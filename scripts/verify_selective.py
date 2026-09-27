"""
verify_selective.py — prove the gates can fail.

Pre-registering a criterion means nothing if the criterion cannot be failed.
Each case below builds prediction data whose truth is known by construction,
and asserts the verdict.  Run before the sweep and after any edit to
evaluation/gates.py or evaluation/selective.py.

    python scripts/verify_selective.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.allocation import allocation_table
from evaluation.gates import (
    gate_a1, gate_a2, gate_s1, gate_s2, gate_s3, gate_s4, gate_s5,
)
from evaluation.signals import MARGIN, NATIVE, STABILITY

PASS, FAIL = "  ok  ", " FAIL "
RESULTS = []


def synth(model, n=800, error_rate=0.18, seed=0, *, saturated=False,
          stability_mode=None, margin_tracks_errors=True,
          latency=0.05):
    """
    One model's predictions with a known error structure.

    saturated            margins so large the sigmoid ties every score at 1.0,
                         which is the pathology the whole study is about
    margin_tracks_errors whether the margin carries information about errors.
                         False is the case the S4 "independent" test needs: a
                         guard whose margin says nothing, where any ranking
                         ability must be coming from somewhere else.
    stability_mode       'informative'  instability tracks errors
                         'redundant'    instability is a noisy copy of the margin
                         'noise'        instability is pure noise
    """
    rng = np.random.default_rng(seed)
    gt = np.where(rng.random(n) < 0.4, "unsafe", "safe")
    err = rng.random(n) < error_rate
    pred = np.where(err, np.where(gt == "unsafe", "safe", "unsafe"), gt)

    if not margin_tracks_errors:
        # The margin is pure noise with respect to correctness.
        mag = rng.normal(44, 2, n)
    elif saturated:
        # Errors and correct answers both far out; the sigmoid cannot separate
        # them, but the margin still can.
        mag = np.where(err, rng.normal(40, 2, n), rng.normal(48, 2, n))
    else:
        mag = np.where(err, rng.normal(1.0, 0.6, n), rng.normal(6.0, 1.5, n))
    mag = np.abs(mag)
    margin = np.where(pred == "unsafe", mag, -mag)
    p = 1.0 / (1.0 + np.exp(-np.clip(margin, -500, 500)))

    frame = pd.DataFrame({
        "prompt_id": [f"p_{i}" for i in range(n)],
        "ground_truth": gt, "prediction": pred, "dataset": "synthetic",
        "model": model, "p_unsafe": p, "margin": margin,
        "latency_sec": latency,
        NATIVE: np.abs(p - 0.5), MARGIN: np.abs(margin),
    })

    if stability_mode == "informative":
        frame[STABILITY] = np.where(err, rng.uniform(0.5, 0.65, n),
                                    rng.uniform(0.9, 1.0, n))
    elif stability_mode == "redundant":
        frame[STABILITY] = np.abs(margin) / np.abs(margin).max() + rng.normal(0, 0.01, n)
    elif stability_mode == "noise":
        frame[STABILITY] = rng.random(n)
    return frame


def check(name, got, expected):
    ok = got == expected
    RESULTS.append(ok)
    print(f"[{PASS if ok else FAIL}] {name:<54} got {got!r}"
          + ("" if ok else f", expected {expected!r}"))


def main():
    print("=" * 84)
    print(" Gate verification against data of known truth")
    print("=" * 84)

    print("\n--- S1: is native confidence usable? ---")
    check("well-separated guard -> native usable",
          gate_s1(synth("llama-guard-3-8b:fp16", seed=1))["status"],
          "NATIVE_USABLE")

    print("\n--- S2: the squashing tax ---")
    # Saturated margins: the probability ties everything, the margin does not.
    check("saturated scores -> margin beats probability",
          gate_s2(synth("llama-guard-3-8b:fp16", seed=2, saturated=True))["status"],
          "MARGIN_BETTER")
    # Unsaturated: sigmoid is order-preserving and loses nothing.
    check("unsaturated scores -> no difference",
          gate_s2(synth("llama-guard-3-8b:fp16", seed=3))["status"],
          "NO_DIFFERENCE")

    print("\n--- S3: does an alternative beat native confidence? ---")
    good = synth("llama-guard-3-8b:fp16", seed=4, saturated=True,
                 margin_tracks_errors=False, stability_mode="informative")
    check("informative stability on a degenerate guard -> beats native",
          gate_s3(good, signals=[STABILITY])["status"],
          "ALTERNATIVE_BEATS_NATIVE")
    # The false-positive check: a useless signal must not win.
    noise = synth("llama-guard-3-8b:fp16", seed=5, stability_mode="noise")
    check("pure-noise stability -> native is best",
          gate_s3(noise, signals=[STABILITY])["status"],
          "NATIVE_IS_BEST")

    print("\n--- S4: is instability independent of the margin? ---")
    # The margin says nothing about correctness here, so any correlation
    # between stability and margin would have to be spurious.
    check("stability informative where the margin is not -> independent",
          gate_s4(synth("llama-guard-3-8b:fp16", seed=6, saturated=True,
                        margin_tracks_errors=False,
                        stability_mode="informative"))["status"],
          "INDEPENDENT")
    check("stability that is a noisy margin -> redundant",
          gate_s4(synth("llama-guard-3-8b:fp16", seed=7,
                        stability_mode="redundant"))["status"],
          "REDUNDANT_WITH_MARGIN")

    print("\n--- S5: does guard kind decide it? ---")
    mixed = pd.concat([
        synth("llama-guard-3-8b:fp16", seed=8, saturated=True),   # degenerate
        synth("encoder-moderation:hf", seed=9),                   # well separated
    ], ignore_index=True)
    check("degenerate generative vs graded encoder -> kind matters",
          gate_s5(mixed)["status"], "KIND_MATTERS")
    same = pd.concat([
        synth("llama-guard-3-8b:fp16", seed=10),
        synth("encoder-moderation:hf", seed=11),
    ], ignore_index=True)
    check("both kinds well separated -> kind does not matter",
          gate_s5(same)["status"], "KIND_DOES_NOT_MATTER")

    print("\n--- A1/A2: how should the oversight budget be spent? ---")

    def priced(frames, perturb_variants=6):
        """Predictions plus a matching perturbation file, ready to price."""
        df = pd.concat(frames, ignore_index=True)
        rows = []
        rng = np.random.default_rng(99)
        for model, sub in df.groupby("model"):
            err = (sub["ground_truth"].to_numpy() != sub["prediction"].to_numpy())
            for pid, is_err in zip(sub["prompt_id"], err):
                flip = 0.45 if is_err else 0.03
                for v in range(perturb_variants):
                    rows.append({"prompt_id": pid, "model": model,
                                 "variant": f"v{v}", "semantics_preserved": True,
                                 "prediction": "unsafe" if rng.random() < flip else "safe"})
        perturbed = pd.DataFrame(rows)
        df[STABILITY] = None
        from evaluation.signals import stability_confidence
        stab = stability_confidence(perturbed)
        idx = pd.MultiIndex.from_arrays([df["model"], df["prompt_id"]])
        df[STABILITY] = stab.reindex(idx).to_numpy()
        return allocation_table(
            df, sorted(df["model"].unique()), [NATIVE, MARGIN, STABILITY],
            perturbed=perturbed, cost_basis="latency")

    # A big guard that is confidently wrong, against a small one whose errors
    # are nearly as rare and whose margins carry information.  The small guard
    # should win outright: fewer bad decisions AND a sixth of the compute.
    cheap_wins = priced([
        synth("llama-guard-3-8b:fp16", seed=20, error_rate=0.18,
              saturated=True, latency=0.090),
        synth("llama-guard-3-1b:fp16", seed=21, error_rate=0.10,
              saturated=False, latency=0.015),
    ])
    check("cheap guard with fewer errors -> cheaper guard wins",
          gate_a1(cheap_wins, budget=0.05)["status"], "CHEAPER_GUARD_WINS")

    # The same big guard against a small one that is simply much worse.  No
    # amount of good selection should rescue it, and A1 must say so.
    default_wins = priced([
        synth("llama-guard-3-8b:fp16", seed=22, error_rate=0.08,
              saturated=False, latency=0.090),
        synth("llama-guard-3-1b:fp16", seed=23, error_rate=0.35,
              saturated=False, latency=0.015),
    ])
    check("small guard far worse -> default is best",
          gate_a1(default_wins, budget=0.05)["status"], "DEFAULT_IS_BEST")

    # A crossover needs the two policies to be good at different things: one
    # guard with few errors it cannot identify, and one with more errors it
    # can.  At zero budget the first wins on raw accuracy; as the reviewer
    # gets hours, the second converts them into caught errors and overtakes.
    crossing = priced([
        synth("llama-guard-3-8b:fp16", seed=24, error_rate=0.12,
              saturated=True, margin_tracks_errors=False, latency=0.090),
        synth("llama-guard-3-1b:fp16", seed=25, error_rate=0.16,
              saturated=False, latency=0.015),
    ])
    a2 = gate_a2(crossing)
    check("guards good at different things -> budget dependent",
          a2["status"], "BUDGET_DEPENDENT")

    # ...and the converse must be reachable, or A2 is not a test.
    check("one dominant policy -> one policy wins",
          gate_a2(cheap_wins)["status"], "ONE_POLICY_WINS")

    passed = sum(RESULTS)
    print("\n" + "=" * 84)
    print(f" {passed}/{len(RESULTS)} checks passed")
    print("=" * 84)
    if passed != len(RESULTS):
        print("\nA gate did not behave as registered. Fix gates.py before collecting data.")
        sys.exit(1)


if __name__ == "__main__":
    main()
