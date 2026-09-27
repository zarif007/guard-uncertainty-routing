"""
The gates.  Fixed before any data is collected, and each one can fail.

The discipline is inherited from the quantization study this repo used to be
(docs/archive_preregistration_quantization.md): criteria are committed in
advance, and scripts/verify_selective.py runs every gate against synthetic
data whose truth is known by construction.  A test that cannot fail on data
built to fail it is decoration.

What changed with the pivot is the BASELINE.  Safety-Flag (arXiv 2609.19072)
established that a guard's native confidence already beats random deferral on
every model they tested.  So "beats random" is no longer a finding -- it is a
sanity check that our pipeline reproduces known results.  The crux is whether
anything beats native confidence, and that is gate S3.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from evaluation.selective import compare_signals, error_vector
from evaluation.signals import MARGIN, NATIVE, PRECISION_AGREE, STABILITY

# --- thresholds, fixed in advance ----------------------------------------
# A signal must beat the baseline by at least this much AURC to count as a
# practical improvement.  Significance alone is not enough: on a large enough
# sample a 0.001 AURC gap is significant and irrelevant.
MIN_AURC_IMPROVEMENT = 0.01

# Bootstrap mass required on the correct side before a difference is called.
CONFIDENCE_LEVEL = 0.95

# Below this fraction of distinct confidence values, a signal is reporting
# input order rather than ranking, and its AURC is not interpretable.
MAX_TIE_FRACTION = 0.90

# Gate B (scorer sanity) survives from the previous study unchanged.
GATE_B_AGREEMENT = 0.99
CACHE_SPEEDUP_TARGET_BY_BACKEND = {"cpu": 5.0, "metal": 3.0, "cuda": 1.2}


def cache_speedup_target(backend: Optional[str] = None) -> float:
    return CACHE_SPEEDUP_TARGET_BY_BACKEND.get((backend or "cpu").lower(), 5.0)


def gate_b(agreement: float, speedup: Optional[float] = None,
           backend: Optional[str] = None) -> Dict:
    """Scorer sanity: p_unsafe >= 0.5 reproduces the argmax label."""
    passed = agreement >= GATE_B_AGREEMENT
    target = cache_speedup_target(backend)
    return {
        "gate": "B", "status": "PASS" if passed else "FAIL",
        "agreement": agreement, "threshold": GATE_B_AGREEMENT,
        "cache_speedup": speedup, "cache_speedup_target": target,
        "reason": ("scores reproduce labels" if passed
                   else f"agreement {agreement:.4f} below {GATE_B_AGREEMENT}"),
    }


# --- helpers --------------------------------------------------------------

def _paired(df: pd.DataFrame, a: str, b: str, seed: int = 42) -> Optional[Dict]:
    """Compare two signals on the rows where both are present."""
    if a not in df.columns or b not in df.columns:
        return None
    mask = df[a].notna() & df[b].notna()
    sub = df[mask]
    if len(sub) < 30:
        return None
    errors = error_vector(sub["ground_truth"], sub["prediction"])
    if errors.sum() < 5 or errors.sum() == len(errors):
        return None
    result = compare_signals(errors, sub[a].to_numpy(), sub[b].to_numpy(), seed=seed)
    result.update({"signal": a, "baseline": b, "n": int(len(sub)),
                   "n_errors": int(errors.sum())})
    return result


def _verdict(result: Optional[Dict]) -> str:
    """
    A signal wins only if it is both significantly and materially better.

    `delta_aurc` is signal minus baseline, so better means MORE NEGATIVE.
    """
    if result is None:
        return "NOT_EVALUABLE"
    if not np.isfinite(result.get("delta_aurc", np.nan)):
        return "NOT_EVALUABLE"
    significant = result["p_a_better"] >= CONFIDENCE_LEVEL
    material = result["delta_aurc"] <= -MIN_AURC_IMPROVEMENT
    if significant and material:
        return "BETTER"
    if result["p_a_better"] <= 1 - CONFIDENCE_LEVEL and \
            result["delta_aurc"] >= MIN_AURC_IMPROVEMENT:
        return "WORSE"
    return "NO_DIFFERENCE"


def _per_model(df: pd.DataFrame, fn) -> Dict[str, Dict]:
    return {model: fn(sub) for model, sub in df.groupby("model")}


# --- the gates ------------------------------------------------------------

def gate_s1(df: pd.DataFrame) -> Dict:
    """
    S1 -- is native confidence usable on OUR models?

    Expected to pass.  This replicates Safety-Flag rather than extending it,
    and it exists so that a failure here is read as "our pipeline is broken"
    rather than "we found something".
    """
    from evaluation.selective import signal_report

    per_model = {}
    for model, sub in df.groupby("model"):
        if NATIVE not in sub.columns:
            continue
        mask = sub[NATIVE].notna()
        sub = sub[mask]
        errors = error_vector(sub["ground_truth"], sub["prediction"])
        if errors.sum() < 5:
            continue
        report = signal_report(errors, sub[NATIVE].to_numpy())
        per_model[model] = {
            "deferral_efficiency": report["deferral_efficiency"],
            "tie_fraction": report["tie_fraction"],
            "usable": bool(report["deferral_efficiency"] > 0
                           and report["tie_fraction"] < MAX_TIE_FRACTION),
        }
    if not per_model:
        return {"gate": "S1", "status": "NOT_EVALUABLE", "per_model": {},
                "reason": "no model had enough errors to rank"}
    usable = [m for m, v in per_model.items() if v["usable"]]
    status = ("NATIVE_USABLE" if len(usable) == len(per_model)
              else "NATIVE_DEGENERATE" if not usable else "MIXED")
    return {
        "gate": "S1", "status": status, "per_model": per_model,
        "n_usable": len(usable), "n_models": len(per_model),
        "reason": f"{len(usable)}/{len(per_model)} models have usable native confidence",
    }


def gate_s2(df: pd.DataFrame) -> Dict:
    """
    S2 -- the squashing tax.  Does the raw margin beat the probability?

    Free to test and free to act on: if it passes, every downstream system
    should rank on the margin instead of the probability, and that is a
    one-line change.
    """
    per_model = _per_model(df, lambda sub: _paired(sub, MARGIN, NATIVE))
    verdicts = {m: _verdict(r) for m, r in per_model.items()}
    better = [m for m, v in verdicts.items() if v == "BETTER"]
    evaluable = [m for m, v in verdicts.items() if v != "NOT_EVALUABLE"]
    return {
        "gate": "S2",
        "status": ("MARGIN_BETTER" if better and len(better) == len(evaluable)
                   else "MARGIN_SOMETIMES_BETTER" if better
                   else "NO_DIFFERENCE" if evaluable else "NOT_EVALUABLE"),
        "per_model": {m: {**(per_model[m] or {}), "verdict": verdicts[m]}
                      for m in per_model},
        "n_better": len(better), "n_evaluable": len(evaluable),
        "reason": f"margin beats native probability on {len(better)}/{len(evaluable)} models",
    }


def gate_s3(df: pd.DataFrame, signals: Sequence[str] = (STABILITY, PRECISION_AGREE)) -> Dict:
    """
    S3 -- THE CRUX.  Does any alternative signal beat native confidence?

    Baseline is native confidence, not random.  Holm correction is applied
    across the signals tested, because testing five signals and reporting the
    best one is how a null result becomes a paper by accident.
    """
    from evaluation.statistical_tests import holm_correction

    raw = {}
    for signal in signals:
        per_model = _per_model(df, lambda sub, s=signal: _paired(sub, s, NATIVE))
        for model, result in per_model.items():
            if result is not None:
                raw[(signal, model)] = result

    if not raw:
        return {"gate": "S3", "status": "NOT_EVALUABLE", "per_signal": {},
                "reason": "no alternative signal was computable"}

    keys = list(raw)
    # One-sided p: bootstrap mass on the wrong side of zero.
    pvals = [1.0 - raw[k]["p_a_better"] for k in keys]
    corrected = holm_correction(pvals, alpha=1 - CONFIDENCE_LEVEL)

    per_signal = {}
    for (signal, model), holm in zip(keys, corrected):
        result = raw[(signal, model)]
        material = result["delta_aurc"] <= -MIN_AURC_IMPROVEMENT
        survives = bool(holm["significant"])
        per_signal.setdefault(signal, {})[model] = {
            **result,
            "p_holm": holm["p_holm"],
            "survives_correction": survives,
            "material": bool(material),
            "verdict": "BETTER" if (survives and material) else "NO_DIFFERENCE",
        }

    winners = {s: [m for m, v in models.items() if v["verdict"] == "BETTER"]
               for s, models in per_signal.items()}
    any_winner = any(winners.values())
    return {
        "gate": "S3",
        "status": "ALTERNATIVE_BEATS_NATIVE" if any_winner else "NATIVE_IS_BEST",
        "per_signal": per_signal,
        "winners": winners,
        "reason": ("; ".join(f"{s} wins on {len(m)} model(s)"
                             for s, m in winners.items() if m)
                   if any_winner else
                   "no alternative signal beat native confidence after correction"),
    }


def gate_s4(df: pd.DataFrame) -> Dict:
    """
    S4 -- is instability independent of the margin?

    The guard against S3 passing for a boring reason.  Prior work (arXiv
    2402.13006) found that examples which flip under perturbation already have
    higher baseline uncertainty.  If stability is just a noisy restatement of
    a small margin, it costs ~10x the inference and adds nothing.

    Tested two ways: the correlation between the two signals, and whether
    stability still ranks errors on the subset where the margin carries no
    information (the tied block at the top of the margin distribution).
    """
    if STABILITY not in df.columns or MARGIN not in df.columns:
        return {"gate": "S4", "status": "NOT_EVALUABLE",
                "reason": "stability signal not computed"}

    per_model = {}
    for model, sub in df.groupby("model"):
        mask = sub[STABILITY].notna() & sub[MARGIN].notna()
        sub = sub[mask]
        if len(sub) < 50:
            continue
        corr = float(np.corrcoef(sub[STABILITY], sub[MARGIN])[0, 1])

        # The saturated region: margins so large the probability is tied at
        # 1.0, where native confidence provably cannot rank anything.
        saturated = sub[sub[MARGIN].abs() > 37.0]
        rescue = None
        if len(saturated) >= 30:
            errors = error_vector(saturated["ground_truth"], saturated["prediction"])
            if 5 <= errors.sum() < len(errors):
                from evaluation.selective import signal_report
                rescue = signal_report(errors, saturated[STABILITY].to_numpy())
        per_model[model] = {
            "correlation_with_margin": corr,
            "n_saturated": int(len(saturated)),
            "saturated_deferral_efficiency": (
                rescue["deferral_efficiency"] if rescue else None),
        }

    if not per_model:
        return {"gate": "S4", "status": "NOT_EVALUABLE",
                "reason": "not enough paired rows"}
    corrs = [v["correlation_with_margin"] for v in per_model.values()]
    independent = float(np.nanmean(np.abs(corrs))) < 0.5
    return {
        "gate": "S4",
        "status": "INDEPENDENT" if independent else "REDUNDANT_WITH_MARGIN",
        "per_model": per_model,
        "mean_abs_correlation": float(np.nanmean(np.abs(corrs))),
        "reason": ("stability carries information the margin does not"
                   if independent else
                   "stability largely restates the margin; its extra cost is not justified"),
    }


def gate_s5(df: pd.DataFrame) -> Dict:
    """
    S5 -- does the KIND of guard determine whether its confidence works?

    The architectural claim: a generative guard's confidence is a by-product
    of next-token prediction, an encoder's is a trained probability.  If the
    two classes differ systematically, that is the headline.
    """
    from evaluation.selective import signal_report
    from models.registry import kind_of

    by_kind: Dict[str, List[float]] = {}
    per_model = {}
    for model, sub in df.groupby("model"):
        if NATIVE not in sub.columns:
            continue
        sub = sub[sub[NATIVE].notna()]
        errors = error_vector(sub["ground_truth"], sub["prediction"])
        if errors.sum() < 5:
            continue
        try:
            kind = kind_of(model)
        except ValueError:
            kind = "unknown"
        report = signal_report(errors, sub[NATIVE].to_numpy())
        per_model[model] = {"kind": kind,
                            "deferral_efficiency": report["deferral_efficiency"],
                            "tie_fraction": report["tie_fraction"]}
        by_kind.setdefault(kind, []).append(report["deferral_efficiency"])

    if len(by_kind) < 2:
        return {"gate": "S5", "status": "NOT_EVALUABLE", "per_model": per_model,
                "reason": f"only one kind of guard present: {list(by_kind)}"}

    means = {k: float(np.nanmean(v)) for k, v in by_kind.items()}
    gap = max(means.values()) - min(means.values())
    best = max(means, key=means.get)
    return {
        "gate": "S5",
        "status": "KIND_MATTERS" if gap >= 0.10 else "KIND_DOES_NOT_MATTER",
        "per_model": per_model, "mean_by_kind": means, "gap": gap,
        "better_kind": best if gap >= 0.10 else None,
        "reason": (f"{best} guards rank their errors better (gap {gap:.3f})"
                   if gap >= 0.10 else
                   f"both kinds rank errors similarly (gap {gap:.3f})"),
    }


# --- claims ---------------------------------------------------------------

CLAIMS = [
    ("Native guard confidence is usable for routing review",
     "gate_s1", ["NATIVE_USABLE", "MIXED"]),
    ("The logit margin beats the probability for routing review",
     "gate_s2", ["MARGIN_BETTER", "MARGIN_SOMETIMES_BETTER"]),
    ("An alternative signal beats native confidence",
     "gate_s3", ["ALTERNATIVE_BEATS_NATIVE"]),
    ("Native confidence is already near the practical ceiling",
     "gate_s3", ["NATIVE_IS_BEST"]),
    ("Behavioural instability carries information confidence cannot express",
     "gate_s4", ["INDEPENDENT"]),
    ("The kind of guard determines whether its confidence is usable",
     "gate_s5", ["KIND_MATTERS"]),
]


def claims_table(evidence: Dict[str, object]) -> pd.DataFrame:
    rows = []
    for claim, key, licensing in CLAIMS:
        observed = evidence.get(key)
        if observed is None or observed in ("NOT_EVALUABLE",):
            status = "NOT_TESTED"
        elif observed in licensing:
            status = "LICENSED"
        else:
            status = "NOT_LICENSED"
        rows.append({"claim": claim, "gate": key,
                     "observed": observed, "status": status})
    return pd.DataFrame(rows)


def hardware_consistency(combined: pd.DataFrame) -> Dict:
    """
    Scores do not depend on the machine, but they do depend on the build.

    Decision metrics are functions of the logits, so they are comparable
    across hosts provided every model was scored on the same engine build.
    This reports what was found rather than refusing to proceed.
    """
    if "env_hash" not in combined.columns:
        return {"status": "UNKNOWN", "reason": "no env_hash recorded",
                "environments": []}
    envs = []
    for env_hash, sub in combined.groupby("env_hash"):
        envs.append({
            "env_hash": env_hash,
            "backend": sub["backend"].iloc[0] if "backend" in sub else None,
            "gpu_name": sub["gpu_name"].iloc[0] if "gpu_name" in sub else None,
            "n_rows": int(len(sub)),
            "models": sorted(sub["model"].unique()),
        })
    single = len(envs) == 1
    return {
        "status": "CONSISTENT" if single else "MIXED",
        "reason": ("all predictions share one environment" if single
                   else f"{len(envs)} environments present"),
        "environments": envs,
        "scores_comparable": True,
    }
