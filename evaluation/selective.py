"""
Which decisions should a human double-check?

A guard classifies every message and gets some wrong.  A human reviewer has
capacity for a fraction of them.  An *uncertainty signal* proposes which ones
to hand over.  This module scores how good that proposal is.

Two equivalent readings of the same curve:

  risk-coverage   at coverage c (the guard answers the c most confident),
                  what is the error rate among the answered?
  errors-caught   at review budget b (the human sees the b least confident),
                  how many of the guard's errors does the human find?

The second is what a deployment actually asks, so it is the headline.  The
first is the standard form and is what AURC summarises.

The baseline that matters is NOT random.  Safety-Flag (arXiv 2609.19072)
already established that a guard's own confidence beats random for every model
they tested.  The open question is whether anything beats *native confidence*,
so `deferral_efficiency` is reported against both and every gate compares
signals pairwise on the same items.
"""
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


def error_vector(y_true: Sequence[str], y_pred: Sequence[str]) -> np.ndarray:
    """1 where the guard was wrong.  Labels are the repo-wide 'safe'/'unsafe'."""
    t = np.asarray(y_true, dtype=object)
    p = np.asarray(y_pred, dtype=object)
    return (t != p).astype(int)


def risk_coverage_curve(errors: np.ndarray, confidence: np.ndarray):
    """
    Sweep coverage from 1/n to 1.  Most confident kept first.

    Ties matter here: a guard whose scores are pinned at 0 and 1 produces huge
    tied blocks, and the order within a block is arbitrary.  mergesort is
    stable so the result is at least reproducible, but a curve dominated by
    ties is reporting the input order as if it were signal.  `tie_fraction`
    below is the diagnostic for that, and it should be read alongside AURC.
    """
    errors = np.asarray(errors, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    order = np.argsort(-confidence, kind="mergesort")
    kept = errors[order]
    n = len(kept)
    k = np.arange(1, n + 1)
    coverage = k / n
    risk = np.cumsum(kept) / k
    return coverage, risk


def aurc(errors: np.ndarray, confidence: np.ndarray) -> float:
    """Area under the risk-coverage curve.  Lower is better."""
    _, risk = risk_coverage_curve(errors, confidence)
    return float(np.mean(risk))


def aurc_oracle(errors: np.ndarray) -> float:
    """Best achievable: every correct answer ranked above every error."""
    e = np.asarray(errors, dtype=float)
    return aurc(e, -e)  # non-errors get confidence 0, errors get -1


def aurc_random(errors: np.ndarray) -> float:
    """
    Expected AURC when cases are handed over at random.

    Under random ordering the expected risk at every coverage is the overall
    error rate, so the expected area is that rate.
    """
    return float(np.mean(errors))


def tie_fraction(confidence: np.ndarray) -> float:
    """Fraction of items sharing a confidence value with at least one other."""
    c = np.asarray(confidence, dtype=float)
    if c.size == 0:
        return float("nan")
    _, counts = np.unique(c, return_counts=True)
    return float((counts[counts > 1].sum()) / c.size)


def errors_caught(errors: np.ndarray, confidence: np.ndarray,
                  budget: float) -> int:
    """How many errors a reviewer finds when shown the `budget` least confident."""
    errors = np.asarray(errors, dtype=int)
    confidence = np.asarray(confidence, dtype=float)
    k = int(round(budget * len(errors)))
    if k <= 0:
        return 0
    order = np.argsort(confidence, kind="mergesort")
    return int(errors[order][:k].sum())


def budget_table(errors: np.ndarray, confidence: np.ndarray,
                 budgets: Sequence[float] = (0.01, 0.05, 0.10, 0.20, 0.50)) -> pd.DataFrame:
    """The deployment-facing table: errors found per review budget."""
    total = int(np.sum(errors))
    rows = []
    for b in budgets:
        found = errors_caught(errors, confidence, b)
        rows.append({
            "budget": b,
            "n_reviewed": int(round(b * len(errors))),
            "errors_found": found,
            "errors_total": total,
            "recall_of_errors": found / total if total else float("nan"),
            # What random selection would have found, in expectation.
            "expected_random": b * total,
            "lift_over_random": (found / (b * total)) if total and b > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def signal_report(errors: np.ndarray, confidence: np.ndarray,
                  budgets: Sequence[float] = (0.01, 0.05, 0.10, 0.20, 0.50)) -> Dict:
    """Everything about one (model, dataset, signal) triple."""
    a = aurc(errors, confidence)
    a_rand = aurc_random(errors)
    a_orac = aurc_oracle(errors)
    span = a_rand - a_orac
    return {
        "n": int(len(errors)),
        "n_errors": int(np.sum(errors)),
        "error_rate": float(np.mean(errors)),
        "aurc": a,
        "aurc_random": a_rand,
        "aurc_oracle": a_orac,
        "excess_aurc": a - a_orac,
        # 1.0 = perfect self-knowledge, 0.0 = no better than random.
        "deferral_efficiency": float((a_rand - a) / span) if span > 0 else float("nan"),
        "tie_fraction": tie_fraction(confidence),
        **{f"errors_found_at_{int(b*100)}pct": errors_caught(errors, confidence, b)
           for b in budgets},
    }


def compare_signals(errors: np.ndarray, a: np.ndarray, b: np.ndarray,
                    n_boot: int = 2000, seed: int = 42) -> Dict:
    """
    Paired bootstrap on AURC(a) - AURC(b), resampling items.

    Paired because both signals rank the *same* items, so the comparison is
    within-item and an unpaired test would throw away the pairing and inflate
    the interval.  Negative delta means signal `a` is better (lower AURC).
    """
    errors = np.asarray(errors, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    observed = aurc(errors, a) - aurc(errors, b)

    rng = np.random.default_rng(seed)
    n = len(errors)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        e = errors[idx]
        if e.sum() == 0 or e.sum() == len(e):
            deltas[i] = np.nan
            continue
        deltas[i] = aurc(e, a[idx]) - aurc(e, b[idx])
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        return {"delta_aurc": observed, "ci_lo": float("nan"),
                "ci_hi": float("nan"), "p_a_better": float("nan"), "n_boot": 0}
    return {
        "delta_aurc": float(observed),
        "ci_lo": float(np.percentile(deltas, 2.5)),
        "ci_hi": float(np.percentile(deltas, 97.5)),
        "p_a_better": float(np.mean(deltas < 0)),
        "n_boot": int(deltas.size),
    }


def signal_table(df: pd.DataFrame, signal_columns: Sequence[str],
                 group_by: Sequence[str] = ("model", "dataset")) -> pd.DataFrame:
    """Run signal_report for every (group, signal) combination."""
    rows = []
    for keys, sub in df.groupby(list(group_by)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        errors = error_vector(sub["ground_truth"], sub["prediction"])
        if errors.sum() == 0:
            continue  # nothing to rank; a perfect guard needs no reviewer
        for column in signal_columns:
            if column not in sub.columns:
                continue
            conf = pd.to_numeric(sub[column], errors="coerce")
            if conf.isna().all():
                continue
            mask = conf.notna().to_numpy()
            rows.append({
                **dict(zip(group_by, keys)),
                "signal": column,
                **signal_report(errors[mask], conf.to_numpy()[mask]),
            })
    return pd.DataFrame(rows)
