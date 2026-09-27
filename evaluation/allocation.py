"""
The oversight allocation problem.

Safety-Flag asks "how reliable is this moderator?"  The unit of analysis is
the model, and the answer is a benchmark.  This module asks a different
question, with the deployment as the unit of analysis:

    You have a guard, a compute budget, and a reviewer with limited hours.
    Every message gets a decision one way or another.  How do you spend the
    oversight budget?

A POLICY is a (guard, signal) pair: which model makes the call, and how the
uncertain cases are picked out for a human.  Every policy has a compute price
and a human price, and they buy the same thing -- fewer bad decisions reaching
production.

Why residual risk, and not errors caught
----------------------------------------
"Errors caught" flatters a bad guard.  A 1B guard making 200 mistakes and
catching 160 of them looks better than an 8B guard making 50 and catching 30,
and ships four times as many bad decisions.  The deployment cares about what
SURVIVES review:

    residual_errors = total_errors(guard) - errors_caught(guard, signal, budget)

So a weak guard has to make up its deficit through selection before it counts
as a win, which is exactly the tradeoff the paper is about.

Why two cost units, and no exchange rate
-----------------------------------------
Compute and human attention are not denominated in the same thing, and any
exchange rate we pick (GPU-hours against a reviewer's wage) is arguable and
dates badly.  So nothing here converts one into the other.  Policies are
compared on a two-dimensional frontier and the reader brings their own prices.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from evaluation.selective import error_vector, errors_caught
from evaluation.signals import (
    MARGIN,
    MODEL_AGREE,
    NATIVE,
    PRECISION_AGREE,
    STABILITY,
)

# Signals that ride along on the forward pass the decision already needed.
# They are free in the only sense that matters: adopting them changes no
# infrastructure and costs no extra inference.
FREE_SIGNALS = (NATIVE, MARGIN)

# 0.01-0.20 is the regime a moderation team actually operates in.  0.50 is
# included only because Safety-Flag reports Risk@0.5, and having one directly
# comparable point turns "we beat the published number" into a checkable claim
# rather than a reimplementation of it.  Nobody reviews half their traffic.
DEFAULT_BUDGETS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50)


def guard_unit_cost(df: pd.DataFrame, model: str,
                    basis: str = "latency") -> float:
    """
    What one decision from this guard costs, in relative units.

    `latency`  measured median seconds per prompt.  Preferred, because it is
               what the deployment actually pays, but it is only comparable
               across guards scored on the same hardware -- which is why
               hardware_consistency() is checked before any of this is used.
    `size`     model size in GB, as a hardware-independent stand-in.  Cruder,
               but it survives a mixed-hardware prediction set and is the
               honest fallback when latency was not recorded.
    """
    sub = df[df["model"] == model]
    if sub.empty:
        return float("nan")
    if basis == "latency" and "latency_sec" in sub.columns:
        value = pd.to_numeric(sub["latency_sec"], errors="coerce").median()
        if np.isfinite(value) and value > 0:
            return float(value)
    from models.registry import get_config
    try:
        return float(get_config(model)["size_gb"])
    except Exception:
        return float("nan")


def signal_multiplier(signal: str, df: pd.DataFrame,
                      perturbed: Optional[pd.DataFrame] = None,
                      members: Optional[Sequence[str]] = None,
                      model: Optional[str] = None) -> float:
    """
    How many guard-inferences one decision costs under this signal.

    Derived from the data rather than assumed: the number of perturbation
    variants actually scored, and the number of ensemble members actually
    run.  Assuming a nominal k would let the paper quote a cost it never paid.
    """
    if signal in FREE_SIGNALS:
        return 1.0
    if signal == STABILITY:
        if perturbed is None or perturbed.empty:
            return float("nan")
        # Count this guard's own variants.  Pooling a multi-model perturbation
        # file would report every guard as costing the sum of all of them.
        sub = perturbed
        if model is not None and "model" in perturbed.columns:
            sub = perturbed[perturbed["model"] == model]
            if sub.empty:
                return float("nan")
        return float(sub.groupby("prompt_id").size().mean())
    if signal in (MODEL_AGREE, PRECISION_AGREE):
        return float(len(members)) if members else float("nan")
    return float("nan")


def policy_row(df: pd.DataFrame, model: str, signal: str, budget: float,
               unit_cost: float, multiplier: float) -> Optional[Dict]:
    """One (guard, signal, human budget) policy, priced and scored."""
    sub = df[df["model"] == model]
    if signal not in sub.columns:
        return None
    mask = sub[signal].notna().to_numpy()
    sub = sub[mask]
    if len(sub) < 30:
        return None

    errors = error_vector(sub["ground_truth"], sub["prediction"])
    total = int(errors.sum())
    if total == 0:
        return None

    caught = errors_caught(errors, sub[signal].to_numpy(), budget) if budget > 0 else 0
    residual = total - caught
    n = len(sub)
    return {
        "model": model,
        "signal": signal,
        "human_budget": budget,
        "n": n,
        "errors_total": total,
        "errors_caught": caught,
        "errors_residual": residual,
        # The deployment objective: how many bad decisions reach production.
        "residual_error_rate": residual / n,
        "baseline_error_rate": total / n,
        "risk_reduction": (total - residual) / total,
        # Compute price, in guard-inferences and in measured units.
        "inferences_per_decision": multiplier,
        "compute_per_decision": unit_cost * multiplier,
        "compute_total": unit_cost * multiplier * n,
        "reviews_per_decision": budget,
    }


def allocation_table(
    df: pd.DataFrame,
    models: Sequence[str],
    signals: Sequence[str],
    budgets: Sequence[float] = DEFAULT_BUDGETS,
    perturbed: Optional[pd.DataFrame] = None,
    ensemble_members: Optional[Dict[str, Sequence[str]]] = None,
    cost_basis: str = "latency",
) -> pd.DataFrame:
    """Every (guard, signal, budget) policy, priced and scored."""
    ensemble_members = ensemble_members or {}
    rows = []
    for model in models:
        unit = guard_unit_cost(df, model, basis=cost_basis)
        for signal in signals:
            mult = signal_multiplier(
                signal, df, perturbed=perturbed,
                members=ensemble_members.get(signal), model=model)
            if not np.isfinite(mult):
                continue
            for budget in budgets:
                row = policy_row(df, model, signal, budget, unit, mult)
                if row:
                    rows.append(row)
    return pd.DataFrame(rows)


def pareto_frontier(table: pd.DataFrame, budget: float) -> pd.DataFrame:
    """
    Policies not beaten on both axes at once, at one human budget.

    A policy is dominated when another is cheaper in compute AND ships fewer
    bad decisions.  What survives is the set worth arguing about; picking
    between them needs prices this module deliberately does not supply.
    """
    sub = table[table["human_budget"] == budget].copy()
    if sub.empty:
        return sub
    sub = sub.sort_values("compute_per_decision")
    best = np.inf
    keep = []
    for _, row in sub.iterrows():
        on_front = row["residual_error_rate"] < best
        keep.append(on_front)
        best = min(best, row["residual_error_rate"])
    sub["on_frontier"] = keep
    return sub


def equal_compute_comparison(table: pd.DataFrame, budget: float,
                             tolerance: float = 0.25) -> pd.DataFrame:
    """
    The centrepiece: at matched compute, does a cheap guard with expensive
    selection beat an expensive guard with free selection?

    The default policy is what everyone actually deploys -- the largest guard
    available, ranked by its own confidence.  Every other policy within
    `tolerance` of that compute budget is a real alternative someone could
    choose instead, and they are compared on residual risk.
    """
    sub = table[table["human_budget"] == budget]
    if sub.empty:
        return sub

    free = sub[sub["signal"].isin(FREE_SIGNALS)]
    if free.empty:
        return pd.DataFrame()
    default = free.loc[free["compute_per_decision"].idxmax()]
    ceiling = default["compute_per_decision"] * (1 + tolerance)

    affordable = sub[sub["compute_per_decision"] <= ceiling].copy()
    affordable["is_default"] = (
        (affordable["model"] == default["model"])
        & (affordable["signal"] == default["signal"]))
    affordable["default_residual_rate"] = default["residual_error_rate"]
    affordable["beats_default"] = (
        affordable["residual_error_rate"] < default["residual_error_rate"])
    affordable["residual_delta"] = (
        affordable["residual_error_rate"] - default["residual_error_rate"])
    affordable["compute_ratio"] = (
        affordable["compute_per_decision"] / default["compute_per_decision"])
    return affordable.sort_values("residual_error_rate")


def crossover_report(table: pd.DataFrame) -> Dict:
    """
    Does the best policy change as the review budget changes?

    If one policy wins everywhere the recommendation is a sentence. If the
    winner changes with budget, the recommendation is a rule, and the crossover
    point is the result -- that is the figure the paper is built around.
    """
    winners = {}
    for budget, sub in table.groupby("human_budget"):
        if sub.empty:
            continue
        best = sub.loc[sub["residual_error_rate"].idxmin()]
        winners[float(budget)] = {
            "model": best["model"], "signal": best["signal"],
            "residual_error_rate": float(best["residual_error_rate"]),
            "compute_per_decision": float(best["compute_per_decision"]),
        }
    distinct = {(v["model"], v["signal"]) for v in winners.values()}
    return {
        "winners_by_budget": winners,
        "n_distinct_winners": len(distinct),
        "crossover_observed": len(distinct) > 1,
        "reason": (f"{len(distinct)} different policies win at different budgets"
                   if len(distinct) > 1 else
                   "one policy wins at every budget tested"),
    }
