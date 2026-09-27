"""
Full analysis: load predictions, compute every uncertainty signal, ask which
one routes human review best, and write the gate verdicts.

    python evaluation/analyze.py
    python evaluation/analyze.py --perturbed results/predictions/perturbed
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.calibration import fit_temperature
from evaluation.gates import (
    claims_table,
    gate_s1,
    gate_s2,
    gate_s3,
    gate_s4,
    gate_s5,
    hardware_consistency,
)
from evaluation.score_range import report as score_range_report
from evaluation.selective import (
    budget_table,
    error_vector,
    risk_coverage_curve,
    signal_table,
)
from evaluation.signals import (
    MARGIN,
    NATIVE,
    PRECISION_AGREE,
    STABILITY,
    attach_signals,
    available_signals,
)
from evaluation.threshold_analysis import auroc, to_labels
from models.registry import PRECISION_LADDER, sort_keys

REVIEW_BUDGETS = [0.01, 0.05, 0.10, 0.20, 0.50]


def load_predictions(predictions_dir):
    files = sorted(glob.glob(os.path.join(predictions_dir, "predictions_*.csv")))
    if not files:
        raise SystemExit(f"No prediction files in {predictions_dir}")
    frames = [pd.read_csv(f) for f in files]
    combined = pd.concat(frames, ignore_index=True)
    return combined[combined["prediction"] != "error"].reset_index(drop=True)


def load_perturbed(path):
    if not path or not os.path.isdir(path):
        return pd.DataFrame()
    files = sorted(glob.glob(os.path.join(path, "*.csv")))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def save(df, tables_dir, name):
    path = os.path.join(tables_dir, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"  wrote {name}.csv ({len(df)} rows)")


def accuracy_summary(combined, models):
    """Plain classification quality, so a signal result is read in context."""
    rows = []
    for model in models:
        sub = combined[combined["model"] == model]
        if sub.empty:
            continue
        y = to_labels(sub["ground_truth"])
        errors = error_vector(sub["ground_truth"], sub["prediction"])
        row = {
            "model": model,
            "n": len(sub),
            "error_rate": float(np.mean(errors)),
            "accuracy": 1.0 - float(np.mean(errors)),
            "flag_rate": float((sub["prediction"] == "unsafe").mean()),
            "base_rate": float((sub["ground_truth"] == "unsafe").mean()),
        }
        if "p_unsafe" in sub and sub["p_unsafe"].notna().any():
            row["auroc"] = auroc(y, sub["p_unsafe"].values)
            if "margin" in sub:
                # AUROC is rank-based, so this should equal the line above
                # except where the sigmoid saturated into ties.  A gap here is
                # the squashing tax, measured directly.
                row["auroc_on_margin"] = auroc(y, sub["margin"].values)
        rows.append(row)
    return pd.DataFrame(rows)


def budget_report(df, models, signals):
    rows = []
    for model in models:
        for dataset, sub in df[df["model"] == model].groupby("dataset"):
            errors = error_vector(sub["ground_truth"], sub["prediction"])
            if errors.sum() == 0:
                continue
            for signal in signals:
                if signal not in sub.columns or sub[signal].isna().all():
                    continue
                mask = sub[signal].notna().to_numpy()
                table = budget_table(errors[mask], sub[signal].to_numpy()[mask],
                                     REVIEW_BUDGETS)
                table["model"] = model
                table["dataset"] = dataset
                table["signal"] = signal
                rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_risk_coverage(df, models, signals, out_dir):
    for model in models:
        sub = df[df["model"] == model]
        errors = error_vector(sub["ground_truth"], sub["prediction"])
        if errors.sum() < 5:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for signal in signals:
            if signal not in sub.columns or sub[signal].isna().all():
                continue
            mask = sub[signal].notna().to_numpy()
            cov, risk = risk_coverage_curve(errors[mask], sub[signal].to_numpy()[mask])
            ax.plot(cov, risk, label=signal.replace("conf_", ""))
        # The two reference lines that make the curves readable.
        cov, risk = risk_coverage_curve(errors, -errors.astype(float))
        ax.plot(cov, risk, "k--", lw=1, label="oracle")
        ax.axhline(float(np.mean(errors)), color="grey", ls=":", lw=1, label="random")
        ax.set_xlabel("coverage (fraction the guard answers)")
        ax.set_ylabel("risk (error rate among answered)")
        ax.set_title(f"Risk-coverage: {model}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        slug = model.replace(":", "__")
        fig.savefig(os.path.join(out_dir, f"risk_coverage_{slug}.png"), dpi=160)
        plt.close(fig)


def plot_budget(budgets, out_dir):
    if budgets.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for (signal,), sub in budgets.groupby(["signal"]):
        agg = sub.groupby("budget")["recall_of_errors"].mean()
        ax.plot(agg.index, agg.values, marker="o", label=signal.replace("conf_", ""))
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="random")
    ax.set_xlabel("human review budget (fraction of traffic)")
    ax.set_ylabel("fraction of guard errors caught")
    ax.set_title("Errors caught per review budget")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "errors_caught_by_budget.png"), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Uncertainty-signal analysis")
    parser.add_argument("--predictions-dir", default="results/predictions")
    parser.add_argument("--perturbed", default="results/predictions/perturbed",
                        help="directory of perturbation runs (signal 3)")
    parser.add_argument("--tables-dir", default="results/tables")
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument("--temperature", type=float, default=None,
                        help="fixed temperature for the native baseline; "
                             "default fits one per model, as Safety-Flag does")
    args = parser.parse_args()

    os.makedirs(args.tables_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    combined = load_predictions(args.predictions_dir)
    models = sort_keys(list(combined["model"].unique()))
    print(f"Loaded {len(combined)} rows | {len(models)} guards")
    print(f"Datasets: {sorted(combined['dataset'].unique())}")

    hardware = hardware_consistency(combined)
    print(f"Environment: {hardware['status']} — {hardware['reason']}")
    save(pd.DataFrame(hardware.get("environments", [])), args.tables_dir, "environments")

    # Read this before anything else.  If scores are pinned at 0 and 1 there
    # is nothing to rank, and every number below is computed on an instrument
    # with one division on its scale.  We TEST this rather than assume it.
    print("\n=== Score dynamic range ===")
    range_report = score_range_report(combined)
    print(f"  {range_report['status']}: {range_report['reason']}")
    save(range_report["table"], args.tables_dir, "score_range")

    # Signal 1 is built the way Safety-Flag builds it: one temperature per
    # model, fitted on that model's own scores.  Temperature cannot change the
    # ordering, so this is about matching their baseline exactly rather than
    # about improving it.
    enriched = []
    for model, sub in combined.groupby("model"):
        temp = args.temperature
        if temp is None and "margin" in sub and sub["margin"].notna().any():
            try:
                # Signature is (margins, y_true) and it returns a dict.
                temp = fit_temperature(sub["margin"].values,
                                       to_labels(sub["ground_truth"]))["temperature"]
            except Exception:
                temp = 1.0
        enriched.append(attach_signals(sub, temperature=temp or 1.0))
    df = pd.concat(enriched, ignore_index=True)

    perturbed = load_perturbed(args.perturbed)
    ladder_present = [m for m in PRECISION_LADDER if m in set(df["model"])]
    df = attach_signals(
        df,
        perturbed=perturbed if not perturbed.empty else None,
        model_members=models if len(models) > 1 else None,
        precision_members=ladder_present if len(ladder_present) > 1 else None,
    )

    signals = available_signals(df)
    print(f"\nSignals available: {signals}")
    if STABILITY not in signals:
        print("  (no perturbation runs found — signal 3 skipped; "
              "run scripts/perturb.py to enable it)")
    if PRECISION_AGREE not in signals:
        print("  (fewer than two precisions scored — signal 5 skipped)")

    save(accuracy_summary(df, models), args.tables_dir, "accuracy_summary")

    signals_table = signal_table(df, signals)
    save(signals_table, args.tables_dir, "signal_performance")

    budgets = budget_report(df, models, signals)
    save(budgets, args.tables_dir, "errors_caught_by_budget")

    alternatives = [s for s in signals if s not in (NATIVE,)]
    verdicts = {
        "gate_s1": gate_s1(df),
        "gate_s2": gate_s2(df),
        "gate_s3": gate_s3(df, signals=[s for s in alternatives if s != MARGIN] or [MARGIN]),
        "gate_s4": gate_s4(df),
        "gate_s5": gate_s5(df),
    }
    evidence = {k: v["status"] for k, v in verdicts.items()}
    claims = claims_table(evidence)
    save(claims, args.tables_dir, "claims_to_evidence")

    plot_risk_coverage(df, models, signals, args.figures_dir)
    plot_budget(budgets, args.figures_dir)

    report = {
        "models": models,
        "datasets": sorted(df["dataset"].unique()),
        "n_rows": len(df),
        "signals": signals,
        "hardware_consistency": hardware,
        "score_range": {k: v for k, v in range_report.items() if k != "table"},
        **verdicts,
        "evidence": evidence,
    }
    with open(os.path.join(args.tables_dir, "gates.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    print("\n=== GATES ===")
    for key in ("gate_s1", "gate_s2", "gate_s3", "gate_s4", "gate_s5"):
        v = verdicts[key]
        print(f"  {v['gate']}: {v['status']:<28} {v.get('reason','')}")

    if not signals_table.empty:
        print("\n=== Which signal routes review best? (lower AURC is better) ===")
        cols = ["model", "dataset", "signal", "aurc", "deferral_efficiency",
                "tie_fraction", "errors_found_at_5pct", "n_errors"]
        print(signals_table[[c for c in cols if c in signals_table.columns]]
              .round(4).to_string(index=False))

    print("\n=== Claims ===")
    print(claims[["claim", "status"]].to_string(index=False))
    print(f"\nTables  -> {args.tables_dir}")
    print(f"Figures -> {args.figures_dir}")


if __name__ == "__main__":
    main()
