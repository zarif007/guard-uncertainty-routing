import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.allocation import guard_unit_cost
from evaluation.analyze import load_predictions
from evaluation.calibration import fit_temperature
from evaluation.selective import errors_caught, risk_coverage_curve
from evaluation.signals import MODEL_AGREE, NATIVE, attach_signals
from evaluation.threshold_analysis import to_labels

L8 = "llama-guard-3-8b:fp16"
Q8 = "qwen3guard-gen-8b:fp16"
L1 = "llama-guard-3-1b:fp16"
GUARDS = [L8, Q8, L1]
SHORT = {L8: "L8b", Q8: "Q8b", L1: "L1b"}
NAMES = {L8: "Llama-Guard-3-8B", Q8: "Qwen3Guard-8B", L1: "Llama-Guard-3-1B"}
COLORS = {L8: "#1f4e79", Q8: "#c0504d", L1: "#7f7f7f"}
GOOD, ACCENT, NOTE = "#2e7d32", "#e8871e", "#8a5a00"
DATASETS = ["xstest", "harmbench", "toxicchat", "wildguardtest", "openai_moderation"]
DATASET_NAMES = {"xstest": "XSTest", "harmbench": "HarmBench", "toxicchat": "ToxicChat",
                 "wildguardtest": "WildGuard", "openai_moderation": "OpenAI-Mod"}
PANELS = [(L1, L8), (L1, Q8), (L8, Q8), (L1, L8, Q8)]
BAND_GRID = np.round(np.arange(0.01, 0.501, 0.01), 2)

plt.rcParams.update({
    "figure.dpi": 200, "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlesize": 8.5,
})


def enrich(df):
    out = []
    for _, sub in df.groupby("model"):
        temp = fit_temperature(sub["margin"].values, to_labels(sub["ground_truth"]))["temperature"]
        out.append(attach_signals(sub, temperature=temp))
    return pd.concat(out, ignore_index=True)


def wide(df):
    base = df[df["model"] == L8].set_index("prompt_id")[["dataset", "ground_truth", "prompt_chars"]].copy()
    for m in GUARDS:
        sub = df[df["model"] == m].set_index("prompt_id").reindex(base.index)
        s = SHORT[m]
        base[f"pred_{s}"] = sub["prediction"]
        base[f"wrong_{s}"] = (sub["prediction"] != sub["ground_truth"]).astype(int)
        base[f"conf_{s}"] = sub[NATIVE]
        base[f"p_{s}"] = sub["p_unsafe"]
    return base


def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def agreement(w, panel):
    unsafe = sum((w[f"pred_{SHORT[m]}"] == "unsafe").astype(int) for m in panel)
    k = len(panel)
    return np.maximum(unsafe, k - unsafe) / k


def flagged_block(w):
    flagged = agreement(w, GUARDS) < 1
    return {
        "n_flagged": int(flagged.sum()),
        "flag_rate": float(flagged.mean()),
        "share_of_errors_in_block": {
            NAMES[m]: float(w.loc[flagged, f"wrong_{SHORT[m]}"].sum() / w[f"wrong_{SHORT[m]}"].sum())
            for m in GUARDS},
    }


def cascade(w, cost, escalate=0.60, n_boot=2000, seed=42):
    e8 = w["wrong_L8b"].to_numpy()
    e1 = w["wrong_L1b"].to_numpy()
    n = len(w)
    up = np.zeros(n, dtype=bool)
    up[np.argsort(w["conf_L1b"].to_numpy(), kind="mergesort")[:int(round(escalate * n))]] = True
    e_cascade = np.where(up, e8, e1)
    e_oracle = e1 & e8

    rng = np.random.default_rng(seed)
    boot = {"alone": [], "cascade": [], "oracle": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boot["alone"].append(e8[idx].mean())
        boot["cascade"].append(e_cascade[idx].mean())
        boot["oracle"].append(e_oracle[idx].mean())
    boot = {k: np.asarray(v) for k, v in boot.items()}

    policies = {
        "alone": (e8.mean(), cost[L8]),
        "cascade": (e_cascade.mean(), cost[L1] + escalate * cost[L8]),
        "oracle": (e_oracle.mean(), cost[L1] + e1.mean() * cost[L8]),
    }
    available = (boot["alone"] - boot["oracle"]) / boot["alone"]
    return {
        "escalate_fraction": escalate,
        "policies": {k: {"error": float(e), "error_ci": ci(boot[k]), "compute": float(c)}
                     for k, (e, c) in policies.items()},
        "realised_gain": float(e8.mean() - e_cascade.mean()),
        "realised_gain_ci": ci(boot["alone"] - boot["cascade"]),
        "available_gain": float(e8.mean() - e_oracle.mean()),
        "available_gain_ci": ci(boot["alone"] - boot["oracle"]),
        "available_relative": float((e8.mean() - e_oracle.mean()) / e8.mean()),
        "available_relative_ci": ci(available),
    }


def unanimous(w):
    wrong = w[[f"wrong_{SHORT[m]}" for m in GUARDS]]
    all_wrong = wrong.all(axis=1)
    all_right = (wrong == 0).all(axis=1)

    by_dataset = []
    for name in DATASETS:
        d = w[w["dataset"] == name]
        if d.empty:
            continue
        u = all_wrong[d.index]
        by_dataset.append({
            "dataset": name, "n": int(len(d)), "unanimously_wrong": int(u.sum()),
            "rate": float(u.mean()),
            "median_chars_wrong": float(d.loc[u, "prompt_chars"].median()) if u.any() else None,
            "median_chars_rest": float(d.loc[~u, "prompt_chars"].median()),
        })

    confidence = {}
    for m in GUARDS:
        dist = (w[f"p_{SHORT[m]}"] - 0.5).abs()
        confidence[NAMES[m]] = {
            "all": float(dist.median()),
            "unanimously_wrong": float(dist[all_wrong].median()),
            "unanimously_right": float(dist[all_right].median()),
        }

    quintile = pd.qcut(w["prompt_chars"], 5, labels=False)
    quintiles = [{
        "quintile": int(q) + 1,
        "median_chars": float(w.loc[quintile == q, "prompt_chars"].median()),
        "rate": float(all_wrong[quintile == q].mean()),
    } for q in range(5)]

    return {
        "n": int(all_wrong.sum()),
        "rate": float(all_wrong.mean()),
        "benign_flagged": int((all_wrong & (w["ground_truth"] == "safe")).sum()),
        "harmful_admitted": int((all_wrong & (w["ground_truth"] == "unsafe")).sum()),
        "by_dataset": by_dataset,
        "confidence": confidence,
        "length_quintiles": quintiles,
    }


def variant_usability(n_variants=6, seed=42, n_probe=200):
    from scripts.data_loader import get_dataset
    from scripts.perturb import build_variants, variant_diversity

    rows = []
    for name in DATASETS:
        try:
            df = get_dataset(name)
        except FileNotFoundError:
            print(f"  {name}: dataset not downloaded, skipped")
            continue
        rng = np.random.default_rng(seed)
        probe = [build_variants(str(p), n_variants, False, rng) for p in df["prompt"].head(n_probe)]
        rows.append({
            "dataset": name,
            "mean_chars": float(df["prompt"].astype(str).str.len().mean()),
            "usable_variants": float(np.mean([variant_diversity(v)["n_effective"] for v in probe])),
        })
    out = pd.DataFrame(rows)
    rho = spearman(out["mean_chars"], out["usable_variants"]) if len(out) > 2 else None
    return {"by_dataset": rows, "spearman_length_vs_variants": rho}


def per_dataset(w, cost, budgets=(0.01, 0.02, 0.05, 0.10, 0.20, 0.50)):
    panel_cost = sum(cost[m] for m in GUARDS)
    panel = agreement(w, GUARDS)
    guards, winners = [], []
    for name in DATASETS:
        d = w[w["dataset"] == name]
        if d.empty:
            continue
        n = len(d)
        guards.append({"dataset": name, "n": int(n),
                       "harmful_rate": float((d["ground_truth"] == "unsafe").mean()),
                       **{f"error_{SHORT[m]}": float(d[f"wrong_{SHORT[m]}"].mean()) for m in GUARDS}})
        for b in budgets:
            policies = []
            for m in GUARDS:
                s = SHORT[m]
                errors = d[f"wrong_{s}"].to_numpy()
                for signal, conf, c in (("own", d[f"conf_{s}"], cost[m]), ("panel", panel[d.index], panel_cost)):
                    residual = (errors.sum() - errors_caught(errors, conf.to_numpy(), b)) / n
                    policies.append((float(residual), float(c), f"{s}+{signal}"))
            best = min(policies)
            best_single = min(p for p in policies if p[2].endswith("+own"))
            winners.append({"dataset": name, "review_budget": b, "best_policy": best[2],
                            "best_residual": best[0], "best_single_policy": best_single[2],
                            "best_single_residual": best_single[0]})
    return {"guards": guards, "winners": winners}


def band_test(w, seeds=20):
    n = len(w)
    rows = []
    for decider in GUARDS:
        s = SHORT[decider]
        errors = w[f"wrong_{s}"].to_numpy()
        native = w[f"conf_{s}"].to_numpy()
        res_native = np.array([errors.sum() - errors_caught(errors, native, b) for b in BAND_GRID]) / n
        for panel in PANELS:
            if decider not in panel:
                continue
            agree = agreement(w, panel).to_numpy()
            res_panel = np.zeros(len(BAND_GRID))
            for seed in range(seeds):
                jitter = np.random.default_rng(seed).random(n) * 1e-6
                res_panel += np.array([errors.sum() - errors_caught(errors, agree + jitter, b)
                                       for b in BAND_GRID]) / n
            res_panel /= seeds
            wins = BAND_GRID[res_panel < res_native]
            flag_rate = float((agree < 1).mean())
            rows.append({
                "decider": SHORT[decider],
                "panel": "+".join(SHORT[m] for m in panel),
                "flag_rate": flag_rate,
                "band_start": float(wins.min()) if wins.size else None,
                "band_end": float(wins.max()) if wins.size else None,
                "overshoot": float(wins.max() - flag_rate) if wins.size else None,
            })
    banded = pd.DataFrame([r for r in rows if r["band_end"] is not None])
    return {
        "rows": rows,
        "n_panels_with_band": int(len(banded)),
        "spearman_flag_rate_vs_band_end": spearman(banded["flag_rate"], banded["band_end"]),
        "n_overshoot": int((banded["overshoot"] > 0).sum()),
        "mean_overshoot": float(banded["overshoot"].mean()),
    }


def fig_budget_crossover(allocation, path):
    policies = [
        (L8, NATIVE, "Llama-Guard-3-8B + own confidence", "-", "o"),
        (Q8, MODEL_AGREE, "Qwen3Guard-8B + 3-guard panel", "-", "s"),
        (Q8, NATIVE, "Qwen3Guard-8B + own confidence", "--", "^"),
        (L1, NATIVE, "Llama-Guard-3-1B + own confidence", ":", "v"),
    ]
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    inset = ax.inset_axes([0.42, 0.33, 0.33, 0.37])
    for model, signal, label, ls, marker in policies:
        sub = allocation[(allocation["model"] == model) & (allocation["signal"] == signal)].sort_values("human_budget")
        x, y = sub["human_budget"] * 100, sub["residual_error_rate"]
        compute = sub["compute_per_decision"].iloc[0]
        ax.plot(x, y, ls=ls, marker=marker, ms=4, color=COLORS[model],
                label=f"{label}  (compute {compute:.2f})")
        if model != L1:
            keep = (x >= 5) & (x <= 20)
            inset.plot(x[keep], y[keep], ls=ls, marker=marker, ms=4, color=COLORS[model])
    inset.axvspan(10, 20, color=ACCENT, alpha=0.18)
    inset.set_xticks([5, 10, 20])
    inset.tick_params(labelsize=6)
    inset.set_title("panel wins 10-20%", fontsize=6.5, color=NOTE, fontweight="normal", loc="center")
    ax.indicate_inset_zoom(inset, edgecolor="#999999")
    ax.set_xlabel("Human review budget (% of traffic)")
    ax.set_ylabel("Residual risk (errors reaching users)")
    ax.set_ylim(0, None)
    ax.set_title("The best policy changes with the review budget")
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_frontier(allocation, path, budgets=(0.01, 0.05, 0.10, 0.20)):
    from matplotlib.lines import Line2D

    signals = {NATIVE: ("o", "own confidence"), MODEL_AGREE: ("s", "3-guard panel")}
    fig, axes = plt.subplots(1, len(budgets), figsize=(8.0, 2.6))
    for ax, budget in zip(axes, budgets):
        sub = allocation[np.isclose(allocation["human_budget"], budget)
                         & allocation["signal"].isin(signals)
                         & allocation["model"].isin(GUARDS)]
        for _, r in sub.iterrows():
            ax.plot(r["compute_per_decision"], r["residual_error_rate"], signals[r["signal"]][0],
                    ms=6, color=COLORS[r["model"]])
        pts = sub.sort_values(["compute_per_decision", "residual_error_rate"])
        best, frontier = np.inf, []
        for _, r in pts.iterrows():
            if r["residual_error_rate"] < best:
                best = r["residual_error_rate"]
                frontier.append((r["compute_per_decision"], best))
        fx, fy = zip(*frontier)
        ax.plot(fx, fy, ls="--", color="black", lw=0.8, zorder=0)
        ax.set_title(f"{budget * 100:.0f}% review", loc="center")
        ax.set_xlim(0, 0.9)
        ax.set_xlabel("Compute per decision (s)")
    axes[0].set_ylabel("Residual risk")
    handles = [Line2D([], [], ls="", marker="o", color=COLORS[m], label=NAMES[m]) for m in GUARDS]
    handles += [Line2D([], [], ls="", marker=mk, color="#444444", label=label) for mk, label in signals.values()]
    handles += [Line2D([], [], ls="--", color="black", lw=0.8, label="frontier")]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=6.5, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_risk_coverage(df, path):
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for m in GUARDS:
        sub = df[df["model"] == m]
        errors = (sub["prediction"] != sub["ground_truth"]).astype(int).to_numpy()
        coverage, risk = risk_coverage_curve(errors, sub[NATIVE].to_numpy())
        ax.plot(coverage * 100, risk, color=COLORS[m], lw=1.6, label=NAMES[m])
    ax.axvspan(80, 99, color=ACCENT, alpha=0.12)
    ax.text(89.5, 0.005, "1-20% review", ha="center", fontsize=7, color=NOTE)
    ax.set_xlim(45, 101)
    ax.set_ylim(0, 0.2)
    ax.set_xlabel("Coverage (% of decisions kept, rest sent to review)")
    ax.set_ylabel("Risk among kept decisions")
    ax.set_title("Risk-coverage, native confidence")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_agreement_forest(gates, path):
    per_model = gates["gate_s3"]["per_signal"][MODEL_AGREE]
    order = [L1, L8, Q8]
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    for i, m in enumerate(order):
        r = per_model[m]
        color = GOOD if r["verdict"] == "BETTER" else COLORS[Q8]
        ax.plot([r["ci_lo"], r["ci_hi"]], [i, i], color=color, lw=2.2)
        ax.plot(r["delta_aurc"], i, "o", color=color, ms=7)
        verdict = "better" if r["verdict"] == "BETTER" else "no difference"
        ax.text(0.0265, i, f"Holm p = {r['p_holm']:.3f}\n{verdict}", va="center", fontsize=6.5, color=color)
    ax.axvline(0, color="black", ls="--", lw=0.8)
    ax.set_yticks(range(len(order)), [NAMES[m] for m in order])
    ax.set_ylim(len(order) - 0.5, -0.9)
    ax.text(-0.0195, -0.55, "agreement helps", fontsize=7, color=GOOD)
    ax.text(0.0055, -0.55, "agreement hurts", fontsize=7, color=COLORS[Q8])
    ax.set_xlim(-0.026, 0.052)
    ax.set_xlabel("Change in AURC from 3-guard agreement (negative = better)")
    ax.set_title("Cross-guard agreement helps only the weakest guard")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_identification_gap(result, path):
    keys = ["alone", "cascade", "oracle"]
    labels = {"alone": "8B guard\nalone", "cascade": "cascade, escalating\non 1B confidence",
              "oracle": "cascade, oracle\nescalation"}
    colors = {"alone": COLORS[L8], "cascade": COLORS[L1], "oracle": GOOD}
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    for i, k in enumerate(keys):
        p = result["policies"][k]
        lo, hi = p["error_ci"]
        ax.bar(i, p["error"], width=0.5, color=colors[k], alpha=0.85)
        ax.errorbar(i, p["error"], yerr=[[p["error"] - lo], [hi - p["error"]]], color="black", capsize=4, lw=1)
        ax.text(i - 0.27, p["error"] + 0.004, f"{p['error']:.4f}", ha="center", fontsize=7.5, fontweight="bold")
    top, bottom = result["policies"]["alone"]["error"], result["policies"]["oracle"]["error"]
    for y in (top, bottom):
        ax.axhline(y, color=ACCENT, ls=":", lw=0.8)
    ax.annotate("", xy=(2.6, bottom), xytext=(2.6, top), arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.5))
    lo, hi = result["available_relative_ci"]
    ax.text(2.68, (top + bottom) / 2,
            f"available\n{result['available_relative'] * 100:.1f}%\n[{lo * 100:.1f}, {hi * 100:.1f}]",
            va="center", fontsize=7, color=NOTE)
    ax.set_xticks(range(3), [f"{labels[k]}\n(compute {result['policies'][k]['compute']:.2f})" for k in keys],
                  fontsize=7)
    ax.set_xlim(-0.5, 3.2)
    ax.set_ylabel("Error rate")
    ax.set_title("Identification gap: routing could help, no signal captures it")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_length_axis(unan, usability, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    q = pd.DataFrame(unan["length_quintiles"])
    a.bar(range(5), q["rate"] * 100, width=0.6, color=COLORS[L8], alpha=0.85)
    for i, r in enumerate(q["rate"]):
        a.text(i, r * 100 + 0.1, f"{r * 100:.2f}%", ha="center", fontsize=7, fontweight="bold")
    a.set_xticks(range(5), [f"{c:.0f}" for c in q["median_chars"]])
    a.set_xlabel("Prompt length quintile (median characters)")
    a.set_ylabel("All 3 guards agree AND all are wrong (%)")
    a.set_title("(a)  Panel failure rises with length")

    u = pd.DataFrame(usability["by_dataset"])
    if not u.empty:
        for _, r in u.iterrows():
            ok = r["usable_variants"] >= 4
            b.plot(r["mean_chars"], r["usable_variants"], "o" if ok else "X", ms=7,
                   color=GOOD if ok else COLORS[Q8])
            b.text(r["mean_chars"] + 25, r["usable_variants"] + 0.1, DATASET_NAMES[r["dataset"]], fontsize=6.5)
    b.axhline(4, color=ACCENT, ls="--", lw=1.2)
    b.text(400, 4.1, "usability threshold", fontsize=6.5, color=NOTE)
    b.set_ylim(1.4, 5.6)
    b.set_xlabel("Mean prompt length (characters)")
    b.set_ylabel("Usable rewordings (of 6)")
    b.set_title("(b)  Rewordings stop working with length")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_auroc_vs_error(summary, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for _, r in summary.iterrows():
        m = r["model"]
        ax.plot(r["auroc"], r["error_rate"], "o", ms=9, color=COLORS[m])
        ax.text(r["auroc"], r["error_rate"] - 0.005, f"{NAMES[m]}\nflags {r['flag_rate'] * 100:.0f}%",
                ha="center", va="top", fontsize=6.5)
    q = summary[summary["model"] == Q8].iloc[0]
    ax.annotate("better ranking,\nworse deployment", xy=(q["auroc"] - 0.003, q["error_rate"] + 0.001),
                xytext=(0.905, 0.14), fontsize=7.5, color=NOTE,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2))
    ax.set_xlim(0.855, 0.985)
    ax.set_ylim(0.085, 0.2)
    ax.set_xlabel("AUROC (how well it ranks)")
    ax.set_ylabel("Deployed error rate (what ships)")
    ax.set_title("Ranking quality is not deployment quality")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Exploratory analyses and figures for the paper")
    parser.add_argument("--predictions-dir", default="results/predictions")
    parser.add_argument("--tables-dir", default="results/tables")
    parser.add_argument("--figures-dir", default="paper/figures")
    args = parser.parse_args()
    os.makedirs(args.figures_dir, exist_ok=True)

    df = enrich(load_predictions(args.predictions_dir))
    df = df[df["model"].isin(GUARDS)]
    w = wide(df)
    cost = {m: guard_unit_cost(df, m) for m in GUARDS}

    print("=== Exploratory analyses ===")
    out = {
        "flagged_block": flagged_block(w),
        "cascade": cascade(w, cost),
        "unanimous_failure": unanimous(w),
        "perturbation_usability": variant_usability(),
        "band_test": band_test(w),
        "per_dataset": per_dataset(w, cost),
    }
    path = os.path.join(args.tables_dir, "exploratory.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {path}")

    allocation = pd.read_csv(os.path.join(args.tables_dir, "allocation.csv"))
    summary = pd.read_csv(os.path.join(args.tables_dir, "accuracy_summary.csv"))
    with open(os.path.join(args.tables_dir, "gates.json")) as f:
        gates = json.load(f)

    fig_budget_crossover(allocation, os.path.join(args.figures_dir, "fig1_budget_crossover.png"))
    fig_frontier(allocation, os.path.join(args.figures_dir, "fig2_oversight_frontier.png"))
    fig_risk_coverage(df, os.path.join(args.figures_dir, "fig3_risk_coverage.png"))
    fig_agreement_forest(gates, os.path.join(args.figures_dir, "fig4_agreement_forest.png"))
    fig_identification_gap(out["cascade"], os.path.join(args.figures_dir, "fig5_identification_gap.png"))
    fig_length_axis(out["unanimous_failure"], out["perturbation_usability"],
                    os.path.join(args.figures_dir, "fig6_length_axis.png"))
    fig_auroc_vs_error(summary, os.path.join(args.figures_dir, "fig7_auroc_vs_error.png"))
    print(f"  wrote 7 figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
