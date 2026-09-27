#!/bin/bash
# Small end-to-end check: score a couple of precisions, then print the metrics.
#
# Use it to confirm a new machine measures what you think it measures before
# committing GPU hours to a full sweep.  Writes to results/smoke/ so it never
# touches the real prediction set.
#
#   bash scripts/smoke_test.sh
#   MODELS="q3 q4 q6" N=60 DATASET=harmbench bash scripts/smoke_test.sh
set -e

MODELS=${MODELS:-"reference"}
DATASET=${DATASET:-"xstest"}
N=${N:-40}
OUT=${OUT:-"results/smoke"}
EXTRA=${EXTRA:-""}

if [ -z "$GPU_LAYERS" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        GPU_LAYERS="-1"
    elif [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
        GPU_LAYERS="-1"
    else
        GPU_LAYERS="0"
    fi
fi

echo "=========================================================="
echo " Smoke test: $MODELS on $DATASET ($N prompts each)"
echo " gpu_layers: $GPU_LAYERS   output: $OUT"
echo "=========================================================="

rm -rf "$OUT"
mkdir -p "$OUT"

for MODEL in $MODELS; do
    echo ""
    echo "--> $MODEL"
    python scripts/run_model.py \
        --model "$MODEL" --dataset "$DATASET" --subset "$N" \
        --n-gpu-layers "$GPU_LAYERS" --warmup 3 --output-dir "$OUT" $EXTRA \
        2>&1 | grep -E "backend:|Memory profile|backend: |Weight file|Device memory|Host RSS|Total reported|Environment |median latency|! " || true
done

echo ""
echo "--> analysis"
python evaluation/analyze.py \
    --predictions-dir "$OUT" \
    --tables-dir "$OUT/tables" \
    --figures-dir "$OUT/figures" >/dev/null

python - "$OUT" <<'PY'
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.getcwd())
from evaluation.score_range import print_report

out = sys.argv[1]
summary = pd.read_csv(f"{out}/tables/accuracy_summary.csv")
envs = pd.read_csv(f"{out}/tables/environments.csv")

pd.set_option("display.width", 200)

print("\n=== Environment (must be a single row) ===")
print(envs[[c for c in ("env_hash", "backend", "gpu_name", "n_rows") if c in envs.columns]]
      .to_string(index=False))
if len(envs) > 1:
    print("  WARNING: more than one environment. Efficiency metrics are not comparable.")

# Read this before any metric below.  A guard whose verdict probabilities are
# pinned at 0 and 1 still produces an AUROC, a TOST interval and a
# TPR-at-target-FPR, and all three are meaningless -- a scale with one
# division cannot show a difference.  It is checked here rather than in
# analyze.py because the fix is a harder prompt set, which changes what gets
# run, and that decision has to happen before the GPU hours are spent.
preds = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{out}/*.csv"))],
                  ignore_index=True)
range_status = print_report(preds[preds["prediction"] != "error"])

print("\n=== Classification quality ===")
cols = ["model", "accuracy", "error_rate", "flag_rate", "base_rate",
        "auroc", "auroc_on_margin"]
print(summary[[c for c in cols if c in summary.columns]].round(4).to_string(index=False))

signals = pd.read_csv(f"{out}/tables/signal_performance.csv")
print("\n=== Which signal routes review best? (lower AURC is better) ===")
cols = ["model", "signal", "aurc", "deferral_efficiency", "tie_fraction",
        "errors_found_at_5pct", "n_errors"]
print(signals[[c for c in cols if c in signals.columns]].round(4).to_string(index=False))

if range_status == "POLARIZED":
    print("\nSTOP: the threshold-free analysis cannot work on this prompt set with "
          "these models.\n      Fix the prompt set before running the sweep; every "
          "number above is\n      a fixed-threshold number wearing a threshold-free "
          "name.")
PY

echo ""
echo "=========================================================="
echo " tables:  $OUT/tables/"
echo " figures: $OUT/figures/"
echo "=========================================================="
