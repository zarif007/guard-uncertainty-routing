import argparse
import os
import shutil
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.hardware import detect_backend, env_fingerprint, fingerprint_hash
from models.llm_loader import DEFAULT_WEIGHTS_DIR
from models.registry import expand_many, get_config, sort_keys, total_disk_gb

PHASES = {
    "1": {
        "name": "Free signals",
        "models": ["llama-guard-3-8b:fp16"],
        "datasets": ["xstest", "harmbench"],
        "gate": "S1, S2: is there a signal, and does the margin beat the probability?",
    },
    "2": {
        "name": "Perturbation",
        "models": ["llama-guard-3-8b:fp16"],
        "datasets": ["xstest"],
        "gate": "S4: does instability add anything beyond the margin? "
                "(run scripts/perturb.py, then analyze)",
    },
    "3": {
        "name": "Full panel",
        "models": ["guard-panel"],
        "datasets": ["toxicchat", "xstest", "harmbench",
                     "wildguardtest", "openai_moderation"],
        "gate": "S3, S5: does anything beat native confidence, "
                "and does guard kind decide it?",
    },
    "4": {
        "name": "Precision ladder",
        "models": ["precision-ladder"],
        "datasets": ["xstest", "harmbench"],
        "gate": "signal 5: cross-precision agreement as an uncertainty signal",
    },
}


def disk_report(keys):
    total = total_disk_gb(keys)
    # Probe the filesystem the weights actually land on.  On a pod the repo
    # sits on the ephemeral container disk and MODEL_WEIGHTS_DIR points at the
    # volume, so probing "." measured the wrong device entirely.
    probe = DEFAULT_WEIGHTS_DIR if os.path.isdir(DEFAULT_WEIGHTS_DIR) else "."
    free = shutil.disk_usage(probe).free / 1024**3
    print(f"Model disk required: {total:.1f} GB   free: {free:.1f} GB")
    if total > free * 0.85:
        print("WARNING: not enough headroom. Use --evict to delete each model after scoring.")
    return total, free


def evict(key):
    config = get_config(key)
    # Walk the configured weights directory, not a hardcoded relative path.
    # setup_runpod.sh points MODEL_WEIGHTS_DIR at the persistent volume, so
    # the old literal found nothing and eviction silently did nothing --
    # exactly when the disk was filling up.
    root = DEFAULT_WEIGHTS_DIR
    removed = 0
    if not config["filename"]:
        return  # transformers-backed guard; nothing to evict
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn == config["filename"]:
                path = os.path.join(dirpath, fn)
                size = os.path.getsize(path)
                os.remove(path)
                removed += size
    if removed:
        print(f"  evicted {config['filename']} ({removed / 1024**3:.2f} GB)")


def main():
    parser = argparse.ArgumentParser(description="Run one phase of the research plan")
    parser.add_argument("--phase", required=True, choices=sorted(PHASES))
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--n-batch", type=int, default=None)
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--latency-repeats", type=int, default=1)
    parser.add_argument("--resume", action="store_true",
                        help="skip prompts already scored (survives a preempted pod)")
    parser.add_argument("--allow-partial-offload", action="store_true")
    parser.add_argument("--evict", action="store_true", help="delete each GGUF after scoring")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()

    phase = PHASES[args.phase]
    keys = sort_keys(expand_many(args.models or phase["models"]))
    datasets = args.datasets or phase["datasets"]

    print(f"=== Phase {args.phase}: {phase['name']} ===")
    print(f"Gate: {phase['gate']}")
    print(f"Models ({len(keys)}): {keys}")
    print(f"Datasets: {datasets}")
    backend = detect_backend(args.n_gpu_layers)
    fingerprint = env_fingerprint(args.n_gpu_layers)
    print(f"Backend: {backend} | {fingerprint.get('gpu_name') or fingerprint['processor']} "
          f"| env {fingerprint_hash(fingerprint)}")
    disk_report(keys)

    if args.dry_run:
        print("\nDry run. Commands that would execute:")
        for key in keys:
            for dataset in datasets:
                print(f"  python scripts/run_model.py --model {key} --dataset {dataset}")
        return

    failures = []
    for key in keys:
        for dataset in datasets:
            cmd = [sys.executable, "scripts/run_model.py", "--model", key, "--dataset", dataset,
                   "--n-gpu-layers", str(args.n_gpu_layers)]
            if args.subset:
                cmd += ["--subset", str(args.subset)]
            if args.n_threads:
                cmd += ["--n-threads", str(args.n_threads)]
            if args.n_batch:
                cmd += ["--n-batch", str(args.n_batch)]
            if args.flash_attn:
                cmd += ["--flash-attn"]
            if args.warmup != 5:
                cmd += ["--warmup", str(args.warmup)]
            if args.latency_repeats > 1:
                cmd += ["--latency-repeats", str(args.latency_repeats)]
            if args.resume:
                cmd += ["--resume"]
            if args.allow_partial_offload:
                cmd += ["--allow-partial-offload"]
            print(f"\n--> {key} on {dataset}")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                failures.append((key, dataset))
        if args.evict:
            evict(key)

    if failures:
        print("\nFailed runs:")
        for key, dataset in failures:
            print(f"  {key} / {dataset}")

    if not args.skip_analysis:
        print("\n--> analysis")
        subprocess.run([sys.executable, "evaluation/analyze.py"])


if __name__ == "__main__":
    main()
