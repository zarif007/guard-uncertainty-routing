import argparse
import os
import sys
import traceback

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.datasets_registry import DATASETS, get_spec

CACHE_DIR = "datasets/cache"
NORMALIZED_DIR = "datasets/normalized"


def normalized_path(name: str) -> str:
    return os.path.join(NORMALIZED_DIR, f"{name}.csv")


def materialize(name: str, force: bool = False) -> str:
    spec = get_spec(name)
    out_path = normalized_path(name)
    if os.path.exists(out_path) and not force:
        print(f"  [skip] {name} already at {out_path}")
        return out_path

    from datasets import load_dataset

    os.makedirs(NORMALIZED_DIR, exist_ok=True)
    kwargs = {"cache_dir": CACHE_DIR, "token": os.environ.get("HF_TOKEN") or None}
    if spec.config:
        ds = load_dataset(spec.hf_id, spec.config, split=spec.split, **kwargs)
    else:
        ds = load_dataset(spec.hf_id, split=spec.split, **kwargs)

    rows = spec.normalize(ds)
    df = pd.DataFrame([r for r in rows if r.get("prompt")])
    df = df.dropna(subset=["prompt", "ground_truth"])
    df["prompt"] = df["prompt"].astype(str).str.strip()
    df = df[df["prompt"].str.len() > 0]
    df = df.drop_duplicates(subset=["prompt"]).reset_index(drop=True)

    df["dataset"] = name
    df["level"] = spec.level
    df["tier"] = spec.tier
    df.to_csv(out_path, index=False)

    counts = df["ground_truth"].value_counts().to_dict()
    base_rate = counts.get("unsafe", 0) / max(len(df), 1)
    print(f"  [ok]   {name}: {len(df)} rows  {counts}  harmful_base_rate={base_rate:.3f}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Materialize datasets into normalized CSVs")
    parser.add_argument("--datasets", nargs="*",
                        default=sorted(n for n, spec in DATASETS.items() if not spec.local_csv))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    print(f"Materializing {len(args.datasets)} dataset(s)")
    failed = []
    for name in args.datasets:
        try:
            materialize(name, force=args.force)
        except Exception as exc:
            failed.append((name, str(exc).split("\n")[0]))
            print(f"  [FAIL] {name}: {exc}")
            if os.environ.get("VERBOSE"):
                traceback.print_exc()

    if failed:
        print("\nFailed datasets (update hf_id/config/split in scripts/datasets_registry.py):")
        for name, err in failed:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
