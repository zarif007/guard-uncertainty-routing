"""
perturb.py — generate meaning-preserving rewordings and score them.

Signal 3 asks whether a guard's verdict survives rewording the message.  It is
the only signal that is behavioural rather than score-based, so it is the one
that can still work on a guard whose scores are pinned at 0 and 1.

The threat to it is stated in docs/preregistration.md commitment 8: a
paraphrase that changes MEANING produces instability that is not uncertainty.
So every variant carries a `semantics_preserved` flag, and
evaluation/signals.stability_confidence expects the failures to be dropped
before the signal is computed.

Two families of perturbation:

  surface   whitespace, capitalisation, terminal punctuation, a polite prefix.
            Meaning-preserving BY CONSTRUCTION -- no judgement call, no model
            needed, and nothing to validate.  This is the default.
  lexical   typo injection.  Cheap, but it can change a word into another
            word, so it is flagged for review rather than trusted.

An LLM paraphraser would give more varied rewordings than either, and is the
obvious upgrade.  It is deliberately not the default: a paraphrase model
introduces its own failure mode (it can sanitise a harmful prompt into a
benign one), and that failure would be indistinguishable from the guard being
unstable.  Add it behind --paraphraser once there is a validation step for it.

    python scripts/perturb.py --model reference --dataset xstest --n-variants 6
"""
import argparse
import os
import sys

import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llm_loader import LLMGuard
from models.registry import get_config
from scripts.data_loader import get_dataset


def surface_variants(prompt: str):
    """(text, name, semantics_preserved) triples.  All meaning-preserving."""
    stripped = prompt.strip()
    no_final_punct = stripped.rstrip(".?!").strip()
    yield stripped, "identity", True
    yield stripped + " ", "trailing_space", True
    yield no_final_punct, "no_final_punct", True
    yield no_final_punct + "?", "question_mark", True
    yield "Please " + stripped[0].lower() + stripped[1:] if stripped else stripped, \
          "polite_prefix", True
    yield stripped[0].upper() + stripped[1:] if stripped else stripped, \
          "capitalised", True
    yield " ".join(stripped.split()), "whitespace_normalised", True


def typo_variants(prompt: str, rng):
    """Single-character transpositions.  Flagged, not trusted."""
    words = prompt.split()
    long_words = [i for i, w in enumerate(words) if len(w) > 4]
    if not long_words:
        return
    for idx in rng.choice(long_words, size=min(2, len(long_words)), replace=False):
        w = words[int(idx)]
        pos = len(w) // 2
        swapped = w[:pos] + w[pos + 1] + w[pos] + w[pos + 2:]
        out = list(words)
        out[int(idx)] = swapped
        # A transposition can produce a real and different word, so this
        # cannot be asserted meaning-preserving without checking.
        yield " ".join(out), f"typo_{idx}", False


def build_variants(prompt: str, n_variants: int, include_typos: bool, rng):
    variants = list(surface_variants(prompt))
    if include_typos:
        variants += list(typo_variants(prompt, rng))
    # Deduplicate: several surface rules collapse on an already-clean prompt,
    # and counting the same string twice would fake agreement.
    seen, unique = set(), []
    for text, name, preserved in variants:
        if text not in seen:
            seen.add(text)
            unique.append((text, name, preserved))
    return unique[:n_variants]


def main():
    import numpy as np

    parser = argparse.ArgumentParser(description="Score reworded variants of each prompt")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--n-variants", type=int, default=6)
    parser.add_argument("--include-typos", action="store_true",
                        help="add flagged lexical variants (semantics_preserved=False)")
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument("--output-dir", default="results/predictions/perturbed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    config = get_config(args.model)
    df = get_dataset(args.dataset, subset_size=args.subset)
    print(f"[{config['key']}] {len(df)} prompts x up to {args.n_variants} variants")

    model = LLMGuard(quant_level=config["key"], n_gpu_layers=args.n_gpu_layers,
                     n_threads=args.n_threads)

    rows = []
    for idx, row in tqdm(list(df.iterrows()), total=len(df)):
        prompt_id = f"{args.dataset.lower()}_{idx}"
        for text, name, preserved in build_variants(
                str(row["prompt"]), args.n_variants, args.include_typos, rng):
            try:
                scored = model.predict_score(text)
            except Exception as exc:
                print(f"  error on {prompt_id}/{name}: {exc}")
                continue
            rows.append({
                "prompt_id": prompt_id,
                "variant": name,
                "semantics_preserved": preserved,
                "prompt": text,
                "ground_truth": row["ground_truth"],
                "dataset": args.dataset.lower(),
                "model": config["key"],
                "prediction": scored["prediction"],
                "p_unsafe": scored["p_unsafe"],
                "margin": scored["margin"],
            })

    out = pd.DataFrame(rows)
    slug = config["key"].replace(":", "__")
    path = os.path.join(args.output_dir, f"perturbed_{slug}_{args.dataset.lower()}.csv")
    out.to_csv(path, index=False)

    dropped = int((~out["semantics_preserved"]).sum())
    flipped = out.groupby("prompt_id")["prediction"].nunique()
    print(f"\n  wrote {len(out)} rows -> {path}")
    print(f"  variants flagged not-meaning-preserving: {dropped} "
          f"({dropped/max(len(out),1):.1%}) — drop these before computing the signal")
    print(f"  prompts whose verdict flipped under rewording: "
          f"{int((flipped > 1).sum())}/{len(flipped)}")


if __name__ == "__main__":
    main()
