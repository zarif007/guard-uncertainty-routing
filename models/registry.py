"""
The guards we score, and how to load each one.

Scope note.  This registry used to be a quantization ladder: two families x
fourteen precisions, because the question was what lower precision does to a
guard's operating point.  The question is now whether any uncertainty signal
beats a guard's native confidence at routing human review, so the axis that
matters is the KIND of guard, not its bit width.

Two kinds, and the difference is the hypothesis:

  generative   the guard writes the word "safe" or "unsafe" and we read that
               token's logit.  Its confidence is a by-product of next-token
               prediction trained on hard labels, so it has every reason to be
               pinned at 0 and 1.
  encoder      a small classifier with a real probability head, trained with a
               proper scoring rule to produce a graded number.

If encoders rank their own errors well and generative guards do not, the
field's move to LLM-based guards cost it the ability to route human review,
and that is the finding.

The precision ladder survives, demoted, as the input to ONE signal:
cross-precision agreement (evaluation/signals.PRECISION_AGREE).  Quantizing a
guard perturbs the model without touching the prompt, which makes it a
cleaner instability probe than paraphrasing.  Nothing else in the default
path uses bit width.
"""
from typing import Dict, List, Optional, Tuple

# --- how a guard is loaded ------------------------------------------------
GGUF = "gguf"          # llama.cpp, via models/llm_loader.py
HF_CAUSAL = "hf_causal"    # transformers causal LM, via models/hf_loader.py
HF_SEQCLS = "hf_seqcls"    # transformers sequence classifier (encoder head)

# --- what kind of guard it is --------------------------------------------
GENERATIVE = "generative"
ENCODER = "encoder"

# Quantization levels, kept only to feed the cross-precision agreement signal.
LLAMA_8B_PRECISIONS = {
    "fp16":   {"file": "Llama-Guard-3-8B.f16.gguf",    "size_gb": 16.07, "bits": 16.0},
    "q8_0":   {"file": "Llama-Guard-3-8B.Q8_0.gguf",   "size_gb": 8.54,  "bits": 8.5},
    "q6_k":   {"file": "Llama-Guard-3-8B.Q6_K.gguf",   "size_gb": 6.59,  "bits": 6.6},
    "q5_k_m": {"file": "Llama-Guard-3-8B.Q5_K_M.gguf", "size_gb": 5.73,  "bits": 5.7},
    "q4_k_m": {"file": "Llama-Guard-3-8B.Q4_K_M.gguf", "size_gb": 4.92,  "bits": 4.8},
    "q3_k_m": {"file": "Llama-Guard-3-8B.Q3_K_M.gguf", "size_gb": 3.93,  "bits": 3.9},
}

QWEN_8B_PRECISIONS = {
    "fp16":   {"file": "Qwen3Guard-Gen-8B.f16.gguf",    "size_gb": 16.4, "bits": 16.0},
    "q8_0":   {"file": "Qwen3Guard-Gen-8B.Q8_0.gguf",   "size_gb": 8.71, "bits": 8.5},
    "q6_k":   {"file": "Qwen3Guard-Gen-8B.Q6_K.gguf",   "size_gb": 6.73, "bits": 6.6},
    "q5_k_m": {"file": "Qwen3Guard-Gen-8B.Q5_K_M.gguf", "size_gb": 5.85, "bits": 5.7},
    "q4_k_m": {"file": "Qwen3Guard-Gen-8B.Q4_K_M.gguf", "size_gb": 5.03, "bits": 4.8},
    "q3_k_m": {"file": "Qwen3Guard-Gen-8B.Q3_K_M.gguf", "size_gb": 4.01, "bits": 3.9},
}

LLAMA_1B_PRECISIONS = {
    "fp16": {"file": "Llama-Guard-3-1B.f16.gguf",  "size_gb": 2.48, "bits": 16.0},
    "q8_0": {"file": "Llama-Guard-3-1B.Q8_0.gguf", "size_gb": 1.32, "bits": 8.5},
}

# Single-precision entry for a guard we run through transformers rather than
# llama.cpp.  `hf` models have one "precision" key so the rest of the
# pipeline -- which addresses everything as family:precision -- needs no
# special casing.
def _hf_only(size_gb: float) -> Dict[str, dict]:
    return {"hf": {"file": None, "size_gb": size_gb, "bits": 16.0}}


# VERIFY: hub ids and GGUF filenames are best-effort.  scripts/preflight.py
# checks every one against the hub before a run and names what is actually
# published, so a wrong string fails loudly in a minute rather than silently
# three hours in.  The encoder entries in particular need checking -- see the
# task-alignment warning below.
FAMILIES = {
    "llama-guard-3-8b": {
        "kind": GENERATIVE, "backend": GGUF,
        "repo": "mradermacher/Llama-Guard-3-8B-GGUF",
        "hf_id": "meta-llama/Llama-Guard-3-8B",
        "template": "llama_guard_3", "n_layers": 32,
        "precisions": LLAMA_8B_PRECISIONS,
        "role": "reference",
    },
    "qwen3guard-gen-8b": {
        "kind": GENERATIVE, "backend": GGUF,
        "repo": "mradermacher/Qwen3Guard-Gen-8B-GGUF",
        "hf_id": "Qwen/Qwen3Guard-Gen-8B",
        "template": "qwen3guard_gen", "n_layers": 36,
        "precisions": QWEN_8B_PRECISIONS,
        "role": "replication",
    },
    "llama-guard-3-1b": {
        "kind": GENERATIVE, "backend": GGUF,
        "repo": "mradermacher/Llama-Guard-3-1B-GGUF",
        "hf_id": "meta-llama/Llama-Guard-3-1B",
        "template": "llama_guard_3", "n_layers": 16,
        "precisions": LLAMA_1B_PRECISIONS,
        "role": "size-axis",
    },
    # The critical comparison.  An encoder head is trained to emit a graded
    # probability rather than a token, which is the whole point of including
    # it.  TASK ALIGNMENT WARNING: many available moderation encoders score
    # *toxicity*, which is not the same label as *harmful request*.  Check the
    # label mapping against harmbench/xstest before trusting any comparison
    # that uses this family -- scripts/label_audit.py is the tool for it.
    "encoder-moderation": {
        "kind": ENCODER, "backend": HF_SEQCLS,
        "repo": None,
        "hf_id": "KoalaAI/Text-Moderation",   # VERIFY before use
        "template": None, "n_layers": None,
        "precisions": _hf_only(0.4),
        "role": "encoder-comparison",
    },
}

DEFAULT_FAMILY = "llama-guard-3-8b"

ALIASES = {
    "reference": "llama-guard-3-8b:fp16",
    "llama8b": "llama-guard-3-8b:fp16",
    "llama1b": "llama-guard-3-1b:fp16",
    "qwen8b": "qwen3guard-gen-8b:fp16",
    "encoder": "encoder-moderation:hf",
    # Short precision aliases still resolve against the reference family, so
    # the cross-precision runs stay convenient to launch.
    "fp16": "llama-guard-3-8b:fp16",
    "q8": "llama-guard-3-8b:q8_0",
    "q6": "llama-guard-3-8b:q6_k",
    "q5": "llama-guard-3-8b:q5_k_m",
    "q4": "llama-guard-3-8b:q4_k_m",
    "q3": "llama-guard-3-8b:q3_k_m",
}

PRECISION_ORDER = ["fp16", "hf", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m"]

# --- groups ---------------------------------------------------------------
# The default scope: one guard of each kind, at full precision.  This is what
# phases 1-3 run, and it is deliberately small -- the core result does not
# need a ladder.
GUARD_PANEL = [
    "llama-guard-3-8b:fp16",
    "qwen3guard-gen-8b:fp16",
    "llama-guard-3-1b:fp16",
    "encoder-moderation:hf",
]

# Input to the cross-precision agreement signal.  One family, six rungs.
PRECISION_LADDER = [f"{DEFAULT_FAMILY}:{p}" for p in
                    ("fp16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m")]

ACTIVE_GROUP = "guard-panel"


def parse_key(key: str) -> Tuple[str, str]:
    if key in ALIASES:
        key = ALIASES[key]
    if ":" in key:
        family, precision = key.split(":", 1)
    else:
        family, precision = DEFAULT_FAMILY, key
    if family not in FAMILIES:
        raise ValueError(f"Unknown guard family '{family}'. Known: {sorted(FAMILIES)}")
    if precision not in FAMILIES[family]["precisions"]:
        raise ValueError(
            f"Unknown precision '{precision}' for {family}. "
            f"Known: {sorted(FAMILIES[family]['precisions'])}"
        )
    return family, precision


def get_config(key: str) -> dict:
    family, precision = parse_key(key)
    fam = FAMILIES[family]
    prec = fam["precisions"][precision]
    return {
        "key": f"{family}:{precision}",
        "family": family,
        "precision": precision,
        "kind": fam["kind"],
        "backend": fam["backend"],
        "repo": fam["repo"],
        "hf_id": fam["hf_id"],
        "filename": prec["file"],
        "size_gb": prec["size_gb"],
        "bits": prec["bits"],
        "template": fam["template"],
        "n_layers": fam["n_layers"],
        "role": fam["role"],
    }


def expand(spec: str) -> List[str]:
    if spec == "guard-panel":
        return list(GUARD_PANEL)
    if spec == "precision-ladder":
        return list(PRECISION_LADDER)
    if spec == "generative":
        return [k for k in GUARD_PANEL if get_config(k)["kind"] == GENERATIVE]
    if spec == "encoders":
        return [k for k in GUARD_PANEL if get_config(k)["kind"] == ENCODER]
    if spec.endswith(":all"):
        family = spec.split(":")[0]
        return [f"{family}:{p}" for p in FAMILIES[family]["precisions"]]
    return [get_config(spec)["key"]]


def expand_many(specs: List[str]) -> List[str]:
    seen, out = set(), []
    for spec in specs:
        for key in expand(spec):
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


def sort_keys(keys: List[str]) -> List[str]:
    fam_rank = {f: i for i, f in enumerate(FAMILIES)}
    prec_rank = {p: i for i, p in enumerate(PRECISION_ORDER)}

    def rank(key):
        family, precision = parse_key(key)
        return (fam_rank.get(family, 99), prec_rank.get(precision, 99))

    return sorted(keys, key=rank)


def total_disk_gb(keys: List[str]) -> float:
    return sum(get_config(k)["size_gb"] for k in keys)


def kind_of(key: str) -> str:
    return get_config(key)["kind"]


def family_of(key: str) -> str:
    return parse_key(key)[0]


MODEL_CONFIGS: Dict[str, dict] = {}
for _family, _fam in FAMILIES.items():
    for _precision in _fam["precisions"]:
        _key = f"{_family}:{_precision}"
        MODEL_CONFIGS[_key] = get_config(_key)
for _alias, _target in ALIASES.items():
    MODEL_CONFIGS[_alias] = get_config(_target)
