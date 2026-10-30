from typing import Dict, List, Tuple

GGUF = "gguf"

GENERATIVE = "generative"
ENCODER = "encoder"

FAMILIES = {
    "llama-guard-3-8b": {
        "kind": GENERATIVE, "backend": GGUF,
        "repo": "mradermacher/Llama-Guard-3-8B-GGUF",
        "hf_id": "meta-llama/Llama-Guard-3-8B",
        "template": "llama_guard_3", "n_layers": 32,
        "precisions": {"fp16": {"file": "Llama-Guard-3-8B.f16.gguf", "size_gb": 16.07, "bits": 16.0}},
        "role": "reference",
    },
    "qwen3guard-gen-8b": {
        "kind": GENERATIVE, "backend": GGUF,
        "repo": "mradermacher/Qwen3Guard-Gen-8B-GGUF",
        "hf_id": "Qwen/Qwen3Guard-Gen-8B",
        "template": "qwen3guard_gen", "n_layers": 36,
        "precisions": {"fp16": {"file": "Qwen3Guard-Gen-8B.f16.gguf", "size_gb": 16.4, "bits": 16.0}},
        "role": "replication",
    },
    "llama-guard-3-1b": {
        "kind": GENERATIVE, "backend": GGUF,
        "repo": "mradermacher/Llama-Guard-3-1B-GGUF",
        "hf_id": "meta-llama/Llama-Guard-3-1B",
        "template": "llama_guard_3", "n_layers": 16,
        "precisions": {"fp16": {"file": "Llama-Guard-3-1B.f16.gguf", "size_gb": 2.48, "bits": 16.0}},
        "role": "size-axis",
    },
    "encoder-moderation": {
        "kind": ENCODER, "backend": None,
        "repo": None,
        "hf_id": "KoalaAI/Text-Moderation",
        "template": None, "n_layers": None,
        "precisions": {"hf": {"file": None, "size_gb": 0.4, "bits": 16.0}},
        "role": "retired",
    },
}

DEFAULT_FAMILY = "llama-guard-3-8b"

ALIASES = {
    "llama8b": "llama-guard-3-8b:fp16",
    "llama1b": "llama-guard-3-1b:fp16",
    "qwen8b": "qwen3guard-gen-8b:fp16",
}

PRECISION_ORDER = ["fp16", "hf"]

GUARD_PANEL = [
    "llama-guard-3-8b:fp16",
    "qwen3guard-gen-8b:fp16",
    "llama-guard-3-1b:fp16",
]


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


MODEL_CONFIGS: Dict[str, dict] = {}
for _family, _fam in FAMILIES.items():
    for _precision in _fam["precisions"]:
        _key = f"{_family}:{_precision}"
        MODEL_CONFIGS[_key] = get_config(_key)
for _alias, _target in ALIASES.items():
    MODEL_CONFIGS[_alias] = get_config(_target)
