from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

TIER_A = "A"


@dataclass
class DatasetSpec:
    name: str
    tier: str
    hf_id: Optional[str]
    config: Optional[str]
    split: str
    normalize: Callable
    level: str = "prompt"
    local_csv: Optional[str] = None
    notes: str = ""
    approx_rows: int = 0
    verified: bool = False
    gated: bool = False


def _row(prompt, label, category=None, language="en", source_extra=None):
    out = {"prompt": prompt, "ground_truth": label, "category": category, "language": language}
    if source_extra:
        out.update(source_extra)
    return out


def norm_harmbench(ds):
    return [_row(r.get("prompt") or r.get("Behavior"), "unsafe", r.get("category") or r.get("SemanticCategory"))
            for r in ds]


def norm_xstest(ds):
    out = []
    for r in ds:
        prompt = r.get("prompt") or r.get("Prompt")
        raw = (r.get("label") or r.get("type") or "").lower()
        label = "unsafe" if raw.startswith("contrast") or raw == "unsafe" else "safe"
        out.append(_row(prompt, label, r.get("type")))
    return out


def norm_toxicchat(ds):
    out = []
    for r in ds:
        label = "unsafe" if int(r.get("toxicity", 0)) == 1 else "safe"
        out.append(_row(r.get("user_input"), label, "jailbreak" if r.get("jailbreaking") else None))
    return out


def norm_wildguard(ds):
    out = []
    for r in ds:
        harm = (r.get("prompt_harm_label") or "").lower()
        if harm not in ("harmful", "unharmful"):
            continue
        out.append(_row(r.get("prompt"), "unsafe" if harm == "harmful" else "safe", r.get("subcategory")))
    return out


OPENAI_MOD_FLAGS = ["S", "H", "V", "HR", "SH", "S3", "H2", "V2"]


def norm_openai_moderation(ds):
    out = []
    for r in ds:
        flags = [f for f in OPENAI_MOD_FLAGS if int(r.get(f, 0) or 0) == 1]
        out.append(_row(r.get("prompt"), "unsafe" if flags else "safe", ",".join(flags) or None))
    return out


DATASETS: Dict[str, DatasetSpec] = {
    "harmbench": DatasetSpec("harmbench", TIER_A, "walledai/HarmBench", "standard", "train",
                             norm_harmbench, approx_rows=200, verified=True,
                             local_csv="datasets/harmbench/harmbench.csv"),
    "xstest": DatasetSpec("xstest", TIER_A, "walledai/XSTest", None, "test",
                          norm_xstest, approx_rows=450, verified=True,
                          local_csv="datasets/xstest/xstest.csv"),
    "toxicchat": DatasetSpec("toxicchat", TIER_A, "lmsys/toxic-chat", "toxicchat0124", "test",
                             norm_toxicchat, approx_rows=4972,
                             notes="real user traffic, naturally low toxic base rate"),
    "wildguardtest": DatasetSpec("wildguardtest", TIER_A, "allenai/wildguardmix", "wildguardtest", "test",
                                 norm_wildguard, approx_rows=1699, gated=True,
                                 notes="gated (auto-approve); accept terms on the hub first"),
    "openai_moderation": DatasetSpec("openai_moderation", TIER_A,
                                     "mmathys/openai-moderation-api-evaluation", None, "train",
                                     norm_openai_moderation, approx_rows=1665),
}


def get_spec(name: str) -> DatasetSpec:
    key = name.lower()
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Known: {sorted(DATASETS)}")
    return DATASETS[key]
