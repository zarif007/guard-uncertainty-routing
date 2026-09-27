import math
import os
from typing import Dict, List, Optional, Sequence

import torch

from evaluation.hardware import check_dtype_supported, default_torch_dtype, torch_device
from models.registry import get_config
from models.templates import get_template, template_fingerprint


class HFGuard:
    """
    PyTorch scorer for a generative guard loaded through transformers.

    dtype
    -----
    bfloat16 by default (float16 on MPS).  float32 is rejected: an 8B model
    in float32 needs ~32 GB, beyond the experiment hardware.  bfloat16 is
    preferred over float16 because it keeps float32's exponent range, so
    logits cannot saturate -- which matters here, because a saturated logit
    is indistinguishable from a confident one and this study is about
    telling those apart.
    """

    def __init__(self, family: str = "llama-guard-3-8b", dtype: str = "auto",
                 device: str = "auto", hf_token: Optional[str] = None,
                 cache_dir: str = "./models/hf"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = torch_device(device)
        if dtype == "auto":
            dtype = default_torch_dtype(device)
        check_dtype_supported(dtype, device)
        hf_token = hf_token or os.environ.get("HF_TOKEN") or None

        config = get_config(f"{family}:fp16")
        self.family = family
        self.hf_id = config["hf_id"]
        self.template = get_template(config["template"])
        self.template_fingerprint = template_fingerprint(self.template)
        self.device = device
        self.torch_dtype = getattr(torch, dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_id, token=hf_token, cache_dir=cache_dir
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            token=hf_token,
            cache_dir=cache_dir,
        ).to(device)
        self.model.eval()

        self.safe_ids = self._resolve(self.template.safe_variants)
        self.unsafe_ids = self._resolve(self.template.unsafe_variants)
        if set(self.safe_ids) & set(self.unsafe_ids):
            raise RuntimeError("safe and unsafe label tokens collide")

        self._prefix_cache = None
        self._prefix_len = 0
        self._cache_supports_crop = False

    def _resolve(self, variants) -> List[int]:
        ids = []
        for v in variants:
            toks = self.tokenizer.encode(v, add_special_tokens=False)
            if toks and toks[0] not in ids:
                ids.append(toks[0])
        return ids

    def layers(self):
        for attr in ("model.layers", "transformer.h", "model.decoder.layers"):
            obj = self.model
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        raise RuntimeError("Could not locate transformer layers on this architecture")

    def lm_head_weight(self) -> torch.Tensor:
        return self.model.get_output_embeddings().weight.data

    @torch.no_grad()
    def build_prefix_cache(self):
        ids = self.tokenizer(
            self.template.cacheable_prefix(), return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)
        out = self.model(ids, use_cache=True)
        self._prefix_cache = out.past_key_values
        self._prefix_len = ids.shape[1]
        self._cache_supports_crop = hasattr(self._prefix_cache, "crop")

    def invalidate_prefix_cache(self):
        self._prefix_cache = None
        self._prefix_len = 0
        self._cache_supports_crop = False

    def _restore_prefix_cache(self):
        """
        Return the KV cache to just the prefix after a prompt has extended it.

        Modern transformers caches (DynamicCache) are mutated in place by the
        forward pass, so the prefix has to be restored between prompts.  A
        crop is O(1) slicing; the deep copy it replaces allocated a fresh copy
        of the whole cache on every single prompt, which on GPU is a VRAM
        allocation per call.
        """
        if self._prefix_cache is None:
            return
        if self._cache_supports_crop:
            self._prefix_cache.crop(self._prefix_len)
        # Legacy tuple caches are immutable: the forward pass builds new
        # tensors and leaves the originals alone, so nothing to restore.

    @torch.no_grad()
    def predict_score(self, prompt: str) -> Dict[str, float]:
        tail = self.template.tail(prompt)
        if self._prefix_cache is not None:
            ids = self.tokenizer(tail, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            out = self.model(ids, past_key_values=self._prefix_cache, use_cache=True)
            self._restore_prefix_cache()
        else:
            full = self.template.render(prompt)
            ids = self.tokenizer(full, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            out = self.model(ids, use_cache=False)

        logits = out.logits[0, -1, :].float()
        logit_safe = float(logits[self.safe_ids].max())
        logit_unsafe = float(logits[self.unsafe_ids].max())
        margin = logit_unsafe - logit_safe
        return {
            "prediction": "unsafe" if margin > 0 else "safe",
            "p_unsafe": 1.0 / (1.0 + math.exp(-max(min(margin, 500), -500))),
            "logit_safe": logit_safe,
            "logit_unsafe": logit_unsafe,
            "margin": margin,
            "raw_output": self.tokenizer.decode([int(logits.argmax())]),
        }

    def score_frame(self, prompts) -> List[Dict[str, float]]:
        return [self.predict_score(p) for p in prompts]


# --------------------------------------------------------------------------


# Label names that mean "this prompt is harmful", lowercased and matched as
# substrings against the model's own id2label.  Moderation encoders do not
# agree on a vocabulary, so this is a mapping decision, it is recorded in the
# run metadata, and it is exactly the thing scripts/label_audit.py exists to
# check.  See the task-alignment warning in models/registry.py.
DEFAULT_UNSAFE_LABELS = (
    "unsafe", "harmful", "toxic", "flagged", "injection", "jailbreak",
    "hate", "harassment", "violence", "sexual", "self-harm", "selfharm",
)
DEFAULT_SAFE_LABELS = ("safe", "ok", "benign", "clean", "neutral", "unflagged")


class EncoderGuard:
    """
    Scorer for an encoder classifier with a real probability head.

    This is the comparison the study turns on.  A generative guard writes the
    token "safe" or "unsafe" and we read that token's logit, so its confidence
    is a by-product of next-token prediction trained on hard labels.  An
    encoder head is trained with a proper scoring rule to emit a graded
    probability, so it has a reason to produce usable resolution and the
    generative guard does not.

    The returned dict matches LLMGuard.predict_score exactly, so run_model.py,
    the signals and the gates need no special casing.

    Label mapping is the weak point, not the model.  Many moderation encoders
    score *toxicity*, which is not the same question as *is this a harmful
    request*.  `label_mapping` in the run metadata records what was assumed;
    audit it before trusting any comparison that uses this guard.
    """

    def __init__(self, model_key: str, device: str = "auto", dtype: str = "auto",
                 hf_token: Optional[str] = None, cache_dir: str = None,
                 unsafe_labels: Sequence[str] = DEFAULT_UNSAFE_LABELS,
                 safe_labels: Sequence[str] = DEFAULT_SAFE_LABELS,
                 max_length: int = 512):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        config = get_config(model_key)
        self.config = config
        self.key = config["key"]
        self.hf_id = config["hf_id"]
        self.max_length = max_length

        self.device = torch_device(device)
        if dtype == "auto":
            dtype = default_torch_dtype(self.device)
        check_dtype_supported(dtype, self.device)
        hf_token = hf_token or os.environ.get("HF_TOKEN") or None
        cache_dir = cache_dir or os.environ.get("MODEL_WEIGHTS_DIR") or "./models/hf"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_id, token=hf_token, cache_dir=cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.hf_id, token=hf_token, cache_dir=cache_dir,
            torch_dtype=getattr(torch, dtype)).to(self.device).eval()

        id2label = getattr(self.model.config, "id2label", None) or {}
        self.labels = {int(i): str(name) for i, name in id2label.items()}
        self.unsafe_ids, self.safe_ids = self._map_labels(unsafe_labels, safe_labels)

    def _map_labels(self, unsafe_patterns, safe_patterns):
        unsafe, safe = [], []
        for idx, name in self.labels.items():
            low = name.lower()
            if any(p in low for p in unsafe_patterns):
                unsafe.append(idx)
            elif any(p in low for p in safe_patterns):
                safe.append(idx)

        # A binary head with uninformative names (LABEL_0 / LABEL_1) is the
        # common case and cannot be resolved by name.  Refuse rather than
        # guess which way round it is -- a silent inversion would make the
        # encoder look catastrophically miscalibrated and we would report it
        # as a finding.
        if not unsafe or not safe:
            raise RuntimeError(
                f"Could not map {self.hf_id} labels {self.labels} onto safe/unsafe.\n"
                f"Pass unsafe_labels=/safe_labels= explicitly, and record the "
                f"choice in the methods section."
            )
        return unsafe, safe

    def predict_score(self, prompt: str) -> dict:
        import torch as _torch

        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=self.max_length).to(self.device)
        with _torch.no_grad():
            logits = self.model(**enc).logits[0].float().cpu().numpy()

        logit_unsafe = float(max(logits[i] for i in self.unsafe_ids))
        logit_safe = float(max(logits[i] for i in self.safe_ids))
        margin = logit_unsafe - logit_safe
        p_unsafe = (1.0 / (1.0 + math.exp(-margin)) if abs(margin) < 500
                    else float(margin > 0))
        top = int(logits.argmax())
        return {
            "eval_tokens": int(enc["input_ids"].shape[-1]),
            "context_tokens": int(enc["input_ids"].shape[-1]),
            "prediction": "unsafe" if margin > 0.0 else "safe",
            "p_unsafe": p_unsafe,
            "logit_safe": logit_safe,
            "logit_unsafe": logit_unsafe,
            "logit_controversial": float("nan"),
            "p_controversial": float("nan"),
            "margin": margin,
            "raw_output": self.labels.get(top, str(top)),
        }

    def predict(self, prompt: str) -> str:
        return self.predict_score(prompt)["prediction"]

    def token_report(self) -> str:
        return (f"labels: {self.labels}\n"
                f"unsafe -> {[self.labels[i] for i in self.unsafe_ids]}\n"
                f"safe   -> {[self.labels[i] for i in self.safe_ids]}")

    def prefix_token_count(self) -> int:
        return 0  # no cacheable prefix; an encoder reads the prompt directly

    def run_metadata(self) -> dict:
        return {
            "model_key": self.key,
            "precision": self.config["precision"],
            "family": self.config["family"],
            "kind": self.config["kind"],
            "bits": self.config["bits"],
            "size_gb": self.config["size_gb"],
            "repo": self.config["repo"],
            "filename": None,
            "hf_id": self.hf_id,
            "loader": self.config["backend"],
            "template_name": None,
            "template_fingerprint": "encoder-none",
            "template_source": "n/a (no chat template; the encoder reads the raw prompt)",
            "template_deviations": [],
            "controversial_policy": None,
            "prefix_tokens": 0,
            "prefix_cache": False,
            "max_length": self.max_length,
            # The audit trail for the mapping decision.
            "label_mapping": {"all": self.labels,
                              "unsafe_ids": self.unsafe_ids,
                              "safe_ids": self.safe_ids},
            "n_ctx": self.max_length,
            "n_gpu_layers": None, "n_threads": None,
            "n_batch": None, "flash_attn": False,
        }
