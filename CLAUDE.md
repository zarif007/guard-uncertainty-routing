# Orientation

Read this before changing anything. It is written for a person or an agent
arriving cold.

## What this is

A research codebase for one question:

> A guard model filters harmful messages and makes mistakes. A human can review
> only a small fraction of its decisions. **Given a fixed budget, how should it
> be spent — on a bigger guard, on checking the guard, or on more review?**

A **policy** is a (guard, signal) pair: which model decides, and how the shaky
cases get picked out for a human. Every policy has a compute price and a human
price. The study prices them and compares them on **residual risk** — mistakes
still reaching users after review.

## Current state — read this before trusting any number

**Nothing has run on a real model.** `results/` is empty. Every number produced
so far is synthetic, generated to verify the gates fire correctly. Phase 1
produces the first real measurements.

The nearest prior work is Safety-Flag (arXiv 2609.19072), read in full and
recorded in `docs/related_work.md` §0. It measured how well a guard's *own*
confidence ranks its errors. It contains no cost analysis, does not report the
1-20% review regime, and has no encoder classifiers. The first two absences are
what this study occupies; the third it leaves open, because no aligned encoder
was available (`preregistration.md` Amendment 1).

## Start here, in order

| File | What it gives you |
|---|---|
| `docs/study_protocol.md` | The whole study in plain language. Start here |
| `docs/preregistration.md` | Which outcome licenses which claim. **Fixed before data** |
| `docs/related_work.md` | Literature, and precisely where we differ |
| `README.md` | Operating manual: install, run, interpret |
| `docs/runbook.md` | What to run, in what order, and what decides whether you continue |

`docs/archive_preregistration_quantization.md` is a historical record of the
study this repo used to be. **Do not update it.** Its value is that it has not
been touched since it was written.

## Rules that are not style preferences

These exist because the study's credibility depends on them.

1. **Never write a claim `results/tables/claims_to_evidence.csv` has not marked
   `LICENSED`.** The table is generated, not curated.

2. **`docs/preregistration.md` is frozen once real predictions exist.** It is a
   commitment device; a pre-registration that can be edited afterwards is worth
   nothing. Amendments go at the top, dated, explaining what changed and why —
   and only while `results/` is still empty.

3. **The thresholds are pre-registered.** `MIN_AURC_IMPROVEMENT = 0.01`,
   `MIN_RESIDUAL_IMPROVEMENT = 0.05`, `GATE_B_AGREEMENT = 0.99`. Changing one
   after seeing data invalidates the gate it governs. If a threshold is wrong,
   say so in writing rather than editing it quietly.

4. **After any edit to `evaluation/gates.py`, `selective.py`, `signals.py` or
   `allocation.py`, run `python scripts/verify_selective.py`.** It asserts every
   gate against data whose answer is known by construction, in both directions.
   It must stay at 13/13. A gate that cannot fail is decoration.

5. **Compute prices are only comparable within one machine.** Every prediction
   row carries an `env_hash`. If a sweep spans environments, the latency-based
   cost model is invalid and the frontier means nothing.

## Repo map

```
evaluation/
  allocation.py    policies, prices, residual risk, the frontier   <- the result
  selective.py     risk-coverage, AURC, errors-caught-at-budget
  signals.py       the five uncertainty signals
  gates.py         A1-A2 (allocation), S1-S5 (signals), claims table
  analyze.py       runs everything, writes tables and figures
  score_range.py   do the scores have any spread?  measured, not assumed
models/
  registry.py      guard families, by KIND (generative vs encoder) and backend
  llm_loader.py    GGUF scorer (llama.cpp)
  hf_loader.py     transformers scorer, incl. EncoderGuard
scripts/
  run_model.py / run_phase.py   scoring and orchestration
  perturb.py                    reworded variants, with a diversity guard
  verify_selective.py           gates vs known-truth data   <- run after edits
  check_encoder_alignment.py    is this encoder answering OUR question?
  preflight.py                  environment, models, datasets, templates
```

## Running it

```bash
python scripts/preflight.py --models guard-panel     # environment + model files
python scripts/verify_selective.py                   # gates, no GPU
python scripts/run_phase.py --phase 1                # first real data, ~10 min
python evaluation/analyze.py                         # tables + figures, no GPU
```

## Gotchas that have already cost time

- **Never `pip install llama-cpp-python` directly.** It gives a CPU-only build
  that runs silently on the host CPU of a GPU pod. Use
  `bash scripts/install_engine.sh`, which detects the backend and verifies
  `llama_supports_gpu_offload()`.
- **On a pod, set `MODEL_WEIGHTS_DIR` to the persistent volume.** `HF_HOME`
  alone is not enough — GGUFs are fetched with an explicit `cache_dir` that
  takes precedence, so weights land on the ephemeral container disk and vanish.
  `scripts/setup_runpod.sh` does this.
- **The encoder is retired and gate S5 is not tested** (2026-10-01).
  `KoalaAI/Text-Moderation` scored AUROC 0.599 against our label — a toxicity
  head, not a harmful-request head. It is out of `GUARD_PANEL`, its family
  entry is kept unused for the record, and `preregistration.md` Amendment 1
  has the reasoning. `gate_s5` returns `NOT_EVALUABLE` on a single-kind panel
  and the claim shows as `NOT_TESTED`, so nothing downstream needed changing.
  **Do not add an encoder without running
  `scripts/check_encoder_alignment.py` first** — many moderation encoders
  score *toxicity*, and a polite request for dangerous information scores low.
- **An encoder family must pin `torch_dtype` in `registry.py`.** DeBERTa
  computes its attention in float32 whatever the weights are, so loading it at
  the device default (float16) raises a dtype mismatch mid-forward.
  `check_dtype_supported` permits float32 below 2 GB; above that it is a
  memory constraint and still refused.
- **`p_unsafe` is `sigmoid(margin)`**, and past a margin of ~37 the float64
  sigmoid returns exactly 1.0. Every prompt beyond that is tied and unrankable.
  This is why `conf_margin` exists as a separate signal and why S2 is worth
  testing for free.
- **`PIN_EPS = 1e-3` in `score_range.py` is a screening heuristic**, not a
  measurement. It flags any margin past ~6.9 as pinned, far inside the range
  where ranking information survives. It must never decide a gate.

## If you are an agent

Work through `docs/study_protocol.md` first — it explains the design in plain
language and names the assumptions. Prefer running `verify_selective.py` over
reasoning about whether a change is safe. Do not relax a threshold to make a
gate pass; report that it failed.
