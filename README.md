# Guard Uncertainty Routing

How should a fixed oversight budget be spent?

> **The unit of analysis is the deployment, not the model.** A guard, a compute
> budget, and a reviewer with limited hours. Every message gets a decision one
> way or another. What is the best way to spend what you have?

Plain-language walkthrough: `docs/study_protocol.md`.
What each outcome licenses: `docs/preregistration.md`.
Literature and differentiation: `docs/related_work.md`.

---

## The problem

A **policy** is a (guard, signal) pair: which model makes the call, and how the
uncertain cases get picked out for a human. Every policy has a compute price
and a human price, and both buy the same thing — fewer bad decisions reaching
production.

The headline experiment is a head-to-head at **matched compute**:

| | compute | selection |
|---|---|---|
| what everyone deploys | large guard | its own confidence (free) |
| the alternative | small guard | perturbation instability (~6x inference) |

A 1B guard costs roughly a sixth of an 8B one, so for the same money you can
run the small guard *and* reword every prompt six times to see whether its
verdict holds. **Which ships fewer bad decisions?** Nobody has measured it.

### The objective: residual risk

Not accuracy, and not errors caught. Both flatter a weak setup. What counts is
what survives review:

```
residual_errors = total_errors(guard) - errors_caught(guard, signal, budget)
```

A guard making 200 mistakes and catching 160 ships more harm than one making
50 and catching 30. A weak guard has to make up its deficit through selection
before it counts as a win.

### No exchange rate

Compute and human attention are not denominated in the same thing, and any
rate we pick is arguable and dates badly. Nothing here converts one into the
other. Policies are compared on a two-dimensional frontier and the reader
brings their own prices.

---

## The signals

Instruments for a policy, not the contribution.

| # | Signal | What it reads | Cost |
|---|---|---|---|
| 1 | `conf_native` | The guard's probability, distance from the 0.5 cut | free — **the baseline** |
| 2 | `conf_margin` | The raw logit gap, before the sigmoid squashes it | free |
| 3 | `conf_stability` | Does the verdict survive rewording the message? | ~k x inference |
| 4 | `conf_model_agree` | Do several different guards agree? | +1 run per guard |
| 5 | `conf_precision_agree` | Does one guard agree with itself across quantizations? | free if the ladder ran |

Signals 3, 4 and 5 never read the confidence score, which is the point: they
remain computable on a guard whose scores are pinned at 0 and 1.

**Why signal 2 is not redundant.** `p_unsafe` is `sigmoid(margin)`. Past a
margin of about 37 the float64 sigmoid returns exactly `1.0`, so every prompt
beyond that collapses into a tie. The ranking survives in the margin and dies
in the probability. Comparing 1 against 2 measures that loss and costs nothing.

---

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/install_engine.sh
```

`llama-cpp-python` is **not** in `requirements.txt`. The correct wheel is
backend-specific, and a plain `pip install` gives a CPU-only build that would
silently run the whole experiment on the host CPU of a GPU pod.
`install_engine.sh` detects the backend and verifies
`llama_supports_gpu_offload()`.

Encoder guards run through `transformers`, not llama.cpp:

```bash
pip install -r requirements-encoder.txt
```

---

## Setup

**1 — Accept the gated datasets** (one click each, auto-approved).

```
https://huggingface.co/datasets/allenai/wildguardmix
https://huggingface.co/datasets/lmsys/lmsys-chat-1m
```

**2 — Authenticate.** `huggingface-cli login`

**3 — Bootstrap the pod.**

```bash
VOLUME=/workspace bash scripts/setup_runpod.sh
```

**4 — Preflight.** Checks engine build, GPU offload, HF auth, every model
filename, disk headroom, template fingerprints. Non-zero exit on anything fatal.

```bash
python scripts/preflight.py --models guard-panel
```

**5 — Verify the gates before spending GPU hours.** Runs every gate against
synthetic score distributions whose truth is known by construction. A test
that cannot fail on data built to fail it is decoration.

```bash
python scripts/verify_selective.py
```

**6 — Datasets, then run.**

```bash
python scripts/download_datasets.py --core
MODELS="guard-panel" bash scripts/run_everything.sh
```

Read `results/tables/gates.json` before anything else.

---

## Phases

| Phase | Command | Decides |
|---|---|---|
| 0 Gate validation | `python scripts/verify_selective.py` | do the gates fire on known-truth data? |
| 1 Free signals | `python scripts/run_phase.py --phase 1` | S1, S2 — is there a signal, and does the margin beat the probability? |
| 2 Perturbation | `python scripts/perturb.py --model reference --dataset xstest` | S4 — does instability add anything beyond the margin? |
| 3 Full panel | `python scripts/run_phase.py --phase 3` | S3, S5 — the crux, and whether guard kind decides it |
| 4 Precision ladder | `python scripts/run_phase.py --phase 4` | signal 5, reusing the quantization runs |

Phases 1 and 2 are genuine stop points. Phase 2 in particular decides whether
the expensive signal earns its cost before it is scaled.

---

## Guards

Keys are `family:precision`. The axis that matters is **kind**, not bit width.

```
generative (token logit)        llama-guard-3-8b, qwen3guard-gen-8b, llama-guard-3-1b
encoder    (probability head)   encoder-moderation
```

A generative guard's confidence is a by-product of next-token prediction
trained on hard labels. An encoder's is a trained probability. If the two
classes differ systematically in whether their confidence ranks errors, that
is an architectural finding and it goes in the abstract.

Groups: `guard-panel` (default, 4 guards, 35 GB), `precision-ladder`
(6 rungs of the reference family, 46 GB), `generative`, `encoders`.

**Model ids are best-effort.** `preflight.py` checks each against the hub and
names what is actually published. The encoder entry in particular needs
verifying — many moderation encoders score *toxicity*, which is not the same
question as *harmful request*. Run this before any comparison that uses it:

```bash
python scripts/check_encoder_alignment.py --model encoder-moderation:hf
```

It scores a labelled probe set and reports whether the encoder tracks our
label at all, plus the harmful prompts it scored lowest — if those are all
polite requests for dangerous information, it is a toxicity head and the
wrong instrument.

---

## Datasets

| Dataset | Rows | Role |
|---|---:|---|
| `toxicchat` | 5,083 | **headline** — real traffic, realistically low harmful rate |
| `xstest` | 450 | benign-but-borderline; committed to git |
| `harmbench` | 200 | all-harmful; committed to git |
| `wildguardtest` | 1,725 | replication; **gated** (auto-approve) |
| `openai_moderation` | 1,680 | replication |

All five ask one question — given a prompt, is it harmful? — so pooling them
is legitimate. ToxicChat is the headline because routing review only matters
where the base rate is realistic.

---

## Gates

Criteria are fixed in `evaluation/gates.py` before any data is seen, and
`scripts/verify_selective.py` proves each one can fail — in both directions.

### The result: how to spend the budget

**A1 — at matched compute, does anything beat the default?** The default is
what people ship: the largest guard available, ranked by its own confidence. A
win means another policy sends fewer bad decisions to production for the same
compute and the same reviewer hours. Must beat it by 5% relative residual risk;
shipping 0.5% fewer bad decisions does not justify rebuilding a pipeline.
`CHEAPER_GUARD_WINS` / `BETTER_SELECTION_WINS` / `DEFAULT_IS_BEST`.

**A2 — does the right policy depend on the budget?** If one policy wins
everywhere the recommendation is a sentence. If the winner changes, the
recommendation is a rule and the crossover point is the result.

### Supporting: why a policy won

**S1 — is native confidence usable here?** Expected to pass. Replicates
Safety-Flag on our models. A failure means the pipeline is broken.

**S2 — the squashing tax.** Does the raw margin beat the probability? Free to
test, free to act on.

**S3 — does any alternative beat native confidence?** By at least 0.01 AURC,
surviving Holm correction across every signal tested. **A signal can win S3 and
still lose A1**, if the compute it costs would have bought a better guard — that
dissociation is why both are run.

**S4 — is instability independent of the margin?** The guard against S3 passing
for a boring reason.

**S5 — does guard kind decide it?** Generative versus encoder.

`claims_to_evidence.csv` marks every claim `LICENSED`, `NOT_LICENSED` or
`NOT_TESTED`. Never write a claim the table has not licensed.

---

## Layout

```
models/
  registry.py        guard families x kind x backend
  templates.py       official Llama Guard 3 and Qwen3Guard prompts
  llm_loader.py      GGUF scorer: predict_score, prefix KV cache, label tokens
  hf_loader.py       transformers scorer for encoder guards
evaluation/
  allocation.py      the oversight allocation problem: policies, prices, frontier
  selective.py       risk-coverage, AURC, errors-caught-at-budget, signal comparison
  signals.py         the five uncertainty signals
  gates.py           A1-A2 (allocation), S1-S5 (signals), claims-to-evidence
  analyze.py         runs everything, writes tables and figures
  score_range.py     can a threshold move here?  measured, not assumed
  calibration.py     ECE, Brier, temperature (baseline construction only)
  statistical_tests.py  bootstrap, Holm, BH, McNemar, DeLong
  threshold_analysis.py AUROC, AUPRC, ROC, TPR@FPR
  disagreement.py    agreement matrix, flips
  hardware.py        backend detection, GPU metadata
scripts/
  perturb.py         meaning-preserving rewordings, with a diversity guard
  check_encoder_alignment.py  is this encoder answering OUR question?
  verify_selective.py  gates vs synthetic data of known truth
  run_model.py       score one guard on one dataset
  run_phase.py       orchestrate a phase
  preflight.py       environment, models, datasets, templates
  label_audit.py     dataset label-noise annotation sheet
```

---

## Known gaps

- **Perturbation quality is a threat, not a detail.** A paraphrase that
  changes meaning produces instability that is not uncertainty.
  `scripts/perturb.py` records a semantics-preserved check per variant;
  variants that fail are dropped before the signal is computed, and the drop
  rate is reported.
- **Cross-model agreement is confounded.** Different guards have different
  decision boundaries, so disagreement partly measures "these are different
  models" rather than "this case is hard." Cross-*precision* agreement does
  not have this problem, which is why it is preferred.
- **`PIN_EPS = 1e-3` in `score_range.py` is a screening heuristic.** It flags
  any margin beyond ~6.9 as pinned, far inside the range where ranking
  information still exists. It never decides a gate.
- **Qwen3Guard emits three labels.** `--controversial-policy` (default
  `strict`) decides how Controversial folds into the binary decision; all
  three logits are written to the predictions CSV so the choice can be
  revisited without re-running.

## Template provenance

Both prompt templates are transcribed from the model's own `chat_template` —
GGUF metadata for Llama Guard 3, `tokenizer_config.json` for Qwen3Guard — not
from prose docs. Each records its deviations in `models/templates.py`. There
is one, common to both: we score the token after the prompt rather than
generating, so a fixed continuation is appended to put the scoring position
exactly on the label.
