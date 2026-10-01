# Runbook

What to run, where, in what order, and what decides whether you continue.

Read `study_protocol.md` first if you have not — this file assumes you know
what the gates are asking. `CLAUDE.md` has the rules that are not style
preferences; nothing here overrides them.

---

## Stage 0 — choose the pod

| | Requirement | Why |
|---|---|---|
| **GPU** | ≥ 24 GB VRAM | The largest guard is 16.4 GB at fp16. Guards are scored one at a time, so peak demand is a single model plus its KV cache |
| **Volume** | 60 GB (100 GB if you will run Phase 4) | Panel 34.9 GB · precision ladder 45.8 GB · union of both **64.7 GB** |

> ### Run every phase on the same pod
>
> The cost model prices each guard by its **measured latency**, and latency is
> only comparable within one machine. Every prediction row carries an
> `env_hash`; `analyze.py` reports `MIXED` if a sweep spans environments, and
> the frontier means nothing in that case.
>
> If you lose the pod partway, you either re-score everything on the new one or
> fall back to `--cost-basis size`, which is hardware-independent and cruder.
> Decide which, and say so in the write-up.

---

## Stage 1 — bootstrap, ~20 minutes

```bash
git clone <this repo> && cd guard-uncertainty-routing
python3 -m venv .venv && source .venv/bin/activate
VOLUME=/workspace bash scripts/setup_runpod.sh
huggingface-cli login
```

`setup_runpod.sh` does four things: points `HF_HOME` and `MODEL_WEIGHTS_DIR` at
the persistent volume and persists them to `~/.bashrc`, installs
`requirements.txt`, runs `install_engine.sh`, and prints the environment
fingerprint. Do not `pip install -r requirements.txt` separately — it is
already done, and a second `install_engine.sh` risks replacing a working CUDA
build.

**Never run `pip install llama-cpp-python`.** It installs a CPU-only wheel that
runs the whole experiment on the pod's host CPU without erroring. The scores
would be unaffected but every latency number would be wrong — and latency is
what prices the frontier, so the headline result would be silently invalid.
`install_engine.sh` detects the backend and verifies
`llama_supports_gpu_offload()`.

### Two gates that cost nothing

```bash
python scripts/preflight.py --models guard-panel
```

Expect `0 fail`. Check specifically that **`weights directory` does not warn** —
if it does, `setup_runpod.sh` did not take, `MODEL_WEIGHTS_DIR` is unset, and
your weights will land on the ephemeral container disk and disappear when the
pod restarts.

```bash
python scripts/verify_selective.py
```

Expect **13/13**. This runs every gate against synthetic data whose answer is
known by construction, in both directions. A gate that cannot fail is
decoration, so this is what makes the rest of the study mean anything.

### Datasets

`harmbench` and `xstest` are committed to git. The other three are fetched:

```bash
python scripts/download_datasets.py --datasets toxicchat wildguardtest openai_moderation
```

Measured counts after normalization, which is what the loader returns:

| Dataset | Rows | Harmful |
|---|---:|---:|
| toxicchat | 4,972 | **7.1%** |
| xstest | 450 | 44.4% |
| harmbench | 200 | 100% |
| wildguardtest | 1,699 | 44.4% |
| openai_moderation | 1,665 | 30.9% |

---

## Stage 2 — Phase 1, free signals · ~10 min · **GO / NO-GO**

```bash
python scripts/run_phase.py --phase 1
```

One guard (Llama-Guard-3-8B), `xstest` + `harmbench`, signals 1 and 2 only.

### Read the verdicts

```bash
python -c "
import json; d=json.load(open('results/tables/gates.json'))
print('score range:', d['score_range']['status'], '-', d['score_range']['reason'])
for k,v in d.items():
    if k.startswith('gate_'): print(f\"{v['gate']:>3}  {v['status']:<24} {v.get('reason','')}\")
"
```

In this order:

| Read | Continue if | Stop if |
|---|---|---|
| `score_range` | `USABLE` | `POLARIZED` — scores are pinned at 0 and 1 and there is nothing to rank. The fix is a harder or more borderline prompt set, **not** a weaker claim |
| **S1** | `NATIVE_USABLE` | `NATIVE_DEGENERATE` — **the pipeline is broken.** S1 replicates Safety-Flag, which found native confidence usable on every model tested. A failure here is a bug, not a finding |
| **S2** | any verdict | — |

S2 is already a result either way. `MARGIN_BETTER` means every downstream
system should rank on the raw margin instead of the probability, which is a
one-line change with a measurable benefit.

**Do not proceed past a failing S1.** Phase 3 is four hours.

---

## Stage 3 — Phase 2, perturbation · ~1 hr · **GO / NO-GO**

```bash
python scripts/perturb.py --model reference --dataset xstest --n-variants 6
```

```bash
python evaluation/analyze.py
```

This is the decision that saves the most time. Read **S4**:

- **`INDEPENDENT`** — instability carries information the margin cannot
  express. Scale it to the full panel in Phase 3.
- **`REDUNDANT_WITH_MARGIN`** — it is a noisy restatement of a small margin at
  roughly 6x the compute. **Do not scale it.** This replicates Safety-Flag's
  Appendix D.1 negative result on a behavioural signal, which
  `preregistration.md` commitment 1d says is reported as a replication rather
  than buried.

Also check the variant drop rate that `perturb.py` reports. If most variants
fail the semantics-preserved check, the signal is measuring paraphrase quality
rather than model uncertainty.

---

## Stage 4 — Phase 3, full panel · ~4 hrs · **the result**

```bash
python scripts/run_phase.py --phase 3
```

Three guards x five datasets. Add `--resume` if the pod was preempted and
`--evict` if the volume is tight.

```bash
python evaluation/analyze.py --headline-budget 0.05
```

Read **A1** first — it is the crux — then **A2**.

| A1 | A2 | The paper |
|---|---|---|
| `CHEAPER_GUARD_WINS` | any | **Not a better guard.** At matched compute a smaller guard with better selection ships fewer bad decisions than the large guard everyone deploys |
| `BETTER_SELECTION_WINS` | any | **Spend on selection.** Same guard, same compute, better routing |
| `DEFAULT_IS_BEST` | `BUDGET_DEPENDENT` | **It depends.** The crossover point is the result |
| `DEFAULT_IS_BEST` | `ONE_POLICY_WINS` | **A bounded negative.** The default is right, and we priced every alternative |
| `NOT_EVALUABLE` | any | Too few errors, or no priced alternative. **Scale the sample** — do not reinterpret |

The paper's main figure is `results/figures/oversight_frontier.png`.

Remember the dissociation: **a signal can win S3 and still lose A1**, if the
compute it cost would have bought a better guard instead. Report both.

---

## Stage 5 — Phase 4, precision ladder · ~5 hrs · optional

Only needed for signal 5 (cross-precision agreement). Costs another 45.8 GB.

```bash
python scripts/run_phase.py --phase 4 --evict
```

Skip it if A1 and A2 already resolved cleanly. It feeds a supporting signal,
not the result.

---

## Stage 6 — write-up

One rule governs everything:

```bash
cat results/tables/claims_to_evidence.csv
```

**Never write a claim this table has not marked `LICENSED`.** It is generated,
not curated. `S5` will read `NOT_TESTED` — that is correct, and
`preregistration.md` Amendment 1 explains it.

What to report alongside the headline, because the pre-registration commits to
it:

- Residual risk is the claim; errors-caught, risk reduction and AURC are
  context (commitment 1b).
- The full 1-20% review regime, with 50% computed **only** so one point is
  directly comparable to Safety-Flag's Risk@0.5. No conclusion rests on it
  (commitment 1c).
- Every signal result carries its compute price (commitment 1).
- The frontier is reported in both units, with no exchange rate between compute
  and human time (commitment 8).

---

## Troubleshooting

| Symptom | Cause | Do this |
|---|---|---|
| Pod preempted mid-run | — | `--resume` skips prompts already scored |
| Volume filling up | Weights accumulating | `--evict` deletes each GGUF after it is scored |
| `analyze.py` says `MIXED` environments | Scoring was split across machines | Re-score on one pod, or switch to `--cost-basis size` and say so |
| Suspiciously good AURC | Ties, not ranking | Check `tie_fraction` in `signal_performance.csv`. Above 0.90 the signal is reporting input order |
| Latency far higher than expected | CPU-only llama.cpp build | Re-run `install_engine.sh`; confirm `preflight.py` shows `engine GPU support True` |
| Weights vanish after restart | `MODEL_WEIGHTS_DIR` unset | Re-run `setup_runpod.sh` with `VOLUME=` pointing at the persistent volume |
| An encoder crashes mid-forward | Reduced-precision dtype | The family must pin `torch_dtype` in `registry.py`; DeBERTa needs float32 |

---

## Things that are never the answer

From `CLAUDE.md`, repeated because this is where the temptation arrives:

1. **Do not relax a threshold to make a gate pass.** Report that it failed.
2. **Do not edit `preregistration.md` once real predictions exist.** It is a
   commitment device.
3. **Do not write an unlicensed claim**, however obvious it looks.
4. **Do not add an encoder without running `check_encoder_alignment.py` first.**
   The last one scored 0.599 and would have made Gate S5 report an artefact.
5. **Re-run `verify_selective.py` after touching `gates.py`, `selective.py`,
   `signals.py` or `allocation.py`.** It must stay at 13/13.
