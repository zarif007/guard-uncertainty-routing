# Buy the Better Guard

Code and data for *Buy the Better Guard: What a Fixed Oversight Budget Buys in
Safety Moderation*. Read the paper:
[`paper/buy_the_better_guard.pdf`](paper/buy_the_better_guard.pdf).

> **Content warning.** The datasets and prediction files contain harmful,
> offensive and unsafe prompts. They are here for safety research only.

## The question

A guard model filters harmful messages and makes mistakes. A human can review
only a small fraction of its decisions. Given a fixed budget, should it be
spent on a bigger guard, on checking the guard, or on more human review?

A **policy** is a guard plus a routing signal that picks which decisions go to
a human. Every policy is priced in compute and in review budget, and compared
on **residual risk**: the mistakes still reaching users after review.

## The answer

Pooled over five datasets, at 1%, 2%, 5% and 50% review, the strongest single
guard ranked by its own confidence ships the fewest mistakes. A three-guard
panel wins only between 10% and 20% review, at 2.5x the compute. The smallest
guard never wins.

Per dataset the picture is more mixed. Which guard is best depends on the
traffic, and on ToxicChat, the only set with a realistic harmful rate, the
panel wins from 2% to 20% review.

The alternatives lose because no available signal can tell which decisions
are wrong. Perfect routing between two of the guards would cut error by 26.3%
at half the compute, and the best real signal captures none of it.

## Repository layout

```
evaluation/
  analyze.py       gates, tables and claims (results/tables/)
  paper.py         exploratory analyses and the seven paper figures
  allocation.py    policies, prices, residual risk
  selective.py     risk-coverage, AURC, errors caught at a budget
  signals.py       the four routing signals
  gates.py         pre-registered gates A1-A2 and S1-S5
models/            guard registry, prompt templates, llama.cpp scorer
scripts/
  run_phase.py     score guards on datasets (GPU)
  run_model.py     score one guard on one dataset (GPU)
  perturb.py       score reworded variants (GPU)
  download_datasets.py
  verify_selective.py   checks every gate against synthetic data
  install_engine.sh     builds llama-cpp-python for your backend
datasets/          HarmBench and XSTest (the other three are downloaded)
results/
  predictions/     26,958 guard predictions, plus perturbation runs
  tables/          generated tables, gates.json, claims_to_evidence.csv
paper/
  buy_the_better_guard.pdf   the paper
  main.tex                   LaTeX source (compiles with pdfLaTeX + BibTeX)
  references.bib             bibliography
  figures/                   the seven figures, numbered as in the paper
docs/
  preregistration.md    hypotheses and thresholds, fixed before data
  study_protocol.md     the design in plain language
  findings.md           where every number comes from
```

## Reproduce the paper

No GPU is needed. The committed predictions are enough.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/verify_selective.py
python evaluation/analyze.py
python evaluation/paper.py
```

`verify_selective.py` runs every gate on synthetic data whose answer is known,
in both directions. It should report 13/13.

`paper.py` needs the downloaded datasets for one panel of figure 6 (usable
rewordings per dataset). Without them it skips those datasets. To get them,
accept the WildGuardMix terms on Hugging Face, log in, and run:

```bash
huggingface-cli login
python scripts/download_datasets.py
```

## Rerun the guards

This needs a GPU. All published numbers come from one NVIDIA L40S
(`env_hash 6ccc83087810`). Compute prices are only comparable within one
machine, and the analysis refuses to mix them.

```bash
bash scripts/install_engine.sh
export MODEL_WEIGHTS_DIR=/path/to/large/disk
python scripts/run_phase.py --phase 3
python scripts/perturb.py --model llama-guard-3-8b:fp16 --dataset xstest
python scripts/perturb.py --model llama-guard-3-8b:fp16 --dataset harmbench
```

Do not `pip install llama-cpp-python` directly. That gives a CPU-only build,
which runs silently on the CPU of a GPU machine. `install_engine.sh` builds the
right one and checks that GPU offload works.

## Pre-registration

`docs/preregistration.md` fixes the hypotheses, thresholds and gates. It last
changed in commit `7a0b7cc` (2026-10-01). The first real predictions arrived
in commit `2d9076b` (2026-10-02). `results/tables/claims_to_evidence.csv` is
generated, not written by hand, and marks which claims the gates license.

## Data

| Dataset | Source | License | Prompt text in this repo |
|---|---|---|---|
| HarmBench | `walledai/HarmBench` | MIT | yes |
| XSTest | `walledai/XSTest` | CC BY 4.0 | yes |
| ToxicChat | `lmsys/toxic-chat` (toxicchat0124) | CC BY-NC 4.0 | no |
| WildGuardTest | `allenai/wildguardmix` | ODC-BY, gated (AI2 Responsible Use) | no |
| OpenAI Moderation | `mmathys/openai-moderation-api-evaluation` | MIT | no |

For the last three, the prediction files keep `prompt_id` and `prompt_chars`
but not the prompt text. `prompt_id` is `<dataset>_<row>`, where the row is the
position in the file `download_datasets.py` writes, so the text joins back
exactly.

## License

Code is MIT licensed (see `LICENSE`). Dataset content stays under each
dataset's own license, listed above.

## Citation

See `CITATION.cff`.
