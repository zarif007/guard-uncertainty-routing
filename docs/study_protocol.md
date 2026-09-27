# How this study works

Plain-language protocol. Written for someone who has not read the code.

Companion documents: `preregistration.md` fixes what each outcome licenses,
`related_work.md` is the literature scan.

---

## The short version

AI chatbots have a filter in front of them called a **guard model**. It reads
what a user types and decides whether to block it. Guards get things wrong —
they block harmless messages, and they let harmful ones through.

In a real company, a human reviewer can double-check some of those decisions.
But only a small slice of them, because people are expensive. So you have to
choose: **out of 10,000 messages, which 500 does the human look at?**

The obvious way to choose is to ask the guard how sure it was, and show the
human the ones it was least sure about. A 2026 benchmark called **Safety-Flag**
already tested that. It works — somewhat, and unevenly across models.

**Our question: is there a better way to choose?**

A guard's confidence score is not the only clue that a decision is shaky. You
could reword the message and see whether the verdict flips. You could run
several guards and see whether they disagree. You could run the same guard at
different compression levels and see whether it changes its mind. These cost
more to compute, and nobody has checked whether they are worth it.

So we line up five ways of spotting a shaky decision, give each one the same
human-review budget, and ask which catches the most of the guard's mistakes.

**Either way we get an answer.** If something beats the guard's own confidence,
that is a practical recommendation people can deploy tomorrow. If nothing
does, that settles the question — guard confidence is already near the
practical ceiling, and the expensive alternatives are not worth their cost.

In one line:

> We are looking for the best way to decide which of a safety filter's
> decisions a human should double-check.

---

## 1. The question

A guard model is a filter. It reads a user's message and decides: safe, or
unsafe. It sits in front of a chatbot and blocks the bad stuff.

Guards make mistakes. In a real deployment, a human reviewer can check some of
those decisions — but only a small fraction, because human attention is the
expensive part. So the practical question is:

> **Out of 10,000 messages, which 500 should a human look at?**

Pick well and you catch most of the guard's mistakes. Pick badly and you waste
the reviewer's time on cases the guard already got right.

To pick well, we need a signal that says *"this particular decision is shaky."*
The question of this study is where that signal comes from.

---

## 2. What is already known, and what is not

A 2026 benchmark called **Safety-Flag** (arXiv 2609.19072) already tested the
obvious approach: use the guard's own confidence score. It covered seven
safety benchmarks, six general-purpose language models and four dedicated
guards, and it found that:

- Confidence-based abstention **does** lower error for every model tested.
- But how much it helps varies a lot, depending on how well a given model's
  confidence actually ranks its own errors.
- Temperature scaling fixes calibration numbers **without changing the
  ordering** of predictions — so it makes the confidence score look better
  without making it better at picking cases.

So "does the guard's own confidence help?" is answered. Yes, somewhat.

**What nobody has tested is whether something else works better.**

A guard's confidence score is not the only available signal. We could also ask
how stable its decision is when we reword the message, or whether several
guards disagree with each other. Those signals cost more to compute, and
nobody has checked whether they are worth it.

That is the gap:

> **When a guard's own confidence is not good enough at ranking its errors, can
> other uncertainty signals do better at the same human-review budget?**

---

## 3. Why the question is well posed

This study cannot produce a non-result:

- **Some alternative signal beats native confidence** → we have a concrete,
  deployable recommendation: use this signal to route human review.
- **No alternative signal beats it** → a precise, bounded finding: guard
  models' native confidence is already close to the best available signal for
  routing review, and the expensive alternatives are not worth their cost.

The second is not a failure. It closes a question that people would otherwise
keep guessing about.

---

## 4. What we measure

### The core experiment

Take 10,000 messages. The guard classifies all of them and gets some wrong.
The human reviewer has capacity for 500.

Each uncertainty signal proposes a different 500. We ask one question:

> **At the same review budget, which signal catches the most guard errors?**

If the guard made 1,000 errors and a reviewer looking at 500 cases finds:

| Signal | Errors found | Reading |
|---|---:|---|
| Random selection | ~50 | The floor |
| Native confidence | ~120 | Safety-Flag's approach. **This is the bar to beat** |
| Some alternative | ~400 | A real finding |

We sweep the budget from 0% to 100% rather than fixing it at 5%, which gives a
full curve. We summarise the curve as one number and compare signals on it.

**Important:** the number to beat is **native confidence**, not random. Random
was the right baseline before Safety-Flag. It is not any more.

---

## 5. The signals we compare

| # | Signal | Plain meaning | Cost |
|---|---|---|---|
| 1 | **Native probability** | The guard's own confidence, temperature-scaled the way Safety-Flag does it | Free — **the baseline** |
| 2 | **Logit margin** | The raw gap between the "safe" and "unsafe" scores, before it gets squashed into a probability | Free |
| 3 | **Perturbation instability** | Reword the message several ways. If the verdict flips, the guard is unsure | ~5–10× inference |
| 4 | **Cross-model disagreement** | Run several different guards. Disagreement means a hard case | ~n× inference |
| 5 | **Cross-precision disagreement** | Run the *same* guard at several quantization levels. Disagreement means a fragile decision | Free if the ladder already ran |

**Why signal 2 is in:** the probability is produced by squashing the margin
through a sigmoid. Past a margin of about 37, the squash returns exactly 1.0
and everything beyond becomes indistinguishable. The ranking information still
exists in the margin and is destroyed in the probability. Signals 1 and 2
measure that loss directly, and signal 2 is free.

**Why signal 3 is the interesting one:** it does not use the confidence score
at all. It is behavioural. So it can work even on a guard whose scores are
pinned at 0 and 1 and carry no ranking information.

**Why signal 5 may be the best value:** perturbing the *model* instead of the
*input* avoids the problem that rewording a message also changes its meaning.
And if the quantization ladder has already been run, this signal is free.

**Why signal 4 needs care:** different guards have different decision
boundaries, so disagreement partly measures "these are different models"
rather than "this case is hard." Usable, but interpret carefully.

---

## 6. The honest risks

Written down in advance so they are findings rather than surprises.

**Perturbation instability may add nothing.** Existing work (arXiv 2402.13006)
found that examples which flip under perturbation already have higher baseline
uncertainty. If instability is just a noisy restatement of a small margin, it
costs 10× the compute and buys nothing. Our hypothesis is that this breaks
down when native confidence is degenerate — but that is a hypothesis, not a
given.

**Rewording changes meaning.** "Instability" mixes genuine model uncertainty
with the paraphrase actually being different. The perturbations have to be
checked as meaning-preserving, or we are measuring bad paraphrases.

**The polarization claim must be tested, not assumed.** An industry benchmark
reports that some guards pin 99.8% of their scores at 0 or 1. That is
motivation, not evidence about *our* models. Safety-Flag in fact found
dedicated guards to be *better* calibrated than general-purpose models. We
measure this on our own models first and report what we find.

---

## 7. What we use

### Models

Two kinds of guard, because the kind may determine whether the signals work.

| Kind | How it scores | Examples |
|---|---|---|
| **Generative guard** | Writes "safe" or "unsafe"; we read that word's score | Llama Guard 3, Qwen3Guard, ShieldGemma, Granite Guardian |
| **Encoder classifier** | A small model with a real probability output | DeBERTa / BERT-style moderation heads |

A generative guard's confidence is a by-product of predicting the next word,
trained on hard right-or-wrong labels. An encoder classifier is trained
specifically to produce a graded probability. They should behave differently,
and if they do, that is part of the answer.

Planned: Llama-Guard-3-8B (the reference), one more generative guard,
Llama-Guard-3-1B for the size question, one encoder classifier.

**Caution:** many encoder classifiers score *toxicity*, which is not the same
label as *harmful request*. Verify the labels line up first.

### Datasets

| Dataset | Rows | Role |
|---|---:|---|
| **toxicchat** | 5,083 | **The headline.** Real traffic, realistically low harmful rate. Where "review 5% of traffic" is a real question |
| harmbench | 200 | All harmful. Stress case. Already in the repo |
| xstest | 450 | Benign but borderline. The other stress case. Already in the repo |
| wildguardtest | 1,725 | Replication |
| openai_moderation | 1,680 | Replication |

Safety-Flag's released item-level scores give us a published comparison point
on overlapping benchmarks.

---

## 8. The gates

Fixed before any data is collected, and each one can fail. Same discipline as
`preregistration.md`.

| Gate | Asks | Passes if |
|---|---|---|
| **S1 — is native confidence usable here?** | Does the guard's own confidence beat random on *our* models? | Confidence interval excludes zero. Expected to pass — this replicates Safety-Flag and validates our setup |
| **S2 — the squashing tax** | Does the raw margin beat the probability? | Paired test on the difference survives correction |
| **S3 — does anything beat native confidence?** | **The crux.** Do signals 3, 4 or 5 beat signal 1 at matched budget? | Paired test per signal, corrected across signals |
| **S4 — is instability independent?** | Does perturbation instability add anything *beyond* the margin? | Measured on cases where the margin is uninformative |
| **S5 — kind of guard** | Do encoder classifiers and generative guards differ? | Per-class comparison with a paired test |

**S3 is the crux.** S1 is a sanity check that our pipeline reproduces known
results. S4 is the guard against the most likely way S3 succeeds for a boring
reason.

---

## 9. How we run it

| Phase | What happens | GPU |
|---|---|---|
| **0** | Build fake score data where the right answer is known by construction; confirm the gates give the right verdict | No |
| **1** | One guard, one dataset, signals 1 and 2. **Go / no-go on the cheap signals** | ~10 min |
| **2** | Add perturbation instability on the same slice. **Go / no-go on the expensive signal** | ~1 hr |
| **3** | All models, all datasets, all signals | ~4 hrs |
| **4** | Cross-precision disagreement, reusing the bit ladder | ~5 hrs |
| **5** | Optional: route to a bigger guard instead of a human | ~1 hr |

Phase 0 uses the same trick `verify_gates.py` already does: a test that cannot
fail on data built to fail it is decoration.

Phases 1 and 2 are genuine stop points. Phase 2 in particular decides whether
the expensive signal earns its cost before we scale it.

---

## 10. What we need

**Reused unchanged:** the scorer, dataset loading and validation, the
statistical tests, the threshold analysis, the run orchestration, the RunPod
path.

**New code:**

| File | Does |
|---|---|
| `evaluation/selective.py` | Errors-caught-at-budget curves, the random and native-confidence baselines, signal comparison |
| `evaluation/signals.py` | Compute the five uncertainty signals from prediction files |
| `scripts/perturb.py` | Generate and validate meaning-preserving rewordings |
| `evaluation/gates.py` | Add gates S1–S5 |
| `scripts/verify_selective.py` | Phase 0 |

**Decide before running anything:** the scorer already saves the margin and
the individual label scores, which covers signals 1, 2, 4 and 5. Adding
full-vocabulary entropy and non-label probability mass costs nothing now and
saves a full re-run later.

---

## 11. What we get

| Outcome | The finding |
|---|---|
| A cheap signal (margin) beats native confidence | Free improvement over the published baseline — change one line and route review better |
| An expensive signal (instability, disagreement) beats it | Behavioural signals carry uncertainty information that confidence scores cannot express. The strongest result |
| Encoders and generative guards differ | The architecture determines whether a guard can support human oversight |
| Nothing beats native confidence | Guard confidence is already near the practical ceiling for routing review; the expensive alternatives are not worth their cost. A bounded, useful negative result |

All four are publishable, and the study is designed so one of them must occur.
