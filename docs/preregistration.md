# Pre-registration: which paper each outcome licenses

Written before any predictions were collected for this study. `results/` was
empty and `scripts/verify_selective.py` passed when this file was committed;
the git history is the proof.

This supersedes `archive_preregistration_quantization.md`, which belongs to the
study this repo used to be. That document is kept unedited: a pre-registration
whose history can be rewritten is worth nothing, and the same applies to one
that gets quietly replaced.

---

## The question

A guard classifies every message and gets some wrong. A human reviewer can
check a fraction of them. An *uncertainty signal* decides which fraction.

> **When a guard's native confidence is insufficient to rank its own errors,
> can an alternative uncertainty signal recover useful error ranking at the
> same human-review budget?**

## What is already settled, and by whom

Safety-Flag (arXiv 2609.19072) put seven safety benchmarks into one
flag/do-not-flag protocol across six general-purpose models and four dedicated
guards, and reported that **confidence-based abstention lowers selective risk
for every model tested**, with gains depending on how well confidence ranks
errors. It also found that fitting one temperature per model reduces
calibration error substantially *without changing predicted labels or
confidence ordering*.

Two consequences, both binding on this study:

1. **The baseline is native confidence, not random.** "Beats random" was the
   right bar before Safety-Flag. It is now a sanity check (Gate S1), and a
   failure there means our pipeline is broken, not that we found something.
2. **Temperature scaling is not a treatment.** It cannot reorder a signal, so
   it is used only to construct the baseline the way Safety-Flag constructs
   it. Any claim that recalibration improved *routing* is ruled out in
   advance, by algebra: `sigmoid(m/T) >= 0.5` exactly when `m >= 0`, for every
   `T > 0`.

---

## The decision table

Gate S3 is the hinge. Read it first, then S4, then S5.

| S3 | S4 | Paper | Headline |
|---|---|---|---|
| `ALTERNATIVE_BEATS_NATIVE` | `INDEPENDENT` | **1 — Behavioural uncertainty** | Signals that do not read the confidence score rank guard errors better than the score does. The strongest available result |
| `ALTERNATIVE_BEATS_NATIVE` | `REDUNDANT_WITH_MARGIN` | **2 — The squashing tax** | The gain is real but free: it is recoverable from the raw margin, so it is a property of the score transform, not of behaviour. Change one line, do not buy the extra inference |
| `NATIVE_IS_BEST` | any | **3 — A bounded negative** | Native confidence is already near the practical ceiling for routing review. The expensive alternatives are not worth their cost, and we say how expensive they were |
| `NOT_EVALUABLE` | any | *No paper yet* | Too few errors to rank, or no alternative signal computed. Scale the sample; this is the only condition authorising it |

### Gate S2, reported alongside and never instead

S2 asks whether the raw logit margin beats the probability. It is free to test
and free to act on.

| S2 | Reading |
|---|---|
| `MARGIN_BETTER` | Every downstream system should rank on the margin. A one-line change with a measurable benefit |
| `MARGIN_SOMETIMES_BETTER` | Model-dependent; report per model, do not average |
| `NO_DIFFERENCE` | The sigmoid is not destroying usable resolution on this prompt set |

**S2 passing changes how S3 must be read.** If the margin already beats native
confidence, then the honest baseline for an expensive signal is the *margin*,
not native confidence — otherwise an expensive signal takes credit for a free
one's gain. Gate S4 exists to catch exactly this.

### Gate S5 — does the kind of guard decide it?

| S5 | Reading |
|---|---|
| `KIND_MATTERS` | Generative and encoder guards differ systematically in whether their confidence ranks errors. This is an architectural finding and goes in the abstract |
| `KIND_DOES_NOT_MATTER` | Self-knowledge is not determined by how the score is produced. Also worth stating, because it is not the expected result |

---

## Commitments

1. **The bar is native confidence.** No result is reported against a random
   baseline alone. Safety-Flag established that floor; clearing it again is
   not a contribution.

2. **A win must be material, not merely significant.** A signal counts as
   better only if it improves AURC by at least 0.01 *and* survives Holm
   correction across every signal tested. On a large enough sample a 0.001
   gap is significant and irrelevant.

3. **Testing five signals and reporting the best one is forbidden.** All
   signals enter the correction together. This is why the correction is
   applied across signals and models jointly in `gate_s3`.

4. **Polarization is measured, not assumed.** An industry benchmark reports
   99.8% of scores pinned at the extremes for some guards. That is motivation.
   `evaluation/score_range.py` runs on our own models first and the verdict is
   reported whatever it says. Safety-Flag in fact found dedicated guards
   *better* calibrated than general-purpose models, so the assumption could
   easily be wrong.

5. **`PIN_EPS` is a choice, not a measurement.** The polarization threshold in
   `score_range.py` flags any margin beyond about 6.9 as pinned. That is far
   inside the range where ranking information still exists. It is a screening
   heuristic and it never decides a gate.

6. **The negative result gets written.** "Native confidence is already near
   the ceiling, and here is what the alternatives cost" is a useful finding
   and closes a question people would otherwise keep guessing about.

7. **No claim is written that `claims_to_evidence.csv` has not marked
   `LICENSED`.**

8. **Perturbation quality is a threat, not a detail.** A paraphrase that
   changes meaning produces instability that is not uncertainty. Variants
   failing the semantics-preserved check are dropped before the signal is
   computed, and the drop rate is reported.
