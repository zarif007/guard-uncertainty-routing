# Related work, and where this project is different

Literature scan of 2026-09-26. Its job is to answer one question honestly:
**is the contribution still there?** The short answer is yes, but not where
the research plan originally put it.

> **The headline finding of this scan.** The original framing -- "quantization
> can improve safety, sometimes non-monotonically" -- is **already published**,
> more than once, for generator models. The pivot away from quantization
> followed. The current question is narrower and better defended: given that
> a guard's own confidence already works *somewhat* for routing human review
> (Safety-Flag, below), can anything work *better*?

> **Amendment, 2026-09-28.** Section 0 was added after Safety-Flag was found.
> It is the nearest neighbour to this study and it moves our baseline. Read it
> before anything else on this page. Sections 1 and 5 are retained because the
> quantization ladder survives as the input to one uncertainty signal
> (cross-precision agreement), not because the old framing survives.

## How to read the status column

Sources were found by search; depth of checking varies and is recorded per
entry. Do not cite anything marked `SNIPPET` without reading it first.

| Status | Means |
|---|---|
| `READ` | Full text or HTML fetched and checked against our claims |
| `ABSTRACT` | Abstract verified directly |
| `SNIPPET` | Search-result summary only — **unverified, must be read** |

---

## 0. Selective prediction and abstention for guards — THE NEIGHBOUR

| Work | Idea | Status |
|---|---|---|
| [Safety-Flag: A Unified Benchmark for the Reliability and Calibration of LLM Content Moderators](https://arxiv.org/abs/2609.19072) | Seven safety benchmarks (BeaverTails, XSTest, Ethics, WildGuard, Aegis, ToxiChat, ToxiGen) in one balanced flag / do-not-flag protocol. Measures three dimensions: error direction, probability calibration, and **confidence-based error ranking for human review**. Six general-purpose LLMs, four dedicated guards, three reference models. Releases item-level decisions and confidence scores. | `ABSTRACT` |

**What it establishes, and what it therefore takes off the table.**

- *"Confidence-based abstention lowers selective risk for every model"* — so
  **"guards cannot rank their own uncertainty" is not an open question.** It
  has been measured and the answer is that native confidence works, unevenly.
  Any framing built on guards being unable to self-assess is dead.
- *"Fitting one temperature per model reduces calibration error by 2.8-6.0x
  without changing predicted labels or confidence ordering"* — this is the
  inertness result from the previous study, observed empirically. We can still
  state it as a general fact with a proof, but it is no longer a headline.
- *"Dedicated guards produce fewer false alarms and are better calibrated, but
  several have higher miss rates outside their documented coverage"* — this
  **contradicts** the assumption that guard scores are uniformly degenerate.
  Polarization must be tested on our own models, not asserted from the
  Artificial Analysis figure.

**Where we differ.** In one sentence, to be defended in one breath:

> Safety-Flag showed a guard's confidence is somewhat useful for routing human
> review. We ask whether signals that ignore the confidence score entirely do
> better -- and what they cost.

Expanded, in the order a reviewer will press on them:

| # | Difference | Why it matters |
|---|---|---|
| 1 | **They test one signal. We test five.** | They use the number the guard already produces. We race five ways of spotting a shaky decision against each other |
| 2 | **Three of our signals never read the confidence score.** | Rewording the message, cross-model agreement and cross-precision agreement are behavioural. They remain computable when the score carries no ranking information -- the regime where Safety-Flag's approach runs out of road |
| 3 | **Their result is our starting line.** | We do not contest their finding. We take it as the floor and ask what is above it. The baseline in `gate_s3` is *their* method, not random deferral |
| 4 | **We question the number itself.** | `p_unsafe` is `sigmoid(margin)`, and past a margin of ~37 the float64 sigmoid ties every score at 1.0. They use the probability; we compare it against the raw margin underneath. Free to test |
| 5 | **We price the signals.** | Their signal is free, so cost never arises. Rewording a prompt k ways costs k x inference, and `gate_s4` exists solely to catch a signal that wins for a boring reason |
| 6 | **We add a kind of guard they do not include.** | They split general-purpose LLMs against dedicated guards -- both of which emit a *token*. An encoder classifier with a trained probability head is in neither category |
| 7 | **We pre-register failable gates.** | Benchmarks describe what they find. `docs/preregistration.md` fixes which outcome licenses which claim, and `scripts/verify_selective.py` proves the criteria can fail |

**Everything rests on 1 and 2.** The rest is supporting detail. If a reviewer
does not accept "they measured one signal, we search for a better one", nothing
further down the list rescues the contribution.

**What is NOT a difference** -- state this before a reviewer does:

- **Our datasets overlap theirs.** XSTest, WildGuard and ToxiChat are in both.
  That is an advantage (a published comparison) but it is not novelty.
- **They have more models.** Thirteen against our four. We are not out-scaling
  them and should not imply it.
- **"Guards are overconfident" is their finding**, as is "temperature scaling
  fixes calibration without changing the ranking". Both are cited, not claimed.
- **Difference 6 is the weakest.** Their dedicated-vs-general-purpose split is
  already a kind-of-guard comparison. Ours is genuinely a different cut --
  encoder heads are in neither of their categories -- but it has to be drawn
  carefully rather than asserted.

**What it gives us.** Item-level decisions and confidence scores are released.
XSTest, WildGuard and ToxiChat overlap our dataset set, so on those items we
can compare against a *published* native-confidence baseline rather than a
reimplementation of one. "We beat the published number on the same items" is a
much stronger claim than "we beat our version of it."

**Open item:** confirm the item-level overlap and reuse their scores directly.

### Adjacent, and each one narrows a different door

| Work | Idea | Status |
|---|---|---|
| [Uncertainty-Aware Abstention in LLMs with Provable Alignment Guarantees](https://arxiv.org/pdf/2607.04430) | Abstention with guarantees — for *generation*, not for a moderation classifier | `SNIPPET` |
| [Entropy Alone is Insufficient for Safe Selective Prediction in LLMs](https://arxiv.org/pdf/2603.21172) | Entropy is a weak uncertainty signal. Supports searching beyond the obvious | `SNIPPET` |
| [Investigating the Impact of Model Instability on Explanations and Uncertainty](https://arxiv.org/pdf/2402.13006) | **The threat to signal 3.** Examples that flip under perturbation already have higher baseline uncertainty. If instability merely restates a small margin it costs ~10x inference for nothing. Gate S4 is built to detect exactly this | `SNIPPET` |
| Semantic entropy / semantic-invariant perturbation sampling (Kuhn & Gal, and successors) | The *technique* behind signal 3 is established for hallucination detection. Its application to guard error-routing is what is new; the method is not | `SNIPPET` |
| [Artificial Analysis guardrail benchmark](https://artificialanalysis.ai/articles/guardrail-safety-benchmark) | Reports verdict probabilities "almost perfectly polarized, with 99.8% of scores pinned at zero or one" on some guards. **Motivation, not evidence about our models** | `SNIPPET` |

---

## 1. Quantization and safety of the *generated* model

The crowded area, and the one that takes our original framing.

| Work | Idea | Status |
|---|---|---|
| [Joint Effect of Quantization and Sampling Temperature on LLM Safety Alignment](https://arxiv.org/html/2606.29581v1) | Factorial study. States directly that quantization's effect on refusal "is not always monotonic — compressing a model can either weaken or unintentionally strengthen its tendency to refuse," and that moderate 4-bit sometimes preserves or improves trustworthiness. Reports INT4 keeping or lowering attack success for 7 of 9 models. | `SNIPPET` |
| [Q-resafe](https://arxiv.org/pdf/2506.20251) | Safety risks of quantized LLMs plus quantization-aware safety patching. | `SNIPPET` |
| [Quantization Undoes Alignment](https://arxiv.org/html/2605.15208v1) | Bias emergence in compressed LLMs across models and precision levels. | `SNIPPET` |
| [QuantiBias](https://arxiv.org/html/2607.21063v1) | Quantization-induced bias that standard safety evaluation misses; short-form safeguards stay flat while open-ended generation degrades. | `SNIPPET` |
| [Alignment-Aware Quantization for LLM Safety](https://www.arxiv.org/pdf/2511.07842) | Contrastive alignment loss during PTQ to keep the quantized model aligned. | `SNIPPET` |
| [Preserving Fairness and Safety via Critical Weight Protection](https://arxiv.org/html/2601.12033v2) | Sensitivity-scores weights, keeps the safety-critical ones at higher precision. | `SNIPPET` |
| [Effect of Quantization on Clinical Benchmarks](https://arxiv.org/html/2609.22216) | Accuracy and safety across model families in a high-stakes domain. | `SNIPPET` |

**Where we differ.** Every one of these quantizes the **generator** and
measures its refusal behaviour, attack success rate or bias. None quantizes
the **guardrail classifier** that sits in front of a generator. That is a
different system with a different failure mode: a generator's refusal is a
behaviour, a guard's verdict is a thresholded score, and only the second can
drift without changing.

**Two that need reading before this section is final:**

- [Silent Alarm: A J-Space Protocol for Comparing Danger Recognition Across
  Models and Quantization Levels](https://arxiv.org/pdf/2607.12792) —
  the title is close enough to our question to matter. `SNIPPET`, **read first.**
- [LiteLMGuard](https://arxiv.org/pdf/2505.05619) — on-device prompt filtering
  against quantization-induced risks. Adjacent: it *defends* against
  quantization damage rather than measuring the guard's own drift. `SNIPPET`

**Critical-weight protection is our Phase 7.** The fairness/safety
critical-weight paper is doing, for the generator, what
`build_mixed_precision.py` does for the guard. Cite it as prior art for the
method and be explicit that the target differs.

---

## 2. Quantization and calibration

| Work | Idea | Status |
|---|---|---|
| [When Quantization Affects Confidence of LLMs](https://arxiv.org/abs/2405.00632) | GPTQ-4bit lowers confidence in true labels; the effect varies by model and scale; quantization loss concentrates on samples where the full model was *already* low-confidence. | `ABSTRACT` |

**Where we differ.** This is the nearest miss on the calibration axis, and it
stops one step short of our argument. It measures confidence and calibration
and leaves it there. It does not ask whether the confidence change moves any
*decision* — and the answer, for a fixed threshold, is that it cannot:

```
sigmoid(m / T) >= 0.5   <=>   m >= 0,   for every T > 0
```

Our `calibration_decomposition` fits `sigmoid(a*m + b)` and splits the result
into temperature (`1/a`, which drives ECE) and boundary location (`-b/a`,
which is the only part that moves a decision). No source found does this
decomposition, and its consequence — that "quantization hurt calibration"
cannot by itself explain a safety-rate change — appears to be ours.

Their finding is also a *prediction* we can test: if quantization loss
concentrates on low-confidence samples, our flip rate should rise sharply as
the margin approaches zero. That is exactly what `flip_rate_by_distance`
measures. Cite it as a hypothesis we confirm or refute in a new domain.

---

## 3. Quantization and decision geometry

| Work | Idea | Status |
|---|---|---|
| [Boundary-Aware Quantization: Finite-Scale Decision Geometry of Neural Classifiers](https://arxiv.org/html/2607.01478v1) | Quantization-induced flips concentrate near the full-precision decision boundary — 0.034 overall vs 0.169 inside the low-margin band at 6 bits. Proposes boundary-preservation as a model-selection criterion. Concludes that "quantization can preserve or improve accuracy while changing the reference decision geometry." | `READ` |

**Where we differ — and why this one helps us.** Read the title and it looks
like a collision with `flip_rate_by_distance`. It is not. It studies
one-hidden-layer MLPs, small CNNs and reduced ResNets on digits, MNIST,
Fashion-MNIST and CIFAR-10. It uses geometric diagnostics (Jaccard distance on
boundary masks, local displacement, junction stability) — **no AUROC, no
discrimination/operating-point separation, no calibration decomposition, no
language models, no safety.**

It is the strongest **supporting** citation in this document. An independent
group, in a completely different domain, found that quantization moves the
decision boundary while leaving or improving accuracy. That is our mechanism,
replicated in advance, on toy vision models. Use it in the introduction to
establish that the phenomenon is general and that we are testing it where it
has consequences.

---

## 4. Guardrail evaluation and operating points

| Work | Idea | Status |
|---|---|---|
| [Reasoning's Razor](https://arxiv.org/pdf/2510.21049) | Reasoning improves accuracy but can hurt recall at critical operating points in safety and hallucination detection. | `SNIPPET` |
| [Artificial Analysis guardrail benchmark](https://artificialanalysis.ai/articles/guardrail-safety-benchmark) | Industry benchmark. Reports that some reasoning guards have verdict-token probabilities "almost perfectly polarized, with 99.8% of scores pinned at zero or one, leaving no range to tune a threshold," and that such guards retain ~10% recall at a 1% FPR budget. | `SNIPPET` |
| [Safety Under Scaffolding](https://arxiv.org/pdf/2603.10044) | Measurement artifacts can produce apparent large safety differences that turn out to be artifactual; safety scores are sensitive to how answers are extracted. | `SNIPPET` |
| [When Benchmarks Lie](https://arxiv.org/pdf/2602.14161) | Malicious-prompt classifiers under true distribution shift. | `SNIPPET` |
| [CPU-deployable safety classification](https://arxiv.org/pdf/2608.21570) | Argues the opposite methodological choice: a *fixed* decision threshold for cross-model comparison, explicitly "forbidding the practice of tuning each model to its own best operating point." | `SNIPPET` |

**Where we differ.** *Reasoning's Razor* is the closest methodological
relative — operating-point-aware evaluation of safety detectors — but its
independent variable is reasoning, not precision. Nobody here varies precision.

**The last row is the objection we must answer in the paper.** There is a
real argument that a fixed threshold is the *correct* comparison, because
per-model threshold tuning flatters whichever model you tuned hardest. Our
answer is not that fixed thresholds are wrong; it is that a fixed-threshold
number is a *joint* measurement of discrimination and operating point, and
reporting it alone makes the two indistinguishable. We report both. Write this
as an explicit paragraph, not a footnote.

**`Safety Under Scaffolding` is our closest ally in framing** — same shape of
argument (a measured safety difference turns out to be a measurement artifact),
different cause.

---

## 5. Quantized guard models as deployed artifacts

| Work | Idea | Status |
|---|---|---|
| [Llama Guard 3-8B model card](https://huggingface.co/meta-llama/Llama-Guard-3-8B) | Meta ships an int8 version: ~40% smaller, "negligible impact on the performance of the model," F1 0.936–0.939. | `SNIPPET` |
| [Llama Guard 3-1B-INT4](https://arxiv.org/pdf/2411.17713) | Pruning plus 4-bit for mobile deployment; reports improved safety *and* efficiency. | `SNIPPET` |
| [Qwen3Guard Technical Report](https://arxiv.org/html/2510.14276v1) | The second family in our ladder. Three sizes, Gen and Stream variants. | `SNIPPET` |

**Where we differ, and this is the paper's opening paragraph.** The canonical
claim that quantizing a guard is safe — Meta's own, for the exact model at the
top of our ladder — rests on **F1 at a fixed threshold**. That is precisely the
statistic that cannot distinguish "discrimination is preserved" from "the
operating point moved and the errors happened to cancel." We are not
contradicting the claim. We are showing it was never tested, and supplying the
test. Two int8/int4 points also say nothing about the shape of the curve
between 3 and 16 bits.

---

## 6. What is unoccupied

Ranked by how much weight each can bear. Rewritten after Safety-Flag.

1. **A comparison of uncertainty signals for routing guard review.**
   Safety-Flag established that native confidence works. Nobody has asked
   whether the raw margin, behavioural instability, or cross-model /
   cross-precision agreement beats it at a matched review budget.
   **This is the load-bearing contribution.**

2. **Signals that do not read the confidence score.** Three of our five are
   behavioural. They remain computable on a guard whose scores are pinned,
   which is precisely the regime where the established approach degrades.

3. **The squashing tax, measured.** `p_unsafe` is `sigmoid(margin)`, and past
   a margin of ~37 the float64 sigmoid returns exactly 1.0, collapsing
   distinct prompts into ties. Comparing the two rankings costs nothing and
   may hand back free resolution. Nothing found does this.

4. **Guard kind as the explanatory variable.** Generative guards emit a token;
   encoder classifiers emit a trained probability. If self-knowledge tracks
   architecture, the field's move to LLM-based guards has a cost nobody has
   priced.

5. **Pre-registered, demonstrably failable gates.**
   `scripts/verify_selective.py` asserts each verdict against synthetic score
   distributions built to trigger it. Unusual in this area.

---

## 7. What would collapse the novelty

Stated plainly so it can be checked rather than hoped about.

- **A paper comparing uncertainty signals for guard error-routing at matched
  budget.** Not found after targeted search, but Safety-Flag was also missed
  on the first pass because it is framed as a calibration benchmark rather
  than as selective prediction. **Search both vocabularies before submission.**
- **Safety-Flag turning out to test alternative signals** in a section the
  abstract does not mention. Read it in full. Highest-priority item on this
  page.
- **Signal 3 collapsing into signal 2.** If perturbation instability is just a
  noisy margin (arXiv 2402.13006), the expensive half of the study returns
  nothing. Gate S4 is the detector, and this is a finding rather than a
  failure — but it changes which paper gets written.
- **Guards turning out to be well calibrated with usable score resolution.**
  Then the motivating pathology is absent, S3 likely returns
  `NATIVE_IS_BEST`, and the honest output is the bounded negative result.

---

## Open items

- [ ] Read [Safety-Flag](https://arxiv.org/abs/2609.19072) in full; confirm it
      does not test alternative uncertainty signals anywhere.
- [ ] Map Safety-Flag's released item-level scores against our five datasets
      (XSTest, WildGuard and ToxiChat look like matches) and reuse their
      baseline directly.
- [ ] Read the semantic-entropy line of work and cite it as the source of the
      signal-3 technique.
- [ ] Verify every `SNIPPET` resolves to a real paper saying what the snippet
      implies, before any of it reaches a bibliography.
- [ ] Re-run this scan against BOTH vocabularies (selective prediction /
      abstention / risk-coverage, and calibration / reliability / confidence)
      closer to submission.
