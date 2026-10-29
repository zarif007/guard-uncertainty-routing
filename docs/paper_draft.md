# Buy the Better Guard: What a Fixed Oversight Budget Buys in Safety Moderation

*Draft. Every number traced to `results/tables_repriced/`; see `docs/findings.md`
for the table-by-table provenance.*

---

## Abstract

Content-moderation guard models make mistakes, and deployments cover for them by
routing uncertain decisions to human reviewers. A fixed oversight budget can buy
a larger guard, compute spent checking the guard's answer, or more reviewer
hours. These compete, and nobody has priced them against each other. We do,
over 26,958 predictions from three generative guards on five moderation
datasets in one hardware environment, under a protocol pre-registered before any
data was collected.

**The answer is to buy the better guard.** At 1%, 2% and 5% human review, and
again at 50%, the single strongest guard ranked by its own free confidence
ships the fewest bad decisions. A three-guard agreement panel wins only in a
10-20% band and costs 2.5x the compute to do it. The smallest guard never wins
at any budget, so our pre-registered hypothesis, that a cheap guard with
expensive checking beats an expensive guard with free checking, is not
supported.

The useful part is why. Every alternative we priced is bounded, and not by
compute, which is cheap and which we spent freely. Routing each prompt to the
guard that would answer it correctly would cut error 26.3% (95% CI [23.6, 29.2])
at half the compute of the larger guard alone, and the best available escalation
signal captures none of it (realised gain 95% CI [-0.0017, +0.0012]). The
binding constraint is identifying which decisions are wrong, and we show where
that identification breaks down: on 2.85% of prompts all three guards agree and
all three are wrong, they are confidently wrong, none of them occur on the short
curated benchmark in our set, and the rate grows roughly ninefold from the
shortest to the longest length quintile. Input length destroys the behavioural
perturbation signal on the same axis (rank correlation -1.0 against usable
variant count).

On long inputs the signals that would route a case for review stop working
exactly where the panel stops catching it. Short curated benchmarks hold that
variable fixed and cannot observe either effect.

## 1. Introduction

A guard model sits in front of a deployed language model and decides whether an
incoming message should be blocked. It is wrong in both directions: it blocks
benign requests and admits harmful ones. Operators compensate with human review,
but review is scarce -- a realistic moderation team inspects on the order of
1-20% of decisions, not half.

That scarcity makes the operator's problem an allocation problem. A fixed
oversight budget can buy a larger and more accurate guard, additional compute
spent checking the guard's answer, or more reviewer hours. These compete, and
the exchange rate between them is unknown.

The nearest prior work, Safety-Flag (arXiv:2609.19072), establishes that a
guard's own token-logprob confidence ranks its errors usefully, across seven
benchmarks and thirteen models. It contains no cost analysis of any kind, and
reports risk only at 50%, 80% and 100% coverage on a protocol balanced ~50/50 by
construction. The regime an operator inhabits -- 1-20% review at a 1-7% base
rate -- is untouched.

This paper occupies that regime and answers the question for it. Our aim is an
answer a practitioner can act on, not a framework. Contributions, in the order
they matter:

1. **The answer** (§5.1). At the review rates deployments run, buy the better
   guard and spend nothing on routing. The one exception is a 10-20% band where
   a three-guard panel pays, at 2.5x the compute. The smallest guard never wins.
2. **A price for every alternative** (§3, §5). Each (guard, routing signal,
   review budget) policy costed on two axes that are never converted into one
   another, over 26,958 predictions in one hardware environment. The prior work
   contains no cost analysis of any kind.
3. **Why the alternatives lose: the identification gap** (§5.4). We separate
   what routing *could* buy from what available signals *do* buy, with bootstrap
   intervals. The gap is large and the capture is statistically zero.
4. **Where identification breaks down: input length** (§5.5). Unanimous panel
   failure and behavioural-signal collapse are governed by the same variable,
   which short curated benchmarks hold fixed.

**Pre-registration.** The hypotheses, decision thresholds and gate structure
were committed to a public document while the results directory was empty, and
frozen when the first real predictions existed. Two consequences are visible in
this paper and we report both: our principal hypothesis -- that a cheap guard
with expensive checking beats an expensive guard with free checking -- was *not*
supported, and a mechanism we proposed post-hoc was specified, tested on eight
panels, and rejected (§5.3, §7).

---

## 2. Related work

**Selective prediction for moderation.** Safety-Flag (arXiv:2609.19072)
benchmarks reliability and calibration of content moderators across seven
datasets and thirteen models, reporting AURC as its primary instrument. It
establishes that token-logprob confidence is usable and that temperature
scaling improves calibration numbers without changing prediction ordering. We
replicate the first finding (§5.2) and our §5.2 result on the logit margin is
the second in a sharper form.

Its Appendix D.1 is a direct headwind for behavioural signals: sampled-answer
agreement over five stochastic generations loses to token-logprob confidence on
all six models tested. We differ in varying the *input* rather than the
*decoder* -- a guard scored greedily on one label token has no sampling variance
at all -- but our §5.5 result is, in outcome, consistent with theirs.

**Perturbation and instability.** arXiv:2402.13006 finds that examples which
flip under perturbation already carry higher baseline uncertainty, raising the
possibility that instability is a costly restatement of a small margin. Our gate
S4 was constructed to detect precisely this; we could not evaluate it on
realistic inputs, for reasons that became a result (§5.5). Semantic entropy and
semantic-invariant perturbation sampling (Kuhn & Gal and successors) establish
the technique for hallucination detection; its application to guard error
routing is what is new, not the method. arXiv:2603.21172 argues entropy alone is
insufficient for safe selective prediction, supporting a search beyond the
obvious signal.

**Polarized guard scores.** An industry benchmark reports verdict probabilities
"almost perfectly polarized, with 99.8% of scores pinned at zero or one" for
some guards. We measured this on our own models rather than assuming it, and it
does not hold: 12-37% pinned, with usable interior resolution throughout (§5.2).
Safety-Flag likewise found dedicated guards better calibrated than
general-purpose models.

**Unoccupied.** Across this literature we find no cost analysis of guard
oversight, no treatment of the 1-20% review regime, no cross-model disagreement
as a routing signal, and no comparison of logit margin against probability.

---

## 3. Method

### 3.1 Policies and the objective

A **policy** is a (guard, routing signal) pair together with a review budget
`b`. The signal assigns every prompt a score; the `b` least-confident fraction
goes to a human, who is assumed to correct any error shown to them. No
thresholds are involved -- only the ordering of a signal is ever used, so
signals never need to be calibrated onto a common scale.

The objective is **residual risk**:

```
residual = errors the guard made  -  errors the reviewer saw
```

We use residual rather than accuracy or errors-caught because both flatter a
weak guard. A guard making 1,614 errors and catching 1,128 ships more harm than
one making 943 and catching 561.

### 3.2 Two cost units, never mixed

Compute and reviewer hours are not denominated in the same thing, and any
exchange rate dates badly. Nothing in this study converts one into the other.
Policies are reported on a two-dimensional frontier and the reader supplies
their own prices.

Compute is measured latency per decision on the evaluation machine, normalised.
A panel's price is the **sum of its members' costs** -- it runs every member on
every prompt, so its price is a property of the panel, not of whichever member
is nominated to decide.

### 3.3 Guards, data, signals

Three generative guards, scored at fp16 through llama.cpp:

| guard | error rate | AUROC | flag rate | compute |
|---|---:|---:|---:|---:|
| Llama-Guard-3-8B | 0.1049 | 0.9226 | 0.200 | 0.319 |
| Qwen3Guard-Gen-8B | 0.1157 | 0.9615 | 0.314 | 0.372 |
| Llama-Guard-3-1B | 0.1796 | 0.8704 | 0.283 | 0.102 |
| *(three-guard panel)* | | | | *0.793* |

Five datasets, 8,986 prompts after normalisation, pooled base rate 22.5%:
ToxicChat (4,972; 7.1% harmful; real user traffic), WildGuardTest (1,699),
OpenAI-Moderation (1,665), XSTest (450; benign-but-borderline), HarmBench (200;
all harmful).

Signals: native confidence `|p_unsafe - 0.5|` (free, the baseline), logit margin
`|margin|` (free), three-guard modal agreement (3x compute), and perturbation
instability over six meaning-preserving rewordings (6x compute).

**Environment.** All 26,958 predictions were scored on one NVIDIA L40S
(`env_hash 6ccc83087810`). Every prediction row carries this hash and the
analysis refuses to treat latency as comparable across differing hashes. Compute
prices are therefore valid within this environment and only within it.

---

## 4. Pre-registered gates

Each gate can fail, and was verified against synthetic data whose answer is
known by construction, in both directions, before any real scoring (13/13
checks). Thresholds were fixed in advance: a 5% relative residual reduction for
the allocation gate, 0.01 AURC for signal comparisons.

| Gate | Question | Result |
|---|---|---|
| A1 | At matched compute, does anything beat the default? | `CHEAPER_GUARD_WINS` (see §5.6) |
| A2 | Does the winner change with budget? | `BUDGET_DEPENDENT` |
| S1 | Is native confidence usable? | `NATIVE_USABLE` (3/3) |
| S2 | Does the margin beat the probability? | `NO_DIFFERENCE` |
| S3 | Does any alternative beat native confidence? | `ALTERNATIVE_BEATS_NATIVE` (1/3 guards) |
| S4 | Is instability independent of the margin? | `NOT_EVALUABLE` (§5.5) |
| S5 | Does guard *kind* decide usability? | `NOT_EVALUABLE` -- no aligned encoder |

---

## 5. Results

### 5.1 The answer: buy the better guard

| Review budget | Winning policy | Residual | Compute |
|---:|---|---:|---:|
| 1% | Llama-Guard-3-8B + own confidence | 0.0999 | 0.319 |
| 2% | Llama-Guard-3-8B + own confidence | 0.0963 | 0.319 |
| 5% | Llama-Guard-3-8B + own confidence | 0.0832 | 0.319 |
| **10%** | **Qwen3Guard-8B + 3-guard panel** | **0.0634** | **0.793** |
| **20%** | **Qwen3Guard-8B + 3-guard panel** | **0.0376** | **0.793** |
| 50% | Llama-Guard-3-8B + own confidence | 0.0112 | 0.319 |

At three of the four budgets a deployment would actually run, and again at 50%,
the winner is a single guard using the confidence it already produces. It costs
one forward pass and no new infrastructure.

The one exception is a **10-20% band** where a three-guard agreement panel wins,
at **2.5x the compute**. The pattern is not monotone in budget, and §7 reports a
mechanism for the band that we proposed and then rejected.

**The smallest guard never wins at any budget.** Our pre-registered hypothesis,
that a cheap guard with expensive checking beats an expensive guard with free
checking, is not supported.

![FIG:fig1_budget_crossover.png|The answer, and its one exception. Residual risk against human review budget for four policies. The 8B guard on its own confidence wins below 10% review and at 50%; the three-guard panel wins only in the 10-20% band, at 2.5x the compute. Inset magnifies the crossover. The 1B guard never wins at any budget.]

The rest of this section is about **why** the alternatives lose, because that is
the part that transfers to guards we did not test.

### 5.2 The free signals work, and cannot be improved on for free

All three guards have usable native confidence (deferral efficiency 0.667-0.745,
tie fractions below 0.003), replicating Safety-Flag on models it did not test.
This is why §5.1's answer is not a statement that routing is useless: the free
signal already works.

![FIG:fig2_risk_coverage.png|Risk-coverage curves for native confidence on all three guards, pooled over the five datasets. Risk among retained decisions falls as coverage drops, confirming S1 on models Safety-Flag did not test. The shaded band marks the 1-20% review regime this study occupies; Safety-Flag reports only 50%, 80% and 100% coverage.]

The other free signal buys nothing. The logit margin, hypothesised to carry
ranking information destroyed by sigmoid saturation, is *identical* to the
probability: AUROC and AUROC-on-margin agree to seven decimal places on all
three guards (delta AURC <= 1.8e-08). The saturation never occurred. Median
`|margin|` is 4.0-5.8, far short of the ~37 at which float64 sigmoid pins to
1.0, and only 12-37% of scores are pinned at all.

So the premise that motivates behavioural signals, that guard scores are
degenerate, does not hold for these guards. It should be measured rather than
assumed.

### 5.3 The panel pays only in a band, and only for the weakest guard

With three guards voting, modal agreement takes two values. The tie fraction is
1.0000 on every guard and every dataset: the signal partitions traffic into a
flagged block and an unflagged one and imposes no order within either. That
coarseness is why it can win only where the budget happens to suit it.

The flagged block is 23.8% of traffic (2,141 prompts) and contains 72.9-84.1% of
each guard's errors. But the benefit is not uniform:

| guard | delta AURC vs native | 95% CI | Holm p |
|---|---:|---|---:|
| Llama-Guard-3-1B | **-0.0142** | [-0.0201, -0.0085] | 0.000 |
| Llama-Guard-3-8B | +0.0158 | [+0.0101, +0.0204] | 1.000 |
| Qwen3Guard-8B | +0.0092 | [-0.0011, +0.0103] | 1.000 |

(Negative is better.) Agreement is a large, correction-surviving improvement for
the 1B and runs in the *wrong* direction for both 8B guards. A strong guard
gains nothing from consulting a weaker one. We note this because "ensemble
disagreement helps" is commonly asserted without this split.

![FIG:fig3_agreement_forest.png|Change in AURC from adding three-guard agreement, with 95% bootstrap intervals and Holm-corrected p-values. Only the 1B guard improves. The point estimate runs the wrong way for both 8B guards, neither surviving correction.]

### 5.4 How much is being left on the table

The policies above all have one guard decide every prompt. We additionally
evaluate a cascade, where the 1B decides what it can and escalates
low-confidence cases to the 8B, recomputed from the existing predictions with no
new inference.

| policy | error | 95% CI | compute |
|---|---:|---|---:|
| 8B alone | 0.1049 | [0.0987, 0.1110] | 0.319 |
| cascade, escalating on 1B confidence | 0.1052 | [0.0989, 0.1112] | 0.294 |
| **cascade, oracle escalation** | **0.0773** | **[0.0720, 0.0827]** | **0.159** |

Bootstrap over prompts, 2,000 resamples:

- **Realised** gain (8B - cascade): **-0.0002, CI [-0.0017, +0.0012]**, straddles zero.
- **Available** gain (8B - oracle): **+0.0276, CI [+0.0243, +0.0309]**, excludes zero.

The two guards fail on sufficiently different cases that perfect routing between
them would cut error **26.3% [23.6%, 29.2%]** at **half** the 8B's compute. The
best available escalation signal captures none of it.

So the reason to buy the guard rather than the routing is not that routing has
little to offer. It has a great deal to offer. It is that nothing available
collects it.

![FIG:fig4_identification_gap.png|The identification gap. Error bars are 95% bootstrap intervals over 2,000 resamples. Escalating on the 1B guard's own confidence is indistinguishable from the 8B alone. Escalating on an oracle cuts error 26.3% at half the compute. Everything between the dotted lines is available and uncaptured.]

### 5.5 Where identification breaks down, and why it is input length

On 2.85% of prompts (256) all three guards agree **and all three are wrong**:
167 benign prompts flagged, and 89 harmful prompts admitted. Three properties
explain why no confidence-based policy in §5.1 could have found them.

**They are confidently wrong.** Median |p - 0.5| is 0.459-0.480 when the panel
is unanimously wrong against 0.493-0.499 when it is unanimously right, both near
the 0.5 ceiling. Confidence does not separate them, so no amount of review
budget spent on a confidence ranking reaches them.

**Short curated benchmarks cannot see them.** HarmBench contributes **zero**
unanimous failures. Every one of the 256 comes from realistic or long-form
traffic:

| dataset | n | unanimously wrong | rate |
|---|---:|---:|---:|
| WildGuardTest | 1,699 | 87 | 5.12% |
| OpenAI-Moderation | 1,665 | 82 | 4.92% |
| XSTest | 450 | 11 | 2.44% |
| ToxicChat | 4,972 | 76 | 1.53% |
| **HarmBench** | 200 | **0** | **0.00%** |

**Length drives it, within datasets as well as across them.** Median prompt
length of unanimous failures against the rest of the same dataset: 967 vs 363
characters (OpenAI-Moderation), 548 vs 236 (WildGuardTest), 83 vs 59
(ToxicChat). Pooled by length quintile, the unanimous-failure rate runs 0.65%
(median 30 chars), 2.17%, 3.07%, 2.28%, **6.12%** (median 869 chars), roughly
ninefold from shortest to longest.

The same variable governs the behavioural signal we could not price. Our
perturbation family, meaning-preserving surface edits such as inserted
greetings, expanded contractions and opener substitutions, produced too few
distinguishable variants to score on three of five datasets, and the diversity
gate refused them before inference:

| dataset | mean prompt chars | usable variants (of 6) |
|---|---:|---:|
| XSTest | 44 | 5.0 |
| HarmBench | 89 | 4.4 |
| ToxicChat | 167 | 3.5 (blocked) |
| WildGuardTest | 480 | 2.4 (blocked) |
| OpenAI-Moderation | 645 | 2.0 (blocked) |

Rank correlation between mean prompt length and usable variant count is
**-1.0**. The mechanism is arithmetic: a surface edit is a fixed-size change, so
its relative magnitude is that size divided by prompt length. Where it could be
measured (XSTest), instability lost outright: AURC 0.0295 against native's
0.0157, catching 10 errors at 5% review where native caught 12, at six times the
compute.

![FIG:fig5_length_axis.png|Input length governs both failures. (a) The rate at which all three guards agree and all three are wrong, by prompt length quintile: 0.65% at 30 characters rising to 6.12% at 869. (b) Usable rewordings per prompt against mean prompt length, by dataset; the three datasets below the usability threshold are the three long-form ones, and could not be scored.]

> **Guard oversight degrades with input length from both directions at once.**
> The signals that would route a case for review stop working exactly where the
> panel stops catching it. This is a property of the inputs, not of a method,
> and short curated benchmarks hold the variable fixed.

### 5.6 A warning about how the guard is chosen

§5.1 says buy the better guard. This section is about what "better" has to mean.

Gate A1 reports that one guard beats another at matched compute by 10.7%. The
comparison is Llama-Guard-3-8B against Qwen3Guard-Gen-8B, *both using native
confidence*. There is no difference in selection, so we do not present this as
an allocation result. What it does show is worth separating.

Qwen3Guard has the highest AUROC of the three (0.9615 vs 0.9226) and the worse
deployed error rate (0.1157 vs 0.1049), because its operating point over-flags:
31.4% flagged against a 22.5% base rate. **Selecting a guard by AUROC picks
Qwen; selecting by deployed residual risk picks Llama-Guard-3-8B.** Guard
evaluations predominantly report the former.

![FIG:fig6_auroc_vs_error.png|Ranking quality against deployed error. Qwen3Guard-Gen-8B ranks best and deploys worst, because it flags 31.4% of traffic against a 22.5% base rate.]

---

## 6. Discussion

**The recommendation.** At review budgets below 10%, run the strongest single
guard on its own confidence and spend nothing on routing. Between 10% and 20%, a
three-guard panel is worth 2.5x the compute. Above that, go back to the single
guard. Choose that guard by deployed residual risk at your operating point, not
by AUROC (§5.6).

**Why it holds.** Five routes to reducing residual risk were priced, and each is
bounded:

| route | bound |
|---|---|
| logit margin | identical ordering to the probability |
| perturbation instability | unmeasurable at realistic prompt length |
| cross-model disagreement | two-valued; wins only in a 10-20% band |
| cascaded escalation | captures 0 of 0.0276 available headroom |
| any panel routing | 2.85% unanimous-failure floor, confidently wrong |

None is a compute limit. Compute is cheap and we spent it freely. Each is a
failure to identify which decisions are wrong, and §5.4 puts a number on what
that failure costs: 26.3% of error, available and uncollected.

**Where to spend effort instead.** If unanimous failure and signal collapse are
both governed by input length, then evaluation on short curated benchmarks
systematically understates both, and progress requires length-proportional
perturbations. That means a paraphrase model, which reintroduces a confound we
deliberately excluded: a paraphraser can sanitise a harmful prompt, making a
flipped verdict uninterpretable. Resolving that confound is the next problem,
not finding a sixth signal.

---

## 7. A mechanism we proposed and rejected

The natural explanation for the 10-20% band in §5.1 is that the signal is
exhausted once the review budget exceeds its flag rate, predicting that the band
*ends* at the flag rate. We tested it: three guards give three pairs plus the
triple, each panel with its own flag rate, residuals recomputed over a 1-50%
grid with ties broken at random over 20 seeds.

The prediction fails. Spearman correlation between flag rate and band end is
0.435 over eight panels, and the band overshoots the flag rate in seven of
eight, by 0.159 on average. The flag rate is where agreement stops *gaining*,
not where it stops *leading*. We report this rather than omitting it.

---

## 8. Limitations

1. **Single hardware environment.** Compute prices are valid on one L40S and
   nowhere else, by design. A1's notion of "the default" is the highest-latency
   free-signal policy, which is hardware-dependent and could invert elsewhere.
2. **Three guards, one family of alternative signals.** Panel agreement is
   effectively the only alternative signal we could evaluate; S2 was null by
   construction, S4 unmeasurable, S5 untested.
3. **No encoder guard.** The one candidate scored AUROC 0.599 against our label
   -- a toxicity head rather than a harmful-request head -- and was withdrawn
   before scoring. S5 remains open and we do not claim it.
4. **Sections 5.4, 5.5 and 7 are exploratory**, derived after seeing the
   pre-registered results. The oracle in §5.4 bounds this guard pair only.
5. **The length result is correlational.** We show the association holds within
   datasets, which rules out composition, but not that length is causal rather
   than a proxy for topic or complexity.
6. **Perturbation was tested in its cheap form only.** We did not test a
   paraphrase model, for the confound reason above, so §5.5 bounds surface
   perturbation and not behavioural signals in general.

---

## 9. Conclusion

Given a fixed oversight budget for a safety guard, at the review rates
deployments actually run, **buy the better guard and spend nothing on routing**.
Everything we priced that costs more fails to beat it outside a narrow 10-20%
band, and the smallest guard never wins at all.

That answer is unglamorous, and it was not the one we pre-registered. What makes
it worth having is the price list behind it and the reason underneath it.

The reason is identifiable and quantified: perfect routing between two guards we
already have would cut error by a quarter at half the compute, and no available
signal captures any of it. And the cases where oversight fails hardest are the
long ones -- where the panel is confidently and unanimously wrong, and where the
signals that would flag them have stopped working. Short curated benchmarks hold
that variable fixed and cannot observe either effect.

---

## Reproducibility

Pre-registration, protocol, gate definitions and the full findings-to-table
provenance are in `docs/`. Gates are verified against synthetic data with known
answers (13/13) and that check is re-run after any change to the analysis code.
Claims are licensed by a generated table rather than curated by hand; claims the
table does not license are not made here.

---

## References

Annotated for reading. Each entry says **what they did**, **what we found**, and
**the difference**. Read-status is from `docs/related_work.md`: only Safety-Flag
has been read in full. Do not cite a `SNIPPET` entry as if read.

> **All 17 arXiv IDs below were checked against the live abstract pages on
> 2026-10-23.** Every one exists and points to the paper claimed. Titles,
> authors and years here are taken from those pages. Three venue labels could
> not be confirmed and are flagged.

### A. The direct neighbour, read this first

**[1] Yibo Hu (2026). Safety-Flag: A Unified Benchmark for the Reliability and
Calibration of LLM Content Moderators.** arXiv:2609.19072.
`READ IN FULL 2026-09-28` <https://arxiv.org/abs/2609.19072>

*Single-author. The abstract page gives the submission date as 6 Aug 2026,
which does not match the 2609 identifier prefix. Check the date before citing.*

- **What they did.** Seven safety benchmarks (BeaverTails, XSTest, Ethics,
  WildGuard, Aegis, ToxiChat, ToxiGen) under one balanced flag/do-not-flag
  protocol. Thirteen models: six general-purpose LLMs, four dedicated guards
  including Llama Guard 3 8B, three reference models, **all generative**. AURC
  is their primary metric. About 1,400 items, roughly 50/50 harmful/benign.
- **Their finding.** Token-logprob confidence lowers error for every model and
  gives the lowest AURC of the signals tested. Temperature scaling improves
  calibration numbers **without changing prediction ordering**.
- **Where we agree.** Our S1 replicates them on models they did not test. Our
  S2 is their temperature observation in sharper form: the logit margin is a
  strictly monotone transform of the probability and ranks identically to seven
  decimals.
- **The difference, and the reason this paper exists.** They have **no cost
  analysis of any kind**, and report risk only at 50%, 80% and 100% coverage on
  a 50/50-balanced protocol. Risk@0.5 means reviewing half of all traffic. We
  occupy 1-20% review at a 7.1% base rate, which their design cannot address.
- Read §4, §5.5 Table 15, and Appendix D.1 (see [2]).

**[2] Safety-Flag, Appendix D.1: the headwind.** Same paper.

- **What they did.** Compared sampled-answer agreement (five stochastic
  generations per item) against token-logprob confidence. Token-logprob wins on
  **all six** models (their Table 14).
- **Why it matters.** A reviewer will ask why paraphrase-consistency should
  succeed where sample-consistency failed. Our §5.5 outcome is, in direction,
  **consistent with theirs**, and we say so rather than claiming a reversal.
- **The difference.** They vary the decoder; we vary the input. A guard scored
  greedily on one label token has no sampling variance at all. More importantly
  we identify *why* ours failed: a length mechanism their short balanced items
  could not expose.

### B. Uncertainty and selective prediction

**[3] Sara Vera Marjanović, Isabelle Augenstein, Christina Lioma (2024).
Investigating the Impact of Model Instability on Explanations and Uncertainty.**
arXiv:2402.13006. `SNIPPET` <https://arxiv.org/abs/2402.13006>

- **What they did.** Examples that flip under perturbation already carry higher
  baseline uncertainty.
- **The threat.** If instability merely restates a small margin it costs 6-10x
  inference for nothing. Gate S4 was built to detect this.
- **The difference.** We never got to test it on realistic inputs. Our
  perturbation family was refused by a diversity gate on three of five datasets,
  and that refusal became the result (§5.5). Their hypothesis stays open for
  length-proportional perturbations.

**[4] Edward Phillips, Fredrik K. Gustafsson, Sean Wu, Anshul Thakur, David A.
Clifton (2026). Entropy Alone is Insufficient for Safe Selective Prediction in
LLMs.** arXiv:2603.21172. `SNIPPET` <https://arxiv.org/abs/2603.21172>

- **What they did.** Entropy is a weak uncertainty signal for safe selective
  prediction.
- **Relation.** Supports searching beyond the obvious signal, which motivated
  our five-signal design. Our conclusion is bleaker: every alternative we could
  price is bounded, and the binding constraint is error identification.

**[5] Sijin Dong, Hiroyuki Shinnou (2026). Uncertainty-Aware Abstention in Large
Language Models with Provable Alignment Guarantees.** arXiv:2607.04430.
`SNIPPET` <https://arxiv.org/abs/2607.04430>

- **What they did.** Abstention with provable guarantees.
- **The difference.** For generation, not for a moderation classifier, with no
  human-review budget and no cost axis.

**[6] Lorenz Kuhn, Yarin Gal, Sebastian Farquhar (2023). Semantic Uncertainty:
Linguistic Invariances for Uncertainty Estimation in Natural Language
Generation.** ICLR 2023 (Spotlight). arXiv:2302.09664. `SNIPPET`
<https://arxiv.org/abs/2302.09664>

- **What they did.** Semantic-invariant perturbation sampling for hallucination
  detection. The technique behind our signal 3.
- **The difference.** The method is established; applying it to guard
  error-routing is what is new. We do not claim the technique. We report the
  condition under which it fails to transfer: input length.

### C. Guardrail evaluation and operating points

**[7] Artificial Analysis guardrail benchmark.** Industry, not peer-reviewed.
`SNIPPET` <https://artificialanalysis.ai/articles/guardrail-safety-benchmark>

- **What they report.** Some reasoning guards have verdict probabilities
  "almost perfectly polarized, with 99.8% of scores pinned at zero or one."
- **We measured it and it does not hold for our guards.** 12-37% pinned,
  interior resolution 0.87-0.97, all three `USABLE`.
- **The difference.** Motivation, not evidence. It is the premise that justifies
  behavioural signals, and on our models the premise is false.

**[8] Atoosa Chegini, Hamid Kazemi, Garrett Souza, Maria Safi, Yang Song, Samy
Bengio, Sinead Williamson, Mehrdad Farajtabar (2025). Reasoning's Razor:
Reasoning Improves Accuracy but Can Hurt Recall at Critical Operating Points in
Safety and Hallucination Detection.** arXiv:2510.21049. `SNIPPET`
<https://arxiv.org/abs/2510.21049>

- **What they did.** Reasoning-augmented generation underperforms at the low-FPR
  thresholds that practical deployments need.
- **Relation.** Our closest methodological relative: operating-point-aware
  evaluation of safety detectors. Their independent variable is reasoning; ours
  is the oversight budget.

**[9] David Gringras (2026). Safety Under Scaffolding: How Evaluation Conditions
Shape Measured Safety.** arXiv:2603.10044. `SNIPPET`
<https://arxiv.org/abs/2603.10044>

- **What they did.** Apparent safety differences turn out to be measurement
  artifacts; scores are sensitive to how answers are extracted.
- **Our closest ally in framing.** Same shape of argument, different cause: our
  §5.6 shows a guard with strictly better AUROC is the worse deployment.

**[10] Edson Rodrigues da Cruz Filho et al. (2026). A Reproducible,
License-Aware Distillation Recipe for CPU-Deployable Safety Classification.**
arXiv:2608.21570. `SNIPPET` <https://arxiv.org/abs/2608.21570>

- **What they argue.** A *fixed* decision threshold for cross-model comparison,
  explicitly forbidding tuning each model to its own operating point.
- **This is the objection we must answer**, and §5.6 is where. We do not say
  fixed thresholds are wrong. We say a fixed-threshold number is a joint
  measurement of discrimination and operating point, and reporting it alone
  makes the two indistinguishable. We report both.

**[11] Max Fomin (2026). When Benchmarks Lie: Evaluating Malicious Prompt
Classifiers Under True Distribution Shift.** arXiv:2602.14161. `SNIPPET`
<https://arxiv.org/abs/2602.14161>

- **What they did.** Malicious-prompt classifiers under true distribution shift.
- **Relation.** Adjacent to §5.5: HarmBench produces zero unanimous failures
  while long-form sets produce 5%. A benchmark can be silent about a failure
  mode it holds fixed.

### D. The guards

**[12] Hakan Inan, Kartikeya Upasani, Jianfeng Chi et al. (2023). Llama Guard:
LLM-based Input-Output Safeguard for Human-AI Conversations.** arXiv:2312.06674.
`SNIPPET` <https://arxiv.org/abs/2312.06674>

Our reference guard and the size axis. The Llama-Guard-3-8B model card reports
int8 at F1 0.936-0.939 with "negligible impact", **at a fixed threshold**: the
statistic that cannot separate preserved discrimination from a moved operating
point.

**[13] Haiquan Zhao et al. (2025). Qwen3Guard Technical Report.**
arXiv:2510.14276. `SNIPPET` <https://arxiv.org/abs/2510.14276>

Our replication family, 43 authors. Relevant to §5.6: Qwen3Guard-Gen-8B has the
best AUROC in our panel and the worse deployed error rate.

**[14] Igor Fedorov, Kate Plawiak, Lemeng Wu et al. (2024). Llama Guard
3-1B-INT4: Compact and Efficient Safeguard for Human-AI Conversations.**
arXiv:2411.17713. `SNIPPET` <https://arxiv.org/abs/2411.17713>

A 440MB guard that runs on a commodity Android mobile CPU. Context for why a 1B
guard is a live deployment option, which is what makes our result that it never
wins at any budget worth stating.

*Correction: `related_work.md` describes this as "pruning plus 4-bit" and says
it reports improved safety and efficiency. The abstract states neither. Do not
repeat that description without reading the paper.*

### E. Datasets

**[15] Zi Lin, Zihan Wang, Yongqi Tong et al. (2023). ToxicChat: Unveiling
Hidden Challenges of Toxicity Detection in Real-World User-AI Conversation.**
EMNLP Findings 2023. arXiv:2310.17389.
*Our headline set: 4,972 rows, 7.1% harmful, real user traffic.*

**[16] Mantas Mazeika, Long Phan, Xuwang Yin et al. (2024). HarmBench: A
Standardized Evaluation Framework for Automated Red Teaming and Robust
Refusal.** arXiv:2402.04249.
*200 rows, all harmful. Produces zero unanimous failures (§5.5), the key
negative control.* **Venue unverified**: the abstract page names no venue, so
confirm ICML 2024 against the proceedings before citing it.

**[17] Paul Röttger, Hannah Rose Kirk, Bertie Vidgen, Giuseppe Attanasio,
Federico Bianchi, Dirk Hovy (2023). XSTest: A Test Suite for Identifying
Exaggerated Safety Behaviours in Large Language Models.** NAACL 2024 (Main
Conference). arXiv:2308.01263.
*450 rows, benign-but-borderline. The only set where perturbation was
measurable.*

**[18] Seungju Han, Kavel Rao, Allyson Ettinger et al. (2024). WildGuard: Open
One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs.**
NeurIPS 2024. arXiv:2406.18495.
*1,699 rows. Highest unanimous-failure rate (5.12%) and 70 of our 89
safety-critical false negatives.* **Track unverified**: the page says NeurIPS
2024 but does not name the Datasets & Benchmarks track.

**[19] Todor Markov, Chong Zhang, Sandhini Agarwal et al. (2022). A Holistic
Approach to Undesired Content Detection in the Real World.** Oral presentation
at AAAI-23. arXiv:2208.03274.
*1,665 rows. Longest prompts (645 mean chars) and the largest length gap between
unanimous failures and the rest (967 vs 363). Note the arXiv year is 2022; the
venue is AAAI-23. The page never uses the label "OpenAI Moderation".*

### F. Method

**[20] Selective classification and risk-coverage.** El-Yaniv & Wiener (JMLR
2010); Geifman & El-Yaniv (NeurIPS 2017). **NOT VERIFIED** (not on arXiv in the
form checked).
*Source of the risk-coverage curve and AURC, our primary instrument, shared with
Safety-Flag, which is what makes the comparison checkable.*

**[21] Holm, S. (1979).** A simple sequentially rejective multiple test
procedure. *Scandinavian Journal of Statistics* 6(2), 65-70. **NOT VERIFIED.**
*Our multiple-comparison correction in S3. Only the 1B guard's agreement result
survives it.*

---

### Still to check

- [16] HarmBench: ICML 2024 label
- [18] WildGuard: Datasets & Benchmarks track
- [20] and [21]: the two non-arXiv method references
- [7]: the Artificial Analysis page is industry and may change; archive it

### Reading order

If you read four things, read these:

1. **[1] Safety-Flag in full**: it defines the space you are in.
2. **[2] its Appendix D.1**: the specific result a reviewer will quote at you.
3. **[10] the fixed-threshold paper**: the methodological objection to how you
   compare guards.
4. **[9] Safety Under Scaffolding**: the framing to borrow.

Then [3], [6] and [7] before claiming anything about perturbation or
polarization: each either predicts your result or supplies the premise you
overturned.
