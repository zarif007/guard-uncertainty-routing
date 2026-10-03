# Where Guard Oversight Fails: Input Length Degrades Error Detection and Panel Agreement Together

*Draft. Every number traced to `results/tables_repriced/`; see `docs/findings.md`
for the table-by-table provenance.*

---

## Abstract

Content-moderation guard models make mistakes, and deployments cover for them by
routing uncertain decisions to human reviewers. Which decisions to route, and
whether the compute spent identifying them would be better spent on a larger
guard, is an allocation question that has not been priced. We evaluate three
generative guards (Llama-Guard-3-1B/8B, Qwen3Guard-Gen-8B) on five moderation
datasets, 26,958 predictions in a single hardware environment, under a
pre-registered protocol fixed before any data was collected. Policies are
compared on residual risk -- errors reaching users after review -- against two
cost axes that we deliberately never convert into one another.

Three results. First, the best policy depends on the review budget: a single 8B
guard on its own confidence wins at 1-5% and 50% review, while a three-guard
agreement panel wins only at 10-20%, at 2.5x the compute. Second, there is a
large and precisely estimated *identification gap*: routing each prompt to the
guard that will answer it correctly would reduce error 26.3% (95% CI [23.6,
29.2]) at half the compute of the 8B alone, and the best available escalation
signal captures none of it (realised gain 95% CI [-0.0017, +0.0012]). Third,
and we argue most consequentially, both failures share an axis. The 2.85% of
prompts on which all three guards agree *and are wrong* are confidently wrong,
are absent entirely from short curated benchmarks, and grow roughly ninefold
from the shortest to the longest length quintile; the same length variable
destroys the behavioural perturbation signal (rank correlation -1.0 between mean
prompt length and usable variant count).

On long inputs the signals that would route a case for review stop working
exactly where the panel stops catching it. Short curated benchmarks cannot
observe this, which we suggest is why it has not been reported.

---

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

This paper occupies that regime and prices it. Our contributions:

1. **An oversight frontier.** Every (guard, routing signal, review budget)
   policy priced on two axes that are never converted into one another, over
   26,958 predictions in one hardware environment (§3).
2. **Budget-dependent policy selection** (§5.2). The winner changes with the
   review budget, non-monotonically.
3. **The identification gap** (§5.4). We separate what routing *could* buy from
   what available signals *do* buy, with bootstrap intervals. The gap is large
   and the capture is statistically zero.
4. **A length axis for guard-oversight failure** (§5.5). Unanimous panel failure
   and behavioural-signal collapse are both governed by input length, which
   short curated benchmarks hold fixed.

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
replicate the first finding (§5.1) and our §5.1 result on the logit margin is
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
does not hold: 12-37% pinned, with usable interior resolution throughout (§5.1).
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

### 5.1 Native confidence is usable, and the margin is exactly the probability

All three guards have usable native confidence (deferral efficiency 0.667-0.745,
tie fractions below 0.003), replicating Safety-Flag on models it did not test.

The logit margin -- free to compute and hypothesised to carry ranking
information destroyed by sigmoid saturation -- is *identical* to the
probability. AUROC and AUROC-on-margin agree to seven decimal places on all
three guards (delta AURC <= 1.8e-08). The saturation never occurred: median
|margin| is 4.0-5.8, far short of the ~37 at which float64 sigmoid pins to 1.0,
and only 12-37% of scores are pinned at all.

The premise motivating behavioural signals -- that guard scores are degenerate
-- does not hold for these guards. It should be measured rather than assumed.

### 5.2 The best policy depends on the review budget

| Review budget | Winning policy | Residual | Compute |
|---:|---|---:|---:|
| 1% | Llama-Guard-3-8B + native | 0.0999 | 0.319 |
| 2% | Llama-Guard-3-8B + native | 0.0963 | 0.319 |
| 5% | Llama-Guard-3-8B + native | 0.0832 | 0.319 |
| **10%** | **Qwen3Guard-8B + panel agreement** | **0.0634** | **0.793** |
| **20%** | **Qwen3Guard-8B + panel agreement** | **0.0376** | **0.793** |
| 50% | Llama-Guard-3-8B + native | 0.0112 | 0.319 |

The expensive signal wins in a **middle band only**, at 2.5x the compute. The
pattern is not monotone in budget, and §7 reports a mechanism for the band that
we proposed and then rejected.

**The 1B guard never wins at any budget.** Our principal pre-registered
hypothesis is not supported.

### 5.3 Panel agreement is two-valued and helps only the weakest guard

With three guards voting, modal agreement takes two values. The tie fraction is
1.0000 on every guard and every dataset: the signal partitions traffic into a
flagged block and an unflagged one and imposes no order within either.

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

### 5.4 The identification gap

Every policy above has one guard decide every prompt. We additionally evaluate a
cascade -- the 1B decides what it can, escalating low-confidence cases to the 8B
-- recomputed from the existing predictions with no new inference.

| policy | error | 95% CI | compute |
|---|---:|---|---:|
| 8B alone | 0.1049 | [0.0987, 0.1110] | 0.319 |
| cascade, escalating on 1B confidence | 0.1052 | [0.0989, 0.1112] | 0.294 |
| **cascade, oracle escalation** | **0.0773** | **[0.0720, 0.0827]** | **0.159** |

Bootstrap over prompts, 2,000 resamples:

- **Realised** gain (8B - cascade): **-0.0002, CI [-0.0017, +0.0012]** -- straddles zero.
- **Available** gain (8B - oracle): **+0.0276, CI [+0.0243, +0.0309]** -- excludes zero.

The two guards fail on sufficiently different cases that perfect routing between
them would cut error **26.3% [23.6%, 29.2%]** at **half** the 8B's compute. The
best available escalation signal captures none of this.

The bottleneck is not compute, which is cheap and was spent freely here. It is
the inability to identify which decisions are wrong.

### 5.5 Where the panel is blind, and why length is the axis

On 2.85% of prompts (256) all three guards agree **and all three are wrong**:
167 benign prompts flagged, and 89 harmful prompts admitted. Three properties
make this set more than a residual.

**They are confidently wrong.** Median |p - 0.5| is 0.459-0.480 when the panel
is unanimously wrong against 0.493-0.499 when it is unanimously right -- both
near the 0.5 ceiling. No confidence-based routing can reach them, because
confidence does not separate them. This is why the 2.85% is a floor rather than
a budget shortfall.

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
(median 30 chars), 2.17%, 3.07%, 2.28%, **6.12%** (median 869 chars) -- roughly
ninefold from shortest to longest.

The same variable governs the behavioural signal. Our perturbation family --
meaning-preserving surface edits: inserted greetings, expanded contractions,
opener substitutions -- produced too few distinguishable variants to score on
three of five datasets, and the diversity gate refused them before inference:

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
measured (XSTest), instability lost outright -- AURC 0.0295 against native's
0.0157, catching 10 errors at 5% review where native caught 12, at six times the
compute.

> **Guard oversight degrades with input length from both directions at once.**
> The signals that would route a case for review stop working exactly where the
> panel stops catching it. This is a property of the inputs, not of a method,
> and short curated benchmarks hold the variable fixed.

### 5.6 Discrimination is not deployment

Gate A1 reports that one guard beats another at matched compute by 10.7%. The
comparison is Llama-Guard-3-8B against Qwen3Guard-Gen-8B, *both using native
confidence* -- there is no difference in selection, and we do not present this
as an allocation result. What it does show is worth separating:

Qwen3Guard has the highest AUROC of the three (0.9615 vs 0.9226) and the worse
deployed error rate (0.1157 vs 0.1049), because its operating point over-flags:
31.4% flagged against a 22.5% base rate. **Selecting a guard by AUROC picks
Qwen; selecting by deployed residual risk picks Llama-Guard-3-8B.** Guard
evaluations predominantly report the former.

---

## 6. Discussion

Five routes to reducing residual risk were priced, and each is bounded:

| route | bound |
|---|---|
| logit margin | identical ordering to the probability |
| perturbation instability | unmeasurable at realistic prompt length |
| cross-model disagreement | two-valued; wins only in a 10-20% band |
| cascaded escalation | captures 0 of 0.0276 available headroom |
| any panel routing | 2.85% unanimous-failure floor, confidently wrong |

None is a compute limit. Each is a failure to identify which decisions are
wrong. The practical recommendation that follows is unglamorous: at review
budgets below 10%, run the best single guard on its own confidence and spend
nothing on routing, because nothing available beats it.

The length result suggests where effort should go instead. If unanimous failure
and signal collapse are both governed by input length, then evaluation on short
curated benchmarks systematically understates both, and progress requires
length-proportional perturbations -- which means a paraphrase model, which
reintroduces a confound we deliberately excluded: a paraphraser can sanitise a
harmful prompt, making a flipped verdict uninterpretable.

---

## 7. A mechanism we proposed and rejected

The natural explanation for the 10-20% band in §5.2 is that the signal is
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
deployments actually run, the answer is to buy the better guard and spend
nothing on routing. Everything we priced that costs more does not beat it
outside a narrow band.

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
