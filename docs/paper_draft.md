# Buy the Better Guard: What a Fixed Oversight Budget Buys in Safety Moderation

**Md. Zariful Huq**, Independent Researcher, zarif.huq.career@gmail.com

Code and data: [github.com/zarif007/guard-uncertainty-routing](https://github.com/zarif007/guard-uncertainty-routing)

---

## Abstract

Content-moderation guard models make mistakes, and deployments cover for them by
routing uncertain decisions to human reviewers. A fixed oversight budget can buy
a larger guard, compute spent checking the guard's answer, or more reviewer
time. These compete, and we know of no work that prices them against each
other. We price them over 26,958 predictions from three generative guards on five moderation
datasets, all scored on one machine, under a protocol pre-registered before any
data was collected.

**The answer is to buy the better guard.** Pooled over the five datasets, at
1%, 2%, 5% and 50% human review, the strongest single guard, ranked by its own
free confidence, ships the fewest bad decisions. A three-guard agreement panel
wins only between 10% and 20% review, at 2.5x the compute. The smallest guard
never wins at any budget, so our pre-registered hypothesis, that a cheap guard
with expensive checking beats an expensive guard with free checking, is not
supported. Within single datasets the picture is more mixed: which guard is
best depends on the traffic, and on real user traffic the panel wins between
2% and 20% review.

The useful part is why. The alternatives are not limited by compute. They are
limited by identification: knowing which decisions are wrong. Routing each
prompt to whichever of two guards would answer it correctly would cut error by
26.3% (95% CI [23.5, 29.2]) at half the compute of the larger guard, and the
best available escalation signal captures none of it (realised gain 95% CI
[-0.0017, +0.0012]). Identification fails most on long inputs. On 2.85% of
prompts all three guards agree and all three are wrong, with high confidence,
and the rate of these failures rises roughly ninefold from the shortest fifth of
prompts to the longest. Length also breaks the perturbation signal: the number
of usable rewordings per prompt falls as prompts get longer (rank correlation
-1.0 across datasets). Benchmarks of short prompts barely vary length, so they
cannot show either effect.

## 1. Introduction

A guard model sits in front of a deployed language model and decides whether an
incoming message should be blocked. It is wrong in both directions: it blocks
benign requests and admits harmful ones. Operators compensate with human review,
but review is scarce. We take 1-20% of decisions as the range a moderation team
can realistically review, and treat 50% as an upper reference point.

That scarcity makes the operator's problem an allocation problem. A fixed
oversight budget can buy a larger and more accurate guard, extra compute spent
checking the guard's answer, or more reviewer time. These compete, and the
exchange rate between them is unknown.

The nearest prior work, Safety-Flag [1], benchmarks thirteen moderation models
on seven datasets and finds that a model's own token-logprob confidence ranks
its errors usefully. It weighs misses against false alarms, but prices neither
compute nor reviewer time. Its tables report risk at 100%, 80% and 50%
coverage, on items balanced roughly 50/50 between harmful and benign, so most
of the 1-20% review range an operator works in is not tabulated.

This paper works in that range and answers the allocation question for it. Our
aim is an answer a practitioner can act on, not a framework. Contributions, in
the order they matter:

1. **The answer** (§5.1). At the review rates deployments run, buy the better
   guard and spend nothing on routing. The one exception is a 10-20% band where
   a three-guard panel pays, at 2.5x the compute. The smallest guard never wins.
2. **A price for every alternative** (§3, §5). Each (guard, routing signal,
   review budget) policy is priced on two axes, compute and review, that are
   never converted into each other, over 26,958 predictions from one machine.
   To our knowledge, guard oversight has not been priced this way before.
3. **Why the alternatives lose: the identification gap** (§5.4). We separate
   what routing *could* buy from what available signals *do* buy, with
   bootstrap intervals. The gap is large and the capture is statistically zero.
4. **Where identification breaks down: input length** (§5.5). Unanimous panel
   failure and the collapse of the perturbation signal both track input length,
   which benchmarks of short prompts barely vary.

**Pre-registration.** The hypotheses, decision thresholds and gate structure
were committed to a public repository before any real prediction existed, and
frozen once the first one did. Two consequences are visible in this paper and
we report both. Our principal hypothesis was *not* supported, and a mechanism we
proposed after seeing the data was specified, tested on eight panels, and
rejected (§7).

---

## 2. Related work

**Selective prediction for moderation.** Safety-Flag [1] benchmarks the
reliability and calibration of content moderators across seven datasets and
thirteen models. Among its reliability metrics is the area under the
risk-coverage curve (AURC) [17, 18], which is also our main ranking measure.
It finds that token-logprob confidence is usable, and that temperature scaling
improves calibration without changing how predictions are ordered. We
replicate the first finding (§5.2), and our result on the logit margin is a
sharper form of the second.

Its Appendix D.1 is a direct headwind for behavioural signals: agreement among
five sampled answers loses to token-logprob confidence on all six models
tested. We vary the *input* rather than the *decoder*, since a guard scored
greedily on one label token has no sampling variance at all. In outcome, our
§5.5 result agrees with theirs.

**Uncertainty beyond confidence.** Semantic uncertainty [2] estimates a
model's uncertainty from how much its sampled outputs agree in meaning. We
apply the related idea of agreement under meaning-preserving variation to
routing guard errors, varying the input rather than the output. A natural
worry is that instability merely restates a small margin at several times the
cost; our gate S4 tests exactly this. Entropy alone has been argued to be
insufficient for safe selective prediction [3], and abstention with provable
guarantees has been developed for question answering [4]. Neither addresses a
moderation classifier with a human-review budget and a compute cost.

**Polarized guard scores.** A common worry is that guard verdict probabilities
sit at 0 or 1, which would leave nothing to rank. We measured this on our own
models rather than assuming it, and it does not hold: 12-37% of scores are
pinned, with usable resolution in between (§5.2).

**Operating points and evaluation conditions.** Reasoning can improve a safety
detector's accuracy and still hurt recall at the low false-positive rates
deployments need [5]. Measured safety can depend on evaluation conditions such
as answer format and scoring rules [6]. Some comparisons fix one decision
threshold across models rather than tuning each [7], and classifier quality can
look very different under true distribution shift [8]. Our §5.6 adds a case of
the same kind: the guard with the better AUROC is the worse deployment.

**What is not covered.** In this literature we find no compute or review-cost
analysis of guard oversight, no results tabulated across the 1-20% review
range, no cross-model disagreement as a routing signal, and no comparison of
logit margin against probability.

---

## 3. Method

### 3.1 Policies and the objective

A **policy** is a (guard, routing signal) pair together with a review budget
`b`. The signal gives every prompt a score, and the `b` least-confident fraction
goes to a human, who is assumed to correct any error shown to them. No
thresholds are involved. Only the ordering a signal induces is used, so signals
never need to be calibrated onto a common scale.

The objective is **residual risk**:

```
residual = errors the guard made  -  errors the reviewer saw
```

reported as a rate over all prompts. We use residual rather than accuracy or
errors caught, because both flatter a weak guard. A guard that makes 1,614
errors and catches 1,128 ships more harm than one that makes 943 and catches
561.

### 3.2 Two cost units, never mixed

Compute and reviewer time are not measured in the same unit, and any exchange
rate dates quickly. Nothing in this study converts one into the other. Policies
are reported on a two-dimensional frontier, and readers can apply their own
prices.

Compute is the median measured latency per decision, in seconds, on the
evaluation machine. A panel runs every member on every prompt, so its price is
the **sum of its members' costs**. It is a property of the panel, not of
whichever member is nominated to decide.

### 3.3 Guards, data, signals

Three generative guards, all scored at fp16 through llama.cpp: Llama-Guard-3-8B
[9], Qwen3Guard-Gen-8B [11] and Llama-Guard-3-1B [10].

| guard | error rate | AUROC | flag rate | compute |
|---|---:|---:|---:|---:|
| Llama-Guard-3-8B | 0.1049 | 0.9226 | 0.200 | 0.319 |
| Qwen3Guard-Gen-8B | 0.1157 | 0.9615 | 0.314 | 0.372 |
| Llama-Guard-3-1B | 0.1796 | 0.8704 | 0.283 | 0.102 |
| *(three-guard panel)* | | | | *0.793* |

Five datasets, 8,986 prompts after normalisation, with a pooled harmful rate of
22.5%: ToxicChat [12] (4,972 prompts; 7.1% harmful; real user traffic),
WildGuardTest [13] (1,699), OpenAI Moderation [14] (1,665), XSTest [15] (450;
benign but borderline) and HarmBench [16] (200; all harmful). Only ToxicChat has
the low harmful rate of real traffic, so the pooled results describe a harder
mix than most deployments see.

Four routing signals:

- **native confidence**, `|p_unsafe - 0.5|` after per-guard temperature scaling
   (free; the baseline);
- **logit margin**, `|margin|` (free);
- **three-guard agreement**, the share of guards that agree with the majority
   verdict (three times the compute);
- **perturbation instability**, how often the verdict flips over six
   meaning-preserving rewordings (six times the compute).

The pre-registration also lists a fifth signal, agreement of one guard with
itself across quantization levels. It was not run, because those quantized
models were never scored.

**Environment.** All 26,958 predictions were scored on one NVIDIA L40S. Every
prediction row carries a hash of the environment, and the analysis refuses to
compare latency across different hashes. Compute prices are therefore valid on
this machine and only on it.

---

## 4. Pre-registered gates

Each gate can fail. Before any real scoring, every gate was run on synthetic
data whose answer is known by construction, in both directions (13 of 13 checks
pass). Thresholds were fixed in advance: a 5% relative reduction in residual
risk for the allocation gate, and 0.01 AURC for signal comparisons.

| Gate | Question | Result |
|---|---|---|
| A1 | At matched compute, does anything beat the default? | `CHEAPER_GUARD_WINS` (a different guard, not a different signal; §5.6) |
| A2 | Does the winner change with budget? | `BUDGET_DEPENDENT` |
| S1 | Is native confidence usable? | `NATIVE_USABLE` (3 of 3 guards) |
| S2 | Does the margin beat the probability? | `NO_DIFFERENCE` |
| S3 | Does any alternative beat native confidence? | `ALTERNATIVE_BEATS_NATIVE` (1 of 3 guards) |
| S4 | Is instability independent of the margin? | `INDEPENDENT`, narrowly, on 650 short prompts (§5.5) |
| S5 | Does guard *kind* decide usability? | Not tested: no suitable encoder guard |

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

Figure 2 shows the whole price list behind this table: every policy as a point
with its compute on one axis and its residual risk on the other. Below 10%
review, nothing more expensive than Llama-Guard-3-8B on its own confidence
lowers residual risk. From 10% review the panel joins the frontier, and it is
the only policy that buys a lower residual with more compute.

The one exception is a **10-20% band**, where a three-guard agreement panel
wins at **2.5x the compute**. The pattern is not monotone in budget, and §7
reports a mechanism for the band that we proposed and then rejected.

**The smallest guard never wins at any budget.** Our pre-registered hypothesis,
that a cheap guard with expensive checking beats an expensive guard with free
checking, is not supported.

![FIG:fig1_budget_crossover.png|The answer, and its one exception. Residual risk against human review budget for four policies. Llama-Guard-3-8B on its own confidence wins below 10% review and at 50%; the three-guard panel wins only in the 10-20% band, at 2.5x the compute. The inset magnifies the crossover. The 1B guard never wins at any budget.]

![FIG:fig7_oversight_frontier.png|The oversight frontier at four review budgets. Each point is a policy: its compute per decision (median seconds on one L40S) against its residual risk. Circles use a guard's own confidence; squares use three-guard agreement, which costs the sum of all three guards. The dashed line joins policies that no cheaper policy beats. The 1B guard sits on it only because it is the cheapest; it never has the lowest risk.]

**Per dataset, the answer depends on the traffic.** These results pool five
datasets, as pre-registered. Appendix A splits them, and two things change.
First, the best single guard differs: Qwen3Guard-Gen-8B has the lowest error on
XSTest, HarmBench and WildGuardTest, and Llama-Guard-3-8B wins the pooled
comparison mainly because Qwen3Guard over-flags OpenAI Moderation (error 0.279
against 0.130). Second, the panel wins more often within a dataset: on
ToxicChat, the only set with a realistic harmful rate, it beats the best single
guard at 2%, 5%, 10% and 20% review, by 2-23% relative. Across all 30
(dataset, budget) cells, a single guard on its own confidence wins 22 and the
panel wins 8. The smallest guard never has the lowest risk outright.

The rest of this section is about **why** the alternatives usually lose,
because that is the part most likely to carry over to guards we did not test.

### 5.2 The free signals work, and cannot be improved for free

All three guards have usable native confidence (deferral efficiency
0.667-0.745, tie fractions below 0.003). This replicates Safety-Flag [1] on
Llama-Guard-3-8B and extends it to two guards it did not test. It is also why
§5.1's answer does not mean routing is useless: the free signal already works.

![FIG:fig2_risk_coverage.png|Risk-coverage curves for native confidence on all three guards, pooled over the five datasets. Risk among the decisions a guard keeps falls as more are sent to review. The shaded band marks 1-20% review, the range this study targets; Safety-Flag tabulates risk only at 50%, 80% and 100% coverage.]

The other free signal adds nothing. We expected the logit margin to keep
ranking information that the sigmoid destroys when it saturates. Instead it
ranks exactly like the probability: AUROC computed on the margin equals AUROC
on the probability for all three guards, and AURC differs by at most 1.8e-08.
The saturation never happened. Median `|margin|` is 4.0-5.8, far below the
roughly 37 at which a float64 sigmoid rounds to exactly 1.0, and only 12-37% of
scores are pinned at all.

So the premise behind behavioural signals, that guard scores are degenerate,
does not hold for these guards. It should be measured rather than assumed.

### 5.3 The panel pays only in a band, and only for the weakest guard

With three guards voting, agreement takes just two values. The tie fraction is
1.0 on every guard and every dataset: the signal splits traffic into a flagged
block, where the guards disagree, and an unflagged one, and orders nothing
within either. That coarseness is why it can win only where the budget happens
to suit it.

The flagged block is 23.8% of traffic (2,141 prompts) and holds 72.9-84.1% of
each guard's errors. But the benefit is not uniform:

| guard | change in AURC vs native | 95% CI | Holm p |
|---|---:|---|---:|
| Llama-Guard-3-1B | **-0.0142** | [-0.0201, -0.0085] | 0.000 |
| Llama-Guard-3-8B | +0.0158 | [+0.0101, +0.0204] | 1.000 |
| Qwen3Guard-8B | +0.0092 | [-0.0011, +0.0103] | 1.000 |

Negative is better; p-values are Holm-corrected [19]. Agreement is a large
improvement that survives correction for the 1B, and runs in the *wrong*
direction for both 8B guards. A strong guard gains nothing from consulting a
weaker one. We note this because "ensemble disagreement helps" is often
asserted without this split.

![FIG:fig3_agreement_forest.png|Change in AURC from adding three-guard agreement, with 95% bootstrap intervals and Holm-corrected p-values. Only the 1B guard improves. For both 8B guards the point estimate runs the wrong way, and neither difference survives correction.]

### 5.4 How much is left on the table

In every policy above, one guard decides every prompt. We also evaluate a
cascade: the 1B decides, and its least confident 60% of prompts are escalated
to the 8B. It is computed from the existing predictions, with no new inference.

| policy | error | 95% CI | compute |
|---|---:|---|---:|
| 8B alone | 0.1049 | [0.0985, 0.1113] | 0.319 |
| cascade, escalating on 1B confidence | 0.1052 | [0.0988, 0.1118] | 0.294 |
| **cascade, oracle escalation** | **0.0773** | **[0.0717, 0.0830]** | **0.159** |

The oracle escalates exactly the prompts the 1B gets wrong. Bootstrapping over
prompts with 2,000 resamples:

- **Realised** gain (8B minus cascade): **-0.0002, CI [-0.0017, +0.0012]**,
   which straddles zero.
- **Available** gain (8B minus oracle): **+0.0276, CI [+0.0243, +0.0310]**,
   which excludes zero.

The two guards fail on different enough cases that perfect routing between them
would cut error by **26.3% [23.5%, 29.2%]** at **half** the 8B's compute. The
best available escalation signal captures none of it.

So the reason to buy the guard rather than the routing is not that routing has
little to offer. It has a lot to offer. Nothing available collects it.

![FIG:fig4_identification_gap.png|The identification gap. Error bars are 95% bootstrap intervals over 2,000 resamples. Escalating on the 1B guard's own confidence is indistinguishable from the 8B alone. Escalating with an oracle cuts error by 26.3% at half the compute. The distance between the dotted lines is available and not captured.]

### 5.5 Where identification breaks down: input length

On 2.85% of prompts (256), all three guards agree **and all three are wrong**:
167 benign prompts flagged, and 89 harmful prompts let through. Three
properties explain why no confidence-based policy in §5.1 could find them.

**They are confidently wrong.** The median distance from the decision
threshold, |p - 0.5|, is 0.459-0.480 when the panel is unanimously wrong and
0.493-0.499 when it is unanimously right, both close to the 0.5 maximum.
Confidence does not separate them, so no review budget spent on a confidence
ranking reaches them.

**They are rare in the shortest, most curated set.** HarmBench contributes
**none**. The rate is highest on the two long-form sets:

| dataset | n | unanimously wrong | rate |
|---|---:|---:|---:|
| WildGuardTest | 1,699 | 87 | 5.12% |
| OpenAI Moderation | 1,665 | 82 | 4.92% |
| XSTest | 450 | 11 | 2.44% |
| ToxicChat | 4,972 | 76 | 1.53% |
| **HarmBench** | 200 | **0** | **0.00%** |

**Length tracks them, within datasets as well as across them.** Median prompt
length of unanimous failures against the rest of the same dataset is 967 vs 363
characters (OpenAI Moderation), 548 vs 236 (WildGuardTest), 83 vs 59 (ToxicChat)
and 45 vs 42 (XSTest). Pooled by length quintile, the unanimous-failure rate is
0.65% (median 30 characters), 2.17%, 3.07%, 2.28% and **6.12%** (median 869
characters): roughly ninefold from shortest to longest.

The same variable governs the perturbation signal. Our rewordings are
meaning-preserving surface edits, such as inserting a greeting, expanding a
contraction or swapping an opening phrase. A diversity check, run before any
inference, refused three of the five datasets because too few rewordings
differed enough from the original:

| dataset | mean prompt characters | usable rewordings (of 6) |
|---|---:|---:|
| XSTest | 44 | 5.0 |
| HarmBench | 89 | 4.4 |
| ToxicChat | 167 | 3.5 (refused) |
| WildGuardTest | 480 | 2.4 (refused) |
| OpenAI Moderation | 645 | 2.0 (refused) |

The rank correlation between mean prompt length and usable rewordings is
**-1.0**. The mechanism is arithmetic. A surface edit is a fixed-size change, so
its relative size is that fixed size divided by prompt length.

On the two datasets that passed (650 prompts, Llama-Guard-3-8B only),
instability is not just a copy of the margin: gate S4 passes, but narrowly, with
a correlation of 0.49 against a cutoff of 0.5. It still ranks errors worse than
native confidence. On XSTest its AURC is 0.0295 against native's 0.0157, and at
5% review it catches 10 errors where native catches 12. On HarmBench it does
worse than random ordering, catching none of the 5 errors at 5% review where
native catches 3. All of this costs six times the compute.

![FIG:fig5_length_axis.png|Input length governs both failures. (a) The rate at which all three guards agree and all three are wrong, by prompt length quintile: 0.65% at a median of 30 characters, rising to 6.12% at 869. (b) Usable rewordings per prompt against mean prompt length, by dataset. The three datasets below the usability threshold are the three with the longest prompts, and could not be scored.]

> **Guard oversight degrades with input length from two directions at once.**
> The signals that would route a case for review stop working on the same long
> inputs where the panel stops catching mistakes. Benchmarks of short prompts
> barely vary length, so they cannot show this.

### 5.6 A warning about how the guard is chosen

§5.1 says buy the better guard. This section is about what "better" has to
mean.

Gate A1 reports that, at matched compute, one policy beats the default by
10.7%. The default is the most expensive free-signal policy, Qwen3Guard-Gen-8B
on its own confidence. The winner is Llama-Guard-3-8B on *its* own confidence,
at 0.86x the compute. Both use the same kind of signal, so this is a
comparison of guards, not of ways to check them, and we do not present it as
an allocation result. What it shows is still worth stating.

Qwen3Guard has the highest AUROC of the three (0.9615 vs 0.9226) and the higher
deployed error rate (0.1157 vs 0.1049), because its operating point over-flags:
it flags 31.4% of traffic against a 22.5% harmful rate. **Choosing a guard by
AUROC picks Qwen; choosing by deployed residual risk picks Llama-Guard-3-8B.**
Guard evaluations mostly report the former.

Even the residual ranking holds only for this mix. Per dataset (Appendix A),
Qwen3Guard has the lower error on three of five sets and Llama-Guard-3-8B on
one, with ToxicChat nearly tied. "The better guard" is a property of the guard *and*
the traffic, and has to be measured on traffic like the deployment's.

![FIG:fig6_auroc_vs_error.png|Ranking quality against deployed error. Qwen3Guard-Gen-8B ranks best and deploys worst, because it flags 31.4% of traffic against a 22.5% harmful rate.]

---

## 6. Discussion

**The recommendation.** At review budgets below 10%, run the strongest single
guard on its own confidence and spend nothing on routing. Between 10% and 20%,
a three-guard panel is worth 2.5x the compute. Above that, go back to the single
guard. Choose that guard by deployed residual risk on your own traffic, not by
AUROC (§5.6). These are pooled results; on traffic like ToxicChat the panel
starts to pay from about 2% review (Appendix A), so the crossover point should
be measured, not assumed.

**Why it holds.** Five routes to lower residual risk were priced, and each one
is bounded:

| route | bound |
|---|---|
| logit margin | ranks identically to the probability |
| perturbation instability | unmeasurable at realistic prompt length; loses to native where measurable |
| cross-model disagreement | two-valued; wins only in a 10-20% band |
| cascaded escalation | captures none of the 0.0276 available |
| any panel routing | 2.85% of prompts are unanimous, confident failures |

None of these is a compute limit. Compute is cheap and we spent it freely. Each
is a failure to identify which decisions are wrong, and §5.4 puts a number on
that failure: 26.3% of error, available and not collected.

**Where to spend effort instead.** If unanimous failure and signal collapse are
both governed by input length, then evaluation on short prompts understates
both, and progress needs perturbations that scale with length. That means a
paraphrase model, which brings back a confound we deliberately excluded: a
paraphraser can sanitise a harmful prompt, which makes a flipped verdict
impossible to interpret. Resolving that confound is the next problem, rather
than finding a sixth signal.

---

## 7. A mechanism we proposed and rejected

The natural explanation for the 10-20% band in §5.1 is that agreement runs out
once the review budget exceeds the share of prompts it flags. That predicts the
band *ends* at the flag rate. We tested it: three guards give three pairs plus
the triple, each panel with its own flag rate, and residuals were recomputed
over a 1-50% grid, with ties broken at random over 20 seeds.

The prediction fails. Over the eight panels that win anywhere, the Spearman
correlation between flag rate and band end is 0.435, and the band runs past the
flag rate in all eight, by 0.155 on average. The flag rate is where agreement
stops *gaining*, not where it stops *leading*. We report this rather than
leaving it out.

---

## 8. Limitations

1. **One machine.** Compute prices hold on one L40S and nowhere else, by design.
   A1's default is the most expensive free-signal policy *on this hardware*;
   elsewhere a different guard could be the default.
2. **Three guards, one family of alternatives.** Panel agreement is effectively
   the only alternative signal with full coverage. S2 was null, S4 could be
   measured only on 650 short prompts and one guard, and S5 was not tested.
3. **No encoder guard.** The one candidate scored an AUROC of 0.599 against our
   label (it detects toxic content, not harmful requests) and was withdrawn
   before scoring. S5 remains open and we make no claim about it.
4. **A harder mix than real traffic, and pooled.** The pooled harmful rate is
   22.5%. Only ToxicChat (7.1%) is close to the low rates of real deployments,
   and on ToxicChat alone the panel wins at more budgets than in the pooled
   result (Appendix A). The headline describes the pooled mix.
5. **Idealised review.** Reviewers are assumed to fix every error they see, and
   dataset labels are taken as correct.
6. **Sections 5.4, 5.5 and 7 are exploratory**, derived after seeing the
   pre-registered results. The oracle in §5.4 bounds this guard pair only.
7. **The length result is correlational.** It holds within datasets, which rules
   out differences between datasets as the cause, but length may stand in for
   topic or complexity.
8. **Perturbation was tested in its cheap form only.** We did not test a
   paraphrase model, for the reason in §6, so §5.5 bounds surface perturbation,
   not behavioural signals in general.

---

## 9. Conclusion

Given a fixed oversight budget for a safety guard, at the review rates
deployments actually run, **buy the better guard and spend nothing on
routing**. On our pooled data, everything else we priced fails to beat it
outside a narrow 10-20% band, and the smallest guard never wins at all. Which
guard is better, and where the panel starts to pay, depend on the traffic, so
both should be measured on traffic like your own.

That answer is unglamorous, and it is not the one we pre-registered. What makes
it worth having is the price list behind it and the reason underneath it.

The reason can be measured: perfect routing between two guards we already have
would cut error by a quarter at half the compute, and no available signal
captures any of it. The cases where oversight fails hardest are the long ones,
where the panel is confidently and unanimously wrong, and where the signals
that would flag them have stopped working.

---

## Ethics statement

This work evaluates safety classifiers on public datasets that contain harmful
and offensive prompts. It creates no new harmful content: the rewordings are
surface edits of existing prompts. We release predictions, not new prompts. For
the three datasets with gated access or non-commercial licenses, the released
files omit the prompt text, which can be rejoined from the original sources.

## Reproducibility

The pre-registration, analysis code, all 26,958 predictions, and the scripts
that produce every table and figure are public at the repository above. Every
gate is checked against synthetic data with known answers (13 of 13 checks
pass). Claims are licensed by a generated table rather than chosen by hand, and
claims the table does not license are not made here. All tables and figures
regenerate without a GPU from the released predictions.

---

## References

[1] Yibo Hu. Safety-Flag: A Unified Benchmark for the Reliability and
Calibration of LLM Content Moderators. arXiv:2609.19072, 2026.

[2] Lorenz Kuhn, Yarin Gal, Sebastian Farquhar. Semantic Uncertainty:
Linguistic Invariances for Uncertainty Estimation in Natural Language
Generation. ICLR 2023. arXiv:2302.09664.

[3] Edward Phillips, Fredrik K. Gustafsson, Sean Wu, Anshul Thakur, David A.
Clifton. Entropy Alone is Insufficient for Safe Selective Prediction in LLMs.
arXiv:2603.21172, 2026.

[4] Sijin Dong, Hiroyuki Shinnou. Uncertainty-Aware Abstention in Large Language
Models with Provable Alignment Guarantees. arXiv:2607.04430, 2026.

[5] Atoosa Chegini, Hamid Kazemi, Garrett Souza, Maria Safi, Yang Song, Samy
Bengio, Sinead Williamson, Mehrdad Farajtabar. Reasoning's Razor: Reasoning
Improves Accuracy but Can Hurt Recall at Critical Operating Points in Safety
and Hallucination Detection. arXiv:2510.21049, 2025.

[6] David Gringras. Safety Under Scaffolding: How Evaluation Conditions Shape
Measured Safety. arXiv:2603.10044, 2026.

[7] Edson Rodrigues da Cruz Filho, Paulo Ricardo Ferreira Neves, Paulo Henrique
Eleuterio Falsetti et al. A Reproducible, License-Aware Distillation Recipe for
CPU-Deployable Safety Classification. arXiv:2608.21570, 2026.

[8] Max Fomin. When Benchmarks Lie: Evaluating Malicious Prompt Classifiers
Under True Distribution Shift. arXiv:2602.14161, 2026. Extended version of a
paper at the AIWILD workshop, ICLR 2026.

[9] Hakan Inan, Kartikeya Upasani, Jianfeng Chi et al. Llama Guard: LLM-based
Input-Output Safeguard for Human-AI Conversations. arXiv:2312.06674, 2023.

[10] Igor Fedorov et al. Llama Guard 3-1B-INT4: Compact and Efficient Safeguard
for Human-AI Conversations. arXiv:2411.17713, 2024.

[11] Haiquan Zhao, Chenhan Yuan, Fei Huang et al. Qwen3Guard Technical Report.
arXiv:2510.14276, 2025.

[12] Zi Lin, Zihan Wang, Yongqi Tong et al. ToxicChat: Unveiling Hidden
Challenges of Toxicity Detection in Real-World User-AI Conversation. Findings
of EMNLP 2023. arXiv:2310.17389.

[13] Seungju Han, Kavel Rao, Allyson Ettinger et al. WildGuard: Open One-Stop
Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs. NeurIPS
2024, Datasets and Benchmarks Track. arXiv:2406.18495.

[14] Todor Markov, Chong Zhang, Sandhini Agarwal et al. A Holistic Approach to
Undesired Content Detection in the Real World. AAAI 2023. arXiv:2208.03274.

[15] Paul Röttger, Hannah Rose Kirk, Bertie Vidgen, Giuseppe Attanasio, Federico
Bianchi, Dirk Hovy. XSTest: A Test Suite for Identifying Exaggerated Safety
Behaviours in Large Language Models. NAACL 2024. arXiv:2308.01263.

[16] Mantas Mazeika, Long Phan, Xuwang Yin et al. HarmBench: A Standardized
Evaluation Framework for Automated Red Teaming and Robust Refusal. ICML 2024,
PMLR 235:35181-35224. arXiv:2402.04249.

[17] Ran El-Yaniv, Yair Wiener. On the Foundations of Noise-free Selective
Classification. Journal of Machine Learning Research, 11(53):1605-1641, 2010.

[18] Yonatan Geifman, Ran El-Yaniv. Selective Classification for Deep Neural
Networks. Advances in Neural Information Processing Systems 30 (NeurIPS 2017),
4878-4887.

[19] Sture Holm. A Simple Sequentially Rejective Multiple Test Procedure.
Scandinavian Journal of Statistics, 6(2):65-70, 1979.

---

## Appendix A. Results per dataset

These tables split the pooled results by dataset. They are exploratory: the
pre-registered analysis is pooled. Review budgets apply within each dataset.

**Error rate of each guard, by dataset.**

| dataset | n | harmful | Llama-Guard-3-8B | Qwen3Guard-8B | Llama-Guard-3-1B |
|---|---:|---:|---:|---:|---:|
| ToxicChat | 4,972 | 7.1% | **0.077** | 0.078 | 0.162 |
| WildGuardTest | 1,699 | 44.4% | 0.172 | **0.089** | 0.201 |
| OpenAI Moderation | 1,665 | 30.9% | **0.130** | 0.279 | 0.234 |
| XSTest | 450 | 44.4% | 0.096 | **0.082** | 0.151 |
| HarmBench | 200 | 100.0% | 0.025 | **0.000** | 0.050 |

**Lowest-risk policy, by dataset and review budget.** "own" is a guard on its
own confidence; "panel" is that guard deciding with three-guard agreement
choosing what to review. Ties go to the cheaper policy, which is why the 1B
appears on HarmBench at 50% review, where every policy reaches zero. The
residual of the best single-guard policy is shown for comparison.

| dataset | budget | best policy | residual | best single guard | residual |
|---|---:|---|---:|---|---:|
| ToxicChat | 1% | Llama-Guard-3-8B + own | 0.0720 | Llama-Guard-3-8B + own | 0.0720 |
| ToxicChat | 2% | **Llama-Guard-3-8B + panel** | 0.0664 | Llama-Guard-3-8B + own | 0.0680 |
| ToxicChat | 5% | **Llama-Guard-3-8B + panel** | 0.0533 | Llama-Guard-3-8B + own | 0.0563 |
| ToxicChat | 10% | **Qwen3Guard-8B + panel** | 0.0396 | Qwen3Guard-8B + own | 0.0426 |
| ToxicChat | 20% | **Llama-Guard-3-8B + panel** | 0.0171 | Qwen3Guard-8B + own | 0.0223 |
| ToxicChat | 50% | Qwen3Guard-8B + own | 0.0040 | Qwen3Guard-8B + own | 0.0040 |
| WildGuardTest | 1% | Qwen3Guard-8B + own | 0.0836 | Qwen3Guard-8B + own | 0.0836 |
| WildGuardTest | 2% | Qwen3Guard-8B + own | 0.0806 | Qwen3Guard-8B + own | 0.0806 |
| WildGuardTest | 5% | Qwen3Guard-8B + own | 0.0689 | Qwen3Guard-8B + own | 0.0689 |
| WildGuardTest | 10% | Qwen3Guard-8B + own | 0.0577 | Qwen3Guard-8B + own | 0.0577 |
| WildGuardTest | 20% | Qwen3Guard-8B + own | 0.0365 | Qwen3Guard-8B + own | 0.0365 |
| WildGuardTest | 50% | Qwen3Guard-8B + own | 0.0129 | Qwen3Guard-8B + own | 0.0129 |
| OpenAI Moderation | 1% | **Llama-Guard-3-8B + panel** | 0.1261 | Llama-Guard-3-8B + own | 0.1267 |
| OpenAI Moderation | 2% | Llama-Guard-3-8B + own | 0.1213 | Llama-Guard-3-8B + own | 0.1213 |
| OpenAI Moderation | 5% | Llama-Guard-3-8B + own | 0.1081 | Llama-Guard-3-8B + own | 0.1081 |
| OpenAI Moderation | 10% | Llama-Guard-3-8B + own | 0.0853 | Llama-Guard-3-8B + own | 0.0853 |
| OpenAI Moderation | 20% | Llama-Guard-3-8B + own | 0.0649 | Llama-Guard-3-8B + own | 0.0649 |
| OpenAI Moderation | 50% | Llama-Guard-3-8B + own | 0.0192 | Llama-Guard-3-8B + own | 0.0192 |
| XSTest | 1% | **Qwen3Guard-8B + panel** | 0.0756 | Qwen3Guard-8B + own | 0.0800 |
| XSTest | 2% | Qwen3Guard-8B + own | 0.0711 | Qwen3Guard-8B + own | 0.0711 |
| XSTest | 5% | **Qwen3Guard-8B + panel** | 0.0578 | Qwen3Guard-8B + own | 0.0622 |
| XSTest | 10% | **Qwen3Guard-8B + panel** | 0.0378 | Llama-Guard-3-8B + own | 0.0422 |
| XSTest | 20% | Llama-Guard-3-8B + own | 0.0244 | Llama-Guard-3-8B + own | 0.0244 |
| XSTest | 50% | Qwen3Guard-8B + own | 0.0000 | Qwen3Guard-8B + own | 0.0000 |
| HarmBench | 1% | Qwen3Guard-8B + own | 0.0000 | Qwen3Guard-8B + own | 0.0000 |
| HarmBench | 2% | Qwen3Guard-8B + own | 0.0000 | Qwen3Guard-8B + own | 0.0000 |
| HarmBench | 5% | Qwen3Guard-8B + own | 0.0000 | Qwen3Guard-8B + own | 0.0000 |
| HarmBench | 10% | Qwen3Guard-8B + own | 0.0000 | Qwen3Guard-8B + own | 0.0000 |
| HarmBench | 20% | Llama-Guard-3-8B + own | 0.0000 | Llama-Guard-3-8B + own | 0.0000 |
| HarmBench | 50% | Llama-Guard-3-1B + own | 0.0000 | Llama-Guard-3-1B + own | 0.0000 |
