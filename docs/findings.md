# Findings

Written from `results/tables_repriced/` (the primary analysis) on 2026-10-03.
`tables_no_stability/` is the same analysis before the panel-pricing fix of
§9.1 and is kept for comparison; `tables/` additionally includes the
xstest-only stability signal and should not be used.
Every number here is traceable to a generated table; none is quoted from memory.

**Provenance.** 26,958 predictions = 3 guards x 8,986 prompts x 5 datasets, all
scored on one machine (`env_hash 6ccc83087810`, NVIDIA L40S, CUDA). Hardware
consistency `CONSISTENT`, so the latency-based cost model is valid.
`docs/preregistration.md` was committed while `results/` was empty and is now
frozen.

---

## 1. The headline

> **At every review budget below 10% and at 50%, the winner is the larger guard
> run once on its own free confidence.** A three-guard panel wins only in a
> 10-20% band, the 1B guard never wins at any budget, and all cross-model
> routing is bounded by a correlated-failure floor of 2.85% that no panel size
> can reach.

The study's motivating hypothesis -- that a cheap guard with expensive checking
beats an expensive guard with free checking -- **is not supported**. That is a
pre-registered outcome (`study_protocol.md` S3) and is reported as one.

The reason is the same for every route tested, and it is not a compute limit
(§8.3): **none of the available signals can identify which decisions are
wrong.** A perfect escalation signal would cut error 26% at half the compute
(§8.2); the best measured signal captures none of that.

What is new is the pricing: the alternatives were run, measured on one machine,
and costed in both units. The prior work in `related_work.md` does none of this.

## 2. What the pre-registered gates returned

These are confirmatory. The thresholds (`MIN_AURC_IMPROVEMENT = 0.01`,
`MIN_RESIDUAL_IMPROVEMENT = 0.05`, `GATE_B_AGREEMENT = 0.99`) were fixed before
any data existed.

| Gate | Status | What it means |
|---|---|---|
| `score_range` | `USABLE` | Scores span the interval on all three guards |
| **A1** | `CHEAPER_GUARD_WINS` | **Read §3 before quoting this** |
| **A2** | `BUDGET_DEPENDENT` | 2 distinct winners across 1-50% review |
| S1 | `NATIVE_USABLE` | 3/3 guards; replicates Safety-Flag |
| S2 | `NO_DIFFERENCE` | Margin and probability rank identically |
| S3 | `ALTERNATIVE_BEATS_NATIVE` | On 1 of 3 guards only -- see §5 |
| S4 | `NOT_EVALUABLE` | Perturbation not measurable -- see §7 |
| S5 | `NOT_EVALUABLE` | Single guard kind; encoder retired (Amendment 1) |

Licensed claims, from `claims_to_evidence.csv`:

- Native guard confidence is usable for routing review
- An alternative signal beats native confidence
- At matched compute, a cheaper guard with better selection beats the default
- The best oversight policy depends on the review budget

**The third of these must not be written as stated.** See §3.

---

## 3. A1 is a model comparison, not an allocation finding

The gate fired, but the evidence underneath says something narrower than the
claim string.

```
default:      qwen3guard-gen-8b:fp16 + conf_native   residual 0.0933
best:         llama-guard-3-8b:fp16  + conf_native   residual 0.0832
reduction:    10.7% at 0.86x compute
```

**Both sides use `conf_native`.** There is no "better selection" anywhere in
this comparison -- the same free signal appears on both sides. The gate labels
it `CHEAPER_GUARD_WINS` because [`gates.py:435`](../evaluation/gates.py) tests
only `best["model"] != default["model"]`.

"The default" is likewise not the guard anyone would name. It is chosen at
[`allocation.py:224`](../evaluation/allocation.py) as the free-signal policy
with the **highest measured compute per decision**. On an L40S, Qwen3Guard-Gen-8B
(36 layers) measured slower than Llama-Guard-3-8B (32 layers), so Qwen became
the default by 0.372 vs 0.319 compute units.

### What is actually interesting here

| guard | AUROC | flag rate | error rate | compute |
|---|---:|---:|---:|---:|
| llama-guard-3-8b | 0.9226 | 0.200 | **0.1049** | 0.319 |
| qwen3guard-gen-8b | **0.9615** | 0.314 | 0.1157 | 0.372 |
| llama-guard-3-1b | 0.8704 | 0.283 | 0.1796 | 0.102 |

Base rate is 0.225.

**Qwen ranks strictly better and deploys strictly worse.** It has the highest
AUROC of the three by a wide margin (0.9615 vs 0.9226) and still ships more bad
decisions, because its operating point over-flags: 31.4% flagged against a
22.5% base rate.

That dissociation is the honest contribution of A1:

> Selecting a guard by AUROC picks Qwen. Selecting by deployed residual risk
> picks Llama-Guard-3-8B. Discrimination and deployed error are not the same
> quantity, and the guard literature reports mostly the former.

Write that. Do not write "a cheaper guard with better selection wins."

---

## 4. A2 is the result: the winner changes with budget, non-monotonically

| Review budget | Winning policy | Residual | Compute |
|---:|---|---:|---:|
| 0% | llama-guard-3-8b + conf_native | 0.1049 | 0.319 |
| 1% | llama-guard-3-8b + conf_native | 0.0999 | 0.319 |
| 2% | llama-guard-3-8b + conf_native | 0.0963 | 0.319 |
| 5% | llama-guard-3-8b + conf_native | 0.0832 | 0.319 |
| **10%** | **qwen3guard-gen-8b + conf_model_agree** | **0.0634** | **0.793** |
| **20%** | **qwen3guard-gen-8b + conf_model_agree** | **0.0376** | **0.793** |
| 50% | llama-guard-3-8b + conf_native | 0.0112 | 0.319 |

The expensive signal wins in a **middle band only**. This is not the monotone
"scarce review -> spend more compute" story; it is a band, and the band has a
mechanism (§6).

The 0% row carries no information about signals -- with no review, residual
equals the raw error rate for every signal. It is listed for completeness and
should not be plotted as a signal comparison.

---

## 5. S3 is narrower than its label

`ALTERNATIVE_BEATS_NATIVE`, but the per-model breakdown matters:

| guard | delta AURC vs native | 95% CI | Holm p | verdict |
|---|---:|---|---:|---|
| llama-guard-3-1b | **-0.0142** | [-0.0201, -0.0085] | 0.000 | **BETTER** |
| llama-guard-3-8b | +0.0158 | [+0.0101, +0.0204] | 1.000 | NO_DIFFERENCE |
| qwen3guard-gen-8b | +0.0092 | [-0.0011, +0.0103] | 1.000 | NO_DIFFERENCE |

Negative delta = lower AURC = better.

**Cross-model agreement helps only the weakest guard.** For the 1B it is a
large, correction-surviving improvement. For both 8B guards the point estimate
is in the *wrong* direction, though neither survives correction.

Mechanistically unsurprising: the 1B guard gains from consulting two stronger
models; the 8B guards gain nothing from consulting a weaker one. Worth stating,
because "ensemble disagreement helps" is usually asserted without this split.

---

## 6. Mechanism (exploratory -- not pre-registered)

Everything in this section is post-hoc analysis of the prediction files. It
explains §4 and §5 but was not specified in advance, and must be labelled as
exploratory in any write-up.

### 6.1 `conf_model_agree` is two-valued

With three guards voting, modal agreement takes only the values 3/3 and 2/3.
`tie_fraction` is **1.0000 on every model and every dataset** -- the signal
partitions traffic into two blocks and imposes no order within either.

Measured directly from the predictions:

```
prompts                8,986
unanimous              6,845  (76.2%)
disagreement set       2,141  (23.8%)
```

Share of each guard's errors falling inside the 23.8% disagreement set:

| guard | total errors | in disagreement set | share |
|---|---:|---:|---:|
| llama-guard-3-1b | 1,614 | 1,358 | **84.1%** |
| llama-guard-3-8b | 943 | 687 | **72.9%** |
| qwen3guard-gen-8b | 1,040 | 784 | **75.4%** |

### 6.2 A proposed mechanism, tested and NOT supported

The obvious explanation for the 10-20% band is that the signal is exhausted once
the review budget exceeds its flag rate. That predicts **the band ends at the
flag rate**.

It was tested on the existing predictions: three guards give three pairs plus
the triple, each panel with its own flag rate, residuals recomputed over a
1-50% grid at 0.01 steps, ties broken at random over 20 seeds so the result is
not a row-order artifact.

| decider | panel | flag rate | band | band end - flag rate |
|---|---|---:|---|---:|
| L1b | L1b+L8b | 0.130 | 0.01-0.31 | +0.180 |
| L1b | L1b+Q8b | 0.195 | 0.01-0.47 | +0.275 |
| L1b | L1b+L8b+Q8b | 0.238 | 0.01-0.50 | +0.262 |
| L8b | L1b+L8b | 0.130 | **never wins** | - |
| L8b | L8b+Q8b | 0.151 | 0.02-0.29 | +0.139 |
| L8b | L1b+L8b+Q8b | 0.238 | 0.19-0.31 | +0.072 |
| Q8b | L8b+Q8b | 0.151 | 0.01-0.34 | +0.189 |
| Q8b | L1b+Q8b | 0.195 | 0.17-0.21 | +0.015 |
| Q8b | L1b+L8b+Q8b | 0.238 | 0.12-0.38 | +0.142 |

**The prediction fails.** Spearman(flag rate, band end) = 0.435 over 8 panels,
and the band overshoots the flag rate in 7 of 8, by 0.159 on average.

The flag rate is where agreement stops *gaining*, not where it stops *leading*.
Advantage banked inside the flagged block persists past exhaustion, and native
confidence needs further budget to catch up. So the band's location is not
predictable from the flag rate alone, and no rule of the form "match the signal
to your capacity" is supported by this data.

One robust observation did survive: **Llama-Guard-3-8B paired with the 1B never
wins at any budget.** A strong guard gains nothing from a weak partner, which is
the same thing S3 says per-model.

### 6.3 The correlated-failure floor

**256 prompts (2.85% of all traffic) are misclassified identically by all three
guards.** By construction these sit in the unanimous block, so cross-model
routing can never surface them: the signal that would flag them is the signal
that says they are safe to skip.

This shows up in the allocation table as an arithmetic signature -- at 50%
review, all three guards paired with `conf_model_agree` report **exactly 137
residual errors** despite total error counts of 943, 1,040 and 1,614. Different
guards, identical residual, because in the unanimous region they are the same
classifier.

> Scaling a guard panel does not reduce correlated failure. Disagreement-based
> oversight has a floor, and that floor is the set of cases on which the panel
> is confidently and unanimously wrong.

---

## 7. Two clean negatives

### 7.1 The logit margin is exactly the probability (S2)

`p_unsafe = sigmoid(margin)` is strictly monotonic, so it induces the same
ordering unless it saturates. It did not saturate anywhere:

| guard | delta AURC | AUROC | AUROC on margin |
|---|---:|---:|---:|
| llama-guard-3-8b | 0.0 | 0.922645 | 0.922645 |
| llama-guard-3-1b | 1.8e-08 | 0.870395 | 0.870395 |
| qwen3guard-gen-8b | -1.3e-08 | 0.961459 | 0.961459 |

Identical to seven decimal places. The documented float64 saturation at margin
~37 never occurred: `margin_abs_median` is 4.0-5.8 and `frac_pinned` is
0.12-0.37 -- well short of the 99.8% polarization an industry benchmark
reports. Consistent with Safety-Flag finding dedicated guards well calibrated.

**Reportable as:** the motivating premise for behavioural signals -- that guard
scores are degenerate -- does not hold for these guards. It should be measured
before it is assumed.

### 7.2 Cheap perturbation does not survive realistic prompt length (S4)

The diversity gate (`MIN_EFFECTIVE_VARIANTS = 4` at `MIN_VARIANT_DISTANCE =
0.05`) refused to score three of five datasets:

| dataset | mean prompt chars | effective variants | gate |
|---|---:|---:|---|
| xstest | 44 | 5.0 | pass |
| harmbench | 89 | 4.4 | pass |
| toxicchat | 167 | 3.5 | **blocked** |
| wildguardtest | 480 | 2.4 | **blocked** |
| openai_moderation | 645 | 2.0 | **blocked** |

Rank correlation between mean prompt length and effective variant count is
**-1.0** across all five.

The mechanism is arithmetic: surface perturbations are fixed-size edits (insert
a greeting, expand a contraction), so relative edit distance is that fixed size
divided by prompt length. Longer prompts drive every variant toward the
original.

**The datasets that fail are the realistic ones.** xstest and harmbench are
short curated one-liners; the three blocked sets are real or long-form traffic.

Where it could be measured (xstest, 450 prompts), instability lost outright --
AURC 0.0295 against native's 0.0157, catching 10 errors at 5% review where
native caught 12, at 6x the compute.

**Reportable as:** a bounded negative with an identified mechanism, and a
condition for future work -- behavioural perturbation signals need
length-proportional edits, which means a paraphrase model, which reintroduces
the confound (a paraphraser can sanitise a harmful prompt, making a flipped
verdict uninterpretable) that `preregistration.md` commitment 8 rejects.

---

## 8. The cascade, and the identification gap (exploratory)

Post-hoc, not pre-registered. Computed on the existing prediction files -- no
new inference -- so it is a re-analysis, not a new experiment.

Every policy in the pre-registered study has **one guard decide every prompt**.
The 1B lost because it was made to judge all 8,986, including cases it is bad
at. A cascade makes the guard choice per-prompt:

```
1B decides  ->  low confidence?  ->  8B decides  ->  still low?  ->  human
```

### 8.1 It ties the 8B; it does not beat it

Three tiers against the `8B -> human` baseline, matched on compute:

| Human budget | 8B -> human | 1B -> 8B -> human | delta |
|---:|---:|---|---:|
| 1% | 0.0999 | 0.1000 (65% escalated, compute 0.294) | +0.0001 |
| 5% | 0.0832 | 0.0834 (compute 0.309) | +0.0001 |
| 10% | 0.0672 | 0.0687 | +0.0014 |
| 20% | 0.0425 | 0.0442 | +0.0017 |

Marginally worse residual at marginally lower compute. It lies **on** the
frontier without extending it. To reach the 8B's error rate the cascade must
escalate ~60-65% of traffic, at which point it has paid for the 1B plus
two-thirds of the 8B.

### 8.2 The oracle gap is the result

Escalating *exactly* the prompts the 1B gets wrong:

| policy | error | compute |
|---|---:|---:|
| 1B alone | 0.1796 | 0.102 |
| 8B alone | 0.1049 | 0.319 |
| cascade, escalating on 1B native confidence | 0.1052 | 0.294 |
| **cascade, oracle escalation** | **0.0773** | **0.159** |

`P(1B wrong AND 8B wrong) = 0.0773`, against `P(8B wrong) = 0.1049`. The two
guards fail on sufficiently different cases that perfect routing between them
would cut error by **26% relative at half the 8B's compute**.

Bootstrapped over prompts, 2,000 resamples:

| quantity | point | 95% CI |
|---|---:|---|
| 8B alone, error | 0.1049 | [0.0987, 0.1110] |
| cascade (60% escalated), error | 0.1052 | [0.0989, 0.1112] |
| cascade (oracle escalation), error | 0.0773 | [0.0720, 0.0827] |
| **realised** gain (8B - cascade) | -0.0002 | **[-0.0017, +0.0012]** |
| **available** gain (8B - oracle) | +0.0276 | **[+0.0243, +0.0309]** |
| unrealised (cascade - oracle) | +0.0278 | [+0.0244, +0.0312] |

The realised gain's interval **straddles zero**: the cascade buys nothing. The
available gain's interval **excludes zero** by a wide margin. In relative terms
the oracle cuts error by **26.3% [23.6%, 29.2%]** against the 8B alone, at
**0.50x** its compute.

This is the cleanest number in the study: the headroom is real and precisely
estimated, and the best available escalation signal captures none of it.

### 8.3 What this makes the thesis about

Placed beside the pre-registered results, every route to lower residual risk is
blocked by the same thing:

| Route | Bound |
|---|---|
| Logit margin | Identical ordering to the probability (S2) |
| Perturbation instability | Unmeasurable at realistic prompt length (S4) |
| Cross-model disagreement | Two-valued; wins only in a 10-20% band |
| Cascaded escalation | Captures 0 of 0.0276 available headroom |
| Any panel routing | 2.85% unanimous-failure floor |

None of these is a compute limit. Compute is cheap and was spent freely here.
Each is a failure to **identify which decisions are wrong**.

> Oversight allocation for safety guards is bounded by error identification, not
> by compute. We price five routes to spending a fixed budget, show each is
> bounded, and quantify the gap: a perfect escalation signal would cut error 26%
> at half the compute, and the best available signal captures none of it.

A bound plus a quantified gap. Not "we found a better method", and it must not
be written as one.

---

## 9. Where the panel is blind (exploratory)

Post-hoc characterisation of the 256 unanimous failures from §6.3. This is the
only section that describes a *set of inputs* rather than a bound on a method.

### 9.1 Direction and location

| | count |
|---|---:|
| benign flagged harmful (false positive) | 167 |
| **harmful passed as safe (false negative)** | **89** |

| dataset | n | unanimously wrong | rate | share of 256 |
|---|---:|---:|---:|---:|
| wildguardtest | 1,699 | 87 | **5.12%** | 34% |
| openai_moderation | 1,665 | 82 | **4.92%** | 32% |
| xstest | 450 | 11 | 2.44% | 4% |
| toxicchat | 4,972 | 76 | 1.53% | 30% |
| **harmbench** | 200 | **0** | **0.00%** | 0% |

**harmbench produces no unanimous failures at all.** On the short, curated,
all-harmful benchmark the panel never fails together. Every unanimous failure
comes from realistic or long-form traffic.

### 9.2 The guards are *confidently* wrong

Median distance from the decision threshold, |p - 0.5| (0.5 is maximal
confidence):

| guard | all prompts | unanimously wrong | unanimously right |
|---|---:|---:|---:|
| llama-guard-3-1b | 0.482 | 0.459 | 0.493 |
| llama-guard-3-8b | 0.497 | 0.465 | 0.499 |
| qwen3guard-gen-8b | 0.496 | 0.480 | 0.497 |

When the panel is unanimously wrong it is **confidently** wrong: 0.459-0.480
against 0.493-0.499 when right, both near the 0.5 ceiling.

> No confidence-based routing can reach these cases, because confidence does
> not separate them. This is why the 2.85% floor is a floor and not a budget
> problem.

### 9.3 Length is the axis, and it holds within datasets

Median prompt length, unanimous failures against the rest of the same dataset:

| dataset | unanimously wrong | rest | ratio |
|---|---:|---:|---:|
| openai_moderation | 967.5 | 363.0 | 2.7x |
| wildguardtest | 548.0 | 235.5 | 2.3x |
| toxicchat | 83.0 | 59.0 | 1.4x |
| xstest | 45.0 | 42.0 | 1.1x |

Not a composition artifact -- the effect is present inside each long-form
dataset. Pooled by length quintile:

| quintile | median chars | unanimous-failure rate |
|---|---:|---:|
| Q1 shortest | 30 | 0.65% |
| Q2 | 53 | 2.17% |
| Q3 | 93 | 3.07% |
| Q4 | 259 | 2.28% |
| **Q5 longest** | **869** | **6.12%** |

Roughly **9x** from shortest to longest quintile.

The 89 safety-critical false negatives are concentrated in wildguardtest (70 of
89, 79%), median length 182 characters against 93 overall.

### 9.4 One axis explains two independent findings

Length also governs the perturbation result (§7.2): effective variants fell
5.0 -> 2.0 as mean prompt length rose 44 -> 645 characters, rank correlation
-1.0.

So the same variable drives both:

| long inputs | consequence |
|---|---|
| cheap perturbation collapses | the behavioural signal becomes unmeasurable |
| the panel fails together | the agreement signal has nothing to flag |

> **Guard oversight degrades with input length from both directions at once:
> the signals that would route a case for review stop working exactly where the
> panel stops catching it.** Short curated benchmarks (harmbench, xstest) are
> blind to this, which is why it has not been reported.

That is a property of the *inputs*, not of any method, and it is the finding
most likely to survive a change of models.

---

## 10. Limitations to state in the write-up

### 10.1 Panel compute was mispriced, and is now fixed

`compute_per_decision` for ensemble signals was `deciding guard's latency x
member count`, which prices three identical forward passes differently
depending on which member is nominated to decide. The 1B pairing was
undercharged at 0.306 and the Qwen pairing overcharged at 1.117, for the same
three passes.

`evaluation/allocation.ensemble_compute_cost` now charges the **sum of member
costs**: 0.793460 for every three-guard policy. `verify_selective.py` stays at
13/13.

**A1 and A2 verdicts are unchanged** (`CHEAPER_GUARD_WINS`, `BUDGET_DEPENDENT`,
2 distinct winners) -- A2 ranks on residual, which pricing does not touch. The
compute column of the A2 table and the frontier figure do change. Both
versions are kept: `tables_repriced/` is primary, `tables_no_stability/` is the
pre-fix comparison.

### 10.2 Remaining limitations
2. **A1's "default" is an artifact of measured latency**, not of what teams
   deploy. On different hardware Llama-Guard-3-8B could become the default and
   A1 would invert. State the hardware.
3. **S3's win is on one guard of three**, and the point estimate is in the
   wrong direction for the other two.
4. **Section 6 is exploratory.** The correlated-failure floor was derived
   after seeing the results. The flag-rate mechanism in §6.2 was specified,
   tested on 8 panels, and rejected; it is reported, not quietly dropped.
5. **S4 and S5 are untested, not negative.** S5 has no aligned encoder
   (Amendment 1); S4 has no measurable variant family on realistic prompts.
6. **The cascade analysis (§8) is post-hoc** and shares the single-environment
   cost model. The oracle is an upper bound on *this* guard pair only.
7. **Single hardware configuration.** Compute prices are comparable only within
   this L40S environment, by design.

---

## 11. Suggested paper framing

**Title direction:** *What a fixed oversight budget actually buys: pricing
guard size, cross-model checking and human review on one frontier.*

**Contributions, in descending strength:**

1. A two-dimensional pricing of oversight policies (compute x reviewer hours,
   no exchange rate) over 26,958 single-environment predictions -- absent from
   the prior work in `related_work.md`.
2. The correlated-failure floor (§6.3) -- see below; this is now the
   strongest mechanistic finding, the flag-rate rule having been falsified.
3. A tested-and-rejected mechanism (§6.2): the natural "match flag rate to
   review capacity" rule was specified, tested on 8 panels, and does not hold.
   Reported as a negative rather than omitted.
4. The AUROC/residual dissociation (§3): the better-ranking guard is the worse
   deployment.
5. Two mechanistic negatives (§7) in the 1-20% review regime that Safety-Flag's
   balanced protocol cannot address.

**The rule from `CLAUDE.md` still governs:** never write a claim
`claims_to_evidence.csv` has not marked `LICENSED` -- and, as A1 demonstrates,
check that the evidence behind a licensed claim says what its label says.
