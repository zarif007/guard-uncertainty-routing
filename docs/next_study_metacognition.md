# Next study: metacognitive monitoring for safety guards

A research direction, not a protocol. Nothing here is pre-registered. The point
is to find the version of this idea that has real computational content rather
than a borrowed vocabulary.

Citations are from memory and **every one needs verifying** before use.

---

## 1. The theoretical claim worth building on

Human metacognition is not a readout of the decision process. The Type-1
judgment ("is this safe?") and the Type-2 judgment ("am I right about that?")
draw on **partly different evidence**. Post-decisional accumulation models
(Pleskac & Busemeyer, two-stage dynamic signal detection) have evidence
continuing to arrive after the decision commits, so confidence can reflect
information the decision never used. Feeling-of-knowing judgments go further:
they are computed from *cue familiarity and fluency* without retrieving the
answer at all.

A generative guard's confidence has none of this. `p_unsafe` is the softmax over
the same forward pass that produced the verdict. It is **decision-congruent by
construction** — it can only be a function of the evidence the decision already
used, so it inherits the decision's blind spots exactly.

> **Hypothesis.** The 2.85% of cases where our three-guard panel is confidently
> and unanimously wrong are not a capacity failure. They are the predicted
> consequence of a monitoring signal that is architecturally incapable of
> disagreeing with the thing it monitors.

That is testable, it explains a measured result, and it motivates a specific
architecture rather than a general appeal to "metacognition".

### What follows for design

If the diagnosis is decision-congruence, the prescription is **evidence
independence**: a monitor earns its place only to the extent its inputs are not
a function of the guard's forward pass. This gives a principled ordering of
monitor designs, from worst to best:

| monitor input | independence | our evidence |
|---|---|---|
| guard's own `p_unsafe` | none | captures 0 of the 0.0276 gap |
| guard's logits / entropy | none | S2: margin is identical to probability |
| a *second* guard's verdict | partial | wins only at 10-20% review |
| input-side cues the guard did not condition on | high | length predicts failure 9x across quintiles |
| post-decisional second pass | high | untested |

The current study walked down the first three rows without knowing it. The
next one should start at row four.

---

## 2. Import the measurement framework: meta-d' and M-ratio

This is the most immediately valuable transfer, and it is quantitative rather
than metaphorical.

Metacognition research separates two things that guard evaluation conflates:

- **d'** -- Type-1 sensitivity. How well the system discriminates safe from
  unsafe. This is what AUROC measures.
- **meta-d'** -- Type-2 sensitivity. How well *confidence* discriminates correct
  from incorrect judgments, expressed in the same units as d' (Maniscalco & Lau).
- **M-ratio = meta-d'/d'** -- metacognitive *efficiency*. How much of the
  information theoretically available to the confidence judgment is actually
  used. M-ratio = 1 is an ideal metacognitive observer; below 1 is a deficit.

Nobody in guard evaluation reports this. Safety-Flag reports AURC, which
confounds Type-1 and Type-2 performance in one number. **M-ratio separates
them**, and the separation is exactly the dissociation we already stumbled into:
Qwen3Guard has the best d' in our panel (AUROC 0.9615) and the worst deployed
error rate.

**First experiment, costs nothing:** compute d', meta-d' and M-ratio for all
three guards on all five datasets from the existing predictions. Predicted
result, given §5.5: M-ratio falls with input length. If it does, you have
converted a correlational finding into a measurement on an established scale.

This alone is a short paper: *guard models have a metacognitive efficiency
deficit, it is measurable on a standard scale, and it worsens with input length.*

---

## 3. Three monitors, each mapped to a system with real computational content

Not decoration -- each maps to a mechanism with a specific signature our data
can test.

### 3.1 Conflict monitor (ACC-like)

Conflict monitoring theory (Botvinick, Braver, Barch, Carter & Cohen) has the
anterior cingulate detect *response conflict* -- simultaneous activation of
competing responses -- and signal the need for more control. Critically, the
conflict detector is **separate from the decider**.

Our `conf_model_agree` already is this, crudely. The neuro version suggests a
refinement we have not tried: conflict should be **graded**, not a vote count.
Competing-response strength, not the identity of the winner. That means
disagreement measured in *logit space* across guards, which turns a two-valued
signal (tie fraction 1.0000) into a continuous one -- directly addressing the
binding limitation of our best alternative signal.

### 3.2 Fluency / feeling-of-knowing monitor

FOK judgments are computed from cue familiarity without content retrieval. The
computational analogue: features of the *input* that predict difficulty without
reference to the verdict. Input perplexity under a reference model,
out-of-distribution scores, length, syntactic depth, code-switching, topic
novelty.

This is the row-four monitor, and our length result is the first evidence it has
signal: 0.65% -> 6.12% unanimous-failure rate across length quintiles, holding
within datasets.

### 3.3 Error monitor (ERN/Pe-like)

The error-related negativity is generated ~100ms after an erroneous response,
**before any feedback** -- the system detects its own error without ground truth.
The error positivity (Pe) tracks *conscious* error awareness, and the two
dissociate: there are errors the brain registers but never becomes aware of.

That dissociation is the sharpest available description of our 256. The
computational analogue is a **post-decisional second pass**: re-encode the
prompt *together with the verdict just produced* and ask a separate head whether
that pairing is coherent. Evidence independence comes from the second pass
seeing something the first never did -- its own output.

---

## 4. The opt-out paradigm gives the evaluation its shape

Animal metacognition research (Hampton's rhesus monkeys; Smith's work on
dolphins and monkeys) established the **uncertain response**: a third option
besides the two task responses, paying less than a correct answer and more than
an error. An animal that uses the opt-out selectively on hard trials is credited
with metacognitive access.

That is our review budget, exactly. It also supplies a paradigm detail worth
copying: the opt-out rate is **not fixed by the experimenter** — the subject
chooses when to use it, and selectivity is what is measured. Our study fixed the
budget and rank-ordered within it. A metacognitive system should instead decide
*how many* cases to escalate, and be scored on whether it spends a variable
budget well.

**Nelson & Narens' monitoring/control framework** is the right organizing
skeleton: metacognition is monitoring (assessing the object-level process) plus
control (acting on that assessment). Our monitors are monitoring. The routing
policy is control. Most ML work on deferral collapses the two; keeping them
separate is both more faithful and more useful, because it lets you ask which
half is failing.

---

## 5. The experimental programme

Every stage below runs on existing predictions except stage 4.

**Stage 1 -- quantify the deficit.** d', meta-d', M-ratio per guard per dataset.
Test whether M-ratio declines with input length. *No GPU. This is the paper's
foundation and may be publishable alone.*

**Stage 2 -- establish decision-congruence.** Show the guard's confidence errors
are *correlated with* its decision errors in a way an independent monitor's need
not be. Concretely: confidence fails to separate the 256 (|p-0.5| 0.465 vs
0.499), while input length does separate them (9x across quintiles). This is the
empirical case for architectural separation.

**Stage 3 -- build the three monitors and ablate.** Each monitor alone, then
combined, scored on **how much of the 0.0276 oracle gap it closes**, leave-one-
dataset-out. The ablation is the result: which evidence source carries the
information, and does independence predict contribution?

> **Leave-one-dataset-out is not optional.** HarmBench yields zero unanimous
> failures and WildGuardTest yields 5.12%. A monitor evaluated in-distribution
> will look excellent and transfer to nothing -- and our own §5.5 argues exactly
> that curated benchmarks hide this failure mode. Evaluating the fix
> in-distribution would contradict the paper that motivates it.

**Stage 4 -- the post-decisional pass.** The only stage needing GPU. Re-encode
(prompt, verdict) and train a coherence head. Expensive, so it should be
justified by stages 1-3 before anyone runs it.

**Stage 5 -- variable-budget control.** Let the system choose its own escalation
rate and score selectivity, per the opt-out paradigm. Compare against our fixed-
budget frontier.

### Be realistic about scale

943-1,614 errors per guard. That supports logistic regression or gradient
boosting over a handful of features. It does not support training anything deep,
and a reviewer will say so. The ceiling on stage 3 is a well-designed shallow
model, which is fine -- the contribution is the framework and the ablation, not
the classifier.

---

## 6. What makes this novel, stated defensively

Learned deferral is an established field. Position against it from the first
page, not in revision:

- Cortes, DeSalvo & Mohri (2016), *Learning with rejection*
- Madras, Pitassi & Zemel (2018), *Predict Responsibly*
- Mozannar & Sontag (2020), *Consistent Estimators for Learning to Defer*
- Geifman & El-Yaniv (2019), *SelectiveNet*
- Corbière et al. (2019), *ConfidNet*; Jiang et al. (2018), *Trust Score*
- FrugalGPT and learned LLM cascades

What they do not have, and what the claim should rest on:

1. **A measurement framework from metacognition research** (meta-d', M-ratio)
   applied to safety guards, separating Type-1 from Type-2 performance where the
   field currently reports one conflated number.
2. **A principled prediction about which monitors work** -- evidence
   independence -- rather than a feature search.
3. **A measured ceiling.** The oracle gap (0.0276, CI [0.0243, 0.0309]) means
   this is scored against a known maximum, not a baseline.
4. **An input-side failure axis** (length) that the field's benchmarks hold
   fixed.

The neuroscience earns its place only through (1) and (2). Keep the ERN and
opt-out material as motivation for specific design choices, and never as an
argument that the system is brain-like.

---

## 7. Reading list -- verify every citation

**Measurement (read first, it is the transferable part)**
- Maniscalco & Lau (2012), *A signal detection theoretic approach for estimating
  metacognitive sensitivity from confidence ratings*, Consciousness & Cognition
- Fleming & Lau (2014), *How to measure metacognition*, Front. Hum. Neurosci.
- Fleming, Weil, Nagy, Dolan & Rees (2010), *Relating introspective accuracy to
  individual differences in brain structure*, Science

**Mechanism**
- Botvinick, Braver, Barch, Carter & Cohen (2001), *Conflict monitoring and
  cognitive control*, Psychological Review
- Yeung & Summerfield (2012), *Metacognition in human decision-making:
  confidence and error monitoring*, Phil. Trans. R. Soc. B
- Pleskac & Busemeyer (2010), *Two-stage dynamic signal detection*,
  Psychological Review

**Framework and paradigm**
- Nelson & Narens (1990), *Metamemory: a theoretical framework* -- the
  monitoring/control distinction
- Hampton (2001), PNAS -- the opt-out paradigm
- Koechlin & Summerfield (2007), *An information theoretical approach to
  prefrontal executive function*, TICS -- hierarchical control, if the cascade
  framing is pursued

---

## 8. The first thing to do

Stage 1. It needs no GPU, no new data, and no new models, and it either produces
a measurable metacognitive deficit that worsens with length -- in which case the
whole programme has a foundation -- or it does not, in which case you have saved
yourself the rest.
