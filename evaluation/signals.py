"""
The five uncertainty signals we compare.

Every signal is expressed as CONFIDENCE, not uncertainty: higher means the
guard's decision is more trustworthy.  `evaluation.selective` keeps the most
confident items and hands the rest to a reviewer, so the direction has to be
consistent or the curve is computed backwards.

  1 native        the guard's own probability, distance from the 0.5 cut
  2 margin        the raw logit gap, before the sigmoid squashes it
  3 stability     does the verdict survive rewording the message?
  4 model_agree   do several different guards agree?
  5 precision_agree  does the same guard agree with itself across quantizations?

Signals 1 and 2 are free -- they are already in every prediction file.
Signal 3 costs k extra forward passes per prompt.  Signals 4 and 5 cost one
extra run per additional model, which the sweep produces anyway.

Why signal 2 is not redundant with signal 1
-------------------------------------------
p_unsafe is sigmoid(margin).  Past a margin of about 37 the sigmoid returns
exactly 1.0 in float64, so every prompt beyond that point becomes a tie.  The
ranking information survives in the margin and is destroyed in the
probability.  Comparing 1 against 2 measures that loss and costs nothing.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

NATIVE = "conf_native"
MARGIN = "conf_margin"
STABILITY = "conf_stability"
MODEL_AGREE = "conf_model_agree"
PRECISION_AGREE = "conf_precision_agree"

FREE_SIGNALS = [NATIVE, MARGIN]
ALL_SIGNALS = [NATIVE, MARGIN, STABILITY, MODEL_AGREE, PRECISION_AGREE]


def native_confidence(df: pd.DataFrame, temperature: float = 1.0) -> pd.Series:
    """
    Distance of the guard's probability from its decision threshold.

    Temperature is applied on the margin, which is where it belongs:
    sigmoid(m/T) >= 0.5 exactly when m >= 0, so temperature never changes a
    label or the *ordering* of this signal -- only the calibration numbers.
    Safety-Flag observed the same thing empirically.  It is exposed anyway so
    the baseline can be built with their recipe rather than an approximation
    of it.
    """
    if "margin" in df.columns and temperature and temperature != 1.0:
        scaled = pd.to_numeric(df["margin"], errors="coerce") / temperature
        p = 1.0 / (1.0 + np.exp(-np.clip(scaled, -500, 500)))
    else:
        p = pd.to_numeric(df["p_unsafe"], errors="coerce")
    return (p - 0.5).abs()


def margin_confidence(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["margin"], errors="coerce").abs()


def stability_confidence(perturbed: pd.DataFrame,
                         id_column: str = "prompt_id") -> pd.Series:
    """
    Agreement rate across rewordings of the same message.

    Input is the output of scripts/perturb.py: one row per (original prompt,
    variant), carrying the original prompt_id.  Confidence is the fraction of
    variants that landed on the modal verdict, so 1.0 means every rewording
    agreed and 0.5 means the guard was split.

    Caveat that belongs in the methods section: a paraphrase changes the text,
    so instability mixes genuine model uncertainty with the variant actually
    meaning something different.  scripts/perturb.py records a
    semantics-preserved check per variant; drop the ones that fail it before
    calling this, or the signal measures paraphrase quality.
    """
    def modal_agreement(group):
        verdicts = group["prediction"].value_counts()
        return verdicts.max() / verdicts.sum()

    return perturbed.groupby(id_column).apply(modal_agreement, include_groups=False)


def agreement_confidence(df: pd.DataFrame, members: Sequence[str],
                         id_column: str = "prompt_id",
                         model_column: str = "model") -> pd.Series:
    """
    Fraction of `members` landing on the modal verdict for each prompt.

    Used for both signal 4 (different guards) and signal 5 (one guard at
    several precisions).  They are the same computation over a different
    member set, and they mean different things:

      signal 4  different architectures with different decision boundaries, so
                disagreement partly measures "these are different models"
                rather than "this case is hard".  Interpret with care.
      signal 5  one model perturbed by quantization.  No boundary difference
                to confound it, and no paraphrase to change the meaning, which
                makes it the cleaner of the two.
    """
    sub = df[df[model_column].isin(members)]
    if sub.empty:
        return pd.Series(dtype=float)

    def modal_agreement(group):
        verdicts = group["prediction"].value_counts()
        return verdicts.max() / verdicts.sum()

    agree = sub.groupby(id_column).apply(modal_agreement, include_groups=False)
    # A prompt only one member scored would read as unanimous; that is an
    # artefact of a missing run, not agreement.
    counts = sub.groupby(id_column)[model_column].nunique()
    return agree.where(counts >= 2)


def attach_signals(
    df: pd.DataFrame,
    perturbed: Optional[pd.DataFrame] = None,
    model_members: Optional[Sequence[str]] = None,
    precision_members: Optional[Sequence[str]] = None,
    temperature: float = 1.0,
) -> pd.DataFrame:
    """
    Add every computable signal column to a prediction frame.

    Signals whose inputs are absent are simply not added; `signal_table` skips
    missing columns, so a phase-1 run with only the free signals works without
    special casing.
    """
    out = df.copy()
    out[NATIVE] = native_confidence(out, temperature=temperature)
    out[MARGIN] = margin_confidence(out)

    if perturbed is not None and not perturbed.empty:
        out[STABILITY] = out["prompt_id"].map(stability_confidence(perturbed))
    if model_members:
        out[MODEL_AGREE] = out["prompt_id"].map(
            agreement_confidence(df, model_members))
    if precision_members:
        out[PRECISION_AGREE] = out["prompt_id"].map(
            agreement_confidence(df, precision_members))
    return out


def available_signals(df: pd.DataFrame) -> List[str]:
    return [s for s in ALL_SIGNALS if s in df.columns and df[s].notna().any()]
