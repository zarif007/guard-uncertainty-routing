from typing import Dict, List, Sequence, Tuple

import numpy as np


def to_labels(ground_truth: Sequence[str]) -> np.ndarray:
    return np.asarray([1 if g == "unsafe" else 0 for g in ground_truth], dtype=int)


def auroc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j < len(s) - 1 and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return float((ranks[y == 1].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))
