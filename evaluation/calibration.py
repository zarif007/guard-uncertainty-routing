from typing import Dict, List, Sequence, Tuple

import numpy as np


GOLDEN = (5.0**0.5 - 1) / 2


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def nll_at_temperature(margins: np.ndarray, y: np.ndarray, temperature: float) -> float:
    p = sigmoid(margins / temperature)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(
    margins: Sequence[float], y_true: Sequence[int], lo: float = 0.01, hi: float = 100.0
) -> Dict[str, float]:
    m = np.asarray(margins, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    keep = np.isfinite(m)
    m, y = m[keep], y[keep]

    phi = GOLDEN
    a, b = lo, hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    for _ in range(200):
        if nll_at_temperature(m, y, c) < nll_at_temperature(m, y, d):
            b = d
        else:
            a = c
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        if abs(b - a) < 1e-6:
            break
    t = (a + b) / 2
    return {
        "temperature": float(t),
        "nll_before": nll_at_temperature(m, y, 1.0),
        "nll_after": nll_at_temperature(m, y, t),
    }
