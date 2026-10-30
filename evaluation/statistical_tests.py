from typing import List, Sequence


def holm_correction(p_values: Sequence[float], alpha: float = 0.05) -> List[dict]:
    pairs = sorted(enumerate(p_values), key=lambda kv: kv[1])
    n = len(pairs)
    out = [None] * n
    running = 0.0
    for rank, (idx, p) in enumerate(pairs):
        adjusted = min(1.0, (n - rank) * p)
        running = max(running, adjusted)
        out[idx] = {
            "p_raw": float(p),
            "p_holm": float(running),
            "significant": bool(running < alpha),
        }
    return out
