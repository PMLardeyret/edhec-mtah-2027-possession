"""Decoding helpers: decode a tracking table with a given configuration, and solve the dead-ball bias b for a target share of D."""

from __future__ import annotations

import numpy as np

import common
import run_real


def real_decode(df, p, cfg: dict, b: float, cost_fn=None, short_poss: float | None = None,
                setpiece: np.ndarray | None = None, rules: tuple[str, ...] | None = None) -> np.ndarray:
    return run_real.labels_real(df, p, cfg | {"bias_d": b}, cost_fn=cost_fn, short_poss=short_poss, setpiece=setpiece,
                                rules=rules)


def solve_bias(df, p, cfg: dict, target: float, cost_fn=None, bounds: tuple[float, float] = (-6.0, 6.0)) -> float:
    """Dichotomie sur b : la part de D décodée croît avec b."""
    graded = df["graded"].to_numpy()
    lo, hi = bounds
    for _ in range(14 if hi - lo <= 12 else 16):
        mid = (lo + hi) / 2
        share = 100 * float((real_decode(df, p, cfg, mid, cost_fn)[graded] == common.D).mean())
        lo, hi = (mid, hi) if share < target else (lo, mid)
    return round((lo + hi) / 2, 3)
