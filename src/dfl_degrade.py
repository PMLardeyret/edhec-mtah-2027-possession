"""Degrade an IDSSE (DFL) match to look like the automatic tracking of the target match (tracks and ball)."""

from __future__ import annotations


import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

import common
import degrade

DFL = common.ROOT / "work" / "external" / "dfl"
SK = common.ROOT / "work" / "external" / "skillcorner"
MATCHES = tuple(f"DFL_{k}" for k in range(1, 8))
IN_PLAY_LOSS, DEAD_LOSS = 0.06, 0.83
CUTS = 0.35  # track cuts per player (overridden: drawn per seed, see variant_cfg)
PROFILE = common.ROOT / "configs" / "s_ball_profile.json"
EVAL_SEED = 100


def load(m: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """(tracking brut, labels, ball_status)."""
    d = SK if m.startswith("SK_") else DFL  
    return pd.read_parquet(d / f"{m}.parquet"), np.load(d / f"{m}_labels.npy"), np.load(d / f"{m}_status.npz")["ball_status"]


def ball_profile(df: pd.DataFrame, fc: float, sigma: float, seed: int) -> pd.DataFrame:
    """Passe-bas puis bruit blanc sur le ballon continu (par mi-temps)."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    sos = butter(2, fc, btype="low", fs=common.FPS, output="sos")
    for c in ("ball_x", "ball_y"):
        v = out[c].to_numpy(float)
        for h in np.unique(df["half"]):
            m = (df["half"].to_numpy() == h) & np.isfinite(v)
            if m.sum() > 30:
                v[m] = sosfiltfilt(sos, v[m]) if fc > 0 else v[m]
        out[c] = v + rng.normal(0.0, sigma, len(v))
    return out


def drop_by_state(df: pd.DataFrame, status: np.ndarray, seed: int, v21: bool | None = None) -> pd.DataFrame:
    """Remove the ball in log-normal gaps: IN_PLAY_LOSS of in-play frames and DEAD_LOSS of dead-ball frames."""
    if v21 or degrade.V21:
        raise NotImplementedError("stop-aware ball degradation is not part of this release")
    rng = np.random.default_rng(seed)
    out = df.copy()
    bx, by = out["ball_x"].to_numpy(float).copy(), out["ball_y"].to_numpy(float).copy()
    n = len(bx)
    for mask, frac in ((status != 0, IN_PLAY_LOSS), (status == 0, DEAD_LOSS)):
        idx = np.flatnonzero(mask)
        # target capped at 90 % of the frames where the ball exists (avoids an endless loop on sparse ball tracks)
        target, lost = min(frac * len(idx), 0.9 * np.isfinite(bx[idx]).sum()), 0
        while lost < target and len(idx):
            s = int(rng.choice(idx))
            ln = max(2, int(rng.lognormal(degrade.GAP_MU, degrade.GAP_SIGMA)))
            seg = np.arange(s, min(s + ln, n))
            seg = seg[mask[seg]]
            lost += int(np.isfinite(bx[seg]).sum())
            bx[seg], by[seg] = np.nan, np.nan
    out["ball_x"], out["ball_y"] = bx, by
    return out


def degrade_match(m: str, seed: int, fc: float, sigma: float, v21: bool | None = None) -> pd.DataFrame:
    df, _lab, status = load(m)
    d = ball_profile(df, fc, sigma, seed)
    d = drop_by_state(d, status, seed + 1, v21)
    old, degrade.CUTS_PER_PLAYER = degrade.CUTS_PER_PLAYER, CUTS
    try:
        t = degrade.degrade(d, seed, extra_ball_loss=0.0)
    finally:
        degrade.CUTS_PER_PLAYER = old
    t["ball_x"], t["ball_y"] = d["ball_x"].to_numpy(), d["ball_y"].to_numpy()
    return t


import os

if os.environ.get("V23C"):
    import sys

    import variant_cfg
    variant_cfg.apply_dfl(sys.modules[__name__])
