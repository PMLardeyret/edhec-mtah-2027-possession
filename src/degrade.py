"""Degradation of open tracking data so that it resembles automatic tracking.

What it does (fixed seed): anonymisation (unflagged goalkeeper, shuffled columns, speeds removed), fragmentation (each player is cut
into several tracks), absences (fraction of frames without a track), extra ball gaps of log-normal length in play frames.
Constants below are defaults; variant s overrides them from configs/ (see variant_cfg.py and the README).
"""

from __future__ import annotations

import argparse
import os
from itertools import pairwise

import numpy as np
import pandas as pd

import common
import load_tracking as lt

PRESENCE_LOSS = 0.07  # 22 -> 20.5 active tracks
MEAN_DROPOUT = 150  # frames; default, variant s4 sets 450 (chosen on IDSSE, configs/v23c_s4.json)
CUTS_PER_PLAYER = 1.1  # ~ 46 tracks for 22 players
MIN_PIECE = 125  # frames; s4 keeps 125 (chosen on IDSSE)
GAP_MU, GAP_SIGMA = np.log(268), 0.9  # default; variant s4 overrides from configs/
V21 = False  # stop-aware ball degradation: not part of this release
RANDOM_PRESENCE: tuple[float, float] | None = None  # set by variant_cfg (strict variant: drawn per seed)
RANDOM_CUTS: tuple[float, float] = (0.5, 2.0)
EXTRA_BALL_LOSS = 0.04  # a-priori choice, not tuned (read from the config by variant_cfg)


def _absent_mask(n: int, rng: np.random.Generator, loss: float | None = None) -> np.ndarray:
    """Épisodes d'absence de durée géométrique couvrant ~PRESENCE_LOSS des frames."""
    m = np.zeros(n, bool)
    n_ep = rng.poisson((PRESENCE_LOSS if loss is None else loss) * n / MEAN_DROPOUT)
    for s in rng.integers(0, n, n_ep):
        m[s:s + 1 + rng.geometric(1 / MEAN_DROPOUT)] = True
    return m


def _cuts(valid: np.ndarray, rng: np.random.Generator, per_player: float | None = None) -> list[tuple[int, int]]:
    """Découpe les frames valides d'un joueur en morceaux (début, fin) avec trou entre morceaux."""
    idx = np.flatnonzero(valid)
    if not len(idx):
        return []
    n_cut = rng.poisson(CUTS_PER_PLAYER if per_player is None else per_player)
    cuts = np.sort(rng.choice(idx, size=min(n_cut, len(idx)), replace=False)) if n_cut else []
    bounds = [idx[0], *cuts, idx[-1] + 1]
    pieces = []
    for i, (a, b) in enumerate(pairwise(bounds)):
        a = a + (rng.integers(1, 100) if i else 0)  # trou de rupture de piste
        if b - a >= MIN_PIECE:
            pieces.append((int(a), int(b)))
    return pieces


def drop_ball(bx: np.ndarray, by: np.ndarray, frac: float, rng: np.random.Generator,
              graded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Retire du ballon jusqu'à `frac` des frames notées supplémentaires, en trous log-normaux."""
    bx, by = bx.copy(), by.copy()
    n = len(bx)
    target = frac * graded.sum()
    lost = 0.0
    while lost < target:
        s = int(rng.integers(0, n))
        ln = max(2, int(rng.lognormal(GAP_MU, GAP_SIGMA)))
        seg = slice(s, min(s + ln, n))
        lost += np.isfinite(bx[seg]).sum()
        bx[seg] = np.nan
        by[seg] = np.nan
    return bx, by


def degrade(df: pd.DataFrame, seed: int, extra_ball_loss: float | None = None, state: np.ndarray | None = None,
            v21: bool | None = None) -> pd.DataFrame:
    """Degrade a tracking table (anonymise, fragment tracks, drop presence, drop ball). `state` and `v21` are kept for call compatibility."""
    if v21 or V21:
        raise NotImplementedError("stop-aware ball degradation is not part of this release")
    extra_ball_loss = EXTRA_BALL_LOSS if extra_ball_loss is None else extra_ball_loss
    rng = np.random.default_rng(seed)
    loss = cuts = None
    if RANDOM_PRESENCE:  # strict variant: presence and cuts drawn per seed from a-priori intervals
        loss, cuts = rng.uniform(*RANDOM_PRESENCE), rng.uniform(*RANDOM_CUTS)
    n = len(df)
    x, y, team = lt.track_array(df, "x"), lt.track_array(df, "y"), lt.track_array(df, "team")
    cols: list[tuple[np.ndarray, np.ndarray, int]] = []
    for k in range(x.shape[1]):
        valid = np.isfinite(x[:, k]) & ~_absent_mask(n, rng, loss)
        t = int(np.nanmax(team[:, k])) if np.isfinite(team[:, k]).any() else -1
        for a, b in _cuts(valid, rng, cuts):
            px = np.full(n, np.nan, np.float32)
            py = np.full(n, np.nan, np.float32)
            keep = valid[a:b]
            px[a:b][keep] = x[a:b, k][keep]
            py[a:b][keep] = y[a:b, k][keep]
            cols.append((px, py, t))
    order = rng.permutation(len(cols))
    out = {c: df[c].to_numpy() for c in ("frame", "half", "graded")}
    bx, by = drop_ball(df["ball_x"].to_numpy(), df["ball_y"].to_numpy(), extra_ball_loss, rng, df["graded"].to_numpy())
    out["ball_x"], out["ball_y"] = bx, by
    out["ball_s"] = np.full(n, np.nan, np.float32)
    for j, i in enumerate(order, start=1):
        px, py, t = cols[i]
        out[f"t{j}_x"], out[f"t{j}_y"] = px, py
        out[f"t{j}_s"] = np.full(n, np.nan, np.float32)
        out[f"t{j}_team"] = np.full(n, t, np.int8)
        out[f"t{j}_gk"] = np.zeros(n, bool)
    return pd.DataFrame(out)


def summary(df: pd.DataFrame) -> dict:
    g = df["graded"].to_numpy()
    present = np.isfinite(lt.track_array(df, "x"))
    return {"pistes": lt.n_tracks(df), "actives_moy": float(present[g].sum(1).mean()),
            "ballon_visible_%": 100 * float(df.loc[df["graded"], "ball_x"].notna().mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra-ball-loss", type=float, default=0.04)
    args = ap.parse_args()
    out = common.ROOT / "work" / "sample" / "degraded"
    out.mkdir(parents=True, exist_ok=True)
    for m in ("Sample_Game_1", "Sample_Game_2", "Sample_Game_3"):
        df = pd.read_parquet(common.ROOT / "work" / "sample" / f"{m}.parquet")
        d = degrade(df, args.seed, args.extra_ball_loss)
        d.to_parquet(out / f"{m}_s{args.seed}.parquet")
        print(m, "avant", summary(df), "après", summary(d))


if __name__ == "__main__":
    main()

if os.environ.get("V23C"):
    import variant_cfg
    variant_cfg.apply_degrade(__import__("sys").modules[__name__])
