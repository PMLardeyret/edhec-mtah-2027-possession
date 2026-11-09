"""Stoppage features that work while the ball is visible (option V21=...feat..., see model.build).

How long the ball has been visible and still (backwards and forwards), time since / until the last / next sharp ball acceleration
(restart signature), mean speed of the 22 players over 2 s and its ratio to the 5 s speed, time the ball spent off the pitch, median
player speed over +-1 s, and "ball more than 1 m outside the lines" extended over a tracking gap of up to 10 s. Not recreated here:
ball still over 1.5 s (`stop_*`), ball distance to the lines (`ball_edge_*`), player speed over 1 s and 5 s.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import common
import dead_ball
import features
import load_tracking as lt

ACC_SHARP = 20.0  # m/s² : entre le p90 (17) et le p99 (32) de l'accélération du ballon dégradé
CAP_S = 60.0  # secondes, plafond des durées
OUT_M = 1.0  # m au-delà des lignes : ballon franchement sorti
HOLD_S = 10.0  # le statut « sorti » est prolongé sur un trou de suivi jusqu'à 10 s
MED_WIN = 51  # ± 1 s à 25 Hz


def _dist(mask: np.ndarray, forward: bool) -> np.ndarray:
    """Images jusqu'à la prochaine (forward) ou depuis la dernière image où `mask` est vrai ; plafonné."""
    n = len(mask)
    idx = np.arange(n, dtype=float)
    pos = pd.Series(np.where(mask, idx, np.nan))
    pos = pos.bfill() if forward else pos.ffill()
    d = (pos.to_numpy() - idx) if forward else (idx - pos.to_numpy())
    return np.nan_to_num(d, nan=CAP_S * common.FPS).clip(0, CAP_S * common.FPS)


def _run_len(mask: np.ndarray, forward: bool) -> np.ndarray:
    """Longueur (images) de la suite de True qui finit (ou commence, si forward) à chaque image ; 0 hors masque."""
    out = np.zeros(len(mask), np.float32)
    e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1), strict=True):
        out[a:b] = np.arange(b - a, 0, -1) if forward else np.arange(1, b - a + 1)
    return out


def add(f: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Colonnes `v21_*` pour les images de `f` (sortie de features.build_features) ; `df` = tracking avec colonnes de ball.clean_ball."""
    obs = (df["ball_source"].to_numpy() == 0) & np.isfinite(df["ball_cx"].to_numpy())
    rng = dead_ball.ball_range(df["ball_cx"].to_numpy(), df["ball_cy"].to_numpy(), obs)
    still = np.nan_to_num(rng, nan=np.inf) < dead_ball.STILL_M
    sharp = np.nan_to_num(np.abs(f["ball_acc"].to_numpy()), nan=0.0) > ACC_SHARP
    out_play = (f["ball_in_play_area"].to_numpy() == 0) & (f["ball_present"].to_numpy() > 0)
    x, y = lt.track_array(df, "x"), lt.track_array(df, "y")
    _, _, psp = features.player_speed(x, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med_sp = pd.Series(np.nanmedian(np.where(np.isfinite(x) & np.isfinite(y), psp, np.nan), axis=1)).rolling(MED_WIN, center=True, min_periods=1).median()
    bx, by = df["ball_cx"].to_numpy(float), df["ball_cy"].to_numpy(float)
    far = (bx < -OUT_M) | (bx > features.L + OUT_M) | (by < -OUT_M) | (by > features.W + OUT_M)
    held = pd.Series(np.where(obs, far.astype(float), np.nan)).ffill(limit=int(HOLD_S * common.FPS)).fillna(0.0)
    sp = f["all_mean_speed"]
    m50 = sp.rolling(50, center=True, min_periods=1).mean()
    g = pd.DataFrame({
        "v21_still_back_s": _run_len(still, False) / common.FPS,
        "v21_still_fwd_s": _run_len(still, True) / common.FPS,
        "v21_since_acc_s": _dist(sharp, False) / common.FPS,
        "v21_until_acc_s": _dist(sharp, True) / common.FPS,
        "v21_players_speed_2s": m50.to_numpy(),
        "v21_players_speed_ratio": (m50 / (f["all_mean_speed_m125"] + 0.1)).to_numpy(),
        "v21_out_back_s": _run_len(out_play, False) / common.FPS,
        "v21_out_fwd_s": _run_len(out_play, True) / common.FPS,
        "v21_players_med_speed_1s": med_sp.to_numpy(),
        "v21_out1m_held": held.to_numpy(),
    }, index=f.index).astype(np.float32)
    return pd.concat([f, g], axis=1)
