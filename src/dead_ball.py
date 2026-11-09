"""Dead-ball detector that works while the ball is visible.

A model trained on missing-ball frames learns "ball absent = stoppage" (79-94 % of D frames of the Metrica matches have no ball).
These features describe a stoppage by what is seen: ball still for more than 1.5 s, slow players, ball near a line or corner.
They are added to the base features (option `stop` of `model.EXTRA`). Thresholds were set on the Metrica matches only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import common

WIN = 38  # frames : fenêtre centrée de 1,5 s
MIN_OBS = 0.7  # part minimale de frames observées dans la fenêtre
STILL_M = 1.5  # m : étendue maximale du ballon « immobile » sur la fenêtre
MIN_STILL_S = 1.5  # s : durée d'immobilité qui qualifie un arrêt
SLOW = 1.2  # m/s : vitesse moyenne des joueurs sur 1 s en dessous de laquelle ils sont « lents »
LINE_M, CORNER_M = 1.5, 3.0


def _runs(mask: np.ndarray) -> np.ndarray:
    """Longueur (frames) de la suite de True qui contient chaque frame, 0 ailleurs."""
    out = np.zeros(len(mask), np.float32)
    e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1), strict=True):
        out[a:b] = b - a
    return out


def ball_range(bx: np.ndarray, by: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Étendue (m) du ballon observé sur la fenêtre centrée ; NaN si moins de MIN_OBS de la fenêtre est vu."""
    x = pd.Series(np.where(observed, bx, np.nan))
    y = pd.Series(np.where(observed, by, np.nan))
    need = int(MIN_OBS * WIN)
    kw = {"window": WIN, "center": True, "min_periods": need}
    rx = x.rolling(**kw).max() - x.rolling(**kw).min()
    ry = y.rolling(**kw).max() - y.rolling(**kw).min()
    return np.hypot(rx, ry).to_numpy(np.float32)


def stop_features(f: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Colonnes `stop_*` pour les frames de `f` (sortie de features.build_features) ; `df` = tracking avec
    colonnes de ball.clean_ball."""
    observed = (df["ball_source"].to_numpy() == 0) & np.isfinite(df["ball_cx"].to_numpy())
    rng = ball_range(df["ball_cx"].to_numpy(), df["ball_cy"].to_numpy(), observed)
    still = np.nan_to_num(rng, nan=np.inf) < STILL_M
    run_s = (_runs(still) + WIN - 1) / common.FPS * still  # la suite de fenêtres immobiles déborde de WIN-1
    slow = np.nan_to_num(f["all_mean_speed_m25"].to_numpy(), nan=99.0) < SLOW
    near_line = np.nan_to_num(np.minimum(f["ball_edge_touch"], f["ball_edge_goal"]), nan=99.0) <= LINE_M
    near_corner = np.nan_to_num(np.hypot(f["ball_edge_goal"], f["ball_edge_touch"]), nan=99.0) < CORNER_M
    core = (run_s > MIN_STILL_S) & slow
    g = pd.DataFrame({"stop_range": rng, "stop_still_s": run_s.astype(np.float32),
                      "stop_core": core.astype(np.float32),
                      "stop_line": (core & (near_line | near_corner)).astype(np.float32)}, index=f.index)
    for c in ("stop_core", "stop_line"):
        g[c + "_m125"] = g[c].rolling(125, center=True, min_periods=1).mean().astype(np.float32)
    return pd.concat([f, g], axis=1)

