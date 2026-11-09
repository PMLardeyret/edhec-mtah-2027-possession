"""Indices par frame (J6), invariants à l'identité des joueurs : seuls comptent les deux blocs d'équipe.

Entrée : DataFrame du chargeur + colonnes de `ball.clean_ball` (voir `ball.with_ball`).
Sortie : une ligne par frame, colonnes préfixées `h_` (équipe 0) et `a_` (équipe 1) quand elles
décrivent une équipe, de sorte qu'échanger les équipes revient à échanger les préfixes.
See the code for the list of features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import common
import load_tracking as lt

L, W = common.PITCH_SAMPLE  # même repère dans les deux régimes à 0,3 % près ; les indices sont relatifs
HALF_SPEED = 3  # vitesse par différence centrale sur +-3 frames
MAX_PLAYER_SPEED = 12.0
STILL, FAST = 0.5, 3.0  # m/s
WINDOWS = (25, 125)
PRE_KICKOFF = (110808, 113742)  # frames stated all-D in the official FAQ
TEAMS = (("h", 0), ("a", 1))
ROLL_BASE = ("d1", "d3", "mean_speed", "spread_x", "spread_y")


def player_speed(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vitesse par piste (vx, vy, norme) en m/s, NaN si la piste n'existe pas sur [t-3, t+3]."""
    k = HALF_SPEED
    vx = np.full_like(x, np.nan, dtype=np.float32)
    vy = np.full_like(y, np.nan, dtype=np.float32)
    s = common.FPS / (2 * k)
    vx[k:-k] = (x[2 * k:] - x[:-2 * k]) * s
    vy[k:-k] = (y[2 * k:] - y[:-2 * k]) * s
    sp = np.hypot(vx, vy)
    bad = sp > MAX_PLAYER_SPEED
    vx[bad], vy[bad], sp[bad] = np.nan, np.nan, np.nan
    return vx, vy, sp


def _nanmean(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Moyenne par ligne sur les colonnes `mask`, NaN s'il n'y en a aucune."""
    n = mask.sum(1)
    s = np.where(mask, a, 0.0).sum(1)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def _nanstd(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = _nanmean(a, mask)[:, None]
    return np.sqrt(_nanmean((a - m) ** 2, mask))


def team_block(t: str, mask: np.ndarray, x: np.ndarray, y: np.ndarray, vx: np.ndarray,
               vy: np.ndarray, sp: np.ndarray, bx: np.ndarray, by: np.ndarray) -> dict:
    """Indices d'une équipe ; `mask` = pistes présentes de cette équipe."""
    out = {f"{t}_n": mask.sum(1).astype(np.float32)}
    cx, cy = _nanmean(x, mask), _nanmean(y, mask)
    out[f"{t}_depth"] = np.abs(cx - L / 2)  # profondeur du bloc par rapport au milieu
    out[f"{t}_cy_off"] = np.abs(cy - W / 2)
    out[f"{t}_spread_x"], out[f"{t}_spread_y"] = _nanstd(x, mask), _nanstd(y, mask)
    out[f"{t}_cvx"], out[f"{t}_cvy"] = _nanmean(vx, mask & np.isfinite(vx)), _nanmean(vy, mask & np.isfinite(vy))
    out[f"{t}_mean_speed"] = _nanmean(sp, mask & np.isfinite(sp))
    valid = mask & np.isfinite(sp)
    nv = np.maximum(valid.sum(1), 1)
    out[f"{t}_still"] = np.where(valid.sum(1) > 0, (valid & (sp < STILL)).sum(1) / nv, np.nan)
    out[f"{t}_fast"] = np.where(valid.sum(1) > 0, (valid & (sp > FAST)).sum(1) / nv, np.nan)
    d = np.where(mask, np.hypot(x - bx[:, None], y - by[:, None]), np.inf)
    d3 = np.sort(d, axis=1)[:, :3]
    d3 = np.where(np.isfinite(d3), d3, np.nan)
    out[f"{t}_d1"] = d3[:, 0]
    out[f"{t}_d3"] = _nanmean(d3, np.isfinite(d3))
    out[f"{t}_n5"] = (d < 5).sum(1).astype(np.float32)
    out[f"{t}_n10"] = (d < 10).sum(1).astype(np.float32)
    near = np.argmin(d, axis=1)
    sp_n = sp[np.arange(len(d)), near]
    out[f"{t}_near_speed"] = np.where(np.isfinite(d.min(1)), sp_n, np.nan)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Indices pour toutes les frames. `df` doit contenir les colonnes de `ball.clean_ball`."""
    x, y = lt.track_array(df, "x"), lt.track_array(df, "y")
    team = lt.track_array(df, "team")
    present = np.isfinite(x) & np.isfinite(y)
    vx, vy, sp = player_speed(x, y)
    bx = df["ball_cx"].to_numpy(np.float32)
    by = df["ball_cy"].to_numpy(np.float32)
    f: dict[str, np.ndarray] = {}
    for t, k in TEAMS:
        f |= team_block(t, present & (team == k), x, y, vx, vy, sp, bx, by)
    for key in ("d1", "d3", "n5", "n10"):
        f[f"diff_{key}"] = f[f"h_{key}"] - f[f"a_{key}"]
    f["diff_depth"] = f["h_depth"] - f["a_depth"]
    valid = present & np.isfinite(sp)
    f["all_mean_speed"] = _nanmean(sp, valid)
    f["all_still"] = np.where(valid.sum(1) > 0, (valid & (sp < STILL)).sum(1) / np.maximum(valid.sum(1), 1),
                              np.nan)
    f["ball_present"] = np.isfinite(bx).astype(np.float32)
    f["ball_in_play_area"] = (np.isfinite(bx) & (bx > 0) & (bx < L) & (by > 0) & (by < W)).astype(np.float32)
    f["ball_edge_touch"] = np.minimum(by, W - by)
    f["ball_edge_goal"] = np.minimum(bx, L - bx)
    f["ball_centre_dist"] = np.hypot(bx - L / 2, by - W / 2)
    for c in ("ball_speed", "ball_acc", "ball_source", "gap_len", "gap_pos", "gap_type"):
        f[c] = df[c].to_numpy(np.float32)
    f["since_obs"] = np.minimum(df["since_obs"].to_numpy(np.float32), 1000)
    f["until_obs"] = np.minimum(df["until_obs"].to_numpy(np.float32), 1000)
    out = pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()})
    return add_temporal(out)


def add_temporal(out: pd.DataFrame) -> pd.DataFrame:
    """Moyennes glissantes centrées (1 s, 5 s) et variation futur - passé (3 s de part et d'autre)."""
    cols = {}
    base = [f"{t}_{b}" for t in ("h", "a") for b in ROLL_BASE] + ["diff_d3", "ball_speed", "all_still",
                                                                   "ball_present", "all_mean_speed"]
    for c in base:
        s = out[c]
        for w in WINDOWS:
            cols[f"{c}_m{w}"] = s.rolling(w, center=True, min_periods=1).mean().to_numpy(np.float32)
    for c in ("diff_d3", "h_mean_speed", "a_mean_speed", "h_spread_x", "a_spread_x"):
        past = out[c].rolling(75, min_periods=1).mean()
        fut = out[c][::-1].rolling(75, min_periods=1).mean()[::-1]
        cols[f"{c}_chg"] = (fut - past).to_numpy(np.float32)
    return pd.concat([out, pd.DataFrame(cols)], axis=1)


def forced_dead(n_frames: int) -> np.ndarray:
    """Frames connues D sans modèle (before the 2nd-half kick-off, per the official FAQ)."""
    m = np.zeros(n_frames, bool)
    if n_frames == common.N_FRAMES_REAL:
        m[PRE_KICKOFF[0]:PRE_KICKOFF[1]] = True
    return m
