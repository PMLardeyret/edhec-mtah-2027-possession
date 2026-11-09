"""V7-bis : signatures des reprises à ballon visible (option --setpiece-rules). Ballon mort (D) si, à la frame f :

  out    le ballon observé est à plus de OUT_M hors du terrain pendant au moins OUT_RUN frames consécutives ;
  corner le ballon est immobile (voir `still`) à moins de CORNER_R d'un point de corner ;
  six    le ballon est immobile dans la petite surface (5,5 m x 18,32 m) et aucun joueur de l'équipe qui n'est pas celle
         du gardien de ce but n'est dans la surface de réparation (16,5 m x 40,32 m) : six mètres ;
  wall   le ballon est immobile et au moins WALL_N joueurs d'une même équipe sont à WALL_LO-WALL_HI m de lui, dans un
         secteur de moins de WALL_SPAN degrés vu du ballon (mur de coup franc).
Immobile = ballon observé (ball_source == 0) de vitesse < STILL_MS pendant plus de STILL_F frames consécutives (toute la suite).
Exception : jamais de D si un gardien est à moins de GK_M du ballon dans sa surface de réparation (le jeu continue
quand le gardien tient le ballon). Dimensions : IFAB, Lois du jeu, lois 1 et 16. Constantes fixées avant mesure.
Interprétation de « aucun adversaire dans la surface de réparation » : adversaire du gardien de ce but.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import common
import load_tracking as lt

OUT_M, OUT_RUN = 0.5, 5
STILL_MS, STILL_F = 0.5, common.FPS  # « plus de 1 s » : suites de plus de STILL_F frames
CORNER_R = 2.0
GOAL_D, GOAL_W = 5.5, 18.32
PEN_D, PEN_W = 16.5, 40.32
WALL_N, WALL_LO, WALL_HI, WALL_SPAN = 3, 8.0, 11.0, 30.0
GK_M = 1.5
RULES = ("out", "corner", "six", "wall")


def _runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Masque des suites vraies d'au moins `min_len` frames."""
    out = np.zeros(len(mask), bool)
    e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1), strict=True):
        if b - a >= min_len:
            out[a:b] = True
    return out


def still(dfb: pd.DataFrame) -> np.ndarray:
    ok = (dfb["ball_source"].to_numpy() == 0) & np.isfinite(dfb["ball_cx"].to_numpy())
    return _runs(ok & (np.nan_to_num(dfb["ball_speed"].to_numpy(), nan=99.0) < STILL_MS), STILL_F + 1)


def _geometry(dfb: pd.DataFrame):
    L, W = common.PITCH_REAL if len(dfb) == common.N_FRAMES_REAL else common.PITCH_SAMPLE
    bx, by = dfb["ball_cx"].to_numpy(), dfb["ball_cy"].to_numpy()
    obs = (dfb["ball_source"].to_numpy() == 0) & np.isfinite(bx)
    x, y = lt.track_array(dfb, "x"), lt.track_array(dfb, "y")
    team, gk = lt.track_array(dfb, "team"), lt.track_array(dfb, "gk").astype(bool)
    return L, W, bx, by, obs, x, y, team, gk


def masks(dfb: pd.DataFrame) -> dict[str, np.ndarray]:
    """Un masque booléen par règle (sans l'exception du gardien) et `gk_hold` : frames où l'exception s'applique."""
    L, W, bx, by, obs, x, y, team, gk = _geometry(dfb)
    n = len(dfb)
    with np.errstate(invalid="ignore"):
        outside = obs & ((bx < -OUT_M) | (bx > L + OUT_M) | (by < -OUT_M) | (by > W + OUT_M))
        st = still(dfb)
        corner = np.zeros(n, bool)
        for cx in (0.0, L):
            for cy in (0.0, W):
                corner |= np.hypot(bx - cx, by - cy) < CORNER_R
        corner &= st
        dx, dy = x - bx[:, None], y - by[:, None]
        dist = np.hypot(dx, dy)
        left = bx < L / 2
        in_goal = (np.minimum(bx, L - bx) <= GOAL_D) & (np.abs(by - W / 2) <= GOAL_W / 2)
        px = np.where(left[:, None], x, L - x)  # distance à la ligne de but défendue
        in_pen = (px <= PEN_D) & (np.abs(y - W / 2) <= PEN_W / 2) & np.isfinite(x)
        on_side = np.where(left[:, None], x < L / 2, x >= L / 2)
        gkt = np.where(gk & on_side & np.isfinite(x), team, np.nan)
        has_gk = np.isfinite(gkt).any(axis=1)
        gk_team = np.where(has_gk, np.nanmax(np.where(np.isfinite(gkt), gkt, -1), axis=1), np.nan)
        opp_in_pen = (in_pen & np.isfinite(team) & (team != gk_team[:, None])).any(axis=1)
        six = st & in_goal & has_gk & ~opp_in_pen
        ang = np.degrees(np.arctan2(dy, dx))
        wall = np.zeros(n, bool)
        for i in np.flatnonzero(st & obs):
            for k in (0, 1):
                m = (team[i] == k) & (dist[i] >= WALL_LO) & (dist[i] <= WALL_HI)
                if m.sum() >= WALL_N:
                    a = np.sort(ang[i][m])
                    a3 = np.concatenate((a, a[:WALL_N - 1] + 360.0))
                    if (a3[WALL_N - 1:WALL_N - 1 + len(a)] - a).min() < WALL_SPAN:
                        wall[i] = True
                        break
        near_gk = (gk & (dist <= GK_M) & in_pen & np.isfinite(dist)).any(axis=1) & obs
    return {"out": _runs(outside, OUT_RUN), "corner": corner, "six": six, "wall": wall, "gk_hold": near_gk}


def apply(codes: np.ndarray, graded: np.ndarray, m: dict[str, np.ndarray], rules: tuple[str, ...] = RULES) -> np.ndarray:
    """D sur les frames notées où au moins une règle choisie s'applique, sauf exception du gardien."""
    dead = np.zeros(len(codes), bool)
    for r in rules:
        dead |= m[r]
    out = codes.copy()
    out[dead & ~m["gk_hold"] & graded] = common.D
    return out
