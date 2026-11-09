"""V7 : arrêt imposé avant les reprises (option --setpiece-anchor).

Un coup de pied arrêté (touche, six mètres, corner, coup franc) du Smart Tagging (ou des événements, en dev) est
précédé d'un arrêt. Si le décodage n'a aucun D dans les LOOK_PRE_S secondes qui précèdent la reprise, on impose D
du début du dernier épisode d'indices de sortie jusqu'à la frame de reprise (exclue). Indice de sortie, à la frame f :
  - ballon à moins de NEAR_LINE_M d'une ligne, ou au-delà (ballon observé ou comblé) ;
  - ballon absent (ball_source == 2) ;
  - joueurs ralentis : médiane des vitesses des pistes, médiane glissante de 1 s, sous SLOW_MS.
L'épisode est le dernier groupe d'indices (trous <= MERGE_F fusionnés) qui touche [reprise - LOOK_BACK_S, reprise) ;
il est tronqué au début de cette fenêtre. Sans indice : D sur FALLBACK_S secondes avant la reprise. Seules les
frames notées changent. Constants are fixed a priori (no tuning on the match).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import common
import features
import load_tracking as lt

NEAR_LINE_M = 1.0  # m : le ballon d'une touche ou d'un six mètres est sur la ligne, à la précision du tracking près (0,3 m)
SLOW_MS = 1.2  # m/s : entre la médiane des frames D du dev (1,0-1,06) et le 25e centile des frames vivantes (1,26-1,43)
LOOK_PRE_S, LOOK_BACK_S, FALLBACK_S = 3.0, 15.0, 2.0  # consigne V7
MERGE_F = common.FPS  # 1 s
KINDS = ("throw-in", "goal kick", "corner kick", "free kick")  # Smart Tagging, minuscules


def exit_cues(dfb: pd.DataFrame) -> np.ndarray:
    """Indice de sortie par frame (booléen)."""
    L, W = common.PITCH_REAL if len(dfb) == common.N_FRAMES_REAL else common.PITCH_SAMPLE
    src = dfb["ball_source"].to_numpy()
    cx, cy = dfb["ball_cx"].to_numpy(), dfb["ball_cy"].to_numpy()
    has = (src <= 1) & np.isfinite(cx) & np.isfinite(cy)
    near = has & ((cx <= NEAR_LINE_M) | (cx >= L - NEAR_LINE_M) | (cy <= NEAR_LINE_M) | (cy >= W - NEAR_LINE_M))
    x, y = lt.track_array(dfb, "x"), lt.track_array(dfb, "y")
    _, _, sp = features.player_speed(x, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames sans piste : médiane NaN
        med = np.nanmedian(np.where(np.isfinite(sp), sp, np.nan), axis=1)
    slow = pd.Series(med).rolling(common.FPS, min_periods=common.FPS // 2, center=True).median().to_numpy() < SLOW_MS
    return near | (src == 2) | np.nan_to_num(slow, nan=False).astype(bool)


def set_piece_frames(tag: pd.DataFrame) -> np.ndarray:
    """Frames de reprise du Smart Tagging réel."""
    sp = tag[(tag["code"] == "set pieces") & tag["type"].isin(KINDS)]
    return np.sort(sp["start_frame"].to_numpy().astype(int))


def set_piece_frames_dev(ev: pd.DataFrame) -> np.ndarray:
    """Restart frames of the events of the Metrica matches (hors coup d'envoi et penalty)."""
    sub = ev["subtype"].astype(str)
    m = (ev["type"] == "SET PIECE") & (sub.str.startswith(("THROW IN", "FREE KICK", "CORNER KICK", "GOAL KICK")))
    return np.sort(ev.loc[m, "frame"].to_numpy().astype(int))


def apply(codes: np.ndarray, graded: np.ndarray, cues: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Retourne `codes` avec D imposé avant les reprises qui n'ont pas de D dans LOOK_PRE_S s (voir `plan`)."""
    out = codes.copy()
    for a, b in plan(codes, graded, cues, frames)[0]:
        seg = graded[a:b]
        out[a:b][seg] = common.D
    return out


def plan(codes: np.ndarray, graded: np.ndarray, cues: np.ndarray, frames: np.ndarray):
    """([(début, fin) imposés], nb de reprises notées, nb sans D avant, nb avec indice)."""
    pre, back, fb = (round(s * common.FPS) for s in (LOOK_PRE_S, LOOK_BACK_S, FALLBACK_S))
    n = len(codes)
    d = codes == common.D
    spans, n_graded, n_miss, n_cue = [], 0, 0, 0
    for s in frames:
        s = int(s)
        if not 0 < s < n or not graded[s]:
            continue
        n_graded += 1
        if d[max(s - pre, 0):s].any():
            continue
        n_miss += 1
        lo = max(s - back, 0)
        c = np.flatnonzero(cues[lo:s])
        if len(c):
            n_cue += 1
            # remonte l'épisode : les trous de moins de MERGE_F frames sont fusionnés
            idx = lo + c
            k = len(idx) - 1
            while k > 0 and idx[k] - idx[k - 1] <= MERGE_F:
                k -= 1
            start = int(idx[k])
        else:
            start = max(s - fb, 0)
        spans.append((start, s))
    return spans, n_graded, n_miss, n_cue
