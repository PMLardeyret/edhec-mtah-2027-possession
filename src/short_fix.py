"""V6 : possessions courtes après une frappe (correction ciblée, option --short-poss).

Le plancher du décodeur (1,5 s) interdit les réceptions brèves qui suivent une frappe, même ballon visible. Règle :
une réception = première frame, après un épisode du ballon observé à plus de SHOT_MS dans les PRE_F frames
précédentes, où le ballon observé est à moins de NEAR_M d'un joueur ; elle est éligible si ce joueur est de
l'équipe qui n'avait pas la possession (étiquette de base 0,2 s avant). Autour de chaque réception éligible
([r - WIN_F, r + WIN_F]), on prend le décodage refait avec une durée minimale de possession abaissée
(`min_poss_s`), à condition que ses étiquettes aux deux bords de la fenêtre soient celles du décodage de base :
aucune frontière nouvelle aux bords. Hors des fenêtres, rien ne change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import common
import decode
import load_tracking as lt

SHOT_MS, NEAR_M = 15.0, 2.0
PRE_F, WIN_F = 50, 50  # 2 s
LOOKBACK = 5  # frames avant la réception pour lire la possession de base


def receptions(dfb: pd.DataFrame, base: np.ndarray, graded: np.ndarray) -> list[int]:
    """Frames de réception éligibles (équipe qui reçoit != équipe en possession juste avant)."""
    src = dfb["ball_source"].to_numpy() == 0
    cx, cy = dfb["ball_cx"].to_numpy(), dfb["ball_cy"].to_numpy()
    sp = np.nan_to_num(dfb["ball_speed"].to_numpy(), nan=0.0)
    pre = pd.Series(sp).rolling(PRE_F, min_periods=1).max().shift(1).fillna(0).to_numpy()
    cand = np.flatnonzero(src & graded & np.isfinite(cx) & (pre > SHOT_MS))
    if not len(cand):
        return []
    x, y, t = (lt.track_array(dfb, k)[cand] for k in ("x", "y", "team"))
    d = np.hypot(x - cx[cand, None], y - cy[cand, None])
    d = np.where(np.isfinite(d) & np.isfinite(t) & (t >= 0), d, np.inf)
    j = d.argmin(1)
    close = d[np.arange(len(cand)), j] < NEAR_M
    team = t[np.arange(len(cand)), j]
    out, last = [], -10**9
    for r, ok, tm in zip(cand, close, team, strict=True):
        if not ok:
            continue
        if r - last > PRE_F:  # début d'un nouveau groupe de réceptions
            prev = base[max(r - LOOKBACK, 0)]
            if prev in (common.H, common.A) and int(tm) in (0, 1) and int(tm) != int(prev):
                out.append(int(r))
        last = r
    return out


def apply(dfb: pd.DataFrame, graded: np.ndarray, p: np.ndarray, base: np.ndarray, cfg: dict,
          min_poss_s: float, cost_fn=None) -> np.ndarray:
    """Retourne `base` avec les fenêtres de réception re-décodées à `min_poss_s`."""
    rec = receptions(dfb, base, graded)
    if not rec:
        return base
    alt = decode.decode_viterbi(p, graded, cost_fn=cost_fn, **(cfg | {"min_poss_s": min_poss_s}))
    out = base.copy()
    for r in rec:
        a, b = max(r - WIN_F, 0), min(r + WIN_F, len(base))
        if alt[a] == base[a] and alt[b - 1] == base[b - 1]:
            out[a:b] = alt[a:b]
    return out
