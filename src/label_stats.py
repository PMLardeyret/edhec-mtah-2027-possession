"""Per-label-set statistics (possessions, durations, share of dead ball) used to describe label definitions."""

from __future__ import annotations

import numpy as np

import common

common.ensure_paths()
import possession_metrics as pm


def stats(codes: np.ndarray, runs: list) -> dict:
    sp = pm.possession_spells(codes, runs)
    dur = np.array([b - a for a, b, _ in sp], float) / common.FPS
    g = np.concatenate([codes[a:b] for a, b in runs])
    minutes = len(g) / common.FPS / 60
    live = float((g != common.D).sum()) / common.FPS / 60
    return {"possessions": len(sp), "duree_moy_s": round(float(dur.mean()), 2), "duree_med_s": round(float(np.median(dur)), 2),
            "part_D_%": round(100 * float((g == common.D).mean()), 1), "part_moins_3s_%": round(100 * float((dur < 3).mean()), 1),
            "poss_par_min_notee": round(len(sp) / minutes, 2), "poss_par_min_jeu": round(len(sp) / live, 2)}
