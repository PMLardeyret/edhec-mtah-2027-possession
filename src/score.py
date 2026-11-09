"""Scoring helpers around the official possession_metrics module (never re-implemented)."""

from __future__ import annotations

import numpy as np

import common

common.ensure_paths()
import possession_metrics as pm


def n_spells(codes: np.ndarray, runs=None) -> tuple[int, float]:
    """(number of possessions, mean duration in s) of a code array, no reference needed."""
    runs = runs if runs is not None else [(0, len(codes))]
    sp = pm.possession_spells(codes, runs)
    return len(sp), (float(np.mean([b - a for a, b, _ in sp])) / common.FPS if sp else 0.0)


def spell_f1(gt: np.ndarray, pred: np.ndarray, runs=None) -> float:
    """Mean spell F1 over the IoU thresholds, within `runs` (derived from gt when absent). Used on external data only."""
    runs = runs if runs is not None else pm.valid_runs(gt)
    return float(pm.spell_scores(gt, pm.fill_abstentions(pred, runs), runs)["mean_f1"])
