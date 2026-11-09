"""Align the IDSSE (DFL) labels with a possession definition based on ball control.

    python src/dfl_relabel.py
A team change (an H or A segment different from the previous H/A segment) only counts if the new team controls the ball: at least
2 consecutive touches by that team (Play / OtherBallAction events, no opposing touch in between), or an IDSSE segment of at least
1.0 s. Otherwise the segment is absorbed by the previous team. Dead-ball (D) frames are unchanged. The rule uses durations and
touches only. Writes DFL_k_labels_aligned10.npy next to the converted tracking.
"""

from __future__ import annotations

import numpy as np

import common
import dfl_events
from dfl_convert import DIR


def runs_of(lab: np.ndarray) -> list[tuple[int, int, int]]:
    cut = np.flatnonzero(np.diff(lab.astype(int))) + 1
    b = np.concatenate([[0], cut, [len(lab)]])
    return [(int(b[i]), int(b[i + 1]), int(lab[b[i]])) for i in range(len(b) - 1)]


def has_control(fr: np.ndarray, tm: np.ndarray, a: int, b: int, team: int) -> bool:
    """Au moins deux touches consécutives de `team` (dans la suite complète des touches) tombant dans [a, b)."""
    i, j = np.searchsorted(fr, a), np.searchsorted(fr, b)
    same = (tm[i:j] == team)
    return bool(np.any(same[1:] & same[:-1]))


def align(lab: np.ndarray, fr: np.ndarray, tm: np.ndarray, min_s: float = 1.0, touches: int = 2) -> np.ndarray:
    out = lab.copy()
    holder = None
    for a, b, c in runs_of(lab):
        if c == common.D:
            continue
        if holder is None or c == holder:
            holder = c
            continue
        if (b - a) >= min_s * common.FPS or has_control(fr, tm, a, b, c):
            holder = c
        else:
            out[a:b] = holder
    return out


def main() -> None:
    for k in range(1, 8):
        lab = np.load(DIR / f"DFL_{k}_labels.npy")
        fr, tm = dfl_events.touch_frames_dfl(k)
        new = align(lab, fr, tm, 1.0, 2)
        assert ((new == common.D) == (lab == common.D)).all()
        np.save(DIR / f"DFL_{k}_labels_aligned10.npy", new)
        print(f"DFL_{k}: aligned labels written", flush=True)


if __name__ == "__main__":
    main()
