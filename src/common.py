"""Chemins et constantes partagés : rend importables les modules officiels (games, métrique)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OFFICIAL = ROOT / "external" / "more-than-a-hack-2027"
SAMPLE_DIR = ROOT / "external" / "sample-data" / "data"

FPS = 25
PITCH_REAL = (105.3, 68.0)  # games.py : from the official games.py
PITCH_SAMPLE = (105.0, 68.0)
N_FRAMES_REAL = 199319
LABELS = ("H", "A", "D", "X")
H, A, D, X = 0, 1, 2, 3  # mêmes codes que possession_metrics


def ensure_paths() -> None:
    """Ajoute au sys.path les dossiers du dépôt officiel (games.py, evaluation/)."""
    for p in (OFFICIAL, OFFICIAL / "evaluation", OFFICIAL / "starter"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


ensure_paths()
