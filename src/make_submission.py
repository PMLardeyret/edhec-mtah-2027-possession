"""Écrit un CSV de soumission (frame,label) puis lance validate_submission.py dessus."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

import common

N = common.N_FRAMES_REAL


def write_submission(codes: np.ndarray, path: Path, validate: bool = True) -> Path:
    """codes : tableau de 0..3 (H, A, D, X) de longueur 199 319."""
    if len(codes) != N:
        raise ValueError(f"{len(codes)} lignes au lieu de {N}")
    letters = np.array(common.LABELS)[codes]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{lab}\n" for i, lab in enumerate(letters))
    if validate:
        r = subprocess.run(
            [sys.executable, str(common.OFFICIAL / "evaluation" / "validate_submission.py"),
             str(path)], capture_output=True, text=True, check=False)
        if r.returncode != 0:
            raise RuntimeError("validation échouée : " + (r.stdout + r.stderr)[-300:])
    return path
