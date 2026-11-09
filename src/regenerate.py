"""Regenerate the submission from a fresh clone: one command.

    python src/regenerate.py [--out submissions/s4.csv]
Output: s4 (every tuned value comes from IDSSE, Metrica or literature).
Steps whose output already exists in work/ are skipped. Prerequisites: `python src/setup_external.py` (pinned external
repositories). Match data are read from MTAH_DATA_DIR (default: ./private_data). Trained weights are read from models/v23s4 (CPU inference, deterministic).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import common

PY, SRC = sys.executable, common.ROOT / "src"


def step(name: str, done: Path, *cmd: str) -> None:
    if done.exists():
        print(f"[skip] {name}", flush=True)
        return
    print(f"[step] {name}", flush=True)
    subprocess.run([PY, str(SRC / cmd[0]), *cmd[1:]], check=True, cwd=common.ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    name = "s4"
    os.environ.setdefault("MTAH_DATA_DIR", str(common.ROOT / "private_data"))
    out = args.out or common.ROOT / "submissions" / f"{name}.csv"
    work = common.ROOT / "work"
    dfl = work / "external" / "dfl"
    step("Metrica tracks", work / "sample" / "Sample_Game_3.parquet", "load_tracking.py", "--regime", "sample")
    step("Metrica pseudo-labels", work / "sample" / "labels" / "Sample_Game_3.npy", "pseudo_labels.py")
    step("match tracks", work / "real" / "tactical.parquet", "load_tracking.py", "--regime", "real")
    step("IDSSE download (SHA-256 checked)", dfl / "MANIFEST.csv", "fetch_dfl.py")
    step("IDSSE conversion", dfl / "DFL_1.parquet", "dfl_convert.py")
    step("IDSSE aligned labels", dfl / "DFL_1_labels_aligned10.npy", "dfl_relabel.py")
    weights = common.ROOT / "models" / f"v23{name}"
    if not (weights / "model_0.txt").exists():
        sys.exit(f"weights missing: {weights}")
    os.environ.update(V23C=name[0], V23C_NAME=name, V21="feat", ALIGNED_LABELS="aligned", LABEL_ALIGN="10", WEIGHTS_DIR=str(weights))
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([PY, str(SRC / "label_match.py"), "--out", str(out)], check=True, cwd=common.ROOT)


if __name__ == "__main__":
    main()
