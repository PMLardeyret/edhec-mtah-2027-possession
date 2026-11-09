"""Common data layer for the Metrica matches (pseudo-labels) and the DFL matches (true labels), degraded tracking.

A name starting with "DFL_" is a DFL match: tracking = dfl_degrade.degrade_match (ball profile from configs/), labels =
work/external/dfl/DFL_k_labels*.npy. Other names go through model.feature_frame.
    python src/train_data.py [--workers 10] [--metrica]    builds all DFL features (seeds 0, 1, 100; swap/flip), with --metrica also Metrica's
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import common
import dfl_degrade as dd
import model
import variant_cfg

DFL = tuple(dd.MATCHES)
CACHE = dd.DFL / "feat"
TRAIN_SEEDS = (0, 1)
ALIGNED = os.environ.get("ALIGNED_LABELS") == "aligned"  # DFL labels aligned to the spell definition (dfl_relabel.py)
SUF = "_aligned" if ALIGNED else ""


ALIGN_TAG = os.environ.get("LABEL_ALIGN", "")  # "10" reads DFL_k_labels_aligned10.npy (alignment 1.0 s)


def is_dfl(name: str) -> bool:
    return name.startswith(("DFL_", "SK_"))  # SK_ prefix: same degradation chain as DFL


def profile() -> tuple[float, float]:
    r = json.loads(dd.PROFILE.read_text(encoding="utf-8"))["retenu"]
    return r["fc"], r["sigma"]


def labels(name: str) -> np.ndarray:
    if not is_dfl(name):
        return np.load(model.LABEL_DIR / f"{name}.npy")
    tag =f"_aligned{os.environ.get('LABEL_ALIGN', ALIGN_TAG) if name.startswith('DFL_') else ''}" if ALIGNED else ""
    return np.load((dd.SK if name.startswith("SK_") else dd.DFL) / f"{name}_labels{tag}.npy")


def degraded(name: str, seed: int, v21: bool | None = None) -> pd.DataFrame:
    fc, sigma = profile()
    return dd.degrade_match(name, seed, fc, sigma, v21)


def feature_frame(name: str, seed: int, swap: bool, flip: bool, v21: bool | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """`v21`: unused ball-degradation switch (default degrade.V21, always False)."""
    v21 = dd.degrade.V21 if v21 is None else v21
    if not is_dfl(name):
        return model.feature_frame(name, seed, swap, flip, v21)
    key = model.cache_key(CACHE, f"{name}_s{seed}_sw{int(swap)}_fl{int(flip)}", v21)
    key.parent.mkdir(parents=True, exist_ok=True)
    lab = labels(name)
    lab = model.swap_labels(lab) if swap else lab

    def make() -> pd.DataFrame:
        df = model.transform(degraded(name, seed, v21), swap, flip)
        f = model.build(df)
        f["graded"] = df["graded"].to_numpy()
        return f

    return model.cached_parquet(key, make), lab


def training_set(names, seeds=TRAIN_SEEDS, augment=True) -> tuple[pd.DataFrame, np.ndarray]:
    xs, ys = [], []
    variants = [(s, fl) for s in (False, True) for fl in (False, True)] if augment else [(False, False)]
    for m in names:
        for seed in seeds:
            for swap, flip in variants:
                f, lab = feature_frame(m, seed, swap, flip)
                keep = np.flatnonzero(f["graded"].to_numpy())[::model.STRIDE]
                xs.append(f.drop(columns="graded").iloc[keep])
                ys.append(lab[keep])
    return pd.concat(xs, ignore_index=True), np.concatenate(ys)


def eval_gt(name: str, graded: np.ndarray):
    lab = labels(name).astype(np.uint8)
    gt = np.where(graded, lab, common.X).astype(np.uint8)
    return gt, [(int(a), int(b)) for a, b in model._runs(graded)]


def _job(a):
    name, seed, swap, flip = a
    model.EXTRA = variant_cfg.extra()
    feature_frame(name, seed, swap, flip)
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--only-eval", action="store_true", help="seulement la graine 100 sans augmentation")
    ap.add_argument("--metrica", action="store_true", help="construit aussi les indices des matchs Metrica (avant ceux de DFL)")
    args = ap.parse_args()
    jobs = []
    if args.metrica:  # Metrica d'abord : le dernier indice construit (DFL_7) sert de témoin d'achèvement à regenerate.py
        jobs += [(m, model.EVAL_SEED, False, False) for m in model.MATCHES]
        if not args.only_eval:
            jobs += [(m, s, sw, fl) for m in model.MATCHES for s in TRAIN_SEEDS for sw in (False, True) for fl in (False, True)]
    jobs += [(m, model.EVAL_SEED, False, False) for m in DFL]
    if not args.only_eval:
        jobs += [(m, s, sw, fl) for m in DFL for s in TRAIN_SEEDS for sw in (False, True) for fl in (False, True)]
    with ProcessPoolExecutor(args.workers) as ex:
        for i, a in enumerate(ex.map(_job, jobs), 1):
            print(i, len(jobs), a, flush=True)


if __name__ == "__main__":
    main()
