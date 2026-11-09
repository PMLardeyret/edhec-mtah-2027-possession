"""Final model: five LightGBM models trained on degraded DFL + Metrica matches (LightGBM seed 7+k, deterministic).

`train(k)` fits model k and writes its text weights when WEIGHTS_SAVE is set (no pickle); `infer(k, dir)` reads the text weights on CPU
and returns the same probabilities. The decoding chain is driven by label_match.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import common
import features
import model
import run_real
import train_data as vd

common.ensure_paths()

K = int(os.environ.get("MODEL_K", "5"))  # number of models
NAMES = [*vd.DFL, *model.MATCHES]
EXTRA = ("stop", "latent")  # feature modules (overridden by the variant configuration)


def p_file(k: int) -> Path:
    return common.ROOT / "work" / "real" / f"p_final{vd.SUF}_{k}.npy"


def train(k: int) -> int:
    model.EXTRA = EXTRA
    df = run_real.real_tracking(None)
    X, y = vd.training_set(NAMES)
    clf = model.fit_lgbm(X, y, **run_real.DETERMINISTIC, random_state=7 + k)
    if os.environ.get("WEIGHTS_SAVE"):  # LightGBM text weights (no pickle)
        out = Path(os.environ["WEIGHTS_SAVE"])
        out.mkdir(parents=True, exist_ok=True)
        clf.booster_.save_model(str(out / f"model_{k}.txt"))
    p = model.apply_temperature(clf.predict_proba(model.build(df)), 1.0)  # T médiane des 7 plis = 1,00
    p[features.forced_dead(len(df))] = (0.0, 0.0, 1.0)
    np.save(p_file(k), p)
    return k


def infer(k: int, wdir: Path) -> int:
    """Inférence CPU depuis les poids texte de `wdir`  ; même sortie que train(k) sans l'entraînement."""
    import lightgbm as lgb
    model.EXTRA = EXTRA
    df = run_real.real_tracking(None)
    bst = lgb.Booster(model_file=str(wdir / f"model_{k}.txt"))
    p = model.apply_temperature(bst.predict(model.build(df)), 1.0)
    p[features.forced_dead(len(df))] = (0.0, 0.0, 1.0)
    np.save(p_file(k), p)
    return k
