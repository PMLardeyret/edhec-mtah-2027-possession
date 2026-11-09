"""Per-frame model: LightGBM on degraded open matches.

Protocol: leave-one-match-out folds. Training on the other matches degraded (training seeds, team swap x pitch mirror augmentation),
evaluation on the held-out degraded match with a distinct seed.
Output: probabilities (p_H, p_A, p_D) per frame, temperature-calibrated (fitted out of fold).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import ball
import common
import dead_ball
import degrade
import features
import latent_ball
import variant_cfg
from score import spell_f1

MATCHES = ("Sample_Game_1", "Sample_Game_2", "Sample_Game_3")
TRAIN_SEEDS = (0, 1)
EVAL_SEED = 100
STRIDE = 5  # sous-échantillonnage à l'entraînement (frames voisines très corrélées)
SEED = 7
CACHE = common.ROOT / "work" / "sample" / "feat"
EXTRA: tuple[str, ...] = ()  # modules V2 actifs : "stop" (M2, arrêt ballon visible), "latent" (M1, balle latente)
LABEL_DIR = common.ROOT / "work" / "sample" / "labels"  # surchargeable (diagnostics sur d'autres étiquettes)
PARAMS = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 100,
          "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.7, "reg_lambda": 1.0,
          "random_state": SEED, "n_jobs": 4, "verbose": -1}


def v21_feat() -> bool:
    """V21 : indices d'arrêt ballon visible (stop_features), activés par la variable d'environnement V21 contenant « feat »."""
    return "feat" in os.environ.get("V21", "")


def cache_key(directory: Path, name: str, v21: bool) -> Path:
    """Feature cache path; the V21 switch has its own directory (feat_v21), the stop features their own suffix."""
    tag = "".join(f"_{e}" for e in EXTRA) + ("_v21f" if v21_feat() else "") + variant_cfg.cache_suffix()
    return (directory.with_name(directory.name + "_v21") if v21 else directory) / f"{name}{tag}.parquet"


def transform(df: pd.DataFrame, swap: bool, flip: bool) -> pd.DataFrame:
    """Augmentation : échange des équipes et/ou miroir du terrain (x -> L-x, y -> W-y)."""
    out = df.copy()
    if flip:
        L, W = common.PITCH_SAMPLE
        for c in out.columns:
            if c.endswith("_x") and c != "ball_cx":
                out[c] = L - out[c]
            elif c.endswith("_y") and c != "ball_cy":
                out[c] = W - out[c]
    if swap:
        for c in out.columns:
            if c.endswith("_team"):
                out[c] = out[c].where(out[c] < 0, 1 - out[c])
    return out


def swap_labels(lab: np.ndarray) -> np.ndarray:
    out = lab.copy()
    out[lab == common.H], out[lab == common.A] = common.A, common.H
    return out


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Base features + active modules, for a tracking table (degraded open match or match to label)."""
    dfb = ball.with_ball(df)
    f = features.build_features(dfb)
    if "stop" in EXTRA:
        f = dead_ball.stop_features(f, dfb)
    if "latent" in EXTRA:
        f = latent_ball.latent_features(f, dfb)
    if v21_feat():
        import stop_features
        f = stop_features.add(f, dfb)
    return f


def cached_parquet(key, make) -> pd.DataFrame:
    """Lit un cache parquet, sinon le construit avec `make()`. Écriture atomique (fichier temporaire puis os.replace) ; un cache
    illisible (incomplet, corrompu) est supprimé et reconstruit au lieu de faire planter le calcul."""
    if key.exists():
        try:
            return pd.read_parquet(key)
        except (OSError, ValueError, EOFError):
            try:
                key.unlink()
            except OSError:
                pass  # un autre processus l'a déjà remplacé ou le lit encore
    f = make()
    tmp = key.with_name(f"{key.name}.{os.getpid()}.tmp")
    f.to_parquet(tmp)
    try:
        os.replace(tmp, key)
    except PermissionError:  # Windows : la cible est ouverte par un autre processus ; son contenu est identique
        tmp.unlink(missing_ok=True)
        if not key.exists():
            raise
    return f


def feature_frame(match: str, seed: int, swap: bool, flip: bool, v21: bool | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Indices + labels d'un match dégradé (graine `seed`), mis en cache. `v21` : dégradation V21 (défaut : degrade.V21)."""
    v21 = degrade.V21 if v21 is None else v21
    key = cache_key(CACHE, f"{match}_s{seed}_sw{int(swap)}_fl{int(flip)}", v21)
    key.parent.mkdir(parents=True, exist_ok=True)
    lab = np.load(LABEL_DIR / f"{match}.npy")
    raw_lab = lab
    lab = swap_labels(lab) if swap else lab

    def make() -> pd.DataFrame:
        df = pd.read_parquet(common.ROOT / "work" / "sample" / f"{match}.parquet")
        df = transform(degrade.degrade(df, seed, state=raw_lab, v21=v21), swap, flip)
        f = build(df)
        f["graded"] = df["graded"].to_numpy()
        return f

    return cached_parquet(key, make), lab


def training_set(matches, seeds=TRAIN_SEEDS, augment=True) -> tuple[pd.DataFrame, np.ndarray]:
    xs, ys = [], []
    variants = [(s, fl) for s in (False, True) for fl in (False, True)] if augment else [(False, False)]
    for m in matches:
        for seed in seeds:
            for swap, flip in variants:
                f, lab = feature_frame(m, seed, swap, flip)
                keep = np.flatnonzero(f["graded"].to_numpy())[::STRIDE]
                xs.append(f.drop(columns="graded").iloc[keep])
                ys.append(lab[keep])
    return pd.concat(xs, ignore_index=True), np.concatenate(ys)


def rules_v1(f: pd.DataFrame) -> np.ndarray:
    """Règles à la main : l'équipe la plus proche du ballon (ou, sans ballon, du bloc moyen sur 5 s)
    a la possession ; arrêt si les joueurs sont lents/immobiles, le ballon absent ou hors jeu."""
    present = f["ball_present"].to_numpy() > 0
    ref = np.where(present, f["diff_d1"], f["diff_d3_m125"])
    ref = np.nan_to_num(ref, nan=0.0)
    p_ha = 1 / (1 + np.exp(ref / 3.0))  # ref < 0 : l'équipe 0 est plus proche
    z = (-1.0 + 3.0 * (np.nan_to_num(f["all_still_m25"]) > 0.25)
         + 1.5 * (1 - f["ball_present_m125"].to_numpy())
         + 3.0 * (present & (f["ball_in_play_area"].to_numpy() == 0))
         - 2.0 * (np.nan_to_num(f["ball_speed_m25"]) > 6))
    p_d = 1 / (1 + np.exp(-z))
    return np.stack([(1 - p_d) * p_ha, (1 - p_d) * (1 - p_ha), p_d], 1)


def fit_temperature(p_logit: np.ndarray, y: np.ndarray) -> float:
    """Température T minimisant la log-vraisemblance négative de softmax(logit / T)."""
    best, best_t = np.inf, 1.0
    for t in np.linspace(0.5, 8.0, 31):
        z = p_logit / t
        z = z - z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        nll = -lp[np.arange(len(y)), y].mean()
        if nll < best:
            best, best_t = nll, t
    return float(best_t)


def apply_temperature(proba: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(proba, 1e-6, 1)) / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_lgbm(X: pd.DataFrame, y: np.ndarray, **extra) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(objective="multiclass", **(PARAMS | extra)).fit(X, y)


def predict_proba(model: lgb.LGBMClassifier, f: pd.DataFrame, forced: np.ndarray | None = None) -> np.ndarray:
    p = model.predict_proba(f.drop(columns="graded", errors="ignore"))
    if forced is not None:
        p[forced] = (0.0, 0.0, 1.0)
    return p


def argmax_smoothed(p: np.ndarray, win: int = 125) -> np.ndarray:
    """Décision provisoire (le vrai décodeur est dans decode.py) : argmax du vote sur `win` frames."""
    v = pd.DataFrame(p).rolling(win, center=True, min_periods=1).mean().to_numpy()
    return v.argmax(1).astype(np.uint8)


def eval_gt(match: str, graded: np.ndarray) -> tuple[np.ndarray, list]:
    lab = np.load(LABEL_DIR / f"{match}.npy").astype(np.uint8)
    gt = np.where(graded, lab, common.X).astype(np.uint8)
    return gt, [(int(a), int(b)) for a, b in _runs(graded)]


def _runs(mask: np.ndarray):
    e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1), strict=True)


def cross_validate(augment: bool = True) -> dict:
    """Retourne spellF1 (provisoire) par pli pour v1 et v2, et les probabilités hors pli."""
    res = {"v1": [], "v2": []}
    oof = {}
    for test in MATCHES:
        train = [m for m in MATCHES if m != test]
        X, y = training_set(train, augment=augment)
        model = fit_lgbm(X, y)
        # température : prédictions croisées entre les deux matchs d'entraînement, graine d'éval
        logits, ys = [], []
        for cal, other in ((train[0], train[1]), (train[1], train[0])):
            Xo, yo = training_set([other], augment=augment)
            m_o = fit_lgbm(Xo, yo)
            fc, labc = feature_frame(cal, EVAL_SEED, False, False)
            g = fc["graded"].to_numpy()
            p = predict_proba(m_o, fc)[g][::STRIDE]
            logits.append(np.log(np.clip(p, 1e-6, 1)))
            ys.append(labc[g][::STRIDE])
        temp = fit_temperature(np.concatenate(logits), np.concatenate(ys))
        f, _ = feature_frame(test, EVAL_SEED, False, False)
        graded = f["graded"].to_numpy()
        gt, runs = eval_gt(test, graded)
        p2 = apply_temperature(predict_proba(model, f), temp)
        p1 = rules_v1(f)
        for name, p in (("v1", p1), ("v2", p2)):
            res[name].append(spell_f1(gt, np.where(graded, argmax_smoothed(p), common.X), runs))
        oof[test] = {"v1": p1, "v2": p2, "T": temp}
        print(test, "T", round(temp, 2), "v1", round(res["v1"][-1], 3), "v2", round(res["v2"][-1], 3))
    return {"scores": res, "oof": oof}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--out", type=Path, default=common.ROOT / "work" / "sample" / "oof")
    args = ap.parse_args()
    r = cross_validate(augment=not args.no_augment)
    args.out.mkdir(parents=True, exist_ok=True)
    for m, d in r["oof"].items():
        np.savez(args.out / f"{m}.npz", v1=d["v1"], v2=d["v2"], T=d["T"])
    for k, v in r["scores"].items():
        print(k, "moyenne", round(float(np.mean(v)), 4), [round(x, 3) for x in v])


if __name__ == "__main__":
    main()
