"""Entry point for the match to label: tracking loading, probabilities, decoding helpers, aggregates only.

Prints and writes only counts, percentages and durations. No data row.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import ball
import common
import decode
import features
import load_tracking as lt
import model

TRAIN_SEEDS = (0, 1, 2)
DETERMINISTIC = {"deterministic": True, "force_row_wise": True, "n_jobs": 1}


def real_tracking(data: Path | None) -> pd.DataFrame:
    path = common.ROOT / "work" / "real" / "tactical.parquet"
    if not path.exists():
        import sys
        sys.argv = ["load_tracking", "--regime", "real", *(["--data", str(data)] if data else [])]
        lt.main()
    return pd.read_parquet(path)


def median_temperature(tag: str = "") -> float:
    d = "oof_" + tag if tag else "oof"
    ts = [float(np.load(common.ROOT / "work" / "sample" / d / f"{m}.npz")["T"]) for m in model.MATCHES]
    return float(np.median(ts))


def predict_real(df: pd.DataFrame, tag: str = "") -> np.ndarray:
    X, y = model.training_set(model.MATCHES, seeds=TRAIN_SEEDS)
    clf = model.fit_lgbm(X, y, **DETERMINISTIC)
    f = model.build(df)
    p = model.apply_temperature(clf.predict_proba(f), median_temperature(tag))
    p[features.forced_dead(len(df))] = (0.0, 0.0, 1.0)
    return p


def learned_cost(df: pd.DataFrame, stick: float):
    """Hazards learned on the 3 Metrica matches (pseudo-labels), per-step cost for the match to label."""
    import hazard
    train = []
    for m in model.MATCHES:
        f, lab = model.feature_frame(m, 0, False, False)
        train.append((hazard.covariates(f), lab, f["graded"].to_numpy()))
    hz = hazard.Hazards().fit(train)
    cov = hazard.covariates(model.build(df))
    return lambda a, b: hz.cost(cov, a, b, stick)


def labels_real(df: pd.DataFrame, p: np.ndarray, config: dict, stick: float | None = None,
                cost_fn=None, short_poss: float | None = None,
                setpiece: np.ndarray | None = None, rules: tuple[str, ...] | None = None) -> np.ndarray:
    graded = df["graded"].to_numpy()
    kw = {k: v for k, v in config.items() if k != "score_dev"}
    codes = decode.decode_viterbi(p, graded, cost_fn=cost_fn if stick is None else learned_cost(df, stick), **kw)
    if short_poss is not None:
        import short_fix
        codes = short_fix.apply(ball.with_ball(df), graded, p, codes, kw, short_poss, cost_fn)
    if setpiece is not None:
        import setpiece_anchor
        codes = setpiece_anchor.apply(codes, graded, setpiece_anchor.exit_cues(ball.with_ball(df)), setpiece)
    if rules:
        import setpiece_rules
        codes = setpiece_rules.apply(codes, graded, setpiece_rules.masks(ball.with_ball(df)), rules)
    codes[codes == common.X] = common.D  # hors frames notées : ignoré par le scoreur
    codes[features.forced_dead(len(df))] = common.D
    return codes.astype(np.uint8)


def aggregates(codes: np.ndarray, graded: np.ndarray) -> dict:
    from score import n_spells
    runs = [(int(a), int(b)) for a, b in model._runs(graded)]
    n, dur = n_spells(codes, runs)
    g = codes[graded]
    return {"frames": len(codes), "possessions": n, "duree_moy_s": round(dur, 2),
            "pct_H": round(100 * float((g == common.H).mean()), 1),
            "pct_A": round(100 * float((g == common.A).mean()), 1),
            "pct_D": round(100 * float((g == common.D).mean()), 1)}
