"""Ball cleaning: jumps, isolated points, smoothing, bridging of short gaps, gap columns.

Input: loader DataFrame (ball_x, ball_y in metres). Added columns:
  ball_cx, ball_cy   cleaned position (NaN where not bridged)
  ball_source        0 observed, 1 filled (gap <= BRIDGE_MAX), 2 missing
  gap_len            length (frames) of the gap the frame belongs to, 0 if observed
  gap_pos            progress inside the gap (0..1), 0 if observed
  gap_type           0 observed, 1 short (<=25), 2 medium (26-125), 3 long (>125)
  since_obs, until_obs  frames since / until the last / next observation (capped)
  ball_vx, ball_vy, ball_speed, ball_acc  m/s and m/s^2, NaN where the ball is missing
Same function for training matches and for the match to label; no data is printed.
"""

from __future__ import annotations

import argparse
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import common

MAX_SPEED = 40.0  # m/s : au-delà, saut
MIN_RUN = 5  # frames : une suite observée plus courte est isolée
SG_WINDOW, SG_ORDER = 7, 2
BRIDGE_MAX = 25  # 1 s
GAP_EDGES = (25, 125)
SIGMA_OBS = 0.3  # m, bruit de mesure du ballon
SIGMA_ACC = 30.0  # m/s², accélération typique d'un coup de pied
PAD = 12  # frames observées de part et d'autre d'un trou pour le lisseur
CAP = 5000


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(e == 1).tolist(), np.flatnonzero(e == -1).tolist(), strict=True))


def remove_outliers(x: np.ndarray, y: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Masque des frames conservées : coupe aux sauts > MAX_SPEED et aux changements de mi-temps,
    supprime les morceaux de moins de MIN_RUN frames."""
    ok = np.isfinite(x) & np.isfinite(y)
    step = np.hypot(np.diff(x), np.diff(y)) * common.FPS
    jump = np.concatenate(([False], np.nan_to_num(step, nan=0.0) > MAX_SPEED))
    jump |= np.concatenate(([False], half[1:] != half[:-1]))
    keep = np.zeros(len(x), bool)
    for a, b in _runs(ok):
        cuts = [a, *(a + np.flatnonzero(jump[a + 1:b]) + 1).tolist(), b]
        for s, e in pairwise(cuts):
            if e - s >= MIN_RUN:
                keep[s:e] = True
    return keep


def _rts_1d(z: np.ndarray) -> np.ndarray:
    """Lisseur RTS à vitesse constante (position, vitesse) ; z contient NaN aux mesures absentes."""
    n, dt = len(z), 1.0 / common.FPS
    Q = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]]) * SIGMA_ACC**2
    F = np.array([[1.0, dt], [0.0, 1.0]])
    R = SIGMA_OBS**2
    first = int(np.flatnonzero(np.isfinite(z))[0])
    m = np.array([z[first], 0.0])
    P = np.diag([R, 100.0])
    xf, Pf, xp, Pp = np.zeros((n, 2)), np.zeros((n, 2, 2)), np.zeros((n, 2)), np.zeros((n, 2, 2))
    for k in range(n):
        if k:
            m, P = F @ m, F @ P @ F.T + Q
        xp[k], Pp[k] = m, P
        if np.isfinite(z[k]):
            g = P[:, 0] / (P[0, 0] + R)
            m = m + g * (z[k] - m[0])
            P = P - np.outer(g, P[0])
        xf[k], Pf[k] = m, P
    xs = xf.copy()
    for k in range(n - 2, -1, -1):
        c = Pf[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xf[k] + c @ (xs[k + 1] - xp[k + 1])
    return xs[:, 0]


def _derivative(v: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Dérivée centrale (par seconde) sur chaque suite finie d'au moins 3 frames."""
    d = np.full(len(v), np.nan)
    for a, b in _runs(ok):
        if b - a >= 3:
            d[a:b] = np.gradient(v[a:b]) * common.FPS
    return d


def clean_ball(df: pd.DataFrame) -> pd.DataFrame:
    x0 = df["ball_x"].to_numpy(np.float64)
    y0 = df["ball_y"].to_numpy(np.float64)
    half = df["half"].to_numpy()
    n = len(df)
    keep = remove_outliers(x0, y0, half)
    x, y = np.where(keep, x0, np.nan), np.where(keep, y0, np.nan)
    for a, b in _runs(keep):  # lissage des suites observées
        if b - a >= SG_WINDOW:
            x[a:b] = savgol_filter(x[a:b], SG_WINDOW, SG_ORDER)
            y[a:b] = savgol_filter(y[a:b], SG_WINDOW, SG_ORDER)
    cx, cy = x.copy(), y.copy()
    source = np.where(keep, 0, 2).astype(np.int8)
    gap_len = np.zeros(n, np.int32)
    gap_pos = np.zeros(n, np.float32)
    for a, b in _runs(~keep):
        gap_len[a:b] = b - a
        gap_pos[a:b] = (np.arange(b - a) + 0.5) / (b - a)
        bridgeable = b - a <= BRIDGE_MAX and a > 0 and b < n and half[a - 1] == half[b]
        if bridgeable and keep[a - 1] and keep[b]:
            lo, hi = max(a - PAD, 0), min(b + PAD, n)
            cx[a:b] = _rts_1d(x[lo:hi])[a - lo:b - lo]
            cy[a:b] = _rts_1d(y[lo:hi])[a - lo:b - lo]
            source[a:b] = 1
    gap_type = np.zeros(n, np.int8)
    gap_type[gap_len > 0] = 1
    gap_type[gap_len > GAP_EDGES[0]] = 2
    gap_type[gap_len > GAP_EDGES[1]] = 3
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(keep, idx, -CAP))
    nxt = np.minimum.accumulate(np.where(keep, idx, 2 * n + CAP)[::-1])[::-1]
    out = pd.DataFrame({
        "ball_cx": cx.astype(np.float32), "ball_cy": cy.astype(np.float32),
        "ball_source": source, "gap_len": gap_len, "gap_pos": gap_pos, "gap_type": gap_type,
        "since_obs": np.minimum(idx - last, CAP).astype(np.int32),
        "until_obs": np.minimum(nxt - idx, CAP).astype(np.int32),
    }, index=df.index)
    ok = np.isfinite(cx)
    vx, vy = _derivative(cx, ok), _derivative(cy, ok)
    ax, ay = _derivative(vx, np.isfinite(vx)), _derivative(vy, np.isfinite(vy))
    out["ball_vx"], out["ball_vy"] = vx.astype(np.float32), vy.astype(np.float32)
    out["ball_speed"] = np.hypot(vx, vy).astype(np.float32)
    out["ball_acc"] = np.hypot(ax, ay).astype(np.float32)
    return out


def with_ball(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df, clean_ball(df)], axis=1)


def bridge_error(df: pd.DataFrame, seed: int = 0, n_test: int = 300) -> dict:
    """Remove real observed windows (5-25 frames), les comble, mesure l'erreur (m)."""
    rng = np.random.default_rng(seed)
    x, y = df["ball_x"].to_numpy(np.float64), df["ball_y"].to_numpy(np.float64)
    ok = remove_outliers(x, y, df["half"].to_numpy())
    full = np.convolve(ok, np.ones(60), "valid") == 60  # fenêtres de 60 frames toutes observées
    starts = np.flatnonzero(full)
    cols = df.columns.get_indexer(["ball_x", "ball_y"])
    errs_b, errs_l = [], []
    for s in rng.choice(starts, size=min(n_test, len(starts)), replace=False):
        ln = int(rng.integers(5, 26))
        sub = df.iloc[s:s + 60].copy()
        sub.iloc[25:25 + ln, cols] = np.nan
        c = clean_ball(sub)
        gx, gy = x[s + 25:s + 25 + ln], y[s + 25:s + 25 + ln]
        errs_b.append(np.hypot(c["ball_cx"].to_numpy()[25:25 + ln] - gx,
                               c["ball_cy"].to_numpy()[25:25 + ln] - gy).mean())
        w = np.linspace(0, 1, ln + 2)[1:-1]
        lx = x[s + 24] + w * (x[s + 25 + ln] - x[s + 24])
        ly = y[s + 24] + w * (y[s + 25 + ln] - y[s + 24])
        errs_l.append(np.hypot(lx - gx, ly - gy).mean())
    return {"n": len(errs_b), "rts_m": float(np.mean(errs_b)), "lineaire_m": float(np.mean(errs_l))}


def summary(df: pd.DataFrame, c: pd.DataFrame) -> dict:
    g = df["graded"].to_numpy()
    src = c["ball_source"].to_numpy()[g]
    seen = np.isfinite(df["ball_x"].to_numpy()[g])
    return {"observe_%": 100 * float((src == 0).mean()), "comble_%": 100 * float((src == 1).mean()),
            "absent_%": 100 * float((src == 2).mean()),
            "retire_(saut/isole)_%": 100 * float((seen & (src != 0)).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bridge-test", action="store_true")
    args = ap.parse_args()
    df = pd.read_parquet(args.parquet)
    c = clean_ball(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    c.to_parquet(args.out)
    print(args.parquet.name, summary(df, c))
    if args.bridge_test:
        print(bridge_error(df))


if __name__ == "__main__":
    main()
