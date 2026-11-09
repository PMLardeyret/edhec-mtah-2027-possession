"""Frontières précises (V2, M3) : coudes de trajectoire du ballon et changement de moyenne (AIC).

Ne crée aucune frontière : déplace celles que le décodeur a posées. Filtrage de tendance ℓ1 (Kim, Koh, Boyd
2009), min 1/2 ||x - y||^2 + lam ||D2 x||_1 par suite de frames où le ballon est observé, résolu par ADMM
(cvxpy et OSQP absents : voir docs/decisions.md). Un coude est un pic de |D2 x|, retenu s'il a un joueur à
moins de PLAYER_M ; sinon, pour une frontière H<->A, minimum du critère AIC (Maeda) de l'indice d'équipe.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

import common

LAM = 10.0  # 24 à 29 coudes par minute de ballon observé on the Metrica matches
RHO = 1.0
MIN_RUN = 10  # frames observées minimales pour filtrer
EPS = 0.03  # m/frame^2 : courbure minimale d'un coude
PLAYER_M = 1.5
SEARCH = 25  # frames : +-1 s
AIC_WIN = 38  # frames : +-1,5 s
USE_AIC = True  # False : seuls les coudes déplacent les frontières


def trend_filter(y: np.ndarray, lam: float = LAM, rho: float = RHO, iters: int = 150) -> np.ndarray:
    """Filtre de tendance ℓ1 d'ordre 2 d'une série 1D par ADMM."""
    n = len(y)
    d2 = sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc")
    lu = splu((sp.identity(n, format="csc") + rho * (d2.T @ d2)).tocsc())
    z, u, x = np.zeros(n - 2), np.zeros(n - 2), y.copy()
    for _ in range(iters):
        x = lu.solve(y + rho * (d2.T @ (z - u)))
        v = d2 @ x + u
        z = np.sign(v) * np.maximum(np.abs(v) - lam / rho, 0)
        u = v - z
    return x


def kink_frames(bx: np.ndarray, by: np.ndarray, observed: np.ndarray, lam: float = LAM, eps: float = EPS) -> np.ndarray:
    """Indices des coudes (pics de la courbure ℓ1) sur toutes les suites observées."""
    e = np.diff(np.concatenate(([0], observed.astype(np.int8), [0])))
    out = []
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1), strict=True):
        if b - a < MIN_RUN:
            continue
        x, y = trend_filter(bx[a:b], lam), trend_filter(by[a:b], lam)
        c = np.hypot(np.diff(x, 2), np.diff(y, 2))  # courbure au centre i+1
        act = c > eps
        s = np.flatnonzero(np.diff(np.concatenate(([0], act.astype(np.int8), [0]))) == 1)
        t = np.flatnonzero(np.diff(np.concatenate(([0], act.astype(np.int8), [0]))) == -1)
        out += [a + 1 + s_ + int(np.argmax(c[s_:t_])) for s_, t_ in zip(s, t, strict=True)]
    return np.array(out, int)


def kinks_per_minute(kinks: np.ndarray, observed: np.ndarray) -> float:
    return len(kinks) / max(observed.sum() / common.FPS / 60, 1e-9)


def aic_change(series: np.ndarray, centre: int, win: int = AIC_WIN) -> int:
    """Instant du minimum AIC de Maeda pour un changement de moyenne de `series` dans centre +- win."""
    a, b = max(centre - win, 0), min(centre + win, len(series))
    s = series[a:b]
    ok = np.isfinite(s)
    if ok.sum() < 20:
        return centre
    idx = np.flatnonzero(ok)
    s = s[idx]
    best, arg = np.inf, centre
    for k in range(5, len(s) - 5):
        aic = k * np.log(s[:k].var() + 1e-6) + (len(s) - k) * np.log(s[k:].var() + 1e-6)
        if aic < best:
            best, arg = aic, a + int(idx[k])
    return arg


def refine(codes: np.ndarray, kinks: np.ndarray, near_player: np.ndarray, team_index: np.ndarray) -> tuple[np.ndarray, int]:
    """Déplace chaque frontière de `codes` vers le coude le plus proche (joueur à < PLAYER_M) dans +-SEARCH,
    à défaut vers le minimum AIC de `team_index` (frontière H<->A seulement). Retourne (codes, nb déplacées)."""
    out = codes.copy()
    cut = np.flatnonzero((codes[1:] != codes[:-1]) & (codes[1:] != common.X) & (codes[:-1] != common.X)) + 1
    good = kinks[near_player[kinks]] if len(kinks) else kinks
    moved = 0
    for t in cut:
        new = t
        near = good[np.abs(good - t) <= SEARCH]
        if len(near):
            new = int(near[np.argmin(np.abs(near - t))])
        elif USE_AIC and {codes[t - 1], codes[t]} == {common.H, common.A}:
            new = aic_change(team_index, t)
        if new == t:
            continue
        lo, hi = (new, t) if new < t else (t, new)
        out[lo:hi] = codes[t] if new < t else codes[t - 1]
        moved += 1
    return out, moved
