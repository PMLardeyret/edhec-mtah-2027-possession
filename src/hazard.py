"""Pénalité de changement apprise (V2, M4) : hasards en temps discret dépendant de covariables.

Remplace les coûts constants cost_ha / cost_d du Viterbi par un coût par pas de 0,2 s : -log(hasard de la
transition) + log(probabilité de rester), mis à l'échelle par `stick` (constante d'auto-transition, réglée sur le
dev). Trois hasards sont appris sur les pseudo-labels des matchs d'entraînement, par régression logistique
(lifelines n'est pas installé : le Cox à covariables variables est remplacé par son analogue en temps discret,
une ligne par pas) : perte de balle (H<->A), arrêt (jeu -> D) et reprise (D -> jeu).
Covariables : moyennes sur le pas, retardées de LAG pas (0,4 s >= 0,3 s demandées) contre l'endogénéité.
Only open (DFL, Metrica) matches are used for learning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import common

STEP = 5  # frames par pas, comme decode.STEP
LAG = 2  # pas de retard sur les covariables
H, A, D = common.H, common.A, common.D
SHARED = ["ball_present", "ball_speed", "gap_len", "all_mean_speed", "edge_min", "ball_in_play_area", "n10"]
ORIENTED = ["own_d1", "opp_d1", "own_n5", "opp_n5"]  # du point de vue de l'équipe qui a le ballon
MIN_LAMBDA, MAX_LAMBDA = 1e-4, 0.5


def covariates(f: pd.DataFrame) -> dict[str, np.ndarray]:
    """Matrices par frame (n, k) : partagées, orientées pour H, orientées pour A."""
    shared = pd.DataFrame({
        "ball_present": f["ball_present"], "ball_speed": f["ball_speed"].fillna(0).clip(0, 40),
        "gap_len": np.log1p(f["gap_len"]), "all_mean_speed": f["all_mean_speed"].fillna(0),
        "edge_min": np.minimum(f["ball_edge_touch"], f["ball_edge_goal"]).fillna(0).clip(0, 30),
        "ball_in_play_area": f["ball_in_play_area"], "n10": f["h_n10"] + f["a_n10"]})
    out = {"shared": shared[SHARED].to_numpy(np.float32)}
    for name, own, opp in (("H", "h", "a"), ("A", "a", "h")):
        out[name] = np.column_stack([
            f[f"{own}_d1"].fillna(40).clip(0, 40), f[f"{opp}_d1"].fillna(40).clip(0, 40),
            f[f"{own}_n5"], f[f"{opp}_n5"]]).astype(np.float32)
    return out


def _lagged_blocks(x: np.ndarray, a: int, b: int) -> np.ndarray:
    n = (b - a + STEP - 1) // STEP
    pad = np.vstack([x[a:b], np.repeat(x[b - 1:b], n * STEP - (b - a), 0)]) if n * STEP > b - a else x[a:b]
    blk = np.nanmean(pad.reshape(n, STEP, -1), 1)
    return np.vstack([np.repeat(blk[:1], LAG, 0), blk[:-LAG]]) if len(blk) > LAG else np.repeat(blk[:1], len(blk), 0)


def _design(cov: dict, a: int, b: int, origin: np.ndarray) -> dict[str, np.ndarray]:
    """Matrices de covariables par pas (retardées) pour une suite [a, b) ; `origin` = état au pas précédent."""
    sh = _lagged_blocks(cov["shared"], a, b)
    ori = np.where((origin == H)[:, None], _lagged_blocks(cov["H"], a, b), _lagged_blocks(cov["A"], a, b))
    return {"play": np.hstack([sh, ori]), "dead": sh}


def _rows(cov: dict, lab: np.ndarray, graded: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Lignes (X, y) des trois hasards à partir des pseudo-labels : un pas = une ligne."""
    rows = {"loss": ([], []), "die": ([], []), "wake": ([], [])}
    e = np.diff(np.concatenate(([0], graded.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1), strict=True):
        nb = (b - a + STEP - 1) // STEP
        blab = lab[a:b][STEP // 2::STEP]
        if nb < 3 or len(blab) == 0:
            continue
        blab = np.concatenate((blab, np.repeat(blab[-1:], nb - len(blab))))
        origin = np.concatenate(([blab[0]], blab[:-1]))
        des = _design(cov, a, b, origin)
        n = len(blab)
        play = np.isin(origin, (H, A))[1:n]
        x_play, x_dead = des["play"][:n][1:][play], des["dead"][:n][1:][~play & (origin[1:] == D)]
        y = blab[1:]
        rows["loss"][0].append(x_play)
        rows["loss"][1].append(y[play] == 1 - origin[1:][play])
        rows["die"][0].append(x_play)
        rows["die"][1].append(y[play] == D)
        rows["wake"][0].append(x_dead)
        rows["wake"][1].append(np.isin(y[origin[1:] == D], (H, A)))
    return {k: (np.vstack(v[0]), np.concatenate(v[1])) for k, v in rows.items()}


class Hazards:
    """Trois régressions logistiques standardisées et leur évaluation (concordance hors match)."""

    def __init__(self, c: float = 1.0):
        self.c, self.m = c, {}

    def fit(self, cov_lab: list[tuple[dict, np.ndarray, np.ndarray]]) -> Hazards:
        parts = [_rows(*x) for x in cov_lab]
        for k in ("loss", "die", "wake"):
            X = np.vstack([p[k][0] for p in parts])
            y = np.concatenate([p[k][1] for p in parts])
            mu, sd = X.mean(0), X.std(0) + 1e-6
            self.m[k] = (mu, sd, LogisticRegression(C=self.c, max_iter=500).fit((X - mu) / sd, y))
        return self

    def concordance(self, cov_lab: list[tuple[dict, np.ndarray, np.ndarray]]) -> dict:
        parts = [_rows(*x) for x in cov_lab]
        out = {}
        for k in ("loss", "die", "wake"):
            X = np.vstack([p[k][0] for p in parts])
            y = np.concatenate([p[k][1] for p in parts])
            mu, sd, clf = self.m[k]
            out[k] = round(float(roc_auc_score(y, clf.predict_proba((X - mu) / sd)[:, 1])), 3)
        return out

    def _lam(self, k: str, X: np.ndarray) -> np.ndarray:
        mu, sd, clf = self.m[k]
        return np.clip(clf.predict_proba((X - mu) / sd)[:, 1], MIN_LAMBDA, MAX_LAMBDA)

    def cost(self, cov: dict, a: int, b: int, stick: float) -> np.ndarray:
        """Coûts (n, 3, 3) d'entrer au pas t depuis l'état i vers j (diagonale ignorée par le Viterbi)."""
        n = (b - a + STEP - 1) // STEP
        out = np.zeros((n, 3, 3))
        for o, name in ((H, "H"), (A, "A")):
            origin = np.full(n, o)
            x = _design(cov, a, b, origin)["play"]
            loss, die = self._lam("loss", x), self._lam("die", x)
            stay = np.clip(1 - loss - die, 1e-3, 1)
            out[:, o, 1 - o] = np.maximum(np.log(stay) - np.log(loss), 0)
            out[:, o, D] = np.maximum(np.log(stay) - np.log(die), 0)
        xd = _design(cov, a, b, np.full(n, D))["dead"]
        wake = self._lam("wake", xd)
        c = np.maximum(np.log(1 - wake) - np.log(wake / 2), 0)  # reprise : H ou A à parts égales
        out[:, D, H] = out[:, D, A] = c
        return stick * out
