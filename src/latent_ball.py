"""Balle latente (V2, M1) par vote de rayons (vision du regard, Park, Jiang, Nguyen et al., NIPS 2012).

Chaque joueur qui court est un capteur qui pointe vers le ballon : b_t est le point le plus proche, aux moindres
carrés pondérés, des demi-droites (position, direction de course). Poids = vitesse x 1[vitesse > 1,5 m/s],
3 itérations IRLS (Huber, c = 3 m), seuls les rayons dirigés vers l'estimation sont conservés. Un ridge faible vers
le barycentre des coureurs rend le système résoluble quand les rayons sont presque parallèles ; la confiance
(kappa, part de poids des rayons à < 3 m) dit alors qu'il est mal observé. Les gardiens ne sont pas identifiés
dans les données (pistes anonymes) : pas de poids par rôle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import features
import load_tracking as lt

MIN_SPEED = 1.5  # m/s
HUBER_C = 3.0  # m
ITERS = 3
RIDGE = 0.02  # part de la trace de N ajoutée vers le barycentre
FLUX_RADIUS = 15.0  # m
FUSE_MIN_GAP = 25  # frames : au-delà d'1 s de trou, le ballon fusionné remplace l'observé pour les distances


def players(df: pd.DataFrame) -> tuple[np.ndarray, ...]:
    x, y = lt.track_array(df, "x"), lt.track_array(df, "y")
    team = lt.track_array(df, "team")
    vx, vy, sp = features.player_speed(x, y)
    return x, y, vx, vy, sp, team


def estimate(x, y, vx, vy, sp, w_extra: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """b̂ (n, 2), kappa, confiance, nombre de rayons utilisés, par frame."""
    n = len(x)
    ok = np.isfinite(sp) & (sp > MIN_SPEED) & np.isfinite(x)
    ux, uy = np.where(ok, vx / np.where(ok, sp, 1), 0.0), np.where(ok, vy / np.where(ok, sp, 1), 0.0)
    px, py = np.where(ok, x, 0.0), np.where(ok, y, 0.0)
    w0 = np.where(ok, sp, 0.0) * (1.0 if w_extra is None else w_extra)
    w = w0.copy()
    bx, by = np.full(n, np.nan), np.full(n, np.nan)
    kappa, conf = np.full(n, np.nan), np.zeros(n)
    for it in range(ITERS + 1):
        # N = sum w (I - u u^T) ; rhs = sum w (I - u u^T) p
        a11, a12, a22 = (w * (1 - ux * ux)).sum(1), (w * (-ux * uy)).sum(1), (w * (1 - uy * uy)).sum(1)
        r1 = (w * ((1 - ux * ux) * px - ux * uy * py)).sum(1)
        r2 = (w * (-ux * uy * px + (1 - uy * uy) * py)).sum(1)
        sw = w.sum(1)
        cx, cy = (w * px).sum(1) / np.maximum(sw, 1e-9), (w * py).sum(1) / np.maximum(sw, 1e-9)
        eps = RIDGE * (a11 + a22) / 2
        a11, a22 = a11 + eps, a22 + eps
        r1, r2 = r1 + eps * cx, r2 + eps * cy
        det = a11 * a22 - a12 * a12
        good = (sw > 0) & (det > 1e-9)
        bx = np.where(good, (a22 * r1 - a12 * r2) / np.where(good, det, 1), np.nan)
        by = np.where(good, (-a12 * r1 + a11 * r2) / np.where(good, det, 1), np.nan)
        tr, disc = a11 + a22, np.sqrt(np.maximum((a11 - a22) ** 2 / 4 + a12 ** 2, 0))
        kappa = np.where(good, (tr / 2 + disc) / np.maximum(tr / 2 - disc, 1e-9), np.nan)
        dx, dy = bx[:, None] - px, by[:, None] - py
        res = np.abs(dx * uy - dy * ux)  # distance du point b̂ au rayon
        ahead = dx * ux + dy * uy > 0
        hub = np.where(res <= HUBER_C, 1.0, HUBER_C / np.maximum(res, 1e-9))
        w = w0 * hub * ahead
        conf = np.where(sw > 0, (w0 * (res < 3.0) * ahead).sum(1) / np.maximum(w0.sum(1), 1e-9), 0.0)
    return {"bx": bx, "by": by, "kappa": kappa, "conf": conf, "n_rays": (w > 0).sum(1)}


def flux(x, y, vx, vy, team, bx, by) -> tuple[np.ndarray, np.ndarray]:
    """Flux radial entrant moyen (m/s, positif = se rapproche du ballon) des joueurs de chaque équipe dans
    FLUX_RADIUS du ballon (bx, by)."""
    dx, dy = bx[:, None] - x, by[:, None] - y
    d = np.hypot(dx, dy)
    inward = (vx * dx + vy * dy) / np.maximum(d, 1e-6)
    out = []
    for k in (0, 1):
        m = (team == k) & np.isfinite(inward) & (d < FLUX_RADIUS)
        out.append(np.where(m.sum(1) > 0, np.where(m, inward, 0).sum(1) / np.maximum(m.sum(1), 1), np.nan))
    return out[0], out[1]


def latent_features(f: pd.DataFrame, dfb: pd.DataFrame) -> pd.DataFrame:
    """Colonnes `lb_*` : b̂, confiance, flux par équipe, et distances à la balle fusionnée (observée, sinon b̂)."""
    x, y, vx, vy, sp, team = players(dfb)
    est = estimate(x, y, vx, vy, sp)
    ox, oy = dfb["ball_cx"].to_numpy(), dfb["ball_cy"].to_numpy()
    long_gap = ~np.isfinite(ox) | (dfb["gap_len"].to_numpy() > FUSE_MIN_GAP)
    fx = np.where(long_gap, est["bx"], ox)
    fy = np.where(long_gap, est["by"], oy)
    fh, fa = flux(x, y, vx, vy, team, fx, fy)
    present = np.isfinite(x) & np.isfinite(y)
    cols = {"lb_kappa": np.log1p(est["kappa"]), "lb_conf": est["conf"], "lb_rays": est["n_rays"],
            "lb_flux_h": fh, "lb_flux_a": fa, "lb_flux_diff": fh - fa}
    for t, k in features.TEAMS:
        d = np.where(present & (team == k), np.hypot(x - fx[:, None], y - fy[:, None]), np.inf).min(1)
        cols[f"lb_{t}_d1"] = np.where(np.isfinite(d), d, np.nan)
    cols["lb_diff_d1"] = cols["lb_h_d1"] - cols["lb_a_d1"]
    g = pd.DataFrame({k: np.asarray(v, np.float32) for k, v in cols.items()}, index=f.index)
    for c in ("lb_flux_diff", "lb_diff_d1", "lb_conf"):
        for w in features.WINDOWS:
            g[f"{c}_m{w}"] = g[c].rolling(w, center=True, min_periods=1).mean().astype(np.float32)
    return pd.concat([f, g], axis=1)


