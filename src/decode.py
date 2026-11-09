"""Décodeur (J8) : de probabilités par frame à une suite de possessions.

Viterbi à 3 états (H, A, D) avec durées minimales (états en chaîne) et coût de changement, comparé au
vote majoritaire et à PELT (ruptures). Tout travaille par bloc de STEP frames puis remonte à la frame.
Les mi-temps (runs notés) sont décodées séparément. Le décodeur ne produit jamais X : un X équivaut à
prolonger l'étiquette précédente, ce que les durées minimales font déjà.
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import ruptures as rpt

import common
import model
from score import spell_f1

STEP = 5  # frames par pas de décodage (0,2 s)
CONFIG = common.ROOT / "configs" / "decode.json"
DEFAULTS = {"min_poss_s": 4.0, "min_dead_s": 2.0, "cost_ha": 8.0, "cost_d": 5.0, "alpha": 1.0, "bias_d": 0.0}


def _blocks(p: np.ndarray, step: int = STEP) -> np.ndarray:
    n = len(p) // step * step
    b = p[:n].reshape(-1, step, p.shape[1]).mean(1)
    if len(p) > n:
        b = np.vstack([b, p[n:].mean(0, keepdims=True)])
    return b


def viterbi(logp: np.ndarray, min_len: tuple[int, int, int], cost: np.ndarray) -> np.ndarray:
    """Chemin d'états (n,) de score maximal. min_len[c] (>= 2) = pas minimum dans l'état c, représenté
    par une chaîne de min_len[c] sous-états ; cost[c1, c2] = pénalité de passage de c1 à c2, ou (n, 3, 3)
    pour une pénalité qui dépend du pas (M4, hasards appris : cost[t] = coût d'entrer au pas t)."""
    n = len(logp)
    m = np.maximum(np.asarray(min_len), 2)
    start = np.concatenate(([0], np.cumsum(m)[:-1]))
    end = start + m - 1
    cls = np.repeat(np.arange(3), m)
    pen = np.where(np.eye(3, dtype=bool), np.inf, cost)  # (origine, destination), ou (n, origine, destination)
    sc = np.full(m.sum(), -np.inf)
    sc[start] = logp[0]
    bp_stay = np.zeros((n, 3), bool)  # dernier sous-état : on y reste (sinon on vient de end-1)
    bp_src = np.zeros((n, 3), np.int8)  # classe d'origine d'une entrée
    for t in range(1, n):
        ent = sc[end][:, None] - (pen[t] if pen.ndim == 3 else pen)
        src = ent.argmax(0)
        enter = ent[src, np.arange(3)]
        stay, adv = sc[end], sc[end - 1]
        new = np.empty_like(sc)
        new[1:] = sc[:-1]
        new[end] = np.maximum(stay, adv)
        new[start] = enter
        bp_stay[t], bp_src[t] = stay >= adv, src
        sc = new + logp[t][cls]
    state = int(np.argmax(sc[end]))
    k = int(m[state]) - 1
    path = np.zeros(n, np.uint8)
    for t in range(n - 1, -1, -1):
        path[t] = state
        if t == 0:
            break
        if k == m[state] - 1 and bp_stay[t, state]:
            continue
        if k == 0:
            state = int(bp_src[t, state])
            k = int(m[state]) - 1
        else:
            k -= 1
    return path


def decode_viterbi(p: np.ndarray, graded: np.ndarray, cost_fn=None, **kw) -> np.ndarray:
    """`cost_fn(a, b)` (option) donne les coûts (n, 3, 3) par pas de la suite [a, b) : pénalité apprise (M4)."""
    par = DEFAULTS | kw
    ml = (round(par["min_poss_s"] * common.FPS / STEP), round(par["min_poss_s"] * common.FPS / STEP),
          round(par["min_dead_s"] * common.FPS / STEP))
    cost = np.array([[0, par["cost_ha"], par["cost_d"]], [par["cost_ha"], 0, par["cost_d"]],
                     [par["cost_d"], par["cost_d"], 0]], float)
    shift = np.array([0.0, 0.0, par["bias_d"]])  # V3 : biais ajouté au log-score de D (après alpha)
    return _per_run(p, graded, lambda b, a=0, e=0: viterbi(
        par["alpha"] * np.log(np.clip(b, 1e-4, 1)) + shift, tuple(max(1, x) for x in ml),
        cost if cost_fn is None else cost_fn(a, e)), ranged=True)


def decode_majority(p: np.ndarray, graded: np.ndarray, win: int = 125) -> np.ndarray:
    return np.where(graded, model.argmax_smoothed(p, win), common.X).astype(np.uint8)


def decode_pelt(p: np.ndarray, graded: np.ndarray, pen: float = 5.0, min_s: float = 3.0) -> np.ndarray:
    def seg(b: np.ndarray) -> np.ndarray:
        bk = rpt.Pelt(model="l2", min_size=max(2, round(min_s)), jump=1).fit(b).predict(pen=pen)
        out = np.zeros(len(b), np.uint8)
        a = 0
        for e in bk:
            out[a:e] = b[a:e].mean(0).argmax()
            a = e
        return out
    return _per_run(p, graded, seg, step=common.FPS)  # blocs de 1 s : PELT est quadratique


def _per_run(p: np.ndarray, graded: np.ndarray, fn, step: int = STEP, ranged: bool = False) -> np.ndarray:
    out = np.full(len(p), common.X, np.uint8)
    e = np.diff(np.concatenate(([0], graded.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1), strict=True):
        path = fn(_blocks(p[a:b], step), a, b) if ranged else fn(_blocks(p[a:b], step))
        out[a:b] = np.repeat(path, step)[: b - a]
    return out


def load_oof(source: str) -> dict:
    d = {}
    for m in model.MATCHES:
        z = np.load(common.ROOT / "work" / "sample" / "oof" / f"{m}.npz")
        f, _ = model.feature_frame(m, model.EVAL_SEED, False, False)
        graded = f["graded"].to_numpy()
        gt, runs = model.eval_gt(m, graded)
        d[m] = (z[source], graded, gt, runs)
    return d


def mean_score(oof: dict, decoder, **kw) -> tuple[float, list[float]]:
    s = [spell_f1(gt, decoder(p, g, **kw), runs) for p, g, gt, runs in oof.values()]
    return float(np.mean(s)), s


def tune(oof: dict, grid: dict) -> tuple[dict, float, list]:
    best, best_s, best_f = None, -1.0, []
    keys = list(grid)
    for vals in itertools.product(*grid.values()):
        kw = dict(zip(keys, vals, strict=True))
        s, f = mean_score(oof, decode_viterbi, **kw)
        if s > best_s:
            best, best_s, best_f = kw, s, f
    return best, best_s, best_f


STEPS = {"min_poss_s": (0.5, 0.5), "min_dead_s": (2.0, 1.0), "cost_ha": (0.25, 0.0), "cost_d": (0.5, 0.0),
         "alpha": (0.5, 0.5)}  # pas de la montée par coordonnée, borne basse du domaine


def refine(oof: dict, start: dict, max_rounds: int = 30) -> tuple[dict, float, list, int]:
    """Montée par coordonnée jusqu'à ce que les deux voisins de chaque paramètre soient moins bons :
    l'optimum est alors intérieur au domaine exploré (sauf borne physique, ex. un coût >= 0).
    Retourne (réglage, score, scores par pli, nombre d'évaluations)."""
    cur = dict(start)
    best, folds = mean_score(oof, decode_viterbi, **cur)
    n_eval = 1
    for _ in range(max_rounds):
        moved = False
        for k, (step, low) in STEPS.items():
            for sign in (1, -1):
                while True:
                    v = round(cur[k] + sign * step, 4)
                    if v < low:
                        break
                    s, f = mean_score(oof, decode_viterbi, **(cur | {k: v}))
                    n_eval += 1
                    if s <= best + 1e-9:
                        break
                    cur, best, folds, moved = cur | {k: v}, s, f, True
        if not moved:
            break
    return cur, best, folds, n_eval


def plateau(oof: dict, centre: dict) -> list[float]:
    """Scores des 3^5 réglages voisins (un pas de chaque côté) : montre la planéité de l'optimum."""
    axes = []
    for k, (step, low) in STEPS.items():
        axes.append([(k, v) for v in (centre[k] - step, centre[k], centre[k] + step) if v >= low])
    return [mean_score(oof, decode_viterbi, **dict(c))[0] for c in itertools.product(*axes)]


def main_refine(save: bool) -> None:
    oof = load_oof("v2")
    start = json.loads(CONFIG.read_text(encoding="utf-8"))
    start = {k: v for k, v in start.items() if k in STEPS}
    best, s, f, n = refine(oof, start)
    print("optimum", best, round(s, 4), [round(x, 3) for x in f], "evaluations", n, flush=True)
    pl = plateau(oof, best)
    print("voisinage (+-1 pas) : n", len(pl), "min", round(min(pl), 4), "médiane", round(float(np.median(pl)), 4),
          "max", round(max(pl), 4), flush=True)
    if save:
        CONFIG.write_text(json.dumps(best | {"score_dev": round(s, 4)}, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="écrit configs/decode.json (v2)")
    ap.add_argument("--refine", action="store_true", help="montée par coordonnée depuis configs/decode.json")
    ap.add_argument("--sources", nargs="+", default=["v1", "v2"])
    args = ap.parse_args()
    if args.refine:
        return main_refine(args.save)
    grid = {"min_poss_s": [1.0, 1.5, 2.5], "min_dead_s": [3.0, 4.0, 6.0], "cost_ha": [0.5, 1.0, 2.0],
            "cost_d": [2.0, 3.0, 5.0, 8.0], "alpha": [2.0, 3.0, 4.0]}
    best_v2 = None
    for src in args.sources:
        oof = load_oof(src)
        print(f"== {src}")
        for w in (50, 125, 250):
            s, f = mean_score(oof, decode_majority, win=w)
            print(f"vote {w:>3}      {s:.4f}", [round(x, 3) for x in f])
        pen = max(((mean_score(oof, decode_pelt, pen=pn, min_s=ms), pn, ms)
                   for pn in (0.3, 1.0, 2.0) for ms in (3.0, 6.0)), key=lambda r: r[0][0])
        print(f"PELT pen={pen[1]} min={pen[2]}  {pen[0][0]:.4f}", [round(x, 3) for x in pen[0][1]])
        best, s, f = tune(oof, grid)
        print(f"Viterbi {best}  {s:.4f}", [round(x, 3) for x in f])
        if src == "v2":
            best_v2 = best | {"score_dev": round(s, 4)}
    if args.save:
        CONFIG.parent.mkdir(exist_ok=True)
        CONFIG.write_text(json.dumps(best_v2, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
