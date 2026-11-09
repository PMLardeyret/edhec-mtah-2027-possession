"""Final labelling of the match, strict variant s; no score of any kind is computed.

    V23C=s python src/label_match.py --train --out submissions/s4.csv        # train 5 models, weights in models/v23s4/
    V23C=s WEIGHTS_DIR=models/v23s4 python src/label_match.py --out ...      # CPU inference from the text weights
Choices read from configs/v23c_<name>.json: feature modules, stick, short_poss, bias b (median of the IDSSE folds). Chain: hazards learned on DFL + Metrica, Viterbi, short_poss, restart anchors (provided Smart Tagging, fixed
rules), `out` rule. Writes reports/v23c/final_<name>_<stem>.json with aggregates of the labelled match only (share of D, number of spells).
"""

from __future__ import annotations

import os

VAR = os.environ.get("V23C", "")
if VAR != "s":
    raise SystemExit("V23C must be s")
NAME = os.environ.get("V23C_NAME") or VAR  # s4: configuration, weights, probabilities and outputs
if NAME[0] != VAR:
    raise SystemExit("V23C_NAME must start with the letter of V23C")
os.environ["V21"] = "feat"  # stop features (read at import of model; spawned workers redo these lines)
os.environ["ALIGNED_LABELS"] = "aligned"
os.environ["LABEL_ALIGN"] = "10"  # DFL labels aligned at 1.0 s + 2 touch-lines
os.environ.setdefault("MTAH_DATA_DIR", str(__import__("pathlib").Path(__file__).resolve().parents[1] / "private_data"))

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import common
import hazard
import memguard
import model
import run_real
import setpiece_anchor as sa
import v3_eval
import train_data as vd
import final_model as vf
from load_tagging import load_tagging
from make_submission import write_submission

common.ensure_paths()
import games

CFG = json.loads((common.ROOT / "configs" / f"v23c_{NAME}.json").read_text(encoding="utf-8")) if (common.ROOT / "configs" / f"v23c_{NAME}.json").exists() else {}
vf.EXTRA = tuple(CFG.get("modules", ("stop", "latent")))
WEIGHTS = common.ROOT / "models" / f"v23{NAME}"


def _p_file(k: int) -> Path:
    return common.ROOT / "work" / "real" / f"p_v23{NAME}_{k}.npy"


vf.p_file = _p_file


def sha256sums() -> None:
    lines = [f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}" for f in sorted(WEIGHTS.glob("model_*.txt"))]
    with open(WEIGHTS / "SHA256SUMS", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=4, help="LightGBM threads for training (the s4 weights were trained with 1)")
    ap.add_argument("--out", type=Path, default=common.ROOT / "submissions" / f"v23{NAME}.csv")
    args = ap.parse_args()
    if not CFG:
        sys.exit(f"configs/v23c_{NAME}.json absent : lancer v23c_run.py (étape choix)")
    K = vf.K
    model.EXTRA = vf.EXTRA
    if args.train:
        os.environ["WEIGHTS_SAVE"] = str(WEIGHTS)
        run_real.DETERMINISTIC = {**run_real.DETERMINISTIC, "n_jobs": args.n_jobs}  # lu par vf.train ; un seul modèle à la fois
        for k in range(K):
            if not memguard.wait_free(label=f"modèle {k}"):
                sys.exit(f"non conclu : mémoire libre insuffisante avant le modèle {k}")
            vf.train(k)
            print("modèle", k, flush=True)
        sha256sums()
    else:
        for k in range(K):
            vf.infer(k, Path(os.environ.get("WEIGHTS_DIR", str(WEIGHTS))))
    cfg4 = json.loads((common.ROOT / "configs" / "decode.json").read_text(encoding="utf-8"))  # réglage de décodage conforme (dev Metrica, re-réglage DFL sans gain)
    df = run_real.real_tracking(None)
    p = np.mean([np.load(vf.p_file(k)) for k in range(K)], axis=0)
    tr = []
    for m in vf.NAMES:
        f, lab = vd.feature_frame(m, 0, False, False)
        tr.append((hazard.covariates(f), lab, f["graded"].to_numpy()))
    hz = hazard.Hazards().fit(tr)
    cov = hazard.covariates(model.build(df))
    cost_fn = lambda a, b: hz.cost(cov, a, b, CFG["stick"])
    b = CFG["b"] if CFG.get("b") is not None else v3_eval.solve_bias(df, p, cfg4, float(CFG["b_target"]), cost_fn)
    frames = sa.set_piece_frames(load_tagging(Path(games.asset_paths()["smart_tagging"])))
    codes = v3_eval.real_decode(df, p, cfg4, b, cost_fn, short_poss=CFG["short_poss"], setpiece=frames, rules=("out",))
    write_submission(codes, args.out)
    out = {"variante": NAME, "b": b, "cfg": CFG, "reel": run_real.aggregates(codes, df["graded"].to_numpy()),
           "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()[:16]}
    d = common.ROOT / "reports" / "v23c"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"final_{NAME}_{args.out.stem}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
