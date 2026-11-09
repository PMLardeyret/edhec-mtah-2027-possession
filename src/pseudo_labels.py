"""H/A/D pseudo-labels per frame, derived from the event data of the Metrica matches.

Output: an array of codes (common.H/A/D) per tracking frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import common

PLAYER_ACTIONS = {"PASS", "SHOT", "CARRY"}  # une action de joueur met fin à une phase morte
OWNER_EVENTS = PLAYER_ACTIONS | {"RECOVERY", "SET PIECE"}  # donnent le ballon à leur équipe
RESTART_DELAY_S = 26.0  # s : une action de joueur plus tard vaut remise en jeu non annotée (réglé : jeu 1 sous 50 % de D)
DEAD_TRIGGERS = {"BALL OUT", "FAULT RECEIVED"}


def subtype_name(st) -> str:
    """Sous-types du JSON : un dict, une liste de dicts (jointe par « - ») ou rien. V3 : les listes étaient
    perdues, donc les tirs hors cadre ou buts du jeu 3 ne déclenchaient aucun arrêt."""
    if isinstance(st, dict):
        return st.get("name", "")
    return "-".join(d.get("name", "") for d in st) if isinstance(st, list) else ""


def load_events(match: str, sample_dir: Path = common.SAMPLE_DIR) -> pd.DataFrame:
    """Événements normalisés : frame (index tracking), period, side (0 dom., 1 ext.), type, subtype."""
    d = sample_dir / match
    if match == "Sample_Game_3":
        js = json.loads((d / f"{match}_events.json").read_text(encoding="utf-8"))["data"]
        rows = [{
            "frame": e["start"]["frame"] - 1, "period": e["period"],
            "side": 0 if e["team"]["id"] == "FIFATMA" else 1,
            "type": e["type"]["name"],
            "subtype": subtype_name(e["subtypes"]),
        } for e in js]
        ev = pd.DataFrame(rows)
    else:
        raw = pd.read_csv(d / f"{match}_RawEventsData.csv")
        ev = pd.DataFrame({
            "frame": raw["Start Frame"] - 1, "period": raw["Period"],
            "side": (raw["Team"] == "Away").astype(int), "type": raw["Type"],
            "subtype": raw["Subtype"].fillna(""),
        })
    return ev.sort_values("frame", kind="stable").reset_index(drop=True)


def is_dead_trigger(row) -> bool:
    t, s = row.type, str(row.subtype)
    if t in DEAD_TRIGGERS:
        return True
    if t == "CHALLENGE" and "FAULT" in s:
        return True
    if t == "BALL LOST" and s == "OFFSIDE":
        return True
    return bool(t == "SHOT" and ("GOAL" in s or "OUT" in s))


def labels_from_events(ev: pd.DataFrame, n_frames: int) -> np.ndarray:
    """Machine à états : la dernière équipe à agir garde le ballon ; arrêt de jeu = D jusqu'au SET PIECE
    suivant (V3), ou à la première action de joueur RESTART_DELAY_S après l arrêt. Avant, une action de joueur ne le termine pas : dans le jeu 3, la passe ou la
    conduite qui sort le ballon est horodatée à la même frame que BALL OUT et rouvrait l'arrêt aussitôt."""
    lab = np.full(n_frames, common.D, np.uint8)
    state, start = common.D, 0
    prev_period, dead, t_dead = None, False, 0
    for row in ev.itertuples():
        f = min(max(int(row.frame), 0), n_frames - 1)
        new = state
        if prev_period is not None and row.period != prev_period:
            new, dead, t_dead = common.D, True, f  # changement de mi-temps : mort jusqu'au coup d'envoi
            lab[start:f] = state
            state, start = new, f
        prev_period = row.period
        if is_dead_trigger(row):
            t_dead = t_dead if dead else f
            new, dead = common.D, True
        elif row.type in OWNER_EVENTS and (row.type == "SET PIECE" or (state != common.D and not dead)
                                           or (dead and row.type in PLAYER_ACTIONS
                                               and f - t_dead >= RESTART_DELAY_S * common.FPS)):
            new, dead = common.H if row.side == 0 else common.A, False
        if new != state:
            lab[start:f] = state
            state, start = new, f
    lab[start:] = state
    return lab


def pseudo_labels(match: str, n_frames: int) -> np.ndarray:
    return labels_from_events(load_events(match), n_frames)


def stats(lab: np.ndarray, graded: np.ndarray) -> dict:
    from score import n_spells
    g = lab[graded]
    return {
        "frames": int(graded.sum()),
        "pct_H": 100 * float((g == common.H).mean()), "pct_A": 100 * float((g == common.A).mean()),
        "pct_D": 100 * float((g == common.D).mean()),
        "possessions_et_duree_moy_s": n_spells(g),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=common.ROOT / "work" / "sample" / "labels")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for m in ("Sample_Game_1", "Sample_Game_2", "Sample_Game_3"):
        df = pd.read_parquet(common.ROOT / "work" / "sample" / f"{m}.parquet")
        lab = pseudo_labels(m, len(df))
        np.save(args.out / f"{m}.npy", lab)
        print(m, stats(lab, df["graded"].to_numpy()))


if __name__ == "__main__":
    main()
