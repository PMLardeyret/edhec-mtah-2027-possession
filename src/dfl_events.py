"""Frames de reprise (touche, six mètres, corner, coup franc) des matchs DFL, en indices du parquet `DFL_k`.

EventTime de l'événement (le coup de pied) est converti en image avec l'horodatage de la première image de chaque mi-temps
(DFL_k.json : debut_ballon, decalage_image). Hors coup d'envoi, penalty et balle à terre, comme setpiece_anchor.set_piece_frames_dev.
"""

from __future__ import annotations

import json
import re
from datetime import datetime

import numpy as np

import common
from dfl_convert import DIR, info, match_ids

KINDS = ("ThrowIn", "GoalKick", "CornerKick", "FreeKick")
EV = re.compile(r'<Event [^>]*EventTime="([^"]+)"')


def _t(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


def set_piece_frames_dfl(k: int) -> np.ndarray:
    mid = match_ids()[k - 1]
    meta = json.loads((DIR / f"DFL_{k}.json").read_text(encoding="utf-8"))
    t0 = {s: _t(v[1]) for s, v in meta["debut_ballon"].items()}
    off = meta["decalage_image"]
    second = t0["secondHalf"]
    out, cur = [], None
    path = next(DIR.glob(f"DFL_03_02_events_raw_*{mid}.xml"))
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = EV.match(line.lstrip())
            if m:
                cur = _t(m.group(1))
            elif cur is not None and any(line.lstrip().startswith(f"<{kd} ") for kd in KINDS):
                sec = "secondHalf" if cur >= second - 1 else "firstHalf"
                i = round((cur - t0[sec]) * common.FPS) + off[sec]
                if i >= 0:
                    out.append(i)
                cur = None
    return np.sort(np.array(out, dtype=int))


TOUCH = re.compile(r'<(?:Play|OtherBallAction) [^>]*?Team="([^"]+)"')
TOUCH_SHOT = re.compile(r'<(?:Play|OtherBallAction|ShotAtGoal|ShotWide|SavedShot|BlockedShot|SuccessfulShot) [^>]*?Team="([^"]+)"')


def touch_frames_dfl(k: int, shots: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Touches de balle (éléments Play et OtherBallAction, plus les tirs si `shots`) : images du parquet `DFL_k` et équipe (0 domicile, 1 extérieur)."""
    mid = match_ids()[k - 1]
    meta = json.loads((DIR / f"DFL_{k}.json").read_text(encoding="utf-8"))
    home = info(mid)["home"]
    t0 = {s: _t(v[1]) for s, v in meta["debut_ballon"].items()}
    off = meta["decalage_image"]
    second = t0["secondHalf"]
    fr, tm, cur = [], [], None
    path = next(DIR.glob(f"DFL_03_02_events_raw_*{mid}.xml"))
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            ls = line.lstrip()
            m = EV.match(ls)
            if m:
                cur = _t(m.group(1))
                continue
            t = (TOUCH_SHOT if shots else TOUCH).match(ls)
            if t and cur is not None:
                sec = "secondHalf" if cur >= second - 1 else "firstHalf"
                i = round((cur - t0[sec]) * common.FPS) + off[sec]
                if i >= 0:
                    fr.append(i)
                    tm.append(0 if t.group(1) == home else 1)
    o = np.argsort(fr, kind="stable")
    return np.array(fr, dtype=int)[o], np.array(tm, dtype=int)[o]


if __name__ == "__main__":
    for k in range(1, 8):
        f = set_piece_frames_dfl(k)
        n = len(np.load(DIR / f"DFL_{k}_labels.npy"))
        lab = np.load(DIR / f"DFL_{k}_labels.npy")
        ok = f[f < n]
        # contrôle : part des reprises précédées d'un D dans les 10 s avant
        pre = np.mean([(lab[max(0, i - 250):i + 1] == common.D).any() for i in ok])
        print(k, len(f), "hors bornes", int((f >= n).sum()), "D dans les 10 s avant", round(float(pre), 2))
