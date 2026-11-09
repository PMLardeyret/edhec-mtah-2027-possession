"""Smart Tagging loader (SportsCode XML): intervals in frame index, round(s * 25)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

import common


def load_tagging(path: Path, fps: int = common.FPS) -> pd.DataFrame:
    """Une ligne par instance : code (minuscule), start_frame, end_frame, team, half, type, tags.

    Les comparaisons de texte sont insensibles à la casse : toutes les valeurs sont passées en
    minuscules sauf `tags` qui garde le texte d'origine.
    """
    rows = []
    for inst in ET.parse(path).iter("instance"):
        tags = {lab.findtext("group"): lab.findtext("text") for lab in inst.findall("label")}
        low = {(k or "").strip().lower(): (v or "").strip().lower() for k, v in tags.items()}
        rows.append({
            "code": (inst.findtext("code") or "").strip().lower(),
            "start_frame": round(float(inst.findtext("start")) * fps),
            "end_frame": round(float(inst.findtext("end")) * fps),
            "team": low.get("team", ""),
            "half": low.get("half", ""),
            "type": low.get("type", ""),
            "tags": tags,
        })
    return pd.DataFrame(rows)


def team_side(team: str, home_name: str = "barcelona") -> int:
    """0 = domicile, 1 = extérieur, -1 = inconnu (N/A). Le XML nomme des clubs, pas des côtés."""
    t = team.strip().lower()
    if not t or t == "n/a":
        return -1
    return 0 if t.startswith(home_name) else 1
