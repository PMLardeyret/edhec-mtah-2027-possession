"""Strict variant s, selected by the environment variable V23C; inactive when it is absent.

- `s`: training degradation constants come from configs/s_ball_profile.json and configs/v23c_s*.json, built from DFL and Metrica
  statistics only; track presence and cuts are drawn per seed from a-priori intervals; feature caches carry a suffix (`_v23cs`);
- V23C_MODULES (`none` or comma-separated list): feature modules (`latent`, `stop`); default ("stop", "latent").
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RANDOM_PRESENCE, RANDOM_CUTS = (0.0, 0.15), (0.5, 2.0)  # intervalles a priori de la variante stricte (REPORT_CONFORMITE.md)


def variant() -> str:
    return os.environ.get("V23C", "")


def name() -> str:
    """Variant name (s4): configuration, profile, weights and caches; V23C_NAME, else V23C."""
    return os.environ.get("V23C_NAME") or variant()


def source_name() -> str:
    """Variant whose feature caches and ball profile are used (`cache_name` key of the config)."""
    c = _cfg()
    return (c or {}).get("cache_name") or name()


def cache_suffix() -> str:
    return f"_v23c{source_name()}" if variant() == "s" else ""


def extra(default: tuple[str, ...] = ("stop", "latent")) -> tuple[str, ...]:
    """Modules d'indices actifs : V23C_MODULES s'il est défini, sinon `default`."""
    v = os.environ.get("V23C_MODULES")
    if v is None:
        return default
    return () if v.strip() in ("", "none") else tuple(x for x in v.split(",") if x)


def _cfg() -> dict | None:
    f = ROOT / "configs" / f"v23c_{name()}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def apply_degrade(mod) -> None:
    if variant() != "s":
        return
    mod.RANDOM_PRESENCE, mod.RANDOM_CUTS = RANDOM_PRESENCE, RANDOM_CUTS
    c = _cfg()
    if c:
        mod.GAP_MU, mod.GAP_SIGMA = c["gap_mu"], c["gap_sigma"]
        mod.MEAN_DROPOUT, mod.MIN_PIECE = c.get("mean_dropout", mod.MEAN_DROPOUT), c.get("min_piece", mod.MIN_PIECE)
        mod.EXTRA_BALL_LOSS = c.get("extra_ball_loss", mod.EXTRA_BALL_LOSS)


def apply_dfl(mod) -> None:
    if variant() != "s":
        return
    c = _cfg()
    if c:
        mod.IN_PLAY_LOSS, mod.DEAD_LOSS = c["in_play_loss"], c["dead_loss"]
        mod.PROFILE = ROOT / "configs" / "s_ball_profile.json"
