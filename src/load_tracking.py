"""Tracking loader: generic EPTS (FIFA-format Metrica matches and the match to label) and Metrica CSV.

Output: a DataFrame (one row per frame) written as parquet.
Colonnes : frame (= frame_id - 1), half (0 = hors mi-temps), graded, ball_x, ball_y, ball_s,
puis pour chaque piste N : tN_x, tN_y, tN_s, tN_team (0 domicile, 1 extérieur, -1 inconnu), tN_gk.
Coordonnées en mètres.

Sur les données privées, la ligne de commande n'affiche que des agrégats.
"""

from __future__ import annotations

import argparse
import io
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

import common


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _chan_kind(channel: str) -> str:
    """Normalise un nom de canal en 'x', 'y' ou 's' (vitesse)."""
    c = channel.strip().lower()
    if c in ("x", "y"):
        return c
    if c.startswith("s") or "speed" in c or "vel" in c:
        return "s"
    if c in ("z",):
        return "z"
    raise ValueError(f"canal inconnu : {channel!r}")


def _walk(node, seps: set, leaves: list) -> None:
    """Parcourt récursivement les registres : collecte séparateurs et feuilles (canaux) dans l'ordre."""
    for child in node:
        tag = _local(child.tag)
        if tag == "SplitRegister":
            seps.add(child.get("separator"))
            _walk(child, seps, leaves)
        elif tag == "PlayerChannelRef":
            leaves.append(("p", child.get("playerChannelId")))
        elif tag == "BallChannelRef":
            leaves.append(("b", _chan_kind(child.get("channelId"))))


def parse_metadata(meta_path: Path) -> dict:
    """Lit le XML EPTS : équipes, joueurs, canaux, spécifications de format, mi-temps."""
    root = ET.parse(meta_path).getroot()
    nodes = list(root.iter())
    teams = [e.get("id") for e in nodes if _local(e.tag) == "Team"]
    players = {}
    for e in nodes:
        if _local(e.tag) == "Player":
            params = {}
            for p in e.iter():
                if _local(p.tag) == "ProviderParameter":
                    name = next((c.text for c in p if _local(c.tag) == "Name"), None)
                    val = next((c.text for c in p if _local(c.tag) == "Value"), None)
                    params[name] = val
            players[e.get("id")] = {
                "team": e.get("teamId"),
                "gk": "goalkeeper" in (params.get("position_type") or "").lower(),
            }
    chan = {}
    for e in nodes:
        if _local(e.tag) == "PlayerChannel":
            chan[e.get("id")] = (e.get("playerId"), _chan_kind(e.get("channelId")))
    field = [(_local(e.tag), e.text) for e in nodes if _local(e.tag) in ("Width", "Height")]
    pitch = None
    if len(field) >= 2:
        pitch = (float(field[0][1]), float(field[1][1]))
    prov = {}
    for e in nodes:
        if _local(e.tag) == "ProviderParameter":
            name = next((c.text for c in e if _local(c.tag) == "Name"), None)
            val = next((c.text for c in e if _local(c.tag) == "Value"), None)
            if name and val and name.endswith(("_half_start", "_half_end")):
                prov[name] = int(val)
    specs = []
    for spec in (e for e in nodes if _local(e.tag) == "DataFormatSpecification"):
        seps = {spec.get("separator")}
        leaves: list[tuple[str, str]] = []
        _walk(spec, seps, leaves)
        specs.append({
            "start": int(spec.get("startFrame")), "end": int(spec.get("endFrame")),
            "seps": [s for s in seps if s], "leaves": leaves,
        })
    return {"teams": teams, "players": players, "chan": chan, "pitch": pitch,
            "prov": prov, "specs": specs}


def _half_from_blocks(n: int, blocks: list[tuple[int, int, int]]) -> np.ndarray:
    half = np.zeros(n, dtype=np.int8)
    for h, first, last in blocks:
        half[first:last + 1] = h
    return half


def load_epts(meta_path: Path, raw_path: Path, pitch: tuple[float, float] | None = None,
              half_blocks: list[tuple[int, int, int]] | None = None,
              speed_scale: float = 1.0) -> pd.DataFrame:
    """Parseur EPTS générique : groupes et canaux lus dans le XML, pas supposés."""
    meta = parse_metadata(Path(meta_path))
    L, W = pitch or meta["pitch"] or common.PITCH_SAMPLE
    lines = Path(raw_path).read_text().split("\n")
    lines = [ln for ln in lines if ln.strip()]
    fids = np.fromiter((int(ln[:ln.index(":")]) for ln in lines), dtype=np.int64, count=len(lines))
    n = int(fids.max())
    home = meta["teams"][0] if meta["teams"] else None

    # une passe par spécification : l'ordre des pistes suit l'ordre d'apparition des joueurs
    parsed = []
    for sp in meta["specs"]:
        order: list[str] = []
        cols: dict[str, dict[str, int]] = {}
        ball: dict[str, int] = {}
        for j, (kind, ref) in enumerate(sp["leaves"], start=1):  # 0 = numéro de frame
            if kind == "b":
                ball[ref] = j
                continue
            pid, ck = meta["chan"][ref]
            if pid not in cols:
                cols[pid] = {}
                order.append(pid)
            cols[pid][ck] = j
        parsed.append((sp, order, cols, ball))
    n_tracks = max(len(o) for _, o, _, _ in parsed)

    shape = (n, n_tracks)
    x = np.full(shape, np.nan, np.float32)
    y = np.full(shape, np.nan, np.float32)
    s = np.full(shape, np.nan, np.float32)
    team = np.full(shape, -1, np.int8)
    gk = np.zeros(shape, bool)
    bx = np.full(n, np.nan, np.float32)
    by = np.full(n, np.nan, np.float32)
    bs = np.full(n, np.nan, np.float32)
    table = {ord(c): "," for sp in meta["specs"] for c in sp["seps"]}

    for sp, order, cols, ball in parsed:
        sel = np.flatnonzero((fids >= sp["start"]) & (fids <= sp["end"]))
        if not len(sel):
            continue
        txt = "\n".join(lines[i] for i in sel).translate(table)
        arr = pd.read_csv(io.StringIO(txt), header=None, engine="c").to_numpy(np.float64)
        if arr.shape[1] != len(sp["leaves"]) + 1:
            raise ValueError(f"colonnes lues {arr.shape[1]} != attendues {len(sp['leaves']) + 1}")
        rows = fids[sel] - 1
        for k, pid in enumerate(order):
            c = cols[pid]
            x[rows, k] = arr[:, c["x"]] * L
            y[rows, k] = arr[:, c["y"]] * W
            if "s" in c:
                s[rows, k] = arr[:, c["s"]] * speed_scale
            info = meta["players"].get(pid, {})
            team[rows, k] = 0 if info.get("team") == home else (1 if info.get("team") else -1)
            gk[rows, k] = info.get("gk", False)
        if "x" in ball:
            bx[rows] = arr[:, ball["x"]] * L
            by[rows] = arr[:, ball["y"]] * W
        if "s" in ball:
            bs[rows] = arr[:, ball["s"]] * speed_scale

    if half_blocks is not None:
        half = _half_from_blocks(n, half_blocks)
    else:
        p = meta["prov"]
        blocks = []
        for h, name in ((1, "first"), (2, "second")):
            if f"{name}_half_start" in p:
                blocks.append((h, p[f"{name}_half_start"] - 1, p[f"{name}_half_end"] - 1))
        half = _half_from_blocks(n, blocks)
    return _assemble(n, half, bx, by, bs, x, y, s, team, gk)


def _assemble(n, half, bx, by, bs, x, y, s, team, gk) -> pd.DataFrame:
    data = {"frame": np.arange(n, dtype=np.int32), "half": half, "graded": half > 0,
            "ball_x": bx, "ball_y": by, "ball_s": bs}
    for k in range(x.shape[1]):
        t = k + 1
        data[f"t{t}_x"] = x[:, k]
        data[f"t{t}_y"] = y[:, k]
        data[f"t{t}_s"] = s[:, k]
        data[f"t{t}_team"] = team[:, k]
        data[f"t{t}_gk"] = gk[:, k]
    return pd.DataFrame(data)


def load_metrica_csv(home_csv: Path, away_csv: Path,
                     pitch: tuple[float, float] = common.PITCH_SAMPLE) -> pd.DataFrame:
    """Format CSV Metrica (Sample_Game_1 et 2) : un fichier par équipe, 3 lignes d'en-tête."""
    L, W = pitch
    parts = []
    for path in (home_csv, away_csv):
        df = pd.read_csv(path, header=2)
        parts.append(df)
    h = parts[0]
    n = int(h["Frame"].max())
    period = np.zeros(n, np.int8)
    period[h["Frame"].to_numpy() - 1] = h["Period"].to_numpy()
    xs, ys, teams, gks = [], [], [], []
    for ti, df in enumerate(parts):
        vals = df.iloc[:, 3:].to_numpy(np.float64)
        rows = df["Frame"].to_numpy() - 1
        npl = vals.shape[1] // 2 - 1  # dernière paire = ballon
        for k in range(npl):
            cx = np.full(n, np.nan, np.float32)
            cy = np.full(n, np.nan, np.float32)
            cx[rows] = vals[:, 2 * k] * L
            cy[rows] = vals[:, 2 * k + 1] * W
            xs.append(cx)
            ys.append(cy)
            teams.append(np.full(n, ti, np.int8))
            gks.append(np.full(n, k == 0, bool))  # convention Metrica : le gardien est listé en premier
        if ti == 0:
            bx = np.full(n, np.nan, np.float32)
            by = np.full(n, np.nan, np.float32)
            bx[rows] = vals[:, -2] * L
            by[rows] = vals[:, -1] * W
    nan = np.full(n, np.nan, np.float32)
    return _assemble(n, period, bx, by, nan, np.stack(xs, 1), np.stack(ys, 1),
                     np.stack([nan] * len(xs), 1), np.stack(teams, 1), np.stack(gks, 1))


def n_tracks(df: pd.DataFrame) -> int:
    return sum(1 for c in df.columns if re.fullmatch(r"t\d+_x", c))


def track_array(df: pd.DataFrame, what: str) -> np.ndarray:
    """Matrice (frames, pistes) pour 'x', 'y', 's', 'team' ou 'gk'."""
    return np.stack([df[f"t{k}_{what}"].to_numpy() for k in range(1, n_tracks(df) + 1)], axis=1)


def sample_paths(match: str, sample_dir: Path = common.SAMPLE_DIR) -> dict:
    d = Path(sample_dir) / match
    if match == "Sample_Game_3":
        return {"kind": "epts", "meta": d / f"{match}_metadata.xml",
                "raw": d / f"{match}_tracking.txt", "events": d / f"{match}_events.json"}
    return {"kind": "csv", "home": d / f"{match}_RawTrackingData_Home_Team.csv",
            "away": d / f"{match}_RawTrackingData_Away_Team.csv",
            "events": d / f"{match}_RawEventsData.csv"}


def load_sample(match: str, sample_dir: Path = common.SAMPLE_DIR) -> pd.DataFrame:
    p = sample_paths(match, sample_dir)
    if p["kind"] == "epts":
        return load_epts(p["meta"], p["raw"])
    return load_metrica_csv(p["home"], p["away"])


def summarize(df: pd.DataFrame) -> dict:
    """Agrégats seulement (sûr pour les données privées)."""
    g = df["graded"].to_numpy()
    nt = n_tracks(df)
    present = ~np.isnan(track_array(df, "x"))
    ball = ~np.isnan(df["ball_x"].to_numpy())
    out = {
        "frames": len(df), "pistes": nt,
        "ballon_visible_toutes_%": 100 * ball.mean(),
        "ballon_visible_notees_%": 100 * ball[g].mean() if g.any() else float("nan"),
        "objets_suivis_moy_notee": present[g].sum(1).mean() if g.any() else float("nan"),
    }
    sp = track_array(df, "s")
    if np.isfinite(sp).any():
        out["vitesse_joueurs_p50"] = float(np.nanpercentile(sp, 50))
        out["vitesse_joueurs_p99"] = float(np.nanpercentile(sp, 99))
        out["vitesse_joueurs_max"] = float(np.nanmax(sp))
    bs = df["ball_s"].to_numpy()
    if np.isfinite(bs).any():
        out["vitesse_ballon_p50"] = float(np.nanpercentile(bs, 50))
        out["vitesse_ballon_p99"] = float(np.nanpercentile(bs, 99))
        out["vitesse_ballon_max"] = float(np.nanmax(bs))
    return out


def _fmt(d: dict) -> dict:
    return {k: (round(float(v), 2) if isinstance(v, float | np.floating) else int(v))
            for k, v in d.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regime", choices=["sample", "real"], required=True)
    ap.add_argument("--data", type=Path, help="dossier de données (sample-data/data ou private)")
    ap.add_argument("--out", type=Path, help="parquet de sortie")
    ap.add_argument("--match", help="sample : Sample_Game_1|2|3 (défaut : les trois)")
    ap.add_argument("--speed-scale", type=float, default=1.0, help="facteur vers m/s")
    args = ap.parse_args()

    if args.regime == "sample":
        data = args.data or common.SAMPLE_DIR
        matches = [args.match] if args.match else ["Sample_Game_1", "Sample_Game_2", "Sample_Game_3"]
        for m in matches:
            df = load_sample(m, data)
            out = args.out or common.ROOT / "work" / "sample" / f"{m}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out)
            print(m, _fmt(summarize(df)))
    else:
        common.ensure_paths()
        import games
        paths = games.asset_paths(data_dir=str(args.data) if args.data else None)
        df = load_epts(paths["atd_meta"], paths["atd_raw"], pitch=common.PITCH_REAL,
                       half_blocks=list(games.half_video_blocks(games.GAME)),
                       speed_scale=args.speed_scale)
        out = args.out or common.ROOT / "work" / "real" / "tactical.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
        print("real", _fmt(summarize(df)))


if __name__ == "__main__":
    main()
