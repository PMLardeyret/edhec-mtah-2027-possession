"""Converts the DFL/Sportec data set (work/external/dfl/*.xml) to the common schema: one parquet per match.

    python src/dfl_convert.py [--match DFL_1]
Output in work/external/dfl/: DFL_k.parquet (frame, half, graded, ball_x/y, t{j}_x/y/s/team/gk; metres, origin top left),
DFL_k_labels.npy (H=0, A=1, D=2: D if BallStatus = 0, else BallPossession 1 -> H, 2 -> A), DFL_k_status.npz (ball_status, possession,
ball_z), DFL_k.json (ids, teams). Match k = rank of the sorted MatchId. Team 0 = home. Referees are dropped; every player keeps a column.
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET

import numpy as np

import common
import load_tracking as lt

DIR = common.ROOT / "work" / "external" / "dfl"
FS = re.compile(r'<FrameSet GameSection="(\w+)" MatchId="[^"]*" TeamId="([^"]*)" PersonId="([^"]*)"')
FR = re.compile(r'<Frame N="(\d+)"[^>]* X="(-?[\d.]+)" Y="(-?[\d.]+)"(?: Z="(-?[\d.]+)")?[^>]*?(?: BallPossession="(\d)" BallStatus="(\d)")?\s*/>')


def match_ids() -> list[str]:
    return sorted(p.name.split("_")[-1][:-4] for p in DIR.glob("DFL_02_01_matchinformation_*.xml"))


def info(mid: str) -> dict:
    root = ET.parse(next(DIR.glob(f"DFL_02_01_matchinformation_*{mid}.xml"))).getroot()
    g = root.find(".//General").attrib
    teams = {t.attrib["TeamId"]: t for t in root.iter("Team")}
    gk = {p.attrib["PersonId"] for t in teams.values() for p in t.iter("Player") if p.attrib.get("PlayingPosition") == "TW"}
    return {"home": g["HomeTeamId"], "guest": g["GuestTeamId"], "title": g["MatchTitle"], "gk": gk}


def parse(mid: str) -> dict:
    """{(section, team, personne): (N, x, y, z, poss, status)} depuis le XML de positions (lecture ligne à ligne)."""
    out: dict = {}
    cur = None
    path = next(DIR.glob(f"DFL_04_03_positions_raw_observed_*{mid}.xml"))
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("<FrameSet"):
                s, team, pid = FS.match(line).groups()
                cur = (s, team, pid)
                out[cur] = ([], [], [], [], [], [])
            elif line.startswith("<Frame ") and cur is not None:
                m = FR.match(line)
                if m is None:
                    continue
                n, x, y, z, po, st = m.groups()
                a = out[cur]
                a[0].append(int(n)); a[1].append(float(x)); a[2].append(float(y))
                a[3].append(float(z) if z else 0.0); a[4].append(int(po) if po else 0); a[5].append(int(st) if st else -1)
    return out


def section_times(mid: str) -> dict:
    """{section: (N, T)} de la première image du ballon de chaque mi-temps : sert à dater les événements."""
    out, want = {}, None
    path = next(DIR.glob(f"DFL_04_03_positions_raw_observed_*{mid}.xml"))
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("<FrameSet"):
                s, team, _ = FS.match(line).groups()
                want = s if team == "BALL" else None
            elif want and line.startswith("<Frame "):
                m = re.match(r'<Frame N="(\d+)" T="([^"]+)"', line)
                out[want] = (int(m.group(1)), m.group(2))
                want = None
                if len(out) == 2:
                    break
    return out


def convert(k: int, mid: str) -> dict:
    inf, raw = info(mid), parse(mid)
    sections = [s for s in ("firstHalf", "secondHalf") if any(key[0] == s for key in raw)]
    span = {}
    for s in sections:
        ns = [np.array(v[0]) for key, v in raw.items() if key[0] == s and key[1] != "referee" and len(v[0])]
        span[s] = (min(n.min() for n in ns), max(n.max() for n in ns))
    off, tot = {}, 0
    for s in sections:
        off[s] = tot
        tot += span[s][1] - span[s][0] + 1
    n = tot
    half = np.zeros(n, np.int8)
    for i, s in enumerate(sections, start=1):
        half[off[s]:off[s] + span[s][1] - span[s][0] + 1] = i
    bx, by = np.full(n, np.nan, np.float32), np.full(n, np.nan, np.float32)
    bz, poss, stat = np.full(n, np.nan, np.float32), np.zeros(n, np.int8), np.full(n, -1, np.int8)
    cols: dict = {}
    for (s, team, pid), (N, X, Y, Z, PO, ST) in raw.items():
        if team == "referee" or not N:
            continue
        idx = np.array(N) - span[s][0] + off[s]
        if team == "BALL":
            bx[idx], by[idx], bz[idx] = np.array(X) + 52.5, np.array(Y) + 34.0, Z
            poss[idx], stat[idx] = PO, ST
            continue
        c = cols.setdefault(pid, {"team": 0 if team == inf["home"] else 1, "x": np.full(n, np.nan, np.float32),
                                   "y": np.full(n, np.nan, np.float32)})
        c["x"][idx], c["y"][idx] = np.array(X) + 52.5, np.array(Y) + 34.0
    order = sorted(cols, key=lambda p: (cols[p]["team"], p))
    x = np.stack([cols[p]["x"] for p in order], 1)
    y = np.stack([cols[p]["y"] for p in order], 1)
    team = np.tile(np.array([cols[p]["team"] for p in order], np.int8), (n, 1))
    gk = np.tile(np.array([p in inf["gk"] for p in order]), (n, 1))
    s = np.full(x.shape, np.nan, np.float32)
    df = lt._assemble(n, half, bx, by, np.full(n, np.nan, np.float32), x, y, s, team, gk)
    lab = np.where(stat == 0, common.D, np.where(poss == 1, common.H, np.where(poss == 2, common.A, common.D))).astype(np.uint8)
    name = f"DFL_{k}"
    df.to_parquet(DIR / f"{name}.parquet")
    np.save(DIR / f"{name}_labels.npy", lab)
    np.savez(DIR / f"{name}_status.npz", ball_status=stat, possession=poss, ball_z=bz)
    meta = {"match_id": mid, "title": inf["title"], "frames": int(n), "joueurs": len(order), "sections": sections,
            "debut_ballon": section_times(mid), "decalage_image": {s: int(off[s]) for s in sections}, "debut_N": {s: int(span[s][0]) for s in sections}}
    (DIR / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", help="ex. DFL_1 (défaut : les sept)")
    args = ap.parse_args()
    for k, mid in enumerate(match_ids(), start=1):
        if args.match and args.match != f"DFL_{k}":
            continue
        print(f"DFL_{k}", convert(k, mid), flush=True)


if __name__ == "__main__":
    main()
