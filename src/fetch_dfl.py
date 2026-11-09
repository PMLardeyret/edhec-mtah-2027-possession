"""Download the open IDSSE/DFL data set (Bassek et al. 2025, figshare 28196177 version 1, CC BY 4.0) into work/external/dfl/.

    python src/fetch_dfl.py [--timeout 1800]
Closed list: the 21 files (name, size, SHA-256, figshare file id) are committed in configs/dfl_sha256.csv. Nothing is read from the
figshare API: file names and URLs come from that list only (https://ndownloader.figshare.com/files/<id>), HTTPS only (redirects
included). A file whose SHA-256 differs from the list is deleted and the script stops with an error. Idempotent: a present file with
the right SHA-256 is skipped. Writes work/external/dfl/MANIFEST.csv (name, size, sha256).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
import urllib.request
from pathlib import Path

import common

LIST = common.ROOT / "configs" / "dfl_sha256.csv"
BASE = "https://ndownloader.figshare.com/files/"
OUT = common.ROOT / "work" / "external" / "dfl"
EXPECTED_FILES = 21


class HttpsOnly(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.startswith("https://"):
            raise SystemExit(f"redirect refused (not HTTPS): {newurl.split('?')[0]}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


OPENER = urllib.request.build_opener(HttpsOnly)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_list() -> list[dict]:
    with open(LIST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != EXPECTED_FILES:
        raise SystemExit(f"{LIST.name}: {len(rows)} entries, expected {EXPECTED_FILES}")
    for r in rows:
        n = r["name"]
        if n != Path(n).name or not n.startswith("DFL_") or not n.endswith(".xml") or not r["file_id"].isdigit():
            raise SystemExit(f"invalid entry in {LIST.name}: {n!r}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=1800, help="total time limit (s)")
    t_end = time.time() + ap.parse_args().timeout
    OUT.mkdir(parents=True, exist_ok=True)
    for r in read_list():
        dst = OUT / r["name"]
        if dst.exists() and dst.stat().st_size == int(r["size"]) and sha256(dst) == r["sha256"]:
            continue
        print("downloading", r["name"], flush=True)
        n = 0
        with OPENER.open(BASE + r["file_id"], timeout=120) as resp, open(dst, "wb") as o:
            if not resp.geturl().startswith("https://"):
                dst.unlink(missing_ok=True)
                raise SystemExit("final URL is not HTTPS")
            while b := resp.read(1 << 20):
                n += len(b)
                if n > int(r["size"]):
                    break
                o.write(b)
                if time.time() > t_end:
                    raise SystemExit("time limit exceeded; rerun to resume")
        if n != int(r["size"]) or sha256(dst) != r["sha256"]:
            dst.unlink(missing_ok=True)
            raise SystemExit(f"SHA-256 or size mismatch, file deleted: {r['name']}")
    with open(OUT / "MANIFEST.csv", "w", newline="", encoding="utf-8") as o:
        w = csv.writer(o)
        w.writerow(["name", "size", "sha256"])
        w.writerows((r["name"], r["size"], r["sha256"]) for r in read_list())
    print("ok", EXPECTED_FILES, "files")


if __name__ == "__main__":
    main()
