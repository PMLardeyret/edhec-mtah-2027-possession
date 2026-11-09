"""Download the external code/data repositories at pinned commits and verify the commit hash.

    python src/setup_external.py
Clones into external/ (not versioned). The IDSSE (DFL) files are fetched and SHA-256-checked by src/fetch_dfl.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOS = {
    "more-than-a-hack-2027": ("https://github.com/metrica-sports/more-than-a-hack-2027", "3de6b60aeabd2475d95879a651d38b72bc4fba00"),
    "sample-data": ("https://github.com/metrica-sports/sample-data", "e706dd506b360d69d9d123d5b8026e7294b13996"),
}


def git(*a: str, cwd: Path | None = None) -> str:
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    (ROOT / "external").mkdir(exist_ok=True)
    for name, (url, commit) in REPOS.items():
        d = ROOT / "external" / name
        if not d.exists():
            git("clone", "--quiet", url, str(d))
        git("checkout", "--quiet", commit, cwd=d)
        head = git("rev-parse", "HEAD", cwd=d)
        if head != commit:
            sys.exit(f"{name}: expected commit {commit}, got {head}")
        if git("status", "--porcelain", cwd=d):
            sys.exit(f"{name}: working tree not clean")
        print(f"[ok] {name} @ {head}", flush=True)


if __name__ == "__main__":
    main()
