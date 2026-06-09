#!/usr/bin/env python3
"""Acquire P1 scenario datasets: LOB (finance) + retail.

Run from repo root:  python scripts/download_p1_data.py [--only lobster,fi2010,m5,favorita]

Access tiers:
  - lobster  : OPEN sample files (level-10 LOB) — downloaded automatically.
  - fi2010   : benchmark LOB; needs Kaggle or the Aalto FAIRDATA mirror (manual).
  - m5       : Kaggle competition `m5-forecasting-accuracy` (needs kaggle creds).
  - favorita : Kaggle competition `favorita-grocery-sales-forecasting` (creds).

Datasets requiring credentials print exact instructions instead of failing hard,
so the open ones still complete. Update DATASET_SOURCES.md after a successful run.
"""

from __future__ import annotations

import argparse
import os
import shutil
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "input"

# Magic-byte signatures we accept as "real" archive/data downloads. Guards
# against a CDN/SPA returning an HTML shell with HTTP 200 (the LOBSTER case).
_ARCHIVE_MAGIC = (b"PK\x03\x04", b"7z\xbc\xaf\x27\x1c", b"\x1f\x8b")  # zip, 7z, gzip


def _download(url: str, dest: Path, expect_archive: bool = True, min_bytes: int = 1024) -> bool:
    """Stream a URL to disk, then validate it is not an HTML error page."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > min_bytes:
        print(f"  SKIP (exists): {dest.name}")
        return True
    print(f"  GET {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
    except Exception as e:  # noqa: BLE001 - report and continue with other files
        print(f"  FAIL {dest.name}: {e}")
        dest.unlink(missing_ok=True)
        return False
    # Content validation: reject tiny files and HTML masquerading as a download.
    head = dest.read_bytes()[:16]
    if dest.stat().st_size < min_bytes or head.lstrip().startswith(b"<"):
        print(f"  REJECT {dest.name}: looks like an HTML/error page, not data")
        dest.unlink(missing_ok=True)
        return False
    if expect_archive and not any(head.startswith(m) for m in _ARCHIVE_MAGIC):
        print(f"  REJECT {dest.name}: not a recognized archive (bad magic bytes)")
        dest.unlink(missing_ok=True)
        return False
    print(f"  OK  {dest.name} ({dest.stat().st_size:,} B)")
    return True


def fetch_lobster() -> None:
    """LOBSTER samples — site is now a SPA; scripted download is blocked."""
    print("[lobster] level-10 sample order books (manual step required)")
    print(
        "  lobsterdata.com migrated to a JS app that returns an HTML shell for\n"
        "  every direct URL, so samples cannot be curl'd. To obtain real LOB:\n"
        "    1) open https://lobsterdata.com/info/DataSamples.php in a browser\n"
        "    2) download the level-10 sample CSV pairs (message + orderbook)\n"
        "    3) place them under input/LOBSTER/<TICKER>/\n"
        "  Alternatively use FI-2010 (below) as the standard public LOB benchmark."
    )


def fetch_fi2010() -> None:
    """FI-2010 benchmark LOB — no stable open direct link; print instructions."""
    print("[fi2010] benchmark LOB (manual step required)")
    print(
        "  Option A (Kaggle):  kaggle datasets download -d "
        "<fi2010-mirror> -p input/FI2010 --unzip\n"
        "  Option B (FAIRDATA): download from "
        "https://etsin.fairdata.fi (search 'FI-2010 limit order book'),\n"
        "                       place the NoAuction_DecPre txt files under input/FI2010/\n"
        "  Expected files: Train_Dst_NoAuction_DecPre_CF_{1..9}.txt + Test_*"
    )


def _fetch_kaggle_competition(slug: str, out_dir: str) -> None:
    """Try the kaggle CLI for a competition; otherwise print manual steps."""
    out = BASE / out_dir
    if shutil.which("kaggle") and Path("~/.kaggle/kaggle.json").expanduser().exists():
        out.mkdir(parents=True, exist_ok=True)
        rc = os.system(f"kaggle competitions download -c {slug} -p '{out}'")
        if rc == 0:
            print(f"  downloaded {slug} -> {out} (unzip the archive next)")
            return
        print(f"  kaggle CLI returned {rc}; falling back to instructions")
    print(
        f"  kaggle creds not found. To fetch {slug}:\n"
        f"    1) pip install kaggle && place kaggle.json in ~/.kaggle/\n"
        f"    2) accept the competition rules on kaggle.com\n"
        f"    3) kaggle competitions download -c {slug} -p input/{out_dir} && unzip"
    )


def fetch_m5() -> None:
    print("[m5] Walmart sales (Kaggle: m5-forecasting-accuracy)")
    _fetch_kaggle_competition("m5-forecasting-accuracy", "M5")


def fetch_favorita() -> None:
    print("[favorita] grocery sales (Kaggle: favorita-grocery-sales-forecasting)")
    _fetch_kaggle_competition("favorita-grocery-sales-forecasting", "Favorita")


_TASKS = {
    "lobster": fetch_lobster,
    "fi2010": fetch_fi2010,
    "m5": fetch_m5,
    "favorita": fetch_favorita,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        default=",".join(_TASKS),
        help="comma-separated subset of: " + ", ".join(_TASKS),
    )
    args = ap.parse_args()
    for name in [s.strip() for s in args.only.split(",") if s.strip()]:
        if name not in _TASKS:
            print(f"unknown task {name!r}; have {list(_TASKS)}")
            continue
        _TASKS[name]()
    print("done.")


if __name__ == "__main__":
    main()
