#!/usr/bin/env python3
"""Download and verify the canonical G-Research Crypto Forecasting benchmark.

This is the Crypto benchmark used by Time-SSM / KRNO-style experiments:
14 crypto assets, minute-level aggregate trading features, and a Target column.
This script downloads the raw canonical source only; model-specific
Time-SSM/KRNO preprocessing should live in separate reproduction scripts.

Prerequisites:
  1. Install/configure Kaggle CLI with ~/.kaggle/kaggle.json.
  2. Preferably open https://www.kaggle.com/competitions/g-research-crypto-forecasting
     while logged in and accept the competition rules.

Run from repo root:
  python scripts/download_gresearch_crypto.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd


SLUG = "g-research-crypto-forecasting"
MIRROR_DATASET = "bariscan07/g-research-crypto-forecasting-dataset"
REQUIRED_FILES = {
    "asset_details.csv",
    "example_sample_submission.csv",
    "example_test.csv",
    "supplemental_train.csv",
    "train.csv",
}
CANONICAL_COLUMNS = [
    "timestamp",
    "Asset_ID",
    "Count",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "VWAP",
    "Target",
]
# These are the official Kaggle competition file sizes listed by
# `kaggle competitions files g-research-crypto-forecasting`.
OFFICIAL_FILE_SIZES = {
    "asset_details.csv": 444,
    "example_sample_submission.csv": 406,
    "example_test.csv": 5923,
    "gresearch_crypto/__init__.py": 59,
    "gresearch_crypto/competition.cpython-37m-x86_64-linux-gnu.so": 468536,
    "supplemental_train.csv": 300931304,
    "train.csv": 2819286393,
}


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def require_kaggle() -> None:
    if shutil.which("kaggle") is None:
        raise SystemExit("kaggle CLI not found. Install with `pip install kaggle` and configure ~/.kaggle/kaggle.json.")
    token = Path("~/.kaggle/kaggle.json").expanduser()
    if not token.exists():
        raise SystemExit("Missing ~/.kaggle/kaggle.json. Create a Kaggle API token and place it there.")


def ensure_entered_competition() -> bool:
    result = run(["kaggle", "competitions", "list", "-s", SLUG])
    if result.returncode != 0:
        raise SystemExit(result.stdout)
    if "False" in result.stdout:
        return False
    return True


def download_competition(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run(["kaggle", "competitions", "download", "-c", SLUG, "-p", str(out_dir)])
    if result.returncode != 0:
        raise SystemExit(result.stdout)
    print(result.stdout.strip())


def download_mirror(out_dir: Path) -> None:
    """Download a Kaggle Dataset mirror whose files match the competition files.

    The competition is closed and Kaggle requires every account to accept rules
    before downloading from the competition endpoint. This mirror preserves the
    same raw filenames and byte sizes, so it is suitable when the competition
    endpoint is blocked.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run(["kaggle", "datasets", "download", "-d", MIRROR_DATASET, "-p", str(out_dir)])
    if result.returncode != 0:
        raise SystemExit(result.stdout)
    print(result.stdout.strip())


def unzip_archives(out_dir: Path) -> None:
    for archive in out_dir.glob("*.zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
        print(f"Extracted {archive.name}")


def verify(out_dir: Path) -> dict[str, object]:
    missing = sorted(name for name in REQUIRED_FILES if not (out_dir / name).exists())
    if missing:
        raise SystemExit(f"Missing required files after download: {missing}")

    train_path = out_dir / "train.csv"
    assets_path = out_dir / "asset_details.csv"
    train_head = pd.read_csv(train_path, nrows=5)
    asset_details = pd.read_csv(assets_path)

    missing_cols = [c for c in CANONICAL_COLUMNS if c not in train_head.columns]
    if missing_cols:
        raise SystemExit(f"train.csv missing canonical columns: {missing_cols}; got {list(train_head.columns)}")
    if list(train_head.columns) != CANONICAL_COLUMNS:
        raise SystemExit(f"train.csv columns differ from the canonical order: {list(train_head.columns)}")

    asset_ids = sorted(asset_details["Asset_ID"].dropna().astype(int).unique().tolist())
    if len(asset_ids) != 14:
        raise SystemExit(f"Expected 14 assets in asset_details.csv, found {len(asset_ids)}: {asset_ids}")

    # Count rows without loading all columns into memory.
    with train_path.open() as f:
        train_rows = sum(1 for _ in f) - 1
    with (out_dir / "supplemental_train.csv").open() as f:
        supplemental_rows = sum(1 for _ in f) - 1

    official_size_matches = {}
    for name, expected_size in OFFICIAL_FILE_SIZES.items():
        path = out_dir / name
        official_size_matches[name] = path.exists() and path.stat().st_size == expected_size

    report = {
        "source": "https://www.kaggle.com/competitions/g-research-crypto-forecasting",
        "mirror_source": f"https://www.kaggle.com/datasets/{MIRROR_DATASET}",
        "slug": SLUG,
        "path": str(out_dir),
        "files": {p.name: p.stat().st_size for p in sorted(out_dir.iterdir()) if p.is_file()},
        "canonical_columns": CANONICAL_COLUMNS,
        "official_file_size_matches": official_size_matches,
        "asset_count": len(asset_ids),
        "asset_ids": asset_ids,
        "train_rows": train_rows,
        "supplemental_train_rows": supplemental_rows,
    }
    (out_dir / "VERIFY.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="input/G-Research-Crypto")
    parser.add_argument(
        "--source",
        choices=["auto", "competition", "mirror"],
        default="auto",
        help="auto tries the official competition first, then the byte-matching Kaggle Dataset mirror.",
    )
    parser.add_argument("--skip-download", action="store_true", help="Only unzip and verify existing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not args.skip_download:
        require_kaggle()
        if args.source == "competition":
            if not ensure_entered_competition():
                raise SystemExit(
                    "Kaggle reports userHasEntered=False for g-research-crypto-forecasting.\n"
                    "Open https://www.kaggle.com/competitions/g-research-crypto-forecasting, "
                    "click Join/Accept Rules, then rerun this script."
                )
            download_competition(out_dir)
        elif args.source == "mirror":
            download_mirror(out_dir)
        else:
            if ensure_entered_competition():
                download_competition(out_dir)
            else:
                print(
                    "Kaggle competition endpoint is blocked by userHasEntered=False; "
                    f"using byte-matching mirror dataset {MIRROR_DATASET}."
                )
                download_mirror(out_dir)
    unzip_archives(out_dir)
    report = verify(out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
