#!/usr/bin/env python3
"""
scripts/download_samples.py
============================
IBVAP P0 — Download public test videos and dataset samples.

All sources are publicly licensed (CC, public domain, or dataset challenge data).
Videos are saved to:  ibvap/data/samples/<category>/

Usage:
    python scripts/download_samples.py
    python scripts/download_samples.py --category vehicles   # only vehicles
    python scripts/download_samples.py --list                # list without downloading
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn, DownloadColumn, Progress,
        TextColumn, TimeRemainingColumn, TransferSpeedColumn,
    )
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    console = None  # type: ignore

# Root data directory (relative to repo root)
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


# =============================================================================
# Sample catalogue
# =============================================================================
@dataclass
class Sample:
    name:     str           # human-readable label
    category: str           # folder name
    url:      str           # direct download URL
    filename: str           # local save name
    md5:      Optional[str] = None   # optional integrity check
    notes:    str = ""


SAMPLES: List[Sample] = [
    # ── Person / pedestrian ──────────────────────────────────────────────────
    Sample(
        name="MOT17-04 sequence (person tracking benchmark)",
        category="persons",
        url="https://motchallenge.net/data/MOT17Det.zip",
        filename="MOT17Det.zip",
        notes="Multiple Object Tracking benchmark, person detection + tracking. "
              "Unzip → use MOT17/train/MOT17-04-DPM/img1/ and seqinfo.ini",
    ),
    Sample(
        name="VIRAT Ground Dataset — S_000200 clip (public surveillance)",
        category="persons",
        url="https://viratdata.org/data/ground/video/VIRAT_S_000200_00_000049_000440.mp4",
        filename="VIRAT_S_000200_clip.mp4",
        notes="Outdoor pedestrian surveillance. Public domain. "
              "Alt URL if down: download from https://viratdata.org/",
    ),
    # ── Vehicles ─────────────────────────────────────────────────────────────
    Sample(
        name="UA-DETRAC sample sequence (vehicle detection + tracking)",
        category="vehicles",
        url="https://detrac-db.rit.albany.edu/Data/DETRAC-train-data.zip",
        filename="DETRAC-train-data.zip",
        notes="Highway + intersection vehicle sequences. "
              "Registration required at detrac-db.rit.albany.edu first.",
    ),
    Sample(
        name="VisDrone2019-MOT sample (drone-view vehicles + people)",
        category="vehicles",
        url="https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-MOT-train-sample.zip",
        filename="VisDrone2019-MOT-train-sample.zip",
        notes="Official VisDrone sample from Ultralytics mirror. ~150 MB.",
    ),
    # ── Indian number plates (ANPR) ──────────────────────────────────────────
    Sample(
        name="IDD (Indian Driving Dataset) — sample frames",
        category="anpr",
        url="https://idd.insaan.iiit.ac.in/",
        filename="IDD_NOTE.txt",
        notes="Manual download required — register at idd.insaan.iiit.ac.in. "
              "Use IDD20k subset. Includes Indian road scenes with plates.",
    ),
    Sample(
        name="Kaggle Indian number plate dataset (placeholder note)",
        category="anpr",
        url="https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset",
        filename="KAGGLE_ANPR_NOTE.txt",
        notes="Download manually via Kaggle CLI: "
              "kaggle datasets download saisirishan/indian-vehicle-dataset. "
              "Contains ~2000 Indian LP images.",
    ),
    # ── Night / low-light ────────────────────────────────────────────────────
    Sample(
        name="ExDARK low-light dataset (sample images)",
        category="night",
        url="https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/raw/master/Dataset/ExDARK_readme.txt",
        filename="ExDARK_README.txt",
        notes="Full dataset at github.com/cs-chan/Exclusively-Dark-Image-Dataset. "
              "12 object classes in 10 lighting conditions. ~3 GB.",
    ),
    Sample(
        name="LOL low-light dataset (note)",
        category="night",
        url="https://daooshee.github.io/BMVC2018website/",
        filename="LOL_NOTE.txt",
        notes="Download from https://daooshee.github.io/BMVC2018website/ "
              "485 train + 15 test low/normal light pairs.",
    ),
    # ── Quick synthetic test clip (always downloadable, no auth) ─────────────
    Sample(
        name="Big Buck Bunny — 360p crowd clip (free test video)",
        category="general",
        url="https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
        filename="BigBuckBunny_360p.mp4",
        notes="Open-source test video. Useful for pipeline smoke-tests "
              "(stream decode, frame rate, colour channels). Not surveillance data.",
    ),
    Sample(
        name="Pexels crowd walking (CC0 test clip)",
        category="persons",
        url="https://www.pexels.com/download/video/854671/",
        filename="pexels_crowd_walking.mp4",
        notes="CC0 crowd walking video. Use as person-detection smoke test. "
              "May require browser download due to Pexels CDN headers.",
    ),
]


# =============================================================================
# Download helper
# =============================================================================
def _md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_note(path: Path, sample: Sample) -> None:
    """Write a .txt placeholder note for manual-download items."""
    path.write_text(
        f"MANUAL DOWNLOAD REQUIRED\n"
        f"========================\n"
        f"Name   : {sample.name}\n"
        f"URL    : {sample.url}\n"
        f"Notes  : {sample.notes}\n",
        encoding="utf-8",
    )
    print(f"  📝  Note written: {path.name}")


def download_sample(sample: Sample, dest_dir: Path) -> bool:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / sample.filename

    if out_path.exists():
        print(f"  ✅  Already exists: {sample.filename}")
        return True

    # For manual-download items, just write a note
    if sample.filename.endswith("_NOTE.txt") or "NOTE" in sample.filename.upper():
        _write_note(out_path, sample)
        return True

    print(f"\n  ⬇  {sample.name}")
    print(f"     URL  : {sample.url}")
    print(f"     Save : {out_path}")

    try:
        req = urllib.request.Request(
            sample.url,
            headers={"User-Agent": "IBVAP/1.0 (research; public datasets)"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            with out_path.open("wb") as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r     {pct:5.1f}%  {downloaded//1024//1024} MB", end="")
        print()

        # Integrity check
        if sample.md5:
            actual = _md5_file(out_path)
            if actual != sample.md5:
                print(f"  ❌  MD5 mismatch! Expected {sample.md5}, got {actual}")
                out_path.unlink()
                return False
        print(f"  ✅  Saved: {out_path.name}")
        return True

    except Exception as exc:
        print(f"  ⚠️   Download failed: {exc}")
        print(f"       Notes: {sample.notes}")
        if out_path.exists():
            out_path.unlink()
        # Write a note so the user knows what to download manually
        note_path = dest_dir / (sample.filename + ".FAILED_NOTE.txt")
        _write_note(note_path, sample)
        return False


# =============================================================================
# Entry point
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download IBVAP public test samples"
    )
    parser.add_argument(
        "--category", "-c",
        help="Only download samples in this category "
             "(persons | vehicles | anpr | night | general)",
    )
    parser.add_argument(
        "--list", "-l", action="store_true",
        help="List all available samples without downloading",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  IBVAP — Sample Video & Dataset Downloader")
    print("=" * 65)

    filtered = SAMPLES
    if args.category:
        filtered = [s for s in SAMPLES if s.category == args.category]
        if not filtered:
            print(f"  Unknown category '{args.category}'. "
                  f"Valid: persons, vehicles, anpr, night, general")
            sys.exit(1)

    if args.list:
        print(f"\n  {'#':<3} {'Category':<12} {'Name'}")
        print("  " + "-" * 60)
        for i, s in enumerate(filtered, 1):
            print(f"  {i:<3} {s.category:<12} {s.name}")
        print(f"\n  Total: {len(filtered)} samples")
        print(f"  Data directory: {DATA_DIR}\n")
        return

    print(f"\n  Data directory: {DATA_DIR}")
    print(f"  Samples to download: {len(filtered)}\n")

    ok = 0
    fail = 0
    for s in filtered:
        dest = DATA_DIR / s.category
        success = download_sample(s, dest)
        if success:
            ok += 1
        else:
            fail += 1

    print("\n" + "=" * 65)
    print(f"  Done. {ok} succeeded, {fail} failed (see *_NOTE.txt files)")
    print(f"  Directory: {DATA_DIR}")
    print("=" * 65)

    # Print dataset notes
    print("""
IMPORTANT — Datasets requiring manual steps:
─────────────────────────────────────────────
1. MOT17 (persons)
   → https://motchallenge.net/data/MOT17Det.zip (~5.5 GB)
   → No login required. Unzip to data/samples/persons/MOT17/

2. UA-DETRAC (vehicles)
   → https://detrac-db.rit.albany.edu  → register → download train data
   → Unzip to data/samples/vehicles/DETRAC/

3. IDD (ANPR, Indian roads)
   → https://idd.insaan.iiit.ac.in  → register → download IDD20k
   → Unzip to data/samples/anpr/IDD/

4. Indian LP dataset (ANPR)
   → kaggle datasets download saisirishan/indian-vehicle-dataset
   → Or use: https://universe.roboflow.com/fypindianplates/india-license-plate

5. ExDARK / LOL (night)
   → https://github.com/cs-chan/Exclusively-Dark-Image-Dataset
   → https://daooshee.github.io/BMVC2018website/

Quick test clips (no login):
   → VisDrone sample → should auto-download via Ultralytics mirror
   → Big Buck Bunny  → should auto-download from Google CDN
""")


if __name__ == "__main__":
    main()
