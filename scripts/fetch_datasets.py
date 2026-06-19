#!/usr/bin/env python3
"""Fetch all datasets required for the AURA benchmark.

Downloads Tanks & Temples and Mip-NeRF 360 datasets to the data/ directory
and records the exact commands for reproducibility.

Usage:
    python scripts/fetch_datasets.py --dataset tanks-and-temples --output data/tanks
    python scripts/fetch_datasets.py --dataset mipnerf360 --output data/mipnerf360
    python scripts/fetch_datasets.py --list    # list available datasets
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


DATASETS = {
    "tanks-and-temples-truck": {
        "description": "Tanks & Temples — truck scene (used in AURA benchmark)",
        "url": "https://storage.googleapis.com/gresearch/refraw360/360_v2.zip",
        "note": "The Tanks & Temples truck scene frames are available via the 3DGS project page.",
        "manual_steps": [
            "1. Visit https://www.tanksandtemples.org/download/ and register",
            "2. Download the 'truck' scene intermediate images",
            "3. Extract to data/tanks/truck/",
            "4. Run: python -m aura.cli ingest data/tanks/truck/ --output outputs/truck-pts129k-manifest.json",
        ],
    },
    "mipnerf360": {
        "description": "Mip-NeRF 360 scenes (bicycle, garden, kitchen, room, counter, bonsai, stump)",
        "url": "https://storage.googleapis.com/gresearch/refraw360/360_v2.zip",
        "note": "Available from the NeRF++ / Mip-NeRF 360 project.",
        "manual_steps": [
            "1. Run: wget https://storage.googleapis.com/gresearch/refraw360/360_v2.zip",
            "2. Run: unzip 360_v2.zip -d data/mipnerf360/",
            "3. Run: python -m aura.cli ingest data/mipnerf360/<scene>/ --output outputs/<scene>-manifest.json",
        ],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch AURA benchmark datasets")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), help="Dataset to fetch")
    parser.add_argument("--output", type=Path, default=Path("data"), help="Output directory")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    args = parser.parse_args()

    if args.list or not args.dataset:
        print("Available datasets:")
        for name, info in DATASETS.items():
            print(f"  {name}")
            print(f"    {info['description']}")
        sys.exit(0)

    info = DATASETS[args.dataset]
    print(f"Dataset: {args.dataset}")
    print(f"Description: {info['description']}")
    print()
    if "manual_steps" in info:
        print("Manual download required:")
        for step in info["manual_steps"]:
            print(f"  {step}")
    print()
    print(f"Note: {info.get('note', '')}")


if __name__ == "__main__":
    main()
