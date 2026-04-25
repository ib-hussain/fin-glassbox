#!/usr/bin/env python3
"""
data/yFinance/rename_and_filter_stooq.py

Phase 1: Rename all .txt files to .csv in a directory tree
Phase 2: Filter — keep only files matching primary_tickers.csv, move others to irrelevant/

Usage:
  python rename_and_filter_stooq.py --dir "data/yFinance/d_us_txt" --tickers "data/primary_tickers.csv" --workers 4
  python rename_and_filter_stooq.py --dir "data/yFinance/d_us_txt" --tickers "data/primary_tickers.csv" --skip-rename
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Rename .txt→.csv and filter Stooq files to primary tickers only")
    parser.add_argument("--dir", required=True, help="Root directory containing .txt/.csv files (recursive)")
    parser.add_argument("--tickers", default="data/primary_tickers.csv", help="Path to primary_tickers.csv")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--skip-rename", action="store_true", help="Skip .txt→.csv rename, only do filtering")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without doing it")
    return parser.parse_args()


def find_all_txt_files(root_dir: Path) -> list[Path]:
    """Recursively find all .txt files."""
    return sorted(root_dir.rglob("*.txt"))


def find_all_csv_files(root_dir: Path) -> list[Path]:
    """Recursively find all .csv files."""
    return sorted(root_dir.rglob("*.csv"))


def rename_file(txt_path: Path, dry_run: bool = False) -> dict:
    """Rename a single .txt to .csv. Returns result dict."""
    csv_path = txt_path.with_suffix(".csv")
    try:
        if not txt_path.exists():
            return {"file": str(txt_path), "status": "missing", "error": "File not found"}
        if csv_path.exists():
            return {"file": str(txt_path), "status": "skipped", "error": "CSV already exists"}
        if not dry_run:
            txt_path.rename(csv_path)
        return {"file": str(txt_path), "status": "renamed", "new_name": str(csv_path)}
    except Exception as e:
        return {"file": str(txt_path), "status": "error", "error": str(e)}


def rename_all_txt_files(root_dir: Path, workers: int = 4, dry_run: bool = False):
    """Phase 1: Rename all .txt to .csv in parallel."""
    txt_files = find_all_txt_files(root_dir)
    if not txt_files:
        print("No .txt files found. Nothing to rename.")
        return
    
    print(f"Found {len(txt_files)} .txt files to rename.")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE'}")
    
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(rename_file, f, dry_run): f for f in txt_files}
        for future in tqdm(as_completed(futures), total=len(txt_files), desc="Renaming .txt→.csv"):
            results.append(future.result())
    
    # Summarize
    statuses = defaultdict(int)
    for r in results:
        statuses[r["status"]] += 1
    print(f"  Renamed: {statuses.get('renamed', 0)}")
    print(f"  Skipped (CSV exists): {statuses.get('skipped', 0)}")
    print(f"  Errors: {statuses.get('error', 0)}")
    if statuses.get('error', 0) > 0:
        for r in results:
            if r["status"] == "error":
                print(f"    {r['file']}: {r['error']}")


def load_primary_tickers(tickers_csv: Path) -> set[str]:
    """Load primary_ticker column from CSV. Returns set of uppercase tickers."""
    df = pd.read_csv(tickers_csv, usecols=["primary_ticker"], dtype={"primary_ticker": str})
    tickers = df["primary_ticker"].dropna().str.upper().unique()
    print(f"Loaded {len(tickers)} unique primary tickers from {tickers_csv}")
    return set(tickers)


def extract_ticker_from_filename(filepath: Path) -> str:
    """
    Extract ticker from Stooq filename: 'aapl.us.csv' → 'AAPL'
    Also handles: 'brk-b.us.csv' → 'BRK-B', 'bf-a.us.csv' → 'BF-A'
    """
    name = filepath.stem  # 'aapl.us' (after removing .csv)
    if name.endswith(".us"):
        ticker = name[:-3].upper()  # 'aapl' → 'AAPL'
        return ticker
    # Fallback: just use the stem uppercase
    return name.upper()


def filter_file(filepath: Path, primary_tickers: set[str], irrelevant_dir: Path, dry_run: bool = False) -> dict:
    """
    Check if file's ticker is in primary_tickers.
    If NOT, move to irrelevant/ directory.
    Returns result dict.
    """
    ticker = extract_ticker_from_filename(filepath)
    
    if ticker in primary_tickers:
        return {"file": str(filepath), "ticker": ticker, "status": "kept"}
    else:
        # Move to irrelevant/
        rel_path = filepath.relative_to(irrelevant_dir.parent) if irrelevant_dir.parent in filepath.parents else filepath
        dest = irrelevant_dir / filepath.name
        
        # Handle name collisions in irrelevant/
        if dest.exists():
            base = filepath.stem
            counter = 1
            while dest.exists():
                dest = irrelevant_dir / f"{base}_{counter}.csv"
                counter += 1
        
        try:
            if not dry_run:
                irrelevant_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(filepath), str(dest))
            return {"file": str(filepath), "ticker": ticker, "status": "moved_to_irrelevant", "dest": str(dest)}
        except Exception as e:
            return {"file": str(filepath), "ticker": ticker, "status": "error", "error": str(e)}


def filter_all_csv_files(root_dir: Path, primary_tickers: set[str], workers: int = 4, dry_run: bool = False):
    """Phase 2: Move non-primary-ticker files to irrelevant/."""
    csv_files = find_all_csv_files(root_dir)
    if not csv_files:
        print("No .csv files found. Nothing to filter.")
        return
    
    irrelevant_dir = root_dir / "irrelevant"
    
    print(f"Found {len(csv_files)} .csv files to check.")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE'}")
    print(f"Irrelevant files will go to: {irrelevant_dir}")
    
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(filter_file, f, primary_tickers, irrelevant_dir, dry_run): f for f in csv_files}
        for future in tqdm(as_completed(futures), total=len(csv_files), desc="Filtering tickers"):
            results.append(future.result())
    
    # Summarize
    statuses = defaultdict(int)
    tickers_kept = set()
    tickers_moved = set()
    for r in results:
        statuses[r["status"]] += 1
        if r["status"] == "kept":
            tickers_kept.add(r["ticker"])
        elif r["status"] == "moved_to_irrelevant":
            tickers_moved.add(r["ticker"])
    
    print(f"  Kept (in primary): {statuses.get('kept', 0)} files, {len(tickers_kept)} unique tickers")
    print(f"  Moved to irrelevant/: {statuses.get('moved_to_irrelevant', 0)} files, {len(tickers_moved)} unique tickers")
    print(f"  Errors: {statuses.get('error', 0)}")
    
    if statuses.get('error', 0) > 0:
        for r in results:
            if r["status"] == "error":
                print(f"    {r['file']}: {r['error']}")
    
    if not dry_run:
        # Remove empty subdirectories
        for subdir in sorted(root_dir.rglob("*"), reverse=True):
            if subdir.is_dir() and subdir != irrelevant_dir and not any(subdir.iterdir()):
                try:
                    subdir.rmdir()
                except OSError:
                    pass


def main():
    args = parse_args()
    root_dir = Path(args.dir).resolve()
    tickers_csv = Path(args.tickers).resolve()
    
    if not root_dir.exists():
        print(f"Error: Directory not found: {root_dir}")
        sys.exit(1)
    if not tickers_csv.exists():
        print(f"Error: Ticker CSV not found: {tickers_csv}")
        sys.exit(1)
    
    # Phase 1: Rename .txt → .csv
    if not args.skip_rename:
        print("=" * 60)
        print("PHASE 1: Renaming .txt → .csv")
        print("=" * 60)
        rename_all_txt_files(root_dir, workers=args.workers, dry_run=args.dry_run)
        print()
    
    # Phase 2: Load primary tickers and filter
    print("=" * 60)
    print("PHASE 2: Filtering to primary tickers")
    print("=" * 60)
    primary_tickers = load_primary_tickers(tickers_csv)
    filter_all_csv_files(root_dir, primary_tickers, workers=args.workers, dry_run=args.dry_run)
    
    print()
    print("Done.")


if __name__ == "__main__":
    main()