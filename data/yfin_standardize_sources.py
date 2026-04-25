#!/usr/bin/env python3
"""
data/yFinance/standardize_sources.py

Standardize all Stooq (.csv) and Huge_Market_Dataset (.csv) files to a common format:
  Date,Open,High,Low,Close,Volume

Stooq format: <TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>
  - DATE is YYYYMMDD, no separators
  - Prices are NOT adjusted

Huge_Market_Dataset format: Date,Open,High,Low,Close,Volume,OpenInt
  - Date is YYYY-MM-DD
  - Prices ARE adjusted for dividends/splits (per dataset description)

Target output format (both sources): Date,Open,High,Low,Close,Volume
  - Date: YYYY-MM-DD
  - Open,High,Low,Close: float
  - Volume: int
  - Skip OpenInt column
  - Keep Ticker as filename identifier

Usage:
  python standardize_sources.py --dir "data/yFinance/d_us_txt" --source stooq --workers 4
  python standardize_sources.py --dir "data/yFinance/Huge_Market_Dataset" --source hugemarket --workers 4
  python standardize_sources.py --all --workers 4  # Process both
"""

import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Standardize Stooq and Huge_Market_Dataset to common format")
    parser.add_argument("--dir", nargs="+", help="Directories to process")
    parser.add_argument("--source", choices=["stooq", "hugemarket"], help="Source type")
    parser.add_argument("--all", action="store_true", help="Process both known directories")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--dry-run", action="store_true", help="Show sample without writing")
    return parser.parse_args()


def get_directories(args) -> list[tuple[Path, str]]:
    """Get list of (directory, source_type) to process."""
    dirs = []
    
    if args.all:
        dirs = [
            (Path("data/yFinance/d_us_txt"), "stooq"),
            (Path("data/yFinance/Huge_Market_Dataset"), "hugemarket"),
        ]
    elif args.dir:
        if not args.source:
            print("Error: --source required with --dir")
            sys.exit(1)
        for d in args.dir:
            dirs.append((Path(d), args.source))
    else:
        print("Error: Specify --dir and --source, or --all")
        sys.exit(1)
    
    # Validate
    for d, s in dirs:
        if not d.exists():
            print(f"Error: Directory not found: {d}")
            sys.exit(1)
    
    return dirs


def find_csv_files(root_dir: Path) -> list[Path]:
    """Find all .csv files recursively, excluding irrelevant/ directory."""
    csv_files = []
    for path in root_dir.rglob("*.csv"):
        # Skip files in irrelevant/ directories
        if "irrelevant" in path.parts:
            continue
        csv_files.append(path)
    return sorted(csv_files)


def extract_ticker_from_filename(filepath: Path) -> str:
    """Extract ticker from filename: 'aapl.us.csv' → 'AAPL'"""
    name = filepath.stem  # 'aapl.us'
    if name.endswith(".us"):
        return name[:-3].upper()
    return name.upper()


def standardize_stooq(filepath: Path, dry_run: bool = False) -> dict:
    """
    Convert Stooq format to standard CSV.
    
    Input:  <TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>
    Output: Date,Open,High,Low,Close,Volume
    """
    ticker = extract_ticker_from_filename(filepath)
    
    try:
        df = pd.read_csv(filepath, dtype=str)
        
        # Skip if empty
        if df.empty:
            return {"file": str(filepath), "ticker": ticker, "status": "empty", "rows": 0}
        
        # Rename columns (strip angle brackets and whitespace)
        df.columns = [c.strip().strip("<>").strip() for c in df.columns]
        
        # Map columns
        col_map = {
            "DATE": "date",
            "OPEN": "open",
            "HIGH": "high",
            "LOW": "low",
            "CLOSE": "close",
            "VOL": "volume",
        }
        
        # Keep only needed columns
        needed = ["date", "open", "high", "low", "close", "volume"]
        df = df.rename(columns=col_map)
        df = df[[c for c in needed if c in df.columns]]
        
        # Handle missing columns
        for c in needed:
            if c not in df.columns:
                df[c] = None
        
        # Parse date: YYYYMMDD → YYYY-MM-DD
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        
        # Convert numeric columns
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
        
        # Drop rows with no date
        df = df.dropna(subset=["date"])
        
        # Sort by date
        df = df.sort_values("date")
        
        # Write back (overwrite original or dry-run)
        if not dry_run:
            df.to_csv(filepath, index=False)
        
        return {
            "file": str(filepath),
            "ticker": ticker,
            "status": "standardized",
            "rows": len(df),
            "date_min": df["date"].iloc[0] if len(df) > 0 else None,
            "date_max": df["date"].iloc[-1] if len(df) > 0 else None,
        }
    
    except Exception as e:
        return {"file": str(filepath), "ticker": ticker, "status": "error", "error": str(e)}


def standardize_hugemarket(filepath: Path, dry_run: bool = False) -> dict:
    """
    Convert Huge_Market_Dataset format to standard CSV.
    
    Input:  Date,Open,High,Low,Close,Volume,OpenInt
    Output: Date,Open,High,Low,Close,Volume
    
    This format is already close to target — just drop OpenInt and ensure date format.
    """
    ticker = extract_ticker_from_filename(filepath)
    
    try:
        df = pd.read_csv(filepath, dtype=str)
        
        if df.empty:
            return {"file": str(filepath), "ticker": ticker, "status": "empty", "rows": 0}
        
        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Rename 'date' if it's capitalized
        col_map = {}
        for col in df.columns:
            if col.lower() == "date":
                col_map[col] = "date"
        df = df.rename(columns=col_map)
        
        # Keep only needed columns
        needed = ["date", "open", "high", "low", "close", "volume"]
        available = [c for c in needed if c in df.columns]
        df = df[available]
        
        # Handle missing columns
        for c in needed:
            if c not in df.columns:
                df[c] = None
        
        # Parse date: YYYY-MM-DD
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        
        # Convert numeric columns
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
        
        # Drop rows with no date
        df = df.dropna(subset=["date"])
        
        # Sort by date
        df = df.sort_values("date")
        
        if not dry_run:
            df.to_csv(filepath, index=False)
        
        return {
            "file": str(filepath),
            "ticker": ticker,
            "status": "standardized",
            "rows": len(df),
            "date_min": df["date"].iloc[0] if len(df) > 0 else None,
            "date_max": df["date"].iloc[-1] if len(df) > 0 else None,
        }
    
    except Exception as e:
        return {"file": str(filepath), "ticker": ticker, "status": "error", "error": str(e)}


def process_directory(root_dir: Path, source_type: str, workers: int = 4, dry_run: bool = False):
    """Process all CSV files in a directory tree."""
    csv_files = find_csv_files(root_dir)
    
    if not csv_files:
        print(f"No .csv files found in {root_dir}")
        return
    
    standardizer = standardize_stooq if source_type == "stooq" else standardize_hugemarket
    
    print(f"Found {len(csv_files)} files to standardize ({source_type} format).")
    print(f"Mode: {'DRY RUN (no writes)' if dry_run else 'LIVE'}")
    
    # Show sample before
    if dry_run and csv_files:
        sample_path = csv_files[0]
        print(f"\nSample file BEFORE: {sample_path}")
        with open(sample_path) as f:
            for i, line in enumerate(f):
                if i < 5:
                    print(f"  {line.rstrip()}")
                else:
                    break
    
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(standardizer, f, dry_run): f for f in csv_files}
        for future in tqdm(as_completed(futures), total=len(csv_files), desc=f"Standardizing {root_dir.name}"):
            results.append(future.result())
    
    # Summarize
    statuses = defaultdict(int)
    total_rows = 0
    for r in results:
        statuses[r["status"]] += 1
        total_rows += r.get("rows", 0)
    
    print(f"  Standardized: {statuses.get('standardized', 0)}")
    print(f"  Empty files:  {statuses.get('empty', 0)}")
    print(f"  Errors:       {statuses.get('error', 0)}")
    print(f"  Total rows:   {total_rows:,}")
    
    if statuses.get('error', 0) > 0:
        for r in results:
            if r["status"] == "error":
                print(f"    {r['file']}: {r['error']}")
    
    # Show sample after (for dry-run)
    if dry_run and csv_files:
        sample_path = csv_files[0]
        print(f"\nSample file AFTER standardization would look like:")
        try:
            if source_type == "stooq":
                r = standardize_stooq(sample_path, dry_run=True)
            else:
                r = standardize_hugemarket(sample_path, dry_run=True)
            if r["status"] == "standardized":
                print(f"  Date range: {r['date_min']} → {r['date_max']}")
                print(f"  Rows: {r['rows']}")
        except:
            pass
    
    return results


def main():
    args = parse_args()
    dirs = get_directories(args)
    
    for root_dir, source_type in dirs:
        print("=" * 60)
        print(f"Processing: {root_dir} (source: {source_type})")
        print("=" * 60)
        process_directory(root_dir, source_type, workers=args.workers, dry_run=args.dry_run)
        print()
    
    print("Done.")


if __name__ == "__main__":
    main()