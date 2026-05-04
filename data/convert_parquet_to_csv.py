#!/usr/bin/env python3
"""
Convert Parquet files to CSV format in-place (same directory).

Usage:
    python data/convert_parquet_to_csv.py --input data/FRED_data 
    python data/convert_parquet_to_csv.py --input data/yFinance --delete-original
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import pandas as pd


def find_parquet_files(root_dir: str | Path) -> List[Path]:
    """Find all .parquet files recursively."""
    root = Path(root_dir)
    return sorted(root.rglob("*.parquet"))


def convert_single(parquet_path: Path, delete_original: bool = False) -> Path:
    """
    Convert one parquet file to CSV.
    CSV is saved in the SAME directory with same name, .csv extension.
    """
    csv_path = parquet_path.with_suffix(".csv")
    
    print(f"Converting: {parquet_path}")
    print(f"       → {csv_path}")
    
    # Read parquet
    df = pd.read_parquet(parquet_path)
    
    # Write CSV
    df.to_csv(csv_path, index=False)
    
    file_size_mb = csv_path.stat().st_size / (1024 * 1024)
    print(f"       Done! {len(df):,} rows, {len(df.columns)} columns, {file_size_mb:.2f} MB")
    
    # Optionally delete original
    if delete_original:
        parquet_path.unlink()
        print(f"       Deleted original: {parquet_path}")
    
    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert Parquet files to CSV format."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data",
        help="Root directory to search for .parquet files.",
    )
    parser.add_argument(
        "--delete-original",
        action="store_true",
        help="Delete original .parquet files after conversion.",
    )
    args = parser.parse_args()
    
    root = Path(args.input)
    
    if not root.exists():
        print(f"Error: Directory not found: {root}")
        return
    
    # Find all parquet files
    parquet_files = find_parquet_files(root)
    
    if not parquet_files:
        print(f"No .parquet files found under: {root}")
        return
    
    print(f"\nFound {len(parquet_files)} parquet file(s)\n")
    print("=" * 60)
    
    # Convert each
    for pq_file in parquet_files:
        try:
            convert_single(pq_file, args.delete_original)
        except Exception as e:
            print(f"ERROR converting {pq_file}: {e}")
        print()
    
    print("=" * 60)
    print(f"Conversion complete. {len(parquet_files)} file(s) processed.")


if __name__ == "__main__":
    main()