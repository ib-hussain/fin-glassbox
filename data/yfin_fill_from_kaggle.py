#!/usr/bin/env python3
"""
data/yFinance/fill_from_kaggle.py

Fill missing data in master_ohlcv_complete.csv using the Kaggle
NASDAQ/NYSE/NYSE-A/OTC 1962-2024 dataset.

Strategy:
  - Use Adjusted Close for consistency with our existing adjusted data
  - Open, High, Low are ALSO adjusted by the Kaggle dataset's methodology
  - Only fill rows where master has NaN
  - Only for tickers already in master

Usage:
  python fill_from_kaggle.py --workers 2
  python fill_from_kaggle.py --workers 2 --dry-run
"""

import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from tqdm import tqdm


# Paths
KAGGLE_DIR = Path("data/yFinance/nasdaq-nyse-nyse-a-otc-daily-stock-1962-2024")
MASTER_FILE = Path("data/yFinance/processed/master_ohlcv_complete.csv")
OUTPUT_FILE = Path("data/yFinance/processed/master_ohlcv_filled.csv")
REPORT_FILE = Path("data/yFinance/processed/fill_report.csv")

KAGGLE_FILES = [
    "NYSE 1962-2024.csv",
    "NASDAQ 1962-2024.csv",
    "NYSE A 1973-2024.csv",
    "OTC 1972-2024.csv",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Fill master panel from Kaggle dataset")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_kaggle_data(kaggle_dir: Path) -> pd.DataFrame:
    """
    Load all 4 Kaggle CSV files, standardize, and return one DataFrame.
    Uses Adjusted Close. Renames columns to match master format.
    """
    frames = []
    total_rows = 0
    
    for fname in KAGGLE_FILES:
        fp = kaggle_dir / fname
        if not fp.exists():
            print(f"  ⚠️  Not found: {fp}")
            continue
        
        print(f"  Loading {fname}...")
        df = pd.read_csv(fp, dtype={"Ticker": str, "Date": str})
        
        # Standardize column names
        df = df.rename(columns={
            "Date": "date",
            "Ticker": "ticker",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Adj Close": "close",  # Use adjusted close as primary close
            "Volume": "volume",
        })
        
        # Keep only needed columns
        cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
        df = df[[c for c in cols if c in df.columns]]
        
        # Upper case tickers
        df["ticker"] = df["ticker"].str.upper()
        
        # Convert numeric
        for c in ["open", "high", "low", "close"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        
        # Drop rows with no close price
        df = df.dropna(subset=["close"])
        
        if df.empty:
            continue
        
        n = len(df)
        total_rows += n
        print(f"    {n:,} valid rows, {df['ticker'].nunique()} unique tickers")
        frames.append(df)
    
    if not frames:
        print("No Kaggle data loaded!")
        return pd.DataFrame()
    
    combined = pd.concat(frames, ignore_index=True)
    print(f"  Total: {len(combined):,} rows, {combined['ticker'].nunique()} unique tickers")
    return combined


def load_master_tickers(master_path: Path, nrows: int = 10000) -> set:
    """Get the set of tickers in the master panel."""
    # Read just the ticker column efficiently
    tickers = set()
    for chunk in pd.read_csv(master_path, usecols=["ticker"], dtype={"ticker": str}, chunksize=500000):
        tickers.update(chunk["ticker"].str.upper().unique())
    return tickers


def fill_ticker(
    ticker: str,
    master_chunk: pd.DataFrame,
    kaggle_data: pd.DataFrame,
    dry_run: bool = False,
) -> dict:
    """
    Fill NaN values in master_chunk (one ticker) with data from kaggle_data.
    Returns the filled DataFrame and stats.
    """
    # Get kaggle data for this ticker
    kg = kaggle_data[kaggle_data["ticker"] == ticker]
    
    if kg.empty:
        return {
            "ticker": ticker,
            "rows_before": master_chunk.dropna(subset=["close"]).shape[0],
            "rows_filled": 0,
            "rows_after": master_chunk.dropna(subset=["close"]).shape[0],
        }
    
    # Create a copy of master chunk
    filled = master_chunk.copy()
    filled = filled.set_index("date")
    kg_indexed = kg.set_index("date")
    
    # Count NaN before
    nan_before = filled["close"].isna().sum()
    rows_with_data_before = filled["close"].notna().sum()
    
    # Fill only where master is NaN
    for col in ["open", "high", "low", "close", "volume"]:
        if col in filled.columns and col in kg_indexed.columns:
            # Get kaggle values for dates that exist in both
            common_dates = filled.index.intersection(kg_indexed.index)
            if len(common_dates) == 0:
                continue
            
            # For each common date where master is NaN, fill with kaggle value
            master_col = filled.loc[common_dates, col]
            kg_col = kg_indexed.loc[common_dates, col]
            
            # Only fill where master is NaN
            mask = master_col.isna()
            fill_dates = common_dates[mask]
            
            if len(fill_dates) > 0:
                filled.loc[fill_dates, col] = kg_indexed.loc[fill_dates, col]
    
    rows_after = filled["close"].notna().sum()
    rows_filled = rows_after - rows_with_data_before
    
    return {
        "ticker": ticker,
        "rows_before": rows_with_data_before,
        "rows_filled": rows_filled,
        "rows_after": rows_after,
        "df": filled.reset_index() if not dry_run else None,
    }


def main():
    args = parse_args()
    
    print("=" * 60)
    print("FILL MASTER PANEL FROM KAGGLE DATASET")
    print("=" * 60)
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print()
    
    # Load Kaggle data
    print("Loading Kaggle dataset...")
    kaggle = load_kaggle_data(KAGGLE_DIR)
    if kaggle.empty:
        print("ERROR: No Kaggle data loaded.")
        sys.exit(1)
    
    kaggle_tickers = set(kaggle["ticker"].unique())
    print()
    
    # Load master tickers
    print("Loading master ticker list...")
    master_tickers = load_master_tickers(MASTER_FILE)
    print(f"  {len(master_tickers)} tickers in master")
    
    # Overlap
    common = master_tickers & kaggle_tickers
    only_master = master_tickers - kaggle_tickers
    print(f"  {len(common)} tickers can be filled from Kaggle")
    print(f"  {len(only_master)} tickers have no Kaggle data")
    print()
    
    if args.dry_run:
        print("DRY RUN — checking sample fills...")
        sample_tickers = sorted(common)[:5]
        for ticker in sample_tickers:
            kg_sub = kaggle[kaggle["ticker"] == ticker]
            print(f"  {ticker}: {len(kg_sub)} Kaggle rows, "
                  f"dates {kg_sub['date'].min()} → {kg_sub['date'].max()}")
        print()
        print("Dry run complete. Run without --dry-run to execute.")
        return
    
    # Process ticker by ticker from master
    # We'll read master in chunks to avoid memory issues
    print("Processing master panel in chunks...")
    
    # First, index kaggle by ticker for fast lookup
    kaggle_indexed = kaggle.set_index(["ticker", "date"]).sort_index()
    
    # Read master in chunks
    chunk_size = 500000  # rows per chunk
    total_filled = 0
    total_rows_before = 0
    total_rows_after = 0
    ticker_stats = []
    
    # We'll write the output in chunks too
    first_chunk = True
    
    for chunk_idx, chunk in enumerate(tqdm(
        pd.read_csv(MASTER_FILE, dtype={"ticker": str, "date": str}, chunksize=chunk_size),
        desc="Processing"
    )):
        chunk["ticker"] = chunk["ticker"].str.upper()
        
        # Get unique tickers in this chunk
        chunk_tickers = chunk["ticker"].unique()
        
        # For each ticker in this chunk, fill from kaggle
        for ticker in chunk_tickers:
            if ticker not in common:
                continue
            
            # Get the rows for this ticker
            mask = chunk["ticker"] == ticker
            ticker_rows = chunk.loc[mask].copy()
            
            # Get kaggle data for this ticker
            if ticker not in kaggle_indexed.index.get_level_values(0):
                continue
            
            kg_ticker = kaggle_indexed.loc[ticker]
            
            # Count before
            nan_before = ticker_rows["close"].isna().sum()
            rows_before = ticker_rows["close"].notna().sum()
            
            # For each column, fill NaN
            ticker_rows = ticker_rows.set_index("date")
            
            for col in ["open", "high", "low", "close", "volume"]:
                if col not in ticker_rows.columns or col not in kg_ticker.columns:
                    continue
                
                # Common dates
                common_dates = ticker_rows.index.intersection(kg_ticker.index)
                if len(common_dates) == 0:
                    continue
                
                # Only fill NaN
                mask_nan = ticker_rows.loc[common_dates, col].isna()
                fill_dates = common_dates[mask_nan]
                
                if len(fill_dates) > 0:
                    ticker_rows.loc[fill_dates, col] = kg_ticker.loc[fill_dates, col]
            
            ticker_rows = ticker_rows.reset_index()
            rows_after = ticker_rows["close"].notna().sum()
            rows_filled = rows_after - rows_before
            
            total_rows_before += rows_before
            total_rows_after += rows_after
            total_filled += rows_filled
            
            if rows_filled > 0:
                ticker_stats.append({
                    "ticker": ticker,
                    "rows_before": rows_before,
                    "rows_filled": rows_filled,
                    "rows_after": rows_after,
                })
            
            # Write back
            chunk.loc[mask, ["open", "high", "low", "close", "volume"]] = \
                ticker_rows[["open", "high", "low", "close", "volume"]].values
        
        # Write chunk to output
        if first_chunk:
            chunk.to_csv(OUTPUT_FILE, index=False, mode="w")
            first_chunk = False
        else:
            chunk.to_csv(OUTPUT_FILE, index=False, mode="a", header=False)
    
    print()
    print("=" * 60)
    print("FILL SUMMARY")
    print("=" * 60)
    print(f"  Total rows with data BEFORE fill: {total_rows_before:,}")
    print(f"  Total rows FILLED from Kaggle:    {total_filled:,}")
    print(f"  Total rows with data AFTER fill:  {total_rows_after:,}")
    print(f"  Tickers improved:                  {len(ticker_stats)}")
    
    # Save fill report
    if ticker_stats:
        report_df = pd.DataFrame(ticker_stats)
        report_df = report_df.sort_values("rows_filled", ascending=False)
        report_df.to_csv(REPORT_FILE, index=False)
        print(f"  Fill report saved: {REPORT_FILE}")
        
        print(f"\n  Top 10 most filled tickers:")
        for _, row in report_df.head(10).iterrows():
            print(f"    {row['ticker']}: {row['rows_before']} → {row['rows_after']} "
                  f"(+{row['rows_filled']} days)")
    
    print(f"\n  Filled master saved: {OUTPUT_FILE}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()