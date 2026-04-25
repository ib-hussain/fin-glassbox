#!/usr/bin/env python3
"""
data/yFinance/build_complete_panel.py

Build a COMPLETE matrix: every ticker × every NYSE trading day (6,288 rows per ticker).
Missing data = NaN, ready to be filled from any source.

Output: data/yFinance/processed/master_ohlcv_complete.csv
  - Exactly 6,288 rows × N tickers (in long format: date, ticker, open, high, low, close, volume)
  - NaN where no data exists
  - Sorted by ticker, then date

Usage:
  python build_complete_panel.py --workers 2
"""

import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm


# Paths
MERGED_DIR = Path("data/yFinance/merged")
PANEL_FILE = Path("data/yFinance/processed/ohlcv_panel.csv")
DATES_FILE = Path("data/market_dates_ONLY_NYSE.csv")
TICKER_LIST = Path("data/tickerList_final.csv")
OUTPUT_FILE = Path("data/yFinance/processed/master_ohlcv_complete.csv")
COVERAGE_FILE = Path("data/yFinance/processed/master_coverage_complete.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Build complete OHLCV matrix with NaN for missing")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_nyse_dates() -> tuple[list[str], set[str]]:
    df = pd.read_csv(DATES_FILE, dtype={"date": str})
    dates_list = df["date"].tolist()
    dates_set = set(dates_list)
    print(f"Loaded {len(dates_list)} NYSE trading dates from {DATES_FILE}")
    return dates_list, dates_set


def load_merged_ticker(ticker: str) -> pd.DataFrame | None:
    """Load merged data for a ticker from merged/ directory."""
    patterns = [
        MERGED_DIR / f"{ticker.lower()}.us.csv",
        MERGED_DIR / f"{ticker.lower()}.csv",
    ]
    for fp in patterns:
        if fp.exists():
            try:
                df = pd.read_csv(fp, dtype={"date": str})
                if df.empty:
                    return None
                cols = ["date", "open", "high", "low", "close", "volume"]
                df = df[[c for c in cols if c in df.columns]]
                for c in ["open", "high", "low", "close"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                if "volume" in df.columns:
                    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
                df["source"] = "merged"
                return df
            except:
                return None
    return None


def load_panel_ticker(ticker: str, panel_cache: dict) -> pd.DataFrame | None:
    """Load ohlcv_panel data for a ticker."""
    if ticker in panel_cache:
        df = panel_cache[ticker].copy()
        df["source"] = "panel"
        return df
    return None


def build_panel_cache() -> dict[str, pd.DataFrame]:
    """Load and group ohlcv_panel.csv by ticker."""
    print("Loading ohlcv_panel.csv...")
    df = pd.read_csv(PANEL_FILE, dtype={"ticker": str, "date": str})
    df["ticker"] = df["ticker"].str.upper()
    cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
    df = df[[c for c in cols if c in df.columns]]
    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    
    cache = {}
    for ticker, g in tqdm(df.groupby("ticker"), desc="Caching panel"):
        cache[ticker] = g.drop(columns=["ticker"])
    print(f"  Cached {len(cache)} tickers")
    return cache


def build_ticker_data(
    ticker: str,
    nyse_dates_list: list[str],
    nyse_dates_set: set[str],
    panel_cache: dict,
) -> pd.DataFrame | None:
    """
    Build complete 6,288-row DataFrame for one ticker.
    Returns DataFrame with all NYSE dates, NaN for missing data.
    Returns None if ticker has NO data at all.
    """
    # Load from both sources
    mg = load_merged_ticker(ticker)
    pn = load_panel_ticker(ticker, panel_cache)
    
    if mg is None and pn is None:
        return None
    
    # Combine, prefer merged source on duplicate dates
    frames = []
    if mg is not None and not mg.empty:
        frames.append(mg)
    if pn is not None and not pn.empty:
        frames.append(pn)
    
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["date", "source"])
    combined = combined.drop_duplicates(subset=["date"], keep="first")
    combined = combined.drop(columns=["source"])
    
    # Filter to only NYSE trading days that exist in our data
    combined = combined[combined["date"].isin(nyse_dates_set)]
    
    if combined.empty:
        return None
    
    # Reindex to ALL 6,288 NYSE dates (this is the key step — creates NaN rows for missing dates)
    combined = combined.set_index("date")
    combined = combined.reindex(nyse_dates_list)
    combined = combined.reset_index()
    combined = combined.rename(columns={"index": "date"})
    
    # Add ticker column
    combined["ticker"] = ticker
    
    # Ensure all OHLCV columns exist
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in combined.columns:
            combined[c] = np.nan
    
    # Column order
    combined = combined[["date", "ticker", "open", "high", "low", "close", "volume"]]
    
    return combined


def main():
    args = parse_args()
    
    print("=" * 60)
    print("BUILD COMPLETE OHLCV MATRIX (NaN for missing)")
    print("=" * 60)
    
    # Load NYSE dates
    nyse_dates_list, nyse_dates_set = load_nyse_dates()
    n_dates = len(nyse_dates_list)
    print(f"Target: {n_dates} rows per ticker\n")
    
    # Load ticker list
    tickers = pd.read_csv(TICKER_LIST, dtype={"ticker": str})["ticker"].str.upper().tolist()
    print(f"Target tickers: {len(tickers)}")
    
    # Build panel cache
    panel_cache = build_panel_cache()
    
    # Process all tickers
    print(f"\nProcessing {len(tickers)} tickers with {args.workers} workers...")
    results = []
    
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(build_ticker_data, t, nyse_dates_list, nyse_dates_set, panel_cache): t
            for t in tickers
        }
        for future in tqdm(as_completed(futures), total=len(tickers), desc="Building"):
            results.append(future.result())
    
    # Filter out None (no data tickers)
    dfs = [r for r in results if r is not None]
    no_data = len(results) - len(dfs)
    
    print(f"\nTickers with data: {len(dfs)}")
    print(f"Tickers with NO data: {no_data}")
    
    # Concatenate
    print("\nConcatenating...")
    master = pd.concat(dfs, ignore_index=True)
    master = master.sort_values(["ticker", "date"])
    
    # Calculate coverage per ticker
    coverage = master.groupby("ticker").apply(
        lambda g: g[["open", "high", "low", "close", "volume"]].notna().any(axis=1).sum()
    )
    coverage = coverage.reset_index(name="days_with_data")
    coverage["pct_complete"] = (coverage["days_with_data"] / n_dates * 100).round(2)
    
    # Find first/last data date per ticker
    first_date = master.dropna(subset=["close"]).groupby("ticker")["date"].first()
    last_date = master.dropna(subset=["close"]).groupby("ticker")["date"].last()
    coverage["date_min"] = coverage["ticker"].map(first_date)
    coverage["date_max"] = coverage["ticker"].map(last_date)
    coverage = coverage.sort_values("days_with_data", ascending=False)
    
    # Coverage stats
    print("\n=== COVERAGE STATISTICS ===")
    print(f"Expected rows per ticker: {n_dates}")
    print(f"Total rows in master: {len(master):,}")
    print(f"Expected total: {len(dfs) * n_dates:,}")
    print(f"\nDays with data distribution:")
    print(coverage["days_with_data"].describe())
    print(f"\nTickers with:")
    for threshold in [6288, 6000, 5000, 4000, 3000, 2000, 1000]:
        count = (coverage["days_with_data"] >= threshold).sum()
        print(f"  ≥{threshold} days: {count}")
    
    # Save
    if not args.dry_run:
        print(f"\nSaving master panel ({len(master):,} rows)...")
        master.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved: {OUTPUT_FILE}")
        
        coverage.to_csv(COVERAGE_FILE, index=False)
        print(f"Saved: {COVERAGE_FILE}")
        
        # Also save ticker list with coverage
        ticker_coverage = coverage[coverage["days_with_data"] > 0]
        ticker_coverage.to_csv("data/yFinance/processed/tickers_with_coverage.csv", index=False)
        print(f"Saved: data/yFinance/processed/tickers_with_coverage.csv")
    else:
        print("\nDRY RUN — no files written.")
    
    print("\nDone.")


if __name__ == "__main__":
    main()