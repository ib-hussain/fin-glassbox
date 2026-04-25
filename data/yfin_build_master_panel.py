#!/usr/bin/env python3
"""
data/yFinance/build_master_panel.py

Build the comprehensive master OHLCV panel aligned to 6,288 NYSE trading days.
Uses merged/ (Stooq+Huge Market) as primary, ohlcv_panel.csv as secondary.
Outputs long-format panel with columns: date, ticker, open, high, low, close, volume

Usage:
  python build_master_panel.py --workers 2 --min-tickers 3500
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
DATES_FILE = Path("data/market_dates_NYSE.csv")
TICKER_LIST = Path("data/tickerList_final.csv")
OUTPUT_FILE = Path("data/yFinance/processed/master_ohlcv_panel.csv")
COVERAGE_FILE = Path("data/yFinance/processed/master_coverage.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Build master OHLCV panel")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--min-tickers", type=int, default=3500, help="Minimum tickers to keep")
    parser.add_argument("--min-days", type=int, default=500, help="Minimum trading days per ticker")
    parser.add_argument("--dry-run", action="store_true", help="Show stats only")
    return parser.parse_args()


def load_nyse_dates() -> tuple[list[str], set[str]]:
    """Load the 6,288 NYSE trading dates."""
    df = pd.read_csv(DATES_FILE, dtype={"date": str})
    dates_list = df["date"].tolist()
    dates_set = set(dates_list)
    print(f"Loaded {len(dates_list)} NYSE trading dates")
    return dates_list, dates_set


def load_merged_ticker(ticker: str) -> pd.DataFrame | None:
    """Load merged data for a ticker."""
    # Try common filename patterns
    patterns = [
        MERGED_DIR / f"{ticker.lower()}.us.csv",
        MERGED_DIR / f"{ticker.lower()}.csv",
    ]
    for fp in patterns:
        if fp.exists():
            df = pd.read_csv(fp, dtype={"date": str})
            if df.empty:
                return None
            # Keep only OHLCV columns
            cols = ["date", "open", "high", "low", "close", "volume"]
            df = df[[c for c in cols if c in df.columns]]
            for c in ["open", "high", "low", "close"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            if "volume" in df.columns:
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(np.int64)
            df["source"] = "merged"
            return df
    return None


def load_panel_ticker(ticker: str, panel_cache: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    """Load ohlcv_panel data for a ticker from cached chunks."""
    if ticker in panel_cache:
        df = panel_cache[ticker].copy()
        df["source"] = "panel"
        return df
    return None


def build_panel_cache() -> dict[str, pd.DataFrame]:
    """Load entire ohlcv_panel and group by ticker."""
    print("Loading ohlcv_panel.csv into memory...")
    df = pd.read_csv(PANEL_FILE, dtype={"ticker": str, "date": str})
    
    # Standardize ticker to uppercase
    df["ticker"] = df["ticker"].str.upper()
    
    # Keep only needed columns
    cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
    df = df[[c for c in cols if c in df.columns]]
    
    # Convert numeric
    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(np.int64)
    
    # Group by ticker
    cache = {}
    for ticker, g in tqdm(df.groupby("ticker"), desc="Building cache"):
        cache[ticker] = g.drop(columns=["ticker"])
    
    print(f"  Cached {len(cache)} tickers from panel")
    return cache


def merge_ticker_sources(
    ticker: str,
    nyse_dates_set: set[str],
    panel_cache: dict[str, pd.DataFrame],
) -> dict:
    """
    Merge merged/ + ohlcv_panel for one ticker.
    Returns dict with stats and the DataFrame (if dry_run=False, else None).
    """
    mg = load_merged_ticker(ticker)
    pn = load_panel_ticker(ticker, panel_cache)
    
    if mg is None and pn is None:
        return {"ticker": ticker, "status": "no_data", "rows": 0}
    
    # Combine
    frames = []
    if mg is not None and not mg.empty:
        frames.append(mg)
    if pn is not None and not pn.empty:
        frames.append(pn)
    
    if not frames:
        return {"ticker": ticker, "status": "no_data", "rows": 0}
    
    combined = pd.concat(frames, ignore_index=True)
    
    # For duplicate dates: prefer "merged" source (already scaled/adjusted)
    combined = combined.sort_values(["date", "source"])
    combined = combined.drop_duplicates(subset=["date"], keep="first")
    
    # Filter to NYSE trading days only
    combined = combined[combined["date"].isin(nyse_dates_set)]
    
    # Drop source column, sort by date
    combined = combined.drop(columns=["source"])
    combined = combined.sort_values("date")
    combined = combined.reset_index(drop=True)
    
    # Ensure all OHLCV columns exist
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in combined.columns:
            combined[c] = np.nan
    
    # Count how many of the 6288 days are present
    nyse_count = len(combined)
    
    return {
        "ticker": ticker,
        "status": "ok",
        "rows": nyse_count,
        "date_min": combined["date"].iloc[0] if nyse_count > 0 else None,
        "date_max": combined["date"].iloc[-1] if nyse_count > 0 else None,
        "df": combined,
    }


def main():
    args = parse_args()
    
    print("=" * 60)
    print("BUILD MASTER OHLCV PANEL")
    print("=" * 60)
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"Min tickers: {args.min_tickers}")
    print(f"Min days per ticker: {args.min_days}")
    print()
    
    # Load NYSE dates
    nyse_dates_list, nyse_dates_set = load_nyse_dates()
    print()
    
    # Load ticker list
    tickers = pd.read_csv(TICKER_LIST, dtype={"ticker": str})["ticker"].str.upper().tolist()
    print(f"Target tickers: {len(tickers)}")
    print()
    
    # Build panel cache
    panel_cache = build_panel_cache()
    print()
    
    # Process all tickers
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(merge_ticker_sources, t, nyse_dates_set, panel_cache): t
            for t in tickers
        }
        for future in tqdm(as_completed(futures), total=len(tickers), desc="Merging sources"):
            results.append(future.result())
    
    # Separate results
    ok_results = [r for r in results if r["status"] == "ok" and r["rows"] >= args.min_days]
    no_data = [r for r in results if r["status"] == "no_data"]
    low_coverage = [r for r in results if r["status"] == "ok" and r["rows"] < args.min_days]
    
    print(f"\n=== RESULTS ===")
    print(f"  Usable tickers (≥{args.min_days} days): {len(ok_results)}")
    print(f"  Low coverage (<{args.min_days} days):  {len(low_coverage)}")
    print(f"  No data:                               {len(no_data)}")
    print()
    
    # Sort by coverage (most days first)
    ok_results.sort(key=lambda r: r["rows"], reverse=True)
    
    # Select top N tickers
    selected = ok_results[:args.min_tickers]
    print(f"Selected top {len(selected)} tickers by coverage.")
    
    if len(selected) < args.min_tickers:
        print(f"⚠️  Only {len(selected)} tickers meet criteria.")
        if len(low_coverage) > 0:
            print(f"   Consider lowering --min-days ({args.min_days}) to include {len(low_coverage)} more.")
    print()
    
    # Coverage stats of selected
    days_col = [r["rows"] for r in selected]
    if days_col:
        print(f"Coverage of selected tickers:")
        print(f"  Max days:     {max(days_col)}")
        print(f"  Min days:     {min(days_col)}")
        print(f"  Mean days:    {np.mean(days_col):.0f}")
        print(f"  Median days:  {np.median(days_col):.0f}")
        print(f"  ≥5000 days:   {sum(1 for d in days_col if d >= 5000)}")
        print(f"  ≥4000 days:   {sum(1 for d in days_col if d >= 4000)}")
        print(f"  ≥3000 days:   {sum(1 for d in days_col if d >= 3000)}")
    print()
    
    # Generate coverage report
    coverage_rows = []
    for r in ok_results:
        coverage_rows.append({
            "ticker": r["ticker"],
            "trading_days": r["rows"],
            "pct_complete": round(r["rows"] / len(nyse_dates_list) * 100, 2),
            "date_min": r["date_min"],
            "date_max": r["date_max"],
        })
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df = coverage_df.sort_values("trading_days", ascending=False)
    
    if not args.dry_run:
        coverage_df.to_csv(COVERAGE_FILE, index=False)
        print(f"Coverage report saved: {COVERAGE_FILE}")
    
    # Build and save master panel
    if not args.dry_run:
        print("Building master panel...")
        selected_tickers = {r["ticker"] for r in selected}
        panel_frames = []
        
        for r in tqdm(selected, desc="Assembling panel"):
            df = r["df"]
            df["ticker"] = r["ticker"]
            # Reorder columns
            df = df[["date", "ticker", "open", "high", "low", "close", "volume"]]
            panel_frames.append(df)
        
        master = pd.concat(panel_frames, ignore_index=True)
        master = master.sort_values(["ticker", "date"])
        master.to_csv(OUTPUT_FILE, index=False)
        
        print(f"Master panel saved: {OUTPUT_FILE}")
        print(f"  Rows: {len(master):,}")
        print(f"  Tickers: {master['ticker'].nunique()}")
        print(f"  Date range: {master['date'].min()} → {master['date'].max()}")
    else:
        print("DRY RUN — no files written.")
        if len(selected) <= 10:
            print(f"\nSelected tickers: {[r['ticker'] for r in selected]}")
    
    print()
    print("Done.")


if __name__ == "__main__":
    main()