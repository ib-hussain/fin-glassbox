#!/usr/bin/env python3
"""
data/yFinance/merge_sources.py

Merge standardized Stooq and Huge_Market_Dataset into a single unified source.

Strategy:
  1. For tickers in BOTH sources:
     a. Compute ratio = Huge_close / Stooq_close on common dates
     b. Scale Stooq prices (OHLC) by this ratio → adjusted to match Huge Market
     c. Take UNION of all dates
     d. For overlapping dates: prefer Huge Market (already adjusted, no scaling needed)
  2. For tickers in ONE source only: use as-is (Stooq will be unadjusted)
  3. Output: one clean CSV per ticker in data/yFinance/merged/

Output format: date,open,high,low,close,volume,source
  source: "stooq_scaled", "hugemarket", "stooq_only", "hugemarket_only"

Usage:
  python merge_sources.py --workers 4
  python merge_sources.py --workers 4 --dry-run
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
STOOQ_DIR = Path("data/yFinance/d_us_txt")
HUGE_DIR = Path("data/yFinance/Huge_Market_Dataset")
MERGED_DIR = Path("data/yFinance/merged")
COVERAGE_FILE = Path("data/yFinance/merged/coverage_report.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Merge Stooq and Huge_Market_Dataset into unified source")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--dry-run", action="store_true", help="Show stats without writing")
    return parser.parse_args()


def find_csv_files(root_dir: Path) -> dict[str, Path]:
    """
    Find all .csv files recursively, excluding irrelevant/.
    Returns dict: {TICKER: filepath}
    """
    ticker_to_path = {}
    for path in root_dir.rglob("*.csv"):
        if "irrelevant" in path.parts:
            continue
        # Extract ticker: 'aapl.us.csv' → 'AAPL'
        name = path.stem
        if name.endswith(".us"):
            ticker = name[:-3].upper()
        else:
            ticker = name.upper()
        ticker_to_path[ticker] = path
    return ticker_to_path


def compute_scaling_ratio(stooq_path: Path, huge_path: Path) -> float | None:
    """
    Compute ratio = Huge_close / Stooq_close on common dates.
    Uses median of daily ratios for robustness.
    Returns None if cannot compute (< 10 common dates).
    """
    try:
        s = pd.read_csv(stooq_path, usecols=["date", "close"], dtype={"date": str, "close": float})
        h = pd.read_csv(huge_path, usecols=["date", "close"], dtype={"date": str, "close": float})
        
        if s.empty or h.empty:
            return None
        
        m = s.merge(h, on="date", suffixes=("_s", "_h"))
        m = m[m["close_s"] > 0]  # Avoid div by zero
        
        if len(m) < 10:
            return None
        
        m["ratio"] = m["close_h"] / m["close_s"]
        
        # Use median — robust to outliers
        ratio = m["ratio"].median()
        
        # Sanity check: ratio should be between 0.01 and 100
        if ratio < 0.01 or ratio > 100:
            return None
        
        return float(ratio)
    
    except Exception:
        return None


def merge_ticker(
    ticker: str,
    stooq_path: Path | None,
    huge_path: Path | None,
    dry_run: bool = False,
) -> dict:
    """
    Merge Stooq and Huge Market data for a single ticker.
    Returns result dict with stats.
    """
    try:
        dfs = []
        sources = []
        
        # Load Stooq
        if stooq_path is not None:
            s = pd.read_csv(stooq_path, dtype={"date": str})
            s["date"] = pd.to_datetime(s["date"])
            s = s.sort_values("date")
            dfs.append(("stooq", s))
        
        # Load Huge Market
        if huge_path is not None:
            h = pd.read_csv(huge_path, dtype={"date": str})
            h["date"] = pd.to_datetime(h["date"])
            h = h.sort_values("date")
            dfs.append(("huge", h))
        
        if not dfs:
            return {"ticker": ticker, "status": "no_data", "rows": 0}
        
        # Case 1: Both sources available — scale Stooq and merge
        if len(dfs) == 2:
            _, s_df = dfs[0]  # stooq
            _, h_df = dfs[1]  # huge
            
            # Compute scaling ratio
            ratio = compute_scaling_ratio(stooq_path, huge_path)
            
            if ratio is not None:
                # Scale Stooq to match Huge adjusted prices
                s_scaled = s_df.copy()
                for col in ["open", "high", "low", "close"]:
                    if col in s_scaled.columns:
                        s_scaled[col] = s_scaled[col] * ratio
                s_scaled["source"] = "stooq_scaled"
                
                h_df["source"] = "hugemarket"
                
                # Concatenate
                merged = pd.concat([s_scaled, h_df], ignore_index=True)
                
                # For duplicate dates: prefer Huge Market (original adjusted)
                merged = merged.sort_values(["date", "source"])
                merged = merged.drop_duplicates(subset=["date"], keep="last")
                
                source_used = "both_scaled"
                ratio_used = ratio
            else:
                # Cannot compute ratio — just concatenate without scaling
                # Mark Stooq as unadjusted
                s_df["source"] = "stooq_unadjusted"
                h_df["source"] = "hugemarket"
                merged = pd.concat([s_df, h_df], ignore_index=True)
                merged = merged.sort_values(["date", "source"])
                merged = merged.drop_duplicates(subset=["date"], keep="last")
                source_used = "both_noscale"
                ratio_used = None
        
        # Case 2: Only one source
        else:
            source_name, merged = dfs[0]
            if source_name == "stooq":
                merged["source"] = "stooq_only"
                source_used = "stooq_only"
                ratio_used = None
            else:
                merged["source"] = "hugemarket_only"
                source_used = "hugemarket_only"
                ratio_used = None
        
        # Sort and clean
        merged = merged.sort_values("date")
        merged = merged.drop_duplicates(subset=["date"])
        merged = merged[merged["date"].notna()]
        
        # Ensure correct column order
        cols = ["date", "open", "high", "low", "close", "volume", "source"]
        merged = merged[[c for c in cols if c in merged.columns]]
        
        # Fill any missing columns
        for c in cols:
            if c not in merged.columns:
                merged[c] = np.nan
        
        # Write
        if not dry_run:
            out_path = MERGED_DIR / f"{ticker.lower()}.us.csv"
            merged.to_csv(out_path, index=False)
        
        return {
            "ticker": ticker,
            "status": "merged",
            "source_used": source_used,
            "rows": len(merged),
            "date_min": merged["date"].min().strftime("%Y-%m-%d") if len(merged) > 0 else None,
            "date_max": merged["date"].max().strftime("%Y-%m-%d") if len(merged) > 0 else None,
            "ratio": ratio_used,
        }
    
    except Exception as e:
        return {"ticker": ticker, "status": "error", "error": str(e), "rows": 0}


def main():
    args = parse_args()
    
    print("=" * 60)
    print("MERGE: Stooq + Huge Market Dataset → Unified Source")
    print("=" * 60)
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print()
    
    # Find all files
    print("Scanning Stooq files...")
    stooq_tickers = find_csv_files(STOOQ_DIR)
    print(f"  Found {len(stooq_tickers)} Stooq tickers")
    
    print("Scanning Huge Market files...")
    huge_tickers = find_csv_files(HUGE_DIR)
    print(f"  Found {len(huge_tickers)} Huge Market tickers")
    
    # All unique tickers
    all_tickers = sorted(set(stooq_tickers.keys()) | set(huge_tickers.keys()))
    common = sorted(set(stooq_tickers.keys()) & set(huge_tickers.keys()))
    stooq_only = sorted(set(stooq_tickers.keys()) - set(huge_tickers.keys()))
    huge_only = sorted(set(huge_tickers.keys()) - set(stooq_tickers.keys()))
    
    print()
    print(f"Total unique tickers: {len(all_tickers)}")
    print(f"  Both sources:       {len(common)}")
    print(f"  Stooq only:         {len(stooq_only)}")
    print(f"  Huge Market only:   {len(huge_only)}")
    print()
    
    if not args.dry_run:
        MERGED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process all tickers in parallel
    results = []
    tasks = []
    for ticker in all_tickers:
        sp = stooq_tickers.get(ticker)
        hp = huge_tickers.get(ticker)
        tasks.append((ticker, sp, hp))
    
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(merge_ticker, ticker, sp, hp, args.dry_run): ticker
            for ticker, sp, hp in tasks
        }
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Merging tickers"):
            results.append(future.result())
    
    # Summarize
    statuses = defaultdict(int)
    source_counts = defaultdict(int)
    total_rows = 0
    ratios = []
    
    for r in results:
        statuses[r["status"]] += 1
        total_rows += r.get("rows", 0)
        if r.get("source_used"):
            source_counts[r["source_used"]] += 1
        if r.get("ratio") is not None:
            ratios.append(r["ratio"])
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Successfully merged: {statuses.get('merged', 0)}")
    print(f"  Errors:              {statuses.get('error', 0)}")
    print(f"  No data:             {statuses.get('no_data', 0)}")
    print(f"  Total rows:          {total_rows:,}")
    print()
    print("Source breakdown:")
    for src, count in sorted(source_counts.items()):
        print(f"  {src}: {count} tickers")
    print()
    
    if ratios:
        print(f"Scaling ratios (for {len(ratios)} tickers with both sources):")
        print(f"  Mean:   {np.mean(ratios):.4f}")
        print(f"  Median: {np.median(ratios):.4f}")
        print(f"  Min:    {np.min(ratios):.4f}")
        print(f"  Max:    {np.max(ratios):.4f}")
        print(f"  % near 1.0 (0.95-1.05): {sum(0.95 <= r <= 1.05 for r in ratios) / len(ratios) * 100:.1f}%")
        print()
    
    if statuses.get('error', 0) > 0:
        print("Errors:")
        for r in results:
            if r["status"] == "error":
                print(f"  {r['ticker']}: {r['error']}")
        print()
    
    # Build coverage report
    print("Generating coverage report...")
    coverage_rows = []
    for r in results:
        if r["status"] == "merged":
            coverage_rows.append({
                "ticker": r["ticker"],
                "source": r["source_used"],
                "rows": r["rows"],
                "date_min": r["date_min"],
                "date_max": r["date_max"],
                "scale_ratio": r.get("ratio"),
            })
    
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df = coverage_df.sort_values("ticker")
    
    if not args.dry_run:
        coverage_df.to_csv(COVERAGE_FILE, index=False)
        print(f"Coverage report saved: {COVERAGE_FILE}")
        print(f"Merged files saved to: {MERGED_DIR}/")
    else:
        print("DRY RUN — no files written.")
    
    print()
    print("Done.")


if __name__ == "__main__":
    main()