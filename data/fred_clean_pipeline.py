#!/usr/bin/env python3
"""
FRED Data Cleaning Pipeline

Takes the grand table and:
1. Filters to NYSE trading days only (removes weekends/holidays)
2. Drops columns with excessive missing data
3. Generates summary statistics
4. Outputs a clean, trading-day-only macro features file

Usage:
    python data/fred_clean_pipeline.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

DEBUG_MODE = bool(int(os.getenv("DEBUG_MODE", 1)))

# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path("data/FRED_data")
GRAND_TABLE = DATA_PATH / "outputs" / "grand_table_raw.csv"
MARKET_DATES = Path("data/market_dates_NYSE.csv")
OUTPUT_DIR = DATA_PATH / "outputs"

# ============================================================
# HELPERS
# ============================================================

def now_ts() -> float:
    return time.time()


def fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60.0
    return f"{hours:.2f}h"


# ============================================================
# STEP 1: LOAD GRAND TABLE + MARKET DATES
# ============================================================

def load_data() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Load the grand table and NYSE trading calendar."""
    
    # Load grand table
    print(f"Loading grand table: {GRAND_TABLE}")
    df = pd.read_csv(GRAND_TABLE, index_col=0, parse_dates=True)
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df.index[0]} → {df.index[-1]}")
    
    # Load NYSE market dates
    print(f"\nLoading market dates: {MARKET_DATES}")
    market_dates_df = pd.read_csv(MARKET_DATES)
    
    # Find the date column (first column)
    date_col = market_dates_df.columns[0]
    print(f"  Date column: '{date_col}'")
    
    # Parse dates - just take the date part, ignore time
    market_dates = pd.to_datetime(market_dates_df[date_col]).dt.normalize()
    
    # Filter to 2000-2024 range
    market_dates = market_dates[
        (market_dates >= "2000-01-01") & (market_dates <= "2024-12-31")
    ]
    
    print(f"  Trading days (2000-2024): {len(market_dates):,}")
    
    return df, pd.DatetimeIndex(market_dates)

# ============================================================
# STEP 2: FILTER TO TRADING DAYS ONLY
# ============================================================

def filter_trading_days(df: pd.DataFrame, market_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Keep only rows that fall on NYSE trading days."""
    
    # Convert market dates to a set of date objects for fast lookup
    trading_days = set(d.date() for d in market_dates)
    
    # Filter dataframe index to only trading days
    mask = df.index.to_series().apply(lambda x: x.date() in trading_days)
    
    filtered = df[mask].copy()
    
    removed = len(df) - len(filtered)
    print(f"\n{'=' * 60}")
    print("FILTERING TO TRADING DAYS")
    print(f"{'=' * 60}")
    print(f"Before: {len(df):,} rows (all calendar days)")
    print(f"After:  {len(filtered):,} rows (NYSE trading days only)")
    print(f"Removed: {removed:,} rows (weekends + holidays)")
    
    return filtered
# ============================================================
# STEP 3: SUMMARY STATISTICS
# ============================================================

def generate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate comprehensive summary statistics."""
    
    print(f"\n{'=' * 60}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 60}")
    
    rows = []
    
    for col in df.columns:
        series = df[col]
        n_total = len(series)
        n_valid = series.count()
        n_missing = n_total - n_valid
        pct_missing = (n_missing / n_total) * 100
        pct_coverage = 100 - pct_missing
        
        stats = {
            "column": col,
            "total_rows": n_total,
            "valid": n_valid,
            "missing": n_missing,
            "pct_missing": round(pct_missing, 2),
            "pct_coverage": round(pct_coverage, 2),
        }
        
        if n_valid > 0:
            stats.update({
                "mean": round(series.mean(), 4),
                "std": round(series.std(), 4),
                "min": round(series.min(), 4),
                "p25": round(series.quantile(0.25), 4),
                "median": round(series.median(), 4),
                "p75": round(series.quantile(0.75), 4),
                "max": round(series.max(), 4),
            })
        else:
            stats.update({
                "mean": None, "std": None, "min": None,
                "p25": None, "median": None, "p75": None, "max": None,
            })
        
        rows.append(stats)
    
    summary = pd.DataFrame(rows)
    
    # Determine quality tier
    def quality_tier(pct: float) -> str:
        if pct >= 95:
            return "⭐⭐⭐ EXCELLENT"
        elif pct >= 80:
            return "⭐⭐ GOOD"
        elif pct >= 50:
            return "⭐ FAIR"
        elif pct >= 20:
            return "⚠️ POOR"
        else:
            return "❌ BAD"
    
    summary["quality"] = summary["pct_coverage"].apply(quality_tier)
    
    # Sort by coverage descending
    summary = summary.sort_values("pct_coverage", ascending=False)
    
    # Print overview
    tiers = summary["quality"].value_counts()
    print(f"\nQuality Distribution:")
    for tier, count in tiers.items():
        print(f"  {tier}: {count} columns")
    
    print(f"\nTop 10 (best coverage):")
    print(summary[["column", "pct_coverage", "mean", "std", "quality"]].head(10).to_string(index=False))
    
    print(f"\nBottom 10 (worst coverage):")
    print(summary[["column", "pct_coverage", "mean", "std", "quality"]].tail(10).to_string(index=False))
    
    return summary


# ============================================================
# STEP 4: DROP BAD COLUMNS
# ============================================================

def drop_bad_columns(df: pd.DataFrame, summary: pd.DataFrame, 
                     min_coverage: float = 50.0) -> pd.DataFrame:
    """Drop columns with coverage below threshold."""
    
    bad_cols = summary[summary["pct_coverage"] < min_coverage]["column"].tolist()
    
    print(f"\n{'=' * 60}")
    print(f"DROPPING COLUMNS (< {min_coverage}% COVERAGE)")
    print(f"{'=' * 60}")
    
    if bad_cols:
        print(f"Dropping {len(bad_cols)} columns:")
        for col in bad_cols:
            cov = summary[summary["column"] == col]["pct_coverage"].values[0]
            print(f"  {col}: {cov:.1f}% coverage")
    else:
        print("No columns to drop!")
    
    cleaned = df.drop(columns=bad_cols, errors="ignore")
    print(f"\nBefore: {df.shape[1]} columns")
    print(f"After:  {cleaned.shape[1]} columns")
    
    return cleaned


# ============================================================
# STEP 5: HANDLE REMAINING MISSING VALUES
# ============================================================

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill remaining gaps. NEVER drop trading days."""
    
    print(f"\n{'=' * 60}")
    print("HANDLING REMAINING MISSING VALUES")
    print(f"{'=' * 60}")
    
    before_nan = df.isna().sum().sum()
    print(f"Missing values before: {before_nan:,}")
    print(f"Total trading days: {len(df):,}")
    
    # Forward-fill with generous limit (1 month = ~21 trading days)
    # This fills gaps where data arrives later
    df = df.ffill(limit=21)
    
    after_ffill = df.isna().sum().sum()
    print(f"Missing after ffill (21-day limit): {after_ffill:,}")
    
    # For data that still has leading NaN (series that start after 2000),
    # backfill with a small limit (1 week = 5 trading days)
    # This handles the very beginning of some series
    df = df.bfill(limit=5)
    
    after_bfill = df.isna().sum().sum()
    print(f"Missing after bfill (5-day limit): {after_bfill:,}")
    
    # Only drop rows that are COMPLETELY empty (all NaN)
    # This preserves ALL trading days
    df = df.dropna(how='all')
    
    # For remaining NaN in individual columns, fill with column median
    # (only affects a few edge cases)
    remaining_nan = df.isna().sum().sum()
    if remaining_nan > 0:
        print(f"Filling {remaining_nan:,} remaining NaN with column medians...")
        df = df.fillna(df.median())
    
    print(f"Final shape: {df.shape}")
    print(f"Final date range: {df.index[0]} → {df.index[-1]}")
    print(f"Trading days preserved: {len(df):,} / 6,288")
    
    return df

# ============================================================
# STEP 6: ENGINEER ADDITIONAL FEATURES
# ============================================================

def engineer_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that benefit from cleaned data."""
    feats = pd.DataFrame(index=df.index)
    
    # Yield Spreads (if not already present)
    if all(c in df.columns for c in ["DGS10", "DGS2"]):
        if "yield_spread_10y2y" not in df.columns:
            feats["yield_spread_10y2y"] = df["DGS10"] - df["DGS2"]
    if all(c in df.columns for c in ["DGS10", "DGS3MO"]):
        if "yield_spread_10y3m" not in df.columns:
            feats["yield_spread_10y3m"] = df["DGS10"] - df["DGS3MO"]
    if all(c in df.columns for c in ["BAA10Y", "AAA10Y"]):
        if "credit_spread_baa_aaa" not in df.columns:
            feats["credit_spread_baa_aaa"] = df["BAA10Y"] - df["AAA10Y"]
    
    # VIX Z-Scores
    if "VIXCLS" in df.columns:
        for w in [20, 60, 252]:
            if f"vix_z_{w}d" not in df.columns:
                roll_mean = df["VIXCLS"].rolling(w).mean()
                roll_std = df["VIXCLS"].rolling(w).std().replace(0, np.nan)
                feats[f"vix_z_{w}d"] = (df["VIXCLS"] - roll_mean) / roll_std
    
    # Regime Flags
    if "yield_spread_10y2y" in df.columns or "yield_spread_10y2y" in feats.columns:
        spread_col = "yield_spread_10y2y"
        spread_data = feats[spread_col] if spread_col in feats.columns else df[spread_col]
        if "regime_yield_inverted" not in df.columns:
            feats["regime_yield_inverted"] = (spread_data < 0).astype(int)
    
    if "VIXCLS" in df.columns:
        if "regime_high_vix" not in df.columns:
            feats["regime_high_vix"] = (df["VIXCLS"] > 25).astype(int)
        if "regime_extreme_vix" not in df.columns:
            feats["regime_extreme_vix"] = (df["VIXCLS"] > 35).astype(int)
    
    if "USRECD" in df.columns:
        if "regime_recession" not in df.columns:
            feats["regime_recession"] = df["USRECD"].fillna(0).astype(int)
    
    if not feats.empty:
        print(f"\n[INFO]: Added {len(feats.columns)} additional derived features")
        df = pd.concat([df, feats], axis=1)
    
    return df


# ============================================================
# MAIN
# ============================================================

def main():
    overall_start = now_ts()
    
    print("=" * 60)
    print("FRED DATA CLEANING PIPELINE")
    print("=" * 60)
    
    # ── Step 1: Load ──
    print("\n[Step 1/6] Loading data...")
    df, market_dates = load_data()
    
    # ── Step 2: Filter to trading days ──
    print("\n[Step 2/6] Filtering to NYSE trading days...")
    df_trading = filter_trading_days(df, market_dates)
    
    # ── Step 3: Summary statistics ──
    print("\n[Step 3/6] Generating summary statistics...")
    summary = generate_summary(df_trading)
    summary.to_csv(OUTPUT_DIR / "column_summary.csv", index=False)
    print(f"\nSummary saved: {OUTPUT_DIR / 'column_summary.csv'}")
    
    # ── Step 4: Drop bad columns ──
    print("\n[Step 4/6] Dropping low-coverage columns...")
    df_clean = drop_bad_columns(df_trading, summary, min_coverage=80.0)    
    
    # ── Step 5: Handle missing ──
    print("\n[Step 5/6] Handling remaining missing values...")
    df_final = handle_missing(df_clean)
    
    # ── Step 6: Additional features ──
    print("\n[Step 6/6] Engineering additional features...")
    df_final = engineer_additional_features(df_final)
    
    # ── Save ──
    output_path = OUTPUT_DIR / "macro_features_trading_days_clean.csv"
    df_final.to_csv(output_path)
    
    # ── Summary ──
    total_elapsed = now_ts() - overall_start
    
    print(f"\n{'=' * 60}")
    print(f"CLEANING PIPELINE COMPLETE — {fmt_elapsed(total_elapsed)}")
    print(f"{'=' * 60}")
    print(f"Output: {output_path}")
    print(f"Shape: {df_final.shape}")
    print(f"Date range: {df_final.index[0].strftime('%Y-%m-%d')} → {df_final.index[-1].strftime('%Y-%m-%d')}")
    print(f"Trading days: {len(df_final):,}")
    print(f"Features: {df_final.shape[1]}")
    
    # Save metadata
    metadata = {
        "date_range": [str(df_final.index[0]), str(df_final.index[-1])],
        "n_trading_days": len(df_final),
        "n_features": df_final.shape[1],
        "feature_names": list(df_final.columns),
        "dropped_columns": list(set(df_trading.columns) - set(df_final.columns)),
    }
    with open(OUTPUT_DIR / "cleaning_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Metadata: {OUTPUT_DIR / 'cleaning_metadata.json'}")
    
    # Final sample
    print(f"\nFinal sample (first 5 rows, first 10 columns):")
    print(df_final.iloc[:5, :10])


if __name__ == "__main__":
    main()