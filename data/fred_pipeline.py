#!/usr/bin/env python3
"""
FRED Macro/Regime Pipeline — Complete Rebuild

Downloads, cleans, aligns, and engineers macro/regime features from FRED.
Only keeps series with sufficient coverage (2000-2024).
CSV-only output. No parquet files.

Usage:
    python data/fred_pipeline.py --full-run          # Download + process
    python data/fred_pipeline.py --skip-download     # Use cached CSVs
    python data/fred_pipeline.py --force-refresh     # Redownload everything
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY_LB", "")
DEBUG_MODE = bool(int(os.getenv("DEBUG_MODE", 0)))

if not FRED_API_KEY:
    raise SystemExit("FRED_API_KEY_LB not found in .env file")

# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path("data/FRED_data")
RAW_DIR = DATA_PATH / "raw" / "series_csv"
PROCESSED_DIR = DATA_PATH / "processed"
TRANSFORMED_DIR = DATA_PATH / "transformed"
ALIGNED_DIR = DATA_PATH / "aligned"

# ============================================================
# CONFIGURATION
# ============================================================

START_DATE = "2000-01-01"
END_DATE = "2024-12-31"

# Minimum fraction of trading days a series must have to be kept
MIN_COVERAGE = 0.30  # 30% of days must have data

# Maximum forward-fill days for monthly series
MAX_FORWARD_FILL_DAYS = 90

# ============================================================
# FRED SERIES DEFINITIONS (Expanded — Only High-Quality)
# ============================================================

FRED_SERIES: dict[str, dict[str, str]] = {
    # === Interest Rates (Daily) ===
    "FEDFUNDS":    {"freq": "Monthly", "cat": "rates", "desc": "Federal Funds Rate"},
    "DGS1MO":      {"freq": "Daily",   "cat": "rates", "desc": "1-Month Treasury"},
    "DGS3MO":      {"freq": "Daily",   "cat": "rates", "desc": "3-Month Treasury"},
    "DGS6MO":      {"freq": "Daily",   "cat": "rates", "desc": "6-Month Treasury"},
    "DGS1":        {"freq": "Daily",   "cat": "rates", "desc": "1-Year Treasury"},
    "DGS2":        {"freq": "Daily",   "cat": "rates", "desc": "2-Year Treasury"},
    "DGS5":        {"freq": "Daily",   "cat": "rates", "desc": "5-Year Treasury"},
    "DGS10":       {"freq": "Daily",   "cat": "rates", "desc": "10-Year Treasury"},
    "DGS30":       {"freq": "Daily",   "cat": "rates", "desc": "30-Year Treasury"},

    # === Inflation (Monthly → forward-filled) ===
    "CPIAUCSL":    {"freq": "Monthly", "cat": "inflation", "desc": "CPI All Items"},
    "CPILFESL":    {"freq": "Monthly", "cat": "inflation", "desc": "CPI Core"},
    "PCEPI":       {"freq": "Monthly", "cat": "inflation", "desc": "PCE Price Index"},
    "PCEPILFE":    {"freq": "Monthly", "cat": "inflation", "desc": "PCE Core"},
    "PPIACO":      {"freq": "Monthly", "cat": "inflation", "desc": "PPI All Commodities"},

    # === Labor Market (Monthly → forward-filled) ===
    "UNRATE":      {"freq": "Monthly", "cat": "labor", "desc": "Unemployment Rate"},
    "PAYEMS":      {"freq": "Monthly", "cat": "labor", "desc": "Nonfarm Payrolls"},
    "ICSA":        {"freq": "Weekly",  "cat": "labor", "desc": "Initial Jobless Claims"},
    "CIVPART":     {"freq": "Monthly", "cat": "labor", "desc": "Labor Force Participation"},

    # === Economic Activity (Monthly → forward-filled) ===
    "INDPRO":      {"freq": "Monthly", "cat": "activity", "desc": "Industrial Production"},
    "HOUST":       {"freq": "Monthly", "cat": "activity", "desc": "Housing Starts"},
    "PERMIT":      {"freq": "Monthly", "cat": "activity", "desc": "Building Permits"},
    "DGORDER":     {"freq": "Monthly", "cat": "activity", "desc": "Durable Goods Orders"},
    "RSAFS":       {"freq": "Monthly", "cat": "activity", "desc": "Retail Sales"},

    # === Financial Stress (Daily) ===
    "VIXCLS":      {"freq": "Daily",   "cat": "stress", "desc": "VIX Volatility Index"},
    "TEDRATE":     {"freq": "Daily",   "cat": "stress", "desc": "TED Spread"},
    "DTWEXBGS":    {"freq": "Daily",   "cat": "stress", "desc": "Trade-Weighted USD"},
    "MORTGAGE30US":{"freq": "Weekly",  "cat": "stress", "desc": "30-Year Mortgage Rate"},

    # === Credit Spreads (Daily, limited history OK) ===
    "BAMLH0A0HYM2":{"freq": "Daily",   "cat": "credit", "desc": "HY OAS"},
    "BAMLC0A0CM":  {"freq": "Daily",   "cat": "credit", "desc": "IG OAS"},

    # === Regime Labels ===
    "USRECD":      {"freq": "Daily",   "cat": "regime", "desc": "NBER Recession Indicator"},

    # === Money & Credit ===
    "M2SL":        {"freq": "Monthly", "cat": "money", "desc": "M2 Money Supply"},
    "WALCL":       {"freq": "Weekly",  "cat": "money", "desc": "Fed Balance Sheet"},

    # === Consumer ===
    "UMCSENT":     {"freq": "Monthly", "cat": "consumer", "desc": "Consumer Sentiment"},
    "PSAVERT":     {"freq": "Monthly", "cat": "consumer", "desc": "Personal Savings Rate"},
}

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


def ensure_dirs() -> None:
    for d in [RAW_DIR, PROCESSED_DIR, TRANSFORMED_DIR, ALIGNED_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ============================================================
# STEP 1: DOWNLOAD
# ============================================================

def fetch_fred_series(series_id: str) -> pd.DataFrame:
    """Fetch a single FRED series from the API."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": START_DATE,
        "observation_end": END_DATE,
        "sort_order": "asc",
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        observations = data.get("observations", [])

        if not observations:
            if DEBUG_MODE:
                print(f"[DEBUG]: No observations for {series_id}")
            return pd.DataFrame()

        df = pd.DataFrame(observations)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df[["date", "value"]].dropna()
        df.columns = ["date", series_id]
        return df

    except Exception as e:
        print(f"  ERROR fetching {series_id}: {e}")
        return pd.DataFrame()


def download_all(force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Download all FRED series with resume support."""
    manifest_path = RAW_DIR.parent / "download_manifest.json"
    series_data: dict[str, pd.DataFrame] = {}

    # Load existing manifest
    manifest: dict[str, Any] = {}
    if manifest_path.exists() and not force_refresh:
        with open(manifest_path) as f:
            manifest = json.load(f)

    for series_id, info in tqdm(FRED_SERIES.items(), desc="Downloading FRED series"):
        csv_path = RAW_DIR / f"{series_id}.csv"

        # Use cached if available (unless force refresh)
        if not force_refresh and csv_path.exists() and series_id in manifest:
            if DEBUG_MODE:
                print(f"[DEBUG]: Using cached {series_id}")
            series_data[series_id] = pd.read_csv(csv_path, parse_dates=["date"])
            continue

        # Download
        df = fetch_fred_series(series_id)
        if not df.empty:
            df.to_csv(csv_path, index=False)
            series_data[series_id] = df

            manifest[series_id] = {
                "downloaded_at": datetime.now().isoformat(),
                "n_observations": len(df),
                "date_range": [df["date"].min().isoformat(), df["date"].max().isoformat()],
            }

    # Save manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDownloaded/Loaded {len(series_data)} series")
    return series_data


# ============================================================
# STEP 2: BUILD MASTER TABLE + FILTER SERIES (SMART)
# ============================================================

def build_master(series_dict: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine all series into master table.
    Judges coverage by NATIVE frequency, not daily.
    Returns: (master_df, metadata_df)
    """
    # Create daily calendar
    daily_cal = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
    master = pd.DataFrame({"date": daily_cal})

    for series_id, df in series_dict.items():
        df = df.rename(columns={series_id: series_id})
        master = master.merge(df, on="date", how="left")

    master = master.set_index("date")
    total_days = len(master)
    total_months = 300  # 25 years × 12 months
    total_weeks = 1300  # ~25 years × 52 weeks
    total_years = 25

    metadata_rows = []
    for series_id in master.columns:
        n_valid = master[series_id].count()
        freq = FRED_SERIES.get(series_id, {}).get("freq", "Daily")
        desc = FRED_SERIES.get(series_id, {}).get("desc", "")

        # Judge coverage by NATIVE frequency
        if freq == "Monthly":
            expected = total_months
        elif freq == "Weekly":
            expected = total_weeks
        elif freq == "Quarterly":
            expected = total_years * 4
        else:  # Daily
            expected = total_days

        coverage = n_valid / expected if expected > 0 else 0

        # Determine if we should keep it
        # Daily: need 20%+ coverage (started late OK if 5+ years)
        # Monthly: need 60%+ coverage (most start 2000)
        # Weekly: need 40%+ coverage
        if freq == "Monthly":
            keep = coverage >= 0.60
        elif freq == "Weekly":
            keep = coverage >= 0.40
        elif freq == "Quarterly":
            keep = coverage >= 0.50
        else:  # Daily
            keep = coverage >= 0.20 and n_valid >= 1250  # At least 5 years

        # EXCEPTIONS: Always keep if important enough
        ALWAYS_KEEP = ["USRECD", "VIXCLS", "FEDFUNDS", "UNRATE", "CPIAUCSL", 
                       "DGS10", "DGS2", "DGS3MO", "TEDRATE", "DGS1MO"]
        if series_id in ALWAYS_KEEP:
            keep = True

        # EXCEPTIONS: Drop if coverage is truly terrible
        if n_valid < 20:  # Less than 20 data points = useless
            keep = False

        metadata_rows.append({
            "series_id": series_id,
            "desc": desc,
            "category": FRED_SERIES.get(series_id, {}).get("cat", "unknown"),
            "frequency": freq,
            "n_valid": n_valid,
            "native_coverage_pct": coverage * 100,
            "keep": keep,
        })

    metadata = pd.DataFrame(metadata_rows)

    # Filter to keep only selected series
    keep_cols = metadata[metadata["keep"]]["series_id"].tolist()
    dropped_cols = metadata[~metadata["keep"]]["series_id"].tolist()

    print(f"\n  Coverage Analysis:")
    print(f"  {'─' * 70}")
    print(f"  {'Series':<16} {'Freq':<10} {'Valid':>8} {'Native Cov%':>12} {'Keep':>6}")
    print(f"  {'─' * 70}")
    for _, row in metadata.iterrows():
        status = "✅" if row["keep"] else "❌"
        print(f"  {row['series_id']:<16} {row['frequency']:<10} {row['n_valid']:>8} {row['native_coverage_pct']:>11.1f}% {status:>6}")
    print(f"  {'─' * 70}")
    print(f"  Keeping {len(keep_cols)}/{len(metadata)} series")
    if dropped_cols:
        print(f"  Dropped: {', '.join(dropped_cols)}")

    master_filtered = master[keep_cols].copy()

    return master_filtered, metadata

# ============================================================
# STEP 3: FORWARD-FILL LOWER FREQUENCY DATA
# ============================================================

def smart_forward_fill(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill with time limit.
    Daily series: no fill needed (already daily)
    Monthly series: fill up to MAX_FORWARD_FILL_DAYS
    Weekly series: fill up to 7 days
    """
    result = df.copy()

    for col in df.columns:
        info = FRED_SERIES.get(col, {})
        freq = info.get("freq", "Daily")

        if freq == "Daily":
            continue  # Already daily, no fill needed

        max_fill = MAX_FORWARD_FILL_DAYS if freq == "Monthly" else 7

        # Forward fill with limit
        result[col] = df[col].ffill(limit=max_fill)

        if DEBUG_MODE:
            before = df[col].count()
            after = result[col].count()
            print(f"[DEBUG]: {col} ({freq}): {before} → {after} valid after fill")

    return result


# ============================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived macro features."""
    feats = pd.DataFrame(index=df.index)

    # --- Yield Spreads ---
    if all(c in df.columns for c in ["DGS10", "DGS2"]):
        feats["yield_spread_10y2y"] = df["DGS10"] - df["DGS2"]
    if all(c in df.columns for c in ["DGS10", "DGS3MO"]):
        feats["yield_spread_10y3m"] = df["DGS10"] - df["DGS3MO"]
    if all(c in df.columns for c in ["DGS2", "FEDFUNDS"]):
        feats["rate_spread_2y_fed"] = df["DGS2"] - df["FEDFUNDS"]

    # --- VIX Z-Scores ---
    if "VIXCLS" in df.columns:
        for w in [20, 60, 252]:
            roll_mean = df["VIXCLS"].rolling(w).mean()
            roll_std = df["VIXCLS"].rolling(w).std().replace(0, np.nan)
            feats[f"vix_z_{w}d"] = (df["VIXCLS"] - roll_mean) / roll_std

    # --- TED Rate Z-Scores ---
    if "TEDRATE" in df.columns:
        for w in [20, 60]:
            roll_mean = df["TEDRATE"].rolling(w).mean()
            roll_std = df["TEDRATE"].rolling(w).std().replace(0, np.nan)
            feats[f"ted_z_{w}d"] = (df["TEDRATE"] - roll_mean) / roll_std

    # --- Inflation MoM (annualized) ---
    for col in ["CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE"]:
        if col in df.columns:
            feats[f"{col}_mom_ann"] = df[col].pct_change() * 100 * 12
            feats[f"{col}_yoy"] = df[col].pct_change(12) * 100

    # --- Unemployment Change ---
    if "UNRATE" in df.columns:
        feats["unrate_change_1m"] = df["UNRATE"].diff()
        feats["unrate_change_3m"] = df["UNRATE"].diff(3)

    # --- Employment Growth ---
    if "PAYEMS" in df.columns:
        feats["payems_mom_pct"] = df["PAYEMS"].pct_change() * 100

    # --- Industrial Production ---
    if "INDPRO" in df.columns:
        feats["indpro_mom_pct"] = df["INDPRO"].pct_change() * 100

    # --- Fed Policy Stance ---
    if "FEDFUNDS" in df.columns:
        fed_ma = df["FEDFUNDS"].rolling(252).mean()
        feats["fed_tightening"] = (df["FEDFUNDS"] > fed_ma + 0.25).astype(int)
        feats["fed_easing"] = (df["FEDFUNDS"] < fed_ma - 0.25).astype(int)

    # --- Regime Flags ---
    if "yield_spread_10y2y" in feats.columns:
        feats["regime_yield_inverted"] = (feats["yield_spread_10y2y"] < 0).astype(int)
    if "VIXCLS" in df.columns:
        feats["regime_high_vix"] = (df["VIXCLS"] > 25).astype(int)
        feats["regime_extreme_vix"] = (df["VIXCLS"] > 35).astype(int)
    if "USRECD" in df.columns:
        feats["regime_recession"] = df["USRECD"].fillna(0).astype(int)

    # --- Dollar Strength Change ---
    if "DTWEXBGS" in df.columns:
        feats["dollar_mom_pct"] = df["DTWEXBGS"].pct_change() * 100

    if DEBUG_MODE:
        print(f"[DEBUG]: Engineered {len(feats.columns)} derived features")

    # Combine levels + derived
    result = pd.concat([df, feats], axis=1)
    return result


# ============================================================
# STEP 5: FINAL CLEANING
# ============================================================

def final_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns with too many NaN, forward-fill remaining, drop early period."""
    # Drop columns with >60% missing after engineering
    missing_pct = df.isna().mean()
    cols_to_drop = missing_pct[missing_pct > 0.60].index.tolist()

    if DEBUG_MODE and cols_to_drop:
        print(f"[DEBUG]: Dropping {len(cols_to_drop)} columns with >60% missing:")
        for c in cols_to_drop:
            print(f"         {c}: {missing_pct[c]*100:.1f}% missing")

    df = df.drop(columns=cols_to_drop)

    # Forward-fill remaining NaN
    df = df.ffill()

    # Start from first date with at least 50% of columns non-NaN
    completeness = df.notna().mean(axis=1)
    valid_start = completeness[completeness > 0.5].index[0]
    df = df[df.index >= valid_start]

    # Drop any remaining NaN rows
    df = df.dropna()

    if DEBUG_MODE:
        print(f"[DEBUG]: Final shape: {df.shape}")
        print(f"[DEBUG]: Date range: {df.index[0]} to {df.index[-1]}")

    return df


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(skip_download: bool = False, force_refresh: bool = False) -> pd.DataFrame:
    """Execute the complete FRED macro pipeline."""
    overall_start = now_ts()

    print("=" * 60)
    print("FRED MACRO/REGIME PIPELINE")
    print(f"Date range: {START_DATE} → {END_DATE}")
    print(f"Series defined: {len(FRED_SERIES)}")
    print("=" * 60)

    ensure_dirs()

    # ── Step 1: Download ──
    print("\n[Step 1/5] Downloading FRED series...")
    t0 = now_ts()

    if skip_download:
        print("  Skipping download, loading from cache...")
        series_dict = {}
        for sid in FRED_SERIES:
            csv_path = RAW_DIR / f"{sid}.csv"
            if csv_path.exists():
                series_dict[sid] = pd.read_csv(csv_path, parse_dates=["date"])
        print(f"  Loaded {len(series_dict)} series from cache")
    else:
        series_dict = download_all(force_refresh=force_refresh)

    print(f"  Done in {fmt_elapsed(now_ts() - t0)}")

    # ── Step 2: Build Master + Filter ──
    print("\n[Step 2/5] Building master table & filtering series...")
    t0 = now_ts()

    master, metadata = build_master(series_dict)

    metadata.to_csv(PROCESSED_DIR / "series_metadata.csv", index=False)
    print(f"  Kept {len(master.columns)} series (coverage ≥ {MIN_COVERAGE*100:.0f}%)")
    print(f"  Done in {fmt_elapsed(now_ts() - t0)}")

    # ── Step 3: Forward-Fill ──
    print("\n[Step 3/5] Smart forward-filling lower-frequency series...")
    t0 = now_ts()

    master_filled = smart_forward_fill(master)

    print(f"  Done in {fmt_elapsed(now_ts() - t0)}")

    # ── Step 4: Feature Engineering ──
    print("\n[Step 4/5] Engineering derived features...")
    t0 = now_ts()

    all_features = engineer_features(master_filled)

    all_features.to_csv(TRANSFORMED_DIR / "macro_all_features.csv")
    print(f"  Total columns: {len(all_features.columns)}")
    print(f"  Done in {fmt_elapsed(now_ts() - t0)}")

    # ── Step 5: Final Clean ──
    print("\n[Step 5/5] Final cleaning...")
    t0 = now_ts()

    clean = final_clean(all_features)

    # Save final output
    clean.to_csv(ALIGNED_DIR / "daily_macro_features_clean.csv")
    print(f"  Final shape: {clean.shape}")
    print(f"  Date range: {clean.index[0].strftime('%Y-%m-%d')} → {clean.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Columns: {len(clean.columns)}")
    print(f"  Done in {fmt_elapsed(now_ts() - t0)}")

    # ── Summary ──
    total_elapsed = now_ts() - overall_start
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE")
    print(f"Total time: {fmt_elapsed(total_elapsed)}")
    print(f"Output: {ALIGNED_DIR / 'daily_macro_features_clean.csv'}")
    print(f"{'=' * 60}")

    # Print a sample
    print("\nSample (first 5 rows, first 10 columns):")
    print(clean.iloc[:5, :10])

    return clean


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FRED Macro/Regime Pipeline")
    parser.add_argument("--full-run", action="store_true", help="Download + process")
    parser.add_argument("--skip-download", action="store_true", help="Use cached CSVs only")
    parser.add_argument("--force-refresh", action="store_true", help="Redownload all series")
    args = parser.parse_args()

    run_pipeline(
        skip_download=args.skip_download,
        force_refresh=args.force_refresh,
    )