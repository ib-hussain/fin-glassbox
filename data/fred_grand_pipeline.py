#!/usr/bin/env python3
"""
FRED Grand Pipeline — Final Version

1. Downloads ALL ~11,000 daily FRED series (if not already downloaded)
2. Deletes series that DON'T cover 2000-01-01 to 2024-12-31
3. Filters to interpretable/important series
4. Builds a grand daily table
5. Engineers macro/regime features

Only CSV output. No parquet.

Usage:
    python data/fred_grand_pipeline.py                  # Use existing files
    python data/fred_grand_pipeline.py --download       # Download + process
    python data/fred_grand_pipeline.py --force-refresh  # Redownload everything
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
API_KEY = os.getenv("FRED_API_KEY_LB", "")
DEBUG_MODE = bool(int(os.getenv("DEBUG_MODE", 1)))

if not API_KEY:
    raise SystemExit("FRED_API_KEY_LB not found in .env file")

# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path("data/FRED_data")
RAW_DIR = DATA_PATH / "raw"
OUTPUT_DIR = DATA_PATH / "outputs"
FRED_LIST_FILE = DATA_PATH / "fred_daily_series_list.csv"

# ============================================================
# DATE RANGE (STRICT)
# ============================================================

TARGET_START = pd.Timestamp("2000-01-01")
TARGET_END = pd.Timestamp("2024-12-31")

# ============================================================
# SERIES SELECTION — Only keep interpretable, useful series
# ============================================================

# These MUST be available. If not in the daily tag, we download them separately.
REQUIRED_SERIES = [
    "VIXCLS",      # VIX (may not be tagged "daily")
    "T10Y2Y",      # 10Y-2Y spread
    "T10Y3M",      # 10Y-3M spread
    "USRECD",      # Recession indicator
    "TEDRATE",     # TED spread
    "DTWEXBGS",    # Trade-weighted dollar
]

KEEP_PATTERNS = [
    # === Treasury Yields ===
    "DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30",
    "DGS1MO", "DGS3MO", "DGS6MO",
    
    # === Treasury Bills ===
    "DTB1YR", "DTB3", "DTB6",
    
    # === Yield Spreads (pre-computed) ===
    "T10Y2Y", "T10Y3M", "T10YFF", "T1YFF", "T3MFF", "T5YFF", "T6MFF",
    
    # === Corporate Bonds ===
    "AAA10Y", "AAAFF", "BAA10Y", "BAAFF", "DAAA", "DBAA",
    
    # === Stock Indices ===
    "NASDAQCOM", "NASDAQ100", "NIKKEI225",
    
    # === Exchange Rates ===
    "DEXUSEU", "DEXJPUS", "DEXUSUK", "DEXCAUS", "DEXCHUS",
    "DEXMXUS", "DEXKOUS", "DEXINUS",
    
    # === Commodities ===
    "DCOILWTICO", "DCOILBRENTEU", "DDFUELLA",
    "DGASNYH", "DGASUSGULF", "DHOILNYH",
    "DPROPANEMBTX", "DHHNGSP",
    
    # === Volatility ===
    "VIXCLS", "VXDCLS",
    
    # === Central Bank Rates ===
    "DFF", "DPRIME", "ECBDFR", "ECBMLFR", "ECBMRRFR",
    
    # === Policy Uncertainty ===
    "USEPUINDXD", "WLEMUINDXD",
    
    # === Recession ===
    "USRECD", "USRECDM", "USRECDP",
    
    # === Weekly/Monthly ===
    "ICSA", "MORTGAGE30US",
    
    # === Commercial Paper ===
    "CPFF", "DCPF1M", "DCPF2M", "DCPF3M", "DCPN2M", "DCPN30", "DCPN3M",
    
    # === Japan Rates ===
    "JPINTDDMEJPY", "JPINTDEXR", "JPINTDUSDJPY", "JPINTDUSDRP",
    
    # === TED Spread ===
    "TEDRATE",
    
    # === Trade-Weighted Dollar ===
    "DTWEXBGS",
    
    # === Other useful ===
    "KCPRU", "DLTIIT", "IUDSOIA", "INFECTDISEMVTRACKD",
]

# Series to ALWAYS exclude (even if they match KEEP_PATTERNS partially)
EXCLUDE_PATTERNS = [
    "BAML", "NASDAQNQ", "NASDAQHX", "NASDAQCX", "NASDAQSX",
    "NASDAQIX", "NASDAQB", "THREEFF", "THREEFY", "RIFSPP",
    "RPAGYD", "RPMBSD", "RPON", "RPTM", "RPTSYD", "RPTTLD",
    "DEXBZUS", "DEXDNUS", "DEXHKUS", "DEXMAUS", "DEXNOUS",
    "DEXSDUS", "DEXSFUS", "DEXSIUS", "DEXSLUS", "DEXSZUS",
    "DEXTAUS", "DEXTHUS", "DEXUSAL", "DEXUSNZ", "DEXVZUS",
    "DJFUELUSGULF",
]


def should_keep(series_id: str) -> bool:
    """Check if a series should be kept based on patterns."""
    for pattern in EXCLUDE_PATTERNS:
        if series_id.startswith(pattern):
            return False
    for pattern in KEEP_PATTERNS:
        if series_id == pattern or series_id.startswith(pattern):
            return True
    return False


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
# STEP 0: DOWNLOAD ALL DAILY FRED SERIES
# ============================================================

def get_daily_series_list(api_key: str) -> list[str]:
    """Fetch list of ALL daily FRED series IDs from the API."""
    base_url = "https://api.stlouisfed.org/fred/tags/series"
    params = {
        "api_key": api_key,
        "tag_names": "daily",
        "file_type": "json",
        "limit": 1000,
        "offset": 0,
    }
    
    all_series = []
    
    while True:
        try:
            resp = requests.get(base_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            if "seriess" in data:
                chunk = data["seriess"]
                all_series.extend(chunk)
                if len(chunk) < params["limit"]:
                    break
                params["offset"] += params["limit"]
            else:
                break
        except Exception as e:
            print(f"Error fetching series list: {e}")
            break
    
    series_ids = sorted(set(s["id"] for s in all_series))
    return series_ids


def download_single_series(series_id: str) -> Optional[pd.DataFrame]:
    """Download a single FRED series."""
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        f"&cosd=2000-01-01&coed=2024-12-31"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text), parse_dates=True)
        if len(df.columns) >= 2:
            df = df.iloc[:, :2]
            df.columns = ["observation_date", series_id]
            df["observation_date"] = pd.to_datetime(df["observation_date"])
            return df
        return None
    except Exception as e:
        if DEBUG_MODE:
            print(f"[DEBUG]: Download failed for {series_id}: {e}")
        return None


def download_all_series(force_refresh: bool = False) -> None:
    """Download all daily FRED series + required series."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    existing = list(RAW_DIR.glob("*.csv"))
    print(f"Existing files in {RAW_DIR}: {len(existing):,}")
    
    # Get list of all daily-tagged series
    series_ids = get_daily_series_list(API_KEY)
    print(f"Total daily FRED series available: {len(series_ids):,}")
    
    # Add required series that might not be tagged "daily"
    all_to_download = set(series_ids)
    for sid in REQUIRED_SERIES:
        if sid not in all_to_download:
            all_to_download.add(sid)
            if DEBUG_MODE:
                print(f"[DEBUG]: Adding required series not in daily tag: {sid}")
    
    all_to_download = sorted(all_to_download)
    
    # Save the list
    pd.DataFrame({"id": all_to_download}).to_csv(FRED_LIST_FILE, index=False)
    print(f"Saved series list to: {FRED_LIST_FILE}")
    
    if not force_refresh and len(existing) > 5000:
        # Only download missing required series
        missing_required = [s for s in REQUIRED_SERIES if not (RAW_DIR / f"{s}.csv").exists()]
        if missing_required:
            print(f"Downloading {len(missing_required)} missing required series: {missing_required}")
            for sid in tqdm(missing_required, desc="Downloading required"):
                df = download_single_series(sid)
                if df is not None and len(df) > 0:
                    df.to_csv(RAW_DIR / f"{sid}.csv", index=False)
                time.sleep(0.1)
        else:
            print("All required series present. Skipping download.")
        return
    
    # Download all missing series
    new_downloads = 0
    for sid in tqdm(all_to_download, desc="Downloading FRED series"):
        csv_path = RAW_DIR / f"{sid}.csv"
        if csv_path.exists() and not force_refresh:
            continue
        
        df = download_single_series(sid)
        if df is not None and len(df) > 0:
            df.to_csv(csv_path, index=False)
            new_downloads += 1
        time.sleep(0.02)
    
    print(f"New downloads: {new_downloads:,}")
    final_count = len(list(RAW_DIR.glob("*.csv")))
    print(f"Total files in {RAW_DIR}: {final_count:,}")


# ============================================================
# STEP 1: DELETE SERIES THAT DON'T COVER 2000-2024
# ============================================================

def delete_incomplete_series() -> tuple[int, int]:
    """
    Scan ALL CSV files, delete those that don't cover 2000-01-01 to 2024-12-31.
    """
    print(f"\n{'=' * 60}")
    print("SCANNING & DELETING SERIES OUTSIDE 2000-2024 RANGE")
    print(f"{'=' * 60}")
    
    csv_files = sorted(RAW_DIR.glob("*.csv"))
    kept = 0
    deleted = 0
    trash_dir = DATA_PATH / "deleted_incomplete"
    trash_dir.mkdir(exist_ok=True)
    
    for csv_path in tqdm(csv_files, desc="Checking date ranges"):
        try:
            # Read just the date column
            df = pd.read_csv(csv_path)
            date_col = None
            for col in df.columns:
                if "date" in col.lower():
                    date_col = col
                    break
            if date_col is None:
                shutil.move(str(csv_path), str(trash_dir / csv_path.name))
                deleted += 1
                continue
            
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col])
            
            if df.empty:
                shutil.move(str(csv_path), str(trash_dir / csv_path.name))
                deleted += 1
                continue
            
            actual_start = df[date_col].min()
            actual_end = df[date_col].max()
            
            starts_ok = actual_start <= TARGET_START + pd.Timedelta(days=15)
            ends_ok = actual_end >= TARGET_END
            
            if starts_ok and ends_ok:
                kept += 1
            else:
                shutil.move(str(csv_path), str(trash_dir / csv_path.name))
                deleted += 1
        
        except Exception:
            shutil.move(str(csv_path), str(trash_dir / csv_path.name))
            deleted += 1
    
    print(f"\nKept (2000-2024): {kept}")
    print(f"Deleted (moved to {trash_dir}): {deleted}")
    return kept, deleted


# ============================================================
# STEP 2: FILTER TO USEFUL SERIES
# ============================================================

def filter_useful_series() -> list[str]:
    """Filter remaining series to only the useful/interpretable ones."""
    csv_files = sorted(RAW_DIR.glob("*.csv"))
    
    kept_series = []
    excluded_series = []
    
    for csv_path in csv_files:
        sid = csv_path.stem
        if should_keep(sid):
            kept_series.append(sid)
        else:
            excluded_series.append(sid)
    
    print(f"\n{'=' * 60}")
    print("FILTERING TO INTERPRETABLE SERIES")
    print(f"{'=' * 60}")
    print(f"Remaining after date check: {len(csv_files)}")
    print(f"Kept (useful): {len(kept_series)}")
    print(f"Excluded (redundant/granular): {len(excluded_series)}")
    
    if DEBUG_MODE:
        print(f"\n[DEBUG] Kept: {', '.join(sorted(kept_series))}")
    
    return kept_series


# ============================================================
# STEP 3: BUILD GRAND TABLE
# ============================================================

def build_grand_table(series_ids: list[str]) -> pd.DataFrame:
    """Load kept series and merge into a single daily table."""
    daily_cal = pd.date_range(start="2000-01-01", end="2024-12-31", freq="D")
    grand = pd.DataFrame({"date": daily_cal})
    
    loaded = 0
    failed = 0
    missing_files = []
    
    for sid in tqdm(series_ids, desc="Building grand table"):
        csv_path = RAW_DIR / f"{sid}.csv"
        
        if not csv_path.exists():
            missing_files.append(sid)
            failed += 1
            continue
        
        try:
            df = pd.read_csv(csv_path)
            
            # Find date column
            date_col = None
            for col in df.columns:
                if "date" in col.lower():
                    date_col = col
                    break
            
            if date_col is None:
                failed += 1
                continue
            
            # The value column is whatever is NOT the date column
            val_cols = [c for c in df.columns if c != date_col]
            if not val_cols:
                failed += 1
                continue
            val_col = val_cols[0]
            
            # Rename value column to series ID for clarity
            df = df.rename(columns={val_col: sid})
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col])
            
            # Keep only date and value
            df = df[[date_col, sid]]
            
            # Merge into grand table
            grand = grand.merge(df, on=date_col, how="left")
            loaded += 1
        
        except Exception as e:
            if DEBUG_MODE:
                print(f"[DEBUG]: Error loading {sid}: {e}")
            failed += 1
    
    grand = grand.set_index("date")
    
    print(f"\nLoaded: {loaded} | Failed: {failed}")
    if missing_files:
        print(f"Missing files: {missing_files}")
    print(f"Grand table shape: {grand.shape}")
    
    return grand


# ============================================================
# STEP 4: SMART FORWARD-FILL
# ============================================================

def smart_fill(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill with limits based on series frequency."""
    result = df.copy()
    
    # Determine fill limit by counting non-NaN values
    for col in df.columns:
        n_valid = df[col].count()
        total_days = len(df)
        ratio = n_valid / total_days
        
        if ratio < 0.05:   # ~300 values over 25 years = monthly
            result[col] = df[col].ffill(limit=90)
        elif ratio < 0.15:  # ~1300 values = weekly
            result[col] = df[col].ffill(limit=7)
        # Daily (>15% coverage): no fill needed
    
    return result


# ============================================================
# STEP 5: FEATURE ENGINEERING
# ============================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived macro/regime features."""
    feats = pd.DataFrame(index=df.index)
    
    # Yield Spreads
    if all(c in df.columns for c in ["DGS10", "DGS2"]):
        feats["yield_spread_10y2y"] = df["DGS10"] - df["DGS2"]
    if all(c in df.columns for c in ["DGS10", "DGS3MO"]):
        feats["yield_spread_10y3m"] = df["DGS10"] - df["DGS3MO"]
    if all(c in df.columns for c in ["DGS2", "DFF"]):
        feats["rate_spread_2y_fed"] = df["DGS2"] - df["DFF"]
    if all(c in df.columns for c in ["BAA10Y", "AAA10Y"]):
        feats["credit_spread_baa_aaa"] = df["BAA10Y"] - df["AAA10Y"]
    if all(c in df.columns for c in ["BAA10Y", "DGS10"]):
        feats["credit_spread_corp_treasury"] = df["BAA10Y"] - df["DGS10"]
    
    # VIX Z-Scores
    if "VIXCLS" in df.columns:
        for w in [20, 60, 252]:
            roll_mean = df["VIXCLS"].rolling(w).mean()
            roll_std = df["VIXCLS"].rolling(w).std().replace(0, np.nan)
            feats[f"vix_z_{w}d"] = (df["VIXCLS"] - roll_mean) / roll_std
    
    # Dollar/Yen/Oil Changes
    if "DEXUSEU" in df.columns:
        feats["dollar_chg_20d"] = df["DEXUSEU"].pct_change(20) * 100
    if "DEXJPUS" in df.columns:
        feats["yen_chg_20d"] = df["DEXJPUS"].pct_change(20) * 100
    if "DCOILWTICO" in df.columns:
        feats["oil_chg_20d"] = df["DCOILWTICO"].pct_change(20) * 100
    
    # Stock Returns
    for idx in ["NASDAQCOM", "NASDAQ100", "NIKKEI225"]:
        if idx in df.columns:
            feats[f"{idx}_ret_20d"] = df[idx].pct_change(20) * 100
    
    # Regime Flags
    if "yield_spread_10y2y" in feats.columns:
        feats["regime_yield_inverted"] = (feats["yield_spread_10y2y"] < 0).astype(int)
    if "VIXCLS" in df.columns:
        feats["regime_high_vix"] = (df["VIXCLS"] > 25).astype(int)
        feats["regime_extreme_vix"] = (df["VIXCLS"] > 35).astype(int)
    if "USRECD" in df.columns:
        feats["regime_recession"] = df["USRECD"].fillna(0).astype(int)
    
    # TED Spread Z-Score
    if "TEDRATE" in df.columns:
        roll_mean = df["TEDRATE"].rolling(60).mean()
        roll_std = df["TEDRATE"].rolling(60).std().replace(0, np.nan)
        feats["ted_z_60d"] = (df["TEDRATE"] - roll_mean) / roll_std
    
    # Policy Uncertainty
    if "USEPUINDXD" in df.columns:
        feats["epu_chg_20d"] = df["USEPUINDXD"].pct_change(20) * 100
    
    if DEBUG_MODE:
        print(f"[DEBUG]: Engineered {len(feats.columns)} derived features")
    
    result = pd.concat([df, feats], axis=1)
    return result


# ============================================================
# STEP 6: FINAL CLEAN
# ============================================================

def final_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns with >60% missing, forward-fill, truncate start."""
    if df.empty:
        print("ERROR: Empty DataFrame. Cannot clean.")
        return df
    
    missing_pct = df.isna().mean()
    cols_to_drop = missing_pct[missing_pct > 0.60].index.tolist()
    
    if DEBUG_MODE and cols_to_drop:
        print(f"[DEBUG]: Dropping {len(cols_to_drop)} columns (>60% missing)")
    
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    if df.empty:
        print("ERROR: All columns dropped. Check your data.")
        return df
    
    df = df.ffill()
    
    completeness = df.notna().mean(axis=1)
    valid_mask = completeness > 0.3
    
    if valid_mask.any():
        valid_start = completeness[valid_mask].index[0]
        df = df[df.index >= valid_start]
    
    df = df.dropna()
    
    if DEBUG_MODE:
        print(f"[DEBUG]: Final shape: {df.shape}")
    
    return df


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="FRED Grand Pipeline")
    parser.add_argument("--download", action="store_true", help="Download missing series")
    parser.add_argument("--force-refresh", action="store_true", help="Redownload all series")
    args = parser.parse_args()
    
    overall_start = now_ts()
    
    print("=" * 60)
    print("FRED GRAND PIPELINE — Complete")
    print(f"Date range: {TARGET_START.strftime('%Y-%m-%d')} → {TARGET_END.strftime('%Y-%m-%d')}")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ── Step 0: Download ──
#    if args.download or args.force_refresh:
#        print("\n[Step 0/6] Downloading FRED daily series...")
#        t0 = now_ts()
#        download_all_series(force_refresh=args.force_refresh)
#        print(f"  Done in {fmt_elapsed(now_ts() - t0)}")
#    else:
#        existing = len(list(RAW_DIR.glob("*.csv")))
#        print(f"\n[Step 0/6] Skipping download. {existing:,} files already exist.")
#        print("  Use --download to download missing series.")
    
    # # ── Step 1: Delete incomplete ──
    # print("\n[Step 1/6] Deleting series outside 2000-2024 range...")
    # t0 = now_ts()
    # kept, deleted = delete_incomplete_series()
    # print(f"  Kept: {kept} | Deleted: {deleted}")
    # print(f"  Done in {fmt_elapsed(now_ts() - t0)}")
    
    # ── Step 2: Filter useful ──
    print("\n[Step 2/6] Filtering to interpretable series...")
    t0 = now_ts()
    useful_series = filter_useful_series()
    print(f"  Done in {fmt_elapsed(now_ts() - t0)}")
    
    if not useful_series:
        print("ERROR: No useful series found. Check your KEEP_PATTERNS.")
        return
    
    # ── Step 3: Build grand table ──
    print(f"\n[Step 3/6] Building grand table from {len(useful_series)} series...")
    t0 = now_ts()
    grand = build_grand_table(useful_series)
    print(f"  Done in {fmt_elapsed(now_ts() - t0)}")
    
    if grand.empty or grand.shape[1] == 0:
        print("ERROR: Grand table is empty. Cannot proceed.")
        return
    
    # Save raw grand table
    grand.to_csv(OUTPUT_DIR / "grand_table_raw.csv")
    
    # ── Step 4: Smart fill ──
    print("\n[Step 4/6] Smart forward-filling...")
    t0 = now_ts()
    grand_filled = smart_fill(grand)
    print(f"  Done in {fmt_elapsed(now_ts() - t0)}")
    
    # ── Step 5: Feature engineering ──
    print("\n[Step 5/6] Engineering derived features...")
    t0 = now_ts()
    all_features = engineer_features(grand_filled)
    all_features.to_csv(OUTPUT_DIR / "macro_all_features.csv")
    print(f"  Total columns: {len(all_features.columns)}")
    print(f"  Done in {fmt_elapsed(now_ts() - t0)}")
    
    # ── Step 6: Final clean ──
    print("\n[Step 6/6] Final cleaning...")
    t0 = now_ts()
    clean = final_clean(all_features)
    
    # if clean.empty:
    #     print("ERROR: Final DataFrame is empty after cleaning.")
    #     return
    
    # clean.to_csv(OUTPUT_DIR / "daily_macro_features_clean.csv")
    # print(f"  Final shape: {clean.shape}")
    # print(f"  Date range: {clean.index[0].strftime('%Y-%m-%d')} → {clean.index[-1].strftime('%Y-%m-%d')}")
    # print(f"  Done in {fmt_elapsed(now_ts() - t0)}")
    
    # ── Summary ──
    total_elapsed = now_ts() - overall_start
    
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE — {fmt_elapsed(total_elapsed)}")
    print(f"Output: {OUTPUT_DIR / 'daily_macro_features_clean.csv'}")
    print(f"Series kept: {len(useful_series)}")
    print(f"Final features: {clean.shape[1]}")
    print(f"Final rows: {clean.shape[0]}")
    print(f"{'=' * 60}")
    
    # Save metadata
    metadata = {
        "date_range": [str(clean.index[0]), str(clean.index[-1])],
        "n_series_kept": len(useful_series),
        "n_features_final": int(clean.shape[1]),
        "n_rows": int(clean.shape[0]),
        "series_list": sorted(useful_series),
        "feature_names": list(clean.columns),
    }
    with open(OUTPUT_DIR / "pipeline_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata: {OUTPUT_DIR / 'pipeline_metadata.json'}")
    
    # Sample
    print("\nSample (first 5 rows, first 10 columns):")
    print(clean.iloc[:5, :10])


if __name__ == "__main__":
    main()

