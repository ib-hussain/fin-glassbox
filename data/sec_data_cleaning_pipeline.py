#!/usr/bin/env python3
"""
SEC Data Cleaning Pipeline

Cleans and standardizes the three core SEC processed files:
1. cik_ticker_map.csv - Exchange filtering, missing value removal
2. issuer_master.csv - US-only filtering, exchange filtering, column pruning, ticker filling
3. fundamentals_features.csv - Deduplication, null column removal, ticker/name filling, outlier capping

All operations are performed in order and outputs are saved as cleaned versions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Optional, Set

# ============================================================
# PATHS
# ============================================================

dataPath = Path(os.getenv("dataPathGlobal", "data"))

# Input files
CIK_TICKER_MAP_IN = dataPath / "sec_edgar" / "processed" / "issuer_master" / "cik_ticker_map.csv"
ISSUER_MASTER_IN = dataPath / "sec_edgar" / "processed" / "issuer_master" / "issuer_master.csv"
FUNDAMENTALS_IN = dataPath / "sec_edgar" / "processed" / "fundamentals" / "fundamentals_features.csv"

# Output files
OUT_DIR = dataPath / "sec_edgar" / "processed" / "cleaned"
CIK_TICKER_MAP_OUT = OUT_DIR / "cik_ticker_map_cleaned.csv"
ISSUER_MASTER_OUT = OUT_DIR / "issuer_master_cleaned.csv"
FUNDAMENTALS_OUT = OUT_DIR / "fundamentals_features_cleaned.csv"
SUMMARY_OUT = OUT_DIR / "cleaning_summary.json"

# ============================================================
# VALID EXCHANGES
# ============================================================

VALID_EXCHANGES = {"Nasdaq", "NYSE"}

# ============================================================
# OUTLIER CAPS (1000% = 10.0 in decimal form)
# ============================================================

RATIO_CAPS = {
    # Profitability margins (-1000% to 1000%)
    "gross_margin": (-10.0, 10.0),
    "operating_margin": (-10.0, 10.0),
    "net_margin": (-10.0, 10.0),
    "opex_to_revenue": (-10.0, 10.0),
    "cogs_to_revenue": (-10.0, 10.0),
    
    # Returns (-1000% to 1000%)
    "roa": (-10.0, 10.0),
    "roe": (-10.0, 10.0),
    
    # Leverage (0 to 100)
    "debt_to_equity": (-10.0, 100.0),
    "debt_to_assets": (0.0, 10.0),
    "current_ratio": (0.0, 100.0),
    "quick_ratio": (0.0, 100.0),
    "cash_ratio": (0.0, 100.0),
    
    # Efficiency (0 to 100)
    "asset_turnover": (0.0, 100.0),
    "ppe_turnover": (0.0, 100.0),
    
    # Cash flow ratios (-1000% to 1000%)
    "ocf_to_revenue": (-10.0, 10.0),
    "fcf_to_revenue": (-10.0, 10.0),
    "capex_to_revenue": (-10.0, 10.0),
    "fcf_to_net_income": (-100.0, 100.0),
    "ocf_to_net_income": (-100.0, 100.0),
    
    # Per share (no cap needed, but keep reasonable)
    "revenue_per_share": (-1e9, 1e9),
    "book_value_per_share": (-1e9, 1e9),
    "ocf_per_share": (-1e9, 1e9),
    "fcf_per_share": (-1e9, 1e9),
    
    # Quality
    "accruals_to_assets": (-10.0, 10.0),
    
    # Growth rates (-1000% to 1000%)
    "revenue_growth_yoy": (-10.0, 10.0),
    "revenue_growth_qoq": (-10.0, 10.0),
    "net_income_growth_yoy": (-10.0, 10.0),
    "net_income_growth_qoq": (-10.0, 10.0),
    "operating_income_growth_yoy": (-10.0, 10.0),
    "operating_income_growth_qoq": (-10.0, 10.0),
    "eps_basic_growth_yoy": (-10.0, 10.0),
    "eps_basic_growth_qoq": (-10.0, 10.0),
    "total_assets_growth_yoy": (-10.0, 10.0),
    "total_assets_growth_qoq": (-10.0, 10.0),
}

# Columns to drop from fundamentals (100% null or not needed)
FUNDAMENTALS_DROP_COLUMNS = [
    "inventory",
    "goodwill", 
    "intangible_assets",
    "short_term_debt",
]


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


def safe_float(value: Any) -> Optional[float]:
    """Convert to float, return None if invalid/missing."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def cap_value(value: Any, cap_range: tuple[float, float]) -> str:
    """Cap a numeric value to the specified range."""
    if value is None or value == "":
        return ""
    
    try:
        num = float(value)
        min_val, max_val = cap_range
        if num < min_val:
            num = min_val
        elif num > max_val:
            num = max_val
        return str(num)
    except (ValueError, TypeError):
        return ""


# ============================================================
# STEP 1: Clean cik_ticker_map.csv
# ============================================================

def clean_cik_ticker_map() -> tuple[int, int, dict]:
    """
    Clean cik_ticker_map.csv:
    - Remove rows with missing primary_exchange
    - Keep only Nasdaq and NYSE exchanges
    """
    print("\n" + "=" * 60)
    print("STEP 1: Cleaning cik_ticker_map.csv")
    print("=" * 60)
    
    if not CIK_TICKER_MAP_IN.exists():
        raise FileNotFoundError(f"Input file not found: {CIK_TICKER_MAP_IN}")
    
    rows_in = 0
    rows_out = 0
    missing_exchange = 0
    invalid_exchange = 0
    
    # Build CIK -> ticker mapping for later use
    cik_to_ticker: dict[str, str] = {}
    cik_to_name: dict[str, str] = {}
    
    with CIK_TICKER_MAP_IN.open("r", newline="", encoding="utf-8") as inf:
        reader = csv.DictReader(inf)
        fieldnames = reader.fieldnames
        
        # Write output
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        tmp_out = CIK_TICKER_MAP_OUT.with_suffix(".csv.part")
        
        with tmp_out.open("w", newline="", encoding="utf-8") as outf:
            writer = csv.DictWriter(outf, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in reader:
                rows_in += 1
                exchange = row.get("primary_exchange", "").strip()
                cik = row.get("cik", "").strip()
                ticker = row.get("primary_ticker", "").strip()
                entity_name = row.get("entity_name", "").strip()
                
                # Check for missing exchange
                if not exchange:
                    missing_exchange += 1
                    continue
                
                # Check for valid exchange
                if exchange not in VALID_EXCHANGES:
                    invalid_exchange += 1
                    continue
                
                # Keep this row
                writer.writerow(row)
                rows_out += 1
                
                # Store mapping
                if cik:
                    cik_to_ticker[cik] = ticker
                    cik_to_name[cik] = entity_name
        
        tmp_out.replace(CIK_TICKER_MAP_OUT)
    
    print(f"  Input rows: {rows_in:,}")
    print(f"  Output rows: {rows_out:,}")
    print(f"  Removed - missing exchange: {missing_exchange:,}")
    print(f"  Removed - invalid exchange: {invalid_exchange:,}")
    print(f"  Output: {CIK_TICKER_MAP_OUT}")
    
    stats = {
        "cik_ticker_map": {
            "rows_in": rows_in,
            "rows_out": rows_out,
            "missing_exchange_removed": missing_exchange,
            "invalid_exchange_removed": invalid_exchange,
        }
    }
    
    return rows_out, cik_to_ticker, cik_to_name, stats


# ============================================================
# STEP 2: Clean issuer_master.csv
# ============================================================

def clean_issuer_master(cik_to_ticker: dict[str, str]) -> tuple[int, dict]:
    """
    Clean issuer_master.csv:
    - Keep only is_us_issuer == 1, then drop this column
    - Keep only Nasdaq, NYSE, or missing primary_exchange
    - Drop columns: all_tickers, padded_cik, entity_type, state_of_incorporation_desc, state_of_incorporation
    - Keep business_city
    - Fill missing tickers using cik_to_ticker mapping
    """
    print("\n" + "=" * 60)
    print("STEP 2: Cleaning issuer_master.csv")
    print("=" * 60)
    
    if not ISSUER_MASTER_IN.exists():
        raise FileNotFoundError(f"Input file not found: {ISSUER_MASTER_IN}")
    
    rows_in = 0
    rows_out = 0
    non_us_removed = 0
    invalid_exchange_removed = 0
    tickers_filled = 0
    
    columns_to_drop = {
        "all_tickers", "padded_cik", "entity_type",
        "state_of_incorporation_desc", "state_of_incorporation", "business_state", "mailing_city", "mailing_state"
    }
    
    with ISSUER_MASTER_IN.open("r", newline="", encoding="utf-8") as inf:
        reader = csv.DictReader(inf)
        original_fieldnames = list(reader.fieldnames or [])
        
        # Determine output fieldnames
        output_fieldnames = [f for f in original_fieldnames if f not in columns_to_drop]
        if "is_us_issuer" in output_fieldnames:
            output_fieldnames.remove("is_us_issuer")
        
        tmp_out = ISSUER_MASTER_OUT.with_suffix(".csv.part")
        
        with tmp_out.open("w", newline="", encoding="utf-8") as outf:
            writer = csv.DictWriter(outf, fieldnames=output_fieldnames)
            writer.writeheader()
            
            for row in reader:
                rows_in += 1
                
                # Filter: keep only US issuers
                is_us = row.get("is_us_issuer", "").strip()
                if is_us != "1":
                    non_us_removed += 1
                    continue
                
                # Filter: keep only valid exchanges or missing
                exchange = row.get("primary_exchange", "").strip()
                if exchange and exchange not in VALID_EXCHANGES:
                    invalid_exchange_removed += 1
                    continue
                
                # Fill missing ticker if possible
                cik = row.get("cik", "").strip()
                ticker = row.get("primary_ticker", "").strip()
                if not ticker and cik and cik in cik_to_ticker:
                    row["primary_ticker"] = cik_to_ticker[cik]
                    tickers_filled += 1
                
                # Build output row (drop specified columns and is_us_issuer)
                out_row = {k: v for k, v in row.items() 
                          if k not in columns_to_drop and k != "is_us_issuer"}
                
                writer.writerow(out_row)
                rows_out += 1
        
        tmp_out.replace(ISSUER_MASTER_OUT)
    
    print(f"  Input rows: {rows_in:,}")
    print(f"  Output rows: {rows_out:,}")
    print(f"  Removed - non-US issuer: {non_us_removed:,}")
    print(f"  Removed - invalid exchange: {invalid_exchange_removed:,}")
    print(f"  Tickers filled from mapping: {tickers_filled:,}")
    print(f"  Columns dropped: {len(columns_to_drop) + 1} (including is_us_issuer)")
    print(f"  Output: {ISSUER_MASTER_OUT}")
    
    stats = {
        "issuer_master": {
            "rows_in": rows_in,
            "rows_out": rows_out,
            "non_us_removed": non_us_removed,
            "invalid_exchange_removed": invalid_exchange_removed,
            "tickers_filled": tickers_filled,
        }
    }
    
    return rows_out, stats


# ============================================================
# STEP 3: Clean fundamentals_features.csv
# ============================================================

def clean_fundamentals(cik_to_ticker: dict[str, str], cik_to_name: dict[str, str]) -> tuple[int, dict]:
    """
    Clean fundamentals_features.csv:
    - Deduplicate (keep first occurrence of each cik+fiscal_year+fiscal_period+filing_date)
    - Drop 100% null columns
    - Fill missing ticker and entity_name using cik_to_ticker and cik_to_name
    - Cap outlier values
    """
    print("\n" + "=" * 60)
    print("STEP 3: Cleaning fundamentals_features.csv")
    print("=" * 60)
    
    if not FUNDAMENTALS_IN.exists():
        raise FileNotFoundError(f"Input file not found: {FUNDAMENTALS_IN}")
    
    rows_in = 0
    rows_out = 0
    duplicates_removed = 0
    tickers_filled = 0
    names_filled = 0
    values_capped = 0
    
    # Track seen periods for deduplication
    seen_periods: Set[tuple] = set()
    
    with FUNDAMENTALS_IN.open("r", newline="", encoding="utf-8") as inf:
        reader = csv.DictReader(inf)
        original_fieldnames = list(reader.fieldnames or [])
        
        # Determine output fieldnames (exclude drop columns)
        output_fieldnames = [f for f in original_fieldnames if f not in FUNDAMENTALS_DROP_COLUMNS]
        
        tmp_out = FUNDAMENTALS_OUT.with_suffix(".csv.part")
        
        with tmp_out.open("w", newline="", encoding="utf-8") as outf:
            writer = csv.DictWriter(outf, fieldnames=output_fieldnames)
            writer.writeheader()
            
            for row in reader:
                rows_in += 1
                
                # Deduplication key
                cik = row.get("cik", "").strip()
                fiscal_year = row.get("fiscal_year", "").strip()
                fiscal_period = row.get("fiscal_period", "").strip()
                filing_date = row.get("filing_date", "").strip()
                
                dedup_key = (cik, fiscal_year, fiscal_period, filing_date)
                if dedup_key in seen_periods:
                    duplicates_removed += 1
                    continue
                seen_periods.add(dedup_key)
                
                # Fill missing ticker
                ticker = row.get("ticker", "").strip()
                if not ticker and cik and cik in cik_to_ticker:
                    row["ticker"] = cik_to_ticker[cik]
                    tickers_filled += 1
                
                # Fill missing entity_name
                entity_name = row.get("entity_name", "").strip()
                if not entity_name and cik and cik in cik_to_name:
                    row["entity_name"] = cik_to_name[cik]
                    names_filled += 1
                
                # Cap outlier values
                for col, cap_range in RATIO_CAPS.items():
                    if col in row:
                        original_val = row[col]
                        capped_val = cap_value(original_val, cap_range)
                        if capped_val != "" and original_val != "" and capped_val != original_val:
                            values_capped += 1
                        row[col] = capped_val
                
                # Build output row (exclude drop columns)
                out_row = {k: v for k, v in row.items() if k not in FUNDAMENTALS_DROP_COLUMNS}
                writer.writerow(out_row)
                rows_out += 1
                
                # Progress indicator
                if rows_in % 1000 == 0:
                    print(f"  Processed {rows_in:,} rows...", flush=True)
        
        tmp_out.replace(FUNDAMENTALS_OUT)
    
    print(f"  Input rows: {rows_in:,}")
    print(f"  Output rows: {rows_out:,}")
    print(f"  Duplicates removed: {duplicates_removed:,}")
    print(f"  Tickers filled: {tickers_filled:,}")
    print(f"  Entity names filled: {names_filled:,}")
    print(f"  Values capped: {values_capped:,}")
    print(f"  Columns dropped: {len(FUNDAMENTALS_DROP_COLUMNS)} (100% null columns)")
    print(f"  Output: {FUNDAMENTALS_OUT}")
    
    stats = {
        "fundamentals_features": {
            "rows_in": rows_in,
            "rows_out": rows_out,
            "duplicates_removed": duplicates_removed,
            "tickers_filled": tickers_filled,
            "entity_names_filled": names_filled,
            "values_capped": values_capped,
        }
    }
    
    return rows_out, stats


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean and standardize SEC processed data files."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cleaned outputs.",
    )
    args = parser.parse_args()
    
    # Check if outputs already exist
    if not args.overwrite:
        existing = []
        if CIK_TICKER_MAP_OUT.exists():
            existing.append(str(CIK_TICKER_MAP_OUT))
        if ISSUER_MASTER_OUT.exists():
            existing.append(str(ISSUER_MASTER_OUT))
        if FUNDAMENTALS_OUT.exists():
            existing.append(str(FUNDAMENTALS_OUT))
        
        if existing:
            print("Some output files already exist:")
            for f in existing:
                print(f"  {f}")
            print("\nUse --overwrite to regenerate.")
            return
    
    overall_start = now_ts()
    
    print("=" * 60)
    print("SEC DATA CLEANING PIPELINE")
    print("=" * 60)
    
    all_stats = {}
    
    # Step 1: Clean cik_ticker_map
    _, cik_to_ticker, cik_to_name, stats1 = clean_cik_ticker_map()
    all_stats.update(stats1)
    
    # Step 2: Clean issuer_master
    _, stats2 = clean_issuer_master(cik_to_ticker)
    all_stats.update(stats2)
    
    # Step 3: Clean fundamentals
    _, stats3 = clean_fundamentals(cik_to_ticker, cik_to_name)
    all_stats.update(stats3)
    
    # Write summary
    summary = {
        "input_files": {
            "cik_ticker_map": str(CIK_TICKER_MAP_IN),
            "issuer_master": str(ISSUER_MASTER_IN),
            "fundamentals_features": str(FUNDAMENTALS_IN),
        },
        "output_files": {
            "cik_ticker_map": str(CIK_TICKER_MAP_OUT),
            "issuer_master": str(ISSUER_MASTER_OUT),
            "fundamentals_features": str(FUNDAMENTALS_OUT),
        },
        "valid_exchanges": list(VALID_EXCHANGES),
        "ratio_caps_applied": {k: list(v) for k, v in RATIO_CAPS.items()},
        "columns_dropped_from_fundamentals": FUNDAMENTALS_DROP_COLUMNS,
        "statistics": all_stats,
        "timing": {
            "total_seconds": now_ts() - overall_start,
        },
    }
    
    with SUMMARY_OUT.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    total_elapsed = now_ts() - overall_start
    
    print("\n" + "=" * 60)
    print("CLEANING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {OUT_DIR}")
    print(f"Summary: {SUMMARY_OUT}")
    print(f"Total elapsed: {fmt_elapsed(total_elapsed)}")
    
    # Final summary
    print("\nFinal cleaned datasets:")
    print(f"  cik_ticker_map: {all_stats['cik_ticker_map']['rows_out']:,} rows")
    print(f"  issuer_master: {all_stats['issuer_master']['rows_out']:,} rows")
    print(f"  fundamentals_features: {all_stats['fundamentals_features']['rows_out']:,} rows")


if __name__ == "__main__":
    main()

# Run cleaning pipeline
# python data/sec_data_cleaning_pipeline.py
# Use --overwrite if needed later
# python data/sec_data_cleaning_pipeline.py --overwrite


