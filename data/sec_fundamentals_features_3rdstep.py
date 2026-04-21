#!/usr/bin/env python3
"""
SEC Fundamentals Feature Engineering

Derives normalized ratios and growth metrics from point-in-time quarterly
fundamentals. All features are percentages or ratios, making them comparable
across companies of different sizes.

Input:  core_fundamentals_quarterly.csv
Output: fundamentals_features.csv (enriched with derived metrics)

Valuation metrics (P/E, P/B, P/S) are intentionally omitted until price data
is available from market data pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

# ============================================================
# PATHS
# ============================================================

dataPath = Path(os.getenv("dataPathGlobal", "data"))

INPUT_FILE = dataPath / "sec_edgar" / "processed" / "fundamentals" / "core_fundamentals_quarterly.csv"
OUTPUT_FILE = dataPath / "sec_edgar" / "processed" / "fundamentals" / "fundamentals_features.csv"
SUMMARY_FILE = dataPath / "sec_edgar" / "processed" / "fundamentals" / "fundamentals_features_summary.json"


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


def safe_divide(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Divide safely, return None if division by zero or missing values."""
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    return numerator / denominator


def safe_growth(current: Optional[float], prior: Optional[float]) -> Optional[float]:
    """Calculate growth rate (current - prior) / abs(prior). Returns None if invalid."""
    if current is None or prior is None:
        return None
    if prior == 0:
        return None
    return (current - prior) / abs(prior)


def calculate_derived_features(row: dict) -> dict:
    """
    Calculate all derived ratios and percentages for a single period.
    These are normalized metrics comparable across companies.
    """
    # Extract raw values
    revenue = safe_float(row.get("revenue"))
    cost_of_revenue = safe_float(row.get("cost_of_revenue"))
    gross_profit = safe_float(row.get("gross_profit"))
    operating_expenses = safe_float(row.get("operating_expenses"))
    operating_income = safe_float(row.get("operating_income"))
    net_income = safe_float(row.get("net_income"))
    
    total_assets = safe_float(row.get("total_assets"))
    current_assets = safe_float(row.get("current_assets"))
    cash = safe_float(row.get("cash_and_equivalents"))
    inventory = safe_float(row.get("inventory"))
    ppe_net = safe_float(row.get("ppe_net"))
    
    total_liabilities = safe_float(row.get("total_liabilities"))
    current_liabilities = safe_float(row.get("current_liabilities"))
    long_term_debt = safe_float(row.get("long_term_debt"))
    short_term_debt = safe_float(row.get("short_term_debt"))
    
    shareholders_equity = safe_float(row.get("shareholders_equity"))
    
    operating_cash_flow = safe_float(row.get("operating_cash_flow"))
    capex = safe_float(row.get("capex"))
    free_cash_flow = safe_float(row.get("free_cash_flow"))
    
    shares_basic = safe_float(row.get("shares_basic"))
    eps_basic = safe_float(row.get("eps_basic"))

    features = {}

    # ============================================================
    # PROFITABILITY RATIOS (all percentages of revenue)
    # ============================================================
    features["gross_margin"] = safe_divide(gross_profit, revenue)
    features["operating_margin"] = safe_divide(operating_income, revenue)
    features["net_margin"] = safe_divide(net_income, revenue)
    
    # Operating efficiency
    features["opex_to_revenue"] = safe_divide(operating_expenses, revenue)
    features["cogs_to_revenue"] = safe_divide(cost_of_revenue, revenue)

    # ============================================================
    # RETURN RATIOS (efficiency of capital use)
    # ============================================================
    features["roa"] = safe_divide(net_income, total_assets)  # Return on Assets
    features["roe"] = safe_divide(net_income, shareholders_equity)  # Return on Equity
    
    # Alternative ROE using average equity would require prior period, skip for now

    # ============================================================
    # LEVERAGE & LIQUIDITY RATIOS
    # ============================================================
    features["debt_to_equity"] = safe_divide(long_term_debt, shareholders_equity)
    features["debt_to_assets"] = safe_divide(total_liabilities, total_assets)
    features["current_ratio"] = safe_divide(current_assets, current_liabilities)
    features["quick_ratio"] = safe_divide(
        (current_assets - inventory) if current_assets and inventory else current_assets,
        current_liabilities
    )
    features["cash_ratio"] = safe_divide(cash, current_liabilities)

    # ============================================================
    # EFFICIENCY RATIOS
    # ============================================================
    features["asset_turnover"] = safe_divide(revenue, total_assets)
    features["ppe_turnover"] = safe_divide(revenue, ppe_net)  # Fixed asset turnover

    # ============================================================
    # CASH FLOW METRICS
    # ============================================================
    features["ocf_to_revenue"] = safe_divide(operating_cash_flow, revenue)
    features["fcf_to_revenue"] = safe_divide(free_cash_flow, revenue)
    features["capex_to_revenue"] = safe_divide(capex, revenue)
    features["fcf_to_net_income"] = safe_divide(free_cash_flow, net_income)
    features["ocf_to_net_income"] = safe_divide(operating_cash_flow, net_income)

    # ============================================================
    # PER-SHARE METRICS (already normalized by shares)
    # ============================================================
    features["revenue_per_share"] = safe_divide(revenue, shares_basic)
    features["book_value_per_share"] = safe_divide(shareholders_equity, shares_basic)
    features["ocf_per_share"] = safe_divide(operating_cash_flow, shares_basic)
    features["fcf_per_share"] = safe_divide(free_cash_flow, shares_basic)

    # ============================================================
    # EARNINGS QUALITY
    # ============================================================
    features["accruals_to_assets"] = safe_divide(
        (net_income - operating_cash_flow) if net_income and operating_cash_flow else None,
        total_assets
    )

    return features


def process_fundamentals() -> tuple[list[dict], dict]:
    """
    Read core fundamentals, calculate derived features and growth rates.
    Returns enriched rows and statistics.
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    print(f"Reading: {INPUT_FILE}", flush=True)
    
    # Load all rows and group by CIK
    rows_by_cik: dict[str, list[dict]] = {}
    
    with INPUT_FILE.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        
        for row in reader:
            cik = row.get("cik", "").strip()
            if not cik:
                continue
            
            # Convert numeric strings to float for calculations
            if cik not in rows_by_cik:
                rows_by_cik[cik] = []
            rows_by_cik[cik].append(row)
    
    print(f"Loaded {sum(len(v) for v in rows_by_cik.values()):,} rows across {len(rows_by_cik):,} CIKs", flush=True)
    
    # Sort each CIK's rows chronologically
    for cik in rows_by_cik:
        rows_by_cik[cik].sort(key=lambda r: (
            r.get("fiscal_year", "0"),
            {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "FY": 5}.get(r.get("fiscal_period", ""), 99)
        ))
    
    # Define new feature columns
    derived_feature_names = [
        "gross_margin", "operating_margin", "net_margin",
        "opex_to_revenue", "cogs_to_revenue",
        "roa", "roe",
        "debt_to_equity", "debt_to_assets",
        "current_ratio", "quick_ratio", "cash_ratio",
        "asset_turnover", "ppe_turnover",
        "ocf_to_revenue", "fcf_to_revenue", "capex_to_revenue",
        "fcf_to_net_income", "ocf_to_net_income",
        "revenue_per_share", "book_value_per_share",
        "ocf_per_share", "fcf_per_share",
        "accruals_to_assets",
    ]
    
    growth_feature_names = []
    base_metrics = ["revenue", "net_income", "operating_income", "eps_basic", "total_assets"]
    for metric in base_metrics:
        growth_feature_names.append(f"{metric}_growth_yoy")
        growth_feature_names.append(f"{metric}_growth_qoq")
    
    all_new_fields = derived_feature_names + growth_feature_names
    output_fieldnames = fieldnames + all_new_fields
    
    enriched_rows = []
    stats = {
        "total_rows": 0,
        "rows_with_growth": 0,
        "ciks_processed": len(rows_by_cik),
        "features_added": len(all_new_fields),
    }
    
    for cik, rows in rows_by_cik.items():
        for i, row in enumerate(rows):
            # Calculate derived features for current period
            features = calculate_derived_features(row)
            
            # Calculate growth rates if prior period exists
            if i > 0:
                prior_row = rows[i - 1]
                
                # YoY growth (same fiscal period, previous year)
                for metric in base_metrics:
                    current_val = safe_float(row.get(metric))
                    
                    # Find same period in prior year
                    yoy_val = None
                    for prior in rows[:i]:
                        if (prior.get("fiscal_period") == row.get("fiscal_period") and
                            int(prior.get("fiscal_year", 0)) == int(row.get("fiscal_year", 0)) - 1):
                            yoy_val = safe_float(prior.get(metric))
                            break
                    
                    features[f"{metric}_growth_yoy"] = safe_growth(current_val, yoy_val)
                    
                # QoQ growth (sequential periods)
                for metric in base_metrics:
                    current_val = safe_float(row.get(metric))
                    prior_val = safe_float(prior_row.get(metric))
                    features[f"{metric}_growth_qoq"] = safe_growth(current_val, prior_val)
                    
                stats["rows_with_growth"] += 1
            else:
                # First period for this CIK, no growth available
                for metric in base_metrics:
                    features[f"{metric}_growth_yoy"] = None
                    features[f"{metric}_growth_qoq"] = None
            
            # Merge features into row
            enriched_row = {**row}
            for key, value in features.items():
                enriched_row[key] = "" if value is None else str(value)
            
            enriched_rows.append(enriched_row)
            stats["total_rows"] += 1
    
    return enriched_rows, stats, output_fieldnames


def write_output(rows: list[dict], fieldnames: list[str]) -> None:
    """Write enriched data to CSV."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    tmp_path = OUTPUT_FILE.with_suffix(".csv.part")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    tmp_path.replace(OUTPUT_FILE)
    print(f"\nWritten: {OUTPUT_FILE}", flush=True)
    print(f"  Rows: {len(rows):,}", flush=True)
    print(f"  Columns: {len(fieldnames)} ({len(fieldnames) - 36} new features)", flush=True)


def write_summary(stats: dict, timing: dict) -> None:
    """Write summary JSON."""
    summary = {
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
        "statistics": stats,
        "feature_groups": {
            "profitability": ["gross_margin", "operating_margin", "net_margin", "opex_to_revenue", "cogs_to_revenue"],
            "returns": ["roa", "roe"],
            "leverage_liquidity": ["debt_to_equity", "debt_to_assets", "current_ratio", "quick_ratio", "cash_ratio"],
            "efficiency": ["asset_turnover", "ppe_turnover"],
            "cash_flow": ["ocf_to_revenue", "fcf_to_revenue", "capex_to_revenue", "fcf_to_net_income", "ocf_to_net_income"],
            "per_share": ["revenue_per_share", "book_value_per_share", "ocf_per_share", "fcf_per_share"],
            "earnings_quality": ["accruals_to_assets"],
            "growth": ["revenue_growth_yoy", "revenue_growth_qoq", "net_income_growth_yoy", "net_income_growth_qoq",
                      "operating_income_growth_yoy", "operating_income_growth_qoq", "eps_basic_growth_yoy",
                      "eps_basic_growth_qoq", "total_assets_growth_yoy", "total_assets_growth_qoq"],
        },
        "notes": [
            "All derived features are normalized ratios or percentages comparable across companies",
            "Growth rates calculated as (current - prior) / abs(prior)",
            "Valuation metrics (P/E, P/B, P/S) omitted - require price data",
            "YoY growth uses same fiscal period in prior year",
            "QoQ growth uses immediate prior period",
        ],
        "timing": timing,
    }
    
    with SUMMARY_FILE.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary: {SUMMARY_FILE}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive normalized features from SEC core fundamentals."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file.",
    )
    args = parser.parse_args()
    
    if not args.overwrite and OUTPUT_FILE.exists():
        print(f"Output already exists: {OUTPUT_FILE}")
        print("Use --overwrite to regenerate.")
        return
    
    overall_start = now_ts()
    
    print("=" * 60, flush=True)
    print("SEC FUNDAMENTALS FEATURE ENGINEERING", flush=True)
    print("=" * 60, flush=True)
    print("\nDeriving normalized ratios and growth metrics...", flush=True)
    
    process_start = now_ts()
    enriched_rows, stats, fieldnames = process_fundamentals()
    process_elapsed = now_ts() - process_start
    
    write_output(enriched_rows, fieldnames)
    
    timing = {
        "processing_seconds": process_elapsed,
        "total_seconds": now_ts() - overall_start,
    }
    
    write_summary(stats, timing)
    
    print("\n=== FEATURE ENGINEERING COMPLETE ===", flush=True)
    print(f"Rows processed: {stats['total_rows']:,}", flush=True)
    print(f"CIKs: {stats['ciks_processed']:,}", flush=True)
    print(f"Features added: {stats['features_added']}", flush=True)
    print(f"Rows with growth metrics: {stats['rows_with_growth']:,}", flush=True)
    print(f"Elapsed: {fmt_elapsed(timing['total_seconds'])}", flush=True)


if __name__ == "__main__":
    main()

# Run feature engineering
# python data/sec_fundamentals_features_3rdstep.py
# Use --overwrite if needed later
# python data/sec_fundamentals_features_3rdstep.py --overwrite