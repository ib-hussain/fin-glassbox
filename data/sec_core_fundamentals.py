#!/usr/bin/env python3
"""
SEC Core Fundamentals Extractor (Strict Point-in-Time)

Extracts a clean, point-in-time quarterly fundamentals table from the flattened
SEC companyfacts table. Uses strict "as-first-reported" logic to prevent lookahead bias.

Inputs:
- companyfacts_flat/facts_part_*.csv (121M+ rows)
- issuer_master/cik_ticker_map.csv (for U.S. issuer filtering)

Outputs:
- core_fundamentals_quarterly.csv: Clean quarterly fundamentals table
- fundamentals_summary.json
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

# ============================================================
# PATHS
# ============================================================

dataPath = Path(os.getenv("dataPathGlobal", "data"))

FACTS_PARTS_DIR = dataPath / "sec_edgar" / "processed" / "companyfacts" / "companyfacts_flat"
ISSUER_MASTER_DIR = dataPath / "sec_edgar" / "processed" / "issuer_master"
OUT_ROOT = dataPath / "sec_edgar" / "processed" / "fundamentals"

CIK_TICKER_MAP = ISSUER_MASTER_DIR / "cik_ticker_map.csv"
CORE_FUNDAMENTALS_FINAL = OUT_ROOT / "core_fundamentals_quarterly.csv"
SUMMARY_FINAL = OUT_ROOT / "fundamentals_summary.json"

TMP_DIR = OUT_ROOT / "_tmp"
PARTS_DIR = OUT_ROOT / "fundamentals_parts"

# ============================================================
# CONCEPT MAPPING (SEC XBRL tags to standard names)
# ============================================================

# Format: (priority_list_of_xbrl_tags, output_column_name, aggregation_mode)
# aggregation_mode: "sum", "first", or "latest_filed" (for point-in-time)
CONCEPT_MAP = [
    # Income Statement
    (["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
      "SalesRevenueNet", "SalesRevenueGoodsNet", "RevenueFromContractWithCustomer"],
     "revenue", "sum"),
    (["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfSales"],
     "cost_of_revenue", "sum"),
    (["GrossProfit"], "gross_profit", "first"),
    (["OperatingExpenses", "OperatingExpense"], "operating_expenses", "sum"),
    (["OperatingIncomeLoss", "OperatingIncome"], "operating_income", "first"),
    (["NetIncomeLoss", "NetIncome", "ProfitLoss"],
     "net_income", "first"),
    (["EarningsPerShareBasic", "EarningsPerShare"],
     "eps_basic", "first"),
    (["EarningsPerShareDiluted"], "eps_diluted", "first"),
    (["WeightedAverageNumberOfSharesOutstandingBasic",
      "WeightedAverageNumberOfDilutedSharesOutstanding"],
     "shares_basic", "first"),

    # Balance Sheet - Assets
    (["Assets", "TotalAssets"], "total_assets", "first"),
    (["AssetsCurrent", "CurrentAssets"], "current_assets", "first"),
    (["CashAndCashEquivalentsAtCarryingValue", "CashAndCashEquivalents",
      "Cash"], "cash_and_equivalents", "first"),
    (["InventoryNet", "Inventory"], "inventory", "first"),
    (["PropertyPlantAndEquipmentNet", "PropertyPlantAndEquipment"],
     "ppe_net", "first"),
    (["Goodwill"], "goodwill", "first"),
    (["IntangibleAssetsNetExcludingGoodwill", "IntangibleAssetsNet"],
     "intangible_assets", "first"),

    # Balance Sheet - Liabilities
    (["Liabilities", "TotalLiabilities"], "total_liabilities", "first"),
    (["LiabilitiesCurrent", "CurrentLiabilities"], "current_liabilities", "first"),
    (["LongTermDebt", "LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"],
     "long_term_debt", "first"),
    (["DebtCurrent", "ShortTermBorrowings"], "short_term_debt", "first"),

    # Balance Sheet - Equity
    (["StockholdersEquity", "ShareholdersEquity", "Equity"],
     "shareholders_equity", "first"),
    (["RetainedEarningsAccumulatedDeficit", "RetainedEarnings"],
     "retained_earnings", "first"),

    # Cash Flow
    (["NetCashProvidedByUsedInOperatingActivities", "OperatingCashFlow"],
     "operating_cash_flow", "first"),
    (["NetCashProvidedByUsedInInvestingActivities"],
     "investing_cash_flow", "first"),
    (["NetCashProvidedByUsedInFinancingActivities"],
     "financing_cash_flow", "first"),
    (["PaymentsToAcquirePropertyPlantAndEquipment", "CapitalExpenditure"],
     "capex", "sum"),
    (["FreeCashFlow"], "free_cash_flow", "first"),
]

# Forms considered for point-in-time quarterly extraction
QUARTERLY_FORMS = {"10-Q", "10-Q/A", "10-K", "10-K/A"}


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


def human_bytes(n: int | float) -> str:
    n = float(n)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if n < 1024.0 or unit == units[-1]:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} B"


def ensure_dirs(overwrite: bool) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if overwrite:
        import shutil
        for p in [PARTS_DIR, TMP_DIR, CORE_FUNDAMENTALS_FINAL, SUMMARY_FINAL]:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                p.unlink()

    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def load_us_ciks() -> set[str]:
    """Load U.S. CIKs from cik_ticker_map."""
    if not CIK_TICKER_MAP.exists():
        print(f"Warning: {CIK_TICKER_MAP} not found. Processing all CIKs.")
        return set()

    us_ciks = set()
    with CIK_TICKER_MAP.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cik = row.get("cik", "").strip()
            if cik:
                us_ciks.add(cik)
    return us_ciks


def build_concept_lookup() -> dict[str, tuple[str, str]]:
    """
    Build mapping from SEC concept name to (output_column, aggregation_mode).
    """
    lookup = {}
    for tags, out_col, agg_mode in CONCEPT_MAP:
        for tag in tags:
            lookup[tag.lower()] = (out_col, agg_mode)
    return lookup


def parse_fiscal_period(fp: str) -> str:
    """
    Normalize fiscal period.
    SEC uses: FY, Q1, Q2, Q3, Q4, H1, H2, etc.
    Returns: "FY", "Q1", "Q2", "Q3", "Q4", or fp as-is.
    """
    fp_upper = fp.upper().strip()
    if fp_upper in ("FY", "Q1", "Q2", "Q3", "Q4"):
        return fp_upper
    return fp_upper


def fiscal_period_sort_key(fp: str) -> int:
    """Sort key for fiscal periods (FY comes after Q4)."""
    order = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "FY": 5}
    return order.get(fp, 99)


def process_facts_parts(us_ciks: set[str]) -> tuple[Path, dict]:
    """
    Process all facts parts, filter for U.S. CIKs, and extract point-in-time
    quarterly fundamentals.

    Returns: (output_path, stats_dict)
    """
    concept_lookup = build_concept_lookup()

    facts_parts = sorted(FACTS_PARTS_DIR.glob("facts_part_*.csv"))
    if not facts_parts:
        raise FileNotFoundError(f"No facts parts found in {FACTS_PARTS_DIR}")

    print(f"Found {len(facts_parts)} facts parts to process", flush=True)

    # We'll write to a single output file (could partition if needed)
    output_path = CORE_FUNDAMENTALS_FINAL

    fieldnames = [
        "cik",
        "ticker",
        "entity_name",
        "fiscal_year",
        "fiscal_period",
        "filing_date",
        "period_end_date",
        "form_type",
        "accession",
        "revenue",
        "cost_of_revenue",
        "gross_profit",
        "operating_expenses",
        "operating_income",
        "net_income",
        "eps_basic",
        "eps_diluted",
        "shares_basic",
        "total_assets",
        "current_assets",
        "cash_and_equivalents",
        "inventory",
        "ppe_net",
        "goodwill",
        "intangible_assets",
        "total_liabilities",
        "current_liabilities",
        "long_term_debt",
        "short_term_debt",
        "shareholders_equity",
        "retained_earnings",
        "operating_cash_flow",
        "investing_cash_flow",
        "financing_cash_flow",
        "capex",
        "free_cash_flow",
    ]

    tmp_path = output_path.with_suffix(".csv.part")
    total_rows = 0
    total_observations_processed = 0

    # Track which (cik, fy, fp) we've already written
    # For point-in-time, we want earliest filing date per period
    processed_periods: dict[tuple[str, str, str], dict] = {}

    with tmp_path.open("w", newline="", encoding="utf-8") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=fieldnames)
        writer.writeheader()

        for part_idx, part_path in enumerate(facts_parts):
            part_size = part_path.stat().st_size
            print(f"\nProcessing part {part_idx+1}/{len(facts_parts)}: {part_path.name} ({human_bytes(part_size)})", flush=True)

            rows_in_part = 0
            observations_in_part = 0

            # Accumulate facts per (cik, fy, fp, concept)
            # Structure: {(cik, fy, fp): {concept: [(val, filed_date, form, end_date, accn)]}}
            period_facts: dict[tuple, dict] = defaultdict(lambda: defaultdict(list))

            with part_path.open("r", newline="", encoding="utf-8") as in_fh:
                reader = csv.DictReader(in_fh)

                for row in reader:
                    observations_in_part += 1

                    cik = row.get("cik", "").strip()
                    if not cik:
                        continue

                    # Filter for U.S. CIKs if we have the list
                    if us_ciks and cik not in us_ciks:
                        continue

                    form = (row.get("form") or "").strip().upper()
                    if form not in QUARTERLY_FORMS:
                        continue

                    fy = (row.get("fy") or "").strip()
                    fp = parse_fiscal_period(row.get("fp") or "")
                    if not fy or not fp:
                        continue

                    # Only quarterly/annual periods
                    if fp not in ("Q1", "Q2", "Q3", "Q4", "FY"):
                        continue

                    concept = (row.get("concept") or "").lower().strip()
                    if concept not in concept_lookup:
                        continue

                    val_str = (row.get("val") or "").strip()
                    if not val_str:
                        continue

                    try:
                        val = float(val_str)
                    except ValueError:
                        continue

                    filed = (row.get("filed") or "").strip()
                    end = (row.get("end") or "").strip()
                    accn = (row.get("accn") or "").strip()

                    key = (cik, fy, fp)
                    period_facts[key][concept].append((val, filed, form, end, accn))
                    rows_in_part += 1

            # Now for each period, select the earliest filing and aggregate facts
            for (cik, fy, fp), concept_dict in period_facts.items():
                # Determine the earliest filing across all concepts for this period
                earliest_filed = "9999-99-99"
                earliest_form = ""
                earliest_end = ""
                earliest_accn = ""

                for concept, obs_list in concept_dict.items():
                    for val, filed, form, end, accn in obs_list:
                        if filed < earliest_filed:
                            earliest_filed = filed
                            earliest_form = form
                            earliest_end = end
                            earliest_accn = accn

                # Now extract only facts from the earliest filing
                row_data = {
                    "cik": cik,
                    "ticker": "",  # Will join later if needed
                    "entity_name": "",
                    "fiscal_year": fy,
                    "fiscal_period": fp,
                    "filing_date": earliest_filed,
                    "period_end_date": earliest_end,
                    "form_type": earliest_form,
                    "accession": earliest_accn,
                }

                for concept, obs_list in concept_dict.items():
                    out_col, agg_mode = concept_lookup[concept]

                    # Filter to only observations from the earliest filing
                    earliest_obs = [v for v, f, _, _, _ in obs_list if f == earliest_filed]

                    if not earliest_obs:
                        continue

                    if agg_mode == "first":
                        row_data[out_col] = str(earliest_obs[0])
                    elif agg_mode == "sum":
                        row_data[out_col] = str(sum(earliest_obs))
                    else:
                        row_data[out_col] = str(earliest_obs[0])

                writer.writerow(row_data)
                total_rows += 1

            total_observations_processed += observations_in_part
            print(f"  Part rows written: {rows_in_part:,} | Cumulative output rows: {total_rows:,}", flush=True)

    tmp_path.replace(output_path)

    stats = {
        "total_output_rows": total_rows,
        "total_observations_processed": total_observations_processed,
        "us_ciks_filtered": len(us_ciks) if us_ciks else 0,
    }

    return output_path, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract point-in-time quarterly fundamentals from SEC companyfacts."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    args = parser.parse_args()

    ensure_dirs(args.overwrite)

    if not args.overwrite and CORE_FUNDAMENTALS_FINAL.exists():
        print(f"Output already exists: {CORE_FUNDAMENTALS_FINAL}")
        print("Use --overwrite to regenerate.")
        return

    overall_start = now_ts()

    print("=" * 60, flush=True)
    print("SEC CORE FUNDAMENTALS EXTRACTOR (Point-in-Time)", flush=True)
    print("=" * 60, flush=True)

    # Load U.S. CIKs for filtering
    print("\nLoading U.S. CIK list...", flush=True)
    us_ciks = load_us_ciks()
    if us_ciks:
        print(f"Loaded {len(us_ciks):,} U.S. CIKs for filtering", flush=True)
    else:
        print("No U.S. CIK filter loaded. Processing all CIKs.", flush=True)

    # Process facts parts
    print("\nProcessing companyfacts parts...", flush=True)
    process_start = now_ts()
    output_path, stats = process_facts_parts(us_ciks)
    process_elapsed = now_ts() - process_start

    print(f"\nProcessing completed in {fmt_elapsed(process_elapsed)}", flush=True)
    print(f"Total output rows: {stats['total_output_rows']:,}", flush=True)
    print(f"Observations processed: {stats['total_observations_processed']:,}", flush=True)

    # Write summary
    summary = {
        "output_file": str(output_path),
        "point_in_time_method": "strict_earliest_filing",
        "quarterly_forms": list(QUARTERLY_FORMS),
        "concepts_extracted": len(CONCEPT_MAP),
        "statistics": stats,
        "timing": {
            "processing_seconds": process_elapsed,
            "total_seconds": now_ts() - overall_start,
        },
    }

    with SUMMARY_FINAL.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    total_elapsed = now_ts() - overall_start

    print("\n=== CORE FUNDAMENTALS COMPLETE ===", flush=True)
    print(f"Output: {output_path}", flush=True)
    print(f"Summary: {SUMMARY_FINAL}", flush=True)
    print(f"Total elapsed: {fmt_elapsed(total_elapsed)}", flush=True)


if __name__ == "__main__":
    main()

# python data/sec_core_fundamentals.py