#!/usr/bin/env python3
"""
SEC Issuer Master Builder

Builds a clean issuer/company master table from processed SEC submissions and companyfacts
inventories. Filters for U.S. issuers and produces a canonical CIK-to-ticker mapping
for downstream fundamental and text processing.

Inputs:
- submissions_inventory.csv (from sec_submissions_pipeline.py)
- companyfacts_inventory.csv (from sec_companyfacts_pipeline.py)
- Optional: entities_part_*.csv for richer metadata

Outputs:
- issuer_master.csv: canonical issuer table with U.S. flag
- cik_ticker_map.csv: CIK to primary ticker mapping
- issuer_master_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

# ============================================================
# PATHS
# ============================================================

dataPath = Path(os.getenv("dataPathGlobal", "data"))

SUBMISSIONS_ROOT = dataPath / "sec_edgar" / "processed" / "submissions"
COMPANYFACTS_ROOT = dataPath / "sec_edgar" / "processed" / "companyfacts"
OUT_ROOT = dataPath / "sec_edgar" / "processed" / "issuer_master"

SUBMISSIONS_INVENTORY = SUBMISSIONS_ROOT / "submissions_inventory.csv"
COMPANYFACTS_INVENTORY = COMPANYFACTS_ROOT / "companyfacts_inventory.csv"
ENTITIES_PARTS_DIR = SUBMISSIONS_ROOT / "submissions_flat"

ISSUER_MASTER_FINAL = OUT_ROOT / "issuer_master.csv"
CIK_TICKER_MAP_FINAL = OUT_ROOT / "cik_ticker_map.csv"
SUMMARY_FINAL = OUT_ROOT / "issuer_master_summary.json"

# U.S. state/territory codes for filtering
US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC", "PR", "VI", "GU", "AS", "MP",
}

# Common non-U.S. incorporation keywords to filter out
NON_US_KEYWORDS = {
    "CANADA", "CAYMAN", "BERMUDA", "BRITISH", "VIRGIN ISLANDS",
    "IRELAND", "UNITED KINGDOM", "UK", "ENGLAND", "WALES", "SCOTLAND",
    "NETHERLANDS", "LUXEMBOURG", "SWITZERLAND", "FRANCE", "GERMANY",
    "ISRAEL", "CHINA", "HONG KONG", "SINGAPORE", "JAPAN", "AUSTRALIA",
    "NEW ZEALAND", "INDIA", "BRAZIL", "MEXICO", "PANAMA", "BAHAMAS",
    "BARBADOS", "CURACAO", "JERSEY", "GUERNSEY", "ISLE OF MAN",
    "MARSHALL ISLANDS", "MAURITIUS", "SEYCHELLES", "LIBERIA",
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
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def parse_pipe_list(value: str) -> list[str]:
    """Parse pipe-separated list from submissions output."""
    if not value:
        return []
    return [v.strip() for v in value.split("|") if v.strip()]


def is_us_issuer(
    state_of_incorporation: str,
    state_desc: str,
    mailing_state: str,
    business_state: str,
) -> bool:
    """
    Determine if issuer is U.S.-based using incorporation and address signals.
    Returns True if likely U.S. issuer, False otherwise.
    """
    # Check incorporation state
    if state_of_incorporation and state_of_incorporation.upper() in US_STATES:
        return True

    # Check incorporation description for non-US keywords
    if state_desc:
        state_desc_upper = state_desc.upper()
        if any(kw in state_desc_upper for kw in NON_US_KEYWORDS):
            return False

    # Check mailing/business address states
    if mailing_state and mailing_state.upper() in US_STATES:
        return True
    if business_state and business_state.upper() in US_STATES:
        return True

    # If we have a state description with US state, it's US
    if state_desc:
        for state in US_STATES:
            if f", {state}" in state_desc or f" {state} " in state_desc:
                return True

    return False


def load_submissions_inventory() -> dict[str, dict[str, Any]]:
    """
    Load submissions inventory and index by CIK.
    Returns: dict[cik] -> inventory row dict
    """
    if not SUBMISSIONS_INVENTORY.exists():
        print(f"Warning: {SUBMISSIONS_INVENTORY} not found. Proceeding without submissions metadata.")
        return {}

    cik_map = {}
    with SUBMISSIONS_INVENTORY.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cik = row.get("cik", "").strip()
            status = row.get("status", "")
            if cik and status == "ok":
                # Keep the first (or merge logic can go here)
                if cik not in cik_map:
                    cik_map[cik] = row
    return cik_map


def load_companyfacts_inventory() -> dict[str, dict[str, Any]]:
    """
    Load companyfacts inventory and index by CIK.
    Returns: dict[cik] -> inventory row dict
    """
    if not COMPANYFACTS_INVENTORY.exists():
        print(f"Warning: {COMPANYFACTS_INVENTORY} not found. Proceeding without companyfacts metadata.")
        return {}

    cik_map = {}
    with COMPANYFACTS_INVENTORY.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cik = row.get("cik", "").strip()
            status = row.get("status", "")
            if cik and status == "ok":
                if cik not in cik_map:
                    cik_map[cik] = row
    return cik_map


def load_entities_metadata() -> dict[str, dict[str, Any]]:
    """
    Load entities metadata from partitioned entities parts.
    Returns: dict[cik] -> entity row dict (latest or merged)
    """
    entities_parts = sorted(ENTITIES_PARTS_DIR.glob("entities_part_*.csv"))
    if not entities_parts:
        print(f"Warning: No entities parts found in {ENTITIES_PARTS_DIR}")
        return {}

    cik_map = {}
    for part_path in entities_parts:
        with part_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cik = row.get("cik", "").strip()
                if not cik:
                    continue

                # Keep the most recent by source_json timestamp or first seen
                if cik not in cik_map:
                    cik_map[cik] = row
                else:
                    # Prefer the one with more complete data
                    existing = cik_map[cik]
                    if len(row.get("entityType", "")) > len(existing.get("entityType", "")):
                        cik_map[cik] = row

    return cik_map


def extract_primary_ticker(tickers_str: str, exchanges_str: str) -> tuple[str, str]:
    """
    Extract primary ticker from pipe-separated strings.
    Returns: (primary_ticker, primary_exchange)
    Prioritizes NYSE/NASDAQ over OTC/OTHER.
    """
    tickers = parse_pipe_list(tickers_str)
    exchanges = parse_pipe_list(exchanges_str)

    if not tickers:
        return "", ""

    # Preferred exchange order (rough heuristic)
    exchange_priority = {
        "NYSE": 1, "New York Stock Exchange": 1,
        "NASDAQ": 2, "Nasdaq": 2,
        "AMEX": 3, "NYSE American": 3,
        "OTC": 10, "OTCBB": 10, "Other OTC": 10,
    }

    best_idx = 0
    best_priority = 999

    for i, (ticker, exchange) in enumerate(zip(tickers, exchanges)):
        if i >= len(tickers):
            break
        priority = exchange_priority.get(exchange, 50)
        if priority < best_priority:
            best_priority = priority
            best_idx = i

    primary_ticker = tickers[best_idx] if best_idx < len(tickers) else tickers[0]
    primary_exchange = exchanges[best_idx] if best_idx < len(exchanges) else ""

    return primary_ticker, primary_exchange


def build_issuer_master() -> tuple[list[dict], dict]:
    """
    Build consolidated issuer master table.
    Returns: (issuer_rows, stats_dict)
    """
    print("Loading submissions inventory...", flush=True)
    submissions_map = load_submissions_inventory()

    print("Loading companyfacts inventory...", flush=True)
    companyfacts_map = load_companyfacts_inventory()

    print("Loading entities metadata...", flush=True)
    entities_map = load_entities_metadata()

    # Combine all unique CIKs
    all_ciks = set(submissions_map.keys()) | set(companyfacts_map.keys()) | set(entities_map.keys())
    print(f"Total unique CIKs across sources: {len(all_ciks):,}", flush=True)

    issuer_rows = []
    us_count = 0
    non_us_count = 0
    unknown_count = 0
    with_ticker = 0

    # Process in batches for progress
    ciks_sorted = sorted(all_ciks)
    total_ciks = len(ciks_sorted)
    progress_interval = max(1, total_ciks // 20)

    for idx, cik in enumerate(ciks_sorted):
        if idx % progress_interval == 0:
            pct = (idx / total_ciks) * 100
            print(f"  Processing CIKs: {idx:,}/{total_ciks:,} ({pct:.1f}%)", flush=True)

        sub_row = submissions_map.get(cik, {})
        fact_row = companyfacts_map.get(cik, {})
        entity_row = entities_map.get(cik, {})

        # Determine entity name
        entity_name = (
            entity_row.get("name") or
            sub_row.get("entity_name") or
            fact_row.get("entity_name") or
            ""
        )

        # U.S. issuer determination
        state_incorp = entity_row.get("stateOfIncorporation", "")
        state_desc = entity_row.get("stateOfIncorporationDescription", "")
        mailing_state = entity_row.get("mailing_stateOrCountry", "")
        business_state = entity_row.get("business_stateOrCountry", "")

        is_us = is_us_issuer(state_incorp, state_desc, mailing_state, business_state)

        if is_us:
            us_count += 1
        elif state_incorp or state_desc:
            non_us_count += 1
        else:
            unknown_count += 1

        # Extract primary ticker
        tickers_str = entity_row.get("tickers", "")
        exchanges_str = entity_row.get("exchanges", "")
        primary_ticker, primary_exchange = extract_primary_ticker(tickers_str, exchanges_str)

        if primary_ticker:
            with_ticker += 1

        # Build row
        row = {
            "cik": cik,
            "padded_cik": cik.zfill(10),
            "entity_name": entity_name,
            "entity_type": entity_row.get("entityType", ""),
            "is_us_issuer": "1" if is_us else "0",
            "primary_ticker": primary_ticker,
            "primary_exchange": primary_exchange,
            "all_tickers": tickers_str,
            "sic": entity_row.get("sic", ""),
            "sic_description": entity_row.get("sicDescription", ""),
            "fiscal_year_end": entity_row.get("fiscalYearEnd", ""),
            "state_of_incorporation": state_incorp,
            "state_of_incorporation_desc": state_desc,
            "business_city": entity_row.get("business_city", ""),
            "business_state": business_state,
            "mailing_city": entity_row.get("mailing_city", ""),
            "mailing_state": mailing_state,
            "has_submissions": "1" if cik in submissions_map else "0",
            "has_companyfacts": "1" if cik in companyfacts_map else "0",
            "earliest_filing_sub": sub_row.get("earliest_filing_date", ""),
            "latest_filing_sub": sub_row.get("latest_filing_date", ""),
            "earliest_filed_facts": fact_row.get("earliest_filed", ""),
            "latest_filed_facts": fact_row.get("latest_filed", ""),
            "n_fact_observations": fact_row.get("n_observations", "0"),
        }
        issuer_rows.append(row)

    stats = {
        "total_ciks": total_ciks,
        "us_issuers": us_count,
        "non_us_issuers": non_us_count,
        "unknown_jurisdiction": unknown_count,
        "issuers_with_ticker": with_ticker,
    }

    return issuer_rows, stats


def write_issuer_master(issuer_rows: list[dict]) -> None:
    """Write issuer master CSV."""
    if not issuer_rows:
        print("No issuer rows to write.")
        return

    fieldnames = list(issuer_rows[0].keys())

    tmp_path = ISSUER_MASTER_FINAL.with_suffix(".csv.part")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(issuer_rows)

    tmp_path.replace(ISSUER_MASTER_FINAL)
    print(f"Written: {ISSUER_MASTER_FINAL}", flush=True)


def write_cik_ticker_map(issuer_rows: list[dict]) -> None:
    """Write CIK to primary ticker mapping for U.S. issuers only."""
    cik_ticker_rows = []

    for row in issuer_rows:
        if row["is_us_issuer"] == "1" and row["primary_ticker"]:
            cik_ticker_rows.append({
                "cik": row["cik"],
                "padded_cik": row["padded_cik"],
                "primary_ticker": row["primary_ticker"],
                "entity_name": row["entity_name"],
                "primary_exchange": row["primary_exchange"],
            })

    if not cik_ticker_rows:
        print("No U.S. issuers with tickers to write.")
        return

    fieldnames = ["cik", "padded_cik", "primary_ticker", "entity_name", "primary_exchange"]

    tmp_path = CIK_TICKER_MAP_FINAL.with_suffix(".csv.part")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cik_ticker_rows)

    tmp_path.replace(CIK_TICKER_MAP_FINAL)
    print(f"Written: {CIK_TICKER_MAP_FINAL}", flush=True)
    print(f"  U.S. issuers with tickers: {len(cik_ticker_rows):,}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SEC issuer master table with U.S. filtering."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    args = parser.parse_args()

    ensure_dirs()

    # Check if outputs already exist (resume behavior)
    if not args.overwrite and ISSUER_MASTER_FINAL.exists() and CIK_TICKER_MAP_FINAL.exists():
        print(f"Outputs already exist. Use --overwrite to regenerate.")
        print(f"  {ISSUER_MASTER_FINAL}")
        print(f"  {CIK_TICKER_MAP_FINAL}")
        return

    overall_start = now_ts()

    print("=" * 60, flush=True)
    print("SEC ISSUER MASTER BUILDER", flush=True)
    print("=" * 60, flush=True)

    # Build issuer master
    build_start = now_ts()
    issuer_rows, stats = build_issuer_master()
    build_elapsed = now_ts() - build_start

    print(f"\nBuild completed in {fmt_elapsed(build_elapsed)}", flush=True)
    print(f"\nStatistics:", flush=True)
    print(f"  Total unique CIKs: {stats['total_ciks']:,}", flush=True)
    print(f"  U.S. issuers: {stats['us_issuers']:,}", flush=True)
    print(f"  Non-U.S. issuers: {stats['non_us_issuers']:,}", flush=True)
    print(f"  Unknown jurisdiction: {stats['unknown_jurisdiction']:,}", flush=True)
    print(f"  Issuers with ticker: {stats['issuers_with_ticker']:,}", flush=True)

    # Write outputs
    print("\nWriting outputs...", flush=True)
    write_issuer_master(issuer_rows)
    write_cik_ticker_map(issuer_rows)

    # Write summary
    summary = {
        "output_root": str(OUT_ROOT),
        "issuer_master": str(ISSUER_MASTER_FINAL),
        "cik_ticker_map": str(CIK_TICKER_MAP_FINAL),
        "statistics": stats,
        "timing": {
            "build_seconds": build_elapsed,
            "total_seconds": now_ts() - overall_start,
        },
    }

    with SUMMARY_FINAL.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    total_elapsed = now_ts() - overall_start
    print(f"\n=== ISSUER MASTER COMPLETE ===", flush=True)
    print(f"Summary JSON: {SUMMARY_FINAL}", flush=True)
    print(f"Total elapsed: {fmt_elapsed(total_elapsed)}", flush=True)


if __name__ == "__main__":
    main()

# pyhton data/sec_issuer_master.py