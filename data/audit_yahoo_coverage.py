#!/usr/bin/env python3
"""
Audit Yahoo Finance/yfinance daily history coverage for a ticker universe.

Input expected by default:
    primary_tickers.csv with columns:
        cik, entity_name, primary_ticker

Outputs:
    data/yahoo_coverage_audit.csv
    data/yahoo_usable_2000_2024.csv
    data/yahoo_partial_or_bad.csv
    data/yahoo_coverage_summary.txt

Run:
    python data/audit_yahoo_coverage.py --input data/primary_tickers.csv --workers 8
"""

import argparse
import concurrent.futures as cf
import math
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
from tqdm import tqdm

try:
    import yfinance as yf
except ImportError:
    print("Missing dependency: yfinance. Install with: pip install yfinance pandas tqdm")
    sys.exit(1)


def normalize_yahoo_symbol(ticker: str) -> str:
    """
    SEC tickers sometimes use dots for class shares, while Yahoo usually uses dashes.
    Your uploaded file already uses dashes for special symbols like BRK-B, so this is safe.
    """
    t = str(ticker).strip().upper()
    t = t.replace(".", "-")
    return t


def clean_history_index(hist: pd.DataFrame) -> pd.DataFrame:
    if hist is None or hist.empty:
        return pd.DataFrame()

    hist = hist.copy()

    # yfinance index may be tz-aware. Remove timezone for clean date comparison.
    try:
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
    except TypeError:
        hist.index = pd.to_datetime(hist.index)

    hist = hist[~hist.index.duplicated(keep="first")].sort_index()

    # Use rows where at least one market-data column is present.
    market_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in hist.columns]
    if market_cols:
        hist = hist.dropna(how="all", subset=market_cols)

    # Prefer rows with a usable close/adjusted close.
    close_col = "Adj Close" if "Adj Close" in hist.columns else "Close" if "Close" in hist.columns else None
    if close_col:
        hist = hist[hist[close_col].notna()]

    return hist


def classify_coverage(first_date, last_date, total_rows, target_rows, target_start, target_end, min_target_rows, stale_days):
    if first_date is None or last_date is None or total_rows == 0:
        return "NO_DATA"

    now = pd.Timestamp.utcnow().tz_localize(None)
    stale_cutoff = now - pd.Timedelta(days=stale_days)

    covers_start = first_date <= target_start
    covers_end = last_date >= target_end
    enough_rows = target_rows >= min_target_rows
    recent = last_date >= stale_cutoff

    if covers_start and covers_end and enough_rows:
        return "USABLE_2000_2024"

    if covers_start and recent and total_rows >= min_target_rows:
        return "USABLE_25Y_TO_RECENT"

    reasons = []
    if not covers_start:
        reasons.append("STARTS_TOO_LATE")
    if not covers_end:
        reasons.append("ENDS_BEFORE_TARGET_END")
    if not enough_rows:
        reasons.append("TOO_FEW_TARGET_ROWS")
    if not recent:
        reasons.append("STALE_OR_DELISTED")
    return "PARTIAL_" + "_".join(reasons)


def audit_one(row, target_start, target_end, min_target_rows, stale_days, sleep_seconds):
    cik = row.get("cik", "")
    entity_name = row.get("entity_name", "")
    original_ticker = row.get("primary_ticker", "")
    yahoo_ticker = normalize_yahoo_symbol(original_ticker)

    result = {
        "cik": cik,
        "entity_name": entity_name,
        "primary_ticker": original_ticker,
        "yahoo_ticker": yahoo_ticker,
        "first_date": "",
        "last_date": "",
        "trading_rows_total": 0,
        "calendar_span_days": 0,
        "span_years": 0.0,
        "target_start": target_start.date().isoformat(),
        "target_end": target_end.date().isoformat(),
        "target_rows_2000_2024": 0,
        "has_data": False,
        "coverage_status": "NO_DATA",
        "error": "",
    }

    try:
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

        hist = yf.Ticker(yahoo_ticker).history(
            period="max",
            interval="1d",
            auto_adjust=False,
            actions=False,
            raise_errors=False,
        )

        hist = clean_history_index(hist)

        if hist.empty:
            return result

        first_date = hist.index.min()
        last_date = hist.index.max()
        total_rows = int(len(hist))

        in_target = hist[(hist.index >= target_start) & (hist.index <= target_end)]
        target_rows = int(len(in_target))

        calendar_span_days = int((last_date.date() - first_date.date()).days + 1)
        span_years = calendar_span_days / 365.25

        result.update({
            "first_date": first_date.date().isoformat(),
            "last_date": last_date.date().isoformat(),
            "trading_rows_total": total_rows,
            "calendar_span_days": calendar_span_days,
            "span_years": round(span_years, 2),
            "target_rows_2000_2024": target_rows,
            "has_data": True,
            "coverage_status": classify_coverage(
                first_date=first_date,
                last_date=last_date,
                total_rows=total_rows,
                target_rows=target_rows,
                target_start=target_start,
                target_end=target_end,
                min_target_rows=min_target_rows,
                stale_days=stale_days,
            ),
        })

    except Exception as e:
        result["coverage_status"] = "ERROR"
        result["error"] = repr(e)

    return result


def write_outputs(results_df: pd.DataFrame, out_csv: str):
    results_df = results_df.sort_values(
        ["coverage_status", "target_rows_2000_2024", "trading_rows_total"],
        ascending=[True, False, False],
    )

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(out_path, index=False)

    usable = results_df[results_df["coverage_status"].isin(["USABLE_2000_2024", "USABLE_25Y_TO_RECENT"])]
    bad = results_df[~results_df.index.isin(usable.index)]

    usable_path = out_path.with_name("data/yahoo_usable_2000_2024.csv")
    bad_path = out_path.with_name("data/yahoo_partial_or_bad.csv")
    summary_path = out_path.with_name("data/yahoo_coverage_summary.txt")

    usable.to_csv(usable_path, index=False)
    bad.to_csv(bad_path, index=False)

    counts = results_df["coverage_status"].value_counts(dropna=False).sort_index()
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Yahoo Finance Coverage Audit Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Created UTC: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"Total tickers audited: {len(results_df)}\n\n")
        f.write("Coverage status counts:\n")
        for status, count in counts.items():
            f.write(f"  {status}: {count}\n")
        f.write("\nTop 20 longest histories:\n")
        longest = results_df.sort_values("trading_rows_total", ascending=False).head(20)
        for _, r in longest.iterrows():
            f.write(
                f"  {r['primary_ticker']}: {r['first_date']} -> {r['last_date']} "
                f"({r['trading_rows_total']} rows, {r['span_years']} years)\n"
            )

    return out_path, usable_path, bad_path, summary_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/primary_tickers.csv", help="Input CSV path.")
    parser.add_argument("--ticker-col", default="primary_ticker", help="Ticker column name.")
    parser.add_argument("--out", default="data/yahoo_coverage_audit.csv", help="Main output CSV.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers. Use 4-12 normally.")
    parser.add_argument("--target-start", default="2000-01-03", help="Target start date for 2000-2024 coverage.")
    parser.add_argument("--target-end", default="2024-12-31", help="Target end date for 2000-2024 coverage.")
    parser.add_argument("--min-target-rows", type=int, default=6288, help="Minimum daily rows inside target range.")
    parser.add_argument("--stale-days", type=int, default=120, help="Last date older than this is considered stale/delisted.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional sleep per ticker to reduce rate pressure.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output CSV.")
    parser.add_argument("--limit", type=int, default=0, help="Test on first N tickers only. 0 = all.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    src = pd.read_csv(input_path)
    if args.ticker_col not in src.columns:
        print(f"Ticker column '{args.ticker_col}' not found. Columns: {list(src.columns)}")
        sys.exit(1)

    src = src.dropna(subset=[args.ticker_col]).copy()
    src[args.ticker_col] = src[args.ticker_col].astype(str).str.strip()
    src = src[src[args.ticker_col] != ""].drop_duplicates(subset=[args.ticker_col])

    if args.limit and args.limit > 0:
        src = src.head(args.limit)

    target_start = pd.Timestamp(args.target_start)
    target_end = pd.Timestamp(args.target_end)

    already = pd.DataFrame()
    done = set()
    if args.resume and Path(args.out).exists():
        already = pd.read_csv(args.out)
        if "primary_ticker" in already.columns:
            done = set(already["primary_ticker"].astype(str))
            src = src[~src[args.ticker_col].astype(str).isin(done)]
            print(f"Resume mode: {len(done)} already done, {len(src)} remaining.")

    records = src.to_dict("records")
    results = [] if already.empty else already.to_dict("records")

    print(f"Auditing {len(records)} tickers from {input_path}...")
    print(f"Target range: {args.target_start} to {args.target_end}")
    print(f"Minimum target rows: {args.min_target_rows}")
    print(f"Workers: {args.workers}")

    with cf.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                audit_one,
                row,
                target_start,
                target_end,
                args.min_target_rows,
                args.stale_days,
                args.sleep,
            )
            for row in records
        ]

        for i, fut in enumerate(tqdm(cf.as_completed(futures), total=len(futures)), start=1):
            results.append(fut.result())

            # checkpoint every 100 completed tickers
            if i % 100 == 0:
                tmp = pd.DataFrame(results)
                tmp.to_csv(args.out, index=False)

    final_df = pd.DataFrame(results)
    paths = write_outputs(final_df, args.out)

    print("\nDone.")
    for p in paths:
        print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
