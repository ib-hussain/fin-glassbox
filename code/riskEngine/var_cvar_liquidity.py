#!/usr/bin/env python3
"""
code/riskEngine/var_cvar_liquidity.py

Risk Engine modules: Historical VaR, CVaR (Expected Shortfall), and Liquidity Risk.

All three are non-parametric / rule-based — no model training required.
Consumes the market feature files produced by the data pipeline.

VaR:  Value at Risk — threshold loss at given confidence (95%, 99%)
CVaR: Conditional VaR — average loss beyond VaR threshold
Liquidity: Trading feasibility scores from volume metrics

Outputs:
  outputs/results/risk/var_cvar.csv       — daily VaR/CVaR per ticker
  outputs/results/risk/liquidity.csv       — daily liquidity scores per ticker

Usage:
  python code/riskEngine/var_cvar_liquidity.py --workers 4
  python code/riskEngine/var_cvar_liquidity.py --workers 4 --chunk 1  # specific chunk
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Paths ──────────────────────────────────────────────────
RETURNS_LONG = Path("data/yFinance/processed/returns_long.csv")
LIQUIDITY_IN = Path("data/yFinance/processed/liquidity_features.csv")
DATES_FILE  = Path("data/market_dates_ONLY_NYSE.csv")
OUT_DIR     = Path("outputs/results/risk")

# ── VaR/CVaR parameters ───────────────────────────────────
VAR_WINDOW     = 504   # 2 years of trading days
VAR_LEVELS     = [0.95, 0.99]

# ── Liquidity thresholds ──────────────────────────────────
MIN_VOLUME_PERCENTILE = 20    # below this = illiquid
MAX_SPREAD_PCT        = 0.5   # above this = high slippage (proxy)
MARKET_CAP_TIERS      = [10e9, 2e9]  # large > $10B, mid, small < $2B
DAYS_TO_LIQUIDATE_MAX = 5     # >5 days = liquidity warning


def parse_args():
    p = argparse.ArgumentParser(description="Compute Historical VaR, CVaR, and Liquidity Risk")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--chunk", type=int, default=0, help="0=all, 1/2/3=specific chunk")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
#  HISTORICAL VaR  &  CVaR
# ══════════════════════════════════════════════════════════════

def compute_var_cvar(returns: np.ndarray, levels: list = VAR_LEVELS) -> dict:
    """
    Compute historical VaR and CVaR from a 1D array of returns.
    
    VaR at level α = the α-percentile of the returns distribution.
    CVaR at level α = mean of returns worse than VaR_α.
    
    Returns are assumed to be simple returns where negative = loss.
    VaR is reported as a negative number (e.g., -0.025 = 2.5% loss threshold).
    CVaR is also negative and |CVaR| >= |VaR|.
    
    Args:
        returns: 1D numpy array of daily returns (simple returns, not log).
        levels: List of confidence levels (e.g., [0.95, 0.99]).
    
    Returns:
        Dict with keys like 'var_95', 'cvar_95', 'var_99', 'cvar_99', 
        'tail_risk_ratio_95', 'tail_risk_ratio_99'.
    """
    clean = returns[~np.isnan(returns)]
    if len(clean) < 100:
        return {f"var_{int(l*100)}": np.nan for l in levels} | \
               {f"cvar_{int(l*100)}": np.nan for l in levels} | \
               {f"tail_ratio_{int(l*100)}": np.nan for l in levels}
    
    result = {}
    for level in levels:
        alpha = 1 - level  # e.g., 5% for 95% VaR
        var = np.percentile(clean, alpha * 100)
        tail = clean[clean <= var]
        cvar = tail.mean() if len(tail) > 0 else var
        
        label = int(level * 100)
        result[f"var_{label}"] = float(var)
        result[f"cvar_{label}"] = float(cvar)
        # Tail risk ratio: how much worse is the tail than the threshold?
        # > 1 means tail is significantly worse than VaR alone suggests
        result[f"tail_ratio_{label}"] = float(abs(cvar / var)) if var != 0 else np.nan
    
    return result


def process_var_cvar_ticker(ticker: str, returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling VaR/CVaR for one ticker across all dates.
    Uses an expanding-then-rolling window: first 2 years expand, then 2-year rolling.
    """
    sub = returns_df[returns_df["ticker"] == ticker].sort_values("date").copy()
    sub["simple_return"] = pd.to_numeric(sub["simple_return"], errors="coerce")
    
    results = []
    returns_series = sub["simple_return"].values
    dates = sub["date"].values
    
    for i in range(len(returns_series)):
        # Use all data up to and including date i, up to 504 days
        start = max(0, i - VAR_WINDOW + 1)
        window = returns_series[start:i+1]
        
        metrics = compute_var_cvar(window)
        metrics["date"] = dates[i]
        metrics["ticker"] = ticker
        metrics["window_size"] = len(window)
        results.append(metrics)
    
    return pd.DataFrame(results)


def build_var_cvar(returns_long_path: Path, dates_file: Path, workers: int) -> pd.DataFrame:
    """Build VaR/CVaR for all tickers."""
    print("\n=== HISTORICAL VaR & CVaR ===")
    print(f"Loading returns from {returns_long_path}...")
    
    # Load returns
    ret = pd.read_csv(returns_long_path, dtype={"ticker": str, "date": str})
    ret["date"] = pd.to_datetime(ret["date"])
    
    tickers = sorted(ret["ticker"].unique())
    print(f"  Tickers: {len(tickers):,}")
    
    # Process in parallel
    all_results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_var_cvar_ticker, t, ret): t for t in tickers}
        for future in tqdm(as_completed(futures), total=len(tickers), desc="  Computing VaR/CVaR"):
            all_results.append(future.result())
    
    result = pd.concat(all_results, ignore_index=True)
    result = result.sort_values(["ticker", "date"])
    
    out_path = OUT_DIR / "var_cvar.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    
    print(f"  Saved: {out_path} ({len(result):,} rows)")
    print(f"  Columns: {list(result.columns)}")
    
    # Quick stats
    for level in VAR_LEVELS:
        label = int(level * 100)
        var_col = f"var_{label}"
        cvar_col = f"cvar_{label}"
        if var_col in result.columns:
            print(f"  VaR {label}% mean: {result[var_col].mean():.4f}")
            print(f"  CVaR {label}% mean: {result[cvar_col].mean():.4f}")
    
    return result


# ══════════════════════════════════════════════════════════════
#  LIQUIDITY RISK
# ══════════════════════════════════════════════════════════════

def compute_liquidity_score(row: pd.Series) -> dict:
    """
    Compute liquidity risk metrics from one row of liquidity features.
    
    Liquidity score: 0 = completely illiquid (untradeable), 1 = highly liquid.
    
    Components:
      - dollar_volume: absolute daily trading value
      - volume_zscore: how unusual today's volume is vs. 21-day history
      - volume_ratio: today's volume / 21-day average
      - turnover_proxy: volume / 252-day average (long-term context)
    """
    dollar_vol = row.get("dollar_volume", 0) or 0
    vol_ratio = row.get("volume_ratio", 1) or 1
    vol_zscore = row.get("volume_zscore", 0) or 0
    turnover = row.get("turnover_proxy", 1) or 1
    
    # Dollar volume score: log-scale, 0 to 1
    # $1B/day = 1.0, $1M/day = ~0.4, $10K/day = ~0.1
    if dollar_vol > 0:
        dv_score = min(1.0, max(0.0, np.log10(max(dollar_vol, 1)) / 9.0))
    else:
        dv_score = 0.0
    
    # Volume ratio score: 1.0 is normal, >3.0 is very active, <0.1 is dead
    vr_score = min(1.0, max(0.0, vol_ratio / 2.0)) if not np.isnan(vol_ratio) else 0.5
    
    # Turnover score: 1.0 is normal, >2.0 is very active
    to_score = min(1.0, max(0.0, turnover / 2.0)) if not np.isnan(turnover) else 0.5
    
    # Composite score: weighted average
    liquidity_score = 0.4 * dv_score + 0.3 * vr_score + 0.3 * to_score
    liquidity_score = round(min(1.0, max(0.0, liquidity_score)), 4)
    
    # Slippage estimate (very rough proxy): higher volume ratio = less slippage
    # Based on square-root impact model: slippage ∝ 1/√(dollar_volume)
    if dollar_vol > 0:
        slippage_est = round(0.01 / np.sqrt(max(dollar_vol / 1e6, 0.001)), 6)
    else:
        slippage_est = 0.05  # 5% slippage if zero volume
    
    # Days to liquidate: how many days to trade $1M worth without >1% impact
    avg_daily_dollar = max(dollar_vol, 1)
    days_to_liquidate = round(1_000_000 / avg_daily_dollar, 1)
    
    # Tradable: boolean flag
    tradable = liquidity_score >= 0.3
    
    return {
        "liquidity_score": liquidity_score,
        "slippage_estimate_pct": slippage_est,
        "days_to_liquidate_1M": days_to_liquidate,
        "tradable": int(tradable),
        "dv_score": round(dv_score, 4),
        "vr_score": round(vr_score, 4),
        "to_score": round(to_score, 4),
    }


def build_liquidity(liquidity_path: Path) -> pd.DataFrame:
    """Build liquidity risk scores for all tickers."""
    print("\n=== LIQUIDITY RISK ===")
    print(f"Loading liquidity features from {liquidity_path}...")
    
    liq = pd.read_csv(liquidity_path, dtype={"ticker": str, "date": str})
    print(f"  Rows: {len(liq):,}")
    
    # Compute scores
    print("  Computing liquidity scores...")
    scores = liq.apply(compute_liquidity_score, axis=1)
    scores_df = pd.DataFrame(scores.tolist())
    
    result = pd.concat([liq[["date", "ticker"]], scores_df], axis=1)
    result = result.sort_values(["ticker", "date"])
    
    out_path = OUT_DIR / "liquidity.csv"
    result.to_csv(out_path, index=False)
    
    print(f"  Saved: {out_path} ({len(result):,} rows)")
    print(f"  Columns: {list(result.columns)}")
    
    # Quick stats
    print(f"\n  Liquidity score distribution:")
    for threshold in [0.9, 0.7, 0.5, 0.3, 0.1]:
        pct = (result["liquidity_score"] >= threshold).mean() * 100
        print(f"    ≥ {threshold:.1f}: {pct:.1f}%")
    
    print(f"  Tradable: {(result['tradable'] == 1).mean() * 100:.1f}%")
    print(f"  Median slippage estimate: {result['slippage_estimate_pct'].median():.6f}")
    
    return result


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    
    print("=" * 60)
    print("RISK ENGINE: VaR, CVaR & Liquidity")
    print("=" * 60)
    print(f"Non-parametric / rule-based — no training required")
    print()
    
    if not RETURNS_LONG.exists():
        print(f"X Returns file not found: {RETURNS_LONG}")
        print("   Run data/yFinance/engineer_features.py first.")
        return
    
    if not LIQUIDITY_IN.exists():
        print(f"X Liquidity file not found: {LIQUIDITY_IN}")
        print("   Run data/yFinance/engineer_features.py first.")
        return
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Build VaR/CVaR
    build_var_cvar(RETURNS_LONG, DATES_FILE, args.workers)
    
    # Build Liquidity
    build_liquidity(LIQUIDITY_IN)
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  {OUT_DIR / 'var_cvar.csv'}")
    print(f"  {OUT_DIR / 'liquidity.csv'}")


if __name__ == "__main__":
    main()

# ── Run instructions ──────────────────────────────────────────
# python code/riskEngine/var_cvar_liquidity.py --workers 6