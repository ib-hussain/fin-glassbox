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
import json
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
    
    generate_var_cvar_xai(result)
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
    
    generate_liquidity_xai(result)
    return result

# ══════════════════════════════════════════════════════════════
#  XAI: RULE TRACE & HISTORICAL CONTEXT
# ══════════════════════════════════════════════════════════════

def generate_var_cvar_xai(var_cvar_df: pd.DataFrame) -> None:
    """Generate XAI explanations for VaR/CVaR module."""
    print("\n=== XAI: VaR/CVaR Explanations ===")
    xai_dir = OUT_DIR / "xai"
    xai_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Per-ticker historical context ──
    # For each ticker, compute historical percentiles of VaR/CVaR
    explanations = []
    
    for ticker in tqdm(var_cvar_df["ticker"].unique(), desc="  Generating VaR/CVaR XAI"):
        sub = var_cvar_df[var_cvar_df["ticker"] == ticker].sort_values("date")
        if len(sub) < 100:
            continue
        
        # Use the most recent date's values
        latest = sub.iloc[-1]
        
        # Compute historical percentiles
        var_95_hist = sub["var_95"].dropna()
        var_99_hist = sub["var_99"].dropna()
        cvar_95_hist = sub["cvar_95"].dropna()
        cvar_99_hist = sub["cvar_99"].dropna()
        
        explanation = {
            "module_name": "HistoricalVaRCVaR",
            "ticker": ticker,
            "date": str(latest["date"])[:10],
            "primary_score": float(round(abs(latest.get("var_95", np.nan)), 4)),
            "raw_value": {
                "var_95": float(round(latest.get("var_95", np.nan), 6)),
                "var_99": float(round(latest.get("var_99", np.nan), 6)),
                "cvar_95": float(round(latest.get("cvar_95", np.nan), 6)),
                "cvar_99": float(round(latest.get("cvar_99", np.nan), 6)),
                "tail_ratio_95": float(round(latest.get("tail_ratio_95", np.nan), 4)),
                "tail_ratio_99": float(round(latest.get("tail_ratio_99", np.nan), 4)),
            },
            "explanation": {
                "percentile_vs_history": {
                    "var_95_percentile": float(round(
                        (var_95_hist <= latest["var_95"]).mean() * 100, 1
                    )) if len(var_95_hist) > 0 else None,
                    "var_99_percentile": float(round(
                        (var_99_hist <= latest["var_99"]).mean() * 100, 1
                    )) if len(var_99_hist) > 0 else None,
                    "cvar_95_percentile": float(round(
                        (cvar_95_hist <= latest["cvar_95"]).mean() * 100, 1
                    )) if len(cvar_95_hist) > 0 else None,
                    "cvar_99_percentile": float(round(
                        (cvar_99_hist <= latest["cvar_99"]).mean() * 100, 1
                    )) if len(cvar_99_hist) > 0 else None,
                },
                "trend": _compute_trend(sub["var_95"].dropna().tail(60)),
                "thresholds_exceeded": _check_var_thresholds(latest),
                "top_positive_factors": [
                    {"factor": "2-year rolling window (504 trading days)", 
                     "weight": 1.0, "direction": "neutral"},
                    {"factor": "Non-parametric empirical distribution", 
                     "weight": 1.0, "direction": "neutral"},
                ],
                "top_negative_factors": [],
                "historical_range": {
                    "var_95_min": float(var_95_hist.min()) if len(var_95_hist) > 0 else None,
                    "var_95_max": float(var_95_hist.max()) if len(var_95_hist) > 0 else None,
                    "var_95_mean": float(var_95_hist.mean()) if len(var_95_hist) > 0 else None,
                    "current_var_95": float(latest.get("var_95", np.nan)),
                    "current_var_95_label": _var_severity_label(latest.get("var_95", np.nan), var_95_hist),
                },
            },
            "metadata": {
                "model_version": "non_parametric_historical",
                "window_size": VAR_WINDOW,
                "confidence_levels": VAR_LEVELS,
            },
        }
        explanations.append(explanation)
    
    # Save
    xai_path = xai_dir / "var_cvar_explanations.json"
    with open(xai_path, "w") as f:
        json.dump({
            "module": "HistoricalVaRCVaR",
            "method": "historical_empirical_distribution",
            "n_tickers": len(explanations),
            "window_size": VAR_WINDOW,
            "confidence_levels": VAR_LEVELS,
            "explanations": explanations,
        }, f, indent=2, default=str)
    
    print(f"  Saved: {xai_path}")


def generate_liquidity_xai(liquidity_df: pd.DataFrame) -> None:
    """Generate XAI explanations for Liquidity Risk module."""
    print("\n=== XAI: Liquidity Risk Explanations ===")
    xai_dir = OUT_DIR / "xai"
    xai_dir.mkdir(parents=True, exist_ok=True)
    
    explanations = []
    
    for ticker in tqdm(liquidity_df["ticker"].unique()[:500], desc="  Generating Liquidity XAI"):
        sub = liquidity_df[liquidity_df["ticker"] == ticker].sort_values("date")
        if len(sub) < 100:
            continue
        
        latest = sub.iloc[-1]
        liq_hist = sub["liquidity_score"].dropna()
        dv_hist = sub["dv_score"].dropna()
        
        # Rule trace — which components contributed
        rule_trace = []
        if latest["dv_score"] < 0.3:
            rule_trace.append({
                "rule": "low_dollar_volume",
                "condition": f"dv_score={latest['dv_score']:.3f} < 0.3",
                "severity": "critical" if latest["dv_score"] < 0.1 else "warning",
                "detail": f"Dollar volume score is low — median daily trading value may be insufficient"
            })
        if latest["vr_score"] < 0.3:
            rule_trace.append({
                "rule": "low_volume_ratio",
                "condition": f"vr_score={latest['vr_score']:.3f} < 0.3",
                "severity": "warning",
                "detail": "Volume relative to 21-day average is low"
            })
        if latest["days_to_liquidate_1M"] > DAYS_TO_LIQUIDATE_MAX:
            rule_trace.append({
                "rule": "slow_liquidation",
                "condition": f"days_to_liquidate={latest['days_to_liquidate_1M']:.1f} > {DAYS_TO_LIQUIDATE_MAX}",
                "severity": "warning",
                "detail": f"Would take {latest['days_to_liquidate_1M']:.1f} days to trade $1M"
            })
        if not latest["tradable"]:
            rule_trace.append({
                "rule": "untradeable",
                "condition": f"liquidity_score={latest['liquidity_score']:.3f} < 0.3",
                "severity": "critical",
                "detail": "Composite liquidity score below tradable threshold"
            })
        
        explanation = {
            "module_name": "LiquidityRisk",
            "ticker": ticker,
            "date": str(latest["date"])[:10],
            "primary_score": float(latest["liquidity_score"]),
            "confidence": 1.0,  # Rule-based = fully confident in the calculation
            "raw_value": {
                "liquidity_score": float(latest["liquidity_score"]),
                "slippage_estimate_pct": float(latest["slippage_estimate_pct"]),
                "days_to_liquidate_1M": float(latest["days_to_liquidate_1M"]),
                "tradable": bool(latest["tradable"]),
                "component_scores": {
                    "dollar_volume_score": float(latest["dv_score"]),
                    "volume_ratio_score": float(latest["vr_score"]),
                    "turnover_score": float(latest["to_score"]),
                },
            },
            "explanation": {
                "rule_trace": rule_trace,
                "percentile_vs_history": float(round(
                    (liq_hist <= latest["liquidity_score"]).mean() * 100, 1
                )) if len(liq_hist) > 0 else None,
                "thresholds_exceeded": [
                    {"threshold": "tradable_minimum", "current_value": float(latest["liquidity_score"]), 
                     "limit": 0.3, "severity": "critical" if latest["liquidity_score"] < 0.3 else "ok"}
                ],
                "top_positive_factors": [
                    {"factor": f"dollar_volume_score", "weight": float(latest["dv_score"]), 
                     "direction": "positive"}
                ] if latest["dv_score"] > 0.5 else [],
                "top_negative_factors": [
                    {"factor": f"dollar_volume_score", "weight": float(1 - latest["dv_score"]), 
                     "direction": "negative"}
                ] if latest["dv_score"] < 0.5 else [],
                "trend": _compute_trend(liq_hist.tail(60)),
                "component_breakdown": {
                    "dv_score_weight": 0.4,
                    "vr_score_weight": 0.3,
                    "to_score_weight": 0.3,
                },
            },
            "metadata": {
                "model_version": "rule_based",
                "scoring_formula": "0.4 * dv_score + 0.3 * vr_score + 0.3 * to_score",
            },
        }
        explanations.append(explanation)
    
    # Save
    xai_path = xai_dir / "liquidity_explanations.json"
    with open(xai_path, "w") as f:
        json.dump({
            "module": "LiquidityRisk",
            "method": "rule_based_composite_score",
            "n_tickers": len(explanations),
            "scoring_weights": {"dv_score": 0.4, "vr_score": 0.3, "to_score": 0.3},
            "explanations": explanations,
        }, f, indent=2, default=str)
    
    print(f"  Saved: {xai_path}")


def _compute_trend(series: pd.Series) -> dict:
    """Compute trend direction and strength from a series."""
    if len(series) < 10:
        return {"direction": "stable", "strength": 0.0}
    
    # Simple linear regression on last N points
    x = np.arange(len(series))
    y = series.values
    mask = ~np.isnan(y)
    if mask.sum() < 5:
        return {"direction": "stable", "strength": 0.0}
    
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 2 or np.std(y_clean) < 1e-10:
        return {"direction": "stable", "strength": 0.0}
    
    coeffs = np.polyfit(x_clean, y_clean, 1)
    slope = coeffs[0]
    
    # Normalize slope by the series std for strength
    strength = min(1.0, abs(slope * len(x_clean)) / (np.std(y_clean) + 1e-10))
    
    if abs(slope) < 1e-8:
        direction = "stable"
    elif slope > 0:
        direction = "increasing" if strength > 0.3 else "stable"
    else:
        direction = "decreasing" if strength > 0.3 else "stable"
    
    return {"direction": direction, "strength": round(float(strength), 3)}


def _check_var_thresholds(latest: pd.Series) -> list:
    """Check VaR values against warning thresholds."""
    thresholds = []
    var_95 = latest.get("var_95", np.nan)
    var_99 = latest.get("var_99", np.nan)
    tail_95 = latest.get("tail_ratio_95", np.nan)
    
    if not np.isnan(var_95) and var_95 < -0.05:
        thresholds.append({
            "threshold": "var_95_severe",
            "current_value": float(round(var_95, 4)),
            "limit": -0.05,
            "severity": "critical",
        })
    elif not np.isnan(var_95) and var_95 < -0.03:
        thresholds.append({
            "threshold": "var_95_elevated",
            "current_value": float(round(var_95, 4)),
            "limit": -0.03,
            "severity": "warning",
        })
    
    if not np.isnan(tail_95) and tail_95 > 1.5:
        thresholds.append({
            "threshold": "fat_tail_risk",
            "current_value": float(round(tail_95, 2)),
            "limit": 1.5,
            "severity": "warning",
        })
    
    return thresholds


def _var_severity_label(current_var: float, var_history: pd.Series) -> str:
    """Label the current VaR relative to history."""
    if np.isnan(current_var) or len(var_history) < 50:
        return "unknown"
    
    pct = (var_history <= current_var).mean()
    if pct > 0.9:
        return "extremely_high_risk"
    elif pct > 0.75:
        return "high_risk"
    elif pct > 0.5:
        return "elevated_risk"
    elif pct > 0.25:
        return "moderate_risk"
    else:
        return "low_risk"
    
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