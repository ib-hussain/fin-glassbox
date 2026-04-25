#!/usr/bin/env python3
"""
data/yfin_engineer_features.py

Build all market-derived feature files for the 2,500 ticker universe.
Memory-optimized: processes in chunks of 200 tickers.
No lookahead bias - all features strictly backward-looking.

Outputs:
  data/yFinance/processed/returns_panel_wide.csv    - 2500 tickers × 6285 days
  data/yFinance/processed/returns_long.csv          - ticker, date, log_return, simple_return
  data/yFinance/processed/liquidity_features.csv    - volume metrics
  data/yFinance/processed/features_temporal.csv     - 10 features for Temporal Encoder

Usage:
  python data/yfin_engineer_features.py --workers 6
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────
INPUT_FILE   = Path("data/yFinance/processed/ohlcv_final.csv")
DATES_FILE   = Path("data/market_dates_ONLY_NYSE.csv")
OUT_DIR      = Path("data/yFinance/processed")
TICKERS_FILE = Path("data/yFinance/processed/common_tickers.csv")

RETURNS_WIDE = OUT_DIR / "returns_panel_wide.csv"
RETURNS_LONG = OUT_DIR / "returns_long.csv"
LIQUIDITY    = OUT_DIR / "liquidity_features.csv"
FEATURES     = OUT_DIR / "features_temporal.csv"

CHUNK_SIZE = 200      # tickers per chunk
MIN_VOL_WINDOW = 5     # minimum days before computing volatility

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING FUNCTIONS
# ══════════════════════════════════════════════════════════════

def rsi(close_series: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI — strictly backward-looking."""
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def macd_hist(close_series: pd.Series) -> pd.Series:
    """MACD histogram: MACD - Signal line."""
    ema12 = close_series.ewm(span=12, adjust=False).mean()
    ema26 = close_series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line - signal


def bollinger_position(close_series: pd.Series, window: int = 20) -> pd.Series:
    """Where is price within Bollinger Bands? 0 = lower, 1 = upper."""
    sma = close_series.rolling(window).mean()
    std = close_series.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (close_series - lower) / (upper - lower).replace(0, np.nan)


def price_position_21(close_series: pd.Series) -> pd.Series:
    """Price position within 21-day range. 0 = 21d low, 1 = 21d high."""
    h = close_series.rolling(21).max()
    l = close_series.rolling(21).min()
    return (close_series - l) / (h - l).replace(0, np.nan)


def process_ticker_block(ticker: str, df: pd.DataFrame, spy_returns: pd.Series) -> dict:
    """
    Process one ticker's full history. Returns dict of DataFrames
    for each output file type.
    """
    df = df.sort_values("date").copy()
    close = df["close"]
    
    # ── Returns ──────────────────────────────────────────
    # All strictly backward-looking: pct_change uses previous close
    df["log_return"]   = np.log(close / close.shift(1))
    df["simple_return"] = close.pct_change()
    
    # ── Volatility (annualized) ──────────────────────────
    df["vol_5d"]  = df["log_return"].rolling(5).std() * np.sqrt(252)
    df["vol_21d"] = df["log_return"].rolling(21).std() * np.sqrt(252)
    
    # ── Technical indicators ─────────────────────────────
    df["rsi_14"]    = rsi(close, 14)
    df["macd_hist"] = macd_hist(close)
    df["bb_pos"]    = bollinger_position(close, 20)
    df["price_pos"] = price_position_21(close)
    
    # ── Range ────────────────────────────────────────────
    df["hl_ratio"] = (df["high"] - df["low"]) / close.replace(0, np.nan)
    
    # ── Volume ───────────────────────────────────────────
    vol_mean_21 = df["volume"].rolling(21).mean()
    vol_std_21  = df["volume"].rolling(21).std().replace(0, np.nan)
    df["dollar_volume"]  = close * df["volume"]
    df["volume_zscore"]  = (df["volume"] - vol_mean_21) / vol_std_21
    df["volume_ratio"]   = df["volume"] / vol_mean_21.replace(0, np.nan)
    df["turnover_proxy"] = df["volume"] / df["volume"].rolling(252).mean().replace(0, np.nan)
    
    # ── SPY correlation ──────────────────────────────────
    if spy_returns is not None:
        
        # Align SPY returns to same index
        spy_aligned = pd.Series(spy_returns, index=df.index)
        df["spy_corr_63d"] = df["log_return"].rolling(63).corr(spy_aligned)
    else:
        df["spy_corr_63d"] = np.nan
    
    # ── Build output DataFrames ──────────────────────────
    
    # Returns long
    ret_long = df[["date", "ticker", "log_return", "simple_return"]].copy()
    
    # Liquidity
    liq = df[["date", "ticker", "dollar_volume", "volume_zscore", 
              "volume_ratio", "turnover_proxy"]].copy()
    
    # Temporal features (10 features for encoder)
    feat = df[["date", "ticker",
               "log_return", "vol_5d", "vol_21d", 
               "rsi_14", "macd_hist", "bb_pos",
               "volume_ratio", "hl_ratio", "price_pos",
               "spy_corr_63d"]].copy()
    
    return {
        "return": df[["date", "log_return"]].copy(),
        "ret_long": ret_long,
        "liquidity": liq,
        "features": feat,
    }


# ══════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    
    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    print(f"Input: {INPUT_FILE}")
    print(f"Chunk size: {CHUNK_SIZE} tickers")
    print()
    
    # ── Load SPY for correlation ─────────────────────────
    print("Loading SPY data for benchmark correlation...")
    spy_data = None
    spy_returns = None
    for chunk in pd.read_csv(INPUT_FILE, dtype={"ticker": str, "date": str}, chunksize=500000):
        spy_chunk = chunk[chunk["ticker"] == "SPY"]
        if len(spy_chunk) > 0:
            if spy_data is None:
                spy_data = spy_chunk
            else:
                spy_data = pd.concat([spy_data, spy_chunk])
    if spy_data is not None:
        spy_data = spy_data.sort_values("date")
        spy_data["log_return"] = np.log(spy_data["close"] / spy_data["close"].shift(1))
        spy_returns = spy_data["log_return"].values
        print(f"  SPY: {len(spy_data)} rows")
    else:
        print("  ⚠️  SPY not found — skipping correlation feature")
    
    # ── Get ticker list ──────────────────────────────────
    tickers = []
    for chunk in pd.read_csv(INPUT_FILE, dtype={"ticker": str}, 
                             usecols=["ticker"], chunksize=500000):
        tickers.extend(chunk["ticker"].unique())
    tickers = sorted(set(tickers))
    print(f"\nTotal tickers: {len(tickers)}")
    
    # ── Process in chunks ────────────────────────────────
    ticker_chunks = [tickers[i:i+CHUNK_SIZE] for i in range(0, len(tickers), CHUNK_SIZE)]
    print(f"Processing {len(ticker_chunks)} chunks of ≤{CHUNK_SIZE} tickers...\n")
    
    # Accumulators for wide returns
    returns_wide_parts = []
    first_ret_long = True
    first_liq = True
    first_feat = True
    
    for chunk_idx, ticker_chunk in enumerate(ticker_chunks):
        # Load only these tickers
        print(f"Chunk {chunk_idx+1}/{len(ticker_chunks)}: loading {len(ticker_chunk)} tickers...")
        
        frames = []
        for chunk in pd.read_csv(INPUT_FILE, dtype={"ticker": str, "date": str}, chunksize=1000000):
            sub = chunk[chunk["ticker"].isin(ticker_chunk)]
            if len(sub) > 0:
                frames.append(sub)
        
        if not frames:
            continue
        panel = pd.concat(frames, ignore_index=True)
        panel["date"] = pd.to_datetime(panel["date"])
        panel = panel.sort_values(["ticker", "date"])
        
        # Process each ticker
        ret_wide_dict = {}
        ret_long_list = []
        liq_list = []
        feat_list = []
        
        for ticker in tqdm(ticker_chunk, desc=f"  Chunk {chunk_idx+1}"):
            sub = panel[panel["ticker"] == ticker]
            if len(sub) < 10:
                continue
            
            result = process_ticker_block(ticker, sub, spy_returns)
            
            # Store return for wide matrix (skip first row = NaN)
            ret_series = result["return"].set_index("date")["log_return"]
            ret_series = ret_series.iloc[1:]  # drop first NaN
            if len(ret_series) > 0:
                ret_wide_dict[ticker] = ret_series
            
            ret_long_list.append(result["ret_long"])
            liq_list.append(result["liquidity"])
            feat_list.append(result["features"])
        
        # ── Write chunk outputs (append mode) ────────────
        
        # Returns wide: accumulate for final pivot
        if ret_wide_dict:
            returns_wide_parts.append(pd.DataFrame(ret_wide_dict))
        
        # Returns long
        if ret_long_list:
            df_long = pd.concat(ret_long_list, ignore_index=True)
            df_long = df_long[df_long["log_return"].notna()]
            if first_ret_long:
                df_long.to_csv(RETURNS_LONG, index=False, mode="w")
                first_ret_long = False
            else:
                df_long.to_csv(RETURNS_LONG, index=False, mode="a", header=False)
        
        # Liquidity
        if liq_list:
            df_liq = pd.concat(liq_list, ignore_index=True)
            if first_liq:
                df_liq.to_csv(LIQUIDITY, index=False, mode="w")
                first_liq = False
            else:
                df_liq.to_csv(LIQUIDITY, index=False, mode="a", header=False)
        
        # Temporal features
        if feat_list:
            df_feat = pd.concat(feat_list, ignore_index=True)
            if first_feat:
                df_feat.to_csv(FEATURES, index=False, mode="w")
                first_feat = False
            else:
                df_feat.to_csv(FEATURES, index=False, mode="a", header=False)
        
        del panel, frames, ret_long_list, liq_list, feat_list
    
    # ── Build wide returns matrix ────────────────────────
    print("\nBuilding returns_panel_wide.csv...")
    if returns_wide_parts:
        wide = pd.concat(returns_wide_parts, axis=1)
        wide = wide.sort_index()
        wide.to_csv(RETURNS_WIDE)
        print(f"  Shape: {wide.shape[0]} dates × {wide.shape[1]} tickers")
        print(f"  Saved: {RETURNS_WIDE}")
    
    # ── Summary ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("OUTPUT FILES")
    print("=" * 60)
    for name, path in [
        ("Returns Wide", RETURNS_WIDE),
        ("Returns Long", RETURNS_LONG),
        ("Liquidity Features", LIQUIDITY),
        ("Temporal Features (10 cols)", FEATURES),
    ]:
        if path.exists():
            size_mb = path.stat().st_size / 1024**2
            print(f"  ✅ {name}: {path} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {name}: NOT CREATED")
    
    print("\nDone.")


if __name__ == "__main__":
    main()
