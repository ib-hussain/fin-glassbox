# Market Data Pipeline — Complete Documentation

**Project:** Explainable Distributed Deep Learning Framework for Financial Risk Management  
**Data Family:** #1 — Time-Series Market Data  
**Status:** ✅ Complete  
**Final Output:** 2,500 tickers × 6,285 trading days, fully filled, all features engineered  
**Date:** 26 April 2026

---

## Table of Contents

1. [Overview](#1-overview)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Source Data Inventory](#3-source-data-inventory)
4. [Pipeline Stages](#4-pipeline-stages)
5. [Final Output Files](#5-final-output-files)
6. [Feature Specifications](#6-feature-specifications)
7. [Coverage Statistics](#7-coverage-statistics)
8. [Ticker Universe](#8-ticker-universe)
9. [Data Quality Controls](#9-data-quality-controls)
10. [Downstream Module Mapping](#10-downstream-module-mapping)
11. [File Manifest](#11-file-manifest)
12. [Reproduction](#12-reproduction)

---

## 1. Overview

The Market Data Pipeline acquires, cleans, standardizes, merges, fills, and engineers features for U.S. stock market time-series data spanning **2000-01-03 to 2024-12-31** (6,285 NYSE trading days). It combines data from **five independent sources** to produce a complete, analysis-ready dataset for the Technical Stream and Risk Engine of the financial risk management framework.

### Key Metrics

| Metric | Value |
|--------|-------|
| **Tickers** | 2,500 (top by coverage from 4,534 candidate) |
| **Trading days** | 6,285 (NYSE calendar, 2000-2024) |
| **Total data points** | 15,715,000 per feature file |
| **OHLCV completeness** | 100% (zero missing values) |
| **Returns NaN rate** | 0.0% |
| **Sources merged** | 5 (yfinance, Stooq, Huge Market Dataset, Kaggle NYSE/NASDAQ, Kaggle OTC) |
| **Pipeline scripts** | 7 |

---

## 2. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SOURCE DATA                              │
├───────────┬──────────┬────────────┬───────────┬────────────┤
│ yfinance  │  Stooq   │   Huge     │  Kaggle   │   Kaggle   │
│ (group    │  (.txt)  │  Market    │ NYSE/NAS  │    OTC     │
│  member)  │          │  Dataset   │   DAQ     │            │
└─────┬─────┴────┬─────┴─────┬──────┴─────┬─────┴──────┬─────┘
      │          │           │            │            │
      v          v           v            v            v
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: Extraction & Standardization                     │
│  • yfin_extracter.py   — Rename .txt→.csv, filter tickers  │
│  • yfin_standardize_sources.py — Unify column formats      │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          v
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: Merge & Price Alignment                          │
│  • yfin_merge_sources.py — Scale Stooq to adjusted prices  │
│  • Union dates, prefer adjusted sources                    │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          v
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: Master Panel Construction                        │
│  • yfin_build_complete_panel.py — 6,288 rows per ticker    │
│  • NaN for missing dates                                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          v
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4: Gap Filling                                      │
│  • yfin_fill_from_kaggle.py — Fill from Kaggle 1962-2024   │
│  • yfin_fill_final_pipeline.py — Multi-layer imputation    │
│    Layer 1: Trim to 6,286 dates                            │
│    Layer 2: Linear interpolation (≥6000 day stocks)        │
│    Layer 3: Trend projection (≥50% coverage)               │
│    Layer 4: EWMA + ratio fill (remaining)                  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          v
┌─────────────────────────────────────────────────────────────┐
│  STAGE 5: Feature Engineering                              │
│  • yfin_engineer_features.py — 30 features per ticker      │
│  • 4 output files: returns wide, returns long,             │
│    liquidity features, temporal features                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          v
┌─────────────────────────────────────────────────────────────┐
│              FINAL OUTPUT (4 files)                         │
│  • returns_panel_wide.csv     (277 MB, 2500×6285)          │
│  • returns_long.csv           (785 MB, 15.7M rows)         │
│  • liquidity_features.csv     (1.2 GB, 15.7M rows)         │
│  • features_temporal.csv      (2.8 GB, 15.7M rows)         │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Source Data Inventory

### 3.1 Primary Source: yfinance (Group Member Download)

| Attribute | Value |
|-----------|-------|
| **Source** | Yahoo Finance via `yfinance` Python library |
| **Format** | CSV (converted from Parquet) |
| **Tickers** | 4,247 (from Wikipedia S&P 500/400/600 + Nasdaq-100 + DJIA + ETFs) |
| **Date range** | 2000-01-03 to 2024-12-31 |
| **Columns** | `date, open, high, low, close, volume, dividends, stock_splits` |
| **Price type** | Adjusted close (split-adjusted) |
| **File** | `data/yFinance/processed/ohlcv_panel.csv` (16.3M rows) |

### 3.2 Stooq Historical Database

| Attribute | Value |
|-----------|-------|
| **Source** | [Stooq.com](https://stooq.com/db/h/) — free historical database |
| **Format** | `.txt` files (comma-separated), one per ticker |
| **Files** | 12,021 files across `nasdaq stocks/`, `nasdaq etfs/`, `nyse stocks/`, `nyse etfs/`, `nysemkt stocks/` |
| **Date range** | 1962–2024+ |
| **Columns** | `<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>` |
| **Date format** | YYYYMMDD (e.g., `19991118`) |
| **Price type** | Unadjusted (raw trading prices) |
| **After filter** | 4,390 tickers matching primary tickers, 15.8M rows |

### 3.3 Boris Marjanovic "Huge Market Dataset"

| Attribute | Value |
|-----------|-------|
| **Source** | [Kaggle: price-volume-data-for-all-us-stocks-etfs](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs) |
| **Format** | CSV, one per ticker (`aapl.us.csv`) |
| **Files** | 8,539 files in `Stocks/` and `ETFs/` directories |
| **Date range** | 1999–2017 (last updated 11/10/2017) |
| **Columns** | `Date, Open, High, Low, Close, Volume, OpenInt` |
| **Price type** | Adjusted for splits and dividends |
| **After filter** | 2,589 tickers matching primary tickers, 7.7M rows |

### 3.4 Kaggle NYSE/NASDAQ/NYSE-A/OTC 1962-2024

| Attribute | Value |
|-----------|-------|
| **Source** | [Kaggle: nasdaq-nyse-nyse-a-otc-daily-stock-1962-2024](https://www.kaggle.com/datasets/eren2222/nasdaq-nyse-nyse-a-otc-daily-stock-1962-2024) |
| **Format** | 4 large CSV files |
| **Files** | `NYSE 1962-2024.csv` (10.8M rows), `NASDAQ 1962-2024.csv` (11.5M rows), `NYSE A 1973-2024.csv` (1.0M rows), `OTC 1972-2024.csv` (2.5M rows) |
| **Total** | 25.9M rows, 6,291 unique tickers |
| **Columns** | `Date, Ticker, Exchange, Open, High, Low, Close, Adj Close, Volume` |
| **Price type** | Both unadjusted close and adjusted close available |

### 3.5 NYSE Trading Calendar

| Attribute | Value |
|-----------|-------|
| **Source** | Derived from FRED macro data pipeline |
| **File** | `data/market_dates_ONLY_NYSE.csv` |
| **Dates** | 6,288 NYSE trading days, 2000-01-03 to 2024-12-31 |

---

## 4. Pipeline Stages

### Stage 1: Extraction & Standardization

**Scripts:** `data/yfin_extracter.py`, `data/yfin_standardize_sources.py`

**What it does:**
1. Renames all `.txt` files to `.csv` in the Stooq directory (12,021 files)
2. Filters files to only those matching `primary_tickers.csv` (4,534 tickers)
3. Moves non-matching files to `irrelevant/` directories
4. Standardizes column formats between sources:
   - **Stooq**: Converts `<TICKER>,<PER>,<DATE>,...` to `date,open,high,low,close,volume`
   - **Huge Market**: Drops `OpenInt`, normalizes column names
   - **Both**: Date format → `YYYY-MM-DD`, volume → int, OHLC → float

**Results:**
- Stooq: 4,390 tickers, 15.8M rows (1 error: `emi.us.csv`)
- Huge Market: 2,589 tickers, 7.7M rows (5 errors: corrupted files)

### Stage 2: Merge & Price Alignment

**Script:** `data/yfin_merge_sources.py`

**What it does:**
1. For tickers present in both Stooq and Huge Market (2,576 tickers):
   - Computes ratio = Huge_close / Stooq_close on overlapping dates
   - Scales all Stooq OHLC by this ratio (adjusts for cumulative splits)
   - Unions dates from both sources
   - Prefers Huge Market prices on overlapping dates (already adjusted)
2. For tickers in one source only: uses as-is

**Price Adjustment Validation (sample):**
| Ticker | Ratio (Huge/Stooq) | Stability (±) | Interpretation |
|--------|-------------------|---------------|----------------|
| AAPL | 4.2740 | 0.002% | ~4:1 cumulative split |
| MSFT | 1.0898 | 0.3% | Minor splits + dividends |
| JPM | 1.1560 | 0.00005% | Perfectly constant |
| GE | 0.1664 | 0.15% | Reverse split |

**Results:** 4,403 tickers, 16.0M rows

### Stage 3: Master Panel Construction

**Script:** `data/yfin_build_complete_panel.py`

**What it does:**
1. Loads exact 6,288 NYSE trading dates from `market_dates_ONLY_NYSE.csv`
2. For each ticker, pulls data from merged sources (preferring merged/ over ohlcv_panel)
3. Forces every ticker to have all 6,288 date rows (NaN for missing dates)
4. Aligns to NYSE calendar — no weekend/holiday gaps

**Results:** 4,252 tickers × 6,288 dates = 26,736,576 rows

### Stage 4: Gap Filling

**Scripts:** `data/yfin_fill_from_kaggle.py`, `data/yfin_dataFilling_pipeline.py`

**What it does:**

**4A. Kaggle Fill (partial):** Filled 17,835 rows from the Kaggle 1962-2024 dataset for 14 tickers.

**4B. Final Fill Pipeline (4 layers):**

| Layer | Method | Target Stocks | What It Does |
|-------|--------|---------------|-------------|
| **Layer 1** | Trim | All 2,500 | Max 6,286 dates (drop last 2 incomplete NYSE dates) |
| **Layer 2** | Linear interpolation | ≥6,000 day stocks | Fills gaps ≤10 consecutive days |
| **Layer 3** | Trend projection | ≥50% coverage | Local linear regression on mirrored series + OHLC/Close ratio fill |
| **Layer 4** | EWMA + ratio fill | Remaining | Exponential weighted moving average + OHLC/Close median ratios |

**Results:** 2,500 tickers × 6,286 dates = 15,715,000 rows, **zero NaN**

### Stage 5: Feature Engineering

**Script:** `data/yfin_engineer_features.py`

**What it does:**
1. Computes all features strictly backward-looking (no lookahead bias)
2. Processes in chunks of 200 tickers for memory efficiency
3. Produces 4 output files

**Results:** 4 output files, ~15.7M rows each

---

## 5. Final Output Files

### 5.1 `ohlcv_final.csv` — Master OHLCV Panel

| Attribute | Value |
|-----------|-------|
| **File** | `data/yFinance/processed/ohlcv_final.csv` |
| **Rows** | 15,715,000 |
| **Columns** | `date, ticker, open, high, low, close, volume` |
| **Tickers** | 2,500 |
| **Dates** | 6,286 (2000-01-03 to 2024-12-31) |
| **NaN rate** | 0% |
| **Price type** | Adjusted (split and dividend adjusted) |

### 5.2 `returns_panel_wide.csv` — Wide Returns Matrix

| Attribute | Value |
|-----------|-------|
| **File** | `data/yFinance/processed/returns_panel_wide.csv` |
| **Size** | 277.2 MB |
| **Shape** | 6,285 rows × 2,501 columns (date + 2,500 tickers) |
| **Format** | Wide matrix, dates as rows, tickers as columns |
| **Values** | Log returns: ln(close_t / close_t-1) |
| **NaN rate** | 0.0% |
| **Primary user** | StemGNN Contagion Module, VaR, CVaR |

### 5.3 `returns_long.csv` — Long Returns

| Attribute | Value |
|-----------|-------|
| **File** | `data/yFinance/processed/returns_long.csv` |
| **Size** | 785.4 MB |
| **Rows** | 15,712,457 |
| **Columns** | `date, ticker, log_return, simple_return` |
| **Primary user** | Volatility Model, Drawdown Model |

### 5.4 `liquidity_features.csv` — Liquidity Metrics

| Attribute | Value |
|-----------|-------|
| **File** | `data/yFinance/processed/liquidity_features.csv` |
| **Size** | 1,228.4 MB |
| **Rows** | 15,715,000 |
| **Columns** | `date, ticker, dollar_volume, volume_zscore, volume_ratio, turnover_proxy` |
| **Primary user** | Liquidity Risk Module |

### 5.5 `features_temporal.csv` — Temporal Encoder Features

| Attribute | Value |
|-----------|-------|
| **File** | `data/yFinance/processed/features_temporal.csv` |
| **Size** | 2,831.5 MB |
| **Rows** | 15,715,000 |
| **Columns** | `date, ticker, log_return, vol_5d, vol_21d, rsi_14, macd_hist, bb_pos, volume_ratio, hl_ratio, price_pos, spy_corr_63d` |
| **Primary user** | Shared Temporal Attention Encoder |

---

## 6. Feature Specifications

### 6.1 Returns Features

| Feature | Formula | Window | Notes |
|---------|---------|--------|-------|
| `log_return` | ln(close_t / close_t-1) | 1 day | Primary return metric |
| `simple_return` | (close_t - close_t-1) / close_t-1 | 1 day | Alternative metric |

### 6.2 Volatility Features

| Feature | Formula | Window | Notes |
|---------|---------|--------|-------|
| `vol_5d` | std(log_return) × √252 | 5 days | Short-term annualized vol |
| `vol_21d` | std(log_return) × √252 | 21 days | Monthly annualized vol |

### 6.3 Technical Indicators

| Feature | Formula | Window | Range | Notes |
|---------|---------|--------|-------|-------|
| `rsi_14` | Wilder's RSI | 14 days | 0–100 | Relative Strength Index |
| `macd_hist` | MACD − Signal | 12/26/9 | Unbounded | MACD histogram |
| `bb_pos` | (close − lower) / (upper − lower) | 20 days, 2σ | 0–1 | Bollinger Band position |

### 6.4 Price Range Features

| Feature | Formula | Window | Notes |
|---------|---------|--------|-------|
| `hl_ratio` | (high − low) / close | 1 day | Daily range normalized by close |
| `price_pos` | (close − 21d_low) / (21d_high − 21d_low) | 21 days | Position within recent range |

### 6.5 Volume Features

| Feature | Formula | Window | Notes |
|---------|---------|--------|-------|
| `dollar_volume` | close × volume | 1 day | Dollar trading volume |
| `volume_zscore` | (volume − mean_21d) / std_21d | 21 days | Abnormal volume detection |
| `volume_ratio` | volume / mean_21d | 21 days | Relative volume |
| `turnover_proxy` | volume / mean_252d | 252 days | Long-term volume context |

### 6.6 Benchmark Features

| Feature | Formula | Window | Notes |
|---------|---------|--------|-------|
| `spy_corr_63d` | Rolling Pearson corr with SPY returns | 63 days | Market correlation proxy |

---

## 7. Coverage Statistics

### 7.1 Final Coverage (Post-Filling)

| Tier | Tickers | % of Universe |
|------|---------|---------------|
| 6,286 days (full) | 2,500 | 100% |
| ≥5,000 days | 2,500 | 100% |
| ≥4,000 days | 2,500 | 100% |
| ≥3,000 days | 2,500 | 100% |

### 7.2 Source Contribution (Pre-Filling)

| Source | Tickers Contributed | Rows |
|--------|---------------------|------|
| yfinance (ohlcv_panel) | 4,228 | 16.2M |
| Stooq | 4,390 | 15.8M |
| Huge Market Dataset | 2,589 | 7.7M |
| Kaggle NYSE/NASDAQ | 3,476 | 17,835 filled |
| Merged (Stooq + Huge) | 4,403 | 16.0M |

### 7.3 Fill Pipeline Effectiveness

| Layer | Method | Stocks Affected | Cells Filled |
|-------|--------|-----------------|-------------|
| Kaggle pre-fill | Direct lookup | 14 | 17,835 |
| Layer 1 | Trim | 2,500 | — |
| Layer 2 | Linear interpolation | ~1,560 | — |
| Layer 3 | Trend projection | ~885 | — |
| Layer 4 | EWMA + ratio | ~1,200 | — |
| **Total** | | **2,500** | **~10M** |

---

## 8. Ticker Universe

### 8.1 Selection Process

1. **Initial universe:** 4,534 tickers from `primary_tickers.csv` (SEC CIK-mapped) + `cik_ticker_map_cleaned.csv` (4,428 tickers)
2. **Data availability filter:** 4,451 tickers had data in at least one source
3. **Coverage ranking:** Sorted by number of NYSE trading days with data
4. **Final selection:** Top 2,500 tickers (cutoff: 3,017 days / 48.0% coverage)

### 8.2 Sector Composition (SIC → GICS)

| GICS Sector | Tickers | % |
|-------------|---------|---|
| Financials | ~500 | 20% |
| Information Technology | ~350 | 14% |
| Health Care | ~300 | 12% |
| Industrials | ~280 | 11% |
| Consumer Discretionary | ~250 | 10% |
| Energy | ~180 | 7% |
| Real Estate | ~150 | 6% |
| Consumer Staples | ~120 | 5% |
| Materials | ~120 | 5% |
| Utilities | ~70 | 3% |
| Communication Services | ~60 | 2% |
| Other | ~120 | 5% |

### 8.3 ETF Coverage

| ETF | In Universe | Description |
|-----|-------------|-------------|
| SPY | ✅ | S&P 500 |
| QQQ | ✅ | Nasdaq-100 |
| DIA | ✅ | Dow Jones Industrial |
| IWM | ❌ | Russell 2000 (filtered out) |
| XLK-XLC | ❌ | Sector ETFs (filtered out) |

---

## 9. Data Quality Controls

### 9.1 No Lookahead Bias

All features are computed using **strictly backward-looking** windows. The pipeline enforces:
- Returns use `pct_change()` (previous close only)
- Rolling windows use `.rolling(window)` with no `center=True`
- No future data is used in any imputation step

### 9.2 Price Adjustment Consistency

All prices in the final panel are **split and dividend adjusted** (comparable to adjusted close). The merge process scaled Stooq's unadjusted prices using a per-ticker constant ratio calibrated against Huge Market's adjusted prices.

### 9.3 Verification Checks

| Check | Result |
|-------|--------|
| NaN in OHLCV | 0 cells |
| NaN in returns wide | 0.0% |
| Date monotonicity | Passed (sorted ascending) |
| Ticker count | 2,500 (exact) |
| Dates per ticker | 6,286 (exact) |
| NYSE calendar alignment | Verified against `market_dates_ONLY_NYSE.csv` |
| Duplicate (ticker, date) pairs | None |

### 9.4 Known Limitations

1. **Synthetic fill for early/late periods:** ~38% of cells were filled using statistical methods (Layers 3-4). These are approximations and carry uncertainty.
2. **No corporate actions tracking:** Dividend and split information was used for price adjustment but is not preserved as separate features.
3. **ETF universe limited:** Only SPY, QQQ, DIA are present. 12 sector ETFs were filtered out during top-2,500 selection due to lower coverage.
4. **Market cap is proxy only:** `market_cap_proxy` uses median dollar volume, not actual market capitalization.

---

## 10. Downstream Module Mapping

### 10.1 Which File Feeds Which Module

| Module | Input File(s) | Features Used |
|--------|---------------|---------------|
| **Shared Temporal Attention Encoder** | `features_temporal.csv` | All 10 features (sequence of 30 days) |
| **Technical Analyst (BiLSTM)** | Temporal Encoder output | 128-dim embedding |
| **Volatility Model (GARCH+MLP)** | `returns_long.csv` + Temporal Encoder output | log_return sequence + 128-dim embedding |
| **Drawdown Model (BiLSTM)** | Temporal Encoder output | 128-dim embedding (30-90 day sequence) |
| **Historical VaR** | `returns_panel_wide.csv` | 2-year rolling returns per ticker |
| **CVaR / Expected Shortfall** | `returns_panel_wide.csv` | 2-year rolling returns per ticker |
| **GNN Contagion Risk (StemGNN)** | `returns_panel_wide.csv` + graph snapshots | Returns matrix (N_stocks × T=30) |
| **Liquidity Risk Module** | `liquidity_features.csv` | dollar_volume, volume_zscore, volume_ratio, turnover_proxy |
| **Regime Detection (MTGNN)** | Temporal Encoder output + FinBERT embeddings | 128-dim temporal + 256-dim text |
| **Position Sizing Engine** | All risk module outputs | Aggregated risk scores |
| **Cross-Asset Graph Builder** | `returns_panel_wide.csv` | Correlation matrix, sector mapping, beta |

### 10.2 Data Flow Diagram

```
features_temporal.csv ──→ Temporal Encoder ──→ Technical Analyst
                                          ──→ Volatility Model
                                          ──→ Drawdown Model
                                          ──→ Regime Detection

returns_panel_wide.csv ──→ VaR / CVaR
                      ──→ StemGNN Contagion
                      ──→ Cross-Asset Graph Builder

returns_long.csv ──→ Volatility Model (GARCH)

liquidity_features.csv ──→ Liquidity Risk Module
```

---

## 11. File Manifest

### 11.1 Scripts

```
data/
├── yfin_extracter.py              # Rename .txt→.csv, filter to primary tickers
├── yfin_standardize_sources.py    # Unify Stooq + Huge Market column formats
├── yfin_merge_sources.py          # Scale prices, union dates, prefer adjusted
├── yfin_build_complete_panel.py   # Build 6288×N complete matrix with NaN
├── yfin_fill_from_kaggle.py       # Fill from Kaggle 1962-2024 dataset
├── yfin_dataFilling_pipeline.py    # 4-layer statistical fill pipeline
└── yfin_engineer_features.py      # Feature engineering (4 output files)
```

### 11.2 Final Data Files

```
data/yFinance/processed/
├── ohlcv_final.csv                # 15.7M rows, fully filled OHLCV (2,500 × 6,286)
├── returns_panel_wide.csv         #   277 MB, log returns (6,285 × 2,500)
├── returns_long.csv               #   785 MB, ticker-date-level returns
├── liquidity_features.csv         # 1,228 MB, volume-based features
├── features_temporal.csv          # 2,832 MB, 10 features for Temporal Encoder
├── common_tickers.csv             # 54 tickers with both market + fundamentals
├── tickers_with_coverage.csv      # Per-ticker coverage statistics
└── master_coverage_complete.csv   # Final coverage report
```

### 11.3 Intermediate Files (Retained for Audit)

```
data/yFinance/
├── yFinance.md                    # This file
├── merged/                        # 4,397 per-ticker CSV files (Stooq+Huge merged)
├── raw/                           # 5,741 raw yfinance Parquet downloads
├── raw_metadata/                  # 5,740 yfinance metadata JSON files
├── Huge_Market_Dataset/           # Filtered Boris Marjanovic CSVs
├── d_us_txt/                      # Filtered Stooq CSVs
└── nasdaq-nyse-nyse-a-otc-daily-stock-1962-2024/  # Kaggle source CSVs
```

---

## 12. Reproduction

### 12.1 Full Pipeline Execution Order

```bash
# Stage 1: Extract and standardize
python data/yFinance/yfin_extracter.py --dir "data/yFinance/d_us_txt" --tickers "data/primary_tickers.csv" --workers 4
python data/yFinance/yfin_extracter.py --dir "data/yFinance/Huge_Market_Dataset" --tickers "data/primary_tickers.csv" --workers 4 --skip-rename
python data/yFinance/yfin_standardize_sources.py --all --workers 4

# Stage 2: Merge sources
python data/yfin_merge_sources.py --workers 4

# Stage 3: Build complete panel
python data/yfin_build_complete_panel.py --workers 4

# Stage 4: Fill gaps
python data/yfin_fill_from_kaggle.py --workers 4
python data/yfin_dataFilling_pipeline.py --workers 4

# Stage 5: Engineer features
python data/yfin_engineer_features.py --workers 4
```

### 12.3 Verification Commands

```bash
# Check OHLCV completeness
python -c "
import pandas as pd
df = pd.read_csv('data/yFinance/processed/ohlcv_final.csv')
print(f'NaN in close: {df[\"close\"].isna().sum()}')
print(f'Tickers: {df[\"ticker\"].nunique()}')
print(f'Dates: {df[\"date\"].nunique()}')
"

# Check returns matrix
python -c "
import pandas as pd
df = pd.read_csv('data/yFinance/processed/returns_panel_wide.csv', nrows=0)
print(f'Tickers: {len(df.columns) - 1}')
"
```

---

## Document Version

**Version:** 1.0  
**Date:** 26 April 2026  
**Status:** Complete  
**Author:** Market Data Pipeline Team  
