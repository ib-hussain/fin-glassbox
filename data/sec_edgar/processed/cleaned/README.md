# SEC Processed Data - Cleaned Datasets

## Overview

This directory contains the cleaned and production-ready versions of the SEC EDGAR processed data. These datasets have undergone rigorous cleaning, filtering, deduplication, and standardization to prepare them for use in the **Explainable Distributed Deep Learning Framework for Financial Risk Management**.

The datasets in this directory are the **definitive source** for:
- **Fundamental Analyst** module input
- **U.S. issuer universe definition**
- **CIK-to-ticker mapping** for joining with market data

---

## Data Lineage Summary

```
SEC EDGAR Bulk Archives (submissions.zip, companyfacts.zip)
                    ↓
            Raw JSON Extraction
                    ↓
    ┌───────────────┴───────────────┐
    ↓                               ↓
submissions_pipeline.py    companyfacts_pipeline.py
    ↓                               ↓
issuer_master.csv (973K)    facts_part_*.csv (121.9M rows)
    ↓                               ↓
    └───────────┬───────────────────┘
                ↓
    sec_issuer_master.py + sec_core_fundamentals.py
                ↓
    cik_ticker_map.csv (5,644) + core_fundamentals_quarterly.csv (7,310)
                ↓
    sec_fundamentals_features.py (derived ratios + growth)
                ↓
    fundamentals_features.csv (7,310 rows, 70 columns)
                ↓
    ┌───────────┴───────────┐
    ↓                       ↓
sec_data_cleaning_pipeline.py (THIS STEP)
    ↓
cleaned/ directory (production-ready datasets)
```

---

## Input Datasets (Before Cleaning)

### 1. `cik_ticker_map.csv`
**Source:** Derived from `issuer_master.csv` by `sec_issuer_master.py`

| Metric | Value |
|--------|-------|
| Original rows | 5,644 |
| Columns | `cik`, `padded_cik`, `primary_ticker`, `entity_name`, `primary_exchange` |
| Issues identified | 2% missing `primary_exchange`, 4 distinct exchange values (Nasdaq, NYSE, OTC, Other) |

### 2. `issuer_master.csv`
**Source:** Built by `sec_issuer_master.py` from submissions and companyfacts inventories

| Metric | Value |
|--------|-------|
| Original rows | 973,279 |
| Columns | 35 columns including entity metadata, filing coverage, addresses |
| Issues identified | ~90% missing `primary_ticker`, non-US issuers included, redundant location columns |

### 3. `fundamentals_features.csv`
**Source:** Built by `sec_fundamentals_features.py` from `core_fundamentals_quarterly.csv`

| Metric | Value |
|--------|-------|
| Original rows | 7,310 |
| Columns | 70 (36 raw + 34 derived features) |
| Issues identified | 100% duplicate rows, 100% null columns, missing ticker/entity_name, extreme outliers |

---

## Cleaning Operations Performed

### Step 1: `cik_ticker_map.csv` Cleaning

| Operation | Details | Rows Affected |
|-----------|---------|---------------|
| Remove missing exchange | Drop rows where `primary_exchange` is empty | ~113 rows (2%) |
| Filter valid exchanges | Keep only `Nasdaq` and `NYSE` | ~30 rows removed |
| **Result** | Clean mapping of CIK → ticker for major U.S. exchanges | **~5,500 rows** |

### Step 2: `issuer_master.csv` Cleaning

| Operation | Details | Impact |
|-----------|---------|--------|
| US-only filter | Keep `is_us_issuer == 1`, then drop column | 851,569 retained (87.5%) |
| Exchange filter | Keep `Nasdaq`, `NYSE`, or missing values | OTC/other exchanges removed |
| Column pruning | Drop `all_tickers`, `padded_cik`, `entity_type`, `state_of_incorporation_desc`, `state_of_incorporation` | 6 columns removed |
| Ticker filling | Fill missing `primary_ticker` using `cik_ticker_map` | ~90% of missing tickers filled |
| **Result** | US-only issuer registry with cleaned columns | **~850,000 rows** |

**Columns retained:**
- Identifiers: `cik`, `entity_name`, `primary_ticker`, `primary_exchange`
- Business: `sic`, `sic_description`, `fiscal_year_end`, `business_city`
- Coverage: `has_submissions`, `has_companyfacts`, filing date ranges, fact counts

### Step 3: `fundamentals_features.csv` Cleaning

| Operation | Details | Impact |
|-----------|---------|--------|
| **Deduplication** | Keep first occurrence of each (`cik`, `fiscal_year`, `fiscal_period`, `filing_date`) | 50% reduction (3,655 rows) |
| **Drop null columns** | Remove columns with 100% missing values: `inventory`, `goodwill`, `intangible_assets`, `short_term_debt` | 4 columns removed |
| **Fill identifiers** | Populate missing `ticker` and `entity_name` from `cik_ticker_map` | All rows now have identifiers |
| **Cap outliers** | Limit all ratios to ±1000% (±10.0) and reasonable bounds | Extreme values clipped |
| **Result** | Clean, deduplicated, normalized fundamentals | **~3,655 rows, 66 columns** |

**Outlier capping details:**
- Profitability margins: `[-10.0, 10.0]` (-1000% to +1000%)
- Return ratios (ROA, ROE): `[-10.0, 10.0]`
- Leverage ratios: `debt_to_equity` `[-10.0, 100.0]`, `current_ratio` `[0.0, 100.0]`
- Growth rates (YoY, QoQ): `[-10.0, 10.0]`

---

## Output Datasets (After Cleaning)

### 1. `cik_ticker_map_cleaned.csv`
**Purpose:** Canonical CIK-to-ticker mapping for U.S. public companies on major exchanges.

| Field | Type | Description |
|-------|------|-------------|
| `cik` | string | SEC Central Index Key (10-digit, no padding) |
| `padded_cik` | string | CIK zero-padded to 10 digits |
| `primary_ticker` | string | Trading symbol (e.g., AAPL, MSFT) |
| `entity_name` | string | Legal entity name |
| `primary_exchange` | string | `Nasdaq` or `NYSE` only |

**Usage:**
- Join with market data (yfinance) using `primary_ticker`
- Map SEC filings to trading symbols
- Define stock universe for prediction

### 2. `issuer_master_cleaned.csv`
**Purpose:** Comprehensive registry of all U.S. SEC filers with coverage metadata.

| Field | Type | Description |
|-------|------|-------------|
| `cik` | string | SEC Central Index Key |
| `entity_name` | string | Legal entity name |
| `primary_ticker` | string | Trading symbol (if publicly traded) |
| `primary_exchange` | string | Exchange (Nasdaq, NYSE, or empty) |
| `sic` | string | Standard Industrial Classification code |
| `sic_description` | string | Industry description |
| `fiscal_year_end` | string | Month/Day of fiscal year end (MMDD) |
| `business_city` | string | Headquarters city |
| `has_submissions` | string | "1" if filings metadata available |
| `has_companyfacts` | string | "1" if XBRL fundamentals available |
| `earliest_filing_sub` | date | First filing date in submissions |
| `latest_filing_sub` | date | Most recent filing date |
| `earliest_filed_facts` | date | First XBRL fact date |
| `latest_filed_facts` | date | Most recent XBRL fact date |
| `n_fact_observations` | integer | Total XBRL facts available |

**Usage:**
- Audit data coverage by company
- Filter companies with sufficient history
- Sector/industry grouping for analysis

### 3. `fundamentals_features_cleaned.csv`
**Purpose:** Point-in-time quarterly fundamentals with derived ratios for Fundamental Analyst module.

**Raw fields (36 columns):**
| Category | Fields |
|----------|--------|
| Identifiers | `cik`, `ticker`, `entity_name` |
| Period | `fiscal_year`, `fiscal_period` (Q1-Q4, FY), `filing_date`, `period_end_date` |
| Filing | `form_type` (10-K, 10-Q), `accession` |
| Income | `revenue`, `cost_of_revenue`, `gross_profit`, `operating_expenses`, `operating_income`, `net_income`, `eps_basic`, `eps_diluted`, `shares_basic` |
| Assets | `total_assets`, `current_assets`, `cash_and_equivalents`, `ppe_net` |
| Liabilities | `total_liabilities`, `current_liabilities`, `long_term_debt` |
| Equity | `shareholders_equity`, `retained_earnings` |
| Cash Flow | `operating_cash_flow`, `investing_cash_flow`, `financing_cash_flow`, `capex`, `free_cash_flow` |

**Derived features (30 columns):**
| Category | Fields |
|----------|--------|
| Profitability | `gross_margin`, `operating_margin`, `net_margin`, `opex_to_revenue`, `cogs_to_revenue` |
| Returns | `roa`, `roe` |
| Leverage | `debt_to_equity`, `debt_to_assets`, `current_ratio`, `quick_ratio`, `cash_ratio` |
| Efficiency | `asset_turnover`, `ppe_turnover` |
| Cash Flow | `ocf_to_revenue`, `fcf_to_revenue`, `capex_to_revenue`, `fcf_to_net_income`, `ocf_to_net_income` |
| Per Share | `revenue_per_share`, `book_value_per_share`, `ocf_per_share`, `fcf_per_share` |
| Quality | `accruals_to_assets` |
| Growth | `revenue_growth_yoy`, `revenue_growth_qoq`, `net_income_growth_yoy`, `net_income_growth_qoq`, `operating_income_growth_yoy`, `operating_income_growth_qoq`, `eps_basic_growth_yoy`, `eps_basic_growth_qoq`, `total_assets_growth_yoy`, `total_assets_growth_qoq` |

**Point-in-Time Guarantee:**
All data uses the **earliest filing date** for each fiscal period, ensuring no lookahead bias from restated values. This is critical for backtesting integrity.

---

## Data Quality Summary

| Dataset | Rows | Completeness | Key Quality Metrics |
|---------|------|--------------|---------------------|
| `cik_ticker_map_cleaned.csv` | ~5,500 | 100% | No missing tickers or exchanges |
| `issuer_master_cleaned.csv` | ~850,000 | 100% for key fields | All U.S. issuers, tickers filled where available |
| `fundamentals_features_cleaned.csv` | ~3,655 | 100% for identifiers | Deduplicated, outliers capped, ratios normalized |

---

## Usage in the Framework

### Fundamental Analyst Module
```python
import pandas as pd

# Load cleaned fundamentals
fundamentals = pd.read_csv("data/sec_edgar/processed/cleaned/fundamentals_features_cleaned.csv")

# Features ready for modeling (all normalized ratios)
feature_columns = [
    "gross_margin", "operating_margin", "net_margin",
    "roa", "roe", "debt_to_equity", "current_ratio",
    "revenue_growth_yoy", "net_income_growth_yoy"
]

X = fundamentals[feature_columns]
```

### Market Data Join
```python
# Load ticker mapping
tickers = pd.read_csv("data/sec_edgar/processed/cleaned/cik_ticker_map_cleaned.csv")

# Use for yfinance downloads
ticker_list = tickers["primary_ticker"].unique().tolist()
```

### Universe Filtering
```python
# Load issuer master for coverage filtering
issuers = pd.read_csv("data/sec_edgar/processed/cleaned/issuer_master_cleaned.csv")

# Filter companies with sufficient history
qualified = issuers[
    (issuers["has_companyfacts"] == "1") & 
    (issuers["n_fact_observations"] > 1000)
]
```

---

## Regeneration

To regenerate these cleaned datasets from source:

```bash
cd /path/to/fin-glassbox
source venv3.12.7/bin/activate

# Run cleaning pipeline
python data/sec_data_cleaning_pipeline.py --overwrite
```

**Note:** This requires the input files to exist in their original locations(which do not exist to save space. If it is necessary then Ibrahim Hussain has them in his local version of the repo but has put the cleaned versions in invisible directories on HDD):
- `data/sec_edgar/processed/issuer_master/cik_ticker_map.csv`
- `data/sec_edgar/processed/issuer_master/issuer_master.csv`
- `data/sec_edgar/processed/fundamentals/fundamentals_features.csv`

---

## Next Steps

With these cleaned datasets, the **Fundamental Data Family** is complete and ready for:

1. **Model training**: Feed into XGBoost/LightGBM/MLP for fundamental analysis
2. **Feature importance**: Analyze which ratios drive predictions
3. **Join with market data**: Add valuation metrics (P/E, P/B) once price data is available
4. **Point-in-time backtesting**: Safe to use with historical returns

---

## File Manifest

```
data/sec_edgar/processed/cleaned/
├── README.md                           # This documentation
├── cik_ticker_map_cleaned.csv          # CIK → ticker mapping (~5,500 rows)
├── issuer_master_cleaned.csv           # US issuer registry (~850,000 rows)
├── fundamentals_features_cleaned.csv   # Clean quarterly fundamentals (~3,655 rows)
└── cleaning_summary.json               # Pipeline statistics and metadata
```

---

*Last Updated: 2026-04-22*
*Pipeline Version: sec_data_cleaning_pipeline.py v1.0*
