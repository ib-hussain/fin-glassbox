# Cleaned SEC EDGAR Datasets

## Overview

This directory contains the cleaned and production-ready versions of the SEC EDGAR processed datasets. These files are the result of a multi-stage data engineering pipeline that transformed raw SEC bulk archives into normalized, analysis-ready tabular data suitable for the **Explainable Multimodal Neural Framework for Financial Risk Management**.

**All datasets are filtered to U.S. public companies listed on NYSE or Nasdaq.**

---

## Pipeline Summary

```
Raw SEC Bulk Archives (ZIP)
    ↓
sec_edgar_download.py (raw acquisition)
    ↓
sec_submissions_pipeline.py + sec_companyfacts_pipeline.py (flattening)
    ↓
sec_issuer_master.py + sec_core_fundamentals.py + sec_fundamentals_features.py (feature engineering)
    ↓
sec_data_cleaning_pipeline.py (cleaning - THIS STEP)
    ↓
Cleaned datasets (this directory)
```

---

## Input Datasets (Before Cleaning)

| File | Source | Original Rows | Issues Addressed |
|------|--------|---------------|------------------|
| `cik_ticker_map.csv` | `issuer_master/` | 5,644 | Missing exchange values (106), non-NYSE/Nasdaq exchanges (1,110) |
| `issuer_master.csv` | `issuer_master/` | 973,279 | Non-US issuers (121,710), invalid exchanges, redundant columns, missing tickers |
| `fundamentals_features.csv` | `fundamentals/` | 7,310 | Duplicate rows (1,050), 100% null columns (4), missing ticker/entity_name (4,689), extreme outliers (4,722 values) |

---

## Output Datasets (After Cleaning)

### 1. `cik_ticker_map_cleaned.csv`
**4,428 rows** | CIK to ticker mapping for U.S. public companies

| Column | Description |
|--------|-------------|
| `cik` | SEC Central Index Key (unique identifier) |
| `padded_cik` | Zero-padded 10-digit CIK |
| `primary_ticker` | Trading symbol |
| `entity_name` | Legal company name |
| `primary_exchange` | **Nasdaq** or **NYSE** only |

**Transformations applied:**
- Removed 106 rows with missing exchange
- Removed 1,110 rows with OTC/other exchanges
- Result: Clean universe of 4,428 NYSE/Nasdaq companies

---

### 2. `issuer_master_cleaned.csv`
**850,459 rows** | Master registry of all U.S. SEC filers

| Column | Description |
|--------|-------------|
| `cik` | SEC Central Index Key |
| `entity_name` | Legal company name |
| `primary_ticker` | Trading symbol (if available) |
| `primary_exchange` | Nasdaq, NYSE, or empty |
| `sic` | Standard Industrial Classification code |
| `sic_description` | Industry description |
| `fiscal_year_end` | Month-day of fiscal year end (MMDD) |
| `business_city` | City of business address |
| `has_submissions` | 1 if filings metadata available |
| `has_companyfacts` | 1 if XBRL facts available |
| `earliest_filing_sub` | First filing date in submissions |
| `latest_filing_sub` | Most recent filing date |
| `earliest_filed_facts` | First XBRL fact date |
| `latest_filed_facts` | Most recent XBRL fact date |
| `n_fact_observations` | Total XBRL facts available |

**Transformations applied:**
- Filtered to U.S. issuers only (`is_us_issuer == 1`): removed 121,710 non-US entities
- Removed OTC/other exchanges: 1,110 rows
- Dropped 9 redundant columns: `is_us_issuer`, `all_tickers`, `padded_cik`, `entity_type`, `state_of_incorporation`, `state_of_incorporation_desc`, `mailing_city`, `mailing_state`, `business_state`
- **Kept** `business_city` for geographic analysis

---

### 3. `fundamentals_features_cleaned.csv`
**6,260 rows** | Point-in-time quarterly fundamentals with derived ratios

This is the **primary dataset** for the Fundamental Analyst module.

#### Identifiers
| Column | Description |
|--------|-------------|
| `cik` | SEC Central Index Key |
| `ticker` | Trading symbol (filled from mapping) |
| `entity_name` | Company name (filled from mapping) |
| `fiscal_year` | Fiscal year |
| `fiscal_period` | Q1, Q2, Q3, Q4, or FY |
| `filing_date` | SEC filing date (point-in-time anchor) |
| `period_end_date` | End date of fiscal period |
| `form_type` | 10-K, 10-Q, or amendments |
| `accession` | SEC accession number |

#### Raw Financials (34 columns)
Revenue, cost_of_revenue, gross_profit, operating_expenses, operating_income, net_income, eps_basic, eps_diluted, shares_basic, total_assets, current_assets, cash_and_equivalents, ppe_net, total_liabilities, current_liabilities, long_term_debt, shareholders_equity, retained_earnings, operating_cash_flow, investing_cash_flow, financing_cash_flow, capex, free_cash_flow

#### Derived Features (28 columns)
| Category | Features |
|----------|----------|
| **Profitability** | gross_margin, operating_margin, net_margin, opex_to_revenue, cogs_to_revenue |
| **Returns** | roa, roe |
| **Leverage** | debt_to_equity, debt_to_assets, current_ratio, quick_ratio, cash_ratio |
| **Efficiency** | asset_turnover, ppe_turnover |
| **Cash Flow** | ocf_to_revenue, fcf_to_revenue, capex_to_revenue, fcf_to_net_income, ocf_to_net_income |
| **Per Share** | revenue_per_share, book_value_per_share, ocf_per_share, fcf_per_share |
| **Quality** | accruals_to_assets |
| **Growth** | revenue_growth_yoy, revenue_growth_qoq, net_income_growth_yoy, net_income_growth_qoq, operating_income_growth_yoy, operating_income_growth_qoq, eps_basic_growth_yoy, eps_basic_growth_qoq, total_assets_growth_yoy, total_assets_growth_qoq |

**Transformations applied:**
- **Deduplication:** Removed 1,050 duplicate rows (each period now appears once)
- **Null columns dropped:** `inventory`, `goodwill`, `intangible_assets`, `short_term_debt` (100% missing)
- **Missing values filled:** 4,689 tickers and entity names filled using CIK mapping
- **Outlier capping:** 4,722 values capped at ±1000% (±10.0) or reasonable domain bounds
- **Point-in-time strict:** All values are "as-first-reported" (earliest filing per period)

---

## Cleaning Statistics Summary

| Metric | Value |
|--------|-------|
| **Total runtime** | 41.5 seconds |
| **cik_ticker_map rows** | 4,428 (from 5,644) |
| **issuer_master rows** | 850,459 (from 973,279) |
| **fundamentals_features rows** | 6,260 (from 7,310) |
| **Duplicates removed** | 1,050 |
| **Missing tickers/names filled** | 4,689 |
| **Outlier values capped** | 4,722 |
| **Columns dropped** | 13 across all files |

---

## Usage Notes

### For Fundamental Analyst Module
Use `fundamentals_features_cleaned.csv` as the primary input. All features are:
- **Normalized ratios** (comparable across companies of any size)
- **Point-in-time correct** (no lookahead bias from restatements)
- **Outlier-capped** (extreme values bounded at ±1000%)

### For Universe Definition
Use `cik_ticker_map_cleaned.csv` to:
- Join with market data (yfinance) using `primary_ticker`
- Filter to valid U.S. public companies
- Map SEC CIKs to trading symbols

### For Company Metadata
Use `issuer_master_cleaned.csv` for:
- Industry classification (SIC codes)
- Data coverage assessment (has_submissions, has_companyfacts)
- Filing date ranges

---

## Related Files

| File | Purpose |
|------|---------|
| `cleaning_summary.json` | Full statistics and parameters from cleaning run |
| `../companyfacts/companyfacts_flat/` | Raw XBRL facts (72 partitioned CSVs, 121.9M rows) |
| `../submissions/submissions_flat/` | Raw filing metadata (partitioned CSVs) |

---

## Regeneration

To regenerate these cleaned files:

```bash
cd /path/to/fin-glassbox
source venv/bin/activate
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
