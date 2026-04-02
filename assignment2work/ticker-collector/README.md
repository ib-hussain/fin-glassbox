# Ticker Collector

A comprehensive data collection pipeline that fetches historical financial time series data (such as cryptocurrencies and foreign exchange fiat currencies) from Yahoo Finance. It automatically collects, merges, aligns, and annotates the closing prices of various assets to construct high-quality multivariate datasets. These datasets are specifically tailored for training sequence-to-sequence deep learning models and Graph Neural Networks (GNNs).

## Features

- **Automated Data Extraction:** Downloads maximum available historical daily data via the Yahoo Finance API.
- **Multivariate Merging:** Aligns disparate individual time series into a unified tabular dataset using inner-joins or backward-filling logic.
- **Missing Data Imputation:** Specifically identifies missing calendar weekdays (especially useful for standard non-crypto markets) and intelligently forward-fills gaps.
- **Temporal Boundary Annotation:** Automatically calculates and marks the final day of each calendar week with `split_point` boolean flags to aid in deep learning temporal windowing.
- **Built-in Visualizations:** Includes visualization tools to plot standardized price movements and categorize macro market phases (e.g., Bull vs. Bear).

## File Structure & Documentation

The codebase is modularized into several specific functional components:

### Pipeline Execution
*   **`run_pipeline.py`**: The primary execution script for the Cryptocurrency data pipeline. It sequentially triggers data collection, merging (with backfilling to recent 2190 days), and week-marking, while logging the time taken for each stage.
*   **`run_pipeline_for_fx.py`**: The main execution script for the Foreign Exchange (FX) data pipeline. Includes an additional step (`time_series_weekdays_filler.py`) to handle and fill missing weekday data specific to standard fiat trading schedules.

### Core Modules
*   **`time_series_collector.py`**: The ingestion module. It connects to Yahoo Finance to download the historical daily close prices for a provided list of tickers and stores each asset as an individual CSV file.
*   **`time_series_merger.py`**: The aggregation module. It merges the individual CSV files into a single, comprehensive dataset matrix. It supports aligning assets either by length-matching to the shortest asset duration (inner join) or by backward-filling the most recent `N` days.
*   **`week_marker.py`**: The temporal annotation module. It processes a merged dataset to append a boolean `split_point` column, which evaluates to `True` for the last recorded day of each calendar week.
*   **`time_series_weekdays_filler.py`**: The data cleaning module (primarily for FX/Stocks). It identifies gaps in trading day sequences (e.g., due to market holidays) and forward-fills (`ffill`) the missing values to maintain a continuous, uninterrupted weekday cadence.
*   **`tickers.py`**: A centralized configuration module defining the standardized Python lists of ticker symbols used in the pipelines, including `crypto_tickers` (top cap coins) and `fx_tickers` (top fiat currencies).

### Utilities
*   **`crypto_bull_bear_viewer.py`**: A visualization utility that generates a Matplotlib/Seaborn plot (`cryptoBullBear.svg`) of standardized closing prices for selected crypto assets (ADA, BTC, ETH, XTZ). It visually divides the time scale into approximate "Bull" and "Bear" market phases.

## Usage

Ensure you have the necessary dependencies installed (e.g., `pandas`, `yfinance`, `matplotlib`, `tqdm`, `scikit-learn`). 

To generate the cryptocurrency dataset, execute:
```bash
python run_pipeline.py
```

To generate the foreign exchange dataset, execute:
```bash
python run_pipeline_for_fx.py
```

The resulting output files will be automatically placed in the `assignment2work/datasetsOut/` directory.
