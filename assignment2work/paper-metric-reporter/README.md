# Paper Metric Reporter — `assignment2work/paper-metric-reporter`

Post-processing and analysis pipeline that ingests **saved model predictions** (`.npy` files from FourierGNN, LSTM, MTGNN, StemGNN, ARIMA, VAR) and **actual marked price CSVs**, then computes portfolio performance, statistical tests, and prediction accuracy metrics as reported in:

> *"Financial asset price prediction with graph neural network-based temporal deep learning models"*

---

## System Arguments (CLI)

Run from the `reporter/` package directory so `import config` resolves correctly:

```bash
cd reporter
python main.py <dataset>
```

| Argument | Position | Purpose | Allowed Values | Example |
|---|---|---|---|---|
| `dataset` | `sys.argv[1]` | Selects which price CSV and model results to load, and whether to split bull/bear | `fx` or `crypto` | `python main.py crypto` |

---

## Directory Layout

```
paper-metric-reporter/
├── README.md
├── requirements.txt
├── risk-free-rate-calculation.xlsx
└── reporter/
    ├── main.py                                    # Entry point; requires argv[1]
    ├── config.py                                  # All paths and portfolio parameters (EDIT HERE)
    ├── models.py                                  # Data containers (Data, DataPath, PortfolioResult …)
    ├── table_to_variability_figure.py             # Standalone bar-chart generator (hard-coded metrics)
    ├── independent_t_testing_for_error_measures.py # Offline statistical significance tests
    └── helpers/
        ├── __init__.py
        ├── reader.py                              # Load CSVs and .npy prediction files
        ├── portfolio.py                           # Long/short simulation, transaction costs
        ├── portfolio_group.py                     # Random, All-asset, and Best-model portfolio factories
        ├── report_transformer.py                  # Price reconstruction, return transforms, trading labels
        ├── reporter.py                            # Print/plot facades (tables, heatmaps, capital curves)
        └── stats.py                               # MAPE, MBD, CFE, cross-sectional standardization
```

---

## Configuration — Edit `reporter/config.py` Before Running

### FX Paths

**Actual price CSV:**
```python
actual = DataPath("../ticker-collector/out/fx/daily_10_4506_marked.csv", "Actual")
```
<!-- #TODO: Check path -->

**Model prediction directories (each must contain `1.npy` … `104.npy`):**
```python
DataPath(f"../experiment-results/fx/mtgnn-results/202312090808", "MTGNN")
```
<!-- #TODO: Check path -->
```python
DataPath(f"../experiment-results/fx/stemgnn-results/202312080300/__fx_daily_simple_returns", "StemGNN")
```
<!-- #TODO: Check path -->
```python
DataPath(f"../experiment-results/fx/lstm-results/202312062100/weeks", "LSTM")
```
<!-- #TODO: Check path -->
```python
DataPath(f"../experiment-results/fx/fouriergnn-results/20250319", "FourierGNN")
```
<!-- #TODO: Check path -->
```python
DataPath(f"../experiment-results/fx/arima-results/20250321", "ARIMA")
```
<!-- #TODO: Check path -->
```python
DataPath(f"../experiment-results/fx/var-results/202503221825", "VAR")
```
<!-- #TODO: Check path -->

### Crypto Paths

**Actual price CSV:**
```python
actual = DataPath("../ticker-collector/out/crypto/daily_20_2190_marked.csv", "Actual")
```
<!-- #TODO: Check path -->

**Model prediction directories:**
```python
DataPath(f"../experiment-results/crypto/fouriergnn-results/20250318", "FourierGNN")
```
<!-- #TODO: Check path -->
```python
DataPath(f"../experiment-results/crypto/arima-results/20250321", "ARIMA")
```
<!-- #TODO: Check path -->
```python
DataPath(f"../experiment-results/crypto/var-results/202503221852", "VAR")
```
<!-- #TODO: Check path -->

### Portfolio & Reporting Parameters

| Parameter | Purpose | Example / Default |
|---|---|---|
| `initial_capital` | Starting cash ($) for simulated portfolios | `100` |
| `num_weeks` | Rolling weeks of evaluation (must match `.npy` file count) | `104` |
| `random_trials` | Monte Carlo samples for random long/short baseline | `1000` |
| `hold_threshold` | Relative price gap below which `get_trading_decisions` labels a "Hold" | `0.01` (1%) |
| `transaction_cost` | One-way transaction cost per rebalance | `0.005` (0.5%) |
| `quick` | If True, shrinks `random_trials` and grid for fast smoke tests | `False` |

### OutputFlags (toggle optional plots)

```python
OutputFlags.plot_standardized_prices = True          # z-scored price series
OutputFlags.plot_pairwise_returns = False             # every pairwise return combination
OutputFlags.plot_predicted_and_actual_portfolio_values = False  # per-model capital curves
```

---

## Running

```bash
# From the reporter/ directory:
cd reporter
python main.py fx       # FX experiment
python main.py crypto   # Cryptocurrency experiment

# Statistical significance of model errors (standalone, no argv):
python independent_t_testing_for_error_measures.py

# Bar charts from hard-coded paper metrics (no argv):
python table_to_variability_figure.py
```

---

## What the Reporter Computes

1. **Correlation matrix** of actual prices
2. **Standardized price** comparison plots (actual vs. predicted)
3. **Portfolio simulation** — equal-weight long/short baskets for each model, plus random and all-asset baselines
4. **Best long/short combination** selected on first 10 weeks, evaluated on all 104
5. **Capital curves** — final portfolio values, Sharpe ratio, drawdown, annualized return
6. **Prediction accuracy** — MAPE, MBD, CFE on reconstructed prices and returns
7. **Direction metrics** — Buy/Hold/Sell accuracy, precision, recall, F1
8. **Statistical significance** — paired t-tests across model capital/return series
9. **Asset selection accuracy** — overlap between model picks and oracle picks

---

## Tests

```bash
cd reporter
python -m pytest tests/
```

---

## Debug Logging

All major functions emit `[Debug_Output]:` prefixed lines showing function name, argument values, array shapes, and intermediate computations.

---

## Paper Reference

Metrics and simulation methodology are described in Sections 3.3–3.5 and Tables 2–5 of the paper. Risk-free rates used for Sharpe ratio calculations are documented in `risk-free-rate-calculation.xlsx`.
