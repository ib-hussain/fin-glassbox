# LSTM Baselines — `assignment2work/LSTM`

Multivariate **LSTM**, **VAR**, and **ARIMA** forecasting scripts used as classical baselines alongside graph-temporal models (FourierGNN, MTGNN, StemGNN) in the study:

> *"Financial asset price prediction with graph neural network-based temporal deep learning models"*

The models are trained on weekly rolling windows over 104 evaluation weeks and produce one prediction per week saved as a `.npy` file — the format consumed by `paper-metric-reporter`.

---

## Directory Layout

```
LSTM/
└── lstm_predictor/
    ├── main.py          # LSTM model with Hyperopt tuning
    ├── main_var.py      # VAR (Vector Autoregression) baseline
    └── main_arima.py    # Per-asset ARIMA with optional Hyperopt order search
```

---

## Model Descriptions

### `main.py` — Hyperopt-Tuned LSTM
A multi-layer LSTM (`LSTMModel`) with a linear prediction head. Hyperparameter search over learning rate, dropout, number of layers, hidden size, batch size, and sequence length is run via `hyperopt` (TPE algorithm) for `Config.hpo_max_evals` trials. On each of the 104 rolling weeks, it saves a `(1, n_assets)` prediction array as:

```
prediction/<timestamp>/weeks/<week_index+1>.npy
```

### `main_var.py` — VAR Baseline
Fits a Vector Autoregression model on the training+validation slice. Automatically selects lag order using AIC. Saves one step-ahead forecasts as:

```
var_prediction-crypto/<timestamp>/weeks/<week_index+1>.npy
```

### `main_arima.py` — Per-Asset ARIMA Baseline
Fits separate `ARIMA(p,d,q)` models per asset column. Optionally performs Hyperopt search for best `(p,d,q)` order. Saves predictions in the same `.npy` format.

---

## Configuration (No CLI Arguments)

All three entry points use **no** `sys.argv` / argparse. Edit paths and hyperparameters in each file's `Config` class **before** running:

### `main.py` Config

| Attribute | Purpose | Example Value |
|---|---|---|
| `prices_path` | Marked price CSV with `Date`, `split_point`, asset columns | `../ticker-collector/out/crypto/daily_20_2190_marked.csv` |
| `predictions_base_path` | Root dir for `.npy` prediction outputs | `prediction` |
| `num_weeks_to_train` | Number of rolling weeks in the outer loop | `104` |
| `total_weeks` | Total weeks used for truncation math | `104` |
| `horizon` | Return horizon in trading days | `7` |
| `training_ratio` | Fraction of rows used for training | `0.6` |
| `validation_ratio` | Fraction of rows used for validation | `0.2` |
| `hpo_max_evals` | Hyperopt trials per week | `100` |
| `num_epochs_to_run` | Max epochs per trial | `500` |
| `early_stopping_patience` | Stop after N non-improving validation checks | `100` |

**Default data path (LSTM — crypto):**
```
../ticker-collector/out/crypto/daily_20_2190_marked.csv
```
<!-- #TODO: Check path -->

### `main_arima.py` Config

**Default data path (ARIMA — FX):**
```
../ticker-collector/out/fx/daily_10_4506_marked.csv
```
<!-- #TODO: Check path -->

### `main_var.py` Config

**Default data path (VAR — crypto):**
```
../ticker-collector/out/crypto/daily_20_2190_marked.csv
```
<!-- #TODO: Check path -->

---

## Data Format

Input CSVs must have:
- `Date` column (string, e.g. `2022-01-07`)
- `split_point` column (boolean/int — marks which rows are weekly evaluation boundaries)
- One column per asset (price level, not return)

The code trims to week `w` using `split_point` markers and converts prices to `horizon`-step simple return ratios.

---

## Running

```bash
cd lstm_predictor
python main.py        # LSTM
python main_var.py    # VAR
python main_arima.py  # ARIMA
```

Run from `lstm_predictor/` so relative paths (e.g. `../ticker-collector/out/...`) resolve correctly.

---

## Outputs

Each script saves weekly predictions to a folder. Example for LSTM:

```
lstm_predictor/prediction/<timestamp>/weeks/1.npy
lstm_predictor/prediction/<timestamp>/weeks/2.npy
...
lstm_predictor/prediction/<timestamp>/weeks/104.npy
```

Copy the `<timestamp>` folder to `experiment-results/crypto/lstm-results/` or `experiment-results/fx/lstm-results/` and update `config.py` in `paper-metric-reporter` accordingly.

---

## Debug Logging

Each script emits `[Debug_Output]:` prefixed lines tracing:
- Data path and shape loaded
- Week index and truncation math
- Splits (train/val/test boundaries)
- Model calls and HPO trials

---

## Paper Reference

Section 4.1 of the paper describes the LSTM baseline architecture. The rolling evaluation methodology (104 weeks, weekly rebalance) is described in Section 3.3. Metrics used: MAPE, MAE, RMSE, A20 Index.
