# Experiment Results — `assignment2work/experiment-results`

This directory is the **canonical drop zone** for numeric model outputs consumed by `paper-metric-reporter/reporter/config.py`.

There are **no Python modules** here by design. Trained models (FourierGNN, LSTM, MTGNN, StemGNN, ARIMA, VAR) write weekly prediction arrays here; the reporter reads them to evaluate portfolio performance and compute paper metrics.

---

## Expected Layout

```
experiment-results/
├── crypto/
│   ├── arima-results/
│   │   └── <timestamp>/          # e.g. 20250321/
│   │       └── 1.npy … 104.npy
│   ├── fouriergnn-results/
│   │   └── <timestamp>/          # e.g. 20250318/
│   │       └── 1.npy … 104.npy
│   ├── lstm-results/
│   │   └── <timestamp>/          # e.g. 202302221651/
│   │       └── weeks/
│   │           └── 1.npy … 104.npy
│   ├── mtgnn-results/
│   │   └── <horizon>/            # e.g. 7/, 21/, 35/ …
│   │       └── 1.npy … 104.npy
│   ├── stemgnn-results/
│   │   └── <timestamp>/
│   │       └── 1.npy … 104.npy
│   └── var-results/
│       └── <timestamp>/
│           └── 1.npy … 104.npy
└── fx/
    ├── arima-results/
    │   └── <timestamp>/
    ├── fouriergnn-results/
    │   └── <timestamp>/
    ├── lstm-results/
    │   └── <timestamp>/
    │       └── weeks/
    ├── mtgnn-results/
    │   └── <timestamp>/
    ├── stemgnn-results/
    │   └── <timestamp>/
    │       └── __fx_daily_simple_returns/
    └── var-results/
        └── <timestamp>/
```

---

## File Format

Each `.npy` file contains a `(1, n_assets)` NumPy array representing one week's prediction:
- **FX**: shape `(1, 10)` — 10 currency pairs
- **Crypto**: shape `(1, 20)` — 20 cryptocurrency assets
- Values are **simple return factors** (e.g. `1.02` = +2% return), except for some LSTM runs tagged `"log-return"` which use `np.exp()` on load.

Files are indexed `1.npy` through `104.npy` (1-indexed, matching `week + 1`).

---

## Paths Referenced in `config.py`

All paths below are relative to `paper-metric-reporter/reporter/`. Update `config.py` whenever you run a new experiment and move outputs here.

### FX Results

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

### Crypto Results

```python
DataPath(f"../experiment-results/crypto/stemgnn-results/202312090602", "StemGNN")
```
<!-- #TODO: Check path -->

```python
DataPath(f"../experiment-results/crypto/mtgnn-results/7", "MTGNN:7")
```
<!-- #TODO: Check path -->

```python
DataPath(f"../experiment-results/crypto/lstm-results/202302221651", "LSTM:202302221651")
```
<!-- #TODO: Check path -->

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

---

## How Model Outputs Get Here

Each model saves predictions in its working directory. Move to this folder and update `config.py`:

| Model | Source directory | Save command used |
|---|---|---|
| **LSTM** | `lstm_predictor/prediction/<timestamp>/weeks/` | `np.save(f"{dir_path}/{week+1}", predictions)` |
| **VAR** | `lstm_predictor/var_prediction-crypto/<timestamp>/weeks/` | same |
| **ARIMA** | `lstm_predictor/arima_prediction-crypto/<timestamp>/weeks/` | same |
| **FourierGNN** | Controlled by `datasets_FourierGNN_output_path` env var | `np.save(f".../{data}/{week-1}.npy", preds)` |
| **MTGNN** | Varies per experiment script | 1.npy … N.npy in numbered folders |
| **StemGNN** | Varies per experiment script | same |

---

## Adding New Experiment Results

1. Copy the run's `weeks/` (or equivalent) folder to the appropriate `experiment-results/<dataset>/<model>-results/<timestamp>/`
2. Open `paper-metric-reporter/reporter/config.py`
3. Add or update the matching `DataPath(...)` entry in `FXPaths` or `CryptoPaths`
4. Add a `<!-- #TODO: Check path -->` comment on the line after if documenting in a README

---

## Note on Empty Directories

If this folder is empty after a fresh clone, populate it from training machine backups or re-run the model pipelines. The reporter will exit with a `FileNotFoundError` (from `np.load`) if any `.npy` file listed in `config.py` is missing.
