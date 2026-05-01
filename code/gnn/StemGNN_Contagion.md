# StemGNN Contagion Risk Module 

**Project:** An Explainable Distributed Deep Learning Framework for Financial Risk Management  
**Module:** StemGNN Contagion Risk  
**Primary file:** `code/gnn/stemgnn_contagion.py`  
**Supporting files:** `stemgnn_base_model.py`, `stemgnn_forecast_dataloader.py`, `stemgnn_handler.py`, `stemgnn_utils.py`  
**Output root:** `outputs/results/StemGNN/`, `outputs/models/StemGNN/`, `outputs/codeResults/StemGNN/`  
**Status:** Implemented, trained for all chunks, packaged for integration  

---

## 1. Purpose of this document

This document is the final comprehensive documentation for the StemGNN component in the fin-glassbox project. It describes the purpose, design, data flow, model architecture, target construction, training protocol, HPO process, checkpointing, XAI outputs, run commands, outputs, and integration role of the StemGNN Contagion Risk Module.

StemGNN is one of the central graph-based risk modules in the project. It belongs to the **Risk Engine**, not the main technical encoder. Its job is not to forecast price directly. Its job is to estimate **contagion risk**: the probability that a stock will suffer an extreme negative movement that appears connected to market-wide or cross-asset spillover effects rather than only its own isolated behaviour.

---

## 2. Role in the overall architecture

```text
INPUTS
└── Time-Series Market Data
    └── returns_panel_wide.csv

RISK ENGINE
└── GNN Contagion Risk Module
    └── StemGNN

SYNTHESIS
└── Quantitative Analyst / Position Sizing / Fusion
```

StemGNN outputs are consumed by the Position Sizing Engine, Quantitative Analyst, future Fusion Engine, and XAI layer.

The module answers:

```text
If market stress propagates through related assets, how vulnerable is each stock over 5, 20, and 60 trading days?
```

---

## 3. Why StemGNN is used

StemGNN is appropriate for contagion risk because it learns both graph structure and temporal structure. It is stronger than a simple correlation matrix because contagion can be non-linear, lagged, multi-hop, and regime-dependent.

StemGNN includes:

- a latent correlation layer;
- graph Fourier transform operations;
- spectral-temporal processing;
- multi-hop propagation through a learned adjacency matrix;
- residual/stacked temporal modelling;
- learned node-level outputs for all stocks simultaneously.

In this project, StemGNN is adapted from a forecasting model into a risk-classification model.

---

## 4. Source files and responsibilities

### `stemgnn_base_model.py`

Contains the cleaned StemGNN base architecture:

- `GLU` — Gated Linear Unit used in the spectral sequence cell;
- `StockBlockLayer` — one spectral-temporal block;
- `Model` — StemGNN base model used by the contagion wrapper.

The base model expects:

```text
input:  [batch, time_step, node_count]
output: forecast tensor + attention matrix
```

### `stemgnn_contagion.py`

Primary project module implementing:

- `ContagionConfig`;
- `ContagionDataset`;
- `ContagionStemGNN`;
- train/validation loops;
- Optuna HPO;
- checkpointing and resume;
- smoke tests;
- prediction generation;
- XAI extraction;
- CLI commands.

### `stemgnn_forecast_dataloader.py`

Legacy forecasting data utilities. It supports the old forecasting baseline and contains z-score/min-max normalisation and `ForecastDataset`.

### `stemgnn_handler.py`

Legacy training/validation/testing helpers for the original forecasting task.

### `stemgnn_utils.py`

Metric helpers for the forecasting baseline:

- MAPE;
- masked MAPE;
- RMSE;
- MAE;
- A20 index;
- unified `evaluate()` helper.

---

## 5. Input data

### Primary input file

```text
data/yFinance/processed/returns_panel_wide.csv
```

Expected format:

```text
date,A,AA,AAL,...
2000-01-04,...
2000-01-05,...
...
```

The module loads this as a date-indexed matrix with tickers as columns.

Current project characteristics:

```text
2,500 tickers
approximately 6,285 trading days
2000-01-04 → 2024-12-27
finite ratio approximately 1.0
```

### Input tensor format

`ContagionStemGNN.forward()` expects:

```text
x: [batch, nodes, window]
```

For the default configuration:

```text
x: [batch, 2500, 30]
```

The wrapper internally permutes the tensor before passing it into the base StemGNN:

```text
[batch, nodes, window] → [batch, window, nodes]
```

---

## 6. Chronological chunking

| Chunk | Train | Validation | Test |
|---|---|---|---|
| Chunk 1 | 2000–2004 | 2005 | 2006 |
| Chunk 2 | 2007–2014 | 2015 | 2016 |
| Chunk 3 | 2017–2022 | 2023 | 2024 |

This prevents look-ahead leakage and keeps all evaluation chronological.

---

## 7. Contagion target construction

StemGNN predicts contagion-event probabilities at three horizons:

```text
5 trading days
20 trading days
60 trading days
```

For each stock and horizon, the binary label is:

```text
1 if:
    forward_h_return < historical 5th percentile threshold
    AND
    forward_h_return is more than 2 standard deviations below recent expected return
else:
    0
```

Key configuration values:

| Parameter | Default | Meaning |
|---|---:|---|
| `extreme_quantile` | `0.05` | Historical bottom 5% threshold |
| `excess_threshold_std` | `2.0` | Excess negative movement threshold |
| `history_days` | `504` | About 2 years of history for threshold estimation |
| `recent_days` | `60` | Recent window for expected return and volatility |
| `min_history_days` | `100` | Minimum history required to form target |

The target is intentionally strict. It tries to identify extreme downside events that are unusually negative relative to the stock’s own recent behaviour.

---

## 8. Dataset construction

`ContagionDataset` creates full-market sliding windows.

Each sample contains:

```text
x:      [nodes, window]
target: [nodes, horizons]
t:      sample time index
```

The dataset:

1. loads the returns matrix as float32;
2. replaces non-finite values with zero;
3. builds sliding windows;
4. builds multi-horizon binary contagion targets;
5. fits z-score normalisation on training windows only;
6. applies the same normalisation to validation/test windows;
7. computes positive class rates and positive-class weights.

Positive class weights are clipped by:

```python
max_pos_weight = 50.0
```

This matters because contagion events are rare.

---

## 9. Model architecture

### Base StemGNN

The base model includes:

```text
GRU latent correlation layer
learned attention adjacency
graph Laplacian
Chebyshev polynomial graph propagation
FFT/IRFFT spectral temporal processing
stacked StockBlockLayer modules
forecast-style output representation
```

The latent adjacency matrix is produced internally from the input sequence, so no external graph is required for the model to run.

### Contagion wrapper

The project wrapper is:

```python
class ContagionStemGNN(nn.Module)
```

It contains:

```text
self.stemgnn = StemGNNBase(...)
self.contagion_heads = ModuleList([...])
```

The wrapper takes the StemGNN output and applies three independent classification heads:

```text
Head 5d  → contagion_5d
Head 20d → contagion_20d
Head 60d → contagion_60d
```

Forward output dictionary:

```text
contagion_logits: [batch, nodes, horizons]
contagion_scores: [batch, nodes, horizons]
attention:         [nodes, nodes]
stemgnn_output:    [batch, nodes, window]
```

The `explain_forward()` method also returns an `xai` dictionary so explanations can be passed through the integrated system.

---

## 10. Main configuration

Important `ContagionConfig` defaults:

| Field | Default | Description |
|---|---:|---|
| `returns_path` | `data/yFinance/processed/returns_panel_wide.csv` | Return matrix path |
| `window_size` | `30` | Lookback window length |
| `multi_layer` | `13` | Spectral expansion depth |
| `stack_cnt` | `2` | Number of stacked StemGNN blocks |
| `dropout_rate` | `0.75` | Dropout in base and heads |
| `contagion_horizons` | `[5,20,60]` | Output horizons |
| `batch_size` | `8` | Default batch size |
| `epochs` | `100` | Final training epochs, with early stopping |
| `learning_rate` | `1e-3` | Default LR before HPO override |
| `optimizer` | `RMSProp` | Default optimiser |
| `early_stop_patience` | `20` | Stop after no validation improvement |
| `amp` | `True` | Mixed precision training |
| `num_workers` | `6` | DataLoader workers |
| `cpu_threads` | `6` | Torch CPU thread limit |
| `xai_sample_size` | `32` | XAI sample count |
| `enable_gnnexplainer` | `True` | Optional level-3 XAI |

---

## 11. HPO design

The module uses Optuna TPE hyperparameter optimisation before final training.

HPO safeguards include:

- SQLite database per chunk;
- finite failure penalty instead of returning `inf`/`nan` to Optuna;
- low worker count during HPO to avoid open-file exhaustion;
- architecture-safe checkpoints;
- immediate trial failure on non-finite losses;
- HPO window limits for manageable search time.

Important HPO controls:

| Field | Default | Description |
|---|---:|---|
| `hpo_trials` | `50` | Trials per chunk |
| `hpo_epochs` | `10` | Max epochs per trial |
| `hpo_num_workers` | `0` | Prevents DataLoader file descriptor buildup |
| `hpo_max_train_windows` | `500` | Training windows during HPO |
| `hpo_max_eval_windows` | `150` | Validation windows during HPO |

HPO results are stored under:

```text
outputs/codeResults/StemGNN/best_params_chunk{N}.json
outputs/codeResults/StemGNN/hpo_chunk{N}.db
```

---

## 12. Training process

The training function performs runtime setup, data construction, model construction, optimiser setup, optional checkpoint resume, epoch training, validation, early stopping, checkpoint saving, and summary saving.

Loss:

```text
BCEWithLogitsLoss
```

with optional positive class weighting.

Training uses:

- gradient clipping;
- AMP on CUDA;
- strict finite-loss checks;
- progress bars showing batch progress;
- resumable checkpoints;
- architecture compatibility checks.

---

## 13. Checkpointing and resume logic

Each chunk has:

```text
outputs/models/StemGNN/chunk{N}/
```

Important files:

```text
best_model.pt
latest_model.pt
final_model.pt
model_freezed/model.pt
model_unfreezed/model.pt
training_metrics.jsonl
training_summary.json
```

The checkpoint contains model state, optimiser state, scheduler state, AMP scaler state, epoch, best validation loss, config, node count, normalisation stats, and architecture signature.

The architecture signature prevents resuming incompatible checkpoints.

---

## 14. Prediction outputs

Prediction output path:

```text
outputs/results/StemGNN/contagion_scores_chunk{N}_{split}.csv
```

Output columns:

| Column | Description |
|---|---|
| `ticker` | Stock ticker |
| `contagion_5d` | Mean predicted contagion probability over the split for 5-day horizon |
| `contagion_20d` | Mean predicted contagion probability over the split for 20-day horizon |
| `contagion_60d` | Mean predicted contagion probability over the split for 60-day horizon |

The prediction file is ticker-level rather than ticker-date-level because the module aggregates predictions over all windows in the requested split.

---

## 15. XAI design

StemGNN provides three levels of XAI.

### Level 1 — Adjacency and top influencers

Saved files:

```text
outputs/results/StemGNN/xai/chunk{N}_{split}_adjacency.npy
outputs/results/StemGNN/xai/chunk{N}_{split}_avg_contagion.npy
outputs/results/StemGNN/xai/chunk{N}_{split}_top_influencers.json
```

### Level 2 — Gradient node/edge importance

Saved files:

```text
outputs/results/StemGNN/xai/chunk{N}_{split}_node_temporal_importance.npy
outputs/results/StemGNN/xai/chunk{N}_{split}_edge_importance.npy
```

### Level 3 — GNNExplainer-style approximation

Optional and activated with:

```text
--enable-gnnexplainer
```

Saved files include:

```text
outputs/results/StemGNN/xai/chunk{N}_{split}_gnnexplainer_mask_*.npy
outputs/results/StemGNN/xai/chunk{N}_{split}_gnnexplainer_top_edges.json
```

---

## 16. Integration contract

Downstream input contract:

```text
ticker
contagion_5d
contagion_20d
contagion_60d
```

Interpretation:

```text
higher value = higher contagion risk
```

In the approved Position Sizing Engine, contagion risk receives the highest risk weight:

```text
contagion weight = 0.25
```

---

## 17. CLI commands

### Compile

```bash
python -m py_compile code/gnn/stemgnn_base_model.py code/gnn/stemgnn_contagion.py code/gnn/stemgnn_forecast_dataloader.py code/gnn/stemgnn_handler.py code/gnn/stemgnn_utils.py
```

### Inspect

```bash
python code/gnn/stemgnn_contagion.py inspect --repo-root .
```

### Synthetic smoke test

```bash
python code/gnn/stemgnn_contagion.py smoke --repo-root . --device cuda --ticker-limit 32 --batch-size 2 --num-workers 0 --cpu-threads 6 --epochs 1 --max-train-windows 4 --max-eval-windows 2
```

### Real-data smoke test

```bash
python code/gnn/stemgnn_contagion.py smoke --repo-root . --device cuda --real --ticker-limit 64 --batch-size 2 --num-workers 0 --cpu-threads 6 --epochs 1 --max-train-windows 4 --max-eval-windows 2
```

### HPO

```bash
python code/gnn/stemgnn_contagion.py hpo --repo-root . --chunk 1 --trials 50 --device cuda --fresh
```

### Train with best HPO parameters

```bash
python code/gnn/stemgnn_contagion.py train-best --repo-root . --chunk 1 --device cuda --fresh
```

### Predict validation/test outputs

```bash
python code/gnn/stemgnn_contagion.py predict --repo-root . --chunk 1 --split val --device cuda --enable-gnnexplainer
python code/gnn/stemgnn_contagion.py predict --repo-root . --chunk 1 --split test --device cuda --enable-gnnexplainer
```

### Full chunk workflow

```bash
python code/gnn/stemgnn_contagion.py hpo --repo-root . --chunk 1 --trials 50 --device cuda --fresh && python code/gnn/stemgnn_contagion.py train-best --repo-root . --chunk 1 --device cuda --fresh && python code/gnn/stemgnn_contagion.py predict --repo-root . --chunk 1 --split val --device cuda --enable-gnnexplainer && python code/gnn/stemgnn_contagion.py predict --repo-root . --chunk 1 --split test --device cuda --enable-gnnexplainer
```

---

## 18. Known debugging lessons from implementation

### Open-file exhaustion during HPO

Earlier HPO runs failed with:

```text
OSError: [Errno 24] Too many open files
```

Fixes included raising the open-file limit, using `hpo_num_workers = 0`, explicitly shutting down loaders, and using separate HPO databases per chunk.

### Shape mismatch from incompatible checkpoints

A failed run had:

```text
ValueError: Expected time_step=60, got 30
```

The final code prevents this by storing and checking an architecture signature before resuming.

### Non-finite losses

Some high-learning-rate HPO trials produced `nan` loss. The final code fails such trials immediately and returns a large finite HPO penalty.

### cuDNN RNN backward issue during XAI

Gradient-based XAI through RNN/GRU layers can fail when cuDNN expects training mode. The final code disables cuDNN for small XAI backward passes while keeping normal training accelerated.

---

## 19. Validation and audit commands

### Verify packaged models and outputs

```bash
for c in 1 2 3; do echo "===== StemGNN chunk$c ====="; ls -lh outputs/models/StemGNN/chunk${c}/best_model.pt outputs/models/StemGNN/chunk${c}/final_model.pt outputs/models/StemGNN/chunk${c}/model_freezed/model.pt outputs/results/StemGNN/contagion_scores_chunk${c}_val.csv outputs/results/StemGNN/contagion_scores_chunk${c}_test.csv; done
```

### Audit prediction files

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path

for p in sorted(Path('outputs/results/StemGNN').glob('contagion_scores_chunk*_*.csv')):
    df = pd.read_csv(p)
    print('\n', p)
    print('shape:', df.shape)
    print('columns:', list(df.columns))
    print(df.describe().to_string())
PY
```

### Audit XAI files

```bash
find outputs/results/StemGNN/xai -type f | sort
```

---

## 20. Final status

The StemGNN module is complete for the risk engine.

Final expected completion state:

```text
outputs/models/StemGNN/chunk1/best_model.pt
outputs/models/StemGNN/chunk1/final_model.pt
outputs/models/StemGNN/chunk1/model_freezed/model.pt
outputs/results/StemGNN/contagion_scores_chunk1_val.csv
outputs/results/StemGNN/contagion_scores_chunk1_test.csv

outputs/models/StemGNN/chunk2/...
outputs/results/StemGNN/contagion_scores_chunk2_val.csv
outputs/results/StemGNN/contagion_scores_chunk2_test.csv

outputs/models/StemGNN/chunk3/...
outputs/results/StemGNN/contagion_scores_chunk3_val.csv
outputs/results/StemGNN/contagion_scores_chunk3_test.csv
```

---

**Document Version:** 1.4  
**Last Updated:** 1 May 2026

