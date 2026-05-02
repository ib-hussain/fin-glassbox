# Technical Analyst
## 1. Module identity

**File:** `code/analysts/technical_analyst.py`  
**Role:** technical/market sequence specialist  
**Branch:** quantitative branch  
**Upstream dependency:** Temporal Encoder embeddings  
**Downstream consumers:** Position Sizing Engine, Quantitative Analyst, Fusion Engine through the quantitative branch

The Technical Analyst consumes Temporal Encoder outputs and produces:

```text
trend_score
momentum_score
timing_confidence
```

It is a downstream analyst. It does not replace the Shared Temporal Attention Encoder. The Temporal Encoder learns the main temporal representation; the Technical Analyst reads sequences of those embeddings to produce task-specific technical scores.

---

## 2. Architectural position

```text
Time-Series Market Data
        ↓
Shared Temporal Attention Encoder
        ↓
Technical Analyst
        ↓
Position Sizing Engine
        ↓
Quantitative Analyst
        ↓
Fusion Engine
        ↓
Final Trade Approver
```

The Technical Analyst contributes a directional and timing signal but does not make final trading decisions.

---

## 3. Input contract

Required Temporal Encoder outputs:

```text
outputs/embeddings/TemporalEncoder/chunk{N}_{split}_embeddings.npy
outputs/embeddings/TemporalEncoder/chunk{N}_{split}_manifest.csv
```

The manifest must contain:

```text
ticker
date
```

Additional market files used for target construction:

```text
data/yFinance/processed/returns_panel_wide.csv
data/yFinance/processed/features_temporal.csv
```

The model builds second-level sequences:

```text
input shape = (batch, analyst_seq_len=30, embedding_dim=256)
```

---

## 4. Chronological chunks

| Chunk | Train | Validation | Test |
|---:|---|---|---|
| 1 | 2000–2004 | 2005 | 2006 |
| 2 | 2007–2014 | 2015 | 2016 |
| 3 | 2017–2022 | 2023 | 2024 |

---

## 5. Target construction

For each embedding date `t`, the module constructs targets using future returns and technical indicators.

### 5.1 Trend target

```text
20d forward return > +0.5% → 1.0
20d forward return < -0.5% → 0.0
otherwise                  → 0.5
```

### 5.2 Momentum target

```text
momentum = abs(5d forward return) / (5d volatility + eps)
momentum = clamp(momentum, 0, 1)
```

### 5.3 Timing target

```text
RSI < 30 and price_pos > 0.5 → 1.0
RSI > 70 and price_pos < 0.5 → 0.0
otherwise                    → 0.5 + 0.5 × normalised MACD histogram
```

The implementation uses robust MACD normalisation with percentile clipping, median, and MAD. This prevents large MACD outliers from collapsing timing targets around 0.5.

---

## 6. Model architecture

Primary model class:

```text
TechnicalAnalystModel
```

Architecture:

```text
Temporal embedding sequence
        ↓
BiLSTM
        ↓
additive attention pooling
        ↓
LayerNorm + Dropout + MLP
        ↓
Sigmoid outputs:
    trend_score
    momentum_score
    timing_confidence
```

Default configuration:

```text
input_dim = 256
analyst_seq_len = 30
lstm_hidden = 64
lstm_layers = 1
bidirectional = True
dropout = 0.20
attention_dim = 64
batch_size = 512
epochs = 50
learning_rate = 1e-3
early_stop_patience = 10
```

Using a BiLSTM here is architecturally acceptable because it is not the main technical encoder. It is a compact specialist reading already-learned temporal embeddings.

---

## 7. Output contract

Model outputs:

```text
outputs/models/TechnicalAnalyst/chunk{N}/best_model.pt
outputs/models/TechnicalAnalyst/chunk{N}/latest_model.pt
outputs/models/TechnicalAnalyst/chunk{N}/final_model.pt
outputs/models/TechnicalAnalyst/chunk{N}/target_stats.json
outputs/models/TechnicalAnalyst/chunk{N}/training_history.csv
```

Prediction outputs:

```text
outputs/results/TechnicalAnalyst/predictions_chunk{N}_{split}.csv
```

Important prediction columns:

```text
ticker
date
trend_score
momentum_score
timing_confidence
target_trend
target_momentum
target_timing
```

XAI outputs:

```text
outputs/results/TechnicalAnalyst/xai/chunk{N}_{split}_attention.json
outputs/results/TechnicalAnalyst/xai/chunk{N}_{split}_embedding_dim_importance.csv
outputs/results/TechnicalAnalyst/xai/chunk{N}_{split}_timestep_importance.csv
outputs/results/TechnicalAnalyst/xai/chunk{N}_{split}_counterfactuals.json
outputs/results/TechnicalAnalyst/xai/chunk{N}_{split}_xai_summary.json
```

---

## 8. Loss and training

The model predicts three continuous values in `[0, 1]`. Training uses supervised regression over the three technical targets. HPO uses Optuna/TPE. The module supports:

```text
inspect
smoke
hpo
train-best
train-best-all
predict
```

Runtime features include:

```text
CUDA/CPU support
mixed precision
gradient clipping
early stopping
DataLoader workers
CPU thread control
fresh HPO starts
checkpoint saving
```

---

## 9. XAI design

The Technical Analyst implements three explanation levels:

### Level 1 — Attention weights

Attention weights identify which of the 30 embedding timesteps mattered most.

### Level 2 — Gradient importance

Gradient attribution identifies important embedding dimensions and timesteps.

### Level 3 — Counterfactuals

Counterfactual tests perturb recent sequence information and measure changes in trend, momentum, and timing outputs.

This gives the technical signal a clear explanation path.

---

## 10. CLI commands

Compile:

```bash
python -m py_compile code/analysts/technical_analyst.py
```

Smoke:

```bash
python code/analysts/technical_analyst.py smoke --repo-root . --device cuda
```

Inspect:

```bash
python code/analysts/technical_analyst.py inspect --repo-root . --device cuda
```

HPO:

```bash
python code/analysts/technical_analyst.py hpo --repo-root . --chunk 1 --trials 40 --device cuda --fresh
```

Train best:

```bash
python code/analysts/technical_analyst.py train-best --repo-root . --chunk 1 --device cuda --fresh
```

Predict:

```bash
python code/analysts/technical_analyst.py predict --repo-root . --chunk 1 --split test --device cuda
```

Full chunk rerun:

```bash
python code/analysts/technical_analyst.py hpo --repo-root . --chunk 1 --trials 40 --device cuda --fresh && python code/analysts/technical_analyst.py train-best --repo-root . --chunk 1 --device cuda --fresh && python code/analysts/technical_analyst.py predict --repo-root . --chunk 1 --split train --device cuda && python code/analysts/technical_analyst.py predict --repo-root . --chunk 1 --split val --device cuda && python code/analysts/technical_analyst.py predict --repo-root . --chunk 1 --split test --device cuda
```

---

## 11. Integration notes

Rerun this module whenever Temporal Encoder embeddings change. Its outputs feed Position Sizing and Quantitative Analyst training. Downstream modules expect stable fields such as:

```text
trend_score
momentum_score
timing_confidence
technical_confidence
```

---
The Technical Analyst is a downstream technical specialist that converts Temporal Encoder embeddings into interpretable trend, momentum, and timing scores. It uses chronological splits, rule-derived future-return targets, a compact BiLSTM with attention pooling, robust target normalisation, and three-level XAI.
