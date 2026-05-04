# Hyperparameter Configuration

## Document Purpose

This document contains **all hyperparameter configurations** for every model in the Explainable Multimodal Neural Framework for Financial Risk Management. These values are based on:

- Baseline reproduction results (MTGNN, StemGNN, FourierGNN)
- Industry standards for financial time series
- Anti-overfitting requirements discussed and approved
- Computational constraints (TPE Bayesian optimization, limited GPU resources)

**Version:** 1.0  
**Status:** Ready for Implementation

---

## Table of Contents

1. [Global Training Configuration](#1-global-training-configuration)
2. [Encoder Layer](#2-encoder-layer)
3. [Analyst Layer](#3-analyst-layer)
4. [Risk Engine](#4-risk-engine)
5. [Fusion Layer](#5-fusion-layer)
6. [Hyperparameter Search Spaces](#6-hyperparameter-search-spaces)
7. [Learning Rate Schedules](#7-learning-rate-schedules)
8. [Regularization Summary](#8-regularization-summary)
9. [YAML Configuration File](#9-yaml-configuration-file)

---

## 1. Global Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `device` | `"cuda"` if available else `"cpu"` | Compute device |
| `seed` | 42 | Random seed for reproducibility |
| `dtype` | `float32` | Default tensor type |
| `num_workers` | 4 | DataLoader workers |
| `pin_memory` | `True` | Faster GPU transfer |
| `checkpoint_dir` | `"./checkpoints/"` | Model save location |
| `log_dir` | `"./logs/"` | TensorBoard logs |
| `mixed_precision` | `True` | FP16 training for speed |

### Training Chunks (Chronological)

```yaml
training_chunks:
  chunk_1:
    train_years: [2000, 2001, 2002, 2003, 2004]
    val_years: [2005]
    test_years: [2006]
    description: "Dot-com recovery period"
  
  chunk_2:
    train_years: [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014]
    val_years: [2015]
    test_years: [2016]
    description: "Financial crisis + recovery"
  
  chunk_3:
    train_years: [2017, 2018, 2019, 2020, 2021, 2022]
    val_years: [2023]
    test_years: [2024]
    description: "COVID + bull market"
```

---

## 2. Encoder Layer

### 2A. Shared Temporal Attention Encoder

| Parameter | Value | Search Range (TPE) | Notes |
|-----------|-------|-------------------|-------|
| `d_model` | 128 | [64, 128, 256] | Embedding dimension |
| `n_layers` | 4 | [2, 3, 4, 5, 6] | Transformer layers |
| `n_heads` | 4 | [2, 4, 8] | Attention heads |
| `d_ff` | 512 | 4 × d_model | Feed-forward dimension |
| `dropout` | 0.1 | [0.05, 0.1, 0.15, 0.2] | General dropout |
| `attention_dropout` | 0.1 | [0.05, 0.1, 0.15] | Attention-specific dropout |
| `activation` | `"gelu"` | - | GELU (standard for transformers) |
| `max_seq_len` | 90 | - | Maximum lookback days |
| `batch_size` | 32 | [16, 32, 64] | Per GPU |
| `epochs` | 100 | [50, 75, 100, 150] | With early stopping |
| `learning_rate` | 1e-4 | `loguniform(5e-5, 5e-4)` | Peak LR after warmup |
| `weight_decay` | 1e-5 | `loguniform(5e-6, 5e-5)` | L2 regularization |
| `warmup_steps` | 4000 | [2000, 4000, 6000] | Linear warmup |
| `lr_schedule` | `"cosine"` | - | Cosine decay with warmup |
| `gradient_clip` | 1.0 | - | Clip gradient norm |
| `label_smoothing` | 0.05 | [0.0, 0.05, 0.1] | Soften labels |
| `early_stop_patience` | 20 | [10, 15, 20, 25] | Epochs without improvement |
| `optimizer` | `"AdamW"` | - | Adam with decoupled weight decay |

**TPE Trials:** 50-100

---

### 2B. FinBERT Financial Text Encoder

| Parameter | Value | Search Range | Notes |
|-----------|-------|-------------|-------|
| `base_model` | `"ProsusAI/finbert"` | - | Pre-trained FinBERT |
| `max_length` | 512 | - | Max tokens per document |
| `projection_dim` | 256 | [128, 256, 384] | Output embedding size |
| `freeze_base` | `False` | - | Fine-tune entire model |
| `batch_size` | 16 | [8, 16, 32] | Small due to memory |
| `epochs_per_chunk` | 3 | [2, 3, 4, 5] | Fine-tuning epochs |
| `learning_rate` | 2e-5 | `loguniform(1e-5, 5e-5)` | Standard BERT fine-tune LR |
| `weight_decay` | 0.01 | `loguniform(0.001, 0.1)` | L2 regularization |
| `warmup_proportion` | 0.1 | [0.05, 0.1, 0.15] | % of steps for warmup |
| `gradient_clip` | 1.0 | - | Clip gradient norm |
| `dropout` | 0.1 | [0.05, 0.1, 0.15] | Classifier dropout |
| `early_stop_patience` | 5 | [3, 5, 7] | Epochs without improvement |
| `optimizer` | `"AdamW"` | - | Adam with decoupled weight decay |
| `scheduler` | `"linear"` | - | Linear decay with warmup |

**TPE Trials:** 20-30

---


## 3. Analyst Layer

### 3A. Technical Analyst (BiLSTM)

| Parameter | Value | Search Range (TPE) | Notes |
|-----------|-------|-------------------|-------|
| `input_dim` | 128 | - | Temporal embedding size |
| `hidden_dim` | 64 | [32, 64, 128] | LSTM hidden size |
| `num_layers` | 1 | [1, 2] | BiLSTM layers |
| `bidirectional` | `True` | - | Use both directions |
| `dropout` | 0.3 | [0.2, 0.3, 0.4] | LSTM dropout |
| `use_attention_pooling` | `True` | - | Weighted sequence pooling |
| `output_dim` | 3 | - | Trend, momentum, timing |
| `batch_size` | 64 | [32, 64, 128] | Per GPU |
| `epochs` | 50 | [30, 50, 75] | With early stopping |
| `learning_rate` | 1e-3 | `loguniform(5e-4, 5e-3)` | Adam LR |
| `weight_decay` | 1e-4 | `loguniform(5e-5, 5e-4)` | L2 regularization |
| `gradient_clip` | 1.0 | - | Clip gradient norm |
| `early_stop_patience` | 20 | [15, 20, 25] | Epochs without improvement |
| `optimizer` | `"Adam"` | - | Adam optimizer |
| `scheduler` | `"reduce_on_plateau"` | - | Reduce LR when loss stalls |

**TPE Trials:** 30-50

---

### 3B. Sentiment Analyst (MLP)

| Parameter | Value | Search Range | Notes |
|-----------|-------|-------------|-------|
| `input_dim` | 256 | - | Text embedding size |
| `hidden_dims` | [128, 64] | [[128], [128, 64], [256, 128, 64]] | Layer sizes |
| `output_dim` | 2 | - | Polarity, confidence |
| `dropout` | 0.2 | [0.1, 0.2, 0.3] | Regularization |
| `activation` | `"relu"` | - | ReLU activation |
| `batch_norm` | `True` | - | Batch normalization |
| `batch_size` | 128 | [64, 128, 256] | Per GPU |
| `epochs` | 30 | [20, 30, 50] | With early stopping |
| `learning_rate` | 1e-3 | `loguniform(5e-4, 5e-3)` | Adam LR |
| `weight_decay` | 1e-5 | `loguniform(5e-6, 5e-5)` | L2 regularization |
| `label_smoothing` | 0.05 | [0.0, 0.05, 0.1] | Soften labels |
| `early_stop_patience` | 15 | [10, 15, 20] | Epochs without improvement |
| `optimizer` | `"Adam"` | - | Adam optimizer |

**Grid Search:** 9 combinations → **TPE Fine-tuning:** 20 trials

---

### 3C. News Analyst (Multi-Head Attention Pooling)

| Parameter | Value | Search Range (TPE) | Notes |
|-----------|-------|-------------------|-------|
| `input_dim` | 256 | - | Per-document embedding size |
| `num_heads` | 4 | [2, 4, 8] | Attention heads |
| `head_dim` | 64 | - | Dimension per head |
| `dropout` | 0.1 | [0.05, 0.1, 0.15] | General dropout |
| `attention_dropout` | 0.1 | [0.05, 0.1, 0.15] | Attention dropout |
| `output_dim` | 2 | - | Impact score, relevance |
| `batch_size` | 64 | [32, 64, 128] | Per GPU (documents vary) |
| `epochs` | 40 | [30, 40, 50] | With early stopping |
| `learning_rate` | 1e-3 | `loguniform(5e-4, 5e-3)` | Adam LR |
| `weight_decay` | 1e-5 | `loguniform(5e-6, 5e-5)` | L2 regularization |
| `gradient_clip` | 1.0 | - | Clip gradient norm |
| `early_stop_patience` | 15 | [10, 15, 20] | Epochs without improvement |
| `optimizer` | `"Adam"` | - | Adam optimizer |

**TPE Trials:** 20-30

---


## 4. Risk Engine

### 4A. Volatility Estimation (GARCH + MLP Hybrid)

#### GARCH Component

| Parameter | Value | Notes |
|-----------|-------|-------|
| `p` | 1 | ARCH order |
| `q` | 1 | GARCH order |
| `dist` | `"normal"` | Error distribution |
| `rolling_window` | 252 | 1 year of trading days |
| `update_frequency` | `"daily"` | Recalculate daily |

#### MLP Component

| Parameter | Value | Search Range (TPE) | Notes |
|-----------|-------|-------------------|-------|
| `input_dim` | 128 | - | Temporal embedding size |
| `hidden_dims` | [64] | [[32], [64], [64, 32]] | Hidden layer sizes |
| `output_dim` | 4 | - | Vol_10d, Vol_30d, Regime, Confidence |
| `dropout` | 0.2 | [0.1, 0.2, 0.3] | Regularization |
| `activation` | `"relu"` | - | ReLU activation |
| `batch_size` | 128 | [64, 128, 256] | Per GPU |
| `epochs` | 40 | [30, 40, 50] | With early stopping |
| `learning_rate` | 1e-3 | `loguniform(5e-4, 5e-3)` | Adam LR |
| `weight_decay` | 1e-5 | `loguniform(5e-6, 5e-5)` | L2 regularization |
| `early_stop_patience` | 15 | [10, 15, 20] | Epochs without improvement |
| `optimizer` | `"Adam"` | - | Adam optimizer |

**TPE Trials:** 30-50

---

### 4B. Drawdown Risk (BiLSTM Dual Horizon)

| Parameter | Value | Search Range (TPE) | Notes |
|-----------|-------|-------------------|-------|
| `input_dim` | 128 | - | Temporal embedding size |
| `hidden_dim` | 64 | [32, 64, 128] | LSTM hidden size |
| `num_layers` | 1 | [1, 2] | BiLSTM layers |
| `bidirectional` | `True` | - | Use both directions |
| `dropout` | 0.3 | [0.2, 0.3, 0.4] | LSTM dropout |
| `output_dim_10d` | 3 | - | Prob, depth, recovery (10-day) |
| `output_dim_30d` | 3 | - | Prob, depth, recovery (30-day) |
| `batch_size` | 64 | [32, 64, 128] | Per GPU |
| `epochs` | 50 | [30, 50, 75] | With early stopping |
| `learning_rate` | 1e-3 | `loguniform(5e-4, 5e-3)` | Adam LR |
| `weight_decay` | 1e-4 | `loguniform(5e-5, 5e-4)` | L2 regularization |
| `gradient_clip` | 1.0 | - | Clip gradient norm |
| `early_stop_patience` | 20 | [15, 20, 25] | Epochs without improvement |
| `optimizer` | `"Adam"` | - | Adam optimizer |
| `scheduler` | `"reduce_on_plateau"` | - | Reduce LR when loss stalls |

**TPE Trials:** 30-50

---

### 4C. VaR & CVaR (Non-parametric)

*No hyperparameters — statistical calculations only.*

| Parameter | Value | Notes |
|-----------|-------|-------|
| `rolling_window` | 504 | 2 years of trading days |
| `confidence_levels` | [0.95, 0.99] | VaR thresholds |
| `update_frequency` | `"daily"` | Recalculate daily |
| `method` | `"historical"` | Empirical distribution |

---

### 4D. GNN Contagion Risk (StemGNN)

*Based on successful baseline reproduction results.*

| Parameter | Value | Search Range (TPE) | Notes |
|-----------|-------|-------------------|-------|
| `num_nodes` | 4428 | - | Number of stocks |
| `window_size` | 30 | [15, 30, 60] | Days of lookback |
| `horizon` | 1 | - | Predict 1 step (contagion, not price) |
| `multi_layer` | 13 | [5, 8, 13, 20] | StemGNN blocks |
| `embed_size` | 32 | [16, 32, 64] | Node embedding dimension |
| `hidden_size` | 64 | [32, 64, 128] | Hidden dimension |
| `learning_rate` | 0.01 | `loguniform(0.001, 0.05)` | RMSprop LR |
| `exponential_decay_step` | 13 | [5, 8, 13] | LR decay step |
| `decay_rate` | 0.5 | [0.3, 0.5, 0.7, 0.9] | LR decay multiplier |
| `dropout_rate` | 0.75 | [0.5, 0.6, 0.75, 0.8] | Regularization |
| `batch_size` | 32 | [16, 32, 64] | Per GPU |
| `epochs` | 100 | [50, 75, 100] | With early stopping |
| `optimizer` | `"RMSprop"` | - | From baseline |
| `norm_method` | `"z_score"` | - | Input normalization |
| `early_stop_patience` | 20 | [15, 20, 25] | Epochs without improvement |
| `gradient_clip` | 1.0 | - | Clip gradient norm |
| `train_length` | 7 | - | Years for training split |
| `valid_length` | 2 | - | Years for validation split |
| `test_length` | 1 | - | Years for test split |
| `leakyrelu_rate` | 0.2 | - | LeakyReLU slope |
| `cheb_k` | 3 | - | Chebyshev polynomial order |
| `top_k_edges` | 66 | [20, 44, 66, 100] | Edges per node (√4428 ≈ 66) |

**TPE Trials:** 50-100

---

### 4E. Liquidity Risk (Rule-based)

*No hyperparameters — rule-based thresholds only.*

| Parameter | Value | Notes |
|-----------|-------|-------|
| `min_volume_percentile` | 20 | Below this = illiquid |
| `max_spread_pct` | 0.5 | Above this = high slippage |
| `market_cap_tiers` | [10e9, 2e9] | Large (>$10B), Mid, Small (<$2B) |
| `days_to_liquidate_threshold` | 5 | >5 days = liquidity warning |
| `update_frequency` | `"daily"` | Recalculate daily |

---

### 4F. Regime Detection (MTGNN Graph Builder + Classifier)

#### Graph Builder Component

| Parameter | Value | Search Range | Notes |
|-----------|-------|-------------|-------|
| `num_nodes` | 4428 | - | Number of stocks |
| `node_embedding_dim` | 64 | [32, 64, 128] | MTGNN embedding size |
| `top_k` | 66 | [20, 44, 66, 100] | Edges per node |
| `cheb_k` | 3 | [2, 3, 5] | Chebyshev order |
| `input_dim` | 384 | - | 128 temporal + 256 text |

#### Classifier Component

| Parameter | Value | Search Range (TPE) | Notes |
|-----------|-------|-------------------|-------|
| `input_features` | 5 | - | Density, modularity, avg_degree, clustering_coef, transitivity |
| `hidden_dims` | [32] | [[16], [32], [32, 16]] | Hidden layer sizes |
| `output_classes` | 4 | - | calm, volatile, crisis, rotation |
| `dropout` | 0.2 | [0.1, 0.2, 0.3] | Regularization |
| `activation` | `"relu"` | - | ReLU activation |
| `batch_size` | 256 | [128, 256, 512] | Per GPU |
| `epochs` | 30 | [20, 30, 40] | With early stopping |
| `learning_rate` | 1e-3 | `loguniform(5e-4, 5e-3)` | Adam LR |
| `weight_decay` | 1e-5 | `loguniform(5e-6, 5e-5)` | L2 regularization |
| `early_stop_patience` | 10 | [5, 10, 15] | Epochs without improvement |
| `optimizer` | `"Adam"` | - | Adam optimizer |

**Training Frequency:** Weekly (graph building only)  
**TPE Trials:** 20-30 (classifier only)

---

### 4G. Position Sizing Engine (Rule-based, User-adjustable)

*No hyperparameters — user-configurable weights.*

| Parameter | Default Value | User Adjustable | Notes |
|-----------|--------------|-----------------|-------|
| `weight_volatility` | 0.20 | Yes | Weight for volatility risk |
| `weight_drawdown` | 0.15 | Yes | Weight for drawdown risk |
| `weight_var_cvar` | 0.15 | Yes | Combined VaR/CVaR weight |
| `weight_contagion` | 0.25 | Yes | Weight for contagion risk |
| `weight_liquidity` | 0.15 | Yes | Weight for liquidity risk |
| `weight_regime` | 0.10 | Yes | Weight for regime risk |
| `threshold_full` | 0.30 | Yes | Below this = 100% position |
| `threshold_high` | 0.50 | Yes | Below this = 75% position |
| `threshold_medium` | 0.70 | Yes | Below this = 50% position |
| `threshold_low` | 0.85 | Yes | Below this = 25% position |
| `veto_threshold` | 0.90 | Yes | Above this = no trade |

---

## 5. Fusion Layer

### Fusion Engine (MLP + Rules)

#### MLP Component (Layer 1)

| Parameter | Value | Search Range (TPE) | Notes |
|-----------|-------|-------------------|-------|
| `input_dim` | 13 | - | Qualitative(2) + Quantitative(2) + 9 module scores |
| `hidden_dims` | [64, 32] | [[32], [64, 32], [128, 64, 32]] | Hidden layer sizes |
| `output_dim` | 3 | - | Buy/Hold/Sell logits |
| `dropout` | 0.2 | [0.1, 0.2, 0.3] | Regularization |
| `activation` | `"relu"` | - | ReLU activation |
| `batch_norm` | `True` | - | Batch normalization |
| `batch_size` | 256 | [128, 256, 512] | Per GPU |
| `epochs` | 50 | [30, 50, 75] | With early stopping |
| `learning_rate` | 1e-3 | `loguniform(5e-4, 5e-3)` | Adam LR |
| `weight_decay` | 1e-5 | `loguniform(5e-6, 5e-5)` | L2 regularization |
| `label_smoothing` | 0.05 | [0.0, 0.05, 0.1] | Soften labels |
| `early_stop_patience` | 15 | [10, 15, 20] | Epochs without improvement |
| `optimizer` | `"Adam"` | - | Adam optimizer |

#### Rule-based Component (Layer 2)

| Rule | Condition | Action |
|------|-----------|--------|
| `liquidity_veto` | liquidity_score < 0.3 | REJECT (no trade) |
| `drawdown_cap` | drawdown_probability > 0.8 | Cap size at 25% |
| `contagion_veto` | contagion_score > 0.9 | REJECT (no trade) |
| `regime_override` | regime == "crisis" AND confidence > 0.7 | Force SELL or HOLD |

**TPE Trials:** 50-100

---

## 6. Hyperparameter Search Spaces

### TPE (Bayesian Optimization) Configuration

```yaml
tpe_config:
  algorithm: "tpe"  # Tree-structured Parzen Estimator
  n_initial_points: 20  # Random exploration before TPE
  n_trials:  # Model-specific (see below)
    temporal_encoder: 75
    finbert: 25
    fundamental_mlp: 30
    technical_analyst: 40
    sentiment_analyst: 20
    news_analyst: 25
    volatility_mlp: 40
    drawdown_bilstm: 40
    stemgnn: 75
    mtgnn_classifier: 25
    fusion_mlp: 75
  early_stop_trials: 20  # Stop if no improvement after 20 trials
  direction: "minimize"  # Minimize validation loss
  pruner: "median"  # Prune unpromising trials
```

### Grid Search Configuration (for LightGBM/XGBoost)

```yaml
grid_search_config:
  cv_folds: 5  # Chronological cross-validation
  scoring: "neg_mean_squared_error"
  n_jobs: -1  # Use all CPU cores
  verbose: 1
```

---

## 7. Learning Rate Schedules

### Cosine Decay with Warmup (Temporal Encoder)

```python
def cosine_decay_with_warmup(step, warmup_steps, total_steps, base_lr):
    if step < warmup_steps:
        # Linear warmup
        return base_lr * (step / warmup_steps)
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))
```

### Reduce on Plateau (BiLSTM Models)

```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,        # Reduce LR by half
    patience=10,       # Wait 10 epochs
    min_lr=1e-6        # Minimum LR
)
```

### Linear Decay with Warmup (FinBERT)

```python
def linear_decay_with_warmup(step, warmup_steps, total_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    else:
        return base_lr * (1 - (step - warmup_steps) / (total_steps - warmup_steps))
```

### Exponential Decay (StemGNN)

```python
def exponential_decay(epoch, base_lr, decay_step, decay_rate):
    return base_lr * (decay_rate ** (epoch // decay_step))
```

---

## 8. Regularization Summary

| Model | Dropout | Attn Dropout | Weight Decay | Label Smooth | Early Stop | Grad Clip |
|-------|---------|--------------|--------------|--------------|------------|-----------|
| Temporal Encoder | 0.1 | 0.1 | 1e-5 | 0.05 | 20 | 1.0 |
| FinBERT | 0.1 | 0.1 | 0.01 | - | 5 | 1.0 |
| Fundamental MLP | 0.2 | - | 1e-5 | - | 15 | - |
| Technical Analyst | 0.3 | - | 1e-4 | - | 20 | 1.0 |
| Sentiment Analyst | 0.2 | - | 1e-5 | 0.05 | 15 | 1.0 |
| News Analyst | 0.1 | 0.1 | 1e-5 | - | 15 | 1.0 |
| Volatility MLP | 0.2 | - | 1e-5 | - | 15 | 1.0 |
| Drawdown BiLSTM | 0.3 | - | 1e-4 | - | 20 | 1.0 |
| StemGNN | 0.75 | - | 1e-5 | - | 20 | 1.0 |
| MTGNN Classifier | 0.2 | - | 1e-5 | - | 10 | 1.0 |
| Fusion MLP | 0.2 | - | 1e-5 | 0.05 | 15 | 1.0 |

### XGBoost/LightGBM Regularization

| Model | L1 (reg_alpha) | L2 (reg_lambda) | Subsampling | Early Stop |
|-------|----------------|-----------------|-------------|------------|
| XGBoost (Fundamental Encoder) | 0.1 | 1.0 | 0.7 | 50 |
| LightGBM (Fundamental Analyst) | 0.1 | 1.0 | 0.7 | 50 |

---

## 9. YAML Configuration File

```yaml
# hyperparameters.yaml
# Complete hyperparameter configuration for all models

version: "1.0"
date: "2026-04-23"

global:
  device: "cuda"
  seed: 42
  dtype: "float32"
  num_workers: 4
  pin_memory: true
  mixed_precision: true
  checkpoint_dir: "./checkpoints/"
  log_dir: "./logs/"

training_chunks:
  chunk_1:
    train: [2000, 2001, 2002, 2003, 2004]
    val: [2005]
    test: [2006]
  chunk_2:
    train: [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014]
    val: [2015]
    test: [2016]
  chunk_3:
    train: [2017, 2018, 2019, 2020, 2021, 2022]
    val: [2023]
    test: [2024]

encoders:
  temporal:
    model: "transformer"
    d_model: 128
    n_layers: 4
    n_heads: 4
    d_ff: 512
    dropout: 0.1
    attention_dropout: 0.1
    activation: "gelu"
    max_seq_len: 90
    batch_size: 32
    epochs: 100
    learning_rate: 1.0e-4
    weight_decay: 1.0e-5
    warmup_steps: 4000
    lr_schedule: "cosine"
    gradient_clip: 1.0
    label_smoothing: 0.05
    early_stop_patience: 20
    optimizer: "AdamW"
    tpe_trials: 75

  finbert:
    base_model: "ProsusAI/finbert"
    max_length: 512
    projection_dim: 256
    freeze_base: false
    batch_size: 16
    epochs_per_chunk: 3
    learning_rate: 2.0e-5
    weight_decay: 0.01
    warmup_proportion: 0.1
    gradient_clip: 1.0
    dropout: 0.1
    early_stop_patience: 5
    optimizer: "AdamW"
    scheduler: "linear"
    tpe_trials: 25

  fundamental:
    xgboost:
      max_depth: 4
      learning_rate: 0.01
      n_estimators: 500
      subsample: 0.7
      colsample_bytree: 0.7
      reg_alpha: 0.1
      reg_lambda: 1.0
      min_child_weight: 5
      early_stopping_rounds: 50
      objective: "reg:squarederror"
      tree_method: "hist"
    mlp:
      input_dim: 70
      hidden_dims: [256]
      output_dim: 128
      dropout: 0.2
      activation: "relu"
      use_layer_norm: true
      batch_size: 128
      epochs: 50
      learning_rate: 1.0e-3
      weight_decay: 1.0e-5
      early_stop_patience: 15
      optimizer: "Adam"
      tpe_trials: 30

analysts:
  technical:
    model: "bilstm"
    input_dim: 128
    hidden_dim: 64
    num_layers: 1
    bidirectional: true
    dropout: 0.3
    use_attention_pooling: true
    output_dim: 3
    batch_size: 64
    epochs: 50
    learning_rate: 1.0e-3
    weight_decay: 1.0e-4
    gradient_clip: 1.0
    early_stop_patience: 20
    optimizer: "Adam"
    scheduler: "reduce_on_plateau"
    tpe_trials: 40

  sentiment:
    model: "mlp"
    input_dim: 256
    hidden_dims: [128, 64]
    output_dim: 2
    dropout: 0.2
    activation: "relu"
    batch_norm: true
    batch_size: 128
    epochs: 30
    learning_rate: 1.0e-3
    weight_decay: 1.0e-5
    label_smoothing: 0.05
    early_stop_patience: 15
    optimizer: "Adam"
    tpe_trials: 20

  news:
    model: "attention_pooling"
    input_dim: 256
    num_heads: 4
    head_dim: 64
    dropout: 0.1
    attention_dropout: 0.1
    output_dim: 2
    batch_size: 64
    epochs: 40
    learning_rate: 1.0e-3
    weight_decay: 1.0e-5
    gradient_clip: 1.0
    early_stop_patience: 15
    optimizer: "Adam"
    tpe_trials: 25

  fundamental_lgb:
    model: "lightgbm"
    input_dim: 128
    objective: "multiclass"
    num_class: 3
    boosting_type: "gbdt"
    num_leaves: 31
    learning_rate: 0.01
    n_estimators: 500
    subsample: 0.7
    colsample_bytree: 0.7
    reg_alpha: 0.1
    reg_lambda: 1.0
    min_child_samples: 20
    min_child_weight: 0.001
    early_stopping_rounds: 50
    verbosity: -1

risk:
  volatility:
    garch:
      p: 1
      q: 1
      dist: "normal"
      rolling_window: 252
      update_frequency: "daily"
    mlp:
      input_dim: 128
      hidden_dims: [64]
      output_dim: 4
      dropout: 0.2
      activation: "relu"
      batch_size: 128
      epochs: 40
      learning_rate: 1.0e-3
      weight_decay: 1.0e-5
      early_stop_patience: 15
      optimizer: "Adam"
      tpe_trials: 40

  drawdown:
    model: "bilstm"
    input_dim: 128
    hidden_dim: 64
    num_layers: 1
    bidirectional: true
    dropout: 0.3
    output_dim_10d: 3
    output_dim_30d: 3
    batch_size: 64
    epochs: 50
    learning_rate: 1.0e-3
    weight_decay: 1.0e-4
    gradient_clip: 1.0
    early_stop_patience: 20
    optimizer: "Adam"
    scheduler: "reduce_on_plateau"
    tpe_trials: 40

  var_cvar:
    rolling_window: 504
    confidence_levels: [0.95, 0.99]
    update_frequency: "daily"
    method: "historical"

  contagion:
    model: "stemgnn"
    num_nodes: 4428
    window_size: 30
    horizon: 1
    multi_layer: 13
    embed_size: 32
    hidden_size: 64
    learning_rate: 0.01
    exponential_decay_step: 13
    decay_rate: 0.5
    dropout_rate: 0.75
    batch_size: 32
    epochs: 100
    optimizer: "RMSprop"
    norm_method: "z_score"
    early_stop_patience: 20
    gradient_clip: 1.0
    train_length: 7
    valid_length: 2
    test_length: 1
    leakyrelu_rate: 0.2
    cheb_k: 3
    top_k_edges: 66
    tpe_trials: 75

  liquidity:
    min_volume_percentile: 20
    max_spread_pct: 0.5
    market_cap_tiers: [10.0e9, 2.0e9]
    days_to_liquidate_threshold: 5
    update_frequency: "daily"

  regime:
    graph_builder:
      num_nodes: 4428
      node_embedding_dim: 64
      top_k: 66
      cheb_k: 3
      input_dim: 384
    classifier:
      input_features: 5
      hidden_dims: [32]
      output_classes: 4
      dropout: 0.2
      activation: "relu"
      batch_size: 256
      epochs: 30
      learning_rate: 1.0e-3
      weight_decay: 1.0e-5
      early_stop_patience: 10
      optimizer: "Adam"
      tpe_trials: 25

  position_sizing:
    weights:
      volatility: 0.20
      drawdown: 0.15
      var_cvar: 0.15
      contagion: 0.25
      liquidity: 0.15
      regime: 0.10
    thresholds:
      full: 0.30
      high: 0.50
      medium: 0.70
      low: 0.85
      veto: 0.90

fusion:
  mlp:
    input_dim: 13
    hidden_dims: [64, 32]
    output_dim: 3
    dropout: 0.2
    activation: "relu"
    batch_norm: true
    batch_size: 256
    epochs: 50
    learning_rate: 1.0e-3
    weight_decay: 1.0e-5
    label_smoothing: 0.05
    early_stop_patience: 15
    optimizer: "Adam"
    tpe_trials: 75
  rules:
    liquidity_veto:
      condition: "liquidity_score < 0.3"
      action: "REJECT"
    drawdown_cap:
      condition: "drawdown_probability > 0.8"
      action: "CAP_SIZE_25%"
    contagion_veto:
      condition: "contagion_score > 0.9"
      action: "REJECT"
    regime_override:
      condition: "regime == 'crisis' AND confidence > 0.7"
      action: "FORCE_SELL_OR_HOLD"

tpe_config:
  algorithm: "tpe"
  n_initial_points: 20
  early_stop_trials: 20
  direction: "minimize"
  pruner: "median"
```

---

**Document Version:** 1.0  
**Status:** Ready for Implementation  
**Next Step:** Generate training scripts based on these configurations.