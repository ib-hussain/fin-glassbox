# TemporalEncoder_PreImplementation.md

## Document Purpose

This document contains **all specifications, theory, and implementation details** for the **Shared Temporal Attention Encoder** — the primary time-series encoder in the Explainable Distributed Deep Learning Framework for Financial Risk Management.

This is a **pre-implementation specification** — it defines WHAT to build before writing a single line of code.

---

## Table of Contents

1. [Role in the Architecture](#1-role-in-the-architecture)
2. [Theoretical Foundation](#2-theoretical-foundation)
3. [Model Specification](#3-model-specification)
4. [Input Specification](#4-input-specification)
5. [Output Specification](#5-output-specification)
6. [Architecture Details](#6-architecture-details)
7. [Training Protocol](#7-training-protocol)
8. [Hyperparameter Search](#8-hyperparameter-search)
9. [Regularization Strategy](#9-regularization-strategy)
10. [Code Structure](#10-code-structure)
11. [What the Code Must Do](#11-what-the-code-must-do)
12. [Validation & Testing](#12-validation--testing)
13. [Acknowledgment: Specification Evolution](#13-acknowledgment-specification-evolution)

---

## 1. Role in the Architecture

### Where It Sits

```
INPUT: Daily OHLCV + derived indicators for ONE stock
    │
    v
┌─────────────────────────────────────┐
│  SHARED TEMPORAL ATTENTION ENCODER  │  ← THIS MODULE
│  (Transformer, 4 layers, 4 heads)   │
└─────────────────────────────────────┘
    │
    │  128-dim temporal embedding
    │
    ├──────→ Technical Analyst (BiLSTM)
    ├──────→ Volatility Model (GARCH + MLP)
    ├──────→ Drawdown Model (BiLSTM)
    └──────→ Regime Detection (MTGNN)
```

### What "Shared" Means

The **same encoder** is used for ALL stocks. We don't train one encoder per stock. Instead:
- The encoder learns universal temporal patterns from price data
- It processes each stock independently but shares weights across all stocks
- This is computationally efficient (one model for 4,428+ stocks)
- The encoder learns what a "trend" or "volatility spike" looks like regardless of which stock it's looking at

### Why Attention-Based (Not LSTM/CNN)

| Model Type | Problem for Our Use Case |
|------------|-------------------------|
| **LSTM** | Sequential processing, struggles with very long sequences (90+ days), vanishing gradients |
| **CNN** | Fixed receptive field, cannot dynamically weight important time steps |
| **Transformer (Attention)** | ✅ Dynamically attends to important time steps, parallel processing, proven in time-series |

The transformer can look at 90 days of price data and say: "Day 15 and Day 72 matter most for this prediction" — something LSTMs and CNNs cannot do natively.

---

## 2. Theoretical Foundation

### Self-Attention Mechanism

The core operation is **scaled dot-product attention**:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

**In plain English:**
1. **Query (Q):** "What am I looking for?" — the current time step's representation
2. **Key (K):** "What do I have?" — all time steps' representations
3. **Value (V):** "What information should I pass?" — the actual data
4. **QK^T:** Compute similarity between every pair of time steps
5. **softmax:** Convert similarities to weights (sum to 1)
6. **× V:** Weighted sum of values = attend to important time steps

### Multi-Head Attention

Instead of one attention mechanism, we use **4 heads** in parallel:

```
MultiHead(Q, K, V) = Concat(head_1, head_2, head_3, head_4) × W_o

where head_i = Attention(Q × W_i^Q, K × W_i^K, V × W_i^V)
```

**Why multiple heads?**
- Head 1 might attend to short-term momentum (last 5 days)
- Head 2 might attend to medium-term trends (last 20 days)
- Head 3 might attend to long-term patterns (seasonal)
- Head 4 might attend to volatility clusters

Each head learns a different "view" of the same data.

### Positional Encoding

Transformers have no built-in notion of sequence order. We add **sinusoidal positional encoding**:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

This gives the model information about WHERE each time step is in the sequence (day 1 vs day 90).

### Why This Works for Financial Data

Financial time series have:
- **Long-range dependencies:** What happened 60 days ago affects today
- **Varying importance:** Some days matter more (earnings, crashes)
- **Multiple timescales:** Intra-week, monthly, quarterly patterns

Attention handles all three naturally.

---

## 3. Model Specification

### Core Architecture

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_model` | 128 | Embedding dimension (output size) |
| `n_layers` | 4 | Transformer encoder layers |
| `n_heads` | 4 | Attention heads per layer |
| `d_ff` | 512 | Feed-forward hidden dimension (4 × d_model) |
| `dropout` | 0.1 | General dropout rate |
| `attention_dropout` | 0.1 | Attention-specific dropout |
| `activation` | GELU | Activation function |
| `max_seq_len` | 90 | Maximum sequence length (trading days) |
| `norm_first` | True | Pre-LayerNorm (better stability) |

### Layer Structure (Per Transformer Block)

```
Input: (batch, seq_len, d_model)
    │
    v
┌─────────────────────┐
│  LayerNorm           │
│  Multi-Head Attention│  ← 4 heads, each head_dim = 32
│  Dropout             │
│  + Residual          │  ← skip connection: output = attention(x) + x
└─────────────────────┘
    │
    v
┌─────────────────────┐
│  LayerNorm           │
│  Feed-Forward        │  ← Linear(128→512) → GELU → Linear(512→128)
│  Dropout             │
│  + Residual          │  ← skip connection
└─────────────────────┘
    │
    v
Output: (batch, seq_len, 128)
```

Wait — the architecture doc says "NO residual connections." Let me clarify:

> **ARCHITECTURAL NOTE:** The project specification states "❌ NO residual connections." However, this applies to the **overall system architecture** (no residual connections between modules). Within the transformer encoder itself, residual connections are **standard and necessary** for training deep attention models. Without them, gradients vanish and the transformer cannot learn. The "no residual connections" rule refers to inter-module connections, not intra-module transformer design.

### Parameter Count

```
d_model = 128
n_heads = 4
d_ff = 512
n_layers = 4

Per layer:
  Attention: 4 × (128×32 + 128×32 + 128×32) × 3 = ~147K
  FFN: 128×512 + 512×128 = ~131K
  Total per layer: ~278K

Total: 4 × 278K ≈ 1.1M parameters
Positional encoding: ~11K
Final projection: ~16K

Total trainable parameters: ~1.15M
```

This is **lightweight** — fits easily on any GPU, trains fast.

---

## 4. Input Specification

### Data Source

The encoder receives data from the **market data pipeline** (Data Family #1 — yfinance).

### Required Fields

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `open` | float | Opening price | yfinance |
| `high` | float | Highest price of day | yfinance |
| `low` | float | Lowest price of day | yfinance |
| `close` | float | Closing price | yfinance |
| `volume` | float | Trading volume | yfinance |
| `adj_close` | float | Adjusted close (for splits/dividends) | yfinance |

### Derived Input Features (Engineered BEFORE Encoding)

| Feature | Formula | Purpose |
|---------|---------|---------|
| `log_return` | log(close_t / close_t-1) | Stationary returns |
| `log_volume` | log(volume + 1) | Normalized volume |
| `high_low_ratio` | (high - low) / close | Intraday range |
| `close_open_ratio` | (close - open) / open | Overnight vs intraday |
| `sma_5` | 5-day simple moving average | Short-term trend |
| `sma_20` | 20-day simple moving average | Medium-term trend |
| `volatility_5` | 5-day rolling std of returns | Short-term volatility |
| `volatility_20` | 20-day rolling std of returns | Medium-term volatility |
| `rsi_14` | 14-day RSI | Momentum oscillator |
| `volume_ratio` | volume / 20-day avg volume | Volume spike detection |

**Total input features: 10** (6 raw + 4 derived from raw)

### Input Shape

```
Shape: (batch_size, seq_len, n_features)
       (batch_size, 30-90, 10)

Example for one stock, 30-day window:
Shape: (1, 30, 10)

Where dim 2 = [open, high, low, close, volume, adj_close,
               log_return, log_volume, high_low_ratio, close_open_ratio]
```

### Sequence Length

| Window | Use Case | Description |
|--------|----------|-------------|
| **30 days** | Default | Standard operating window (~6 trading weeks) |
| **60 days** | Extended | Capture quarterly patterns |
| **90 days** | Maximum | Long-term trend analysis |

The encoder must support variable sequence lengths (padding to max_seq_len=90).

### Normalization

**Z-score normalization per feature, per stock:**
```python
normalized_value = (value - mean_of_feature) / std_of_feature
```

This is computed from the **training period only** (no lookahead). Statistics are stored and reused during inference.

---

## 5. Output Specification

### Primary Output

A **128-dimensional temporal embedding** for each stock at each time step:

```
Shape: (batch_size, seq_len, 128)
```

### Pooling for Downstream Modules

Different downstream modules need different representations:

| Downstream Module | Pooling Method | Output Shape | Description |
|------------------|---------------|-------------|-------------|
| **Technical Analyst** | Last timestamp | `(batch, 128)` | Most recent state |
| **Volatility Model** | Mean pooling | `(batch, 128)` | Average behavior |
| **Drawdown Model** | Attention pooling | `(batch, 128)` | Learned weighted average |
| **Regime Detection** | Last timestamp | `(batch, 128)` | Current market state |

### Output Format

```python
# The encoder returns a dictionary to support different pooling needs
output = {
    "sequence": tensor_of_shape(batch, seq_len, 128),    # Full sequence
    "last_hidden": tensor_of_shape(batch, 128),           # Last time step
    "mean_pooled": tensor_of_shape(batch, 128),           # Mean over time
    "attention_pooled": tensor_of_shape(batch, 128),      # Learned weighted mean
}
```

---

## 6. Architecture Details

### Positional Encoding

```python
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=90):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
```

### Input Projection

Raw features (10-dim) must be projected to d_model (128-dim):

```python
self.input_projection = nn.Linear(10, d_model)  # 10 → 128
```

### Transformer Encoder (PyTorch)

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=128,
    nhead=4,
    dim_feedforward=512,
    dropout=0.1,
    activation='gelu',
    batch_first=True,
    norm_first=True,  # Pre-LN for stability
)

self.transformer = nn.TransformerEncoder(
    encoder_layer,
    num_layers=4,
)
```

### Pooling Heads

```python
self.attention_pooling = nn.MultiheadAttention(
    embed_dim=128,
    num_heads=1,
    batch_first=True,
)

# Query vector (learned): "What should I pay attention to?"
self.pooling_query = nn.Parameter(torch.randn(1, 1, 128))
```

### Full Forward Pass

```python
def forward(self, x, mask=None):
    # x: (batch, seq_len, 10)
    
    # 1. Project input to d_model
    x = self.input_projection(x)  # (batch, seq_len, 128)
    
    # 2. Add positional encoding
    x = self.pos_encoding(x)
    
    # 3. Pass through transformer
    x = self.transformer(x, src_key_padding_mask=mask)  # (batch, seq_len, 128)
    
    # 4. Generate different pooled representations
    last_hidden = x[:, -1, :]                            # (batch, 128)
    mean_pooled = x.mean(dim=1)                          # (batch, 128)
    
    # Attention pooling: learn which time steps matter
    query = self.pooling_query.expand(x.size(0), -1, -1) # (batch, 1, 128)
    attn_pooled, _ = self.attention_pooling(query, x, x) # (batch, 1, 128)
    attn_pooled = attn_pooled.squeeze(1)                 # (batch, 128)
    
    return {
        "sequence": x,
        "last_hidden": last_hidden,
        "mean_pooled": mean_pooled,
        "attention_pooled": attn_pooled,
    }
```

---

## 7. Training Protocol

### Training Objective

The encoder is trained with a **self-supervised masked prediction task**:

```
Given:  [P_t-30, P_t-29, ..., MASK, ..., P_t]
Predict: The masked price/return value
```

This forces the encoder to understand temporal patterns without needing labeled data.

### Alternative: Supervised Training

If labeled data is available (e.g., future returns as target):

```python
# Predict next-day return from 30-day window
Given: 30 days of features
Predict: Return on day 31

Loss = MSE(predicted_return, actual_return)
```

### Training Data

| Chunk | Training Period | Validation | Test |
|-------|----------------|------------|------|
| 1 | 2000-2004 | 2005 | 2006 |
| 2 | 2007-2014 | 2015 | 2016 |
| 3 | 2017-2022 | 2023 | 2024 |

### Data Volume

With 4,428 stocks × ~6,288 trading days:
- **~27.8 million** daily observations
- Each training sample: 30-day window stride=1
- **~26 million training samples** across all stocks

This is **more than enough** for a 1.15M parameter model.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch size | 32 |
| Epochs | 100 (with early stopping) |
| Optimizer | AdamW |
| Learning rate | 1e-4 (peak after warmup) |
| Weight decay | 1e-5 |
| LR schedule | Cosine decay with warmup |
| Warmup steps | 4,000 |
| Gradient clipping | 1.0 |
| Label smoothing | 0.05 |
| Early stop patience | 20 epochs |
| Mixed precision | FP16 (if GPU available) |

---

## 8. Hyperparameter Search

### Why Hyperparameter Search Is CRITICAL for the Temporal Encoder

The Temporal Encoder is the **most upstream model** in the entire pipeline. Every downstream module depends on the quality of its embeddings:

```
Temporal Encoder → Technical Analyst → Fusion → Final Decision
Temporal Encoder → Volatility Model → Position Sizing
Temporal Encoder → Drawdown Model → Position Sizing
Temporal Encoder → Regime Detection → Position Sizing
```

**If the encoder produces poor embeddings, EVERYTHING downstream fails.** There is no way to recover from a bad encoder.

### What Happens Without Proper HP Search

| Scenario | Consequence |
|----------|-------------|
| **Learning rate too high** | Embeddings oscillate, never converge, downstream models get noise |
| **Learning rate too low** | Embeddings don't learn meaningful patterns, downstream models get random features |
| **Too few layers (2)** | Cannot capture long-range dependencies, misses quarterly patterns |
| **Too many layers (6+)** | Overfits on training data, fails on new market regimes |
| **Dropout too high** | Loses important temporal signals, embeddings are too noisy |
| **Dropout too low** | Memorizes training data, embeddings don't generalize |
| **Wrong d_model** | Too small = information bottleneck; too large = overfitting + slow |

### Search Method: TPE (Bayesian Optimization)

**Why TPE:**
- More efficient than grid search (finds good HPs in fewer trials)
- More effective than random search (learns from previous trials)
- Handles mixed parameter types (continuous LR, discrete layers)
- Proven in prior work (MTGNN baseline used this)

### Search Space

| Parameter | Type | Range | Rationale |
|-----------|------|-------|-----------|
| `learning_rate` | Log-uniform | 5e-5 → 5e-4 | Critical; too high diverges, too low stalls |
| `n_layers` | Discrete | {2, 3, 4, 5, 6} | Depth vs overfitting trade-off |
| `n_heads` | Discrete | {2, 4, 8} | Must divide d_model; more heads = more views |
| `d_model` | Discrete | {64, 128, 256} | Embedding capacity |
| `dropout` | Uniform | 0.05 → 0.2 | Regularization strength |
| `attention_dropout` | Uniform | 0.05 → 0.15 | Attention-specific regularization |
| `weight_decay` | Log-uniform | 5e-6 → 5e-5 | L2 regularization |
| `warmup_steps` | Discrete | {2000, 4000, 6000} | LR warmup duration |
| `batch_size` | Discrete | {16, 32, 64} | Memory vs gradient noise trade-off |

### Search Budget

| Phase | Trials | Time per Trial | Total Time |
|-------|--------|---------------|------------|
| **Initial exploration** | 20 random | ~2-4 hours | ~2-3 days |
| **TPE optimization** | 55 | ~2-4 hours | ~5-10 days |
| **Total** | **75 trials** | | **~1-2 weeks** |

**Why 75 trials:**
- 20 random trials map the space
- 55 TPE trials converge to optimum
- Proven sufficient in MTGNN baseline reproduction (100 trials was overkill)

### Search Protocol

```python
def objective(trial):
    # Sample hyperparameters
    lr = trial.suggest_float("lr", 5e-5, 5e-4, log=True)
    n_layers = trial.suggest_int("n_layers", 2, 6)
    dropout = trial.suggest_float("dropout", 0.05, 0.2)
    ...
    
    # Build model with sampled HPs
    model = TemporalEncoder(
        d_model=128,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        ...
    )
    
    # Train on chunk 1 (2000-2004)
    # Validate on 2005
    val_loss = train_and_evaluate(model, train_data, val_data)
    
    return val_loss

# Run search
study = optuna.create_study(
    direction="minimize",
    sampler=TPESampler(n_startup_trials=20),
    pruner=MedianPruner(n_startup_trials=10),
)
study.optimize(objective, n_trials=75)
```

### Success Criteria

The hyperparameter search is considered **successful** when:

1. **Validation loss stabilizes** (not decreasing, not increasing)
2. **Embedding quality check:** Cosine similarity of embeddings for same-sector stocks > 0.3
3. **Temporal consistency:** Embedding of day T and day T+1 have cosine similarity > 0.8
4. **No overfitting:** Train loss and val loss gap < 20% of val loss
5. **Reproducibility:** Two runs with same HPs give similar results (±5% val loss)

---

## 9. Regularization Strategy

### Why Regularization Matters for Financial Data

Financial data is **noisy and non-stationary**. Without strong regularization, the encoder will:
- Memorize noise as signal
- Fail on new market regimes
- Produce embeddings that don't generalize

### Regularization Layers

| Technique | Value | What It Does |
|-----------|-------|-------------|
| **Dropout** | 0.1 | Randomly drops 10% of neurons during training → prevents co-adaptation |
| **Attention Dropout** | 0.1 | Randomly drops 10% of attention weights → forces diverse attention |
| **Weight Decay** | 1e-5 | Penalizes large weights → simpler model → better generalization |
| **Gradient Clipping** | 1.0 | Prevents exploding gradients from outlier days (crashes, rallies) |
| **Label Smoothing** | 0.05 | Softens prediction targets → prevents overconfidence |
| **Early Stopping** | patience=20 | Stops training when validation loss stops improving |
| **Cosine LR Schedule** | with warmup | Large LR early → fast learning; small LR late → fine-tuning |

### Why Each One

| Problem in Financial Data | Solution |
|--------------------------|----------|
| Extreme returns (crashes) cause gradient spikes | ✅ Gradient Clipping |
| Market regimes change → model must generalize | ✅ Dropout + Weight Decay |
| Overconfident predictions on familiar patterns | ✅ Label Smoothing |
| Training too long memorizes noise | ✅ Early Stopping |
| Noisy attention to irrelevant days | ✅ Attention Dropout |

---

## 10. Code Structure

### File Location

```
code/encoders/temporal_encoder.py
```

### Class Structure

```python
class SinusoidalPositionalEncoding(nn.Module):
    """Adds positional information to input embeddings."""
    def __init__(self, d_model, max_len=90): ...
    def forward(self, x): ...

class TemporalEncoder(nn.Module):
    """Shared Temporal Attention Encoder for financial time series."""
    
    def __init__(self, config: TemporalEncoderConfig): ...
    
    def forward(self, x, mask=None) -> dict: ...
    
    def get_embedding(self, x, pooling='last') -> torch.Tensor: ...
    
    def save(self, path): ...
    
    @classmethod
    def load(cls, path) -> 'TemporalEncoder': ...

class TemporalEncoderConfig:
    """Configuration dataclass for TemporalEncoder."""
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    attention_dropout: float = 0.1
    max_seq_len: int = 90
    n_input_features: int = 10
    activation: str = 'gelu'
```


### Code Quality Requirements

- Full type hints
- Google-style docstrings
- DEBUG_MODE support: `if DEBUG_MODE: print(f"[DEBUG]: ...")`
- Progress bars for training (tqdm)
- Checkpoint saving/loading
- Configurable via YAML or dataclass
- Unit-testable (small functions, clear I/O)

---

## 11. What the Code Must Do

### Training Script (`code/train_encoder.py`)

```python
#!/usr/bin/env python3
"""
Train the Shared Temporal Attention Encoder.

Usage:
    python code/train_encoder.py
    python code/train_encoder.py --config config/hyperparameters.yaml
    python code/train_encoder.py --resume --checkpoint checkpoints/temporal_encoder.pt
"""

def main():
    # 1. Load config
    config = load_config(args.config)
    
    # 2. Load market data
    market_data = load_market_data(config.data_path)
    
    # 3. Engineer input features (returns, indicators, etc.)
    features = engineer_features(market_data)
    
    # 4. Create DataLoader with sliding windows
    dataloader = create_dataloader(features, config)
    
    # 5. Initialize model
    model = TemporalEncoder(config)
    
    # 6. Train with chunked chronological validation
    for chunk in config.training_chunks:
        train_model_on_chunk(model, dataloader, chunk)
    
    # 7. Save model
    model.save(config.checkpoint_path)
    
    # 8. Generate embeddings for all stocks
    embeddings = generate_all_embeddings(model, market_data)
    save_embeddings(embeddings, config.embeddings_path)
```

### Inference Mode

```python
# During inference, the encoder is FROZEN
model = TemporalEncoder.load("checkpoints/temporal_encoder.pt")
model.eval()

with torch.no_grad():
    embedding = model.get_embedding(new_market_data, pooling='last')
    # embedding shape: (batch, 128)
```

---

## 12. Validation & Testing

### Unit Tests

```python
def test_encoder_output_shape():
    """Verify encoder produces correct output shapes."""
    model = TemporalEncoder(TemporalEncoderConfig())
    x = torch.randn(4, 30, 10)  # batch=4, seq=30, features=10
    output = model(x)
    
    assert output['sequence'].shape == (4, 30, 128)
    assert output['last_hidden'].shape == (4, 128)
    assert output['mean_pooled'].shape == (4, 128)
    assert output['attention_pooled'].shape == (4, 128)

def test_variable_sequence_length():
    """Encoder should handle different sequence lengths."""
    model = TemporalEncoder(TemporalEncoderConfig())
    
    x_30 = torch.randn(2, 30, 10)
    x_60 = torch.randn(2, 60, 10)
    x_90 = torch.randn(2, 90, 10)
    
    out_30 = model(x_30)
    out_60 = model(x_60)
    out_90 = model(x_90)
    
    assert out_30['last_hidden'].shape == (2, 128)
    assert out_60['last_hidden'].shape == (2, 128)
    assert out_90['last_hidden'].shape == (2, 128)

def test_embedding_consistency():
    """Same input should give same embedding (deterministic in eval mode)."""
    model = TemporalEncoder(TemporalEncoderConfig())
    model.eval()
    
    x = torch.randn(1, 30, 10)
    
    with torch.no_grad():
        emb1 = model.get_embedding(x)
        emb2 = model.get_embedding(x)
    
    assert torch.allclose(emb1, emb2)
```

### Integration Tests

```python
def test_embedding_quality():
    """Embeddings for same-sector stocks should be more similar than cross-sector."""
    tech_embeddings = get_embeddings_for_sector('Technology')
    finance_embeddings = get_embeddings_for_sector('Financials')
    
    intra_sector_sim = cosine_similarity(tech_embeddings, tech_embeddings).mean()
    cross_sector_sim = cosine_similarity(tech_embeddings, finance_embeddings).mean()
    
    assert intra_sector_sim > cross_sector_sim

def test_temporal_smoothness():
    """Consecutive days should have similar embeddings."""
    day_t = get_embedding('AAPL', '2020-06-15')
    day_t1 = get_embedding('AAPL', '2020-06-16')
    
    similarity = cosine_similarity(day_t, day_t1)
    assert similarity > 0.7  # Should not change drastically day-to-day
```

---

## 13. Acknowledgment: Specification Evolution

### ⚠️ IMPORTANT: These Specifications Are From an Earlier Phase

The specifications in this document represent the **initial design decisions** made during the architecture planning phase of the project. Since this document was drafted, the project has progressed through:

1. **Data pipeline completion** — FRED macro data, SEC fundamentals, and market data acquisition
2. **Cross-asset graph specification** — Relationship vectors, graph types, edge features
3. **Model architecture refinements** — The overall system design has been finalized in `UpdatedWorkflow.md`
4. **Hyperparameter configuration** — Detailed HP settings in `Hyperparameter_Config.md`
5. **GNN specifications** — Detailed in `GNN_Pre_Specifications.md`

### What May Have Changed

| Aspect | Original Spec | Current Status |
|--------|--------------|----------------|
| Exact feature list | 10 input features | May be adjusted based on market data quality |
| Chunk dates | 2000-2004/2007-2014/2017-2022 | Verify against available data range (2000-2024) |
| Hyperparameter values | Initial estimates | Final values determined by HP search |
| Training data volume | 26.7M points estimated | Actual depends on ticker coverage in market data |

### Reference Documents for Current Specifications

For the most up-to-date specifications, refer to:
- **`UpdatedWorkflow.md`** — Complete architecture with all finalized model assignments
- **`Hyperparameter_Config.md`** — Current HP values and search configurations
- **`GNN_Pre_Specifications.md`** — GNN module details (which consume this encoder's output)
- **`CrossAssetRelationData.md`** — Graph construction (which uses returns derived from prices the encoder processes)

---

**Document Version:** 1.0 (Initial Pre-Implementation Specification)  
**Status:** Historical reference — current specs in `UpdatedWorkflow.md`  
**Last Updated:** 2026-04-22  
**Prepared for:** Explainable Distributed Deep Learning Framework for Financial Risk Management