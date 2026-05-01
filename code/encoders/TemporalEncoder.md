# Shared Temporal Attention Encoder

## 1. Document Purpose

It should be treated as the reference for:

- why the Temporal Encoder exists,
- what data it consumes,
- how the model is structured,
- how it is trained,
- how embeddings are produced,
- how manifests align embeddings with ticker/date rows,
- how downstream modules consume its outputs,
- how XAI is produced,
- and how to validate the finished module.

The Temporal Encoder is one of the most important upstream modules in the project because its embeddings feed multiple downstream risk and analyst models. If its outputs are missing, misaligned, or low quality, the Technical Analyst, Volatility Model, Drawdown Model, and Regime Detection module are all affected.

---

## 2. Role in the Full Architecture

The Temporal Encoder belongs to the **encoder layer** of the system.

```text
INPUT DATA FAMILY
└── Time-Series Market Data
    └── features_temporal.csv
        └── Shared Temporal Attention Encoder
            ├── Technical Analyst
            ├── Volatility Model
            ├── Drawdown Risk Model
            └── MTGNN Regime Detection
```

The encoder does **not** make a trading decision by itself. Its job is to convert a rolling 30-day market-feature sequence into a dense representation of recent price behaviour.

The output embedding is then reused by downstream modules, which avoids forcing every model to relearn basic temporal market structure from scratch.

---

## 3. Why This Module Exists

Financial market behaviour is sequential. A single daily row is not enough to understand:

- trend continuation,
- momentum reversal,
- volatility clustering,
- drawdown warning patterns,
- volume shocks,
- regime transitions,
- and broader temporal context.

The Temporal Encoder converts a 30-day sequence into a compressed vector that represents the recent state of a ticker. This gives downstream models access to richer context than raw single-day features.

The project intentionally uses an **attention-based encoder** rather than making a GNN the main time-series encoder. GNNs are reserved for correlation/contagion and regime graph modules. This preserves clean architectural boundaries:

```text
Temporal Encoder  → learns per-ticker temporal behaviour
StemGNN / MTGNN   → learns cross-asset or graph-based structure
```

---

## 4. Design Decision: Why Transformer-Based Attention

The main alternatives were LSTM, CNN, and attention-based models.

| Candidate | Strength | Limitation for this project |
|---|---|---|
| LSTM | Good sequential baseline | Processes sequentially, can be slow, can struggle with long-range dependency and parallel GPU utilisation |
| CNN | Efficient local pattern extraction | Fixed receptive field; does not dynamically decide which days matter |
| Transformer Encoder | Parallel, attention-based, flexible dependency modelling | Requires careful regularisation and enough data |

The project chose a **Transformer-style temporal encoder** because it can learn which days in a window matter most and can process time steps in parallel on GPU.

This is especially useful for financial data because relevant signals may not always be the most recent row. An abnormal volume spike, a volatility cluster, or a momentum reversal several days earlier may matter more than yesterday’s movement.

---

## 5. Input Data

### 5.1 Source file

The encoder consumes the final engineered market-feature file:

```text
data/yFinance/processed/features_temporal.csv
```

This file is created by the market data pipeline after yFinance/Stooq/Kaggle filling, calendar alignment, missing-value handling, and no-leakage feature engineering.

### 5.2 Required columns

The active model uses the following 10 engineered features:

| Feature | Meaning | Use |
|---|---|---|
| `log_return` | Daily log return | Basic return/momentum movement |
| `vol_5d` | 5-day realised volatility | Short-term instability |
| `vol_21d` | 21-day realised volatility | Medium-term instability |
| `rsi_14` | 14-day RSI | Momentum/overbought/oversold signal |
| `macd_hist` | MACD histogram | Trend/momentum divergence |
| `bb_pos` | Bollinger band position | Relative price location in band |
| `volume_ratio` | Volume relative to recent average | Volume abnormality |
| `hl_ratio` | High-low range ratio | Intraday range / volatility proxy |
| `price_pos` | Price position indicator | Relative price state |
| `spy_corr_63d` | Rolling correlation with SPY proxy | Market co-movement/context |

The code checks these fields during inspection. Missing features indicate the market pipeline has not been completed correctly.

### 5.3 Input shape

Each training or embedding sample is a rolling sequence:

```text
(batch_size, seq_len, n_features)
```

The final active configuration uses:

```text
seq_len = 30
n_features = 10
```

So a typical batch has shape:

```text
(batch_size, 30, 10)
```

Each sample corresponds to one ticker and one end date. The sequence covers the 30 trading days ending on that date.

---

## 6. Chronological Split Design

The project uses chronological chunks to avoid look-ahead bias.

| Chunk | Training period | Validation period | Test period | Purpose |
|---|---:|---:|---:|---|
| Chunk 1 | 2000–2004 | 2005 | 2006 | Early historical period |
| Chunk 2 | 2007–2014 | 2015 | 2016 | Crisis/post-crisis period |
| Chunk 3 | 2017–2022 | 2023 | 2024 | Recent market period |

This split structure is important because financial data is time ordered. Random train/test splitting would leak future market conditions into training.

The encoder must fit its normalisation and model only on the training portion for the relevant chunk, then apply that fitted state to validation and test embeddings.

---

## 7. Model Architecture

### 7.1 High-level structure

```text
Input sequence: (batch, 30, 10)
    │
    ├── Linear input projection: 10 → d_model
    │
    ├── Sinusoidal positional encoding
    │
    ├── Transformer Encoder layers
    │
    ├── Pooling
    │   ├── last_hidden
    │   ├── mean_pooled
    │   └── attention_pooled
    │
    └── Temporal embedding: (batch, d_model)
```

### 7.2 Input projection

The raw 10-dimensional feature vector at each time step is projected into the model dimension:

```python
self.input_projection = nn.Linear(n_input_features, d_model)
```

This allows the transformer to operate in a richer hidden space.

### 7.3 Positional encoding

Because transformer attention does not inherently know sequence order, sinusoidal positional encoding is added to the projected inputs.

The positional encoding lets the model distinguish early, middle, and recent days inside the 30-day window.

### 7.4 Transformer encoder

The encoder uses PyTorch’s `nn.TransformerEncoderLayer` with:

- `batch_first=True`,
- GELU activation,
- pre-layer normalisation via `norm_first=True`,
- multi-head self-attention,
- feed-forward layers,
- dropout,
- residual connections inside the transformer block.

Important architectural clarification:

The project’s “no residual shortcuts between major modules” rule does **not** forbid residual connections inside a Transformer block. Transformer residuals are part of the standard internal mechanism required for stable deep attention training.

### 7.5 Pooling outputs

The model returns a dictionary rather than a single tensor:

```python
{
    "sequence": x,
    "last_hidden": last_hidden,
    "mean_pooled": mean_pooled,
    "attention_pooled": attn_pooled,
}
```

This design makes the encoder flexible for different downstream modules.

| Output | Shape | Meaning |
|---|---:|---|
| `sequence` | `(batch, seq_len, d_model)` | Full hidden sequence |
| `last_hidden` | `(batch, d_model)` | Representation of the latest state |
| `mean_pooled` | `(batch, d_model)` | Average sequence representation |
| `attention_pooled` | `(batch, d_model)` | Learned weighted representation |

The final operational embeddings used by downstream modules are 256-dimensional after HPO selected `d_model=256` for later chunks.

---

## 8. Training Objective

The encoder is trained with a self-supervised masked prediction task.

### 8.1 Masked temporal reconstruction

Random time steps are masked and the model is trained to reconstruct the original feature values at masked positions.

```text
Original sequence:
[t1, t2, t3, ..., t30]

Masked input:
[t1, 0, t3, ..., t30]

Target:
recover the true feature vector at the masked position
```

The loss is mean squared error on masked positions only:

```text
loss = MSE(predicted_masked_values, true_masked_values)
```

This makes the encoder learn temporal structure without needing hand-written supervised labels.

### 8.2 Why self-supervised training is suitable here

The encoder is shared across multiple downstream tasks, so it should learn a general-purpose temporal representation rather than optimising directly for one task only.

Self-supervised masked reconstruction helps the encoder learn:

- return dynamics,
- volatility persistence,
- momentum/mean-reversion structure,
- volume-price relationships,
- and cross-feature interactions inside a 30-day market window.

---

## 9. Normalisation and Leakage Control

The encoder uses feature normalisation so that features with large numeric ranges do not dominate the model.

### 9.1 Normalisation rule

The feature normaliser stores:

```text
mean(feature)
std(feature)
```

and transforms:

```text
x_normalised = (x - mean) / std
```

### 9.2 No-leakage rule

Normalisation must be fitted on the **training split only** for a chunk.

Validation and test embeddings must reuse the training-fitted normaliser. They must not fit their own normalisers using validation/test data because that would leak future distribution information into inference.

In the final operational run, train-only normalisers were saved under each chunk model folder, for example:

```text
outputs/models/TemporalEncoder/chunk2/normalizer.npz
outputs/models/TemporalEncoder/chunk3/normalizer.npz
```

---

## 10. Hyperparameter Optimisation

The Temporal Encoder uses Optuna TPE search before final training.

### 10.1 Why HPO matters

The encoder is upstream of many models. Poor temporal embeddings reduce the quality of:

- Technical Analyst,
- Volatility Model,
- Drawdown Model,
- Regime Detection,
- Quantitative Analyst,
- Position Sizing,
- Fusion Engine.

HPO is therefore not optional for the thesis-quality version.

### 10.2 HPO search space

The code searches over:

| Parameter | Search space / type |
|---|---|
| `n_layers` | 2 to 6 |
| `n_heads` | 2, 4, 8 |
| `d_model` | 64, 128, 256 |
| `dropout` | continuous range |
| `attention_dropout` | continuous range |
| `learning_rate` | log-scale range |
| `weight_decay` | log-scale range |
| `warmup_steps` | candidate values |
| `batch_size` | candidate values |

The best parameters are saved under:

```text
outputs/codeResults/TemporalEncoder/hpo/best_params_chunk1.json
outputs/codeResults/TemporalEncoder/hpo/best_params_chunk2.json
outputs/codeResults/TemporalEncoder/hpo/best_params_chunk3.json
```

### 10.3 Known final HPO examples from the production run

Chunk 2 used a large 256-dimensional configuration:

```json
{
  "n_layers": 6,
  "n_heads": 8,
  "d_model": 256,
  "dropout": 0.17854162005663654,
  "attention_dropout": 0.19488416358230604,
  "learning_rate": 0.0004982755066410893,
  "weight_decay": 0.0002305298448157034,
  "warmup_steps": 1000,
  "batch_size": 32
}
```

Chunk 3 used:

```json
{
  "n_layers": 4,
  "n_heads": 4,
  "d_model": 256,
  "dropout": 0.12580726647987625,
  "attention_dropout": 0.1243864375089663,
  "learning_rate": 0.00045877596861583427,
  "weight_decay": 0.000009225880372153381,
  "warmup_steps": 1000,
  "batch_size": 32
}
```

The exact active values should always be checked from the saved HPO JSON files in the repository.

---

## 11. Training, Resume, and Checkpointing

### 11.1 Checkpoint files

Each chunk stores model checkpoints under:

```text
outputs/models/TemporalEncoder/chunk{n}/
```

Expected files include:

```text
best_model.pt
latest_model.pt
training_history.csv
training_summary.json
effective_config.json
normalizer.npz
model_freezed/model.pt
model_unfreezed/model.pt
```

### 11.2 Resume behaviour

The training function supports resuming from:

```text
latest_model.pt
training_history.csv
```

If interrupted, the next run can continue from the last saved epoch. The training history is saved incrementally so progress is not lost when a remote session disconnects.

### 11.3 Practical runtime lesson

The Temporal Encoder training stage was the slowest part of the downstream setup because full training over millions of 30-day windows is expensive. In practice, once the validation loss had plateaued and a usable `best_model.pt` existed, embeddings could be generated from the best checkpoint.

This was especially important for Chunk 2 and Chunk 3, where the project needed embeddings urgently to unblock downstream risk modules.

The key practical rule is:

```text
If best_model.pt exists, validation loss has stabilised, and embeddings are the blocking dependency, generate embeddings from best_model.pt rather than waiting for unnecessary extra epochs.
```

---

## 12. Embedding Generation

### 12.1 Output directory

Embeddings are saved under:

```text
outputs/embeddings/TemporalEncoder/
```

### 12.2 Output files

For every chunk and split, the encoder produces:

```text
chunk{n}_{split}_embeddings.npy
chunk{n}_{split}_manifest.csv
```

Example:

```text
outputs/embeddings/TemporalEncoder/chunk2_train_embeddings.npy
outputs/embeddings/TemporalEncoder/chunk2_train_manifest.csv
```

### 12.3 Final production embedding shapes

The completed production run produced finite 256-dimensional embeddings for all chunks.

| Chunk | Split | Embedding shape | Manifest shape |
|---|---|---:|---:|
| Chunk 1 | train | `(3,065,000, 256)` | `(3,065,000, 2)` |
| Chunk 1 | val | `(555,000, 256)` | `(555,000, 2)` |
| Chunk 1 | test | `(552,500, 256)` | `(552,500, 2)` |
| Chunk 2 | train | `(4,960,000, 256)` | `(4,960,000, 2)` |
| Chunk 2 | val | `(555,000, 256)` | `(555,000, 2)` |
| Chunk 2 | test | `(555,000, 256)` | `(555,000, 2)` |
| Chunk 3 | train | `(3,700,000, 256)` | `(3,700,000, 2)` |
| Chunk 3 | val | `(550,000, 256)` | `(550,000, 2)` |
| Chunk 3 | test | `(547,500, 256)` | `(547,500, 2)` |

All sampled embeddings were verified as finite during the final audit.

---

## 13. Manifest Alignment

### 13.1 Why manifests are required

The `.npy` embedding arrays contain only numeric vectors. Downstream modules need to know:

```text
embedding row i → which ticker and which date?
```

This mapping is stored in manifest files:

```text
chunk{n}_{split}_manifest.csv
```

The minimal manifest columns are:

```text
ticker,date
```

Some manifest-building tools may also include:

```text
seq_start,seq_end
```

### 13.2 Manifest generation logic

The manifest reconstructs the same rolling-window order used by the `MarketSequenceDataset`:

```text
For each ticker:
    sort by date
    build 30-day windows
    assign the embedding date to the final date of the window
```

The helper script:

```text
code/encoders/build_embedding_manifest.py
```

exists to rebuild manifest files if embeddings already exist but row metadata is missing.

### 13.3 Alignment validation rule

For every split:

```text
len(embeddings) == len(manifest)
```

If this is false, downstream model training must not proceed until alignment is fixed.

---

## 14. XAI Outputs

The Temporal Encoder contributes XAI at the embedding-generation stage.

### 14.1 Attention XAI

The attention pooling mechanism identifies which time steps in the rolling window mattered most for the embedding.

Expected outputs:

```text
outputs/results/TemporalEncoder/xai/chunk{n}_{split}_attention_weights.npy
outputs/results/TemporalEncoder/xai/chunk{n}_{split}_attention_weights.csv
```

These files help answer:

```text
Which days in the 30-day window were most important for the temporal representation?
```

### 14.2 Gradient feature importance

Gradient-based importance is computed over a small sample of embeddings.

Expected outputs:

```text
outputs/results/TemporalEncoder/xai/chunk{n}_{split}_feature_importance.npy
outputs/results/TemporalEncoder/xai/chunk{n}_{split}_feature_importance.csv
```

These files help answer:

```text
Which engineered market features had the strongest influence on the embedding?
```

### 14.3 XAI limitations

Temporal Encoder XAI should be interpreted as representation-level explanation, not final decision explanation.

The encoder explains what shaped the embedding. It does not explain the final Buy/Hold/Sell decision. Final explanation is produced later by module-level XAI, position sizing XAI, quantitative/qualitative synthesis, and fusion explanation.

---

## 15. Downstream Consumers

### 15.1 Technical Analyst

Consumes Temporal Encoder embeddings and learns directional technical scores:

```text
trend_score
momentum_score
timing_confidence
```

### 15.2 Volatility Model

Consumes embeddings as learned market-state features and predicts future volatility outputs used by the risk engine and position sizing.

### 15.3 Drawdown Risk Model

Consumes embeddings to estimate expected drawdown and related downside path-risk signals.

### 15.4 Regime Detection

Consumes temporal embeddings, together with FinBERT/text and macro context, to help classify market regime state.

---

## 16. File Structure

### 16.1 Code files

```text
code/encoders/temporal_encoder.py
code/encoders/build_embedding_manifest.py
```

### 16.2 Input file

```text
data/yFinance/processed/features_temporal.csv
```

### 16.3 HPO files

```text
outputs/codeResults/TemporalEncoder/hpo/best_params_chunk1.json
outputs/codeResults/TemporalEncoder/hpo/best_params_chunk2.json
outputs/codeResults/TemporalEncoder/hpo/best_params_chunk3.json
```

### 16.4 Model files

```text
outputs/models/TemporalEncoder/chunk1/
outputs/models/TemporalEncoder/chunk2/
outputs/models/TemporalEncoder/chunk3/
```

### 16.5 Embedding files

```text
outputs/embeddings/TemporalEncoder/chunk1_train_embeddings.npy
outputs/embeddings/TemporalEncoder/chunk1_train_manifest.csv
...
outputs/embeddings/TemporalEncoder/chunk3_test_embeddings.npy
outputs/embeddings/TemporalEncoder/chunk3_test_manifest.csv
```

### 16.6 XAI files

```text
outputs/results/TemporalEncoder/xai/
```

---

## 17. CLI Commands

All commands below are single-line commands to match the project execution preference.

### 17.1 Inspect data

```bash
cd ~/fin-glassbox && python code/encoders/temporal_encoder.py inspect --repo-root .
```

### 17.2 Run HPO for a chunk

```bash
cd ~/fin-glassbox && python code/encoders/temporal_encoder.py hpo --repo-root . --chunk 1 --device cuda
```

### 17.3 Train best model for one chunk

```bash
cd ~/fin-glassbox && python code/encoders/temporal_encoder.py train-best --repo-root . --chunk 1 --device cuda
```

### 17.4 Train all chunks

```bash
cd ~/fin-glassbox && python code/encoders/temporal_encoder.py train-best --repo-root . --chunk 1 --device cuda && python code/encoders/temporal_encoder.py train-best --repo-root . --chunk 2 --device cuda && python code/encoders/temporal_encoder.py train-best --repo-root . --chunk 3 --device cuda
```

### 17.5 Fast embedding generation for one chunk

If the active code version supports performance flags, use a large embedding batch:

```bash
cd ~/fin-glassbox && python code/encoders/temporal_encoder.py embed --chunk 1 --split train --device cuda --batch-size 4096 --num-workers 8 --prefetch-factor 4 && python code/encoders/temporal_encoder.py embed --chunk 1 --split val --device cuda --batch-size 4096 --num-workers 8 --prefetch-factor 4 && python code/encoders/temporal_encoder.py embed --chunk 1 --split test --device cuda --batch-size 4096 --num-workers 8 --prefetch-factor 4
```

### 17.6 Build or repair manifests

```bash
cd ~/fin-glassbox && python code/encoders/build_embedding_manifest.py
```

### 17.7 Verify all embedding outputs

```bash
cd ~/fin-glassbox && python -c "import numpy as np,pandas as pd,pathlib; base=pathlib.Path('outputs/embeddings/TemporalEncoder'); files=['chunk1_train','chunk1_val','chunk1_test','chunk2_train','chunk2_val','chunk2_test','chunk3_train','chunk3_val','chunk3_test']; [print(f, np.load(base/f'{f}_embeddings.npy',mmap_mode='r').shape, pd.read_csv(base/f'{f}_manifest.csv').shape, 'finite_sample=', float(np.isfinite(np.load(base/f'{f}_embeddings.npy',mmap_mode='r')[:10000]).mean())) for f in files]"
```

---

## 18. Validation Checklist

The Temporal Encoder is considered complete only if all of the following are true:

| Check | Required result |
|---|---|
| `features_temporal.csv` exists | Yes |
| All 10 input features exist | Yes |
| Chunk HPO files exist | Yes, for chunks used in final system |
| `best_model.pt` exists | Yes, per chunk |
| `model_freezed/model.pt` exists | Yes, per chunk |
| `normalizer.npz` exists | Yes, per chunk |
| Train/val/test embeddings exist | Yes, per chunk |
| Train/val/test manifests exist | Yes, per chunk |
| Embedding rows equal manifest rows | Yes |
| Embedding finite sample ratio | 1.0 expected |
| XAI sample files exist | Strongly preferred |

---

## 19. Troubleshooting

### 19.1 Training is too slow

This happened during production. The practical fix was to use the best available checkpoint and generate embeddings directly once validation had plateaued.

Also check:

- GPU utilisation,
- DataLoader workers,
- batch size,
- whether normalisation is repeatedly moving tensors to GPU inside the training loop,
- whether the dataset is being rebuilt unnecessarily,
- whether HDD I/O is bottlenecking the run.

### 19.2 Embeddings exist but downstream modules fail

Check manifest alignment:

```text
len(embeddings) must equal len(manifest)
```

Also verify manifest columns include ticker/date and that dates are parseable.

### 19.3 Normaliser missing

If validation/test embedding uses a split-fitted normaliser, leakage can occur. Rebuild or copy the train-only normaliser for that chunk.

### 19.4 Chunk 2 or Chunk 3 missing

The downstream modules require all chunks eventually. If only Chunk 1 exists, Temporal Encoder is not complete for final backtesting.

---

## 20. Final Status

At the current final project state, the Temporal Encoder is complete for all three chunks:

```text
Chunk 1 train/val/test embeddings + manifests: complete
Chunk 2 train/val/test embeddings + manifests: complete
Chunk 3 train/val/test embeddings + manifests: complete
```

This unblocked the rest of the risk engine and analyst stack.

---

## 21. Summary

The Temporal Encoder is the project’s shared market-sequence representation model. It converts 30-day sequences of engineered market features into 256-dimensional embeddings used by technical, volatility, drawdown, and regime modules.

Its importance comes from being upstream of several models. The final system depends on it being:

- chronologically trained,
- leakage-safe,
- HPO-tuned,
- checkpointed,
- resumable,
- manifest-aligned,
- finite and validated,
- and explainable through attention and gradient XAI.

This module is now a completed core encoder component in the `fin-glassbox` architecture.
