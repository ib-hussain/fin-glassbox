# Textual Analysts Documentation

**Project:** `fin-glassbox` — Explainable Distributed Deep Learning Framework for Financial Risk Management  
**Module family:** Textual Analyst Layer  
**Files covered:**

```text
code/analysts/text_market_label_builder.py
code/analysts/sentiment_analyst.py
code/analysts/news_analyst.py
```

**Current architecture note:** The Fundamental Encoder and Fundamental Analyst have been removed from the workflow. The textual analyst layer documented here uses only the FinBERT text stream, market-data-derived labels, and filing metadata. It does not depend on a fundamental embedding stream.

**British English convention:** All project-facing documentation should use British English.

---

## 1. Purpose of this document

This document explains the complete supervised-data construction and modelling workflow for the textual analyst modules. It records:

1. what data the analysts consume;
2. how the supervised labels were generated;
3. how the label builder joins SEC text metadata to market outcomes;
4. how the Sentiment Analyst works;
5. how the News Analyst works;
6. how hyperparameter search, checkpointing, and prediction exports are handled;
7. what outputs are expected from each file;
8. which commands should be used for inspection, HPO, training, and prediction.

The core principle is simple:

```text
FinBERT provides real 256-dimensional text embeddings.
Market data provides real future-outcome labels.
Sentiment Analyst and News Analyst learn supervised mappings from text embeddings to future market-derived targets.
```

No dummy labels, synthetic embeddings, or fake supervised data are used.

---

## 2. Current textual-analysis workflow

The textual analyst layer now follows this pipeline:

```text
SEC filing text chunks
        ↓
FinBERT Text Encoder
        ↓
256-dimensional chunk-level text embeddings + row-aligned metadata
        ↓
text_market_label_builder.py
        ↓
real market-derived supervised labels
        ↓
Sentiment Analyst and News Analyst
        ↓
textual analyst scores, uncertainty, attention explanations, and analyst embeddings
```

The data flow is:

```text
outputs/embeddings/FinBERT/chunk{N}_{split}_embeddings.npy
outputs/embeddings/FinBERT/chunk{N}_{split}_metadata.csv
        ↓
outputs/results/analysts/labels/text_market_labels_chunk{N}_{split}.csv
        ↓
outputs/models/analysts/sentiment/chunk{N}/best.pt
outputs/models/analysts/news/chunk{N}/best.pt
        ↓
outputs/results/analysts/sentiment/*
outputs/results/analysts/news/*
outputs/embeddings/analysts/sentiment/*
outputs/embeddings/analysts/news/*
```

---

## 3. Important architecture correction: no Fundamental Analyst

Earlier project specifications included:

```text
Temporal embedding:     128 dimensions
Text embedding:         256 dimensions
Fundamental embedding:  128 dimensions
```

That design has now been changed. The Fundamental Encoder and Fundamental Analyst are removed. The textual analyst modules should not assume that fundamental embeddings, fundamental labels, or a fundamental analyst output exist.

The textual analyst layer remains valid because it only requires:

```text
FinBERT 256-dimensional embeddings
FinBERT metadata CSVs
CIK → ticker mapping
market returns panel
market-derived supervised labels
```

The SEC CIK-to-ticker mapping is still allowed and required as an identity bridge. Using that mapping does not reintroduce a Fundamental Analyst.

---

## 4. Source data used by the textual analysts

### 4.1 FinBERT text embeddings

The FinBERT encoder produces chunk-level embeddings from SEC filing text. The final downstream text embedding shape is:

```text
single row: (256,)
batch:      (batch_size, 256)
dtype:      float32
format:     .npy
```

The expected embedding files are:

```text
outputs/embeddings/FinBERT/chunk1_train_embeddings.npy
outputs/embeddings/FinBERT/chunk1_val_embeddings.npy
outputs/embeddings/FinBERT/chunk1_test_embeddings.npy
outputs/embeddings/FinBERT/chunk2_train_embeddings.npy
outputs/embeddings/FinBERT/chunk2_val_embeddings.npy
outputs/embeddings/FinBERT/chunk2_test_embeddings.npy
outputs/embeddings/FinBERT/chunk3_train_embeddings.npy
outputs/embeddings/FinBERT/chunk3_val_embeddings.npy
outputs/embeddings/FinBERT/chunk3_test_embeddings.npy
```

Each embedding matrix must have a matching metadata CSV:

```text
outputs/embeddings/FinBERT/chunk1_train_metadata.csv
outputs/embeddings/FinBERT/chunk1_val_metadata.csv
outputs/embeddings/FinBERT/chunk1_test_metadata.csv
outputs/embeddings/FinBERT/chunk2_train_metadata.csv
outputs/embeddings/FinBERT/chunk2_val_metadata.csv
outputs/embeddings/FinBERT/chunk2_test_metadata.csv
outputs/embeddings/FinBERT/chunk3_train_metadata.csv
outputs/embeddings/FinBERT/chunk3_val_metadata.csv
outputs/embeddings/FinBERT/chunk3_test_metadata.csv
```

The expected metadata columns are:

```text
chunk_id
doc_id
year
form_type
cik
filing_date
accession
source_name
chunk_index
word_count
```

The embedding matrix and metadata file must be row-aligned. Row `i` in the `.npy` matrix must describe the same SEC text chunk as row `i` in the metadata CSV.

### 4.2 FinBERT split design

The textual analysts preserve the same chronological split structure as the FinBERT encoder:

| Chunk | Train years | Validation year | Test year |
|---:|---|---:|---:|
| 1 | 2000–2004 | 2005 | 2006 |
| 2 | 2007–2014 | 2015 | 2016 |
| 3 | 2017–2022 | 2023 | 2024 |

Each chunk is treated as a separate chronological experiment. The analysts do not train on validation or test rows.

### 4.3 Market data

The label builder uses the completed market-data pipeline, especially the wide returns panel:

```text
data/yFinance/processed/returns_panel_wide.csv
```

The market data context is:

| Item | Value |
|---|---:|
| Tickers | 2,500 |
| Trading days | 6,285 return rows |
| Date span | 2000-01-04 to 2024-12-27 for returns panel |
| Return type | log return |
| Missing values | 0.0% in final returns panel |
| Benchmark default | SPY |

The label builder uses this file to calculate forward stock returns, forward benchmark returns, excess returns, future volatility, volatility ratios, and forward drawdown.

### 4.4 CIK-to-ticker mapping

The label builder requires:

```text
data/sec_edgar/processed/cleaned/cik_ticker_map_cleaned.csv
```

This file maps SEC Central Index Keys to tickers. The relevant columns are:

```text
cik
primary_ticker
```

The label builder normalises CIK values and maps each FinBERT metadata row to a ticker. Only rows whose mapped ticker exists in the market returns panel can receive market-derived targets.

---

## 5. Why supervised labels had to be generated

FinBERT itself does not produce market labels. The current FinBERT training stage is domain-adaptive Masked Language Modelling. That means the encoder learns SEC filing language and produces better text representations, but it does not learn whether a filing was followed by positive returns, negative returns, volatility spikes, or drawdowns.

Therefore, the supervised label CSVs had to be created separately from market data.

The logic is:

```text
FinBERT embedding = input feature X
future market outcome = supervised target y
```

For each SEC filing text chunk:

```text
CIK + filing_date + doc_id
        ↓
CIK → ticker
        ↓
first trading day after filing_date
        ↓
future stock return and future benchmark return
        ↓
future excess return, future volatility, future drawdown
        ↓
sentiment/news/risk targets
```

This is not leakage because the future market outcome is used only as the training target. The model input remains the text embedding and metadata available at the filing date.

---

## 6. Label generation file: `text_market_label_builder.py`

### 6.1 Purpose

`code/analysts/text_market_label_builder.py` creates real supervised labels for the textual analysts. It reads FinBERT metadata, joins each row to a market ticker, calculates future outcome labels from the returns panel, and writes row-aligned label CSVs.

The file does not train a model. It only constructs supervised training data.

### 6.2 Main inputs

| Input | Default path | Purpose |
|---|---|---|
| FinBERT metadata | `outputs/embeddings/FinBERT/chunk{N}_{split}_metadata.csv` | SEC chunk identity and filing date |
| FinBERT embeddings | `outputs/embeddings/FinBERT/chunk{N}_{split}_embeddings.npy` | Optional contract validation; not modified |
| CIK map | `data/sec_edgar/processed/cleaned/cik_ticker_map_cleaned.csv` | Maps SEC CIK to ticker |
| Market returns | `data/yFinance/processed/returns_panel_wide.csv` | Computes future market outcomes |
| Benchmark ticker | `SPY` by default | Market-relative excess return |

### 6.3 Main outputs

The label builder writes nine split-level CSV files:

```text
outputs/results/analysts/labels/text_market_labels_chunk1_train.csv
outputs/results/analysts/labels/text_market_labels_chunk1_val.csv
outputs/results/analysts/labels/text_market_labels_chunk1_test.csv
outputs/results/analysts/labels/text_market_labels_chunk2_train.csv
outputs/results/analysts/labels/text_market_labels_chunk2_val.csv
outputs/results/analysts/labels/text_market_labels_chunk2_test.csv
outputs/results/analysts/labels/text_market_labels_chunk3_train.csv
outputs/results/analysts/labels/text_market_labels_chunk3_val.csv
outputs/results/analysts/labels/text_market_labels_chunk3_test.csv
```

It also writes per-chunk threshold files:

```text
outputs/results/analysts/labels/text_market_label_thresholds_chunk1.json
outputs/results/analysts/labels/text_market_label_thresholds_chunk2.json
outputs/results/analysts/labels/text_market_label_thresholds_chunk3.json
```

When `--write-combined` is used, it also writes:

```text
outputs/results/analysts/labels/text_market_labels_all.csv
```

It writes run/configuration records into:

```text
outputs/codeResults/analysts/labels/
```

### 6.4 Default label horizons

The default horizon list is:

```text
1, 5, 10, 20, 30 trading days
```

Default primary horizons are:

| Target family | Default horizon |
|---|---:|
| Sentiment target | 10 trading days |
| News/event impact target | 10 trading days |
| Risk target | 30 trading days |
| Trailing volatility window | 60 trading days |

### 6.5 Event start date

The label builder does not use the filing date itself as the first market-return day. It selects the first trading day after the filing date:

```text
event_start_idx = first returns-panel date strictly after filing_date
```

This avoids assuming that an SEC filing was available before market close. If intraday filing timestamps are not available, using the next trading day is the safer anti-leakage rule.

### 6.6 Forward return columns

For each horizon `h`, the builder calculates:

```text
future_log_return_{h}d
benchmark_log_return_{h}d
future_simple_return_{h}d
benchmark_simple_return_{h}d
future_excess_log_return_{h}d
future_excess_simple_return_{h}d
```

The excess return is calculated relative to the benchmark:

```text
future_excess_log_return_h = stock_forward_log_return_h - benchmark_forward_log_return_h
```

The default benchmark is `SPY`.

### 6.7 Volatility and drawdown columns

For the risk horizon, the builder calculates:

```text
future_realised_vol_30d
trailing_realised_vol_60d
future_vol_ratio_30d_vs_trailing_60d
future_max_drawdown_30d
```

Realised volatility is annualised. Drawdown is calculated from the forward path over the risk horizon.

### 6.8 Train-only threshold fitting

Each chronological chunk has its own thresholds. Thresholds are fitted only on that chunk's train split.

Default quantiles:

| Parameter | Default |
|---|---:|
| negative quantile | 0.33 |
| positive quantile | 0.67 |
| high-risk quantile | 0.80 |
| scale quantile | 0.95 |

This means validation/test distributions are never used to choose class thresholds or scaling constants.

### 6.9 Final target columns

The label builder adds these direct training targets:

| Column | Meaning | Range/type |
|---|---|---|
| `sentiment_score_target` | signed scaled excess return | approximately `[-1, 1]` via `tanh` |
| `sentiment_class_target` | bearish/neutral/bullish class | `-1`, `0`, `1` |
| `news_event_impact_target` | signed event impact | approximately `[-1, 1]` via `tanh` |
| `news_importance_target` | absolute event importance | `[0, 1]` |
| `volatility_spike_30d_target` | whether future volatility ratio crosses train threshold | `0` or `1` |
| `drawdown_risk_30d_target` | whether forward drawdown crosses train threshold | `0` or `1` |
| `risk_relevance_target` | combined volatility/drawdown relevance score | `[0, 1]` or NaN if both components missing |
| `primary_excess_return_target` | primary future excess log return | continuous |
| `primary_abs_excess_return_target` | absolute primary excess return | continuous |
| `target_schema_version` | schema tag | `text_market_labels_v1` |

### 6.10 Label availability and missing targets

A row can fail to receive a complete target if:

1. CIK cannot be mapped to a ticker;
2. ticker is not present in the 2,500-ticker market panel;
3. filing date cannot be converted;
4. event start date is too close to the end of the market panel for the required future horizon;
5. trailing volatility history is unavailable near the start of the panel.

The builder preserves rows by default because row alignment with FinBERT embeddings is critical. Training code later filters to valid supervised rows.

### 6.11 Actual label-build result from the current run

The label build completed successfully and wrote all nine split files plus the combined file.

Valid supervised-row counts from the audit were:

| File | Rows | Valid `sentiment_score_target` | Valid `sentiment_class_target` | Valid `news_importance_target` | Valid `risk_relevance_target` |
|---|---:|---:|---:|---:|---:|
| `chunk1_train` | 189,244 | 184,695 | 184,695 | 184,695 | 184,695 |
| `chunk1_val` | 40,000 | 38,694 | 38,694 | 38,694 | 38,694 |
| `chunk1_test` | 40,000 | 39,201 | 39,201 | 39,201 | 39,201 |
| `chunk2_train` | 320,000 | 311,402 | 311,402 | 311,402 | 311,402 |
| `chunk2_val` | 40,000 | 39,301 | 39,301 | 39,301 | 39,301 |
| `chunk2_test` | 40,000 | 39,054 | 39,054 | 39,054 | 39,054 |
| `chunk3_train` | 240,000 | 207,651 | 207,651 | 207,651 | 207,651 |
| `chunk3_val` | 40,000 | 22,008 | 22,008 | 22,008 | 22,008 |
| `chunk3_test` | 40,000 | 36,097 | 36,097 | 36,097 | 36,097 |

The `RuntimeWarning: All-NaN slice encountered` seen during label building was not a fatal error. It occurred when both the volatility-derived score and drawdown-derived score were missing for a row, causing `risk_relevance_target` to remain NaN. The audit confirms that enough valid supervised rows exist for every split.

---

## 7. Sentiment Analyst file: `sentiment_analyst.py`

### 7.1 Purpose

`code/analysts/sentiment_analyst.py` trains a supervised sentiment model from real FinBERT embeddings and real market-derived labels.

The Sentiment Analyst answers:

```text
Given the SEC text embedding for a filing chunk, did the subsequent market reaction behave more bearish, neutral, or bullish relative to the benchmark?
```

It is not a human-labelled sentiment classifier. It is a market-supervised sentiment model.

### 7.2 Input level

The Sentiment Analyst is row-level / chunk-level.

Each training sample is:

```text
FinBERT embedding row: (256,)
optional metadata features: year, word_count, chunk_index, form_type one-hot
label row: same row index in text_market_labels_chunk{N}_{split}.csv
```

The model validates that:

```text
len(label_csv) == embedding_matrix.shape[0]
embedding_matrix.shape[1] == 256
metadata_row_index is sequential if present
years match approved chronological split
```

### 7.3 Required label columns

The file requires at least:

```text
chunk_id
doc_id
year
form_type
cik
filing_date
accession
source_name
chunk_index
word_count
metadata_row_index
split
ticker
ticker_in_market_panel
label_available
sentiment_score_target
sentiment_class_target
```

Rows are eligible for supervised training only if:

```text
label_available is true
sentiment_score_target is not NaN
sentiment_class_target is one of -1, 0, 1
```

### 7.4 Metadata preprocessing

The class `MetadataPreprocessor` fits metadata transformations only on the train split.

Numeric metadata columns:

```text
year
word_count
chunk_index
```

Categorical metadata:

```text
form_type
```

The numeric features are standardised using train-only means and standard deviations. `form_type` is one-hot encoded using a train-only vocabulary with `<UNK>` for unseen forms.

The fitted preprocessor is saved to:

```text
outputs/models/analysts/sentiment/chunk{N}/metadata_preprocessor.json
```

### 7.5 Model architecture

The main class is:

```text
SentimentAnalystMLP
```

Default architecture:

```text
input: 256 + metadata_dim
hidden: 128 → 64
representation: 64
activation: tanh
regularisation: LayerNorm + Dropout
```

The trunk is a sequence of:

```text
Linear
LayerNorm
Tanh
Dropout
```

The model has three heads:

| Head | Output | Activation/loss |
|---|---|---|
| polarity head | `sentiment_score` | `tanh`, MSE against `sentiment_score_target` |
| class head | `class_logits` | cross-entropy against `sentiment_class_target` mapped to 3 classes |
| magnitude head | `magnitude` | `sigmoid`, BCE against `abs(sentiment_score_target)` |

The class mapping is:

```text
-1 → 0
 0 → 1
 1 → 2
```

### 7.6 Default loss composition

The default composite training loss is:

```text
loss = 1.00 * polarity_loss + 0.50 * class_loss + 0.25 * magnitude_loss
```

The weights are HPO-searchable.

### 7.7 HPO design

The Sentiment Analyst is HPO-first. The intended workflow is not fixed-epoch final training. The correct workflow is:

```text
inspect → hpo / hpo-all → train-best / train-best-all → predict / predict-all
```

HPO uses Optuna/TPE with SQLite persistence.

The HPO search covers:

```text
learning_rate
weight_decay
dropout
batch_size
hidden_dims
representation_dim
epochs
early_stop_patience
gradient_clip
class_loss_weight
magnitude_loss_weight
optional metadata feature usage
```

Default search examples include:

```text
learning_rate: 1e-5 to 8e-4
weight_decay: 1e-7 to 5e-3
dropout: 0.05 to 0.35
batch sizes: 256, 512, 1024
hidden architectures: 128,64; 256,128; 256,128,64; 512,256,128; 128,128,64
representation dimensions: 32, 64, 128
HPO epochs: 5 to 25 by default
```

The HPO objective is:

```text
minimise validation composite loss
```

### 7.8 Sentiment checkpoints and outputs

For each chunk:

```text
outputs/models/analysts/sentiment/chunk{N}/latest.pt
outputs/models/analysts/sentiment/chunk{N}/best.pt
outputs/models/analysts/sentiment/chunk{N}/epoch_XXX.pt
outputs/models/analysts/sentiment/chunk{N}/metadata_preprocessor.json
```

Training history:

```text
outputs/results/analysts/sentiment/chunk{N}_training_history.csv
outputs/results/analysts/sentiment/chunk{N}_training_summary.json
```

Prediction outputs:

```text
outputs/results/analysts/sentiment/chunk{N}_{split}_predictions.csv
outputs/results/analysts/sentiment/chunk{N}_{split}_metrics.json
outputs/embeddings/analysts/sentiment/chunk{N}_{split}_sentiment_embeddings.npy
```

The sentiment analyst embedding has default shape:

```text
(N_rows, 64)
```

If HPO selects another `representation_dim`, the shape follows that selected value.

### 7.9 Prediction columns

The prediction CSV includes the original label/metadata columns plus:

```text
predicted_sentiment_score
predicted_sentiment_class_index
predicted_sentiment_class
predicted_sentiment_confidence
predicted_sentiment_uncertainty
predicted_sentiment_magnitude
sentiment_embedding_file
```

---

## 8. News Analyst file: `news_analyst.py`

### 8.1 Purpose

`code/analysts/news_analyst.py` trains a supervised attention-pooling model from real FinBERT embeddings and real market-derived labels.

The News Analyst answers:

```text
Did this filing/document correspond to an important market-moving event?
Was the event impact positive or negative?
Was the event relevant to future risk?
Which chunks inside the document mattered most?
```

The News Analyst is more event-oriented than the Sentiment Analyst.

### 8.2 Input level

FinBERT embeddings are chunk-level, but news/event impact is better treated at document level. Therefore, `news_analyst.py` groups chunk rows by document identity.

The default grouping key uses:

```text
doc_id
accession
cik
filing_date
```

Each document group contains multiple chunk embeddings:

```text
chunk-level embeddings: (num_chunks_in_document, 256)
attention mask:         (num_chunks_in_document,)
metadata features:      document-level features
```

To control memory and runtime, each document is capped by:

```text
max_chunks_per_document = 64 by default
```

If a document has more chunks than the cap, the code selects evenly spaced chunks to preserve document coverage instead of only taking the beginning.

### 8.3 Required data

The News Analyst uses the same embedding and label files as the Sentiment Analyst:

```text
outputs/embeddings/FinBERT/chunk{N}_{split}_embeddings.npy
outputs/results/analysts/labels/text_market_labels_chunk{N}_{split}.csv
```

It validates row alignment before grouping.

### 8.4 News targets

The News Analyst uses:

| Target | Source label column | Meaning |
|---|---|---|
| impact | `news_event_impact_target` | signed event impact in approximately `[-1, 1]` |
| importance | `news_importance_target` | absolute event importance in `[0, 1]` |
| risk relevance | `risk_relevance_target` | future risk relevance in `[0, 1]` |
| volatility spike | `volatility_spike_30d_target` | binary volatility event target |
| drawdown risk | `drawdown_risk_30d_target` | binary drawdown event target |

The code can detect optional volatility/drawdown target column names from the label CSV.

### 8.5 Document-level target aggregation

Because labels were generated per chunk row, but News Analyst trains per document group, document targets are aggregated from rows within each document. The grouping preserves traceability by storing source row indices.

The document-level view is used for model training and prediction. Predictions can still be traced back to the contributing chunk rows through the attention export.

### 8.6 Metadata preprocessing

The class `DocumentMetadataPreprocessor` prepares document-level metadata features. It is fitted only on training groups.

It uses filing-time metadata only, not future outcomes. Examples include:

```text
year
word_count statistics
chunk count
chunk_index statistics
form_type
source_name
```

Train-only fitting avoids leakage.

### 8.7 Model architecture

The main model class is:

```text
NewsAnalystAttentionModel
```

Default architecture:

```text
input embedding dimension: 256
projection dimension d_model: 128
attention heads: 4
self-attention layers: 1
representation dimension: 128
activation in MLP blocks: tanh
regularisation: LayerNorm + Dropout
```

The high-level architecture is:

```text
chunk embeddings
        ↓
input projection 256 → d_model
        ↓
optional TransformerEncoder context layers
        ↓
MultiHeadAttentionPooling
        ↓
document representation
        ↓
metadata fusion
        ↓
MLP trunk with tanh activations
        ↓
output heads
```

### 8.8 Attention pooling

The class `MultiHeadAttentionPooling` learns multiple attention heads over the chunks of a document. This produces:

```text
pooled document representation
attention weights per document chunk
```

The attention weights are exported later for explainability. They indicate which chunks within a filing/document contributed most strongly to the News Analyst decision.

### 8.9 Output heads

The model predicts:

| Head | Output | Activation/loss |
|---|---|---|
| impact head | `event_impact_score` | `tanh`, regression against signed impact |
| importance head | `news_importance_score` | `sigmoid`, BCE/regression-style score against importance |
| risk head | `risk_relevance_score` | `sigmoid`, masked BCE/regression-style score against risk relevance |
| volatility head | `volatility_spike_score` | `sigmoid`, masked BCE when target exists |
| drawdown head | `drawdown_risk_score` | `sigmoid`, masked BCE when target exists |
| representation | `news_embedding` | default 128-dimensional embedding for downstream fusion/risk modules |

### 8.10 Default loss composition

The default loss weights are:

```text
impact_loss_weight = 1.00
importance_loss_weight = 0.75
risk_loss_weight = 0.75
volatility_loss_weight = 0.25
drawdown_loss_weight = 0.25
```

The loss is masked for optional or missing targets. This matters because not every row/document has every future risk target available.

### 8.11 HPO design

The News Analyst is also HPO-first. It uses Optuna/TPE and SQLite-backed resumable studies.

The HPO search covers:

```text
learning_rate
weight_decay
dropout
batch_size
epochs
early_stop_patience
gradient_clip
d_model
attention_heads
self_attention_layers
hidden_dims
representation_dim
max_chunks_per_document
use_metadata_features
impact_loss_weight
importance_loss_weight
risk_loss_weight
volatility_loss_weight
drawdown_loss_weight
```

Default HPO options include:

```text
learning_rate: 2e-5 to 8e-4
weight_decay: 1e-6 to 5e-3
dropout: 0.05 to 0.35
batch sizes: 32, 64, 96, 128, 192
attention configs: 96x2, 96x3, 96x4, 96x6, 128x2, 128x4, 128x8, 192x2, 192x3, 192x4, 192x6, 192x8, 256x2, 256x4, 256x8
self-attention layers: 0 to 2
representation dimensions: 64 or 128
max chunks per document: 16, 32, 64, 96
```

The objective is:

```text
minimise validation composite loss
```

### 8.12 CPU and CUDA optimisation

The News Analyst is designed for both CPU and CUDA execution.

CUDA path:

```text
PyTorch model on GPU
mixed precision enabled unless --no-mixed-precision is passed
pin_memory enabled for CUDA DataLoaders
checkpointing after every epoch
```

CPU path:

```text
--device cpu
--torch-num-threads controls PyTorch CPU threading
--num-workers can parallelise DataLoader preparation
small HPO subsets can be used through --hpo-max-train-groups and --hpo-max-val-groups
```

Because the model aggregates document groups, CPU tests should use `--hpo-max-train-groups` and `--hpo-max-val-groups` first.

### 8.13 News checkpoints and outputs

For each chunk:

```text
outputs/models/analysts/news/chunk{N}/latest.pt
outputs/models/analysts/news/chunk{N}/best.pt
outputs/models/analysts/news/chunk{N}/epoch_XXX.pt
outputs/models/analysts/news/chunk{N}/document_metadata_preprocessor.json
```

Training history:

```text
outputs/results/analysts/news/chunk{N}_training_history.csv
outputs/results/analysts/news/chunk{N}_training_summary.json
```

Prediction outputs:

```text
outputs/results/analysts/news/chunk{N}_{split}_predictions.csv
outputs/results/analysts/news/chunk{N}_{split}_metrics.json
outputs/results/analysts/news/chunk{N}_{split}_attention.csv
outputs/embeddings/analysts/news/chunk{N}_{split}_news_embeddings.npy
```

The default News Analyst embedding shape is:

```text
(N_document_groups, 128)
```

If HPO selects `representation_dim = 64`, the embedding file shape becomes:

```text
(N_document_groups, 64)
```

### 8.14 Attention export

The attention CSV is an explainability output. It records document-group attention weights over the source chunk rows. This makes the News Analyst more transparent than a plain pooled document embedding.

Expected attention file:

```text
outputs/results/analysts/news/chunk{N}_{split}_attention.csv
```

This file should be used later for:

```text
module-level XAI
identifying influential SEC filing chunks
analysing form-type and section-level behaviour
building final explanation reports
```

---

## 9. Relationship between Sentiment Analyst and News Analyst

The two analysts are complementary.

| Feature | Sentiment Analyst | News Analyst |
|---|---|---|
| Main question | Was the market reaction bullish, neutral, or bearish? | Was the document/event important and risk-relevant? |
| Input level | chunk-level row | document-level group |
| Main architecture | MLP | attention pooling + MLP heads |
| Main target | `sentiment_score_target`, `sentiment_class_target` | `news_event_impact_target`, `news_importance_target`, `risk_relevance_target` |
| Representation output | default 64-dim | default 128-dim |
| Explainability | prediction confidence/uncertainty, metadata trace | attention over chunks + prediction scores |
| Best use | polarity and market-supervised sentiment | event salience, novelty/intensity, risk relevance |

The Sentiment Analyst is more direct and row-level. The News Analyst is more suitable for aggregation and event understanding.

---

## 10. Anti-leakage rules preserved in the textual analyst layer

The workflow follows these rules:

1. FinBERT embeddings are generated chronologically by chunk and split.
2. Market labels use prices after the filing date only.
3. Event start is the next available trading day after `filing_date`.
4. Thresholds for classes and risk flags are fitted only on the train split of each chunk.
5. Validation/test rows are never used to fit thresholds.
6. Metadata preprocessors are fitted only on train rows/groups.
7. Analyst models train only on train split rows/groups.
8. Validation split is used only for early stopping, HPO objective evaluation, and model selection.
9. Test split is held out for final evaluation and prediction export.
10. Missing future targets are filtered during supervised training, not fabricated.

---

## 11. File format policy

The textual analyst workflow follows the project file-format policy:

| Artefact | Format |
|---|---|
| Text embeddings | `.npy` |
| Analyst embeddings | `.npy` |
| Metadata | `.csv` |
| Labels | `.csv` |
| Predictions | `.csv` |
| Attention exports | `.csv` |
| Configs/manifests/HPO best params | `.json` |
| PyTorch model checkpoints | `.pt` |
| PCA/sklearn objects | `.pkl` only where necessary |
| Parquet | NOT TO BE USED |

---

## 12. Expected final directory structure

```text
code/analysts/
├── text_market_label_builder.py
├── sentiment_analyst.py
├── news_analyst.py
└── TextualAnalysts.md

outputs/results/analysts/
├── labels/
│   ├── text_market_labels_chunk1_train.csv
│   ├── text_market_labels_chunk1_val.csv
│   ├── text_market_labels_chunk1_test.csv
│   ├── text_market_labels_chunk2_train.csv
│   ├── text_market_labels_chunk2_val.csv
│   ├── text_market_labels_chunk2_test.csv
│   ├── text_market_labels_chunk3_train.csv
│   ├── text_market_labels_chunk3_val.csv
│   ├── text_market_labels_chunk3_test.csv
│   ├── text_market_labels_all.csv
│   ├── text_market_label_thresholds_chunk1.json
│   ├── text_market_label_thresholds_chunk2.json
│   └── text_market_label_thresholds_chunk3.json
├── sentiment/
│   ├── chunk1_training_history.csv
│   ├── chunk1_training_summary.json
│   ├── chunk1_val_predictions.csv
│   └── chunk1_val_metrics.json
└── news/
    ├── chunk1_training_history.csv
    ├── chunk1_training_summary.json
    ├── chunk1_val_predictions.csv
    ├── chunk1_val_metrics.json
    └── chunk1_val_attention.csv

outputs/models/analysts/
├── sentiment/
│   └── chunk1/
│       ├── latest.pt
│       ├── best.pt
│       ├── epoch_001.pt
│       └── metadata_preprocessor.json
└── news/
    └── chunk1/
        ├── latest.pt
        ├── best.pt
        ├── epoch_001.pt
        └── document_metadata_preprocessor.json

outputs/embeddings/analysts/
├── sentiment/
│   └── chunk1_val_sentiment_embeddings.npy
└── news/
    └── chunk1_val_news_embeddings.npy
```

The example above shows chunk 1 outputs only. Full runs should create equivalent files for chunks 1, 2, and 3 and for train/validation/test splits.

---

## 13. Recommended execution order

### 13.1 Compile checks

```bash
cd ~/fin-glassbox && source venv3.12.7/bin/activate && python -m py_compile code/analysts/text_market_label_builder.py
```

```bash
cd ~/fin-glassbox && source venv3.12.7/bin/activate && python -m py_compile code/analysts/sentiment_analyst.py
```

```bash
cd ~/fin-glassbox && source venv3.12.7/bin/activate && python -m py_compile code/analysts/news_analyst.py
```

### 13.2 Build real labels

```bash
cd ~/fin-glassbox && source venv3.12.7/bin/activate && python code/analysts/text_market_label_builder.py --repo-root ~/fin-glassbox --chunks 1,2,3 --splits train,val,test --benchmark-ticker SPY --overwrite --write-combined
```

### 13.3 Audit label availability

```bash
cd ~/fin-glassbox && source venv3.12.7/bin/activate && python -c "import pandas as pd, glob; files=sorted(glob.glob('outputs/results/analysts/labels/text_market_labels_chunk*_*.csv')); cols=['label_available','sentiment_score_target','sentiment_class_target','news_importance_target','risk_relevance_target']; [print('\\n'+f, '\\n', pd.DataFrame({'rows':[len(pd.read_csv(f))], **{c:[pd.read_csv(f)[c].notna().sum()] for c in cols if c in pd.read_csv(f).columns}}).to_string(index=False)) for f in files]"
```

### 13.4 Inspect Sentiment Analyst inputs

```bash
cd ~/fin-glassbox && source venv3.12.7/bin/activate && python code/analysts/sentiment_analyst.py inspect --repo-root ~/fin-glassbox --chunk 1 --device cpu
```

```bash
cd ~/fin-glassbox && source venv3.12.7/bin/activate && python code/analysts/sentiment_analyst.py inspect --repo-root ~/fin-glassbox --chunk 2 --device cpu
```

```bash
cd ~/fin-glassbox && source venv3.12.7/bin/activate && python code/analysts/sentiment_analyst.py inspect --repo-root ~/fin-glassbox --chunk 3 --device cpu
```

### 13.5 Sentiment Analyst CPU-safe HPO smoke test

```bash
cd ~/fin-glassbox && source venv3.12.7/bin/activate && python code/analysts/sentiment_analyst.py hpo --repo-root ~/fin-glassbox --chunk 1 --device cpu --trials 3 --hpo-max-train-rows 10000 --hpo-max-val-rows 3000 --hpo-epochs-min 2 --hpo-epochs-max 5
```

### 13.6 Train Sentiment Analyst from best HPO result

```bash
cd ~/fin-glassbox && source venv3.12.7/bin/activate && python code/analysts/sentiment_analyst.py train-best --repo-root ~/fin-glassbox --chunk 1 --device cpu
```

### 13.7 Full Sentiment Analyst HPO-first workflow

Use `cuda` only when the GPU is free. Use `cpu` if running on CPU.

```bash
cd ~/fin-glassbox && source venv3.12.7/bin/activate && python code/analysts/sentiment_analyst.py hpo-all --repo-root ~/fin-glassbox --chunks 1,2,3 --trials 30 --device cuda
```

```bash
cd ~/fin-glassbox && source venv3.12.7/bin/activate && python code/analysts/sentiment_analyst.py train-best-all --repo-root ~/fin-glassbox --chunks 1,2,3 --device cuda
```

```bash
cd ~/fin-glassbox && source venv3.12.7/bin/activate && python code/analysts/sentiment_analyst.py predict-all --repo-root ~/fin-glassbox --chunks 1,2,3 --splits train,val,test --checkpoint best --device cuda
```

### 13.8 Inspect News Analyst inputs

```bash
cd ~/fin-glassbox && source venv3.12.7/bin/activate && python code/analysts/news_analyst.py inspect --repo-root ~/fin-glassbox --chunk 1 --device cpu
```

```bash
cd ~/fin-glassbox && source venv3.12.7/bin/activate && python code/analysts/news_analyst.py inspect --repo-root ~/fin-glassbox --chunk 2 --device cpu
```

```bash
cd ~/fin-glassbox && source venv3.12.7/bin/activate && python code/analysts/news_analyst.py inspect --repo-root ~/fin-glassbox --chunk 3 --device cpu
```

### 13.9 News Analyst CPU-safe HPO smoke test

```bash
cd ~/fin-glassbox && source venv3.12.7/bin/activate && python code/analysts/news_analyst.py hpo --repo-root ~/fin-glassbox --chunk 1 --device cpu --trials 3 --hpo-max-train-groups 5000 --hpo-max-val-groups 1500 --hpo-epochs-min 2 --hpo-epochs-max 5
```

### 13.10 Train News Analyst from best HPO result

```bash
cd ~/fin-glassbox && source venv3.12.7/bin/activate && python code/analysts/news_analyst.py train-best --repo-root ~/fin-glassbox --chunk 1 --device cpu
```

### 13.11 Full News Analyst HPO-first workflow

Use `cuda` only when the GPU is free. Use `cpu` if running on CPU.

```bash
cd ~/fin-glassbox && source venv3.12.7/bin/activate && python code/analysts/news_analyst.py hpo-all --repo-root ~/fin-glassbox --chunks 1,2,3 --trials 30 --device cuda
```

```bash
cd ~/fin-glassbox && source venv3.12.7/bin/activate && python code/analysts/news_analyst.py train-best-all --repo-root ~/fin-glassbox --chunks 1,2,3 --device cuda
```

```bash
cd ~/fin-glassbox && source venv3.12.7/bin/activate && python code/analysts/news_analyst.py predict-all --repo-root ~/fin-glassbox --chunks 1,2,3 --splits train,val,test --checkpoint best --device cuda
```

### 13.12 News Analyst CPU tuning example

```bash
cd ~/fin-glassbox && source venv3.12.7/bin/activate && python code/analysts/news_analyst.py hpo --repo-root ~/fin-glassbox --chunk 1 --device cpu --torch-num-threads 8 --num-workers 4 --trials 5 --hpo-max-train-groups 5000 --hpo-max-val-groups 1500
```

---

## 14. What should be committed

The following source files should be committed:

```text
code/analysts/text_market_label_builder.py
code/analysts/sentiment_analyst.py
code/analysts/news_analyst.py
code/analysts/TextualAnalysts.md
```

Generated outputs should be handled according to repository policy. Large `.npy`, `.pt`, and large `.csv` outputs should generally not be committed directly unless the repository is configured for them. They should be reproducible from the code and stored as experiment artefacts.

---

## 15. Final implementation status

| Component | Status |
|---|---|
| FinBERT 256-dimensional embedding interface | Ready |
| Market-data pipeline | Complete |
| Real supervised label construction | Complete |
| Sentiment Analyst code | Implemented, HPO-first |
| News Analyst code | Implemented, HPO-first |
| Dummy/synthetic data usage | Not used |
| Next major task | Run inspections, HPO smoke tests, then full HPO/training when compute is available |

---

## 16. Summary

The textual analyst pipeline is now a real supervised-learning pipeline built from actual project artefacts:

```text
FinBERT embeddings provide text features.
Market returns provide future outcome labels.
The label builder creates leakage-safe targets.
The Sentiment Analyst learns row-level market-supervised sentiment.
The News Analyst learns document-level event importance and risk relevance using attention pooling.
```

This design is compatible with the broader explainable distributed framework because each module has explicit inputs, outputs, checkpoints, metrics, and explanation artefacts.
