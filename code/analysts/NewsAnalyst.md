# News Analyst Module

## 1. Module identity

**File:** `code/analysts/news_analyst.py`  
**Role:** supervised document-level financial event analyst  
**Branch:** qualitative branch  
**Upstream dependency:** FinBERT embeddings + text-market labels  
**Downstream consumers:** Qualitative Analyst and Fusion Engine through the qualitative branch

The News Analyst groups FinBERT text-chunk embeddings into document-level units and predicts event impact, importance, risk relevance, volatility-spike likelihood, and drawdown-risk likelihood. It provides the event-analysis half of the qualitative branch.

Although the current corpus is SEC filing text, the module’s interface is suitable for future company-news and macro-news inputs if they are converted into the same embedding + metadata + label format.

---

## 2. Architectural position

```text
Financial text / SEC filings
        ↓
FinBERT Encoder
        ↓
News Analyst
        ↓
Qualitative Analyst
        ↓
Fusion Engine
        ↓
Final Trade Approver
```

The Sentiment Analyst focuses on market-aligned polarity. The News Analyst focuses on event importance, risk relevance, and document-level impact.

---

## 3. Input contract

Required embeddings:

```text
outputs/embeddings/FinBERT/chunk{N}_{split}_embeddings.npy
```

Required labels:

```text
outputs/results/analysts/labels/text_market_labels_chunk{N}_{split}.csv
```

Expected embedding dimension:

```text
256
```

Important metadata columns:

```text
doc_id
accession
ticker
cik
filing_date
year
form_type
source_name
chunk_index
word_count
```

Important target columns:

```text
news_event_impact_target
news_importance_target
risk_relevance_target
volatility_spike_{risk_horizon}d_target, optional
drawdown_risk_{risk_horizon}d_target, optional
```

---

## 4. Document grouping

The News Analyst does not treat each text chunk as a separate event. It groups chunks by:

```text
doc_id
accession
```

Within each document, chunks are sorted by `chunk_index`. Each grouped sample stores:

```text
row_indices
chunk_indices
n_chunks_original
total_word_count
mean_word_count
max_chunk_index
document metadata
document-level targets
```

The maximum number of chunks per document is configurable using:

```text
max_chunks_per_document
```

Default:

```text
64
```

This is essential for keeping long filings computationally manageable while still preserving document-level context.

---

## 5. Chronological split policy

| Chunk | Train | Validation | Test |
|---:|---|---|---|
| 1 | 2000–2004 | 2005 | 2006 |
| 2 | 2007–2014 | 2015 | 2016 |
| 3 | 2017–2022 | 2023 | 2024 |

One model is trained per chunk. Validation and test data are never used to fit thresholds or preprocessing state.

---

## 6. Label construction

The labels come from `text_market_label_builder.py`.

### 6.1 Event impact

```text
news_event_impact_target = tanh(future_excess_return / train_fitted_news_scale)
```

Range:

```text
[-1, 1]
```

### 6.2 News importance

```text
news_importance_target = clip(abs(future_excess_return) / train_fitted_news_scale, 0, 1)
```

Range:

```text
[0, 1]
```

### 6.3 Risk relevance

```text
risk_relevance_target = max(volatility_score, drawdown_score)
```

Range:

```text
[0, 1]
```

### 6.4 Optional risk heads

When available, the model also learns:

```text
volatility_spike_score
drawdown_risk_score
```

These allow the News Analyst to contribute directly to risk-aware qualitative reasoning.

---

## 7. Model architecture

Primary model class:

```text
NewsAnalystAttentionModel
```

Architecture:

```text
Document chunk embeddings, shape = (batch, chunks, 256)
        ↓
token projection to d_model
        ↓
optional metadata projection
        ↓
self-attention context layers
        ↓
multi-head attention pooling over chunks
        ↓
MLP trunk
        ↓
event impact head
importance head
risk relevance head
volatility spike head
drawdown risk head
```

Default core settings:

```text
input_dim = 256
d_model = 128
attention_heads = 4
self_attention_layers = 1
hidden_dims = [128, 64]
representation_dim = 128
dropout = 0.15
batch_size = 96
eval_batch_size = 192
epochs = 40
```

The model keeps PyTorch-only execution by disabling TensorFlow-related environment paths.

---

## 8. Output heads

| Output | Activation | Range | Purpose |
|---|---|---:|---|
| `event_impact_score` | `tanh` | `[-1, 1]` | directional document/event impact |
| `news_importance_score` | `sigmoid` | `[0, 1]` | importance/materiality |
| `risk_relevance_score` | `sigmoid` | `[0, 1]` | risk relevance |
| `volatility_spike_score` | `sigmoid` | `[0, 1]` | volatility event likelihood |
| `drawdown_risk_score` | `sigmoid` | `[0, 1]` | drawdown event likelihood |
| `news_embedding` | trunk vector | configurable | downstream document representation |
| `attention_weights` | per head/chunk | `[0, 1]` | XAI trace |

---

## 9. Loss function

The objective is:

```text
loss =
    impact_loss_weight × MSE(event_impact_score, target_impact)
  + importance_loss_weight × BCE(news_importance_score, target_importance)
  + risk_loss_weight × BCE(risk_relevance_score, target_risk)
  + volatility_loss_weight × masked_BCE(volatility_spike_score, target_volatility)
  + drawdown_loss_weight × masked_BCE(drawdown_risk_score, target_drawdown)
```

Default weights:

```text
impact_loss_weight = 1.0
importance_loss_weight = 0.75
risk_loss_weight = 0.75
volatility_loss_weight = 0.25
drawdown_loss_weight = 0.25
```

The volatility and drawdown losses are masked because those targets may be absent or partially unavailable.

---

## 10. Output contract

Model files:

```text
outputs/models/analysts/news/chunk{N}/latest.pt
outputs/models/analysts/news/chunk{N}/best.pt
outputs/models/analysts/news/chunk{N}/document_metadata_preprocessor.json
```

Result files:

```text
outputs/results/analysts/news/chunk{N}_training_history.csv
outputs/results/analysts/news/chunk{N}_{split}_metrics.json
outputs/results/analysts/news/chunk{N}_{split}_news_predictions.csv
outputs/results/analysts/news/chunk{N}_{split}_attention.csv
```

Embedding files:

```text
outputs/embeddings/analysts/news/chunk{N}_{split}_news_embeddings.npy
```

---

## 11. Prediction schema

The prediction output is document-level. Typical fields include:

```text
ticker
date / filing_date
doc_id
accession
form_type
source_name
news_event_impact
news_importance
risk_relevance
volatility_spike
drawdown_risk
news_uncertainty
news_event_present
```

The attention file records which chunks and attention heads drove the document prediction.

---

## 12. XAI design

The News Analyst’s primary XAI mechanism is document-level attention:

```text
attention head → chunk index → importance weight
```

This lets the system explain which parts of a filing or document mattered most. The module also keeps provenance fields such as `doc_id`, `accession`, `form_type`, and `source_name`, so explanations can be traced back to document origin.

---

## 13. CLI commands

Inspect:

```bash
python code/analysts/news_analyst.py inspect --repo-root . --chunk 1
```

HPO:

```bash
python code/analysts/news_analyst.py hpo --repo-root . --chunk 1 --trials 30 --device cuda
```

Train best:

```bash
python code/analysts/news_analyst.py train-best --repo-root . --chunk 1 --device cuda
```

Predict one split:

```bash
python code/analysts/news_analyst.py predict --repo-root . --chunk 1 --split test --device cuda
```

Full rerun after FinBERT regeneration:

```bash
python code/analysts/news_analyst.py hpo-all --repo-root . --chunks 1,2,3 --trials 30 --device cuda && python code/analysts/news_analyst.py train-best-all --repo-root . --chunks 1,2,3 --device cuda && python code/analysts/news_analyst.py predict-all --repo-root . --chunks 1,2,3 --splits train,val,test --device cuda
```

---

## 14. Integration notes

The News Analyst should be rerun whenever FinBERT embeddings are regenerated. The Qualitative Analyst should then be rerun because it consumes both Sentiment and News outputs.

Correct text-side sequence:

```text
FinBERT
        ↓
Sentiment Analyst
        ↓
News Analyst
        ↓
Qualitative Analyst
        ↓
Fusion Engine
```

---
The News Analyst is a supervised document-level event-impact model. It uses FinBERT chunk embeddings, document grouping, market-derived labels, self-attention, multi-head attention pooling, and attention trace export. Its role is to provide explainable event impact and risk relevance to the qualitative branch.
