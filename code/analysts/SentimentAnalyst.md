# Sentiment Analyst 

## 1. Module identity

**File:** `code/analysts/sentiment_analyst.py`  
**Role:** supervised market-aligned sentiment specialist  
**Branch:** qualitative branch  
**Upstream dependency:** FinBERT embeddings + text-market labels  
**Downstream consumers:** Qualitative Analyst and, indirectly, Fusion Engine

The Sentiment Analyst converts FinBERT text embeddings into supervised financial sentiment outputs. It is not a dictionary-based sentiment model and it is not trained on dummy labels. Its labels are built by `text_market_label_builder.py` from future market behaviour after the SEC filing date. In practical terms, it answers:

> Given this financial text representation, what market-aligned sentiment signal does it imply?

The model is deliberately modular. FinBERT remains the general financial-language encoder; the Sentiment Analyst is the task-specific supervised head that maps those text embeddings into sentiment, confidence, uncertainty, and compact downstream sentiment embeddings.

---

## 2. Architectural position

```text
SEC / financial text
        ↓
FinBERT financial text encoder
        ↓
Sentiment Analyst
        ↓
Qualitative Analyst
        ↓
Fusion Engine
        ↓
Final Trade Approver
```

The Sentiment Analyst does not make the final Buy/Hold/Sell decision. It contributes one qualitative signal that is later combined with News Analyst outputs by the Qualitative Analyst.

---

## 3. Input contract

For each chunk and split:

```text
outputs/embeddings/FinBERT/chunk{N}_{split}_embeddings.npy
outputs/results/analysts/labels/text_market_labels_chunk{N}_{split}.csv
```

where:

```text
N ∈ {1, 2, 3}
split ∈ {train, val, test}
```

The embedding file must be row-aligned with the label file. Row `i` in the embedding matrix must correspond to row `i` in the label CSV.

Expected embedding shape:

```text
(number_of_text_rows, 256)
```

Important label columns include:

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

---

## 4. Chronological split policy

The module follows the project-wide chronological chunks:

| Chunk | Train | Validation | Test |
|---:|---|---|---|
| 1 | 2000–2004 | 2005 | 2006 |
| 2 | 2007–2014 | 2015 | 2016 |
| 3 | 2017–2022 | 2023 | 2024 |

Validation and test rows are never used to fit preprocessing statistics, label thresholds, or HPO decisions beyond validation scoring.

---

## 5. Label logic

Labels are produced by `text_market_label_builder.py`. The label builder maps each filing to the first trading day strictly after `filing_date`, then computes forward excess market returns.

The continuous sentiment target is:

```text
sentiment_score_target = tanh(future_excess_log_return / train_fitted_scale)
```

The class target is based on train-fitted quantile thresholds:

```text
future excess return <= negative threshold → negative class
future excess return >= positive threshold → positive class
otherwise                                  → neutral class
```

This makes the Sentiment Analyst a **market-aligned sentiment model**, not a general natural-language polarity model.

---

## 6. Model architecture

Primary model class:

```text
SentimentAnalystMLP
```

Architecture:

```text
FinBERT embedding, 256-dim
        + optional metadata features
        ↓
MLP trunk
    Linear
    LayerNorm
    Tanh
    Dropout
        ↓
sentiment representation embedding
        ↓
polarity head
class head
magnitude head
```

Outputs:

| Output | Type / range | Purpose |
|---|---:|---|
| `sentiment_score` | `[-1, 1]` via `tanh` | directional market-aligned sentiment |
| `class_logits` | 3 logits | negative / neutral / positive class |
| `magnitude` | logit | strength of sentiment signal |
| `sentiment_embedding` | configurable, default 64-dim | compact downstream representation |

The model uses `tanh` activations in the hidden stack, matching the project’s preference for MLP hidden activations.

---

## 7. Loss function

The training objective combines polarity regression, class prediction, and magnitude learning:

```text
loss =
    polarity_loss_weight × MSE(sentiment_score, sentiment_score_target)
  + class_loss_weight × CE(class_logits, sentiment_class_target)
  + magnitude_loss_weight × BCEWithLogits(magnitude, sentiment_magnitude_target)
```

Default weights:

```text
polarity_loss_weight = 1.0
class_loss_weight = 0.5
magnitude_loss_weight = 0.25
```

This design preserves both fine-grained continuous signal and coarse class-level direction.

---

## 8. Main configuration

The core defaults from `SentimentConfig` include:

```text
input_dim = 256
hidden_dims = [128, 64]
representation_dim = 64
dropout = 0.15
batch_size = 512
eval_batch_size = 1024
epochs = 40
learning_rate = 1e-4
weight_decay = 1e-4
gradient_clip = 1.0
early_stop_patience = 8
mixed_precision = True
```

Configurable directories:

```text
outputs/embeddings/FinBERT
outputs/results/analysts/labels
outputs/embeddings/analysts/sentiment
outputs/models/analysts/sentiment
outputs/results/analysts/sentiment
outputs/codeResults/analysts/sentiment
```

---

## 9. Output contract

Model outputs:

```text
outputs/models/analysts/sentiment/chunk{N}/latest.pt
outputs/models/analysts/sentiment/chunk{N}/best.pt
outputs/models/analysts/sentiment/chunk{N}/metadata_preprocessor.json
```

Result outputs:

```text
outputs/results/analysts/sentiment/chunk{N}_training_history.csv
outputs/results/analysts/sentiment/chunk{N}_{split}_metrics.json
outputs/results/analysts/sentiment/chunk{N}_{split}_predictions.csv
```

Embedding outputs:

```text
outputs/embeddings/analysts/sentiment/chunk{N}_{split}_sentiment_embeddings.npy
```

These sentiment embeddings allow later modules to consume a compact learned sentiment representation instead of raw FinBERT outputs.

---

## 10. Metrics and validation

The module computes:

```text
loss
polarity_loss
class_loss
magnitude_loss
MSE
MAE
RMSE
correlation
classification accuracy
3-class confusion matrix
```

The main HPO objective is validation loss.

---

## 11. XAI and explainability

The Sentiment Analyst exposes interpretability through:

1. Row-level prediction columns: score, class, magnitude, confidence, and uncertainty.
2. A saved sentiment embedding stream for downstream traceability.
3. Optional attribution mode through `--xai-method`, currently supporting `gradient` and `shap`.

For large-scale runs, gradient-based attribution is the practical default because SHAP can be computationally expensive on hundreds of thousands of text embeddings.

---

## 12. Leakage prevention

The module protects against leakage by:

```text
using chronological splits only
using train split for fitting metadata preprocessing
using train-fitted market thresholds from the label builder
not fitting scalers on validation/test
requiring row alignment between embeddings and labels
serialising preprocessing state with the model
```

Because targets are derived from future returns, these controls are essential.

---

## 13. CLI commands

Inspect:

```bash
python code/analysts/sentiment_analyst.py inspect --repo-root . --chunk 1
```

HPO:

```bash
python code/analysts/sentiment_analyst.py hpo --repo-root . --chunk 1 --trials 30 --device cuda
```

Train from best HPO:

```bash
python code/analysts/sentiment_analyst.py train-best --repo-root . --chunk 1 --device cuda
```

Predict one split:

```bash
python code/analysts/sentiment_analyst.py predict --repo-root . --chunk 1 --split test --device cuda
```

Predict all chunks and splits:

```bash
python code/analysts/sentiment_analyst.py predict-all --repo-root . --chunks 1,2,3 --splits train,val,test --device cuda
```

Full rerun after FinBERT regeneration:

```bash
python code/analysts/sentiment_analyst.py hpo-all --repo-root . --chunks 1,2,3 --trials 30 --device cuda && python code/analysts/sentiment_analyst.py train-best-all --repo-root . --chunks 1,2,3 --device cuda && python code/analysts/sentiment_analyst.py predict-all --repo-root . --chunks 1,2,3 --splits train,val,test --device cuda
```

---

## 14. Integration notes

Rerun this module whenever FinBERT embeddings are refreshed. Do not mix old Sentiment Analyst outputs with new News Analyst or Qualitative Analyst outputs. The correct text-side dependency chain is:

```text
FinBERT embeddings
        ↓
text_market_label_builder.py
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
The Sentiment Analyst is a supervised, market-aligned financial text module. It uses FinBERT embeddings, train-only market-derived labels, checkpointing, HPO, explicit confidence/uncertainty outputs, and downstream embedding export. It contributes interpretable sentiment evidence to the qualitative branch without pretending to be the final trading decision-maker.
