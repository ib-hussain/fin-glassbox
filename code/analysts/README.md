# `code/analysts/` Folder Documentation

## 1. Folder purpose

The `code/analysts/` folder contains the specialist analysis layer for the fin-glassbox project:

```text
An Explainable Distributed Deep Learning Framework for Financial Risk Management
```

The analysts convert encoder outputs and risk-engine outputs into interpretable module-level and branch-level signals. They are not final trade approvers. They exist to preserve the project philosophy:

```text
specialisation + modularity + explainability + risk-aware synthesis
```

---

## 2. Folder files and references

| File | Documentation | Role |
|---|---|---|
| [`text_market_label_builder.py`](text_market_label_builder.py) | Section 6 of this document | Builds supervised market-derived labels for text analysts |
| [`sentiment_analyst.py`](sentiment_analyst.py) | [`SentimentAnalyst.md`](SentimentAnalyst.md) | Learns market-aligned sentiment from FinBERT embeddings |
| [`news_analyst.py`](news_analyst.py) | [`NewsAnalyst.md`](NewsAnalyst.md) | Learns document/event impact, importance, and risk relevance |
| [`technical_analyst.py`](technical_analyst.py) | [`TechnicalAnalyst.md`](TechnicalAnalyst.md) | Learns trend, momentum, and timing confidence from Temporal Encoder embeddings |
| [`qualitative_analyst.py`](qualitative_analyst.py) | [`QualitativeAnalyst.md`](QualitativeAnalyst.md) | Combines sentiment and news outputs into qualitative branch outputs |
| [`quantitative_analyst.py`](quantitative_analyst.py) | [`QuantitativeAnalyst.md`](QuantitativeAnalyst.md) | Learns risk-attention pooling over quantitative risk/position outputs |

---

## 3. Architectural placement

```text
ENCODERS
├── FinBERT
└── Shared Temporal Attention Encoder
        ↓
ANALYSTS
├── Sentiment Analyst
├── News Analyst
├── Technical Analyst
├── Qualitative Analyst
└── Quantitative Analyst
        ↓
FUSION ENGINE
        ↓
FINAL TRADE APPROVER
```

The folder contains both first-level specialist analysts and branch-level synthesis analysts.

---

## 4. Analyst categories

### 4.1 Textual/qualitative path

```text
text_market_label_builder.py
sentiment_analyst.py
news_analyst.py
qualitative_analyst.py
```

This path turns SEC text and FinBERT embeddings into daily qualitative branch outputs.

### 4.2 Technical/market path

```text
technical_analyst.py
```

This path turns Temporal Encoder embeddings into technical scores.

### 4.3 Quantitative synthesis path

```text
quantitative_analyst.py
```

This path consumes Position Sizing output and learns attention-weighted risk synthesis.

---

## 5. End-to-end analyst data flow

### 5.1 Qualitative branch

```text
SEC textual filings
        ↓
FinBERT embeddings
        ↓
text_market_label_builder.py
        ↓
Sentiment Analyst + News Analyst
        ↓
Qualitative Analyst
        ↓
Fusion Engine
```

### 5.2 Quantitative branch

```text
Market data
        ↓
Temporal Encoder embeddings
        ↓
Technical Analyst
        ↓
Risk Engine modules
        ↓
Position Sizing Engine
        ↓
Quantitative Analyst
        ↓
Fusion Engine
```

---

## 6. `text_market_label_builder.py`

### 6.1 Purpose

This file builds real supervised labels for the text analysts. It joins SEC text metadata with CIK/ticker mapping and market returns.

It creates labels such as:

```text
sentiment_score_target
sentiment_class_target
news_event_impact_target
news_importance_target
risk_relevance_target
volatility_spike_{risk_horizon}d_target
drawdown_risk_{risk_horizon}d_target
```

### 6.2 Design rules

```text
Inputs are SEC metadata available at filing time.
Targets are future market outcomes after the filing date.
The event start is the first trading day strictly after filing_date.
Train-only thresholds are fitted per chronological chunk.
Validation/test thresholds are inherited from the training split.
No dummy labels are created.
```

### 6.3 Default horizons

```text
horizons = 1,5,10,20,30
primary_sentiment_horizon = 10
primary_news_horizon = 10
risk_horizon = 30
```

### 6.4 Why it matters

This file makes the text analysts thesis-defensible because the supervision comes from future market outcomes rather than subjective sentiment labels.

---

## 7. Sentiment Analyst

Documentation: [`SentimentAnalyst.md`](SentimentAnalyst.md)

Primary input:

```text
outputs/embeddings/FinBERT/chunk{N}_{split}_embeddings.npy
outputs/results/analysts/labels/text_market_labels_chunk{N}_{split}.csv
```

Primary output:

```text
outputs/results/analysts/sentiment/chunk{N}_{split}_predictions.csv
outputs/embeddings/analysts/sentiment/chunk{N}_{split}_sentiment_embeddings.npy
```

Core role:

```text
FinBERT embedding → market-aligned sentiment score, class, confidence, uncertainty, magnitude
```

---

## 8. News Analyst

Documentation: [`NewsAnalyst.md`](NewsAnalyst.md)

Primary input:

```text
outputs/embeddings/FinBERT/chunk{N}_{split}_embeddings.npy
outputs/results/analysts/labels/text_market_labels_chunk{N}_{split}.csv
```

Primary output:

```text
outputs/results/analysts/news/chunk{N}_{split}_news_predictions.csv
outputs/results/analysts/news/chunk{N}_{split}_attention.csv
outputs/embeddings/analysts/news/chunk{N}_{split}_news_embeddings.npy
```

Core role:

```text
document chunks → event impact, importance, risk relevance, volatility spike, drawdown risk
```

---

## 9. Technical Analyst

Documentation: [`TechnicalAnalyst.md`](TechnicalAnalyst.md)

Primary input:

```text
outputs/embeddings/TemporalEncoder/chunk{N}_{split}_embeddings.npy
outputs/embeddings/TemporalEncoder/chunk{N}_{split}_manifest.csv
```

Primary output:

```text
outputs/results/TechnicalAnalyst/predictions_chunk{N}_{split}.csv
outputs/results/TechnicalAnalyst/xai/
```

Core role:

```text
Temporal Encoder sequence → trend_score, momentum_score, timing_confidence
```

---

## 10. Qualitative Analyst

Documentation: [`QualitativeAnalyst.md`](QualitativeAnalyst.md)

Primary input:

```text
outputs/results/analysts/sentiment/chunk{N}_{split}_predictions.csv
outputs/results/analysts/news/chunk{N}_{split}_news_predictions.csv
```

Primary output:

```text
outputs/results/QualitativeAnalyst/qualitative_events_chunk{N}_{split}.csv
outputs/results/QualitativeAnalyst/qualitative_daily_chunk{N}_{split}.csv
outputs/results/QualitativeAnalyst/xai/
```

Core role:

```text
Sentiment + News → daily qualitative score, qualitative risk, qualitative confidence
```

---

## 11. Quantitative Analyst

Documentation: [`QuantitativeAnalyst.md`](QuantitativeAnalyst.md)

Primary input:

```text
outputs/results/PositionSizing/position_sizing_chunk{N}_{split}.csv
```

Primary output:

```text
outputs/results/QuantitativeAnalyst/quantitative_analysis_chunk{N}_{split}.csv
outputs/results/QuantitativeAnalyst/xai/quantitative_analysis_chunk{N}_{split}_xai_summary.json
```

Core role:

```text
Risk scores + technical context + position sizing → risk-attention quantitative branch output
```

Fusion requires the trained attention schema:

```text
top_attention_risk_driver
risk_attention_volatility
risk_attention_drawdown
risk_attention_var_cvar
risk_attention_contagion
risk_attention_liquidity
risk_attention_regime
attention_pooled_risk_score
```

---

## 12. Shared implementation standards

All analyst files should follow these standards:

```text
Importable module + executable CLI.
Chronological chunks only.
No dummy data for real training.
HPO with Optuna/TPE where applicable.
Checkpointed training.
Prediction outputs preserve ticker/date/document provenance.
XAI outputs are either embedded in prediction rows or saved as sidecar JSON/CSV files.
CUDA and CPU execution are supported.
Commands should be written as single lines.
CSV/NPY/JSON/PT are the primary output formats.
```

---

## 13. Dependency order

Recommended order after upstream embeddings exist:

```text
1. text_market_label_builder.py
2. sentiment_analyst.py
3. news_analyst.py
4. qualitative_analyst.py
5. technical_analyst.py
6. riskEngine/position_sizing.py
7. quantitative_analyst.py
8. fusion layer
```

The Technical Analyst can run independently of the text-side modules as long as Temporal Encoder embeddings exist.

---

## 14. Full rerun commands

Compile:

```bash
python -m py_compile code/analysts/text_market_label_builder.py code/analysts/sentiment_analyst.py code/analysts/news_analyst.py code/analysts/technical_analyst.py code/analysts/qualitative_analyst.py code/analysts/quantitative_analyst.py
```

Sentiment full rerun:

```bash
python code/analysts/sentiment_analyst.py hpo-all --repo-root . --chunks 1,2,3 --trials 30 --device cuda && python code/analysts/sentiment_analyst.py train-best-all --repo-root . --chunks 1,2,3 --device cuda && python code/analysts/sentiment_analyst.py predict-all --repo-root . --chunks 1,2,3 --splits train,val,test --device cuda
```

News full rerun:

```bash
python code/analysts/news_analyst.py hpo-all --repo-root . --chunks 1,2,3 --trials 30 --device cuda && python code/analysts/news_analyst.py train-best-all --repo-root . --chunks 1,2,3 --device cuda && python code/analysts/news_analyst.py predict-all --repo-root . --chunks 1,2,3 --splits train,val,test --device cuda
```

Technical chunk rerun:

```bash
python code/analysts/technical_analyst.py hpo --repo-root . --chunk 1 --trials 40 --device cuda --fresh && python code/analysts/technical_analyst.py train-best --repo-root . --chunk 1 --device cuda --fresh && python code/analysts/technical_analyst.py predict --repo-root . --chunk 1 --split train --device cuda && python code/analysts/technical_analyst.py predict --repo-root . --chunk 1 --split val --device cuda && python code/analysts/technical_analyst.py predict --repo-root . --chunk 1 --split test --device cuda
```

Qualitative chunk rerun:

```bash
python code/analysts/qualitative_analyst.py hpo --repo-root . --chunk 1 --trials 30 --device cuda --fresh && python code/analysts/qualitative_analyst.py train-best --repo-root . --chunk 1 --device cuda --fresh && python code/analysts/qualitative_analyst.py predict-all --repo-root . --chunks 1 --splits train val test --device cuda
```

Quantitative chunk rerun:

```bash
python code/analysts/quantitative_analyst.py hpo --repo-root . --chunk 1 --trials 30 --device cuda --fresh && python code/analysts/quantitative_analyst.py train-best --repo-root . --chunk 1 --device cuda --fresh && python code/analysts/quantitative_analyst.py predict-all --repo-root . --chunks 1 --splits train val test --device cuda
```

---

## 15. Validation commands

Check Quantitative Analyst schema before Fusion:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path
for p in sorted(Path('outputs/results/QuantitativeAnalyst').glob('quantitative_analysis_chunk*_*.csv')):
    cols = pd.read_csv(p, nrows=0).columns
    print(p, 'attention_schema=', 'top_attention_risk_driver' in cols, 'old_schema=', 'top_risk_driver' in cols and 'top_attention_risk_driver' not in cols)
PY
```

Check Qualitative daily outputs:

```bash
python - <<'PY'
from pathlib import Path
for c in [1,2,3]:
    for s in ['train','val','test']:
        p = Path(f'outputs/results/QualitativeAnalyst/qualitative_daily_chunk{c}_{s}.csv')
        print(f'chunk{c}_{s}:', 'OK' if p.exists() else 'MISSING', p)
PY
```

---

## 16. Integration risk notes

Do not mix old and new Quantitative Analyst outputs. Fusion must consume the trained attention schema.

If FinBERT is regenerated, rerun:

```text
Sentiment Analyst
News Analyst
Qualitative Analyst
Fusion Engine
```

If Temporal Encoder embeddings are regenerated, rerun:

```text
Technical Analyst
Position Sizing
Quantitative Analyst
Fusion Engine
```

Fusion training requires train outputs from both branch-level analysts:

```text
outputs/results/QuantitativeAnalyst/quantitative_analysis_chunk{N}_train.csv
outputs/results/QualitativeAnalyst/qualitative_daily_chunk{N}_train.csv
```

---
The `code/analysts/` folder implements the specialist reasoning layer of the project. It converts text embeddings, temporal embeddings, and risk-engine outputs into explainable branch signals. The folder supports the project’s central claim that a distributed, specialised, explainable architecture is more defensible than a single monolithic black-box predictor.
