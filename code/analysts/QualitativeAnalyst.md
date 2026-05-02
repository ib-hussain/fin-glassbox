# Qualitative Synthesis Layer Analyst 

## 1. Module Identity

**File:** `code/analysts/qualitative_analyst.py`  
**Module name:** Qualitative Analyst  
**Project:** *An Explainable Distributed Deep Learning Framework for Financial Risk Management*  
**Layer:** Qualitative synthesis branch  
**Status:** Trainable downstream analyst module  
**Consumes:** Sentiment Analyst outputs + News Analyst outputs  
**Produces:** Event-level and daily ticker-level qualitative signals for the Fusion Engine  
**Fundamentals used:** No

The Qualitative Analyst is the trained synthesis module that combines text-side information from the Sentiment Analyst and News Analyst. It does not perform raw text encoding and it does not read SEC filings directly. Raw financial text is already encoded upstream by FinBERT, then processed by specialist downstream text analysts. This module learns how to combine those specialist outputs into a structured qualitative branch signal.

The module is intentionally separated from the Quantitative Analyst. The Quantitative Analyst uses technical, risk, and position-sizing outputs. The Qualitative Analyst uses text-derived outputs only.

---

## 2. Purpose in the Overall Architecture

The final system follows a modular, distributed design rather than a single monolithic model. The Qualitative Analyst exists to summarise **textual market evidence** into a form that can be merged later with the quantitative branch.

Its role is to answer:

> “What does the available financial text evidence suggest about this ticker/date, and how risky or reliable is that text evidence?”

It does **not** answer:

> “Should the final system buy or sell?”

That final action is handled later by the Fusion Engine and Final Trade Approver.

The simplified location of this module is:

```text
FinBERT Encoder
    ↓
Sentiment Analyst        News Analyst
    ↓                    ↓
        Qualitative Analyst
                ↓
            Fusion Engine
                ↓
        Final Trade Approver
```

---

## 3. Why This Module Is Trainable

The first design possibility was a purely rule-based qualitative synthesis layer. However, training is important in the project architecture because each major analyst module should learn a calibrated mapping rather than only applying fixed rules.

The implemented design therefore uses a **lightweight trainable MLP calibration model**.

The model learns from structured upstream analyst outputs, not from raw text. This is important because the raw language understanding is already handled by FinBERT and the text-specific analysts.

The Qualitative Analyst is trained using weak-supervised targets derived from sentiment and news outputs. This makes the module trainable even when no manually labelled qualitative synthesis dataset exists.

This is defensible because:

1. The module is not inventing labels from nothing; it learns from upstream analyst signals.
2. The weak labels are transparent and formula-defined.
3. The learned model can smooth, calibrate, and combine noisy upstream signals.
4. The XAI output still exposes the original drivers and the learned feature importance.

---

## 4. Inputs

The module consumes two upstream prediction families.

### 4.1 Sentiment Analyst Outputs

Expected locations include:

```text
outputs/results/analysts/sentiment/
outputs/results/SentimentAnalyst/
```

The script automatically searches for chunk/split-specific sentiment prediction files.

Expected file naming patterns include:

```text
chunk{chunk}_{split}_predictions.csv
sentiment_predictions_chunk{chunk}_{split}.csv
predictions_chunk{chunk}_{split}.csv
```

Typical sentiment columns include:

```text
ticker
filing_date or date
doc_id
accession
form_type
source_name
predicted_sentiment_score
predicted_sentiment_class
predicted_sentiment_confidence
predicted_sentiment_uncertainty
predicted_sentiment_magnitude
```

The module supports flexible column aliases. For example, `predicted_sentiment_score`, `sentiment_score`, and `pred_sentiment_score` can all be mapped to the internal `sentiment_score` field.

### 4.2 News Analyst Outputs

Expected locations include:

```text
outputs/results/analysts/news/
outputs/results/NewsAnalyst/
```

Expected file naming patterns include:

```text
chunk{chunk}_{split}_news_predictions.csv
news_predictions_chunk{chunk}_{split}.csv
predictions_chunk{chunk}_{split}.csv
```

Typical news columns include:

```text
ticker
filing_date or date
doc_id
accession
form_type
source_name
predicted_news_event_impact
predicted_news_importance
predicted_risk_relevance
predicted_volatility_spike
predicted_drawdown_risk
predicted_news_uncertainty
```

The News Analyst is treated as an event-impact and event-risk source, not only a direction source.

---

## 5. Standardised Internal Feature Schema

The module converts all available sentiment and news columns into a fixed feature vector.

```text
sentiment_score
sentiment_confidence
sentiment_uncertainty
sentiment_magnitude
sentiment_event_present
news_event_impact
news_importance
risk_relevance
volatility_spike
drawdown_risk
news_uncertainty
news_event_present
has_both_sources
```

These 13 features form the model input vector.

### Feature Meanings

| Feature | Meaning |
|---|---|
| `sentiment_score` | Directional sentiment score, normally from -1 to +1 |
| `sentiment_confidence` | Confidence of the Sentiment Analyst prediction |
| `sentiment_uncertainty` | Uncertainty of the sentiment output |
| `sentiment_magnitude` | Strength/magnitude of the sentiment signal |
| `sentiment_event_present` | Binary indicator that sentiment evidence exists |
| `news_event_impact` | Directional market impact estimated by News Analyst |
| `news_importance` | Importance or relevance of the news/filing event |
| `risk_relevance` | How relevant the news event is to financial risk |
| `volatility_spike` | Estimated likelihood or severity of volatility increase |
| `drawdown_risk` | Estimated downside/drawdown risk from the news event |
| `news_uncertainty` | Uncertainty of the News Analyst output |
| `news_event_present` | Binary indicator that news evidence exists |
| `has_both_sources` | Binary indicator that both sentiment and news evidence exist |

---

## 6. Output Targets

The model predicts three continuous outputs:

```text
qualitative_score
qualitative_risk_score
qualitative_confidence
```

### 6.1 `qualitative_score`

Range:

```text
-1 to +1
```

Interpretation:

```text
-1 = strongly negative qualitative evidence
 0 = neutral or no clear qualitative direction
+1 = strongly positive qualitative evidence
```

### 6.2 `qualitative_risk_score`

Range:

```text
0 to 1
```

Interpretation:

```text
0 = low text-driven risk
1 = severe text-driven risk
```

This is separate from direction. A text event can be positive but still risky.

### 6.3 `qualitative_confidence`

Range:

```text
0 to 1
```

Interpretation:

```text
0 = unreliable or missing text evidence
1 = strong, confident qualitative evidence
```

---

## 7. Weak-Supervised Target Construction

Because there is no manually labelled qualitative synthesis dataset, the module constructs transparent weak-supervised targets from the upstream analyst outputs.

### 7.1 Direction Target

```text
sentiment_direction = sentiment_score
news_direction = news_event_impact × news_importance
qualitative_score_target = 0.55 × sentiment_direction + 0.45 × news_direction
```

The result is clipped to:

```text
[-1, +1]
```

Sentiment receives slightly higher weight because it is already a direct polarity signal, while news impact is event-based and may be noisier.

### 7.2 Risk Target

```text
qualitative_risk_target =
    0.35 × risk_relevance
  + 0.30 × volatility_spike
  + 0.25 × drawdown_risk
  + 0.10 × news_uncertainty
```

The result is clipped to:

```text
[0, 1]
```

This makes the model sensitive to text that warns of tail risk, instability, volatility, or downside pressure.

### 7.3 Confidence Target

```text
qualitative_confidence_target =
    0.40 × sentiment_confidence
  + 0.25 × (1 - sentiment_uncertainty)
  + 0.20 × news_importance
  + 0.15 × (1 - news_uncertainty)
```

The result is clipped to:

```text
[0, 1]
```

If neither sentiment nor news evidence exists, confidence is forced to 0.

---

## 8. Model Architecture

The Qualitative Analyst uses a small MLP.

```text
Input: 13-dimensional feature vector
    ↓
Linear layer(s)
    ↓
Tanh activation
    ↓
Dropout
    ↓
Linear output head
    ↓
Output transformations
```

The output transformations are:

```text
qualitative_score      = tanh(raw_output_0)
qualitative_risk       = sigmoid(raw_output_1)
qualitative_confidence = sigmoid(raw_output_2)
```

This ensures valid output ranges:

```text
qualitative_score      ∈ [-1, +1]
qualitative_risk       ∈ [0, 1]
qualitative_confidence ∈ [0, 1]
```

Default architecture:

```text
input_dim = 13
hidden_dim = 64
n_layers = 2
dropout = 0.15
activation = tanh
output_dim = 3
```

The hidden activation is `tanh`, consistent with the project preference for MLP hidden layers.

---

## 9. Training Objective

The module uses a weighted multi-output regression loss.

```text
score_loss = MSE(predicted_qualitative_score, qualitative_score_target)
risk_loss  = MSE(predicted_qualitative_risk, qualitative_risk_target)
conf_loss  = MSE(predicted_qualitative_confidence, qualitative_confidence_target)
```

Final loss:

```text
loss = 0.40 × score_loss + 0.35 × risk_loss + 0.25 × confidence_loss
```

The score and risk outputs receive higher emphasis because they are more directly used in the Fusion Engine.

---

## 10. Hyperparameter Optimisation

The module supports Optuna TPE hyperparameter search.

Search parameters:

```text
hidden_dim:     [32, 64, 96, 128]
n_layers:       1 to 3
dropout:        0.05 to 0.40
learning_rate:  1e-4 to 3e-3
weight_decay:   1e-7 to 1e-3
batch_size:     [256, 512, 1024]
```

HPO is stored in SQLite:

```text
outputs/codeResults/QualitativeAnalyst/hpo_chunk{chunk}.db
```

Best parameters are saved as:

```text
outputs/codeResults/QualitativeAnalyst/best_params_chunk{chunk}.json
```

The HPO command is resumable unless `--fresh` is used.

---

## 11. Training Outputs

For each chunk, the module saves:

```text
outputs/models/QualitativeAnalyst/chunk{chunk}/best_model.pt
outputs/models/QualitativeAnalyst/chunk{chunk}/final_model.pt
outputs/models/QualitativeAnalyst/chunk{chunk}/scaler.npz
outputs/models/QualitativeAnalyst/chunk{chunk}/training_history.csv
outputs/models/QualitativeAnalyst/chunk{chunk}/model_freezed/model.pt
outputs/models/QualitativeAnalyst/chunk{chunk}/model_unfreezed/model.pt
```

The model checkpoint contains:

```text
state_dict
config
feature_names
target_names
best_val_loss
```

The scaler contains train-fitted feature mean and standard deviation. The scaler is fitted only on the training split to avoid validation/test leakage.

---

## 12. Prediction Outputs

The module produces two prediction outputs per chunk/split.

### 12.1 Event-Level Output

Path:

```text
outputs/results/QualitativeAnalyst/qualitative_events_chunk{chunk}_{split}.csv
```

Purpose:

```text
Preserve individual text-event evidence for explanation and audit.
```

Important columns:

```text
ticker
date
chunk
split
doc_id
accession
form_type
source_name
sentiment_score
sentiment_confidence
sentiment_uncertainty
sentiment_magnitude
news_event_impact
news_importance
risk_relevance
volatility_spike
drawdown_risk
news_uncertainty
sentiment_direction_score
news_direction_score
qualitative_score
qualitative_risk_score
qualitative_confidence
qualitative_recommendation
dominant_qualitative_driver
xai_summary
```

### 12.2 Daily Ticker-Level Output

Path:

```text
outputs/results/QualitativeAnalyst/qualitative_daily_chunk{chunk}_{split}.csv
```

Purpose:

```text
Provide ticker/date aligned qualitative signals for the Fusion Engine.
```

Important columns:

```text
ticker
date
chunk
split
event_count
sentiment_event_count
news_event_count
qualitative_score
qualitative_risk_score
qualitative_confidence
qualitative_recommendation
max_event_risk_score
mean_event_risk_score
mean_sentiment_score
mean_news_impact_score
mean_news_importance
dominant_qualitative_driver
xai_summary
```

---

## 13. Daily Aggregation Logic

The event-level output is aggregated by:

```text
ticker, date
```

### 13.1 Daily Qualitative Score

```text
daily_qualitative_score = confidence-weighted mean of event qualitative_score
```

### 13.2 Daily Qualitative Risk

```text
daily_qualitative_risk_score =
    0.60 × max(event_qualitative_risk_score)
  + 0.40 × weighted_mean(event_qualitative_risk_score)
```

This prevents severe risk disclosures from being averaged away by many neutral events.

### 13.3 Daily Qualitative Confidence

```text
daily_qualitative_confidence = mean(event_confidence) × min(1, log1p(event_count) / log(5))
```

This rewards repeated supporting evidence but caps the boost.

---

## 14. Recommendation Logic

The module converts qualitative score, risk, and confidence into a qualitative recommendation.

```text
BUY:
    qualitative_score > +0.20
    qualitative_risk_score < 0.70
    qualitative_confidence ≥ 0.40

SELL:
    qualitative_score < -0.20
    OR qualitative_risk_score > 0.85

HOLD:
    otherwise
```

This conservative logic ensures that the qualitative branch does not overreact to weak or uncertain text signals.

---

## 15. Explainability Design

The module provides both row-level and file-level XAI.

### 15.1 Row-Level XAI

Each event and daily row contains an `xai_summary` column.

Example:

```text
SELL: qualitative_score=-0.421, risk=0.881, confidence=0.624; driver=drawdown_risk; sentiment=-0.350; news_impact=-0.220.
```

### 15.2 Dominant Driver

Each event row includes:

```text
dominant_qualitative_driver
```

Possible values include:

```text
sentiment
news_impact
risk_relevance
volatility_spike
drawdown_risk
news_uncertainty
```

This tells which upstream factor most influenced the qualitative output.

### 15.3 Gradient-Based Feature Importance

The module computes input-gradient feature importance for a sample of rows.

XAI output path:

```text
outputs/results/QualitativeAnalyst/xai/qualitative_chunk{chunk}_{split}_xai_summary.json
```

The JSON contains:

```text
module
chunk
split
config
model metadata
event row count
daily row count
ticker count
date range
event recommendation counts
daily recommendation counts
dominant driver counts
gradient feature importance
top positive event examples
top negative event examples
highest risk event examples
plain-English explanation
summary statistics
```

---

## 16. Integration with Fusion Engine

The Fusion Engine should consume the daily qualitative file:

```text
outputs/results/QualitativeAnalyst/qualitative_daily_chunk{chunk}_{split}.csv
```

It should merge on:

```text
ticker, date
```

with:

```text
outputs/results/QuantitativeAnalyst/quantitative_analysis_chunk{chunk}_{split}.csv
outputs/results/PositionSizing/position_sizing_chunk{chunk}_{split}.csv
```

Important interpretation rule:

```text
Missing qualitative row = no text evidence available
```

It should not be interpreted as positive or negative sentiment.

Recommended Fusion defaults for missing qualitative evidence:

```text
qualitative_score = 0.0
qualitative_confidence = 0.0
qualitative_risk_score = 0.5 or neutral default
```

This ensures that absent text does not falsely strengthen the decision.

---

## 17. No Fundamentals Policy

This module intentionally excludes fundamentals.

It does not read:

```text
fundamental embeddings
fundamental analyst outputs
SEC tabular fundamentals
valuation ratios
balance-sheet strength outputs
```

The current qualitative branch is limited to:

```text
Sentiment Analyst + News Analyst
```

This preserves the current approved architecture.

---

## 18. Leakage Prevention

The module follows these anti-leakage rules:

1. It trains on the training split and validates on the validation split.
2. The feature scaler is fitted only on training data.
3. Validation and test features are transformed using the saved training scaler.
4. The module uses existing upstream predictions rather than future market outcomes.
5. It outputs event-level and daily-level signals using the event/filing date only.

The weak-supervised targets are derived from same-row upstream analyst outputs, so they do not introduce future price information.

---

## 19. Strengths

The Qualitative Analyst has several strengths:

1. It is trainable, satisfying the architecture requirement for a learned analyst layer.
2. It is lightweight and fast compared to retraining FinBERT.
3. It remains explainable because all inputs are interpretable upstream analyst scores.
4. It produces both event-level and daily-level outputs.
5. It is fusion-ready because daily outputs are ticker/date aligned.
6. It supports HPO, checkpointing, scaler persistence, validation, and XAI.

---

## 20. Limitations

This version has limitations that should be acknowledged in the report.

1. It uses weak-supervised targets, not manually labelled qualitative decisions.
2. It depends on the quality of upstream Sentiment and News Analyst outputs.
3. It does not directly read article text or SEC filing text.
4. It does not include fundamentals.
5. Text coverage is sparse, so many ticker-days may not have qualitative evidence.
6. If sentiment/news models are rerun, this module should also be rerun.

These limitations are acceptable because this module is a synthesis component, not a primary text encoder.

---

## 21. Main CLI Commands

### Compile

```bash
cd ~/fin-glassbox && python -m py_compile code/analysts/qualitative_analyst.py
```

### Inspect Inputs

```bash
cd ~/fin-glassbox && python code/analysts/qualitative_analyst.py inspect --repo-root .
```

### Smoke Test

```bash
cd ~/fin-glassbox && python code/analysts/qualitative_analyst.py smoke --repo-root . --device cuda
```

### HPO for One Chunk

```bash
cd ~/fin-glassbox && python code/analysts/qualitative_analyst.py hpo --repo-root . --chunk 1 --trials 30 --device cuda --fresh
```

### Train One Chunk with Best HPO Parameters

```bash
cd ~/fin-glassbox && python code/analysts/qualitative_analyst.py train-best --repo-root . --chunk 1 --device cuda --fresh
```

### Predict One Split

```bash
cd ~/fin-glassbox && python code/analysts/qualitative_analyst.py predict --repo-root . --chunk 1 --split test --device cuda
```

### Validate One Output

```bash
cd ~/fin-glassbox && python code/analysts/qualitative_analyst.py validate --repo-root . --chunk 1 --split test
```

---

## 22. Recommended Test Run

Before launching the full run, execute a small sanity run:

```bash
cd ~/fin-glassbox && python -m py_compile code/analysts/qualitative_analyst.py && python code/analysts/qualitative_analyst.py inspect --repo-root . && python code/analysts/qualitative_analyst.py smoke --repo-root . --device cuda
```

Then run one chunk with a small number of trials:

```bash
cd ~/fin-glassbox && python code/analysts/qualitative_analyst.py hpo --repo-root . --chunk 1 --trials 3 --device cuda --fresh && python code/analysts/qualitative_analyst.py train-best --repo-root . --chunk 1 --device cuda --fresh && python code/analysts/qualitative_analyst.py predict --repo-root . --chunk 1 --split val --device cuda && python code/analysts/qualitative_analyst.py predict --repo-root . --chunk 1 --split test --device cuda && python code/analysts/qualitative_analyst.py validate --repo-root . --chunk 1 --split test
```

If this passes, run the full version.

---

## 23. Full Qualitative Analyst Run

Use this after sentiment and news outputs exist for all chunks.

```bash
cd ~/fin-glassbox && python code/analysts/qualitative_analyst.py hpo --repo-root . --chunk 1 --trials 30 --device cuda --fresh && python code/analysts/qualitative_analyst.py train-best --repo-root . --chunk 1 --device cuda --fresh && python code/analysts/qualitative_analyst.py predict-all --repo-root . --chunks 1 --splits val test --device cuda && python code/analysts/qualitative_analyst.py hpo --repo-root . --chunk 2 --trials 30 --device cuda --fresh && python code/analysts/qualitative_analyst.py train-best --repo-root . --chunk 2 --device cuda --fresh && python code/analysts/qualitative_analyst.py predict-all --repo-root . --chunks 2 --splits val test --device cuda && python code/analysts/qualitative_analyst.py hpo --repo-root . --chunk 3 --trials 30 --device cuda --fresh && python code/analysts/qualitative_analyst.py train-best --repo-root . --chunk 3 --device cuda --fresh && python code/analysts/qualitative_analyst.py predict-all --repo-root . --chunks 3 --splits val test --device cuda
```

---

## 24. Comprehensive Upstream Rerun Command

After FinBERT, Sentiment Analyst, and News Analyst are fully rerun, the complete qualitative branch can be refreshed with this command:

```bash
cd ~/fin-glassbox && python code/analysts/sentiment_analyst.py hpo --repo-root . --chunk 1 --trials 30 --device cuda --fresh && python code/analysts/sentiment_analyst.py train-best --repo-root . --chunk 1 --device cuda --fresh && python code/analysts/sentiment_analyst.py predict --repo-root . --chunk 1 --split val --device cuda && python code/analysts/sentiment_analyst.py predict --repo-root . --chunk 1 --split test --device cuda && python code/analysts/news_analyst.py hpo --repo-root . --chunk 1 --trials 30 --device cuda --fresh && python code/analysts/news_analyst.py train-best --repo-root . --chunk 1 --device cuda --fresh && python code/analysts/news_analyst.py predict --repo-root . --chunk 1 --split val --device cuda && python code/analysts/news_analyst.py predict --repo-root . --chunk 1 --split test --device cuda && python code/analysts/qualitative_analyst.py hpo --repo-root . --chunk 1 --trials 30 --device cuda --fresh && python code/analysts/qualitative_analyst.py train-best --repo-root . --chunk 1 --device cuda --fresh && python code/analysts/qualitative_analyst.py predict-all --repo-root . --chunks 1 --splits val test --device cuda && python code/analysts/sentiment_analyst.py hpo --repo-root . --chunk 2 --trials 30 --device cuda --fresh && python code/analysts/sentiment_analyst.py train-best --repo-root . --chunk 2 --device cuda --fresh && python code/analysts/sentiment_analyst.py predict --repo-root . --chunk 2 --split val --device cuda && python code/analysts/sentiment_analyst.py predict --repo-root . --chunk 2 --split test --device cuda && python code/analysts/news_analyst.py hpo --repo-root . --chunk 2 --trials 30 --device cuda --fresh && python code/analysts/news_analyst.py train-best --repo-root . --chunk 2 --device cuda --fresh && python code/analysts/news_analyst.py predict --repo-root . --chunk 2 --split val --device cuda && python code/analysts/news_analyst.py predict --repo-root . --chunk 2 --split test --device cuda && python code/analysts/qualitative_analyst.py hpo --repo-root . --chunk 2 --trials 30 --device cuda --fresh && python code/analysts/qualitative_analyst.py train-best --repo-root . --chunk 2 --device cuda --fresh && python code/analysts/qualitative_analyst.py predict-all --repo-root . --chunks 2 --splits val test --device cuda && python code/analysts/sentiment_analyst.py hpo --repo-root . --chunk 3 --trials 30 --device cuda --fresh && python code/analysts/sentiment_analyst.py train-best --repo-root . --chunk 3 --device cuda --fresh && python code/analysts/sentiment_analyst.py predict --repo-root . --chunk 3 --split val --device cuda && python code/analysts/sentiment_analyst.py predict --repo-root . --chunk 3 --split test --device cuda && python code/analysts/news_analyst.py hpo --repo-root . --chunk 3 --trials 30 --device cuda --fresh && python code/analysts/news_analyst.py train-best --repo-root . --chunk 3 --device cuda --fresh && python code/analysts/news_analyst.py predict --repo-root . --chunk 3 --split val --device cuda && python code/analysts/news_analyst.py predict --repo-root . --chunk 3 --split test --device cuda && python code/analysts/qualitative_analyst.py hpo --repo-root . --chunk 3 --trials 30 --device cuda --fresh && python code/analysts/qualitative_analyst.py train-best --repo-root . --chunk 3 --device cuda --fresh && python code/analysts/qualitative_analyst.py predict-all --repo-root . --chunks 3 --splits val test --device cuda
```

---

## 25. Expected Final Outputs

After a successful full run, the following files should exist:

```text
outputs/models/QualitativeAnalyst/chunk1/best_model.pt
outputs/models/QualitativeAnalyst/chunk2/best_model.pt
outputs/models/QualitativeAnalyst/chunk3/best_model.pt

outputs/results/QualitativeAnalyst/qualitative_events_chunk1_val.csv
outputs/results/QualitativeAnalyst/qualitative_events_chunk1_test.csv
outputs/results/QualitativeAnalyst/qualitative_events_chunk2_val.csv
outputs/results/QualitativeAnalyst/qualitative_events_chunk2_test.csv
outputs/results/QualitativeAnalyst/qualitative_events_chunk3_val.csv
outputs/results/QualitativeAnalyst/qualitative_events_chunk3_test.csv

outputs/results/QualitativeAnalyst/qualitative_daily_chunk1_val.csv
outputs/results/QualitativeAnalyst/qualitative_daily_chunk1_test.csv
outputs/results/QualitativeAnalyst/qualitative_daily_chunk2_val.csv
outputs/results/QualitativeAnalyst/qualitative_daily_chunk2_test.csv
outputs/results/QualitativeAnalyst/qualitative_daily_chunk3_val.csv
outputs/results/QualitativeAnalyst/qualitative_daily_chunk3_test.csv

outputs/results/QualitativeAnalyst/xai/qualitative_chunk1_val_xai_summary.json
outputs/results/QualitativeAnalyst/xai/qualitative_chunk1_test_xai_summary.json
outputs/results/QualitativeAnalyst/xai/qualitative_chunk2_val_xai_summary.json
outputs/results/QualitativeAnalyst/xai/qualitative_chunk2_test_xai_summary.json
outputs/results/QualitativeAnalyst/xai/qualitative_chunk3_val_xai_summary.json
outputs/results/QualitativeAnalyst/xai/qualitative_chunk3_test_xai_summary.json
```

---

## 26. Validation Checklist

A qualitative run is considered successful if:

```text
1. Training completes and best_model.pt is saved.
2. scaler.npz exists for each chunk.
3. Event-level outputs exist for validation and test splits.
4. Daily outputs exist for validation and test splits.
5. XAI JSON outputs exist for validation and test splits.
6. qualitative_score is always within [-1, +1].
7. qualitative_risk_score is always within [0, 1].
8. qualitative_confidence is always within [0, 1].
9. Recommendation labels are only BUY, HOLD, SELL.
10. Fusion can merge daily qualitative outputs on ticker/date.
```

---

## 27. Report Description


> The Qualitative Analyst is a trainable synthesis module that combines outputs from the Sentiment Analyst and News Analyst into a daily ticker-level qualitative signal. Rather than directly reprocessing raw text, it learns a calibrated mapping from upstream text-specialist predictions into qualitative direction, qualitative risk, and qualitative confidence. The module is trained using transparent weak-supervised targets derived from sentiment polarity, news impact, risk relevance, volatility-spike likelihood, drawdown-risk likelihood, and uncertainty. It produces event-level outputs for auditability and ticker-date daily outputs for integration into the Fusion Engine. Explainability is provided through row-level explanations, dominant-driver attribution, and gradient-based feature importance.

---

## 28. Academic Talking Points


1. The module separates qualitative text evidence from quantitative market/risk evidence.
2. It does not duplicate FinBERT; it uses the specialised outputs already produced by FinBERT-based analysts.
3. It is trainable, but still interpretable because its input features are human-readable analyst scores.
4. It outputs direction, risk, and confidence separately.
5. Daily aggregation prevents the Fusion Engine from needing to handle raw event-level text rows.
6. XAI is built in through dominant-driver explanations and gradient feature importance.
7. Missing qualitative evidence is treated as absence of text evidence, not as positive or negative information.
8. The module can be rerun easily whenever FinBERT, Sentiment Analyst, or News Analyst outputs improve.

---

## 29. Summary

The Qualitative Analyst completes the qualitative branch of the architecture. It transforms sparse, event-level text analysis into structured, trainable, explainable daily qualitative signals. It is designed to be lightweight, reproducible, auditable, and directly compatible with the upcoming Fusion Engine.

Its main contribution is:

```text
Sentiment + News → trained qualitative direction/risk/confidence → Fusion-ready daily signal
```

This keeps the full system modular, explainable, and defensible.
