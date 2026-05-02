# Workflow Architecture Blueprint 

## 1. Document Purpose

This document is the **workflow specification** for the project:

**An Explainable Multimodal Neural Framework for Financial Risk Management**


This file documents:

- the final project workflow,
- the module boundaries,
- the current data flow,
- the training and inference sequence,
- the file contracts between modules,
- the integration rules,
- the no-leakage discipline,
- the remaining final-stage tasks,
- and the thesis-defensible reasoning behind the system.


---

## 2. Critical Architecture Update

The current implementation **does not include fundamentals**.

Older versions of the project architecture contained:

- Fundamental Encoder,
- Fundamental Analyst,
- fundamental embedding stream,
- intrinsic-value/fundamental-quality branch.

These are now intentionally excluded from the active workflow.

The project may still contain historical SEC fundamentals data and documentation because those datasets were collected and processed earlier. However, the **current model pipeline must not silently include fundamentals** unless they are explicitly reintroduced later.

### Active data families in the current build

The active implementation uses four modelling families:

1. **Time-series market data**
2. **Financial text data**
3. **Macro/regime data**
4. **Cross-asset relation/graph data**

Fundamentals are treated as historical project work, not an active model input.

---

## 3. Core Project Philosophy

The project does not attempt to solve financial risk management using a single monolithic black-box model.

Instead, it uses a **distributed, modular, explainable architecture**:

```text
Specialised modules → risk-aware synthesis → hybrid fusion → final decision + explanations
```

The system is built around the following principles:

| Principle | Meaning in This Project |
|---|---|
| Specialisation | Each model performs a bounded task: sentiment, news, volatility, drawdown, contagion, regime, etc. |
| Multimodality | The system combines market time series, text, macro data, and graph relations. |
| Explainability | Each module emits its own explanation trace; fusion also explains the combined decision. |
| Risk-first design | Risk modules are not decorative; they control sizing, gating, and final approval. |
| Chronological discipline | All training, validation, testing, normalisation, and PCA fitting must respect time order. |
| Buildability | The project prioritises a working, defensible system over unnecessary complexity. |
| Auditability | Every final decision should be traceable to module outputs and rule-barrier effects. |

The research argument is that transparency improves when a large decision problem is decomposed into specialised, inspectable components rather than hidden inside one oversized neural network.

---

## 4. Final High-Level Architecture

```text
INPUT DATA
├── Market data
│   ├── OHLCV
│   ├── returns
│   ├── engineered technical features
│   └── liquidity features
│
├── Financial text data
│   ├── SEC textual filings
│   ├── section-level text chunks
│   ├── filing metadata
│   └── FinBERT embeddings
│
├── Macro / regime data
│   ├── FRED interest-rate series
│   ├── yield curve features
│   ├── credit spread features
│   └── regime stress indicators
│
└── Cross-asset relation data
    ├── rolling correlation snapshots
    ├── ticker universe
    ├── sector metadata
    ├── beta estimates
    └── graph edge lists

ENCODERS
├── Temporal Encoder
│   └── 256-dimensional market embeddings
│
└── FinBERT Encoder
    └── 256-dimensional financial text embeddings

ANALYST MODULES
├── Technical Analyst
├── Sentiment Analyst
├── News Analyst
├── Qualitative Analyst
└── Quantitative Analyst

RISK ENGINE
├── Volatility Model
├── Drawdown Risk Model
├── Historical VaR Module
├── CVaR / Expected Shortfall Module
├── StemGNN Contagion Risk Module
├── Liquidity Risk Module
├── MTGNN Regime Risk Module
└── Position Sizing Engine

SYNTHESIS
├── Qualitative branch
│   └── Sentiment + News → daily qualitative signal
│
├── Quantitative branch
│   └── Technical + Risk + Position Sizing → trained risk-attention output
│
└── Fusion Engine
    ├── Layer 1: learned fusion weighting
    └── Layer 2: user-defined rule barrier

FINAL DECISION
└── Buy / Hold / Sell + confidence + position size + explanation
```

---

## 5. Chronological Chunking Strategy

The system is trained and evaluated using chronological chunks. This protects against look-ahead bias and creates realistic out-of-sample validation periods.

| Chunk | Train Period | Validation Period | Test Period | Purpose |
|---|---:|---:|---:|---|
| Chunk 1 | 2000–2004 | 2005 | 2006 | Early historical regime and first full pipeline integration |
| Chunk 2 | 2007–2014 | 2015 | 2016 | Financial crisis/post-crisis period and mid-sample robustness |
| Chunk 3 | 2017–2022 | 2023 | 2024 | Recent market period and final current-era evaluation |

### Mandatory chronological rules

1. Training data must never use validation or test information.
2. Normalisers must be fit on train only.
3. PCA projection must be fit on train only.
4. Rolling risk features must use only historical windows available at that date.
5. Future returns may be used only for target construction during training/evaluation, never as inference inputs.
6. Output files must preserve `ticker` and `date` for downstream joins.

---

## 6. Data Layer

### 6.1 Market data

Primary market files are stored under:

```text
data/yFinance/processed/
```

Important files include:

```text
returns_panel_wide.csv
returns_long.csv
ohlcv_final.csv
features_temporal.csv
liquidity_features.csv
```

The market pipeline produced a complete panel over approximately 2,500 tickers and about 6,286 trading days. This data drives:

- Temporal Encoder,
- Technical Analyst,
- Volatility Model,
- Drawdown Risk Model,
- VaR/CVaR,
- Liquidity Risk,
- cross-asset graph construction,
- Position Sizing,
- Quantitative Analyst,
- Fusion.

### 6.2 Text data

Financial text comes primarily from SEC filings. The text pipeline includes:

- raw filing downloads,
- cleaning,
- section extraction,
- quality filtering,
- text chunking,
- FinBERT MLM fine-tuning,
- 768-dimensional embedding extraction,
- train-only PCA projection to 256 dimensions,
- metadata preservation.

FinBERT outputs live under:

```text
outputs/embeddings/FinBERT/
```

Expected files:

```text
chunk{n}_train_embeddings.npy
chunk{n}_train_metadata.csv
chunk{n}_val_embeddings.npy
chunk{n}_val_metadata.csv
chunk{n}_test_embeddings.npy
chunk{n}_test_metadata.csv
```

### 6.3 Macro/regime data

FRED macro/regime data lives under:

```text
data/FRED_data/outputs/
```

Important file:

```text
macro_features_trading_days_clean.csv
```

This file contains trading-day-aligned macro features such as:

- Treasury yield levels,
- yield curve spreads,
- credit spread proxies,
- policy-rate indicators,
- energy and currency series,
- recession/regime proxy columns.

The active regime model uses these features alongside graph and embedding signals.

### 6.4 Cross-asset relation data

Cross-asset relation data lives under:

```text
data/graphs/
```

Important subfolders:

```text
data/graphs/snapshots/
data/graphs/metadata/
data/graphs/returns/
data/graphs/combined/
```

The relation graph data includes rolling correlation edge snapshots and metadata such as sector mapping, beta estimates, market-cap proxies, and universe membership. These outputs support:

- StemGNN contagion risk,
- MTGNN regime graph learning,
- graph-level XAI,
- systemic risk interpretation.

---

## 7. Encoder Layer

## 7.1 Temporal Encoder

The Temporal Encoder is the shared market representation module.

### Role

It converts market sequences into 256-dimensional embeddings. These embeddings are reused by multiple downstream market/risk modules.

### Why it exists

Market data is sequential. A single day of OHLCV is not enough to understand trend, momentum, volatility context, or pre-drawdown behaviour. The Temporal Encoder learns a reusable representation of recent market behaviour.

### Current output contract

```text
outputs/embeddings/TemporalEncoder/chunk{n}_{split}_embeddings.npy
outputs/embeddings/TemporalEncoder/chunk{n}_{split}_manifest.csv
```

The `.npy` file contains the embedding matrix. The manifest maps each row to:

```text
ticker,date
```

### Known final embedding examples

```text
chunk1_train_embeddings.npy: 3,065,000 × 256
chunk1_val_embeddings.npy:     555,000 × 256
chunk1_test_embeddings.npy:    552,500 × 256

chunk2_train_embeddings.npy: 4,960,000 × 256
chunk2_val_embeddings.npy:     555,000 × 256
chunk2_test_embeddings.npy:    555,000 × 256

chunk3_train_embeddings.npy: 3,700,000 × 256
chunk3_val_embeddings.npy:     550,000 × 256
chunk3_test_embeddings.npy:    547,500 × 256
```

### Downstream consumers

- Technical Analyst
- Volatility Model
- Drawdown Risk Model
- MTGNN Regime Risk
- Quantitative Analyst through downstream outputs

---

## 7.2 FinBERT Encoder

The FinBERT Encoder is the shared text representation module.

### Role

It turns SEC filing text chunks into 256-dimensional financial text embeddings.

### Why it exists

Financial text contains qualitative information that cannot be captured from price data alone:

- risk factors,
- management discussion,
- business changes,
- governance signals,
- legal disclosures,
- event tone,
- uncertainty,
- sentiment.

### Current output contract

```text
outputs/embeddings/FinBERT/chunk{n}_{split}_embeddings.npy
outputs/embeddings/FinBERT/chunk{n}_{split}_metadata.csv
outputs/embeddings/FinBERT/chunk{n}_{split}_manifest.json
```

### Projection discipline

The model may extract 768-dimensional FinBERT embeddings first, then project to 256 dimensions using IncrementalPCA.

The PCA must be fit on the train split only:

```text
Projection: IncrementalPCA fitted on train split only
```

### Downstream consumers

- Sentiment Analyst
- News Analyst
- Qualitative Analyst through Sentiment/News outputs
- MTGNN Regime Risk when text embeddings are used in graph state construction
- Fusion indirectly through qualitative outputs

---

## 8. Analyst Layer

The analyst layer converts encoder outputs and risk outputs into interpretable domain-specific signals.

---

## 8.1 Sentiment Analyst

### Input

```text
outputs/embeddings/FinBERT/chunk{n}_{split}_embeddings.npy
outputs/embeddings/FinBERT/chunk{n}_{split}_metadata.csv
```

### Output

```text
outputs/results/analysts/sentiment/chunk{n}_{split}_predictions.csv
outputs/embeddings/analysts/sentiment/chunk{n}_{split}_sentiment_embeddings.npy
```

### Main outputs

- `sentiment_score`
- `sentiment_confidence`
- `sentiment_uncertainty`
- `sentiment_magnitude`
- row-level XAI summary

### Purpose

The module provides the sentiment component of the qualitative branch.

---

## 8.2 News Analyst

### Input

FinBERT embeddings and text metadata.

### Output

```text
outputs/results/analysts/news/chunk{n}_{split}_news_predictions.csv
outputs/embeddings/analysts/news/chunk{n}_{split}_news_embeddings.npy
```

### Main outputs

- `news_event_impact`
- `news_importance`
- `risk_relevance`
- `volatility_spike`
- `drawdown_risk`
- `news_uncertainty`

### Purpose

The module interprets event relevance and qualitative risk from text.

---

## 8.3 Technical Analyst

### Input

Temporal Encoder embeddings and manifest.

### Output

```text
outputs/results/TechnicalAnalyst/technical_analysis_chunk{n}_{split}.csv
```

### Main outputs

- `trend_score`
- `momentum_score`
- `timing_confidence`
- `technical_confidence`
- technical direction signal
- attention/counterfactual/gradient explanations

### Purpose

The module converts temporal market embeddings into directional technical signals.

---

## 8.4 Qualitative Analyst

### Input

- Sentiment Analyst predictions
- News Analyst predictions

### Output

Event-level:

```text
outputs/results/QualitativeAnalyst/qualitative_events_chunk{n}_{split}.csv
```

Daily ticker-level:

```text
outputs/results/QualitativeAnalyst/qualitative_daily_chunk{n}_{split}.csv
```

### Main outputs

- `qualitative_score`
- `qualitative_risk_score`
- `qualitative_confidence`
- `qualitative_recommendation`
- `event_count`
- `dominant_qualitative_driver`
- `xai_summary`

### Purpose

The module creates the qualitative branch consumed by Fusion.

### Important sparsity note

Qualitative daily outputs are sparse compared with quantitative market rows. Fusion must handle missing qualitative rows by inserting a neutral no-text state.

---

## 8.5 Quantitative Analyst

### Input

The Quantitative Analyst consumes:

- Technical Analyst outputs,
- Volatility outputs,
- Drawdown outputs,
- VaR/CVaR outputs,
- StemGNN contagion outputs,
- Liquidity outputs,
- Regime outputs,
- Position Sizing outputs.

### Output

```text
outputs/results/QuantitativeAnalyst/quantitative_analysis_chunk{n}_{split}.csv
```

### Required final schema

The final Fusion-ready Quantitative Analyst output must contain:

```text
attention_pooled_risk_score
top_attention_risk_driver
risk_attention_volatility
risk_attention_drawdown
risk_attention_var_cvar
risk_attention_contagion
risk_attention_liquidity
risk_attention_regime
```

If these columns are absent, the file is from the older non-attention schema and must not be used for final Fusion training.

### Main outputs

- `quantitative_recommendation`
- `risk_adjusted_quantitative_signal`
- `technical_direction_score`
- `quantitative_risk_score`
- `quantitative_risk_state`
- `quantitative_confidence`
- `quantitative_action_strength`
- `recommended_capital_fraction`
- `attention_pooled_risk_score`
- `top_attention_risk_driver`
- risk attention weights
- detailed XAI summary

### Purpose

The module acts as the market/risk synthesis branch before final Fusion.

---

## 9. Risk Engine

The Risk Engine is the control centre of the project. It estimates different types of financial risk, then constrains decision confidence and position size.

---

## 9.1 Volatility Model

### Purpose

Volatility measures uncertainty and instability in price movement. Higher volatility means future price movement is less stable, so position sizing and confidence should be more conservative.

### Output examples

- `vol_10d`
- `vol_30d`
- `volatility_risk_score`
- `volatility_regime_label`
- `garch_vol`
- `recent_vol`
- `volatility_confidence`

### Integration

Consumed by:

- Position Sizing Engine
- Quantitative Analyst
- Fusion indirectly

---

## 9.2 Drawdown Risk Model

### Purpose

Drawdown risk estimates how far an asset could fall from a recent or future local peak. It captures path-dependent downside risk, not just volatility.

### Output examples

- `expected_drawdown_10d`
- `expected_drawdown_30d`
- `drawdown_risk_10d`
- `drawdown_risk_30d`
- `drawdown_risk_score`
- `recovery_days_10d`
- `recovery_days_30d`
- `confidence_10d`
- `confidence_30d`

### Integration

Drawdown risk caps position size and influences Quantitative Analyst risk attention.

---

## 9.3 Historical VaR Module

### Purpose

Value at Risk estimates a threshold loss under a historical return distribution.

Example interpretation:

```text
VaR 95% = -0.03
```

means that, historically, the asset lost more than 3% on about 5% of days in the rolling window.

### Output examples

- `var_95`
- `var_99`

### Integration

VaR contributes to tail-risk scoring and position sizing.

---

## 9.4 CVaR / Expected Shortfall Module

### Purpose

CVaR estimates the average loss beyond the VaR threshold. It is more informative than VaR when the tail loss is severe.

### Output examples

- `cvar_95`
- `cvar_99`
- `tail_ratio_95`
- `tail_ratio_99`

### Integration

CVaR is used by Position Sizing and Quantitative Analyst as a tail-risk component.

---

## 9.5 StemGNN Contagion Risk Module

### Purpose

Contagion risk estimates whether risk is spreading through cross-asset relationships. It captures systemic or relational risk that a single-asset time-series model may miss.

### Model

StemGNN is used for cross-asset contagion modelling.

### Output examples

```text
outputs/results/StemGNN/contagion_scores_chunk{n}_{split}.csv
```

Columns include multi-horizon contagion scores such as:

- `contagion_5d`
- `contagion_20d`
- `contagion_60d`

### Integration

Contagion risk is one of the dominant drivers in Quantitative Analyst risk attention and Position Sizing caps.

---

## 9.6 Liquidity Risk Module

### Purpose

Liquidity risk estimates whether a trade can be executed safely. An asset may look attractive but still be dangerous if volume is weak or execution cost is high.

### Output examples

- `liquidity_score`
- `slippage_estimate_pct`
- `days_to_liquidate_1M`
- `tradable`

### Integration

The Fusion rule barrier may force HOLD and zero position when liquidity is too low or `tradable == False`.

---

## 9.7 MTGNN Regime Risk Module

### Purpose

Regime risk identifies the market environment: calm, volatile, crisis, or rotation.

### Why this module exists

Sentiment and news can explain why a regime may be changing, but the regime model captures the actual market-behaviour state through graph, macro, and embedding features.

### Output examples

- `regime_label`
- `regime_confidence`
- `prob_calm`
- `prob_volatile`
- `prob_crisis`
- `prob_rotation`
- `graph_density`
- `avg_degree_norm`
- `learned_graph_stress`
- `macro_stress_score`

### Integration

Regime outputs influence:

- Position Sizing
- Quantitative Analyst
- Fusion rule barrier
- final decision explanation

---

## 9.8 Position Sizing Engine

### Purpose

Position sizing converts risk outputs into a recommended capital allocation.

### User-approved exposure policy

| Mode | Maximum per stock |
|---|---:|
| Conservative | 5% |
| Moderate / Default | 10% |
| Aggressive | 15% |

Additional crisis constraints:

| Situation | Maximum exposure |
|---|---:|
| Crisis regime, short/default horizon | 5% |
| Crisis regime, long horizon | 3% |

### Output examples

```text
outputs/results/PositionSizing/position_sizing_chunk{n}_{split}.csv
```

Important columns:

- `recommended_capital_fraction`
- `recommended_capital_pct`
- `position_fraction_of_max`
- `binding_cap_source`
- `hard_cap_applied`
- `size_bucket`
- `risk_budget_used`
- `size_reduction_reasons`

### Integration rule

Fusion may reduce the Position Sizing recommendation, but should not increase it beyond the risk-approved exposure.

---

## 10. Synthesis Layer

## 10.1 Qualitative branch

The qualitative branch compresses text-derived evidence into daily ticker-level signals.

```text
FinBERT → Sentiment Analyst
       → News Analyst
       → Qualitative Analyst
       → qualitative_daily_chunk{n}_{split}.csv
```

Qualitative branch output meaning:

| Field | Meaning |
|---|---|
| `qualitative_score` | Directional text-based signal, usually from negative to positive |
| `qualitative_risk_score` | Text-derived risk or uncertainty |
| `qualitative_confidence` | Confidence based on event count and model uncertainty |
| `event_count` | Number of matched events for ticker-date |
| `dominant_qualitative_driver` | Main qualitative explanation driver |

## 10.2 Quantitative branch

The quantitative branch compresses technical and risk-engine outputs into a dense market/risk decision signal.

```text
Technical Analyst
Risk Engine modules
Position Sizing
       ↓
Quantitative Analyst
       ↓
quantitative_analysis_chunk{n}_{split}.csv
```

Quantitative branch output meaning:

| Field | Meaning |
|---|---|
| `risk_adjusted_quantitative_signal` | Directional signal after risk adjustment |
| `quantitative_risk_score` | Learned/rule-combined risk intensity |
| `quantitative_confidence` | Confidence in quantitative signal |
| `recommended_capital_fraction` | Position sizing recommendation |
| `top_attention_risk_driver` | Highest-weight risk source |
| `risk_attention_*` | Learned attention weights over risk modules |

---

## 11. Fusion Engine

The Fusion Engine is hybrid.

It has two layers:

```text
Layer 1: Learned Fusion Model
Layer 2: User Rule Barrier
```

## 11.1 Layer 1 — Learned fusion

The learned layer combines quantitative and qualitative evidence.

It learns:

- quantitative branch weight,
- qualitative branch weight,
- fused signal,
- fused risk score,
- fused confidence,
- learned position multiplier,
- learned Buy/Hold/Sell logits.

Inputs include:

- Quantitative signal/risk/confidence,
- risk attention weights,
- Position Sizing recommendation,
- Qualitative score/risk/confidence,
- event count features,
- branch agreement/disagreement,
- text availability,
- crisis/high-risk/liquidity flags.

## 11.2 Layer 2 — User rule barrier

The rule barrier is the final safety layer.

The learned model proposes. The rule barrier approves, caps, vetoes, or modifies.

Rules include:

- if `tradable == False`, force HOLD and position 0,
- if liquidity is below threshold, force HOLD and position 0,
- if contagion risk is too high, disallow BUY and cap exposure,
- if drawdown risk is too high, cap exposure,
- if quantitative risk is too high, disallow BUY,
- if regime is crisis, cap exposure,
- final exposure must not exceed Position Sizing recommendation.

The final position should follow:

```text
final_position = min(
    position_sizing_recommendation,
    learned_position_suggestion,
    user_rule_cap
)
```

## 11.3 Fusion output contract

Expected output:

```text
outputs/results/FusionEngine/fused_decisions_chunk{n}_{split}.csv
```

Important columns:

- `final_recommendation`
- `final_fusion_signal`
- `final_fusion_risk_score`
- `final_fusion_confidence`
- `final_position_fraction`
- `final_position_pct`
- `learned_recommendation`
- `learned_quantitative_weight`
- `learned_qualitative_weight`
- `rule_changed_action`
- `rule_barrier_reasons`
- `fusion_xai_summary`

---

## 12. Final Decision Layer

The Final Trade Approver consumes Fusion output and emits the final user-facing decision.

Final output should include:

```text
Buy / Hold / Sell
Confidence Score
Position Size Recommendation
Risk Summary
Module-wise Explanation Trace
Final Explanation
```

In the current architecture, Fusion already performs much of the final decision logic. A thin final approver may still be useful for:

- formatting final outputs,
- enforcing global portfolio constraints,
- preparing Streamlit/UI output,
- creating final explanation cards,
- generating report-ready decision traces.

---

## 13. End-to-End Training Workflow

### 13.1 Data preparation

1. Build market data.
2. Build SEC text corpus.
3. Build FRED macro/regime table.
4. Build cross-asset graph snapshots.
5. Validate ticker/date alignment.

### 13.2 Encoder training

1. Run Temporal Encoder HPO.
2. Train Temporal Encoder per chunk.
3. Generate Temporal Encoder embeddings.
4. Run FinBERT HPO/fine-tuning.
5. Generate FinBERT embeddings.
6. Fit train-only PCA and project embeddings to 256 dimensions.

### 13.3 Analyst training

1. Train Sentiment Analyst.
2. Train News Analyst.
3. Train Technical Analyst.
4. Train Qualitative Analyst.
5. Generate qualitative daily outputs.

### 13.4 Risk engine training/calculation

1. Run VaR/CVaR.
2. Run Liquidity Risk.
3. Train Volatility Model.
4. Train Drawdown Risk Model.
5. Train/run StemGNN Contagion Risk.
6. Train/run MTGNN Regime Risk.
7. Run Position Sizing.

### 13.5 Quantitative synthesis

1. Train Quantitative Analyst.
2. Generate train/val/test quantitative outputs.
3. Validate attention schema.

### 13.6 Fusion

1. Inspect Fusion inputs.
2. Ensure train/val/test exists for Quantitative and Qualitative branches.
3. Ensure Quantitative files use trained attention schema.
4. Run Fusion HPO.
5. Train best Fusion model.
6. Predict val/test.
7. Validate Fusion outputs.
8. Generate final XAI reports.

---

## 14. End-to-End Inference Workflow

A future production inference day should follow this order:

```text
1. Load latest market data
2. Update market features
3. Generate temporal embedding
4. Load latest text/filings/news if available
5. Generate FinBERT embedding
6. Run Sentiment Analyst
7. Run News Analyst
8. Run Technical Analyst
9. Run risk modules
10. Run Position Sizing
11. Run Qualitative Analyst
12. Run Quantitative Analyst
13. Run Fusion Engine
14. Run Final Trade Approver
15. Export final decision + XAI report
```

For missing text on a given day, the system should use a neutral qualitative state:

```text
qualitative_score = 0.0
qualitative_risk_score = 0.5
qualitative_confidence = 0.0
event_count = 0
dominant_qualitative_driver = no_text_event
```

This prevents missing text from becoming falsely bullish or bearish.

---

## 15. File and Folder Contract

### Encoders

```text
code/encoders/temporal_encoder.py
code/encoders/finbert_encoder.py
outputs/embeddings/TemporalEncoder/
outputs/embeddings/FinBERT/
outputs/models/TemporalEncoder/
outputs/models/FinBERT/
```

### Analysts

```text
code/analysts/sentiment_analyst.py
code/analysts/news_analyst.py
code/analysts/technical_analyst.py
code/analysts/qualitative_analyst.py
code/analysts/quantitative_analyst.py
outputs/results/analysts/
outputs/results/TechnicalAnalyst/
outputs/results/QualitativeAnalyst/
outputs/results/QuantitativeAnalyst/
```

### Risk engine

```text
code/riskEngine/volatility.py
code/riskEngine/drawdown.py
code/riskEngine/var_cvar_liquidity.py
code/riskEngine/regime_gnn.py
code/riskEngine/position_sizing.py
outputs/results/risk/
outputs/results/StemGNN/
outputs/results/MTGNNRegime/
outputs/results/PositionSizing/
```

### GNN modules

```text
code/gnn/stemgnn_contagion.py
code/gnn/stemgnn_base_model.py
code/gnn/mtgnn_regime.py
```

### Fusion

```text
code/fusion/fusion_layer.py
code/fusion/final_fusion.py
outputs/results/FusionEngine/
outputs/models/FusionEngine/
outputs/codeResults/FusionEngine/
```

---

## 16. Validation and Readiness Audits

### 16.1 Quantitative schema audit

Fusion must not consume old Quantitative files.

A valid Fusion-ready Quantitative output must satisfy:

```text
attention_schema=True
old_schema=False
```

Required columns:

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

### 16.2 Train/val/test availability audit

Before Fusion training, confirm:

```text
QuantitativeAnalyst:
quantitative_analysis_chunk{n}_train.csv
quantitative_analysis_chunk{n}_val.csv
quantitative_analysis_chunk{n}_test.csv

QualitativeAnalyst:
qualitative_daily_chunk{n}_train.csv
qualitative_daily_chunk{n}_val.csv
qualitative_daily_chunk{n}_test.csv
```

### 16.3 XAI audit

Each module should produce:

- predictions/results CSV,
- XAI JSON or XAI summary column,
- finite numeric values,
- explanation fields that match the model output.

---

## 17. No-Leakage Policy

This project depends heavily on point-in-time correctness.

The following are forbidden:

- fitting normalisers on validation/test data,
- fitting PCA on validation/test embeddings,
- using future returns as inference features,
- using test labels for HPO,
- using future graph snapshots for earlier dates,
- merging text by year only when exact filing date is available,
- training Fusion on stale old-schema outputs without detection,
- allowing final Fusion to increase position size above Position Sizing output.

---

## 18. Current Practical Completion State

At the time this replacement workflow was written, the project had reached the final integration stage.

Completed or largely completed:

- market data pipeline,
- SEC text pipeline,
- FRED macro data pipeline,
- cross-asset graph builder,
- Temporal Encoder embeddings for all three chunks,
- FinBERT embeddings and improved runs in progress/completing,
- Sentiment Analyst,
- News Analyst,
- Technical Analyst,
- Qualitative Analyst,
- Volatility Model,
- Drawdown Model,
- VaR/CVaR,
- Liquidity Risk,
- StemGNN Contagion Risk,
- MTGNN Regime Risk,
- Position Sizing Engine,
- Quantitative Analyst with trained risk attention,
- Fusion module smoke test.

Remaining final-stage work:

1. Ensure Quantitative Analyst chunk 2 and chunk 3 have final attention-schema outputs for train/val/test.
2. Generate Qualitative Analyst train outputs for all chunks.
3. Run Fusion chunk 1 fully.
4. Run Fusion chunks 2 and 3 after schema readiness.
5. Build or finalise Final Trade Approver if needed.
6. Run end-to-end output audit.
7. Run XAI audit.
8. Prepare final evaluation/backtesting and thesis/report material.

---

## 19. Thesis-Defensible System Description

This project proposes An Explainable Multimodal Neural Framework for Financial Risk Management. Instead of relying on a monolithic black-box model, the system decomposes the decision process into specialised modules for market sequence understanding, textual analysis, risk estimation, graph-based contagion modelling, regime detection, position sizing, and final fusion. Each module produces interpretable intermediate outputs and explanation traces. A hybrid Fusion Engine combines learned evidence weighting with a user-controlled rule barrier, ensuring that the final decision remains risk-aware, auditable, and practically controllable. The architecture is evaluated using chronological train/validation/test chunks to reduce look-ahead bias and improve realism.

---

## 20. Final Notes

This workflow is the active project workflow.

Do not reintroduce fundamentals, Bull/Bear debaters, or monolithic agent orchestration unless explicitly approved.

The current priority is not adding more architectural complexity. The priority is:

```text
complete integration → validate outputs → audit XAI → evaluate results → document clearly
```
