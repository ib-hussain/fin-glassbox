# Quantitative Synthesis Layer Analyst 

## 1. Module identity

**File:** `code/analysts/quantitative_analyst.py`  
**Role:** trained quantitative branch synthesis module  
**Branch:** quantitative branch  
**Upstream dependency:** Position Sizing Engine output  
**Downstream consumer:** Fusion Engine

The Quantitative Analyst combines technical and risk-engine outputs into a trained quantitative branch signal. It does not make the final decision. Its main architectural requirement is:

```text
attention-weighted pooling across risk scores
```

This allows the model to explain which risk family dominated the quantitative judgement for each ticker-date.

---

## 2. Architectural position

```text
Technical Analyst
Risk Engine modules
Position Sizing Engine
        ↓
Quantitative Analyst
        ↓
Fusion Engine
        ↓
Final Trade Approver
```

The Position Sizing Engine remains an upstream risk-control block. The Quantitative Analyst learns a synthesis of its output and carries the risk context forward to Fusion.

---

## 3. Input contract

The module reads:

```text
outputs/results/PositionSizing/position_sizing_chunk{chunk}_{split}.csv
```

This file already contains joined outputs from:

```text
Technical Analyst
Volatility model
Drawdown model
VaR/CVaR
StemGNN contagion risk
Liquidity module
Regime model
Position Sizing rules
```

The Quantitative Analyst therefore does not independently load every risk module. It uses the integrated Position Sizing output as its base frame.

---

## 4. Risk attention inputs

The six risk-token inputs are:

```text
volatility_risk_score
drawdown_risk_score
var_cvar_risk_score
contagion_risk_score
liquidity_risk_score
regime_risk_score
```

These correspond to the project’s risk-engine submodules.

---

## 5. Context features

The model also uses context features such as:

```text
trend_score
momentum_score
timing_confidence
technical_confidence
combined_risk_score
regime_confidence
recommended_capital_fraction
recommended_capital_pct
position_fraction_of_max
max_single_stock_exposure
prob_calm
prob_volatile
prob_crisis
prob_rotation
```

A train-fitted `ContextScaler` is saved as:

```text
outputs/models/QuantitativeAnalyst/chunk{chunk}/scaler.npz
```

---

## 6. Target construction

The model learns a rule-derived, risk-aware quantitative target.

### 6.1 Technical direction

```text
technical_direction_score =
    0.40 × trend_score
  + 0.35 × momentum_score
  + 0.25 × timing_confidence
```

### 6.2 Risk-adjusted signal

```text
risk_gate = 1 - combined_risk_score
position_gate = 0 if position_fraction_of_max is effectively zero else 0.5 + 0.5 × position_fraction_of_max
target_quantitative_signal = technical_direction_score × risk_gate × position_gate
```

The target signal is clipped to `[-1, 1]`.

### 6.3 Risk target

```text
target_quantitative_risk = combined_risk_score
```

### 6.4 Confidence target

```text
target_quantitative_confidence =
    0.40 × technical_confidence
  + 0.20 × risk_gate
  + 0.25 × position_fraction
  + 0.15 × regime_confidence
```

This produces a continuous confidence target rather than a binary confidence flag.

---

## 7. Model architecture

Primary model class:

```text
QuantitativeRiskAttentionModel
```

Architecture:

```text
risk_scores, shape = (batch, 6)
context_features, shape = (batch, context_dim)
        ↓
risk value projection
risk identity embedding
context projection
        ↓
attention score per risk driver
        ↓
attention-weighted pooled risk
        ↓
MLP with Tanh activations
        ↓
quantitative signal
quantitative risk
quantitative confidence
```

Outputs:

```text
risk_adjusted_quantitative_signal
quantitative_risk_score
quantitative_confidence
attention_pooled_risk_score
risk_attention_volatility
risk_attention_drawdown
risk_attention_var_cvar
risk_attention_contagion
risk_attention_liquidity
risk_attention_regime
top_attention_risk_driver
```

---

## 8. Decision mapping

The model maps continuous outputs to:

```text
BUY
HOLD
SELL
```

using thresholds including:

```text
buy_threshold
sell_threshold
max_risk_for_buy
severe_risk_sell_threshold
min_confidence_for_buy
min_position_fraction
```

This recommendation is a **branch-level quantitative recommendation**, not the final system recommendation.

---

## 9. Output contract

Model files:

```text
outputs/models/QuantitativeAnalyst/chunk{chunk}/best_model.pt
outputs/models/QuantitativeAnalyst/chunk{chunk}/final_model.pt
outputs/models/QuantitativeAnalyst/chunk{chunk}/scaler.npz
outputs/models/QuantitativeAnalyst/chunk{chunk}/training_history.csv
```

HPO files:

```text
outputs/codeResults/QuantitativeAnalyst/hpo_chunk{chunk}.db
outputs/codeResults/QuantitativeAnalyst/best_params_chunk{chunk}.json
```

Prediction files:

```text
outputs/results/QuantitativeAnalyst/quantitative_analysis_chunk{chunk}_{split}.csv
outputs/results/QuantitativeAnalyst/xai/quantitative_analysis_chunk{chunk}_{split}_xai_summary.json
```

---

## 10. Required Fusion schema

Fusion should consume only the trained attention schema. Required fields include:

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

Old outputs with `top_risk_driver` instead of `top_attention_risk_driver` should be regenerated.

---

## 11. XAI design

The Quantitative Analyst explains itself through:

1. Risk attention weights.
2. Top attention risk driver.
3. Attention-pooled risk score.
4. Row-level `xai_summary`.
5. Split-level XAI JSON summaries.

This makes it clear whether the model relied mostly on volatility, drawdown, VaR/CVaR, contagion, liquidity, or regime risk.

---

## 12. CLI commands

Inspect:

```bash
python code/analysts/quantitative_analyst.py inspect --repo-root .
```

Smoke:

```bash
python code/analysts/quantitative_analyst.py smoke --repo-root . --device cuda
```

HPO:

```bash
python code/analysts/quantitative_analyst.py hpo --repo-root . --chunk 1 --trials 30 --device cuda --fresh
```

Train best:

```bash
python code/analysts/quantitative_analyst.py train-best --repo-root . --chunk 1 --device cuda --fresh
```

Predict all Fusion-required splits for one chunk:

```bash
python code/analysts/quantitative_analyst.py predict-all --repo-root . --chunks 1 --splits train val test --device cuda
```

Validate:

```bash
python code/analysts/quantitative_analyst.py validate --repo-root . --chunk 1 --split test
```

Full rerun after risk modules are refreshed:

```bash
python code/analysts/quantitative_analyst.py hpo --repo-root . --chunk 1 --trials 30 --device cuda --fresh && python code/analysts/quantitative_analyst.py train-best --repo-root . --chunk 1 --device cuda --fresh && python code/analysts/quantitative_analyst.py predict-all --repo-root . --chunks 1 --splits train val test --device cuda && python code/analysts/quantitative_analyst.py hpo --repo-root . --chunk 2 --trials 30 --device cuda --fresh && python code/analysts/quantitative_analyst.py train-best --repo-root . --chunk 2 --device cuda --fresh && python code/analysts/quantitative_analyst.py predict-all --repo-root . --chunks 2 --splits train val test --device cuda && python code/analysts/quantitative_analyst.py hpo --repo-root . --chunk 3 --trials 30 --device cuda --fresh && python code/analysts/quantitative_analyst.py train-best --repo-root . --chunk 3 --device cuda --fresh && python code/analysts/quantitative_analyst.py predict-all --repo-root . --chunks 3 --splits train val test --device cuda
```

---
The Quantitative Analyst is a trained risk-attention synthesis module. It preserves risk-module identities, exposes learned risk attention weights, produces quantitative signal/risk/confidence outputs, and remains subordinate to the Fusion Engine and final rule barrier.
