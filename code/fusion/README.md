# `code/fusion/` — Hybrid Fusion Engine

## 1. Purpose

The `code/fusion/` directory contains the final synthesis layer for **An Explainable Multimodal Neural Framework for Financial Risk Management**. Its job is to combine the two high-level analysis branches produced by the system:

1. **Quantitative Analyst** — dense ticker-date market/risk analysis based on technical signals, volatility, drawdown, VaR/CVaR, liquidity, contagion, regime risk, and position sizing.
2. **Qualitative Analyst** — sparse event/date text analysis based on FinBERT-derived sentiment and news/event impact.

The Fusion Engine converts these branches into a final risk-aware decision object:

```text
Buy / Hold / Sell
├── final fused signal
├── final fused risk score
├── final confidence score
├── final position size
├── learned quantitative/qualitative branch weights
├── rule-barrier override reasons
└── system-level XAI explanation
```

This module is intentionally **hybrid**. It is not a pure neural black box. It uses a learned fusion layer to estimate branch weights and final signal components, then applies a user-controlled rule barrier as the final line of defence.

---

## 2. Directory contents

```text
code/fusion/
├── fusion_layer.py
└── final_fusion.py
```

### `fusion_layer.py`

Core implementation file. It contains:

- configuration dataclasses,
- input schema validation,
- quantitative/qualitative branch loading,
- branch merge logic,
- feature engineering,
- target construction,
- learned fusion model,
- Optuna HPO objective,
- training loop,
- prediction loop,
- user rule barrier,
- validation utilities,
- smoke test,
- and XAI report generation.

Important classes and functions include:

| Object | Role |
|---|---|
| `UserRuleBarrierConfig` | Stores user-controlled safety rules and exposure caps. |
| `FusionConfig` | Stores paths, training settings, HPO settings, schema behaviour, and rule settings. |
| `HybridFusionModel` | Neural fusion model that learns branch weights, signal, risk, confidence, position multiplier, and Buy/Hold/Sell logits. |
| `FusionScaler` | Train-fitted feature normaliser saved with the model. |
| `merge_branches()` | Joins quantitative and qualitative branch outputs by `ticker` and `date`. |
| `prepare_fusion_dataframe()` | Cleans and engineers fusion features. |
| `construct_fusion_targets()` | Builds self-supervised risk-aware training targets. |
| `apply_user_rule_barrier()` | Applies hard safety constraints after learned prediction. |
| `predict_fusion()` | Produces final fused decisions and XAI outputs. |
| `validate_predictions()` | Verifies output range, branch weights, and required schema. |

### `final_fusion.py`

Thin CLI wrapper around `fusion_layer.py`. It exposes the runnable interface:

```text
inspect
smoke
hpo
train-best
predict
predict-all
validate
run
run-all
```

The file is intentionally small. All substantive modelling logic remains in `fusion_layer.py`, while `final_fusion.py` provides a clean command-line entry point for experiments, validation, and production-style execution.

---

## 3. Architectural position

The Fusion Engine sits after the two analysis branches:

```text
Risk Engine + Technical Analyst
        │
        ▼
Quantitative Analyst
        │
        ├──────────────┐
        │              ▼
        │        Fusion Engine ─────► Final Decision / Trade Approver
        │              ▲
        └──────────────┤
                       │
Sentiment + News + FinBERT
        │
        ▼
Qualitative Analyst
```

The Fusion Engine is the final synthesis stage before final trade approval. It does not replace the risk engine. Instead, it respects the risk engine by enforcing the principle:

```text
Fusion may reduce exposure, but it must not exceed the Position Sizing recommendation.
```

---

## 4. Hybrid fusion design

### 4.1 Layer 1 — learned fusion model

The learned layer estimates how to combine quantitative and qualitative evidence. It learns:

- `learned_quantitative_weight`,
- `learned_qualitative_weight`,
- `learned_fusion_signal`,
- `learned_fusion_risk_score`,
- `learned_fusion_confidence`,
- `learned_position_multiplier`,
- `learned_sell_prob`,
- `learned_hold_prob`,
- `learned_buy_prob`,
- `learned_recommendation`.

The model is intentionally compact. It uses an MLP backbone with `tanh` activations and separate heads for branch weights, action logits, signal, risk, confidence, and position scaling. This keeps the layer trainable, inspectable, and defensible.

### 4.2 Layer 2 — user rule barrier

The user rule barrier is not learned. It is a deterministic final safety layer. It applies constraints such as:

- do not trade if `tradable == False`,
- block or reduce exposure when liquidity is too low,
- disallow BUY when quantitative risk is too high,
- cap exposure under high drawdown risk,
- cap exposure under high contagion risk,
- cap exposure during crisis regimes,
- ensure final exposure never exceeds the Position Sizing recommendation.

The final position is computed using the conservative rule:

```text
final_position = min(
    position_sizing_recommendation,
    learned_position_suggestion,
    user_rule_cap
)
```

This design preserves transparency and ensures the learned model cannot override explicit user/risk constraints.

---

## 5. Input contracts

### 5.1 Quantitative Analyst input

Expected path:

```text
outputs/results/QuantitativeAnalyst/quantitative_analysis_chunk{chunk}_{split}.csv
```

Required attention-schema columns include:

```text
ticker
date
quantitative_recommendation
risk_adjusted_quantitative_signal
technical_direction_score
quantitative_risk_score
quantitative_confidence
quantitative_action_strength
recommended_capital_fraction
recommended_capital_pct
position_fraction_of_max
max_single_stock_exposure
attention_pooled_risk_score
top_attention_risk_driver
risk_attention_volatility
risk_attention_drawdown
risk_attention_var_cvar
risk_attention_contagion
risk_attention_liquidity
risk_attention_regime
```

The Fusion Engine deliberately fails if it detects the older Quantitative Analyst schema containing `top_risk_driver` without `top_attention_risk_driver`. This prevents stale rule-only quantitative outputs from silently corrupting fusion training.

### 5.2 Qualitative Analyst input

Expected path:

```text
outputs/results/QualitativeAnalyst/qualitative_daily_chunk{chunk}_{split}.csv
```

Important columns include:

```text
ticker
date
event_count
sentiment_event_count
news_event_count
qualitative_score
qualitative_risk_score
qualitative_confidence
qualitative_recommendation
mean_sentiment_score
mean_news_impact_score
dominant_qualitative_driver
xai_summary
```

Qualitative data is sparse. Most ticker-date rows do not have text events. When a qualitative row is missing for a quantitative ticker-date, the Fusion Engine uses a neutral qualitative state:

```text
qualitative_score = 0.0
qualitative_risk_score = 0.5
qualitative_confidence = 0.0
event_count = 0
dominant_qualitative_driver = no_text_event
```

This prevents missing text from being interpreted as bullish or bearish.

---

## 6. Output contracts

Predictions are written to:

```text
outputs/results/FusionEngine/fused_decisions_chunk{chunk}_{split}.csv
```

XAI summaries are written to:

```text
outputs/results/FusionEngine/xai/fused_decisions_chunk{chunk}_{split}_xai_summary.json
```

Model artefacts are written to:

```text
outputs/models/FusionEngine/chunk{chunk}/
├── best_model.pt
├── final_model.pt
├── scaler.npz
├── training_history.csv
├── training_summary.json
├── model_freezed/model.pt
└── model_unfreezed/model.pt
```

HPO artefacts are written to:

```text
outputs/codeResults/FusionEngine/
├── hpo_chunk{chunk}.db
└── best_params_chunk{chunk}.json
```

---

## 7. Main output columns

The fused CSV contains both final outputs and audit traces. Important columns include:

| Column | Meaning |
|---|---|
| `final_recommendation` | Final post-rule Buy/Hold/Sell decision. |
| `final_fusion_signal` | Final fused directional signal after learned model. |
| `final_fusion_risk_score` | Final fused risk score. |
| `final_fusion_confidence` | Final confidence score. |
| `final_position_fraction` | Final capital allocation as a fraction. |
| `final_position_pct` | Final capital allocation as a percentage. |
| `learned_recommendation` | Learned model recommendation before rule barrier. |
| `learned_quantitative_weight` | Learned trust weight assigned to quantitative branch. |
| `learned_qualitative_weight` | Learned trust weight assigned to qualitative branch. |
| `branch_weight_dominance` | Which branch dominated the fusion decision. |
| `rule_changed_action` | Whether the rule barrier changed the learned action. |
| `rule_barrier_reasons` | Human-readable reason for safety caps/vetoes. |
| `fusion_xai_summary` | Compact final explanation. |

---

## 8. XAI design

Fusion-level XAI has three layers:

### 8.1 Branch-weight explanation

The model reports how much it trusted each branch:

```text
learned_quantitative_weight
learned_qualitative_weight
branch_weight_dominance
```

This answers: **Was the decision primarily market/risk-driven or text/event-driven?**

### 8.2 Risk-driver explanation

The quantitative branch carries risk attention weights:

```text
risk_attention_volatility
risk_attention_drawdown
risk_attention_var_cvar
risk_attention_contagion
risk_attention_liquidity
risk_attention_regime
top_attention_risk_driver
```

This answers: **Which risk type most influenced the quantitative branch?**

### 8.3 Rule-barrier explanation

The rule layer records why the final output was capped or modified:

```text
rule_changed_action
user_rule_cap_fraction
rule_barrier_reasons
```

This answers: **Did the safety layer intervene, and why?**

---

## 9. CLI usage

### 9.1 Compile

```bash
cd ~/fin-glassbox && python -m py_compile code/fusion/fusion_layer.py code/fusion/final_fusion.py
```

### 9.2 Inspect inputs

```bash
cd ~/fin-glassbox && python code/fusion/final_fusion.py inspect --repo-root .
```

### 9.3 Smoke test

```bash
cd ~/fin-glassbox && python code/fusion/final_fusion.py smoke --repo-root . --device cuda
```

### 9.4 HPO

```bash
cd ~/fin-glassbox && python code/fusion/final_fusion.py hpo --repo-root . --chunk 1 --trials 30 --device cuda --fresh
```

### 9.5 Train best model

```bash
cd ~/fin-glassbox && python code/fusion/final_fusion.py train-best --repo-root . --chunk 1 --device cuda --fresh
```

### 9.6 Predict one split

```bash
cd ~/fin-glassbox && python code/fusion/final_fusion.py predict --repo-root . --chunk 1 --split test --device cuda
```

### 9.7 Predict multiple splits

```bash
cd ~/fin-glassbox && python code/fusion/final_fusion.py predict-all --repo-root . --chunks 1 --splits val test --device cuda
```

### 9.8 Validate output

```bash
cd ~/fin-glassbox && python code/fusion/final_fusion.py validate --repo-root . --chunk 1 --split test
```

### 9.9 Full chunk run

```bash
cd ~/fin-glassbox && python code/fusion/final_fusion.py run --repo-root . --chunk 1 --trials 30 --device cuda --fresh --predict-splits val test
```

### 9.10 Full all-chunk run

Only run this after all chunks have the trained Quantitative attention schema:

```bash
cd ~/fin-glassbox && python code/fusion/final_fusion.py run-all --repo-root . --chunks 1 2 3 --trials 30 --device cuda --fresh --predict-splits val test
```

---

## 10. Readiness checks before full run

Before training Fusion on a chunk, verify:

1. Quantitative train/val/test files exist.
2. Quantitative files use the attention schema.
3. Qualitative train/val/test files exist, or missing qualitative behaviour is intentionally neutral.
4. `ticker` and `date` columns are present in both branches.
5. Position Sizing outputs are already reflected inside Quantitative Analyst outputs.
6. No old rule-only Quantitative files are accidentally being used.

Recommended schema audit:

```bash
cd ~/fin-glassbox && python - <<'PY'
import pandas as pd
from pathlib import Path
for p in sorted(Path("outputs/results/QuantitativeAnalyst").glob("quantitative_analysis_chunk*_*.csv")):
    df = pd.read_csv(p, nrows=5)
    has_attention = "top_attention_risk_driver" in df.columns
    has_old = "top_risk_driver" in df.columns
    print(f"{p}: attention_schema={has_attention}, old_schema={has_old}, rows~", sum(1 for _ in open(p))-1)
PY
```

---

## 11. Rule barrier defaults

Default user-facing exposure profile:

```text
conservative: 5% max per stock
moderate:     10% max per stock
aggressive:   15% max per stock
```

Crisis-specific caps:

```text
short-horizon crisis cap: 5%
long-horizon crisis cap:  3%
```

Important default constraints:

```text
not tradable            -> HOLD, 0% position
low liquidity           -> HOLD, 0% position
high contagion          -> BUY veto / severe cap
high drawdown           -> cap exposure
high quantitative risk  -> BUY veto / cap exposure
position sizing output  -> upper bound on final exposure
```

---

## 12. Why this design is thesis-defensible

A pure learned fusion model would be difficult to explain and could override risk constraints. A pure rule-based fusion model would be transparent but less adaptive. This module combines both:

- the learned layer captures how quantitative and qualitative signals should be weighted;
- the rule barrier preserves safety, transparency, and user control;
- branch weights expose the model’s internal trust allocation;
- rule reasons expose final overrides;
- all outputs remain auditable at ticker-date level.

This is aligned with the project philosophy:

```text
specialisation + multimodality + explainability + modular integration + risk-aware decision-making
```

---

## 13. Common failure modes

### Old Quantitative schema detected

Cause: chunk output came from the old Quantitative Analyst version.

Fix: rerun the trained attention-based Quantitative Analyst for that chunk/split.

### Missing train outputs

Cause: only val/test predictions were generated.

Fix: generate train predictions for both Quantitative and Qualitative branches before Fusion HPO/training.

### All outputs are HOLD

Possible causes:

- learned signal thresholds are conservative,
- position size is capped to zero or near zero,
- risk barrier is frequently intervening,
- upstream Quantitative/Qualitative outputs are near-neutral,
- qualitative coverage is sparse.

Check:

```text
learned_recommendation
final_recommendation
rule_changed_action
rule_barrier_reasons
learned_fusion_signal
final_position_pct
```

### Qualitative influence is near zero

This may be correct when no text event exists for most ticker-date rows. Check:

```text
text_available
event_count
qualitative_confidence
learned_qualitative_weight
```

---


`code/fusion/` contains the final hybrid synthesis layer of the system. It is deliberately designed as a compact but explainable module:

```text
learned branch weighting
+ learned signal/risk/confidence estimation
+ user-controlled rule barrier
+ final position cap
+ module-level and system-level XAI
```

It converts the project from a collection of specialised modules into a coherent final decision system while preserving the central role of risk management and explainability.
