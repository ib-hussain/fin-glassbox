# `code/riskEngine/` Folder Documentation

## 1. Document Purpose

This document is the replacement folder-level documentation for:

```text
code/riskEngine/
```

inside the `fin-glassbox` project:

**An Explainable Distributed Deep Learning Framework for Financial Risk Management**

It documents the role of the Risk Engine directory, the files inside it, the execution order, the input/output contracts, the XAI design, and how this directory connects to the rest of the system.


---

## 2. Risk Engine Role in the Full System

The Risk Engine is the control layer of the project. It converts market, graph, macro, and learned embedding evidence into risk estimates and position constraints.

```text
Temporal Encoder
Technical Analyst
Cross-asset graph data
FRED macro data
Market returns / OHLCV / liquidity features
        │
        ▼
code/riskEngine/
        ├── Volatility Risk
        ├── Drawdown Risk
        ├── Historical VaR
        ├── CVaR / Expected Shortfall
        ├── Liquidity Risk
        ├── Regime Detection
        └── Position Sizing
        │
        ▼
Quantitative Analyst
        │
        ▼
Fusion Engine
        │
        ▼
Final Trade Approver
```

The Risk Engine does not exist merely to add extra features. It is central to the project’s thesis: a financial AI system becomes more defensible when risk is decomposed into interpretable, specialised modules.

---

## 3. Directory Files

The current Risk Engine directory contains the following important files:

| File | Role | Documentation |
|---|---|---|
| `volatility.py` | GARCH + MLP hybrid volatility estimator | [`Volatility_Risk_Module.md`](Volatility_Risk_Module.md) |
| `volatility_manifest_generator.py` | helper for embedding manifest generation/auditing | [`Volatility_Risk_Module.md`](Volatility_Risk_Module.md) |
| `drawdown.py` | BiLSTM + attention dual-horizon drawdown model | [`Drawdown_Risk_Module.md`](Drawdown_Risk_Module.md) |
| `var_cvar_liquidity.py` | historical VaR, CVaR, and liquidity risk calculations | [`VaR_CVaR_Liquidity.md`](VaR_CVaR_Liquidity.md) |
| `regime_gnn.py` | thin Risk Engine wrapper around `code/gnn/mtgnn_regime.py` | [`Regime_Detection_Module.md`](Regime_Detection_Module.md) |
| `position_sizing.py` | rule-based, user-adjustable capital allocation engine | [`Position_Sizing_Engine.md`](Position_Sizing_Engine.md) |
| `README.md` | this folder-level documentation | Current file |

The graph-heavy implementation for regime detection is intentionally kept in `code/gnn/`. The Risk Engine wrapper exposes it under the risk namespace without duplicating graph code.

---

## 4. Module Categories

The Risk Engine contains three types of modules.

### 4.1 Learned neural risk models

| Module | File | Model |
|---|---|---|
| Volatility | `volatility.py` | GARCH-style baseline + MLP |
| Drawdown | `drawdown.py` | BiLSTM + attention + dual horizon heads |
| Regime | `regime_gnn.py` → `code/gnn/mtgnn_regime.py` | MTGNN-inspired graph builder + classifier |

These modules support HPO, training, checkpointing, prediction, and XAI.

### 4.2 Classical / rule-based financial risk modules

| Module | File | Method |
|---|---|---|
| Historical VaR | `var_cvar_liquidity.py` | rolling empirical quantile |
| CVaR / Expected Shortfall | `var_cvar_liquidity.py` | average tail loss beyond VaR |
| Liquidity Risk | `var_cvar_liquidity.py` | rule-based liquidity score |

These modules do not need model training because their calculations are finance-defined and interpretable.

### 4.3 Capital allocation module

| Module | File | Method |
|---|---|---|
| Position Sizing | `position_sizing.py` | weighted risk score + hard caps + technical multiplier |

Position Sizing is deliberately rule-based because it is the main interpretable exposure-control layer.

---

## 5. Standard Chronological Chunking

All modules follow the project’s chronological split strategy:

| Chunk | Train | Validation | Test |
|---|---:|---:|---:|
| chunk 1 | 2000–2004 | 2005 | 2006 |
| chunk 2 | 2007–2014 | 2015 | 2016 |
| chunk 3 | 2017–2022 | 2023 | 2024 |

This protects against look-ahead bias. Normalisers, GARCH baselines, HPO choices, PCA fits, label rules, and learned parameters must be based on the appropriate training data only.

---

## 6. Execution Order

The recommended execution order is:

```text
1. Temporal Encoder embeddings
2. Technical Analyst outputs
3. VaR/CVaR/Liquidity outputs
4. Volatility Risk outputs
5. Drawdown Risk outputs
6. StemGNN Contagion outputs from code/gnn/
7. MTGNN Regime outputs through regime_gnn.py
8. Position Sizing outputs
9. Quantitative Analyst outputs
10. Fusion Engine outputs
```

Position Sizing should only be considered final after all upstream risk outputs exist for the required chunk/split.

---

## 7. Module Input/Output Contracts

### 7.1 Volatility

Input:

```text
outputs/embeddings/TemporalEncoder/chunk{chunk}_{split}_embeddings.npy
outputs/embeddings/TemporalEncoder/chunk{chunk}_{split}_manifest.csv
data/yFinance/processed/features_temporal.csv
data/yFinance/processed/returns_panel_wide.csv
```

Output:

```text
outputs/results/Volatility/predictions_chunk{chunk}_{split}.csv
outputs/results/Volatility/xai/
```

### 7.2 Drawdown

Input:

```text
outputs/embeddings/TemporalEncoder/chunk{chunk}_{split}_embeddings.npy
outputs/embeddings/TemporalEncoder/chunk{chunk}_{split}_manifest.csv
data/yFinance/processed/ohlcv_final.csv
```

Output:

```text
outputs/results/Drawdown/predictions_chunk{chunk}_{split}.csv
outputs/results/Drawdown/xai/
```

### 7.3 VaR/CVaR/Liquidity

Input:

```text
data/yFinance/processed/returns_long.csv
data/yFinance/processed/liquidity_features.csv
```

Output:

```text
outputs/results/risk/var_cvar.csv
outputs/results/risk/liquidity.csv
outputs/results/risk/chunks/var_cvar_chunk{chunk}_{split}.csv
outputs/results/risk/chunks/liquidity_chunk{chunk}_{split}.csv
outputs/results/risk/xai/
```

### 7.4 Regime Detection

Input:

```text
outputs/embeddings/TemporalEncoder/
outputs/embeddings/FinBERT/
data/FRED_data/outputs/macro_features_trading_days_clean.csv
data/graphs/snapshots/edges_YYYY-MM-DD.csv
```

Output:

```text
outputs/results/MTGNNRegime/predictions_chunk{chunk}_{split}.csv
outputs/results/MTGNNRegime/xai/
```

### 7.5 Position Sizing

Input:

```text
outputs/results/TechnicalAnalyst/
outputs/results/Volatility/
outputs/results/Drawdown/
outputs/results/StemGNN/
outputs/results/MTGNNRegime/
outputs/results/risk/chunks/
```

Output:

```text
outputs/results/PositionSizing/position_sizing_chunk{chunk}_{split}.csv
outputs/results/PositionSizing/xai/position_sizing_chunk{chunk}_{split}_xai_summary.json
```

---

## 8. CLI Reference

### 8.1 Volatility

```bash
python code/riskEngine/volatility.py inspect --repo-root . --device cuda
```

```bash
python code/riskEngine/volatility.py hpo --repo-root . --chunk 1 --trials 40 --device cuda --fresh
```

```bash
python code/riskEngine/volatility.py train-best --repo-root . --chunk 1 --device cuda --fresh
```

```bash
python code/riskEngine/volatility.py predict --repo-root . --chunk 1 --split test --device cuda
```

### 8.2 Drawdown

```bash
python code/riskEngine/drawdown.py inspect --repo-root .
```

```bash
python code/riskEngine/drawdown.py hpo --repo-root . --chunk 1 --trials 40 --device cuda --fresh
```

```bash
python code/riskEngine/drawdown.py train-best --repo-root . --chunk 1 --device cuda --fresh
```

```bash
python code/riskEngine/drawdown.py predict --repo-root . --chunk 1 --split test --device cuda
```

### 8.3 VaR/CVaR/Liquidity

```bash
python code/riskEngine/var_cvar_liquidity.py --workers 6 --chunk 0
```

### 8.4 Regime Detection

```bash
python code/riskEngine/regime_gnn.py hpo --repo-root . --chunk 1 --trials 30 --device cuda --fresh --node-limit 2500
```

```bash
python code/riskEngine/regime_gnn.py train-best --repo-root . --chunk 1 --device cuda --fresh --node-limit 2500
```

```bash
python code/riskEngine/regime_gnn.py predict --repo-root . --chunk 1 --split test --device cuda --node-limit 2500
```

### 8.5 Position Sizing

```bash
python code/riskEngine/position_sizing.py inspect --repo-root .
```

```bash
python code/riskEngine/position_sizing.py run-all --repo-root . --chunks 1 2 3 --splits train val test
```

```bash
python code/riskEngine/position_sizing.py validate --repo-root . --chunk 1 --split test
```

---

## 9. XAI Policy for the Risk Engine

Every Risk Engine module must support explanation, either through neural XAI or explicit rule trace.

| Module | XAI type |
|---|---|
| Volatility | gradient feature importance, counterfactual volatility scenarios, GARCH parameter summary |
| Drawdown | attention over timesteps, gradient importance, counterfactual drawdown scenarios |
| VaR/CVaR | empirical quantile trace, tail severity, historical percentile context |
| Liquidity | rule trace, liquidity component scores, tradability explanation |
| Regime | graph properties, macro stress, transition explanation, optional GNN-style edge importance |
| Position Sizing | weighted risks, binding cap source, reduction reasons, XAI summary |

These explanations are later consumed by the Quantitative Analyst, Fusion Engine, and final Streamlit/user-facing output.

---

## 10. Integration with Quantitative Analyst

The Quantitative Analyst consumes the Risk Engine and learns attention-weighted pooling across risk scores. It needs the Risk Engine to produce stable and aligned columns such as:

```text
volatility_risk_score
drawdown_risk_score
var_cvar_risk_score
contagion_risk_score
liquidity_risk_score
regime_risk_score
combined_risk_score
recommended_capital_fraction
recommended_capital_pct
binding_cap_source
xai_summary
```

The Quantitative Analyst is only as good as these upstream contracts.

---

## 11. Integration with Fusion

The Fusion Engine uses Quantitative Analyst and Qualitative Analyst outputs. The Risk Engine still controls final exposure because:

```text
Fusion final_position <= Position Sizing recommendation
```

This preserves the project’s risk-first architecture. The learned Fusion layer can propose a decision, but the rule barrier and Position Sizing recommendation remain final safety constraints.

---

## 12. Common Validation Checklist

Before moving to Quantitative Analyst or Fusion, verify:

```text
Volatility predictions exist for required chunks/splits.
Drawdown predictions exist for required chunks/splits.
StemGNN contagion scores exist for required chunks/splits.
Regime predictions exist for required chunks/splits.
VaR/CVaR chunk files exist for required chunks/splits.
Liquidity chunk files exist for required chunks/splits.
Position Sizing outputs exist for required chunks/splits.
All outputs have ticker/date keys.
No required numeric columns contain NaN/inf.
XAI JSON files exist.
```

Suggested audit command pattern:

```bash
python code/riskEngine/position_sizing.py inspect --repo-root .
```

Then validate each needed split:

```bash
python code/riskEngine/position_sizing.py validate --repo-root . --chunk 1 --split test
```

---

## 13. Hardware and Performance Notes

The learned modules support CUDA and CPU execution. For large runs:

- use CUDA where possible;
- avoid running multiple GPU-heavy models simultaneously;
- use `--num-workers` carefully because too many workers can trigger open-file or memory issues;
- use smoke tests before long runs;
- keep checkpoints and resume logic intact;
- and never ignore NaN losses.

Volatility and Drawdown use PyTorch. Regime uses graph processing and may be memory-sensitive with high node limits. VaR/CVaR and Liquidity use CPU parallelism through worker threads.

---

## 14. Why the Risk Engine is Thesis-Defensible

The Risk Engine makes the project defensible because it separates financial risk concepts into understandable modules:

| Risk concept | Implementation |
|---|---|
| instability | Volatility |
| downside path loss | Drawdown |
| tail-loss threshold | VaR |
| tail-loss severity | CVaR |
| systemic spillover | StemGNN contagion |
| tradability | Liquidity |
| market environment | Regime detection |
| capital control | Position Sizing |

This modular design gives better explanation than a single monolithic black-box predictor.

---

## 15. Current Implementation Boundaries

The current active implementation does **not** include fundamentals. The Risk Engine should not expect fundamental module outputs unless that architecture is explicitly reintroduced.

The Risk Engine currently depends on:

- market data,
- temporal embeddings,
- graph data,
- macro/regime data,
- technical analyst outputs,
- and text-derived regime inputs where relevant.

---

## 16. Final Status

The `code/riskEngine/` directory is the completed risk-control backbone of `fin-glassbox`. It supplies the quantitative risk evidence required for Position Sizing, Quantitative Analyst, Fusion, final decision-making, and XAI audit.


This directory should be maintained carefully because schema changes here propagate to the entire final decision system.
