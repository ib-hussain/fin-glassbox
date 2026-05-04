# Volatility Risk Module

## 1. Document Purpose

This document is the replacement documentation for the **Volatility Risk Module** in the `fin-glassbox` project:

**An Explainable Multimodal Neural Framework for Financial Risk Management**

The module is implemented in:

```text
code/riskEngine/volatility.py
```

It is one of the central learned risk modules in the Risk Engine. Its purpose is to estimate near-term and medium-term price instability from the market embedding stream and classical volatility baselines, then pass volatility-aware risk information to the Position Sizing Engine, Quantitative Analyst, Fusion Engine, and final decision layer.

---

## 2. Role in the Full Architecture

The Volatility Risk Module belongs to the Risk Engine:

```text
Temporal Encoder embeddings
        │
        ▼
Volatility Risk Module
        ├── GARCH-style volatility baseline
        ├── recent realised volatility
        ├── Temporal Encoder embedding signal
        └── learned MLP correction / forecast head
        │
        ▼
outputs/results/Volatility/predictions_chunk{chunk}_{split}.csv
        │
        ▼
Position Sizing Engine → Quantitative Analyst → Fusion Engine
```

It does **not** directly produce Buy/Hold/Sell decisions. It estimates the amount of uncertainty in future price movement. The downstream system uses this estimate to reduce confidence, cap position size, and identify dangerous high-volatility environments.

In financial terms, volatility is the magnitude of price variation. A stock can move upward and still be high-risk if the path is unstable. This module therefore measures instability, not directional opportunity.

---

## 3. Financial Meaning

Volatility answers:

```text
How unstable is the asset likely to be over the next horizon?
```

For this system, the module estimates:

```text
vol_10d
vol_30d
volatility regime: low / medium / high
volatility confidence
```

The 10-day horizon is useful for short tactical risk control. The 30-day horizon is useful for medium-horizon exposure control. Downstream modules treat high volatility as a warning signal, especially when it aligns with drawdown, VaR/CVaR, contagion, or crisis-regime evidence.

A high volatility forecast does not automatically mean “sell”. It means:

- price movement is uncertain,
- position size should usually be smaller,
- risk-adjusted confidence should decrease,
- and the final fusion layer should be more cautious.

---

## 4. Implementation Summary

The implemented file uses a **hybrid GARCH + MLP design**:

```text
Input sources
├── Temporal Encoder embedding: 256-dim
├── GARCH-style volatility baseline
└── recent realised volatility

Model
├── VolatilityMLP
├── hidden MLP layers with tanh-style non-linearity through PyTorch layers
├── dropout regularisation
├── volatility output heads
├── volatility regime probability head
└── confidence output
```

The hybrid design is deliberate. A purely neural volatility model may be flexible but less interpretable. A purely statistical volatility model is interpretable but may miss information embedded in the learned market representation. Combining both provides:

| Component | Purpose |
|---|---|
| GARCH baseline | Classical volatility anchor using past return dynamics |
| recent realised volatility | Simple empirical recent movement estimate |
| Temporal Encoder embedding | learned market-sequence representation |
| MLP | nonlinear correction and horizon-specific forecast |

This makes the module stronger and more thesis-defensible because it uses both classical finance logic and learned representation learning.

---

## 5. Input Data

### 5.1 Temporal Encoder embeddings

The module consumes 256-dimensional Temporal Encoder embeddings:

```text
outputs/embeddings/TemporalEncoder/chunk{chunk}_{split}_embeddings.npy
outputs/embeddings/TemporalEncoder/chunk{chunk}_{split}_manifest.csv
```

Expected manifest columns:

```text
ticker
date
```

The manifest aligns each embedding row to a ticker-date pair.

### 5.2 Market feature file

The module uses engineered market features:

```text
data/yFinance/processed/features_temporal.csv
```

Relevant feature examples include:

```text
log_return
vol_5d
vol_21d
rsi_14
macd_hist
bb_pos
volume_ratio
hl_ratio
price_pos
spy_corr_63d
```

### 5.3 Returns panel

The GARCH baseline and recent-volatility estimates use:

```text
data/yFinance/processed/returns_panel_wide.csv
```

This file contains a date-indexed return matrix for the ticker universe.

---

## 6. Chronological Chunking

The module follows the project’s chronological split discipline:

| Chunk | Train | Validation | Test |
|---|---:|---:|---:|
| chunk 1 | 2000–2004 | 2005 | 2006 |
| chunk 2 | 2007–2014 | 2015 | 2016 |
| chunk 3 | 2017–2022 | 2023 | 2024 |

This protects against leakage. The module must never fit normalisation, GARCH parameters, HPO choices, or target statistics on validation or test periods.

---

## 7. Target Construction

The volatility target is built from historical realised future volatility over configured horizons:

```text
vol_horizons = [10, 30]
```

The target is clipped to keep training stable:

```text
min_target_vol = 0.01
max_target_vol = 5.0
fallback_vol = 0.30
```

The fallback value prevents training collapse when an unusual ticker/date combination does not have a stable volatility estimate. The module includes finite-value cleaning and clipping because volatility pipelines are highly sensitive to NaN, infinity, extremely illiquid assets, and anomalous return spikes.

---

## 8. Model Outputs

Prediction output is written to:

```text
outputs/results/Volatility/predictions_chunk{chunk}_{split}.csv
```

Important columns:

| Column | Meaning |
|---|---|
| `ticker` | asset identifier |
| `date` | prediction date |
| `vol_10d` | predicted 10-day volatility |
| `vol_30d` | predicted 30-day volatility |
| `regime_id` | numerical volatility regime class |
| `regime_label` | low / medium / high volatility regime |
| `regime_probs_low` | probability of low-volatility state |
| `regime_probs_medium` | probability of medium-volatility state |
| `regime_probs_high` | probability of high-volatility state |
| `confidence` | model confidence score |
| `garch_vol` | GARCH-style volatility baseline |
| `recent_vol` | recent realised volatility estimate |

Downstream code often maps these into:

```text
volatility_risk_score
volatility_regime_label
volatility_confidence
```

inside Position Sizing and Quantitative Analyst outputs.

---

## 9. Training Workflow

The Volatility module supports full independent execution.

### Inspect inputs

```bash
python code/riskEngine/volatility.py inspect --repo-root . --device cuda
```

### Smoke test

```bash
python code/riskEngine/volatility.py smoke --repo-root . --device cuda
```

The smoke test checks that the model can run forward, compute loss, backpropagate, and generate XAI-compatible gradients.

### HPO

```bash
python code/riskEngine/volatility.py hpo --repo-root . --chunk 1 --trials 40 --device cuda --fresh
```

HPO uses Optuna TPE and SQLite persistence. Failed trials must return a large finite penalty rather than NaN or infinity because Optuna/SQLite can fail when non-finite values are committed.

### Train with best HPO parameters

```bash
python code/riskEngine/volatility.py train-best --repo-root . --chunk 1 --device cuda --fresh
```

### Predict and generate XAI

```bash
python code/riskEngine/volatility.py predict --repo-root . --chunk 1 --split test --device cuda
```

### Train all chunks

```bash
python code/riskEngine/volatility.py train-best-all --repo-root . --device cuda --fresh
```

---

## 10. Checkpointing and Saved Model Files

The module saves trained models under:

```text
outputs/models/Volatility/chunk{chunk}/
```

Important files include:

```text
best_model.pt
latest_model.pt
training_history.csv
garch_models.pkl
garch_params.json
```

The production checkpoint contains model configuration to support reliable reloading. This is important because the integrated system may later import the model rather than run it from CLI.

---

## 11. XAI Integration

The Volatility module returns a structured prediction object:

```python
{
    "predictions": pd.DataFrame,
    "xai": dict,
    "paths": dict
}
```

This matches the project-wide XAI requirement that modules must be usable both independently and in the integrated system.

XAI outputs are written to:

```text
outputs/results/Volatility/xai/
```

Important XAI files:

```text
chunk{chunk}_{split}_feature_importance.csv
chunk{chunk}_{split}_counterfactuals.json
chunk{chunk}_{split}_garch_summary.json
chunk{chunk}_{split}_xai_summary.json
```

### 11.1 Gradient feature importance

Gradient XAI ranks which embedding dimensions most influenced predicted volatility. This does not directly map each dimension to a human-readable technical indicator, but it identifies whether the learned Temporal Encoder representation is materially affecting volatility output.

### 11.2 Counterfactual XAI

Counterfactual scenarios alter GARCH or recent-volatility inputs and measure how predicted volatility changes. This answers:

```text
How sensitive is the model to classical volatility evidence?
```

### 11.3 GARCH parameter summary

The module records aggregate GARCH parameter behaviour:

```text
omega
alpha
beta
persistence
```

This gives a classical volatility explanation anchor and helps defend the hybrid design.

---

## 12. Integration with Position Sizing

Position Sizing consumes volatility outputs and turns them into a bounded risk score. The current Position Sizing weighting gives volatility a default weight of:

```text
volatility weight = 0.20
```

This is high enough to influence sizing but not high enough to dominate the whole Risk Engine. Volatility becomes especially important when it aligns with:

- high drawdown risk,
- severe CVaR,
- high contagion,
- low liquidity,
- or crisis regime.

---

## 13. Validation Expectations

A healthy run should show:

- finite embeddings,
- finite targets,
- no NaN train/validation losses,
- non-infinite best validation loss,
- prediction CSVs with expected row counts,
- XAI files written,
- volatility values inside configured clipping range,
- and regime probabilities summing approximately to 1.

If training loss becomes NaN, this is not acceptable. It usually indicates non-finite targets, too aggressive learning rate, corrupted features, or missing clipping/sanitisation.

---

## 14. Known Practical Issues and Fixes

Earlier versions of the module produced NaN/inf HPO results. The stable version fixes this through:

- finite-value sanitisation,
- volatility clipping,
- large finite HPO failure values,
- isolated HPO trial directories,
- safe checkpoint loading,
- and explicit data inspection.

This is important documentation because volatility modelling is numerically fragile. The module must remain conservative about non-finite values.

---

## 15. Limitations

The module is not a perfect market-risk oracle. It has these limitations:

- volatility is not direction;
- historical volatility patterns can fail during structural breaks;
- GARCH assumptions may not hold for all assets;
- embedding dimensions are not directly human-readable;
- illiquid or anomalous tickers can distort realised volatility;
- and predicted volatility should be interpreted with drawdown, VaR/CVaR, contagion, liquidity, and regime risk.

These limitations are controlled downstream by the Position Sizing Engine and Fusion rule barrier.

---

The Volatility Risk Module is a production-stage Risk Engine component. It supports:

- inspection,
- smoke testing,
- Optuna TPE HPO,
- training from best parameters,
- prediction,
- checkpointing,
- XAI output,
- and integration through returned dictionaries.

It should be treated as one of the core quantitative risk signals used by Position Sizing, Quantitative Analyst, Fusion, and final decision approval.
