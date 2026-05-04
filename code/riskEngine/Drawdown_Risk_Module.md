# Drawdown Risk Module

## 1. Document Purpose

This document is the replacement documentation for the **Drawdown Risk Module** in the `fin-glassbox` project:

**An Explainable Multimodal Neural Framework for Financial Risk Management**

The module is implemented in:

```text
code/riskEngine/drawdown.py
```

Its role is to estimate **future downside path risk** from Temporal Encoder embeddings and future close-price paths. It predicts continuous drawdown severity, soft drawdown risk, recovery delay, and confidence for 10-day and 30-day horizons.


---

## 2. Role in the Full Architecture

The Drawdown Risk Module is a learned component of the Risk Engine:

```text
Temporal Encoder embeddings
        │
OHLCV close-price panel
        │
        ▼
Drawdown Risk Module
        ├── BiLSTM sequence encoder
        ├── attention pooling
        ├── 10-day downside head
        └── 30-day downside head
        │
        ▼
outputs/results/Drawdown/predictions_chunk{chunk}_{split}.csv
        │
        ▼
Position Sizing Engine → Quantitative Analyst → Fusion Engine
```

The module does not decide whether to buy or sell. It answers a narrower risk question:

```text
How severe could the future peak-to-trough loss be over the next 10 and 30 trading days?
```

This is different from volatility. Volatility measures instability in both directions. Drawdown focuses specifically on downside path damage.

---

## 3. Financial Meaning

Drawdown is the fall from a previous peak to a later low. It is path-dependent: two assets can have the same final return but very different drawdown paths.

Example:

```text
Asset A: rises smoothly from 100 to 110
Asset B: falls to 70, then recovers to 110
```

Both end at 110, but Asset B had a severe drawdown. The Drawdown Risk Module captures this type of downside path risk.

The module estimates:

```text
expected_drawdown_10d
expected_drawdown_30d
drawdown_risk_10d
drawdown_risk_30d
recovery_days_10d
recovery_days_30d
confidence_10d
confidence_30d
drawdown_risk_score
```

A high drawdown score means that the asset may experience a severe fall even if other directional signals look positive.

---

## 4. Why the Output is Continuous, Not Binary

The module is intentionally **not** a simple binary classifier. A binary output such as “drawdown/no drawdown” would throw away too much information.

The current design outputs real-valued information:

| Output | Why it matters |
|---|---|
| expected drawdown | how large the future loss path may be |
| soft drawdown risk | how close the forecast is to a danger zone |
| recovery days | how long recovery may take after the drop |
| confidence | how reliable the module believes the horizon estimate is |

This gives downstream components richer information for explanation, sizing, and final fusion.

---

## 5. Input Data

### 5.1 Temporal Encoder embeddings

The module consumes:

```text
outputs/embeddings/TemporalEncoder/chunk{chunk}_{split}_embeddings.npy
outputs/embeddings/TemporalEncoder/chunk{chunk}_{split}_manifest.csv
```

Each row is a ticker-date embedding produced by the Temporal Encoder.

### 5.2 Close prices

The module uses close prices from:

```text
data/yFinance/processed/ohlcv_final.csv
```

Required columns:

```text
date
ticker
close
```

The code builds a cached wide close-price panel:

```text
outputs/cache/Drawdown/close_panel_wide_float32.npz
```

This cache prevents repeatedly reloading and pivoting the 1.3GB OHLCV file.

---

## 6. Chronological Chunking

The module follows the same chronological split as the rest of the project:

| Chunk | Train | Validation | Test |
|---|---:|---:|---:|
| chunk 1 | 2000–2004 | 2005 | 2006 |
| chunk 2 | 2007–2014 | 2015 | 2016 |
| chunk 3 | 2017–2022 | 2023 | 2024 |

Target construction uses future windows relative to date `t`, but the validation/test splits remain chronological. The model is never trained on future validation/test data.

---

## 7. Target Construction

For each ticker-date at time `t`, the module looks ahead over two horizons:

```text
horizon_10d = 10 trading days
horizon_30d = 30 trading days
```

For each horizon, the target calculation estimates:

1. maximum future drawdown over the horizon,
2. soft drawdown risk relative to a threshold,
3. recovery delay normalised by horizon,
4. confidence target.

The configured soft thresholds are:

```text
10-day drawdown threshold = 5%
30-day drawdown threshold = 8%
```

Instead of a hard binary threshold, a softness parameter creates a smooth risk score. This allows the model to learn risk gradients near the danger boundary.

---

## 8. Model Architecture

The module uses:

```text
BiLSTM + Attention Pooling + Dual Horizon Heads
```

Detailed structure:

```text
Input sequence
├── shape: (batch, seq_len=30, input_dim=256)
│
BiLSTM encoder
├── hidden dimension: configurable, default 64
├── layers: configurable, default 1
├── bidirectional: true
│
Attention pooling
├── produces weighted sequence summary
├── exposes timestep importance for XAI
│
Prediction head
└── 8 continuous outputs:
    ├── expected_drawdown_10d
    ├── drawdown_risk_10d
    ├── recovery_norm_10d
    ├── confidence_target_10d
    ├── expected_drawdown_30d
    ├── drawdown_risk_30d
    ├── recovery_norm_30d
    └── confidence_target_30d
```

The sequence length is 30 because the upstream Temporal Encoder embeddings represent rolling market context.

---

## 9. Model Outputs

Prediction output is written to:

```text
outputs/results/Drawdown/predictions_chunk{chunk}_{split}.csv
```

Important columns:

| Column | Meaning |
|---|---|
| `ticker` | asset identifier |
| `date` | prediction date |
| `expected_drawdown_10d` | predicted maximum drawdown over next 10 trading days |
| `drawdown_risk_10d` | soft 10-day drawdown danger score |
| `recovery_days_10d` | estimated recovery delay in days |
| `confidence_10d` | confidence for 10-day horizon |
| `expected_drawdown_30d` | predicted maximum drawdown over next 30 trading days |
| `drawdown_risk_30d` | soft 30-day drawdown danger score |
| `recovery_days_30d` | estimated recovery delay in days |
| `confidence_30d` | confidence for 30-day horizon |
| `drawdown_risk_score` | combined downstream risk score |
| `target_*` columns | validation targets for evaluation/audit |

---

## 10. Training Workflow

### Inspect inputs

```bash
python code/riskEngine/drawdown.py inspect --repo-root .
```

### Smoke test

```bash
python code/riskEngine/drawdown.py smoke --repo-root . --device cuda
```

The smoke test checks forward pass, loss, gradients, attention shape, and XAI gradient shape.

### HPO

```bash
python code/riskEngine/drawdown.py hpo --repo-root . --chunk 1 --trials 40 --device cuda --fresh
```

### Train with best HPO parameters

```bash
python code/riskEngine/drawdown.py train-best --repo-root . --chunk 1 --device cuda --fresh
```

### Predict

```bash
python code/riskEngine/drawdown.py predict --repo-root . --chunk 1 --split test --device cuda
```

### Train all chunks

```bash
python code/riskEngine/drawdown.py train-best-all --repo-root . --device cuda --fresh
```

---

## 11. Checkpointing

Models are saved under:

```text
outputs/models/Drawdown/chunk{chunk}/
```

Important files:

```text
best_model.pt
latest_model.pt
training_history.csv
training_summary.json
```

The code is designed to be both importable and independently executable. Checkpoints contain the configuration necessary to reload models safely during prediction.

---

## 12. XAI Integration

The prediction function returns:

```python
{
    "predictions": pd.DataFrame,
    "xai": dict,
    "paths": dict
}
```

XAI outputs are written to:

```text
outputs/results/Drawdown/xai/
```

The module implements three XAI levels.

### 12.1 Level 1: Attention XAI

Attention weights show which timesteps in the 30-day input window contributed most to the drawdown forecast. This answers:

```text
Which recent market days warned the model about downside path risk?
```

### 12.2 Level 2: Gradient XAI

Gradient-based XAI measures sensitivity across:

- embedding dimensions,
- timesteps,
- and output heads.

This helps diagnose whether the model is relying on specific parts of the Temporal Encoder representation.

### 12.3 Level 3: Counterfactual XAI

Counterfactual explanations perturb the input conditions and measure how the drawdown forecast changes. This answers:

```text
What would need to change for predicted drawdown risk to fall or rise?
```

---

## 13. Integration with Position Sizing

The Position Sizing Engine consumes:

```text
drawdown_risk_score
expected_drawdown_10d
expected_drawdown_30d
drawdown_risk_10d
drawdown_risk_30d
recovery_days_10d
recovery_days_30d
```

The default Position Sizing weight for drawdown is:

```text
drawdown weight = 0.15
```

Drawdown also participates in hard caps. If drawdown risk becomes high, position size can be capped even when technical signals are positive.

---

## 14. Validation Expectations

A healthy run should show:

- finite close-price panel,
- finite embeddings,
- finite targets,
- no in-place autograd errors,
- no NaN losses,
- expected prediction columns,
- attention shape `(batch, seq_len)`,
- gradient XAI shape `(sample, seq_len, input_dim)`,
- and reasonable target means for 10-day and 30-day drawdown.

Earlier debugging fixed an in-place tensor operation issue where `torch.nan_to_num_` was applied to a leaf tensor requiring gradients. In-place operations must be avoided when generating gradient XAI.

---

## 15. Limitations

The Drawdown module has these limitations:

- it depends on close-price quality;
- future drawdown targets are noisy;
- severe crashes are rare and hard to learn;
- recovery-day estimation is approximate;
- and continuous drawdown scores should not be interpreted without other Risk Engine outputs.

The module is strongest when combined with VaR/CVaR, volatility, contagion, liquidity, and regime outputs.

---


The Drawdown Risk Module is a completed learned Risk Engine module with:
- inspection,
- smoke testing,
- Optuna TPE HPO,
- training,
- prediction,
- checkpointing,
- continuous dual-horizon outputs,
- and XAI support.

It is a key downside-risk control signal used by Position Sizing, Quantitative Analyst, Fusion, and final trade approval.
