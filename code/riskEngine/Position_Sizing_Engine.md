# Position Sizing Engine

## 1. Document Purpose

This document is the replacement documentation for the **Position Sizing Engine** in the `fin-glassbox` project:

**An Explainable Multimodal Neural Framework for Financial Risk Management**

The module is implemented in:

```text
code/riskEngine/position_sizing.py
```

Its purpose is to convert risk-engine outputs into an interpretable recommended capital fraction for each ticker-date. It is not a neural model. It is a rule-based, user-adjustable, risk-control engine.


---

## 2. Role in the Full Architecture

The Position Sizing Engine is the bridge between raw risk estimates and actionable exposure control.

```text
Technical Analyst
Volatility Risk
Drawdown Risk
VaR / CVaR
StemGNN Contagion
Liquidity Risk
MTGNN Regime
        │
        ▼
Position Sizing Engine
        ├── weighted risk score
        ├── risk bucket
        ├── technical-confidence multiplier
        ├── module hard caps
        └── final recommended capital fraction
        │
        ▼
Quantitative Analyst → Fusion Engine → Final Trade Approver
```

It ensures that optimistic signals cannot freely increase exposure when risk modules disagree. In the project’s design, the Risk Engine remains central; Position Sizing is one of the main mechanisms enforcing that principle.

---

## 3. Financial Meaning

Position sizing answers:

```text
How much capital should be allocated to this asset under current risk conditions?
```

It is separate from direction. A model may identify a potentially positive signal, but the system still needs to decide whether that signal deserves:

- no exposure,
- small exposure,
- moderate exposure,
- or maximum allowed exposure.

This module makes that decision using interpretable risk weights and hard caps.

---

## 4. Design Philosophy

The Position Sizing Engine is intentionally rule-based. This is not a weakness. It is a safety and explainability feature.

A fully learned position sizing model would be harder to defend because it could silently learn aggressive behaviour. The current implementation keeps capital allocation transparent:

```text
weighted risk → risk bucket → technical multiplier → hard caps → final size
```

The module’s guiding rule is:

```text
Hard risk caps always override optimistic signals.
```

---

## 5. Input Modules

The module reads outputs from the following components:

```text
outputs/results/TechnicalAnalyst/predictions_chunk{chunk}_{split}.csv
outputs/results/Volatility/predictions_chunk{chunk}_{split}.csv
outputs/results/Drawdown/predictions_chunk{chunk}_{split}.csv
outputs/results/StemGNN/contagion_scores_chunk{chunk}_{split}.csv
outputs/results/MTGNNRegime/predictions_chunk{chunk}_{split}.csv
outputs/results/risk/chunks/var_cvar_chunk{chunk}_{split}.csv
outputs/results/risk/chunks/liquidity_chunk{chunk}_{split}.csv
```

The module merges these inputs by ticker-date. It is designed to tolerate some missing modules during development, but final production runs should have all upstream modules available.

---

## 6. Default Risk Weights

The approved default weights are:

| Risk component | Weight |
|---|---:|
| volatility | 0.20 |
| drawdown | 0.15 |
| VaR/CVaR | 0.15 |
| contagion | 0.25 |
| liquidity | 0.15 |
| regime | 0.10 |

Contagion has the largest weight because cross-asset spillover risk can create systemic danger that single-stock models miss.

The combined risk score is computed as:

```text
combined_risk_score =
    0.20 × volatility_risk
  + 0.15 × drawdown_risk
  + 0.15 × var_cvar_risk
  + 0.25 × contagion_risk
  + 0.15 × liquidity_risk
  + 0.10 × regime_risk
```

---

## 7. Exposure Modes

The module supports user-adjustable exposure modes:

| Mode | Maximum single-stock exposure |
|---|---:|
| conservative | 5% |
| moderate/default | 10% |
| aggressive | 15% |

The approved default is:

```text
moderate = 10% maximum per stock
```

These are portfolio-level capital fractions, not model confidence values.

---

## 8. Regime Hard Caps

Regime controls can reduce exposure even if the weighted risk bucket is moderate.

Configured caps:

| Regime / mode | Hard cap |
|---|---:|
| volatile | 6% |
| rotation | 5% |
| crisis, short horizon | 5% |
| crisis, long horizon | 3% |

The long-horizon crisis cap is stricter because holding risky exposure through crisis regimes is more dangerous than short tactical exposure.

---

## 9. Module Hard Caps

The module applies hard caps for severe risk signals:

```text
severe_module_risk_threshold = 0.85
high_module_risk_threshold = 0.75
severe_module_cap = 2%
high_module_cap = 5%
```

These caps apply to module risk scores such as volatility, drawdown, VaR/CVaR, contagion, and regime.

Liquidity has its own special logic:

```text
low_liquidity_threshold = 0.35
severe_liquidity_threshold = 0.20
low_liquidity_cap = 5%
severe_liquidity_cap = 2%
```

This prevents the system from recommending large positions in assets that may be difficult or costly to trade.

---

## 10. Technical Confidence Scaling

The engine uses Technical Analyst confidence as a multiplier if enabled:

```text
technical_multiplier_min = 0.75
technical_multiplier_max = 1.10
```

This means strong technical confidence can slightly increase the fraction of the allowed risk budget, but it cannot override hard caps. Technical confidence helps size within allowed risk boundaries; it does not create permission to ignore risk.

---

## 11. Position Sizing Algorithm

The module follows this pipeline:

```text
1. Load and merge upstream module outputs by ticker-date.
2. Compute module-specific normalised risk scores.
3. Compute weighted combined_risk_score.
4. Convert combined risk to a risk bucket and bucket fraction.
5. Compute technical confidence multiplier.
6. Compute pre-cap position fraction.
7. Compute hard caps from regime and each risk module.
8. Choose the most restrictive hard cap.
9. Final recommended capital fraction = min(pre-cap size, binding hard cap).
10. Save output CSV and XAI JSON.
```

The final size is therefore always explainable through:

- weighted risk contributors,
- size bucket,
- technical multiplier,
- binding cap source,
- and reduction reasons.

---

## 12. Output Files

Outputs are written to:

```text
outputs/results/PositionSizing/position_sizing_chunk{chunk}_{split}.csv
outputs/results/PositionSizing/xai/position_sizing_chunk{chunk}_{split}_xai_summary.json
```

Important columns:

| Column | Meaning |
|---|---|
| `volatility_risk_score` | normalised volatility risk |
| `drawdown_risk_score` | normalised drawdown risk |
| `var_cvar_risk_score` | normalised tail-risk score |
| `contagion_risk_score` | normalised contagion score |
| `liquidity_risk_score` | normalised liquidity risk |
| `regime_risk_score` | normalised regime risk |
| `combined_risk_score` | weighted aggregate risk |
| `size_bucket` | interpreted risk bucket |
| `risk_bucket_fraction` | base allocation fraction within max exposure |
| `technical_multiplier` | technical-confidence scaling factor |
| `pre_cap_capital_fraction` | position before hard caps |
| `recommended_capital_fraction` | final recommended position fraction |
| `recommended_capital_pct` | final recommendation in percent |
| `binding_cap_source` | module/rule that most restricted position |
| `hard_cap_applied` | whether a hard cap reduced the size |
| `size_reduction_reasons` | human-readable explanation string |
| `xai_summary` | compact row-level explanation |

---

## 13. CLI Workflow

### Inspect upstream availability

```bash
python code/riskEngine/position_sizing.py inspect --repo-root .
```

### Smoke test

```bash
python code/riskEngine/position_sizing.py smoke --repo-root .
```

### Run one chunk/split

```bash
python code/riskEngine/position_sizing.py run --repo-root . --chunk 1 --split test
```

### Run all chunks and splits

```bash
python code/riskEngine/position_sizing.py run-all --repo-root . --chunks 1 2 3 --splits train val test
```

### Validate output

```bash
python code/riskEngine/position_sizing.py validate --repo-root . --chunk 1 --split test
```

---

## 14. XAI Integration

The Position Sizing Engine produces one of the clearest explanation traces in the project. It explains not only the final number, but the reason for reduction.

XAI includes:

```text
risk weight summary
module availability
position summary
risk score summary
binding cap counts
size bucket counts
regime counts
top position examples
plain-English explanation
```

Row-level explanation appears in:

```text
xai_summary
size_reduction_reasons
binding_cap_source
hard_cap_applied
```

A typical explanation can be read as:

```text
Top weighted risks: contagion, volatility, var_cvar; Hard cap applied: regime_hard_cap; Crisis regime constrained exposure.
```

This is essential for thesis defence and for a future Streamlit interface.

---

## 15. Integration with Quantitative Analyst

The Quantitative Analyst consumes Position Sizing output. It uses:

```text
recommended_capital_fraction
recommended_capital_pct
combined_risk_score
binding_cap_source
hard_cap_applied
size_bucket
risk scores
xai_summary
```

The Quantitative Analyst then learns attention-weighted pooling across risk scores. Position Sizing remains the interpretable capital-allocation foundation under that learned branch.

---

## 16. Integration with Fusion

Fusion must not increase risk beyond Position Sizing. The final Fusion rule barrier should enforce:

```text
final_position <= recommended_capital_fraction
```

This keeps the Risk Engine central even after learned fusion. The learned fusion model can reduce, accept, or veto exposure, but it should not enlarge exposure beyond the Position Sizing recommendation.

---

## 17. Validation Expectations

A healthy Position Sizing run should show:

- all core upstream modules available,
- finite combined risk scores,
- recommended positions between 0 and max exposure,
- no negative capital fractions,
- sensible hard-cap counts,
- clear binding cap source values,
- and XAI JSON written.

The validation command should pass before running Quantitative Analyst or Fusion.

---

## 18. Limitations

The module is rule-based and therefore depends on the quality of upstream risk estimates. It cannot fix bad volatility, drawdown, contagion, or regime outputs. It also uses fixed weights, which are defensible and interpretable but not automatically optimised.

This is acceptable because Fusion later learns branch weights, while Position Sizing remains the conservative risk-control layer.

---

The Position Sizing Engine is a completed Risk Engine module. It is the main interpretable capital allocation layer and provides downstream modules with:

- recommended exposure,
- risk budget usage,
- risk bucket,
- binding hard-cap source,
- and explanation traces.

It should be run after all Risk Engine modules and before Quantitative Analyst and Fusion.
