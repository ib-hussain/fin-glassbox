# xAI.md

# Explainability and XAI Integration

## 1. Document Purpose

This document is the XAI specification for the project:

**An Explainable Distributed Deep Learning Framework for Financial Risk Management**


The purpose of this document is to define:

- why explainability is central to the project,
- what each module must explain,
- how explanation outputs should be structured,
- how XAI moves through the pipeline,
- how module-level explanations combine into system-level explanations,
- how Fusion explanations work,
- what files should be produced,
- and how explanation quality should be audited.


---

## 2. XAI Philosophy

The project is not simply a prediction system with explanation added afterward. Explainability is part of the architecture.

The central claim is:

```text
A distributed financial AI system becomes more transparent when each specialised module exposes its own reasoning trace, and the final decision explicitly shows how those traces were fused and constrained.
```

The XAI design has three levels:

1. **Module-level explanation** — each model explains its own output.
2. **Branch-level explanation** — Qualitative and Quantitative Analysts explain their synthesis.
3. **System-level explanation** — Fusion explains the final decision, branch weights, rule-barrier changes, risk caps, and position size.

This is stronger than only applying a generic post-hoc method to the final model because it preserves the internal structure of the decision process.

---

## 3. Explanation Goals

The XAI system must answer five questions for any final decision:

| Question | Required explanation |
|---|---|
| What did the system decide? | Buy / Hold / Sell, confidence, position size |
| Why did it decide that? | Fused quantitative + qualitative evidence |
| What risks mattered most? | Risk attention, rule caps, dominant risk drivers |
| Did any rule override the learned model? | Rule barrier trace |
| Can the decision be audited later? | Stored CSV/JSON XAI outputs with ticker-date keys |

The final explanation should be understandable to:

- a data science evaluator,
- a finance-aware examiner,
- a thesis supervisor,
- a technical developer,
- and a future UI/Streamlit user.

---

## 4. XAI Architecture Overview

```text
ENCODERS
├── Temporal Encoder
│   ├── attention over timesteps
│   ├── embedding audit
│   └── finite/alignment checks
│
└── FinBERT Encoder
    ├── text chunk metadata
    ├── token/document provenance
    ├── PCA projection audit
    └── embedding trace

ANALYSTS
├── Sentiment Analyst
│   ├── sentiment score
│   ├── confidence / uncertainty
│   └── gradient or feature contribution trace
│
├── News Analyst
│   ├── event impact
│   ├── news importance
│   ├── risk relevance
│   └── event-driver explanation
│
├── Technical Analyst
│   ├── timestep attention
│   ├── gradient feature importance
│   └── counterfactual direction/timing explanation
│
├── Qualitative Analyst
│   ├── dominant qualitative driver
│   ├── event aggregation explanation
│   └── daily text evidence summary
│
└── Quantitative Analyst
    ├── attention over risk modules
    ├── top risk driver
    ├── risk-adjusted signal explanation
    └── position/risk context

RISK ENGINE
├── Volatility
│   └── volatility component explanation
├── Drawdown
│   └── attention + gradient + counterfactual explanation
├── VaR/CVaR
│   └── empirical tail-risk trace
├── StemGNN Contagion
│   └── adjacency + edge/node importance + optional GNNExplainer
├── Liquidity
│   └── rule/component trace
├── MTGNN Regime
│   └── graph property + macro stress explanation
└── Position Sizing
    └── cap/reduction/risk-budget rule trace

FUSION
├── learned quantitative weight
├── learned qualitative weight
├── learned signal/risk/confidence
├── rule barrier reasons
├── final position cap
└── final explanation summary
```

---

## 5. Standard XAI Output Schema

Every major module should return or store an explanation object using a consistent structure.

Recommended schema:

```json
{
  "module": "ModuleName",
  "chunk": 1,
  "split": "test",
  "ticker": "AAPL",
  "date": "2024-03-28",
  "prediction": {
    "primary_output": 0.123,
    "recommendation": "HOLD",
    "confidence": 0.57,
    "risk_score": 0.44
  },
  "xai": {
    "summary": "Plain-English explanation of the output.",
    "top_drivers": [
      {"name": "contagion", "value": 0.25, "direction": "risk-increasing"},
      {"name": "volatility", "value": 0.20, "direction": "risk-increasing"}
    ],
    "method": "attention | gradient | rule_trace | graph_explainer | counterfactual",
    "confidence_notes": "What makes the explanation reliable or uncertain.",
    "limitations": "Known limitation of this explanation."
  },
  "provenance": {
    "input_files": [],
    "model_checkpoint": "",
    "generated_at": "",
    "row_id": "optional"
  }
}
```

CSV outputs may contain a simplified version through:

```text
xai_summary
fusion_xai_summary
risk_summary
size_reduction_reasons
rule_barrier_reasons
```

JSON files should contain richer explanation reports.

---

## 6. Explanation Levels

The project uses five explanation levels.

| Level | Name | Purpose | Example |
|---:|---|---|---|
| L0 | Data provenance | Shows where the input came from | ticker/date/filing metadata |
| L1 | Local driver explanation | Explains one prediction | attention weights, top risk driver |
| L2 | Mechanism explanation | Explains how the module produced the score | rule trace, graph properties |
| L3 | Counterfactual explanation | Shows what would change the output | lower drawdown risk would increase position |
| L4 | System-level explanation | Explains final fused decision | branch weights + rule barrier |

Not every module needs every level. However, every final user-facing decision should include L0, L1, L2, and L4. L3 should be included where computationally practical.

---

## 7. Module-by-Module XAI Specification

---

## 7.1 Temporal Encoder XAI

### What must be explained

The Temporal Encoder explains which parts of the historical sequence contributed most to the learned market representation.

### XAI methods

| Method | Purpose |
|---|---|
| Attention weights | Identify important timesteps |
| Gradient feature importance | Identify sensitive input dimensions/features |
| Embedding audit | Validate finite embeddings and row alignment |

### Outputs

Expected explanation artefacts:

```text
outputs/embeddings/TemporalEncoder/*_manifest.csv
outputs/embeddings/TemporalEncoder/*_embeddings.npy
outputs/results/TemporalEncoder/xai/
```

### Explanation example

```text
The Temporal Encoder placed highest attention on the most recent and mid-window timesteps, indicating that both immediate price behaviour and recent historical context influenced the embedding.
```

### Limitations

Attention is not a perfect causal explanation. It shows where the model focused, not necessarily which feature caused the final downstream decision.

---

## 7.2 FinBERT Encoder XAI

### What must be explained

FinBERT must preserve text provenance and embedding traceability.

### XAI methods

| Method | Purpose |
|---|---|
| Metadata trace | Links embedding rows to filings/chunks |
| Token/chunk provenance | Identifies form type, source section, filing date |
| PCA manifest | Shows 768→256 projection fitted on train only |
| Downstream explanation | Sentiment/News modules explain semantic effect |

### Required metadata

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
```

### Important XAI rule

FinBERT itself is an encoder. Its primary explainability is provenance and downstream semantic interpretation. Sentiment and News Analysts are responsible for producing explicit text-impact explanations.

---

## 7.3 Sentiment Analyst XAI

### What must be explained

The Sentiment Analyst explains the emotional/market tone inferred from financial text.

### XAI methods

| Method | Purpose |
|---|---|
| Prediction decomposition | sentiment score, confidence, uncertainty, magnitude |
| Gradient feature importance | embedding dimensions that influenced sentiment |
| Metadata trace | filing section and document source |
| Summary sentence | plain-English sentiment interpretation |

### Important output fields

```text
sentiment_score
sentiment_confidence
sentiment_uncertainty
sentiment_magnitude
xai_summary
```

### Explanation example

```text
The text produced mildly negative sentiment with moderate confidence. The uncertainty remains material, so the qualitative branch should not dominate the final decision.
```

---

## 7.4 News Analyst XAI

### What must be explained

The News Analyst explains event importance and event risk relevance.

### XAI methods

| Method | Purpose |
|---|---|
| Event-impact score | Indicates positive/negative event pressure |
| News importance | Indicates how important the event is |
| Risk relevance | Indicates whether the event is risk-related |
| Driver summary | Shows whether volatility, drawdown, or uncertainty dominates |

### Important output fields

```text
news_event_impact
news_importance
risk_relevance
volatility_spike
drawdown_risk
news_uncertainty
xai_summary
```

### Explanation example

```text
The filing section was treated as risk-relevant and mildly negative, increasing qualitative risk but not enough to override the quantitative branch.
```

---

## 7.5 Technical Analyst XAI

### What must be explained

The Technical Analyst explains market-direction evidence from temporal embeddings.

### XAI methods

| Level | Method | Output |
|---:|---|---|
| L1 | Attention weights | important timesteps |
| L2 | Gradient feature importance | embedding dimensions driving trend/momentum/timing |
| L3 | Counterfactuals | what change would alter technical call |

### Important output fields

```text
trend_score
momentum_score
timing_confidence
technical_confidence
technical_direction_score
xai_summary
```

### Explanation example

```text
The technical branch shows positive momentum but only moderate timing confidence, so the signal is supportive but not strong enough by itself to force a BUY.
```

---

## 7.6 Volatility Model XAI

### What must be explained

The Volatility Model explains expected instability over short and medium horizons.

### XAI methods

| Method | Purpose |
|---|---|
| GARCH component trace | classical volatility baseline |
| Recent realised volatility | recent observed risk |
| Neural adjustment explanation | learned correction from temporal embedding |
| Regime probability | low/medium/high volatility state |

### Important output fields

```text
vol_10d
vol_30d
volatility_risk_score
volatility_regime_label
volatility_confidence
garch_vol
recent_vol
```

### Explanation example

```text
Volatility risk is high because both recent realised volatility and the GARCH baseline are elevated, causing the position sizing module to reduce exposure.
```

---

## 7.7 Drawdown Risk Model XAI

### What must be explained

The Drawdown Risk Model explains downside path risk.

### XAI methods

| Level | Method | Output |
|---:|---|---|
| L1 | Attention weights | timesteps warning of drawdown |
| L2 | Gradient importance | embedding dimensions tied to drawdown risk |
| L3 | Counterfactuals | what would reduce drawdown estimate |

### Important output fields

```text
expected_drawdown_10d
expected_drawdown_30d
drawdown_risk_10d
drawdown_risk_30d
drawdown_risk_score
recovery_days_10d
recovery_days_30d
confidence_10d
confidence_30d
```

### Explanation example

```text
The model estimates moderate 30-day drawdown risk and a longer recovery period, so the risk engine reduces position size even if the directional signal is positive.
```

---

## 7.8 Historical VaR XAI

### What must be explained

VaR explains the historical threshold loss at a confidence level.

### XAI method

VaR is statistical, so its explanation is not neural. It should expose:

- rolling window length,
- percentile threshold,
- confidence level,
- resulting loss threshold,
- data availability.

### Important output fields

```text
var_95
var_99
```

### Explanation example

```text
The 95% historical VaR indicates that losses worse than this threshold occurred in approximately the worst 5% of days in the historical rolling window.
```

---

## 7.9 CVaR / Expected Shortfall XAI

### What must be explained

CVaR explains the average loss beyond the VaR threshold.

### XAI method

Like VaR, CVaR is statistical and should expose:

- VaR threshold,
- number of tail observations,
- average tail loss,
- tail ratio.

### Important output fields

```text
cvar_95
cvar_99
tail_ratio_95
tail_ratio_99
```

### Explanation example

```text
CVaR is more severe than VaR, meaning that once the loss threshold is breached, the average tail loss is materially larger than the VaR cutoff.
```

---

## 7.10 Liquidity Risk XAI

### What must be explained

Liquidity explains whether a position can be traded safely.

### XAI method

Liquidity is best explained through a rule/component trace.

### Important output fields

```text
liquidity_score
slippage_estimate_pct
days_to_liquidate_1M
tradable
dv_score
vr_score
to_score
```

### Explanation example

```text
The asset is considered tradable because dollar volume and turnover are sufficient, and estimated slippage is low. Liquidity does not block the position.
```

or:

```text
The asset is not tradable under the rule barrier because liquidity score is below the minimum threshold, forcing HOLD and zero position.
```

---

## 7.11 StemGNN Contagion Risk XAI

### What must be explained

StemGNN explains cross-asset contagion risk: whether risk may spread through relationships among assets.

### XAI levels

| Level | Method | Purpose |
|---:|---|---|
| L1 | Learned adjacency / top influencers | Which assets influence the target most |
| L2 | Gradient node/edge importance | Which relationships changed the score |
| L3 | GNNExplainer approximation | Local explanatory subgraph |

### Important output fields

```text
contagion_5d
contagion_20d
contagion_60d
contagion_risk_score
```

### Optional heavy XAI

GNNExplainer-style explanations are more expensive and should be opt-in for large runs.

Recommended mode:

```text
always-on: adjacency + top influencers + gradient importance
optional: --enable-gnnexplainer
```

### Explanation example

```text
Contagion is the top risk driver because the asset is strongly connected to a stressed cluster in the learned cross-asset graph.
```

---

## 7.12 MTGNN Regime Risk XAI

### What must be explained

The regime model explains the current market state using learned graph structure and macro/regime features.

### XAI methods

| Method | Purpose |
|---|---|
| Graph properties | density, degree, entropy, edge weight stress |
| Macro stress score | macro contribution to regime state |
| Regime probabilities | probability of calm/volatile/crisis/rotation |
| Graph diff | optional comparison to previous period |
| Key edges | important graph connections |

### Important output fields

```text
regime_label
regime_confidence
prob_calm
prob_volatile
prob_crisis
prob_rotation
graph_density
avg_degree_norm
std_degree_norm
mean_edge_weight
max_edge_weight
graph_entropy
learned_graph_stress
macro_stress_score
label_graph_stress_score
```

### Explanation example

```text
The system classifies the market as crisis because graph stress and macro stress are elevated, and the crisis probability dominates other regime probabilities.
```

---

## 7.13 Position Sizing XAI

### What must be explained

Position Sizing explains why a certain capital allocation was recommended.

### XAI method

This module should use rule trace explanations.

### Important output fields

```text
recommended_capital_fraction
recommended_capital_pct
position_fraction_of_max
binding_cap_source
hard_cap_applied
size_bucket
risk_budget_used
size_reduction_reasons
```

### Required explanation content

- base exposure mode,
- risk bucket,
- technical multiplier,
- binding hard cap,
- crisis cap if applied,
- whether exposure was reduced due to volatility, drawdown, VaR/CVaR, contagion, liquidity, or regime.

### Explanation example

```text
The recommended exposure is 3% because the asset is in a crisis regime; the regime hard cap is binding even though the technical signal is positive.
```

---

## 7.14 Qualitative Analyst XAI

### What must be explained

The Qualitative Analyst explains the daily ticker-level text view.

### XAI methods

| Method | Purpose |
|---|---|
| Event aggregation trace | number of events and source types |
| Dominant driver | risk relevance, news uncertainty, sentiment, etc. |
| Score decomposition | sentiment + news impact + confidence |
| Row-level summary | plain-English daily explanation |

### Important output fields

```text
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

### Explanation example

```text
The qualitative branch remains HOLD because the text signal is weak and confidence is low, even though risk relevance is present.
```

---

## 7.15 Quantitative Analyst XAI

### What must be explained

The Quantitative Analyst explains how technical and risk-engine evidence were combined.

### XAI methods

| Method | Purpose |
|---|---|
| Risk attention weights | learned importance across risk modules |
| Top risk driver | most influential risk source |
| Risk-adjusted signal | final quantitative directional signal |
| Position context | recommended exposure and hard-cap reason |

### Required Fusion-ready XAI fields

```text
attention_pooled_risk_score
top_attention_risk_driver
risk_attention_volatility
risk_attention_drawdown
risk_attention_var_cvar
risk_attention_contagion
risk_attention_liquidity
risk_attention_regime
xai_summary
```

### Explanation example

```text
The quantitative branch recommends HOLD because the technical signal is positive but contagion and regime risks dominate the risk attention, limiting the risk-adjusted signal.
```

### Critical schema rule

If `top_attention_risk_driver` and the `risk_attention_*` columns are missing, the file is not final Fusion-ready.

---

## 7.16 Fusion Engine XAI

### What must be explained

Fusion explains the final decision.

It must show:

- learned quantitative branch weight,
- learned qualitative branch weight,
- learned recommendation,
- final recommendation,
- whether rules changed the action,
- user rule cap,
- final position,
- dominant risk driver,
- text availability,
- rule barrier reasons.

### Important output fields

```text
final_recommendation
final_fusion_signal
final_fusion_risk_score
final_fusion_confidence
final_position_fraction
final_position_pct
learned_recommendation
learned_quantitative_weight
learned_qualitative_weight
branch_weight_dominance
rule_changed_action
user_rule_cap_fraction
pre_rule_learned_position_fraction
rule_barrier_reasons
fusion_xai_summary
```

### Fusion explanation pattern

```text
The learned fusion layer weighted the quantitative branch at 0.82 and the qualitative branch at 0.18. The final signal was mildly positive, but the user rule barrier capped position size because the regime was crisis. Final decision: HOLD with 3% maximum allowed exposure.
```

### Rule barrier explanation pattern

```text
Learned model proposed BUY. Rule barrier changed action to HOLD because quantitative risk exceeded the buy-veto threshold and liquidity score was below the minimum threshold.
```

---

## 7.17 Final Trade Approver XAI

The Final Trade Approver should be a thin, auditable layer.

It should not hide Fusion logic. It should format and preserve it.

Final explanation should include:

```text
Final decision
Final confidence
Final position size
Quantitative branch summary
Qualitative branch summary
Top risk drivers
Rule barrier trace
Module-level evidence links
```

---

## 8. Always-On vs Optional XAI

Some explanations are cheap and should always be generated. Others are expensive and should be optional.

| XAI type | Runtime cost | Policy |
|---|---:|---|
| CSV `xai_summary` text | Low | Always on |
| Rule trace | Low | Always on |
| Attention weights | Low/medium | Always on when model supports it |
| Gradient importance | Medium | On validation/test or samples |
| Counterfactuals | Medium/high | Samples or requested tickers |
| GNNExplainer | High | Opt-in with flag |
| Full SHAP on large neural models | High | Sampled only |

Recommended approach:

```text
Production full run: lightweight XAI for all rows + rich XAI for samples
Debug/defence run: enable heavier XAI for selected assets/dates
```

---

## 9. XAI File Structure

Recommended output structure:

```text
outputs/results/
├── analysts/
│   ├── sentiment/
│   └── news/
│
├── TechnicalAnalyst/
│   └── xai/
│
├── QualitativeAnalyst/
│   └── xai/
│
├── QuantitativeAnalyst/
│   └── xai/
│
├── risk/
│   └── xai/
│
├── StemGNN/
│   └── xai/
│
├── MTGNNRegime/
│   └── xai/
│
├── PositionSizing/
│   └── xai/
│
└── FusionEngine/
    └── xai/
```

Each prediction CSV should include a compact explanation column. Each XAI folder should contain JSON summaries or richer sampled explanations.

---

## 10. XAI Integration into the Data Flow

The explanation trace should move forward with the prediction.

```text
Module output CSV
├── numeric prediction columns
├── confidence/risk columns
├── compact xai_summary
└── optional path to rich XAI JSON
```

Downstream modules should not discard upstream explanations.

For example:

```text
Sentiment xai_summary
News xai_summary
       ↓
Qualitative xai_summary
       ↓
Fusion qualitative_xai_summary
```

And:

```text
Risk module summaries
Position sizing rule trace
Quantitative risk attention
       ↓
Fusion quantitative_xai_summary
       ↓
Final decision explanation
```

---

## 11. XAI for Missing Qualitative Data

Text data is sparse. Many ticker-date rows have no matching filing/news event.

The system must explain missing text explicitly instead of treating it as positive or negative.

Neutral qualitative state:

```text
qualitative_score = 0.0
qualitative_risk_score = 0.5
qualitative_confidence = 0.0
event_count = 0
dominant_qualitative_driver = no_text_event
```

Explanation:

```text
No qualitative text event matched this ticker-date; the qualitative branch was kept neutral and received low fusion weight.
```

---

## 12. XAI for User Rule Barrier

The user rule barrier is one of the most important explanation components.

It must always expose:

- whether it changed the learned action,
- whether it reduced the position,
- which rule was binding,
- final cap value,
- whether crisis mode applied,
- whether liquidity blocked the trade,
- whether contagion/drawdown/high-risk vetoed BUY.

Recommended fields:

```text
rule_changed_action
user_rule_cap_fraction
pre_rule_learned_position_fraction
final_position_fraction
rule_barrier_reasons
```

Example values:

```text
rule_barrier_reasons = crisis_cap_0.03; position_reduced_by_rule_barrier; action_changed_by_rule_barrier
```

---

## 13. XAI Quality Audits

XAI outputs must be audited, not just generated.

### 13.1 Finite value audit

All numeric explanation columns must be finite.

Audit checks:

```text
No NaN in attention weights
No infinite risk score
No invalid confidence outside [0, 1]
No branch weights that fail to sum to 1
```

### 13.2 Attention audit

For attention distributions:

```text
sum(attention_weights) ≈ 1
all weights >= 0
no missing risk modules
```

### 13.3 Rule trace audit

If final position is less than recommended position, `rule_barrier_reasons` must explain why.

If learned action differs from final action, `rule_changed_action` must be 1 and reasons must be non-empty.

### 13.4 Schema audit

Fusion-ready Quantitative files must contain:

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

### 13.5 Counterfactual plausibility audit

Counterfactuals should be plausible. For example:

- reducing risk should not increase a risk warning,
- increasing liquidity should not trigger a liquidity block,
- crisis regime should reduce or cap exposure.

### 13.6 Stability audit

Explanations should not change wildly for tiny input perturbations unless the decision is near a threshold.

Recommended check:

```text
Same ticker over neighbouring dates should show gradually changing drivers unless there is an actual event/regime shift.
```

---

## 14. Explanation Metrics

Suggested explanation-quality metrics:

| Metric | Applies to | Meaning |
|---|---|---|
| Attention entropy | Technical, Quantitative, GNN | Whether attention is concentrated or diffuse |
| Top-driver stability | Quantitative, Fusion | Whether dominant drivers are stable across nearby dates |
| Rule-trace completeness | Position/Fusion | Whether every override has a reason |
| Counterfactual validity | Technical/Drawdown/Fusion | Whether proposed changes alter output as expected |
| Feature-importance consistency | Gradient/SHAP-style outputs | Whether important features remain meaningful across runs |
| Human readability | All modules | Whether explanation is understandable in report/UI |

---

## 15. XAI and Streamlit/UI Integration

A future interface should not show all raw columns by default. It should show layered explanation cards.

### Recommended UI cards

1. **Final Decision Card**
   - Buy/Hold/Sell
   - confidence
   - position size
   - final explanation

2. **Risk Summary Card**
   - overall risk score
   - top risk driver
   - volatility/drawdown/contagion/regime/liquidity status

3. **Quantitative Evidence Card**
   - risk-adjusted signal
   - risk attention weights
   - Position Sizing recommendation

4. **Qualitative Evidence Card**
   - qualitative score
   - confidence
   - event count
   - dominant qualitative driver

5. **Rule Barrier Card**
   - whether learned decision was changed
   - cap applied
   - reasons

6. **Audit Trail Card**
   - model checkpoints
   - chunk/split
   - input files
   - ticker/date provenance

---

## 16. XAI in the Report / Thesis

The report should describe XAI as architectural, not decorative.

Suggested wording:

```text
The proposed framework integrates explainability at multiple levels. Individual modules expose local explanations such as attention weights, gradient-based feature sensitivity, rule traces, graph properties, and event-level text drivers. These module-level explanations are then propagated into qualitative and quantitative synthesis branches. The final hybrid Fusion Engine provides system-level explainability by reporting learned branch weights, final signal and risk estimates, position caps, and user-rule overrides. This enables the final Buy/Hold/Sell decision to be audited through both learned evidence and explicit risk-control logic.
```

---

## 17. Known XAI Limitations

The system must be honest about limitations.

1. Attention is not identical to causality.
2. Gradient importance can be noisy.
3. GNN explanations can be expensive and approximate.
4. FinBERT embeddings are compressed representations; direct token-level explanation may be limited unless implemented separately.
5. Qualitative data is sparse and imbalanced across tickers/dates.
6. Weak-supervised targets explain the system policy, not human-labelled trading truth.
7. Rule-based explanations are transparent but depend on chosen thresholds.
8. Fusion explanations are only as reliable as upstream module outputs.

These limitations do not invalidate the design. They define the boundary of what the explanations can claim.

---

## 18. Required XAI Behaviour by Module Type

### Neural modules

Must provide at least one of:

- attention weights,
- gradient feature importance,
- counterfactual examples,
- sampled SHAP/LIME-style feature explanations.

### Rule/statistical modules

Must provide:

- formula/method trace,
- input window/threshold details,
- rule trigger reason,
- output interpretation.

### Graph modules

Must provide:

- graph-level properties,
- important nodes/edges,
- top influencer links,
- optional GNNExplainer subgraph.

### Fusion/final decision modules

Must provide:

- branch weights,
- learned vs final action,
- rule barrier reasons,
- final position cap,
- system-level explanation.

---

## 19. XAI Checklist
```text
[ ] Temporal Encoder has embedding manifests and attention/XAI samples.
[ ] FinBERT embeddings preserve metadata and PCA provenance.
[ ] Sentiment Analyst outputs sentiment explanation fields.
[ ] News Analyst outputs event/risk explanation fields.
[ ] Technical Analyst outputs attention/gradient/counterfactual XAI.
[ ] Volatility outputs component-level risk explanation.
[ ] Drawdown outputs attention/gradient/counterfactual XAI.
[ ] VaR/CVaR outputs statistical tail-risk trace.
[ ] Liquidity outputs rule/component trace.
[ ] StemGNN outputs adjacency/top-influencer/gradient XAI and optional GNNExplainer.
[ ] MTGNN Regime outputs graph property and macro/regime explanation.
[ ] Position Sizing outputs cap/reduction reasons.
[ ] Qualitative Analyst outputs daily qualitative XAI summaries.
[ ] Quantitative Analyst outputs risk attention weights and top risk driver.
[ ] Fusion outputs branch weights, rule barrier reasons, and final explanation.
[ ] Final decision output preserves module-level and system-level explanation trace.
```

---


The XAI strategy of this project is based on traceable modular reasoning.

Each model explains its own part of the decision. The qualitative and quantitative branches summarise their evidence. The Fusion Engine then explains how those branches were weighted, how risk affected the final decision, and whether user-defined safety rules changed the learned recommendation.

This makes the system defensible as an explainable financial AI framework because the final output is not just a prediction. It is a structured decision trace.
