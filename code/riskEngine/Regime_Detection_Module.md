# Regime Detection Module

## 1. Document Purpose

This document is the replacement documentation for the **Regime Detection Module** in the `fin-glassbox` project:

**An Explainable Multimodal Neural Framework for Financial Risk Management**

The Risk Engine entrypoint is implemented in:

```text
code/riskEngine/regime_gnn.py
```

The actual graph-learning implementation is kept in:

```text
code/gnn/mtgnn_regime.py
```

`regime_gnn.py` is intentionally a thin wrapper. It exposes a Risk Engine-facing command while preserving the graph implementation inside `code/gnn/`, where graph models belong.

---

## 2. Role in the Full Architecture

The Regime Detection Module identifies the current broad market state. It acts as a bridge between market behaviour, macro stress, and graph structure.

```text
Temporal Encoder embeddings
        │
FinBERT / text embeddings where available
        │
FRED macro/regime features
        │
Cross-asset graph snapshots
        │
        ▼
MTGNN Regime Detection
        ├── graph building / adjacency learning
        ├── graph property extraction
        ├── macro stress features
        └── regime classifier
        │
        ▼
outputs/results/MTGNNRegime/predictions_chunk{chunk}_{split}.csv
        │
        ▼
Position Sizing Engine → Quantitative Analyst → Fusion Engine
```

The module does not forecast a specific stock’s price. It classifies the market environment so downstream risk logic can behave differently in calm, volatile, crisis, or rotation states.

---

## 3. Financial Meaning

A market regime is a behavioural state of the market. Examples:

| Regime | Meaning |
|---|---|
| calm | low stress, weaker systemic risk, more stable conditions |
| volatile | unstable movement, increased uncertainty |
| crisis | broad stress, high correlation, high risk of systemic drawdowns |
| rotation | sector/cluster transition where capital moves between groups |

The same stock signal should be interpreted differently across regimes. A moderate BUY-like signal in a calm market may be acceptable. The same signal during a crisis may need a much smaller position or may be rejected by the Fusion rule barrier.

---

## 4. Why Regime Detection is Mandatory

NLP/text explains **why** market conditions may be changing. Regime detection captures **what the market is actually doing** structurally.

This module is mandatory because:

- volatility alone does not describe graph-wide stress;
- news sentiment alone does not prove market behaviour changed;
- single-stock models miss systemic context;
- position sizing needs environment-aware caps;
- and Fusion needs regime-aware final gating.

The Regime Detection Module therefore provides context for risk interpretation.

---

## 5. Wrapper Design

The file:

```text
code/riskEngine/regime_gnn.py
```

imports and re-exports the implementation from:

```text
code/gnn/mtgnn_regime.py
```

The wrapper imports:

```text
MTGNNRegimeConfig
MTGNNRegimeModel
RegimeSnapshotDataset
build_graph_summary
run_hpo
train_regime_model
predict_with_xai
cmd_inspect
cmd_smoke
main
```

This means the Risk Engine can call the regime module as:

```bash
python code/riskEngine/regime_gnn.py predict --repo-root . --chunk 1 --split test --device cuda --node-limit 2500
```

while the actual graph model remains documented and maintained in the GNN folder.

---

## 6. Input Data

The module can use several data families.

### 6.1 Temporal Encoder embeddings

```text
outputs/embeddings/TemporalEncoder/chunk{chunk}_{split}_embeddings.npy
outputs/embeddings/TemporalEncoder/chunk{chunk}_{split}_manifest.csv
```

These provide per-ticker market behaviour representations.

### 6.2 FinBERT embeddings

```text
outputs/embeddings/FinBERT/chunk{chunk}_{split}_embeddings.npy
outputs/embeddings/FinBERT/chunk{chunk}_{split}_metadata.csv
```

Text embeddings are useful for macro/event context when they can be aligned.

### 6.3 FRED macro/regime data

```text
data/FRED_data/outputs/macro_features_trading_days_clean.csv
```

Key macro features include:

```text
yield_spread_10y2y
yield_spread_10y3m
credit_spread_baa_aaa
regime_yield_inverted
DFF
DGS10
DGS2
DGS3MO
BAA10Y
AAA10Y
```

These features give the regime model explicit macro stress context.

### 6.4 Cross-asset graph snapshots

```text
data/graphs/snapshots/edges_YYYY-MM-DD.csv
```

Each snapshot contains edges such as:

```text
window_start
source
target
correlation
```

These snapshots encode cross-asset co-movement and dependency structure.

---

## 7. Modelling Approach

The implementation uses an MTGNN-inspired graph builder and classifier. In this project, MTGNN is **not** used as a full general-purpose forecasting engine. Its role here is narrower:

```text
learn / summarise graph structure → extract graph stress properties → classify regime
```

This limited usage is intentional and should be defended as follows:

- StemGNN handles contagion-style risk propagation.
- MTGNN-style graph learning is used for regime structure.
- The regime model is not the main technical encoder.
- The main technical stream remains the Temporal Encoder.

The regime model therefore respects the project boundary that GNNs live in graph/risk modules rather than replacing the technical encoder.

---

## 8. Regime Labels

The model outputs regime classes such as:

```text
calm
volatile
crisis
rotation
```

The module can use rule-derived labels when no human-labelled regime dataset exists. Label construction uses graph and macro stress logic such as:

```text
high graph density + high stress → crisis
moderate density/stress → volatile
low stress + stable graph → calm
sector/cluster changes → rotation
```

This is acceptable because the model is learning a defensible internal risk-state policy rather than claiming access to external proprietary regime labels.

---

## 9. Output Files

Prediction outputs are written to:

```text
outputs/results/MTGNNRegime/predictions_chunk{chunk}_{split}.csv
```

Important columns include:

| Column | Meaning |
|---|---|
| `ticker` or snapshot key | date/snapshot identification depending on output granularity |
| `date` | regime date |
| `regime_id` | numeric predicted regime |
| `regime_label` | calm / volatile / crisis / rotation |
| `regime_confidence` | confidence in predicted regime |
| `prob_calm` | probability of calm state |
| `prob_volatile` | probability of volatile state |
| `prob_crisis` | probability of crisis state |
| `prob_rotation` | probability of rotation state |
| `regime_transition_probability` | estimated probability of state transition |
| `graph_density` | graph connectedness / stress proxy |
| `avg_degree_norm` | normalised average graph degree |
| `std_degree_norm` | dispersion in graph degree |
| `mean_edge_weight` | mean edge strength |
| `max_edge_weight` | strongest edge weight |
| `graph_entropy` | graph dispersion/uncertainty measure |
| `learned_graph_stress` | learned graph stress score |
| `macro_stress_score` | macro-derived stress estimate |
| `label_graph_stress_score` | graph stress used in label construction/audit |

These outputs are merged into Position Sizing and Quantitative Analyst.

---

## 10. Training Workflow

### Inspect

```bash
python code/riskEngine/regime_gnn.py inspect --repo-root .
```

### Smoke test

```bash
python code/riskEngine/regime_gnn.py smoke --repo-root . --device cuda
```

### HPO

```bash
python code/riskEngine/regime_gnn.py hpo --repo-root . --chunk 1 --trials 30 --device cuda --fresh --node-limit 2500
```

### Train

```bash
python code/riskEngine/regime_gnn.py train-best --repo-root . --chunk 1 --device cuda --fresh --node-limit 2500
```

### Predict validation and test

```bash
python code/riskEngine/regime_gnn.py predict --repo-root . --chunk 1 --split val --device cuda --node-limit 2500
```

```bash
python code/riskEngine/regime_gnn.py predict --repo-root . --chunk 1 --split test --device cuda --node-limit 2500
```

### Full chunk command example

```bash
python code/riskEngine/regime_gnn.py train-best --repo-root . --chunk 1 --device cuda --fresh --node-limit 2500 && python code/riskEngine/regime_gnn.py predict --repo-root . --chunk 1 --split val --device cuda --node-limit 2500 && python code/riskEngine/regime_gnn.py predict --repo-root . --chunk 1 --split test --device cuda --node-limit 2500
```

---

## 11. XAI Integration

Regime XAI is graph- and macro-based. It should explain both predicted state and state drivers.

XAI output is written under:

```text
outputs/results/MTGNNRegime/xai/
```

The module’s explanation levels include:

### 11.1 Level 1: Graph property explanation

The model reports graph properties such as:

- density,
- average degree,
- edge weight strength,
- graph entropy,
- learned graph stress,
- and macro stress score.

These explain why the model saw a calm, volatile, crisis, or rotation state.

### 11.2 Level 2: Graph difference explanation

A useful regime explanation is not only “what is the graph now?” but also:

```text
what changed from the previous period?
```

This can include changes in graph density, edge concentration, transition probability, and macro stress.

### 11.3 Level 3: GNN-style explanation

The regime module may include GNNExplainer-style or approximation-based edge explanations to identify which relationships contributed most to the regime classification.

---

## 12. Integration with Position Sizing

Position Sizing uses regime information for two purposes:

1. **weighted risk score**
2. **hard capital caps**

The default Position Sizing weight for regime risk is:

```text
regime weight = 0.10
```

Regime risk also triggers hard caps:

```text
volatile regime cap = 6%
rotation regime cap = 5%
crisis short-horizon cap = 5%
crisis long-horizon cap = 3%
```

This makes regime a final risk-control context rather than a mere classifier.

---

## 13. Validation Expectations

A healthy regime run should show:

- graph snapshots loaded,
- finite node features,
- finite macro features,
- non-empty train/validation samples,
- label counts across regimes,
- finite validation loss,
- saved predictions,
- saved XAI summary,
- and sensible regime probabilities.

A validation accuracy around a simple majority baseline is not automatically failure because regime labels are rule-derived and often imbalanced. The more important requirement is that the module generates stable, finite, explainable market-state context.

---

## 14. Known Practical Notes

The regime module can be memory-heavy with large node limits. If CUDA memory is under pressure, reduce `--node-limit` or run after other GPU jobs finish.

Earlier runs showed CUDA out-of-memory when another GPU process was active. That is not necessarily a model failure; it can be resource contention.

---

## 15. Limitations

The Regime Detection Module has these limitations:

- regime labels are not external human labels;
- graph snapshots are lower-frequency than daily stock predictions;
- some text embeddings are sparse relative to market data;
- graph properties may simplify complex market structure;
- and node limits can affect full-universe coverage.

These limitations are controlled by using regime as a contextual risk signal rather than the only decision source.

---


The Regime Detection Module is a Risk Engine component exposed through `code/riskEngine/regime_gnn.py` and implemented through `code/gnn/mtgnn_regime.py`. It supplies market-state context to Position Sizing, Quantitative Analyst, Fusion, and final trade approval.
