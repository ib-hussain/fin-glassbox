# MTGNN Regime Detection Module

**Project:** An Explainable Multimodal Neural Framework for Financial Risk Management  
**Component:** MTGNN-style learned graph builder  
**Primary implementation context:** `code/gnn/mtgnn_regime.py`  
**Scope of this document:** only the MTGNN graph-building usage inside the current project  
**Not covered here:** full regime classifier documentation, regime training details, final regime-risk interpretation  

---

## 1. Purpose of this document

This document explains the **MTGNN graph-building component** as it is used in the current fin-glassbox system.

The project does not use the full original MTGNN architecture as a general forecasting model. Instead, it uses the **graph construction idea** from MTGNN: learning a sparse, feature-aware adjacency matrix between assets. That learned graph is then converted into graph properties and passed into the regime module.

Therefore, this document intentionally focuses only on:

- what the MTGNN-style graph builder does;
- what inputs it consumes;
- what graph it constructs;
- what graph properties it outputs;
- how it supports the regime risk module;
- how it differs from using a full MTGNN forecasting model.

The full regime-risk model, classifier, labels, macro integration, and final regime outputs should be documented separately in the Regime Risk Module documentation.

---

## 2. What “MTGNN usage” means in this project

In this project, MTGNN is used as:

```text
Feature-aware graph builder
```

not as:

```text
full MTGNN forecasting model
```

The implemented component learns an adjacency matrix from node features. Each node is a stock. Each node feature vector combines temporal market embeddings and text context embeddings.

The core output is:

```text
learned adjacency matrix: [batch, nodes, nodes]
```

This adjacency matrix is summarised into graph-level properties such as density, average degree, mean edge weight, entropy, and learned graph stress.

---

## 3. Position in the architecture

```text
Temporal Encoder embeddings
        +
FinBERT text embeddings
        ↓
MTGNN-style graph learner
        ↓
learned sparse adjacency
        ↓
graph properties
        ↓
Regime Risk Module
        ↓
Position Sizing / Quantitative Analyst / Fusion
```

The MTGNN graph builder is not exposed as a standalone final decision module. It is an internal structural component of the regime module.

---

## 4. Why only the graph builder is used

The project uses only the MTGNN-style graph builder because:

1. The system needs market-state structure, not another price forecaster.
2. The Temporal Encoder already handles temporal sequence representation.
3. StemGNN already handles contagion risk.
4. The regime module needs graph properties that describe market connectivity.
5. A learned graph is more adaptive than a fixed correlation matrix.

This keeps module boundaries clean:

```text
Temporal Encoder → market sequence representation
StemGNN → contagion risk
MTGNN graph builder → learned market connectivity graph for regime analysis
```

---

## 5. Input data used by the graph builder

### 5.1 Temporal Encoder embeddings

Path pattern:

```text
outputs/embeddings/TemporalEncoder/chunk{N}_{split}_embeddings.npy
outputs/embeddings/TemporalEncoder/chunk{N}_{split}_manifest.csv
```

The manifest supplies:

```text
ticker,date
```

The embedding file supplies a 256-dimensional temporal embedding per ticker-date row.

For a given graph snapshot date, the dataset finds the latest available temporal embedding for each selected node ticker at or before that date.

### 5.2 FinBERT text embeddings

Path pattern:

```text
outputs/embeddings/FinBERT/chunk{N}_{split}_embeddings.npy
outputs/embeddings/FinBERT/chunk{N}_{split}_metadata.csv
```

The text context is aggregated over a lookback window, default:

```text
text_lookback_days = 30
```

The text embedding contributes a 256-dimensional representation. When a ticker mapping exists, the module aggregates text by ticker. If ticker mapping is unavailable, it falls back to broadcasting a global market-level text vector.

### 5.3 Cross-asset graph snapshots

Path:

```text
data/graphs/snapshots/edges_YYYY-MM-DD.csv
```

These snapshots come from the Cross-Asset Relation Data Builder. They are used to define the snapshot dates and to compute existing graph features such as density and correlation strength.

### 5.4 FRED macro/regime features

Path:

```text
data/FRED_data/outputs/macro_features_trading_days_clean.csv
```

Default selected macro columns include:

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

Macro features are not part of the graph builder itself, but they are concatenated with graph properties later inside the regime module.

---

## 6. Node feature construction

Each node is a stock ticker.

For each selected node ticker at a snapshot date, the module constructs:

```text
node_feature = [temporal_embedding, text_embedding]
```

Default dimensions:

| Component | Dimension | Source |
|---|---:|---|
| Temporal embedding | 256 | Temporal Encoder |
| Text embedding | 256 | FinBERT aggregation |
| Combined node feature | 512 | Concatenation |

So for a graph with `N` selected stocks:

```text
node_features: [N, 512]
```

During training/inference batches:

```text
node_features: [batch, N, 512]
```

The default `node_limit` is 2,500, but HPO and fast runs may use a smaller node limit such as 768 or 512.

---

## 7. Node universe selection

The node universe is selected from the temporal manifest and graph snapshots. The module tries to select tickers that are available in the embeddings and appear in the graph snapshots.

Important controls:

| Parameter | Meaning |
|---|---|
| `node_limit` | Maximum number of stock nodes retained |
| `hpo_node_limit` | Smaller node count used during HPO |
| `top_k` | Number of outgoing learned graph neighbours per node |

The purpose is to keep graph learning computationally manageable while preserving enough market coverage for regime structure.

---

## 8. MTGNN-style graph learner

The actual graph builder class is:

```python
class MTGNNGraphLearner(nn.Module)
```

It contains:

```text
node_encoder
query projection
key projection
top-k sparse adjacency construction
```

### 8.1 Node encoder

Each node feature is passed through an encoder:

```text
Linear(input_dim → hidden_dim)
LayerNorm
GELU
Dropout
Linear(hidden_dim → graph_dim)
LayerNorm
GELU
```

This produces a graph-space node representation.

### 8.2 Query-key graph score

The graph learner projects node representations into query and key vectors:

```text
q = W_q h
k = W_k h
```

Then computes pairwise scores:

```text
score_ij = (q_i · k_j) / sqrt(graph_dim)
```

The diagonal is masked out so a node does not connect to itself.

### 8.3 Edge weight construction

Scores are passed through a sigmoid:

```text
weight_ij = sigmoid(score_ij)
```

Then for each node only the top `k` neighbours are retained:

```text
adjacency_i = top_k(weight_i)
```

All other edge weights are set to zero.

The final output is a sparse directed adjacency matrix:

```text
adjacency: [batch, N, N]
```

---

## 9. Graph properties extracted from learned adjacency

The learned adjacency is converted into 7 graph-level properties by:

```python
graph_properties_from_adjacency(adj)
```

| Index | Property | Meaning |
|---:|---|---|
| 0 | `density` | Fraction of possible edges retained |
| 1 | `mean_degree_norm` | Average normalised node degree |
| 2 | `std_degree_norm` | Dispersion of node degree |
| 3 | `mean_weight` | Average learned edge strength |
| 4 | `max_weight` | Strongest learned edge weight |
| 5 | `graph_entropy` | How dispersed/uncertain edge distributions are |
| 6 | `graph_stress` | Simple stress proxy: `0.5 × density + 0.5 × mean_weight` |

These properties are compact, explainable summaries of the learned market graph.

---

## 10. Existing graph snapshots vs learned MTGNN graph

The module uses two graph concepts.

### Existing correlation graph snapshots

Precomputed by `build_cross_asset_graph.py`:

```text
data/graphs/snapshots/edges_YYYY-MM-DD.csv
```

They provide realised correlation-based graph features such as existing density, average degree, mean/max absolute correlation, and sector concentration.

### Learned MTGNN-style adjacency

Generated inside the model from node features:

```text
Temporal Encoder embedding + FinBERT text context → learned adjacency
```

It provides feature-aware dynamic edge weights, top-k learned links, graph properties for regime classification, and XAI top edges.

The existing graph is data-engineered from realised returns. The learned graph is model-generated from multimodal node embeddings.

---

## 11. Current implementation flow

```text
RegimeSnapshotDataset
    ├── loads Temporal Encoder embeddings
    ├── aggregates FinBERT text context
    ├── loads graph snapshot files
    ├── loads FRED macro rows
    └── produces node_features + macro_features + label metadata

MTGNNGraphLearner
    ├── node encoder
    ├── query/key scoring
    └── top-k adjacency

graph_properties_from_adjacency
    └── converts learned adjacency into 7 graph properties

MTGNNRegimeModel
    ├── calls MTGNNGraphLearner
    ├── extracts graph properties
    └── passes graph properties to downstream classifier
```

This document focuses on the graph-building parts, not the full classifier.

---

## 12. Output artefacts relevant to graph building

### 12.1 Graph summary files

The `build-graph` command writes:

```text
outputs/results/MTGNNRegime/graph_summary_chunk{N}_{split}.csv
```

These files summarise snapshot-level graph and stress metrics.

Typical columns include:

```text
date
label
macro_stress_score
graph_stress_score
market_vol_21d
market_ret_21d
market_drawdown_63d
xsec_dispersion
existing_edges
existing_density
existing_avg_degree_norm
existing_mean_abs_corr
existing_max_abs_corr
sector_concentration
```

### 12.2 Learned graph properties in prediction files

When full regime prediction is run, learned graph properties are saved in:

```text
outputs/results/MTGNNRegime/predictions_chunk{N}_{split}.csv
```

Relevant graph-builder columns include:

```text
graph_density
avg_degree_norm
std_degree_norm
mean_edge_weight
max_edge_weight
graph_entropy
learned_graph_stress
```

### 12.3 XAI graph edge outputs

XAI files are saved under:

```text
outputs/results/MTGNNRegime/xai/
```

Graph-builder-relevant XAI includes Level 1 graph properties, Level 2 top learned edges, graph-diff records showing added/removed top edges between snapshots, and counterfactual behaviour when node/text features are altered.

---

## 13. CLI commands relevant to graph building

### Compile

```bash
python -m py_compile code/gnn/mtgnn_regime.py
```

### Inspect required inputs

```bash
python code/gnn/mtgnn_regime.py inspect --repo-root .
```

### Smoke test graph learner

```bash
python code/gnn/mtgnn_regime.py smoke --repo-root . --device cuda
```

### Build graph/regime summary table

```bash
python code/gnn/mtgnn_regime.py build-graph --repo-root . --chunk 1 --split train --device cuda --node-limit 512 --max-snapshots 5
```

This command is the cleanest graph-builder-specific run.

### Build graph summary for validation/test

```bash
python code/gnn/mtgnn_regime.py build-graph --repo-root . --chunk 1 --split val --device cuda --node-limit 768
python code/gnn/mtgnn_regime.py build-graph --repo-root . --chunk 1 --split test --device cuda --node-limit 768
```

---

## 14. Graph-builder validation checks

A successful graph-builder run should show:

```text
snapshots > 0
nodes > 0
macro_cols > 0
node_feature_finite = 1.000000 or very close
label_counts printed for snapshot records
graph_summary_chunk{N}_{split}.csv saved
```

Example style of expected output:

```text
Building chunk1_train: snapshots=63, nodes=768, macro_cols=10
chunk1_train: 61 samples | label_counts={...} | node_feature_finite=1.000000
Saved graph summary: outputs/results/MTGNNRegime/graph_summary_chunk1_train.csv
```

Validation command:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path

for p in sorted(Path('outputs/results/MTGNNRegime').glob('graph_summary_chunk*_*.csv')):
    df = pd.read_csv(p)
    print('\n', p)
    print('shape:', df.shape)
    print('columns:', list(df.columns))
    print(df.head().to_string(index=False))
PY
```

---

## 15. Important design limitations

This MTGNN documentation is intentionally limited. The current project does **not** use MTGNN as a full time-series forecasting model. Therefore:

- no MTGNN forecasting loss is documented here;
- no autoregressive output is documented here;
- no price prediction target is documented here;
- no standalone MTGNN trading signal is produced here.

The graph builder is used only to construct a learned graph representation and graph-property features for regime analysis.

The full regime-risk module documentation should separately explain regime labels, classifier training, macro stress interpretation, prediction outputs, regime XAI, and how regime risk affects position sizing.

---

## 16. Final interpretation

The MTGNN-style graph builder gives the project a learned, multimodal market-connectivity graph. It combines:

```text
Temporal Encoder features
FinBERT text context
Top-k learned adjacency
Graph property extraction
```

This is the correct use of MTGNN for the current architecture because it avoids adding another redundant forecasting model and instead uses MTGNN where it is most useful: **adaptive graph construction**.

Final role:

```text
MTGNN graph builder = internal graph construction layer for regime risk
```

not:

```text
MTGNN = standalone final predictive module
```

This keeps the architecture clean, explainable, and consistent with the project’s distributed modular design.
