# `code/gnn/` Folder Documentation

**Project:** An Explainable Multimodal Neural Framework for Financial Risk Management  
**Folder:** `code/gnn/`  
**Purpose:** Graph construction, graph neural network modelling, contagion-risk estimation, and regime graph-learning support  
**Status:** Active production/research folder used by the downstream Risk Engine, Position Sizing Engine, Quantitative Analyst, and future Fusion Engine  

---

## 1. Purpose of this document

This document provides the folder-level documentation for the entire `code/gnn/` package in the fin-glassbox project. It is intended to act as the high-level index and integration guide for all graph-related code and graph-related documentation.

The folder contains two different graph responsibilities:

1. **Cross-Asset Relation Data Construction**  
   This creates graph artefacts from market returns, sector information, beta estimates, ETF/sector similarity, and rolling correlation snapshots.

2. **Graph Neural Network Modules**  
   These consume graph-like market structure or learn graph structure internally for risk modelling. In the current architecture, this includes:
   - **StemGNN** for contagion risk;
   - **MTGNN-style graph learning** inside the Regime Risk module.

The folder should be understood as the project’s graph intelligence layer. It is not the main temporal encoder and it is not the final decision layer. Its role is to build and model cross-asset relationships that are needed for systemic risk, contagion risk, and regime-state understanding.

---

## 2. Relationship to the overall architecture

In the final workflow, graph-based components live mainly inside the **Risk Engine**:

```text
INPUT DATA
├── Time-Series Market Data
├── Macro / Regime Data
├── Financial Text Embeddings
└── Cross-Asset Relation Data

code/gnn/
├── Cross-Asset Relation Data Builder
├── StemGNN Contagion Risk Module
└── MTGNN-style Graph Builder for Regime Risk

RISK ENGINE
├── GNN Contagion Risk
└── Regime Risk

SYNTHESIS
├── Position Sizing Engine
├── Quantitative Analyst
└── Fusion Engine
```

The key architectural rule is:

```text
GNNs are not used as the main technical encoder.
GNNs are used specifically for relationship-aware risk modelling.
```

This avoids architecture drift. The Temporal Encoder remains responsible for market-sequence embeddings, while the graph modules model inter-asset structure and systemic spillover.

---

## 3. Folder responsibilities

The `code/gnn/` folder has four responsibilities:

### 3.1 Build cross-asset graph data

The graph builder creates data under `data/graphs/`. These artefacts include:

- ticker universe;
- market-return matrix;
- sector mapping;
- beta estimates;
- dollar-volume market-cap proxy;
- sector-similarity data;
- static graph edges;
- rolling correlation snapshots;
- a sample combined graph for inspection.

### 3.2 Implement StemGNN contagion risk

StemGNN estimates whether each stock is vulnerable to future downside spillover at multiple horizons. It consumes market returns and produces ticker-level contagion scores.

### 3.3 Implement MTGNN-style graph learning for regime risk

The MTGNN component is only partially used in this project. It is used as a **learned graph builder**, not as a full MTGNN forecasting system. It learns sparse adjacency matrices from node features and converts them into graph properties for the regime model.

### 3.4 Provide XAI-ready graph outputs

The graph modules provide interpretable outputs such as:

- adjacency summaries;
- top connected/influential nodes;
- graph density and graph stress;
- gradient-based importance;
- GNNExplainer-style approximations where applicable;
- module-level XAI JSON summaries.

---

## 4. Related documentation index

This folder-level document should be read together with the module-specific documents below.

### 4.1 Cross-Asset Relation Data

- [CrossAssetRelationData.md](CrossAssetRelationData.md)  
  Older/broader cross-asset relation data reference.

- [CrossAssetRelationData_Builder.md](CrossAssetRelationData_Builder.md)  
  Final builder-level documentation explaining construction logic, inputs, outputs, graph snapshots, static metadata, and downstream usage.

### 4.2 StemGNN

- [StemGNN.md](StemGNN.md)  
  Earlier StemGNN module documentation.

- [StemGNN_Contagion.md](StemGNN_Contagion.md)  
  Updated StemGNN contagion-risk documentation.

- [StemGNN_Contagion_Documentation.md](StemGNN_Contagion_Documentation.md)  
  Final comprehensive StemGNN contagion module documentation.

### 4.3 MTGNN

- [MTGNN.md](MTGNN.md)  
  MTGNN graph-building usage documentation.

- [MTGNN_GraphBuilder_Documentation.md](MTGNN_GraphBuilder_Documentation.md)  
  Final MTGNN graph-builder documentation limited to how MTGNN is used inside this project.

### 4.4 Pre-implementation specifications

- [GNN_Pre_Specifications.md](GNN_Pre_Specifications.md)  
  Initial specification document describing why StemGNN and MTGNN were selected and how they were intended to fit into the architecture.

---

## 5. Source files in `code/gnn/`

The current graph-related source files are expected to include the following.

| File | Purpose | Current role |
|---|---|---|
| `build_cross_asset_graph.py` | Builds static and dynamic cross-asset graph artefacts | Data builder |
| `stemgnn_base_model.py` | Contains the cleaned StemGNN base architecture | Model backbone |
| `stemgnn_contagion.py` | Wraps StemGNN for contagion-risk training, HPO, prediction, and XAI | Main StemGNN module |
| `stemgnn_forecast_dataloader.py` | Compatibility/support dataloader utilities from the StemGNN baseline context | Supporting utility |
| `stemgnn_handler.py` | Compatibility/handler utilities from the baseline reproduction context | Supporting utility |
| `stemgnn_utils.py` | Utility functions used by the StemGNN baseline/support code | Supporting utility |
| `mtgnn_regime.py` | MTGNN-style graph learner plus regime classifier implementation | Graph builder + regime module implementation |

The `stemgnn_*` files are retained because the project has a baseline-reproduction history. The actively integrated contagion model is driven by `stemgnn_contagion.py`, but the base/support files document and preserve the reproducibility path from the baseline work.

---

## 6. Cross-Asset Relation Data Builder

### 6.1 Primary file

```text
code/gnn/build_cross_asset_graph.py
```

### 6.2 Purpose

The builder constructs graph data from market-derived signals and metadata. It turns flat ticker-level data into relational artefacts that describe how assets connect to each other.

It exists because financial risk is not independent across stocks. Stocks may be connected through:

- sector membership;
- common ETF baskets;
- return correlations;
- similar beta exposure;
- macro-sensitive co-movement;
- crisis-period correlation spikes;
- shared liquidity stress.

The cross-asset graph family therefore supports systemic-risk reasoning and provides the relational market structure used by downstream GNN modules.

### 6.3 Core inputs

Typical inputs are:

```text
data/yFinance/processed/returns_panel_wide.csv
data/yFinance/processed/ohlcv_final.csv
data/sec_edgar/processed/cleaned/issuer_master_cleaned.csv
data/sec_edgar/processed/cleaned/cik_ticker_map_cleaned.csv
```

The exact availability depends on the local repository state, but the builder is designed around market returns, OHLCV information, and issuer metadata.

### 6.4 Core outputs

The builder writes under:

```text
data/graphs/
├── metadata/
├── returns/
├── static/
├── snapshots/
└── combined/
```

Important outputs include:

| Output | Meaning |
|---|---|
| `data/graphs/metadata/ticker_universe.csv` | Final graph ticker list |
| `data/graphs/metadata/sector_map.csv` | Ticker-sector mapping |
| `data/graphs/metadata/market_cap_proxy.csv` | Dollar-volume market-cap proxy |
| `data/graphs/metadata/betas.csv` | Beta estimates versus benchmark where possible |
| `data/graphs/metadata/sector_similarity.csv` | Sector-level return similarity |
| `data/graphs/returns/returns_matrix.csv` | Graph-ready returns matrix |
| `data/graphs/static/edges.csv` | Static graph relationships |
| `data/graphs/snapshots/edges_YYYY-MM-DD.csv` | Dynamic rolling correlation graph snapshots |
| `data/graphs/combined/sample_graph.pkl` | Inspectable sample graph object |

### 6.5 Dynamic snapshots

Dynamic snapshots represent how market correlation structure changes through time. Each snapshot is an edge list with columns such as:

```text
window_start
source
target
correlation
```

Each file describes the strongest relationships observed within a rolling historical window. These snapshots are important because market connectivity is regime-dependent: the graph in calm markets is not the same as the graph in stress regimes.

### 6.6 Static graph data

Static graph edges are slower-moving relationships such as:

- ticker-sector membership;
- sector similarity;
- ETF/return-similarity relationships;
- metadata-derived relationships.

Static edges are useful for auditability and can support graph initialisation or later graph regularisation.

---

## 7. StemGNN Contagion Risk Module

### 7.1 Primary files

```text
code/gnn/stemgnn_base_model.py
code/gnn/stemgnn_contagion.py
code/gnn/stemgnn_forecast_dataloader.py
code/gnn/stemgnn_handler.py
code/gnn/stemgnn_utils.py
```

### 7.2 Purpose

StemGNN estimates contagion risk across assets. Its central question is:

```text
If market stress spreads through the asset network, how vulnerable is each stock over future horizons?
```

The module outputs ticker-level probabilities/scores for contagion-style downside risk over:

```text
5 trading days
20 trading days
60 trading days
```

### 7.3 Why StemGNN is used

StemGNN is appropriate because it combines:

- learned latent correlation;
- graph Fourier/spectral processing;
- temporal sequence modelling;
- multi-hop relationship propagation;
- node-level outputs for many stocks simultaneously.

This is stronger than using a simple correlation matrix because contagion may be non-linear, lagged, and regime-dependent.

### 7.4 Input data

The module primarily consumes:

```text
data/yFinance/processed/returns_panel_wide.csv
```

This file contains a wide daily return matrix:

```text
date × ticker returns
```

The module builds sliding windows from the return matrix and constructs future-horizon contagion targets.

### 7.5 Target construction

The implemented contagion framing is risk-oriented rather than price-forecast oriented. Instead of predicting the exact future price, it predicts whether a future negative movement crosses an extreme-risk threshold at several horizons.

The target is conceptually:

```text
future downside event over horizon h → contagion target for that horizon
```

The model predicts three outputs per stock:

```text
contagion_5d
contagion_20d
contagion_60d
```

### 7.6 Outputs

The main output root is:

```text
outputs/results/StemGNN/
```

Expected prediction files:

```text
outputs/results/StemGNN/contagion_scores_chunk1_val.csv
outputs/results/StemGNN/contagion_scores_chunk1_test.csv
outputs/results/StemGNN/contagion_scores_chunk2_val.csv
outputs/results/StemGNN/contagion_scores_chunk2_test.csv
outputs/results/StemGNN/contagion_scores_chunk3_val.csv
outputs/results/StemGNN/contagion_scores_chunk3_test.csv
```

Expected model files:

```text
outputs/models/StemGNN/chunk1/best_model.pt
outputs/models/StemGNN/chunk1/final_model.pt
outputs/models/StemGNN/chunk1/model_freezed/model.pt
outputs/models/StemGNN/chunk1/model_unfreezed/model.pt

outputs/models/StemGNN/chunk2/...
outputs/models/StemGNN/chunk3/...
```

Expected HPO output:

```text
outputs/codeResults/StemGNN/best_params_chunk1.json
outputs/codeResults/StemGNN/best_params_chunk2.json
outputs/codeResults/StemGNN/best_params_chunk3.json
```

### 7.7 XAI outputs

StemGNN provides graph-specific explainability information such as:

- learned adjacency summaries;
- influential nodes;
- gradient node/edge importance;
- GNNExplainer approximation where enabled;
- top-risk ticker explanations.

XAI outputs are saved under:

```text
outputs/results/StemGNN/xai/
```

### 7.8 Integration usage

StemGNN outputs feed:

```text
Position Sizing Engine
Quantitative Analyst
Fusion Engine
XAI Layer
```

The Position Sizing Engine uses StemGNN contagion as one of its risk weights. In the approved weighting scheme, contagion risk has the highest single risk weight because systemic contagion can invalidate diversification during stress periods.

---

## 8. MTGNN graph-builder usage

### 8.1 Primary file

```text
code/gnn/mtgnn_regime.py
```

### 8.2 Scope limitation

The project does **not** use MTGNN as a full forecasting architecture. It uses the MTGNN-inspired graph-construction component inside the regime module.

The MTGNN usage here is limited to:

```text
learn graph adjacency from node features → summarise graph properties → support regime risk
```

The full regime classifier and regime-risk interpretation should be documented in the Regime Risk Module documentation. This folder-level document only explains why MTGNN belongs in `code/gnn/` and how its graph-learning component interacts with the graph folder.

### 8.3 Inputs

The MTGNN regime implementation may consume:

```text
outputs/embeddings/TemporalEncoder/chunk*_split_embeddings.npy
outputs/embeddings/TemporalEncoder/chunk*_split_manifest.csv
outputs/embeddings/FinBERT/chunk*_split_embeddings.npy
outputs/embeddings/FinBERT/chunk*_split_metadata.csv
data/FRED_data/outputs/macro_features_trading_days_clean.csv
data/graphs/snapshots/edges_YYYY-MM-DD.csv
```

The temporal embeddings represent stock-market sequence behaviour. FinBERT embeddings provide text context. FRED data provides macro context. Existing graph snapshots provide graph-stress context and historical market connectivity.

### 8.4 Learned graph construction

The MTGNN-style learner produces a sparse adjacency matrix:

```text
[batch, nodes, nodes]
```

That adjacency is not the final output. It is converted into graph properties such as:

- graph density;
- normalised average degree;
- degree dispersion;
- mean edge weight;
- maximum edge weight;
- graph entropy;
- learned graph stress.

These graph properties are then used in the regime module.

### 8.5 Outputs

Important outputs include:

```text
outputs/results/MTGNNRegime/graph_summary_chunk{chunk}_{split}.csv
outputs/results/MTGNNRegime/predictions_chunk{chunk}_{split}.csv
outputs/results/MTGNNRegime/xai/chunk{chunk}_{split}_xai.json
outputs/models/MTGNNRegime/chunk{chunk}/best_model.pt
```

The graph summary is useful for inspecting the graph-building component directly. The predictions are consumed by downstream risk modules.

---

## 9. Data flow across the graph folder

The graph layer has two partly independent flows.

### 9.1 Explicit graph-data flow

```text
market returns + metadata
        ↓
build_cross_asset_graph.py
        ↓
data/graphs/
├── static edges
├── dynamic snapshots
└── graph metadata
        ↓
MTGNN regime support / graph inspection / graph stress features
```

### 9.2 Learned contagion flow

```text
returns_panel_wide.csv
        ↓
StemGNN sliding-window dataset
        ↓
StemGNN spectral-temporal GNN
        ↓
contagion_5d / contagion_20d / contagion_60d
        ↓
Position Sizing + Quantitative Analyst + Fusion
```

### 9.3 Learned regime-graph flow

```text
Temporal embeddings + FinBERT embeddings + macro context
        ↓
MTGNN-style graph learner
        ↓
learned adjacency matrix
        ↓
graph properties
        ↓
regime classification / regime risk
        ↓
Position Sizing + Quantitative Analyst + Fusion
```

---

## 10. Chunking and chronology

The wider project uses chronological chunks:

| Chunk | Train | Validation | Test |
|---|---|---|---|
| Chunk 1 | 2000–2004 | 2005 | 2006 |
| Chunk 2 | 2007–2014 | 2015 | 2016 |
| Chunk 3 | 2017–2022 | 2023 | 2024 |

The graph modules must preserve chronological safety. For training and validation, only historical information available up to the relevant date should be used. The graph snapshot date and market-return windows must not look into the future relative to the prediction date.

---

## 11. Integration contracts

### 11.1 StemGNN contract

Expected prediction schema:

```text
ticker
contagion_5d
contagion_20d
contagion_60d
```

The output may also include auxiliary confidence, rank, or XAI fields depending on the final code version.

### 11.2 MTGNN regime contract

Expected prediction schema:

```text
date
pred_regime_id
pred_regime_label
confidence
prob_calm
prob_volatile
prob_crisis
prob_rotation
graph_density
avg_degree_norm
learned_graph_stress
macro_stress_score
```

The Position Sizing Engine uses regime labels and probabilities to apply hard risk caps.

### 11.3 Cross-asset graph contract

Expected snapshot schema:

```text
window_start
source
target
correlation
```

Expected metadata outputs include ticker universe, sector map, beta estimates, and market-cap proxy.

---

## 12. XAI expectations

The graph folder contributes to explainability in three ways.

### 12.1 Structural XAI

Graph data is interpretable because edges have explicit meanings:

```text
AAPL ↔ MSFT because rolling correlation is high
AAPL → Technology because sector membership links it
Sector A ↔ Sector B because sector return similarity is high
```

### 12.2 Model XAI

StemGNN and MTGNN provide model-level explanations:

- learned graph structure;
- top influential nodes;
- node/edge importance;
- top edges from adjacency;
- gradient-based importance;
- counterfactual-style graph stress changes where implemented.

### 12.3 Downstream XAI

Graph outputs appear in downstream explanations:

```text
Position size was reduced because contagion risk was high.
Market regime was classified as crisis because graph density and macro stress increased.
```

This makes graph modelling visible to the final system explanation.

---

## 13. Recommended operating sequence

A full graph-layer run usually follows this sequence:

```text
1. Build or verify market data.
2. Build cross-asset graph artefacts.
3. Train/HPO StemGNN contagion module.
4. Predict StemGNN validation/test contagion scores.
5. Build MTGNN graph summaries or train regime module.
6. Predict regime validation/test outputs.
7. Run Position Sizing and Quantitative Analyst.
```

Example single-line commands are intentionally kept without shell line continuations.

### 13.1 Build graph data

```bash
cd ~/fin-glassbox && python code/gnn/build_cross_asset_graph.py
```

### 13.2 StemGNN smoke and training

```bash
cd ~/fin-glassbox && python -m py_compile code/gnn/stemgnn_base_model.py code/gnn/stemgnn_contagion.py code/gnn/stemgnn_forecast_dataloader.py code/gnn/stemgnn_handler.py code/gnn/stemgnn_utils.py && python code/gnn/stemgnn_contagion.py smoke --repo-root . --device cuda --ticker-limit 64 --batch-size 2 --num-workers 0 --epochs 1 --max-train-windows 4 --max-eval-windows 2
```

```bash
cd ~/fin-glassbox && python code/gnn/stemgnn_contagion.py hpo --repo-root . --chunk 1 --trials 50 --device cuda --fresh && python code/gnn/stemgnn_contagion.py train-best --repo-root . --chunk 1 --device cuda --fresh && python code/gnn/stemgnn_contagion.py predict --repo-root . --chunk 1 --split val --device cuda && python code/gnn/stemgnn_contagion.py predict --repo-root . --chunk 1 --split test --device cuda
```

### 13.3 MTGNN graph/regime smoke and use

```bash
cd ~/fin-glassbox && python -m py_compile code/gnn/mtgnn_regime.py && python code/gnn/mtgnn_regime.py smoke --repo-root . --device cuda
```

```bash
cd ~/fin-glassbox && python code/gnn/mtgnn_regime.py build-graph --repo-root . --chunk 1 --split train --device cuda --node-limit 768
```

```bash
cd ~/fin-glassbox && python code/gnn/mtgnn_regime.py hpo --repo-root . --chunk 1 --trials 30 --device cuda --fresh --node-limit 768 && python code/gnn/mtgnn_regime.py train-best --repo-root . --chunk 1 --device cuda --fresh --node-limit 768 && python code/gnn/mtgnn_regime.py predict --repo-root . --chunk 1 --split val --device cuda --node-limit 768 && python code/gnn/mtgnn_regime.py predict --repo-root . --chunk 1 --split test --device cuda --node-limit 768
```

---

## 14. Validation commands

### 14.1 StemGNN output audit

```bash
cd ~/fin-glassbox && for c in 1 2 3; do echo "===== StemGNN chunk$c ====="; ls -lh outputs/models/StemGNN/chunk${c}/best_model.pt outputs/models/StemGNN/chunk${c}/final_model.pt outputs/models/StemGNN/chunk${c}/model_freezed/model.pt outputs/results/StemGNN/contagion_scores_chunk${c}_val.csv outputs/results/StemGNN/contagion_scores_chunk${c}_test.csv; done
```

### 14.2 MTGNN output audit

```bash
cd ~/fin-glassbox && for c in 1 2 3; do for s in val test; do echo "===== MTGNNRegime chunk${c}_${s} ====="; ls -lh outputs/results/MTGNNRegime/predictions_chunk${c}_${s}.csv outputs/results/MTGNNRegime/xai/chunk${c}_${s}_xai.json 2>/dev/null || true; done; done
```

### 14.3 Graph data audit

```bash
cd ~/fin-glassbox && find data/graphs -maxdepth 3 -type f | sort | head -100
```

```bash
cd ~/fin-glassbox && python -c "import pandas as pd, pathlib; files=sorted(pathlib.Path('data/graphs/snapshots').glob('edges_*.csv')); print('snapshots=', len(files)); print('first=', files[0] if files else None); print('last=', files[-1] if files else None); df=pd.read_csv(files[0]); print(df.head().to_string(index=False)); print('rows=', len(df)) if files else None"
```

---

## 15. Common risks and safeguards

### 15.1 Look-ahead bias

Graph snapshots and labels must use only information available at or before the relevant prediction date. Rolling windows must not include future returns.

### 15.2 Over-large graph models

The full 2,500-stock universe can produce large adjacency structures. Node limits, sparse top-k edges, batch-size control, and checkpointing are necessary for stable GPU use.

### 15.3 NaN loss in GNN training

GNN models can become unstable with high learning rates or extreme class imbalance. The StemGNN implementation should guard against non-finite losses, clip gradients, and avoid unstable HPO ranges.

### 15.4 File handle pressure

Large DataLoader worker counts can trigger open-file limits. The StemGNN implementation should keep worker count conservative and raise the open-file soft limit where safe.

### 15.5 Sparse or missing text embeddings for MTGNN

The MTGNN graph builder may need to aggregate sparse FinBERT text context. Missing text context should be treated as absent evidence, not positive evidence.

---


The folder is therefore a completed core graph layer for the current project scope, while still allowing future extension into more advanced graph fusion, graph regularisation, or graph-aware portfolio construction.
