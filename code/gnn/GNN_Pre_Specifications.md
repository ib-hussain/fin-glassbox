# GNN_Pre_Specifications.md

## Document Purpose

This document contains **all finalized decisions, specifications, and technical requirements** for the two Graph Neural Networks (GNNs) to be implemented in the **Explainable Distributed Deep Learning Framework for Financial Risk Management**.

This is a **pre-implementation specification** — it defines WHAT to build before writing code. The actual number of tickers will be finalized after market data acquisition (expected: 4,428-4,534 stocks).

---

## Table of Contents

1. [Overview: Two GNNs in the System](#overview-two-gnns-in-the-system)
2. [GNN 1: StemGNN — Contagion Risk Module](#gnn-1-stemgnn--contagion-risk-module)
3. [GNN 2: MTGNN Graph Builder — Regime Detection Module](#gnn-2-mtgnn-graph-builder--regime-detection-module)
4. [Data Requirements for Both GNNs](#data-requirements-for-both-gnns)
5. [Shared Infrastructure](#shared-infrastructure)
6. [Implementation Checklist](#implementation-checklist)
7. [XAI Requirements for GNNs](#xai-requirements-for-gnns)
8. [Validation Metrics](#validation-metrics)
9. [File Structure Plan](#file-structure-plan)

---

## Overview: Two GNNs in the System

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: 4,428-4,534 Stocks                    │
│              30-90 days of daily price/returns data              │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
                v                               v
┌──────────────────────────────┐  ┌──────────────────────────────┐
│                              │  │                              │
│     StemGNN (Full Model)     │  │   MTGNN (Graph Part Only)    │
│                              │  │                              │
│   CONTAGION RISK MODULE      │  │   REGIME DETECTION MODULE    │
│                              │  │                              │
└──────────────────────────────┘  └──────────────────────────────┘
                │                               │
                v                               v
┌──────────────────────────────┐  ┌──────────────────────────────┐
│  Output: Contagion Score     │  │  Output: Regime Label        │
│  per stock (0-1 scale)       │  │  (calm/volatile/crisis/rot)  │
└──────────────────────────────┘  └──────────────────────────────┘
                │                               │
                └───────────────┬───────────────┘
                                v
                    ┌─────────────────────┐
                    │   POSITION SIZING   │
                    │       ENGINE        │
                    └─────────────────────┘
```

---

## GNN 1: StemGNN — Contagion Risk Module

### Purpose (Plain English)

**Question it answers:** *"If stock A crashes, how badly does stock B get hurt?"*

This module models how financial distress spreads through the market. It identifies which stocks are "super-spreaders" of risk and which stocks are most vulnerable to contagion from others.

### Why StemGNN Was Chosen

| Reason | Explanation |
|--------|-------------|
| **Proven performance** | Achieved best MAPE (9.29%) on crypto dataset in baseline reproduction |
| **Handles noise** | Financial contagion is noisy and non-linear — StemGNN excels here |
| **Joint learning** | Learns connections AND time patterns together, not separately |
| **Spectral approach** | Works in frequency domain, capturing hidden cyclical relationships |
| **Existing code** | You already have working StemGNN implementation from Assignment 2 |

### Model Specifications

| Specification | Value |
|--------------|-------|
| **Model Type** | Full StemGNN (spectral-temporal graph neural network) |
| **Input Data** | Daily returns for all stocks (log or simple returns) |
| **Input Shape** | `(N_stocks, T_timesteps, F_features)` |
| **N_stocks** | 4,428-4,534 (exact number TBD after data collection) |
| **T_timesteps** | 30 days (primary), 90 days (extended), 10 days (crisis mode) |
| **F_features** | 1 (returns only, or up to 5 with OHLCV derived features) |
| **Output** | Contagion risk score per stock (float, 0-1 scale) |
| **Training Frequency** | Monthly or quarterly (due to computational cost) |
| **Inference Frequency** | Daily or weekly (using cached trained model) |

### What StemGNN Does — Step by Step

```
Step 1: INPUT
    Takes 30 days of returns for all 4,428 stocks
    Shape: (4428, 30, 1)
    
Step 2: LATENT CORRELATION LAYER
    Uses GRU + Self-Attention to learn:
    "Which stocks move together?"
    Output: Adjacency matrix (4428 × 4428)
    Each cell: Connection strength between stock i and stock j
    
Step 3: GRAPH FOURIER TRANSFORM (GFT)
    Converts the adjacency matrix to spectral domain
    Finds hidden patterns that simple correlation misses
    
Step 4: SPECTRAL-TEMPORAL BLOCKS
    Multiple layers of:
    - Graph convolution (spreads information across connections)
    - Temporal convolution (captures time patterns)
    - Skip connections (prevents information loss)
    
Step 5: MESSAGE PASSING
    Each stock receives "risk signals" from its connected neighbors
    After multiple rounds, each stock knows the risk level of its entire network
    
Step 6: OUTPUT LAYER
    Fully connected layer produces:
    - Contagion risk score per stock (0 = safe, 1 = highly vulnerable)
    - Optional: Risk propagation paths (for XAI)
```

### Input Data Details

| Field | Description | Source |
|-------|-------------|--------|
| `returns` | Daily log or simple returns | Derived from yfinance closing prices |
| `window_length` | 30 days (configurable: 10, 30, 90) | Config parameter |
| `normalization` | Z-score normalization per stock | Standard practice |

**Input Example (conceptual):**
```
Stock      Day1    Day2    Day3    ...    Day30
AAPL      0.012  -0.008   0.003   ...    0.015
MSFT      0.008  -0.005   0.001   ...    0.012
GOOGL     0.015  -0.010   0.004   ...    0.018
...       ...     ...      ...     ...    ...
XYZ      -0.003   0.002  -0.001   ...    0.005
```

### Output Data Details

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `contagion_score` | float | 0.0 - 1.0 | Overall vulnerability to spillover risk |
| `network_centrality` | float | 0.0 - 1.0 | How influential this stock is (super-spreader) |
| `cluster_id` | int | 0 to K | Which "crash cluster" this stock belongs to |
| `top_influencers` | list[str] | - | List of tickers that most affect this stock |
| `top_influencees` | list[str] | - | List of tickers most affected by this stock |

**Output Example:**
```json
{
  "AAPL": {
    "contagion_score": 0.72,
    "network_centrality": 0.89,
    "cluster_id": 3,
    "top_influencers": ["MSFT", "NVDA", "QQQ"],
    "top_influencees": ["SPY", "XLK", "AAPL_suppliers"]
  }
}
```

### Hyperparameters (Initial Values)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `window_size` | 30 | Days of lookback |
| `horizon` | 1 | Predict 1 step ahead (contagion, not price) |
| `multi_layer` | 13 | Number of StemGNN blocks |
| `learning_rate` | 0.01 | From baseline experiments |
| `exponential_decay_step` | 13 | LR scheduler |
| `decay_rate` | 0.5 | LR scheduler |
| `dropout_rate` | 0.75 | Regularization |
| `batch_size` | 32 | Training batch size |
| `epochs` | 100 | Training epochs (with early stopping) |
| `optimizer` | RMSProp | From baseline |
| `norm_method` | z_score | Input normalization |

*Note: These are starting points from successful baseline runs. Hyperparameter tuning will be performed.*

### Training Protocol

```
For each training cycle (monthly/quarterly):
    1. Collect latest 2 years of daily returns
    2. Split: 75% train, 15% validation, 10% test (chronological)
    3. Normalize using z-score
    4. Create sliding windows (window_size=30, stride=1)
    5. Train StemGNN with early stopping (patience=20)
    6. Evaluate on test set
    7. Save model checkpoint
    8. (Optional) Hyperparameter tuning every 3 months
```


---

## GNN 2: MTGNN Graph Builder — Regime Detection Module

### Purpose (Plain English)

**Question it answers:** *"What mood is the market in right now?"*

This module analyzes the structure of stock relationships to classify the current market environment. It receives input from both price patterns (Temporal Encoder) and news sentiment (FinBERT).

### Why MTGNN Graph Builder Was Chosen

| Reason | Explanation |
|--------|-------------|
| **Clean graph learning** | MTGNN builds sparse, interpretable relationship graphs |
| **Multi-input capable** | Can combine temporal features AND text sentiment |
| **Fast** | Graph building only (no full training) is computationally light |
| **Graph structure is regime** | The shape of the graph IS the regime — no complex prediction needed |
| **Existing code** | You already have MTGNN graph learning layer code |

### What "Graph Builder Only" Means

```
Full MTGNN:
    [Graph Learning] → [Graph Conv] → [Temporal Conv] → [Prediction]
    
What We Use:
    [Graph Learning] → [Graph Properties Analysis] → [Regime Label]
    
We STOP after building the graph. We don't predict prices. 
We just look at what the graph looks like.
```

### Model Specifications

| Specification | Value |
|--------------|-------|
| **Model Type** | MTGNN Graph Learning Layer + Simple Classifier |
| **Input 1 (Temporal)** | Encoded features from Shared Temporal Attention Encoder |
| **Input 1 Shape** | `(N_stocks, D_temporal)` where D_temporal = 64 or 128 |
| **Input 2 (Text)** | FinBERT embeddings aggregated per stock |
| **Input 2 Shape** | `(N_stocks, D_text)` where D_text = 768 (FinBERT base) |
| **Combined Input** | Concatenated: `(N_stocks, D_temporal + D_text)` |
| **N_stocks** | 4,428-4,534 |
| **Output** | Regime label + confidence score |
| **Training Frequency** | Weekly |
| **Inference Frequency** | Daily |

### What MTGNN Graph Builder Does — Step by Step
```
Step 1: INPUT
    Takes combined features for all stocks
    Shape: (4428, 832)  [64 temporal + 768 text]
    
Step 2: NODE EMBEDDING
    Projects each stock to a lower-dimensional space
    Shape: (4428, 64)
    
Step 3: GRAPH LEARNING LAYER (MTGNN core)
    Computes similarity between all pairs of node embeddings
    Uses self-attention: "How much does stock i attend to stock j?"
    Applies Top-K sparsification (K = 66 edges per node)
    Output: Adjacency matrix (4428 × 4428)
    
Step 4: GRAPH PROPERTY EXTRACTION
    Calculates metrics from the adjacency matrix:
    - Graph density (how connected is the market?)
    - Modularity (are there distinct clusters?)
    - Average degree (how many connections per stock?)
    - Clustering coefficient (do friends of friends connect?)
    
Step 5: REGIME CLASSIFICATION
    Simple classifier (MLP or rule-based) maps graph properties to regime:

    | Density | Modularity | Degree | → | Regime                    |
    |---------|------------|--------|---|---------------------------|
    | Low     | High       | Low    | → | CALM (normal)             |
    | High    | Low        | High   | → | CRISIS (panic)            |
    | Medium  | Medium     | Medium | → | VOLATILE (uncertain)      |
    | Medium  | High       | Medium | → | ROTATION (sector shift)   |

Step 6: OUTPUT
    Returns:
    - regime_label: "calm" | "volatile" | "crisis" | "rotation"
    - regime_confidence: 0.0 - 1.0
    - graph_metrics: Dictionary of all computed properties
```
### Input Data Details

#### Input 1: Temporal Features (from Shared Temporal Attention Encoder)

| Field | Description | Source |
|-------|-------------|--------|
| `temporal_embedding` | Learned representation of price patterns | Temporal Attention Encoder |
| `dimension` | 64 or 128 | Configurable |
| `update_frequency` | Daily | Computed from latest 30-day window |

#### Input 2: Text Features (from FinBERT)

| Field | Description | Source |
|-------|-------------|--------|
| `sentiment_embedding` | FinBERT [CLS] token or pooled output | FinBERT forward pass |
| `dimension` | 768 | Fixed (FinBERT base) |
| `aggregation` | Mean of all news/filings for that stock in past 7 days | News aggregator |
| `update_frequency` | Daily | As new filings/news arrive |

### Output Data Details

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| `regime_label` | string | "calm", "volatile", "crisis", "rotation" | Current market state |
| `regime_confidence` | float | 0.0 - 1.0 | Model confidence in regime classification |
| `graph_density` | float | 0.0 - 1.0 | % of possible edges that exist |
| `modularity` | float | -1.0 - 1.0 | How clustered the graph is |
| `avg_degree` | float | 0 - 4428 | Average connections per stock |
| `transition_probability` | float | 0.0 - 1.0 | Likelihood of regime change next period |

**Output Example:**
```json
{
  "timestamp": "2026-04-22",
  "regime_label": "volatile",
  "regime_confidence": 0.78,
  "graph_density": 0.042,
  "modularity": 0.31,
  "avg_degree": 186.4,
  "transition_probability": 0.35,
  "regime_history": ["calm", "calm", "volatile"]
}
```

### Graph Property → Regime Mapping (Rule-Based Fallback)

If learned classifier is not ready, use these rules:

| Condition | Regime | Confidence Formula |
|-----------|--------|-------------------|
| density < 0.02 AND modularity > 0.4 | CALM | 1.0 - density |
| density > 0.08 | CRISIS | min(1.0, density * 10) |
| density between 0.02-0.08 AND modularity between 0.2-0.4 | VOLATILE | 0.5 + abs(0.05 - density) |
| modularity > 0.3 AND density between 0.03-0.06 | ROTATION | modularity |

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `node_embedding_dim` | 64 | MTGNN graph learner embedding size |
| `top_k` | 66 | Edges per node (√N ≈ √4428) |
| `classifier_hidden` | [128, 64] | MLP layers for regime classification |
| `temporal_feature_dim` | 64 | From Temporal Encoder |
| `text_feature_dim` | 768 | From FinBERT |
| `cheb_k` | 3 | Chebyshev polynomial order (graph conv) |
| `learning_rate` | 0.001 | For classifier training |
| `weight_decay` | 1e-5 | Regularization |

### Training Protocol

```
Weekly training:
    1. Get latest temporal embeddings (from Temporal Encoder)
    2. Get latest text embeddings (from FinBERT aggregator)
    3. Concatenate features
    4. Build graph using MTGNN graph learner
    5. Extract graph properties
    6. (If labeled regime data available) Train classifier
    7. (If not) Use rule-based mapping
    8. Save regime label and confidence
```


---

## Data Requirements for Both GNNs

### Shared Data Source

Both GNNs depend on **market price data** currently being collected via yfinance.

| Data | Format | Frequency | Source |
|------|--------|-----------|--------|
| Daily closing prices | CSV/Parquet | Daily | yfinance |
| Ticker list | CSV | Static | `cik_ticker_map_cleaned.csv` (4,428 tickers) |
| Sector/Industry mapping | CSV | Static | yfinance `info['sector']` |
| ETF holdings | CSV | Quarterly | SEC N-PORT or yfinance |

### Preprocessing Pipeline

```
Raw yfinance Data
       │
       v
[Return Calculator] → Daily log returns
       │
       ├──────────────────┬──────────────────┐
       v                  v                  v
[StemGNN Input]    [Temporal Encoder]  [Correlation Matrix]
(30-day windows)   (Technical features)  (For graph building)
```

---

## Shared Infrastructure

### Common Utilities Needed

| Utility | Purpose | Used By |
|---------|---------|---------|
| `GraphDataset` class | PyTorch Geometric data loader | Both |
| `adjacency_to_edge_index` | Convert matrix to edge list | Both |
| `normalize_returns` | Z-score normalization | Both |
| `rolling_window` | Create sequences for training | StemGNN |
| `save_graph_snapshot` | Cache graphs for analysis | MTGNN |

---

## Implementation Checklist

### Phase 1: Setup (Before Data Arrives)

- [ ] Create directory structure for GNN modules
- [ ] Copy MTGNN graph learning layer code from `assignment2work/MTGNN/layer.py`
- [ ] Copy StemGNN model code from `assignment2work/StemGNN/`
- [ ] Create base classes for both modules
- [ ] Write data loading utilities (skeletons)

### Phase 2: MTGNN Regime Module

- [ ] Implement `RegimeDetectionModule` class
- [ ] Implement graph property extraction functions
- [ ] Implement rule-based regime classifier
- [ ] Add MLP classifier for learned regime detection
- [ ] Write tests with synthetic data
- [ ] Add XAI explanations (graph visualization)

### Phase 3: StemGNN Contagion Module

- [ ] Adapt StemGNN for contagion scoring (not price prediction)
- [ ] Modify output layer for risk scores
- [ ] Implement contagion score calculation
- [ ] Add network centrality metrics
- [ ] Write tests with synthetic data
- [ ] Add XAI explanations (top influencers)

### Phase 4: Integration

- [ ] Connect to Temporal Encoder output
- [ ] Connect to FinBERT output
- [ ] Connect to Position Sizing Engine
- [ ] End-to-end testing
- [ ] Performance benchmarking

---

## XAI Requirements for GNNs

### StemGNN Contagion Module Explanations

| Explanation | Method | Output |
|-------------|--------|--------|
| "Why high contagion score?" | GNNExplainer | Top 3 stocks influencing this score |
| "Which connections matter most?" | Attention weights | Heatmap of strongest edges |
| "What cluster is this stock in?" | Spectral clustering | Cluster ID + member list |
| "How does shock propagate?" | Message passing trace | Path visualization |
More details can be and WILL be added later to the explainations module.

### MTGNN Regime Module Explanations

| Explanation | Method | Output |
|-------------|--------|--------|
| "Why this regime?" | Graph property values | Density, modularity scores |
| "What changed from yesterday?" | Graph diff | Added/removed edges |
| "Which sectors are rotating?" | Community detection | Sector cluster labels |
| "Confidence in regime call?" | Softmax probability | Confidence percentage |
More details can be and WILL be added later to the explainations module.

---

## Validation Metrics

### StemGNN Contagion Module

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Contagion score calibration | Scores correlate with actual drawdown contagion | Backtest: during known crashes, high-score stocks dropped more |
| Stability | Scores don't fluctuate wildly day-to-day | Rolling standard deviation < 0.1 |
| Sparsity | Graph has K≈66 edges per node | Actual average degree |

### MTGNN Regime Module

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Regime accuracy | >70% agreement with VIX-based labels | Compare to VIX thresholds |
| Transition smoothness | < 3 regime changes per month | Count transitions |
| Confidence calibration | High confidence = correct regime | Brier score |

---

## File Structure Plan

```
fin-glassbox/
│
├── code/
│   ├── gnn/
│   │   ├── __init__.py
│   │   ├── stemgnn_contagion.py      # Contagion Risk Module
│   │   ├── mtgnn_regime.py           # Regime Detection Module
│   │   ├── graph_utils.py            # Shared graph utilities
│   │   ├── xai_gnn.py                # GNN-specific explanations
│   │   ├── config_gnn.py             # Hyperparameters
│   │   ├── build_cross_asset_graph.py    # Graph construction
│   │   ├── train_contagion_gnn.py        # StemGNN training
│   │   └── run_regime_detection.py       # MTGNN inference
│   │   └── GNN_Pre_Specifications.md         # THIS FILE
│   ├── Some other directories
│   │   ├── technical_encoder.py          # Shared Temporal Attention
│   │   └── finbert_encoder.py            # Text encoder
│
├── data/
│   ├── yFinance/                  # yfinance output (coming soon)
│   ├── graphs/                       # Saved graph snapshots
│   │   ├── regime_graphs/
│   │   └── contagion_graphs/
│   └── sec_edgar/processed/cleaned/  # Already have
│
├── assignment2work/                   # Your baseline code (reference)
│   ├── MTGNN/
│   ├── StemGNN/
│   └── FourierGNN/
```

---

## Summary Table

| Aspect | StemGNN (Contagion) | MTGNN (Regime) |
|--------|---------------------|----------------|
| **Purpose** | Risk spillover between stocks | Market mood classification |
| **Input** | 30-day returns | Temporal + Text embeddings |
| **Output** | Contagion score (0-1) | Regime label + confidence |
| **Training** | Monthly/Quarterly | Weekly |
| **Inference** | Daily | Daily |
| **GPU Needed** | Yes (T4, 8-12 GB) | Light (2-4 GB) |
| **XAI Method** | GNNExplainer + Attention | Graph properties + Diff |
| **Code Source** | Assignment 2 StemGNN | Assignment 2 MTGNN layer |

---

## Final Approval

This specification document defines:

- ✅ StemGNN for Contagion Risk Module
- ✅ MTGNN Graph Builder for Regime Detection Module
- ✅ Two GNNs total, no more
- ✅ Clear inputs, outputs, and training protocols
- ✅ Implementation checklist ready

**Status: APPROVED FOR IMPLEMENTATION**

*Document Version: 1.0*  
*Last Updated: 2026-04-22*  
*Prepared for: Explainable Distributed Deep Learning Framework for Financial Risk Management*