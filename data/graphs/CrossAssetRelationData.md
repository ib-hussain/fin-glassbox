# CrossAssetRelationData.md

## Document Purpose

This document defines **everything** about the Cross-Asset Relation Data family — what it is, why it's needed, exactly how to build it from market data, and how it feeds into the two GNN modules (StemGNN Contagion and MTGNN Regime Detection).

This is a **build specification** — it defines WHAT to build and HOW to build it.

---

## Table of Contents

1. [What Is Cross-Asset Relation Data?](#1-what-is-cross-asset-relation-data)
2. [Why Is It Needed?](#2-why-is-it-needed)
3. [Where It's Used in the Architecture](#3-where-its-used-in-the-architecture)
4. [Data Sources](#4-data-sources)
5. [Graph Types to Build](#5-graph-types-to-build)
6. [Relationship Vector Specification](#6-relationship-vector-specification)
7. [Build Steps](#7-build-steps)
8. [Output Files](#8-output-files)
9. [File Structure](#9-file-structure)
10. [Complete Build Script Specification](#10-complete-build-script-specification)

---

## 1. What Is Cross-Asset Relation Data?

Cross-Asset Relation Data captures **how stocks relate to each other**. Unlike the other four data families which are **asset-centric** (each row is about one company), this data is **relationship-centric** (each edge represents a connection between two assets).

### Core Concept

In financial markets, assets don't move independently. When one stock drops, it pulls down related stocks. Cross-asset data models these relationships so the GNN can learn **propagation patterns**.

### Data Types

| Type | Description | Example |
|------|-------------|---------|
| **Nodes** | Individual assets (stocks, ETFs) | AAPL, MSFT, SPY |
| **Edges** | Relationships between assets | AAPL ←→ MSFT |
| **Edge Features** | Strength/type of relationship | correlation=0.85, same_sector=1 |
| **Node Features** | Properties of each asset | sector=Technology, market_cap=3T |

---

## 2. Why Is It Needed?

### For StemGNN Contagion Module:

The Contagion module answers: *"If stock A crashes, how badly does stock B get hurt?"*

It needs:
- **A graph structure** showing which stocks are connected
- **Edge weights** showing connection strength
- **Node features** for initial risk state
- **Returns data** to learn dynamic relationships

### For MTGNN Regime Detection Module:

The Regime module answers: *"What mood is the market in right now?"*

It needs:
- **Graph structure** to analyze market connectivity
- **Graph properties** (density, modularity, clustering) that indicate regime
- **Node features** from temporal + text encoders
- **Dynamic graphs** that change over time

---

## 3. Where It's Used in the Architecture

```
INPUT: 4,428 stocks × daily returns
              │
              v
    ┌─────────────────────────┐
    │  CROSS-ASSET GRAPH      │
    │  BUILDER                │
    │  (THIS MODULE)          │
    └─────────────────────────┘
              │
    ┌─────────┴─────────┐
    │                   │
    v                   v
┌──────────────┐  ┌──────────────┐
│  StemGNN     │  │  MTGNN       │
│  Contagion   │  │  Regime      │
│  Module      │  │  Module      │
└──────────────┘  └──────────────┘
    │                   │
    v                   v
┌──────────────┐  ┌──────────────┐
│ Contagion    │  │ Regime Label │
│ Scores       │  │ + Confidence │
└──────────────┘  └──────────────┘
    │                   │
    └─────────┬─────────┘
              v
    ┌─────────────────┐
    │ POSITION SIZING │
    │ ENGINE          │
    └─────────────────┘
```

---

## 4. Data Sources

### Primary Sources (All Free)

| Source | What It Provides | Used For |
|--------|-----------------|----------|
| **Market Data (yfinance)** | Daily closing prices for 4,428 stocks | Correlation computation, returns |
| **cik_ticker_map_cleaned.csv** | Ticker list (4,428 stocks) | Node universe definition |
| **yfinance `info['sector']`** | Sector/industry for each ticker | Sector similarity graph |
| **Market Cap (from yfinance)** | Company size | Node feature, market cap ratio |
| **ETF Holdings (yfinance/SEC)** | Which ETFs hold which stocks | ETF membership graph |
| **Index Membership** | SP500, Nasdaq-100, Russell constituents | Index membership graph |

### Derived Features (Computed from Market Data)

| Feature | Computation | Window |
|---------|-------------|--------|
| Daily Returns | `(close_t - close_t-1) / close_t-1` | 1 day |
| Rolling Correlation | Pearson correlation of returns | 30 days |
| Partial Correlation | Correlation after removing market factor | 30 days |
| Beta | Covariance with SPY / Variance of SPY | 252 days |
| Volume Correlation | Correlation of log volumes | 30 days |

---

## 5. Graph Types to Build

### Overview

We build **FOUR complementary graph types**, each capturing different aspects of asset relationships:

| Graph | Nodes | Edges | Edge Weight | Update Frequency |
|-------|-------|-------|-------------|------------------|
| **1. Correlation Network** | All tickers | Top-K correlated pairs | Rolling 30-day correlation | Every 20 trading days |
| **2. Sector Hierarchy Graph** | Tickers + Sectors | Ticker → Sector membership | Sector similarity (0-1) | Static |
| **3. ETF Membership Graph** | Tickers + ETFs | Ticker → ETF membership | Normalized weight | Quarterly |
| **4. Index Membership Graph** | Tickers + Indices | Ticker → Index membership | Binary | Quarterly |

---

### Graph 1: Correlation Network

**Purpose:** Directly captures which stocks move together.

**Nodes:** All 4,428 tickers

**Edges:** Top-K strongest correlations per node
- K = √4428 ≈ **66 edges per node**
- Total: ~292,000 edges (1.5% dense)
- Edge weight: Rolling 30-day Pearson correlation

**Build Process:**
```python
1. Load daily closing prices for all tickers
2. Compute daily returns: returns = close.pct_change()
3. For each 20-trading-day window:
   a. Compute correlation matrix (4428 × 4428)
   b. For each ticker, find top 66 correlations
   c. Store edge list: (ticker_i, ticker_j, correlation)
4. Save each snapshot with date label
```

**Output Format:**
```
window_start, ticker_i, ticker_j, correlation
2020-01-02, AAPL, MSFT, 0.85
2020-01-02, AAPL, GOOGL, 0.72
...
```

---

### Graph 2: Sector Hierarchy Graph

**Purpose:** Captures same-sector clustering and sector-level spillover.

**Nodes:** 4,428 tickers + ~11 GICS sectors

**Edges:**
- Ticker → Sector: Membership edge (weight = 1.0)
- Sector → Sector: Correlation-based similarity (data-driven, not arbitrary)

**Sector Similarity Computation:**
```python
1. For each sector, compute equal-weighted average daily returns
2. Compute correlation matrix of sector returns (252-day rolling)
3. Normalize to 0-1: similarity = (correlation + 1) / 2
```

**Example Sector Similarity Matrix:**
| | Tech | Healthcare | Financials | Energy |
|--|------|------------|------------|--------|
| Tech | 1.00 | 0.62 | 0.58 | 0.31 |
| Healthcare | 0.62 | 1.00 | 0.47 | 0.23 |
| Financials | 0.58 | 0.47 | 1.00 | 0.52 |
| Energy | 0.31 | 0.23 | 0.52 | 1.00 |

**Output Format:**
```
source, target, relationship_type, weight
AAPL, Technology, sector_membership, 1.0
Technology, Healthcare, sector_similarity, 0.62
Technology, Financials, sector_similarity, 0.58
...
```

---

### Graph 3: ETF Membership Graph

**Purpose:** Captures basket-driven selling and ETF flow effects.

**Nodes:** 4,428 tickers + major ETFs

**ETF Universe:**
| ETF | Description | Holdings |
|-----|-------------|----------|
| SPY | S&P 500 | ~500 |
| QQQ | Nasdaq-100 | ~100 |
| IWM | Russell 2000 | ~2000 |
| DIA | Dow Jones | 30 |
| XLK | Technology | ~70 |
| XLF | Financials | ~70 |
| XLV | Healthcare | ~65 |
| XLE | Energy | ~25 |
| XLI | Industrials | ~75 |
| XLP | Consumer Staples | ~35 |
| XLY | Consumer Discretionary | ~55 |
| XLU | Utilities | ~30 |
| XLB | Materials | ~30 |
| XLRE | Real Estate | ~30 |
| XLC | Communication Services | ~25 |

**Edges:** Ticker → ETF (weight = percentage of ETF portfolio)

**Build Process:**
```python
1. Download ETF holdings from yfinance or SEC N-PORT filings
2. For each ETF, extract constituent tickers and weights
3. Build bipartite graph: Ticker node ←→ ETF node
4. Edge weight: percentage of ETF invested in that ticker
```

**Note:** If ETF holdings data is unavailable, use binary membership (weight=1.0 if ticker is in ETF, else no edge).

**Output Format:**
```
ticker, etf, weight, date
AAPL, SPY, 7.2, 2024-12-31
AAPL, QQQ, 12.5, 2024-12-31
...
```

---

### Graph 4: Index Membership Graph

**Purpose:** Captures passive flow and benchmark effects.

**Nodes:** 4,428 tickers + Major indices

**Indices:**
| Index | Description |
|-------|-------------|
| SP500 | S&P 500 |
| NASDAQ100 | Nasdaq-100 |
| RUSSELL2000 | Russell 2000 |
| DOW30 | Dow Jones Industrial |

**Edges:** Ticker → Index (binary, weight = 1.0 if member)

**Build Process:**
```python
1. Download index constituents (Wikipedia/yfinance)
2. Map constituent tickers to our universe
3. Build bipartite graph
```

**Output Format:**
```
ticker, index, weight
AAPL, SP500, 1.0
AAPL, NASDAQ100, 1.0
AAPL, DOW30, 1.0
...
```

---

## 6. Relationship Vector Specification

### For Each Edge, We Compute an 8-Dimensional Feature Vector

| Dimension | Name | Range | Computation | Source Graph |
|-----------|------|-------|-------------|-------------|
| 1 | `correlation_30d` | -1 to 1 | Rolling Pearson correlation of daily returns | Correlation Network |
| 2 | `sector_similarity` | 0 to 1 | Sector return correlation (252-day) | Sector Graph |
| 3 | `etf_overlap_jaccard` | 0 to 1 | Jaccard similarity of ETF memberships | ETF Graph |
| 4 | `index_co_membership` | 0 to 1 | Both in same index = 1, else 0 | Index Graph |
| 5 | `market_cap_ratio` | 0 to 1 | min(mcap_i, mcap_j) / max(mcap_i, mcap_j) | Market Data |
| 6 | `volume_correlation` | -1 to 1 | Correlation of log daily volumes | Market Data |
| 7 | `beta_similarity` | 0 to 1 | 1 - |beta_i - beta_j| (clipped to 0-1) | Market Data |
| 8 | `partial_correlation_30d` | -1 to 1 | Correlation after removing SPY effect | Correlation Network |

### Computation Details

```python
def compute_relationship_vector(ticker_i, ticker_j, returns_df, volumes_df, 
                                 sector_map, etf_holdings, index_membership,
                                 market_caps, betas, spy_returns):
    """Compute 8-dim relationship vector for a pair of tickers."""
    
    # 1. Rolling correlation (30-day)
    corr_30d = returns_df[ticker_i].rolling(30).corr(returns_df[ticker_j]).iloc[-1]
    
    # 2. Sector similarity (pre-computed sector correlation matrix)
    sector_sim = sector_similarity_matrix[sector_map[ticker_i]][sector_map[ticker_j]]
    
    # 3. ETF overlap (Jaccard)
    etfs_i = set(etf_holdings.get(ticker_i, []))
    etfs_j = set(etf_holdings.get(ticker_j, []))
    jaccard = len(etfs_i & etfs_j) / len(etfs_i | etfs_j) if (etfs_i | etfs_j) else 0
    
    # 4. Index co-membership
    indices_i = set(index_membership.get(ticker_i, []))
    indices_j = set(index_membership.get(ticker_j, []))
    index_co = 1.0 if (indices_i & indices_j) else 0.0
    
    # 5. Market cap ratio
    mcap_i = market_caps.get(ticker_i, 0)
    mcap_j = market_caps.get(ticker_j, 0)
    mcap_ratio = min(mcap_i, mcap_j) / max(mcap_i, mcap_j) if max(mcap_i, mcap_j) > 0 else 0
    
    # 6. Volume correlation
    vol_corr = volumes_df[ticker_i].rolling(30).corr(volumes_df[ticker_j]).iloc[-1]
    
    # 7. Beta similarity
    beta_i = betas.get(ticker_i, 1.0)
    beta_j = betas.get(ticker_j, 1.0)
    beta_sim = max(0, 1 - abs(beta_i - beta_j))
    
    # 8. Partial correlation (remove SPY)
    partial_corr = compute_partial_correlation(
        returns_df[ticker_i], returns_df[ticker_j], spy_returns
    )
    
    return np.array([corr_30d, sector_sim, jaccard, index_co, 
                     mcap_ratio, vol_corr, beta_sim, partial_corr])
```

---

## 7. Build Steps

### Step 1: Load Universe

**Inputs:**
- `data/sec_edgar/processed/cleaned/cik_ticker_map_cleaned.csv` → 4,428 tickers
- Market data (yfinance output) → daily OHLCV

**Actions:**
1. Load ticker list
2. Load market data for all tickers
3. Verify alignment to 6,288/6286 NYSE trading days
4. Filter to tickers present in market data

**Output:** Validated ticker universe (N tickers, may be less than 4,428)

---

### Step 2: Fetch Static Metadata

**Inputs:** Ticker list

**Actions:**
1. Fetch sector/industry via yfinance (one-time API call per ticker):
   ```python
   stock = yf.Ticker(ticker)
   sector = stock.info.get('sector', 'Unknown')
   industry = stock.info.get('industry', 'Unknown')
   market_cap = stock.info.get('marketCap', 0)
   ```
2. Cache results to avoid re-downloading
3. Fetch ETF holdings (yfinance or SEC N-PORT)
4. Fetch index constituents (Wikipedia/yfinance)

**Outputs:**
- `sector_map.csv`: ticker → sector mapping
- `market_caps.csv`: ticker → market cap
- `etf_holdings.csv`: etf → list of tickers
- `index_membership.csv`: index → list of tickers

---

### Step 3: Compute Returns and Features

**Inputs:** Market data (daily prices)

**Actions:**
1. Compute daily log returns: `returns = log(close_t / close_t-1)`
2. Compute daily log volumes
3. Compute beta vs SPY (252-day rolling)
4. Store in standardized format

**Outputs:**
- `returns_matrix.csv`: N_tickers × 6,288 days
- `volumes_matrix.csv`: N_tickers × 6,288 days
- `betas.csv`: ticker → beta

---

### Step 4: Build Sector Similarity Matrix

**Inputs:** Returns matrix, sector_map

**Actions:**
1. Aggregate returns by sector (equal-weighted average)
2. Compute 252-day rolling correlation of sector returns
3. Normalize to 0-1 similarity
4. Store as static matrix

**Output:** `sector_similarity.csv`: 11×11 matrix (or fewer sectors)

---

### Step 5: Build Correlation Snapshots

**Inputs:** Returns matrix

**Actions:**
1. For each 20-trading-day window (stride=20):
   a. Compute 30-day correlation matrix
   b. For each ticker, find top-K=66 edges
   c. Save edge list with window date
2. Total snapshots: ~314 (6288 / 20)

**Output:** `correlation_graphs/` directory with:
- `edges_YYYY-MM-DD.csv`: edge list for each window

---

### Step 6: Build Static Graphs

**Inputs:** Ticker list, sector_map, etf_holdings, index_membership

**Actions:**
1. Build sector hierarchy graph (tickers → sectors, sectors → sectors)
2. Build ETF membership graph (tickers → ETFs)
3. Build index membership graph (tickers → indices)
4. Merge all into unified static graph

**Outputs:**
- `static_graph_nodes.csv`: All nodes with features
- `static_graph_edges.csv`: All edges with types and weights

---

### Step 7: Combine into Final Graph Objects

**Inputs:** Correlation snapshots, static graphs, relationship vectors

**Actions:**
1. For each correlation snapshot:
   a. Merge with static graph edges
   b. Compute 8-dim relationship vector for each edge
   c. Build PyTorch Geometric Data object
   d. Save as .pt file
2. Also save in NetworkX format for analysis

**Outputs:**
- `graphs/correlation_snapshots/`: `.pt` files
- `graphs/static/`: Static graph `.pt` file
- `graphs/combined/`: Combined graphs per window

---

## 8. Output Files

### Directory Structure

```
data/graphs/
├── metadata/
│   ├── ticker_universe.csv          # Final ticker list used
│   ├── sector_map.csv               # Ticker → sector mapping
│   ├── market_caps.csv              # Ticker → market cap
│   ├── betas.csv                    # Ticker → beta
│   └── sector_similarity.csv        # Sector × sector similarity matrix
├── static/
│   ├── nodes.csv                    # All nodes (tickers + ETFs + indices + sectors)
│   ├── edges.csv                    # Static edges (sector, ETF, index)
│   └── static_graph.pt              # PyTorch Geometric Data object
├── returns/
│   ├── returns_matrix.csv           # N × T returns matrix
│   └── volumes_matrix.csv           # N × T volumes matrix
├── correlation_snapshots/
│   ├── edges_2000-01-24.csv         # Top-K edges for each window
│   ├── edges_2000-02-22.csv
│   └── ...                          # ~314 snapshots
└── combined/
│   ├── graph_2000-01-24.pt          # PyTorch Geometric Data (full)
│   ├── graph_2000-02-22.pt
│   └── ...                          # ~314 snapshots
└── CrossAssetRelationData.md   # Current file
```

### Final Deliverable Files

| File | Format | Rows | Description |
|------|--------|------|-------------|
| `ticker_universe.csv` | CSV | 4,428 | Final ticker list |
| `sector_map.csv` | CSV | 4,428 | Sector per ticker |
| `sector_similarity.csv` | CSV | 11×11 | Sector similarity matrix |
| `returns_matrix.csv` | CSV | 4,428×6,288 | Daily returns |
| `static_graph.pt` | PyTorch | - | Static graph object |
| `graph_YYYY-MM-DD.pt` | PyTorch | 314 files | Combined graph per window |

---

## 9. File Structure (Project-Wide)

```
fin-glassbox/
├── code/
│   ├── gnn/
│   │   ├── build_cross_asset_graph.py    # ← THE MAIN SCRIPT
│   │   ├── graph_utils.py                # Graph utilities
│   │   ├── stemgnn_contagion.py           # Contagion module
│   │   └── mtgnn_regime.py                # Regime module
├── data/
│   ├── graphs/                            # ← ALL OUTPUTS GO HERE
│   │   ├── metadata/
│   │   ├── static/
│   │   ├── returns/
│   │   ├── correlation_snapshots/
│   │   └── combined/
│   ├── yFinance/                          # Market data (input)
│   │   └── ...                            # OHLCV files
│   └── sec_edgar/processed/cleaned/
│       └── cik_ticker_map_cleaned.csv     # Ticker universe (input)
```

---

## 10. Complete Build Script Specification

### Script: `code/gnn/build_cross_asset_graph.py`

```python
#!/usr/bin/env python3
"""
Cross-Asset Graph Builder

Builds all cross-asset relationship graphs from market data.
Produces static graphs, correlation snapshots, and combined PyTorch Geometric objects.

Usage:
    python code/gnn/build_cross_asset_graph.py
    python code/gnn/build_cross_asset_graph.py --workers 8
    python code/gnn/build_cross_asset_graph.py --skip-correlations  # Static only
"""

# ============================================================
# CONFIGURATION
# ============================================================

# Graph parameters
K_EDGES_PER_NODE = 66        # Top-K correlations per node (√4428 ≈ 66)
CORRELATION_WINDOW = 30      # Days for rolling correlation
SNAPSHOT_STRIDE = 20         # Trading days between snapshots
BETA_WINDOW = 252            # Days for beta computation

# ETF Universe (for membership graph)
ETFS = [
    'SPY', 'QQQ', 'IWM', 'DIA',
    'XLK', 'XLF', 'XLV', 'XLE', 'XLI',
    'XLP', 'XLY', 'XLU', 'XLB', 'XLRE', 'XLC'
]

# Indices (for membership graph)
INDICES = {
    'SP500': 'S&P 500',
    'NASDAQ100': 'Nasdaq-100',
    'RUSSELL2000': 'Russell 2000',
    'DOW30': 'Dow Jones Industrial'
}

# ============================================================
# SCRIPT FLOW
# ============================================================

"""
Step 1: Load Universe
    - Read cik_ticker_map_cleaned.csv → ticker list
    - Load market data from data/yFinance/
    - Verify alignment to NYSE trading days
    - Filter to tickers present in market data

Step 2: Fetch Static Metadata (with caching)
    - Fetch sector/industry via yfinance
    - Fetch market caps
    - Fetch ETF holdings
    - Fetch index constituents
    - Save all to data/graphs/metadata/

Step 3: Compute Returns and Features
    - Compute daily log returns
    - Compute daily log volumes
    - Compute beta vs SPY (252-day rolling)
    - Save to data/graphs/returns/

Step 4: Build Sector Similarity Matrix
    - Aggregate returns by sector
    - Compute sector return correlations
    - Normalize to 0-1
    - Save to data/graphs/metadata/sector_similarity.csv

Step 5: Build Correlation Snapshots
    - For each 20-day window:
        - Compute 30-day correlation matrix
        - Find top-K=66 edges per node
        - Save edge list
    - Save to data/graphs/correlation_snapshots/

Step 6: Build Static Graphs
    - Sector hierarchy: tickers → sectors, sectors → sectors
    - ETF membership: tickers → ETFs
    - Index membership: tickers → indices
    - Merge into unified static graph
    - Save to data/graphs/static/

Step 7: Build Combined Graph Objects
    - For each correlation snapshot:
        - Merge with static graph edges
        - Compute 8-dim relationship vector per edge
        - Build PyTorch Geometric Data object
        - Save as .pt file
    - Save to data/graphs/combined/
"""
```

### Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| **K edges per node** | 66 (√4428) | Balance sparsity and coverage |
| **Correlation window** | 30 days | Monthly patterns, responsive |
| **Snapshot stride** | 20 days | ~314 snapshots, manageable |
| **Relationship vector** | 8 dimensions | Start simple, expandable |
| **Graph format** | PyTorch Geometric `.pt` | Directly loadable by GNNs |
| **Static graphs** | Built once, cached | Sector/ETF/index change slowly |
| **Dynamic graphs** | Built per window | Correlations change with market |

---

## Summary

| Aspect | Detail |
|--------|--------|
| **Script** | `code/gnn/build_cross_asset_graph.py` |
| **Input** | Market data (yfinance) + ticker list |
| **Output** | Static graph + ~314 correlation snapshots (PyTorch Geometric) |
| **Feeds** | StemGNN Contagion Module + MTGNN Regime Module |
| **Relationship vector** | 8-dim: correlation, sector_sim, etf_overlap, index_co, mcap_ratio, vol_corr, beta_sim, partial_corr |
| **K edges per node** | 66 (√4428) |
| **Snapshot frequency** | Every 20 trading days (~314 total) |

