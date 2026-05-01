# Cross-Asset Relation Data Builder 

**Project:** An Explainable Distributed Deep Learning Framework for Financial Risk Management  
**Module family:** Cross-Asset Relation Data  
**Primary file:** `code/gnn/build_cross_asset_graph.py`  
**Primary output root:** `data/graphs/`  
**Status:** Implemented and used by downstream GNN/regime components  

---

## 1. Purpose of this document

This document is the final documentation for the **Cross-Asset Relation Data Builder**. It explains how the cross-asset graph data is constructed, why it exists, what files it produces, what assumptions it makes, and how the resulting graph artefacts are consumed by the rest of the fin-glassbox architecture.

The Cross-Asset Relation Data family is one of the five core data families in the project. Unlike the market, text, macro, and fundamentals data families, this family is **relationship-centric** rather than asset-centric. Its core purpose is to represent how assets are connected to each other through market co-movement, sector membership, ETF similarity, beta exposure, and dynamic correlation structure.

In the broader system, this data supports graph-aware risk modelling, especially:

- the **StemGNN Contagion Risk Module**, which learns dynamic spillover risk directly from market return windows;
- the **MTGNN graph-building component** used inside the regime risk module;
- graph-stress and market-connectivity features for regime detection;
- auditability of the relational market structure used by the risk engine.

---

## 2. Why cross-asset relation data is needed

Financial assets do not behave independently. A single stock may move because of its own fundamentals, but it may also move because of:

- sector-wide shocks;
- benchmark/index movement;
- ETF basket flows;
- shared macro exposures;
- liquidity stress;
- correlation spikes during crisis regimes;
- contagion from influential or highly connected assets.

A standard table of ticker-level features cannot directly represent these relationships. A graph representation makes the relationships explicit:

```text
nodes = stocks, ETFs, sectors
edges = relationships between those nodes
edge weights = relationship strength
snapshots = relationship structure over time
```

This is important for a financial risk system because risk is not only a property of a single asset. In market stress, diversification can fail because correlations increase and losses propagate across related assets. The graph data family is therefore the structural foundation for modelling **contagion**, **systemic exposure**, and **regime-level connectivity**.

---

## 3. Final implementation summary

The implemented builder is:

```text
code/gnn/build_cross_asset_graph.py
```

It reads market-derived returns, OHLCV data, and SEC issuer metadata, then writes all graph artefacts under:

```text
data/graphs/
├── metadata/
├── returns/
├── static/
├── snapshots/
└── combined/
```

The current implemented graph builder constructs:

1. **Ticker universe** from the return matrix columns.
2. **Sector mapping** from SEC issuer SIC descriptions mapped into GICS-like sectors.
3. **Market-cap proxy** using median dollar volume.
4. **Beta estimates** against SPY where SPY exists in the return matrix.
5. **Sector similarity matrix** based on equal-weighted sector return correlations.
6. **Static graph edges** for ticker-sector membership, ETF similarity, and sector similarity.
7. **Dynamic correlation snapshots** using rolling 30-trading-day windows with 20-trading-day stride.
8. **Sample NetworkX combined graph** containing the latest dynamic correlation graph plus static edges.

The implementation is intentionally practical: it uses the already-cleaned market panel instead of requiring a new external graph dataset. This makes the graph family reproducible and aligned with the exact ticker universe used in the rest of the project.

---

## 4. Input files

### 4.1 Market return matrix

```text
data/yFinance/processed/returns_panel_wide.csv
```

Expected format:

```text
date,A,AA,AAL,...
2000-01-04,...
2000-01-05,...
...
```

The script loads this file with `date` as the index and tickers as columns. This file defines the active universe for graph construction.

### 4.2 Final OHLCV panel

```text
data/yFinance/processed/ohlcv_final.csv
```

Required columns:

```text
date,ticker,open,high,low,close,volume
```

This file is used to compute the market-cap proxy:

```text
dollar_volume = close × volume
market_cap_proxy = rolling median dollar_volume over 252 trading days
```

This is not a true market capitalisation. It is a liquidity/size proxy suitable for graph node metadata.

### 4.3 Cleaned SEC issuer metadata

```text
data/sec_edgar/processed/cleaned/issuer_master_cleaned.csv
```

Required columns used by the builder:

```text
primary_ticker
sic_description
```

The script maps SEC SIC descriptions to GICS-style sectors using the internal `SIC_TO_GICS` dictionary. Any missing or unmapped ticker is assigned to `Other`.

---

## 5. Core construction parameters

The implemented script uses the following graph construction constants:

| Parameter | Value | Purpose |
|---|---:|---|
| `K_EDGES` | `50` | Top correlated neighbours retained per node in each dynamic snapshot |
| `CORR_WINDOW` | `30` trading days | Rolling return window used for correlation snapshots |
| `SNAP_STRIDE` | `20` trading days | Distance between consecutive graph snapshots |
| `BETA_WINDOW` | `252` trading days | Window for beta estimation against SPY |
| `MCAP_WINDOW` | `252` trading days | Window for median dollar-volume market-cap proxy |
| `ETF_TICKERS` | `SPY`, `QQQ`, `DIA` | ETF nodes used for ETF-similarity static edges |

The `K_EDGES = 50` design is consistent with the current 2,500 ticker universe because `sqrt(2500) = 50`. This keeps each snapshot sparse enough to store and process while preserving local dependency structure.

---

## 6. Step-by-step pipeline

### Step 1 — Load returns matrix

The script loads:

```text
data/yFinance/processed/returns_panel_wide.csv
```

The columns become `tickers_all`, and the index becomes `dates_all`. These two arrays define the graph node universe and the snapshot timeline.

The script expects a fully cleaned market panel. Missing and inconsistent ticker handling should already have been completed upstream in the market data pipeline.

### Step 2 — Build metadata

#### 2A. Sector mapping

The script reads `issuer_master_cleaned.csv`, filters it to tickers present in the return matrix, and maps SEC SIC descriptions into GICS-like sectors.

Output:

```text
data/graphs/metadata/sector_map.csv
```

Format:

```text
ticker,gics_sector
A,Health Care
MSFT,Information Technology
...
```

Possible sector values include:

```text
Information Technology
Financials
Health Care
Energy
Industrials
Consumer Discretionary
Consumer Staples
Utilities
Communication Services
Materials
Real Estate
Other
```

`Other` is used for missing or unmapped SIC descriptions.

#### 2B. Market-cap proxy

The script computes:

```text
dollar_vol = close × volume
market_cap_proxy = rolling_median(dollar_vol, 252 trading days).last_value
```

Output:

```text
data/graphs/metadata/market_cap_proxy.csv
```

This proxy is not a true market cap. It is a size/liquidity proxy designed to provide useful graph node metadata without requiring paid market-cap data.

#### 2C. Beta versus SPY

If `SPY` exists in the return matrix, each ticker receives a rolling beta estimate:

```text
beta_i = Cov(return_i, return_SPY) / Var(return_SPY)
```

using a 252-day rolling window. If SPY is unavailable or insufficient history exists, beta defaults to `1.0`.

Output:

```text
data/graphs/metadata/betas.csv
```

#### 2D. Ticker universe

Output:

```text
data/graphs/metadata/ticker_universe.csv
```

---

### Step 3 — Save graph-specific returns matrix copy

The script writes a graph-module copy of the return matrix:

```text
data/graphs/returns/returns_matrix.csv
```

This makes the graph data family self-contained and auditable. It also prevents downstream graph modules from needing to know the original market data path.

---

### Step 4 — Build sector similarity matrix

For each sector, the script computes the equal-weighted mean return across all tickers assigned to that sector. It then computes the sector-to-sector correlation matrix and normalises it:

```text
sector_similarity = (sector_correlation + 1) / 2
```

Output:

```text
data/graphs/metadata/sector_similarity.csv
```

The 0–1 normalisation makes the sector similarity suitable as a graph edge weight.

---

### Step 5 — Build static graph edges

Static graph edges represent relationships that do not change every 20 trading days.

Output:

```text
data/graphs/static/edges.csv
```

Columns:

```text
source,target,edge_type,weight
```

The script creates three static edge groups.

#### 5A. Sector membership edges

Each ticker is connected to its sector node:

```text
source = ticker
target = SECTOR_<sector name>
edge_type = sector_membership
weight = 1.0
```

#### 5B. ETF similarity edges

For each ETF in `ETF_TICKERS` that exists in the return matrix, the script computes the positive return correlation between each ticker and the ETF. If the correlation is above `0.1`, an edge is retained:

```text
source = ticker
target = ETF_SPY / ETF_QQQ / ETF_DIA
edge_type = etf_similarity
weight = positive correlation
```

This approximates ETF/index-like exposure without requiring holdings files.

#### 5C. Sector similarity edges

Sectors are connected when their normalised similarity is above `0.3`:

```text
source = SECTOR_<sector 1>
target = SECTOR_<sector 2>
edge_type = sector_similarity
weight = sector_similarity value
```

---

### Step 6 — Build dynamic correlation snapshots

Dynamic snapshots are the core time-varying graph outputs.

For each 30-trading-day window, with a stride of 20 trading days, the script:

1. selects the current return window;
2. computes the correlation matrix among valid tickers;
3. for each ticker, finds the top `K_EDGES = 50` correlated neighbours;
4. keeps only positive correlations above `0.1`;
5. writes an edge-list file labelled by the snapshot start date.

Output folder:

```text
data/graphs/snapshots/
```

File pattern:

```text
data/graphs/snapshots/edges_YYYY-MM-DD.csv
```

Columns:

```text
window_start,source,target,correlation
```

Example:

```text
2000-01-04,A,HAS,0.681966
2000-01-04,A,ARE,0.655350
2000-01-04,A,CIEN,0.627203
```

The current completed graph dataset contains approximately:

```text
313 correlation snapshots
first snapshot: 2000-01-04
last snapshot: 2024-10-23
sample snapshot rows: about 82,200 edges
```

---

### Step 7 — Build sample combined NetworkX graph

If `networkx` is installed, the script builds a sample combined graph from the latest snapshot and the static graph edges.

Output:

```text
data/graphs/combined/sample_graph.pkl
```

The graph includes ticker nodes, ETF nodes, sector nodes, dynamic correlation edges, static edges, and node metadata such as sector, market-cap proxy, and beta.

The observed final combined graph contains approximately:

```text
2,515 graph nodes
105,545 sample graph edges
```

The exact number can vary depending on the ticker universe, available ETFs, and whether the `Other` sector node is present.

---

## 7. Output file structure

```text
data/graphs/
├── metadata/
│   ├── ticker_universe.csv
│   ├── sector_map.csv
│   ├── market_cap_proxy.csv
│   ├── betas.csv
│   └── sector_similarity.csv
├── returns/
│   └── returns_matrix.csv
├── static/
│   ├── edges.csv
│   └── nodes.csv
├── snapshots/
│   ├── edges_2000-01-04.csv
│   ├── edges_2000-02-02.csv
│   ├── ...
│   └── edges_2024-10-23.csv
└── combined/
    └── sample_graph.pkl
```

---

## 8. Output file specifications

| File | Key columns | Purpose |
|---|---|---|
| `metadata/ticker_universe.csv` | `ticker` | Active graph ticker universe |
| `metadata/sector_map.csv` | `ticker`, `gics_sector` | SIC-derived sector mapping |
| `metadata/market_cap_proxy.csv` | `ticker`, `market_cap_proxy` | Size/liquidity proxy |
| `metadata/betas.csv` | `ticker`, `beta` | Rolling beta against SPY |
| `metadata/sector_similarity.csv` | sector matrix | Normalised sector similarity |
| `static/edges.csv` | `source`, `target`, `edge_type`, `weight` | Static relation edges |
| `static/nodes.csv` | `node_id` | Static graph nodes |
| `snapshots/edges_YYYY-MM-DD.csv` | `window_start`, `source`, `target`, `correlation` | Dynamic rolling correlation graph |
| `combined/sample_graph.pkl` | NetworkX graph object | Latest combined sample graph |

---

## 9. Relationship to downstream modules

### 9.1 StemGNN Contagion

StemGNN does not require the static edge files as direct input in the current implementation. The StemGNN module learns its own latent adjacency matrix from returns through the StemGNN correlation layer. However, the cross-asset graph family remains conceptually important because it defines the relation-data family and provides graph artefacts for audit, comparison, and regime features.

```text
StemGNN learns dynamic relationships internally from returns.
Cross-asset graph snapshots provide explicit relation snapshots and audit artefacts.
```

### 9.2 MTGNN Regime Graph Builder

The MTGNN regime module uses the dynamic snapshot files under:

```text
data/graphs/snapshots/
```

It reads the snapshot edge lists, computes graph features such as density and correlation strength, and combines these with learned MTGNN-style graph properties, Temporal Encoder embeddings, FinBERT context, and FRED macro data.

### 9.3 Position Sizing Engine

The Position Sizing Engine does not read cross-asset graph files directly. It consumes graph-aware risk outputs, especially:

```text
StemGNN contagion risk
MTGNN regime risk
```

The graph data affects position sizing through those risk outputs.

---

## 10. Data leakage and point-in-time considerations

The graph builder uses market return history and rolling windows. To avoid look-ahead bias:

- correlation snapshots should be used only when their full 30-day window is available;
- downstream splits must remain chronological;
- graph snapshots should not be randomly shuffled across train/validation/test boundaries;
- metadata such as sector mapping and market-cap proxy should be documented as practical free-data approximations;
- a production-grade system would need true point-in-time sector membership, ETF membership, and market-cap histories.

---

## 11. Build commands

### Metadata-only dry run

```bash
python code/gnn/build_cross_asset_graph.py --workers 4 --skip-snapshots
```

### Full graph build

```bash
python code/gnn/build_cross_asset_graph.py --workers 4
```

### Verify output files

```bash
find data/graphs -maxdepth 3 -type f | sort | head -100
```

### Count snapshots

```bash
python -c "from pathlib import Path; files=sorted(Path('data/graphs/snapshots').glob('edges_*.csv')); print(len(files)); print(files[0] if files else None); print(files[-1] if files else None)"
```

### Inspect one snapshot

```bash
python -c "import pandas as pd; from pathlib import Path; f=sorted(Path('data/graphs/snapshots').glob('edges_*.csv'))[0]; df=pd.read_csv(f); print(f); print(df.head().to_string(index=False)); print('rows=', len(df)); print('cols=', list(df.columns))"
```

---

## 12. Validation command

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path

base = Path('data/graphs')
required = [
    'metadata/ticker_universe.csv',
    'metadata/sector_map.csv',
    'metadata/market_cap_proxy.csv',
    'metadata/betas.csv',
    'metadata/sector_similarity.csv',
    'returns/returns_matrix.csv',
    'static/edges.csv',
    'static/nodes.csv',
]

for rel in required:
    p = base / rel
    print(('OK      ' if p.exists() else 'MISSING ') + str(p))
    if p.exists() and p.suffix == '.csv':
        df = pd.read_csv(p, nrows=5)
        print('        columns:', list(df.columns))

snaps = sorted((base / 'snapshots').glob('edges_*.csv'))
print('snapshots:', len(snaps))
if snaps:
    df = pd.read_csv(snaps[0])
    print('first snapshot:', snaps[0], 'rows=', len(df), 'columns=', list(df.columns))
PY
```

---

## 13. Known limitations

The current graph builder is strong enough for the academic system, but it has explicit limitations:

1. ETF similarity is correlation-based, not true holdings-based.
2. Market-cap proxy is dollar-volume based, not true market capitalisation.
3. Sector mapping is SIC-derived, not official point-in-time GICS membership.
4. Static sector edges are not time-varying.
5. Snapshot correlations keep positive top-K relationships and therefore focus on co-movement rather than negative hedging relationships.
6. Graph snapshots are structural inputs/features, not prediction labels.

These are acceptable trade-offs for a free-data, large-scale research framework, but they should be acknowledged in the methodology.

---

## 14. Final role in the architecture

```text
cleaned market data + issuer metadata
        ↓
cross-asset graph construction
        ↓
static graph metadata + dynamic correlation snapshots
        ↓
GNN contagion/regime modules
        ↓
risk engine + position sizing + fusion
```

The Cross-Asset Relation Data Builder is therefore a data-engineering component, not a predictive model. It makes the relational market structure explicit and reusable.
