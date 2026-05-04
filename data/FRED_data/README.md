# FRED Data 

## The Two GNNs and Their Inputs

### GNN 1: StemGNN (Contagion Module)
- **Input:** Cross-asset graph built from **market returns only**
- **FRED data?** ❌ NOT used here
- **Why?** Contagion is about how stocks affect each other. Macro data doesn't help answer "if AAPL crashes, does MSFT crash too?"

### GNN 2: MTGNN (Regime Detection Module)
- **Input 1:** Temporal embeddings from the Shared Temporal Attention Encoder (market price patterns)
- **Input 2:** Text embeddings from FinBERT (sentiment from SEC filings)
- **FRED data?** ✅ THIS is where it enters

---

## Where FRED Data Actually Flows

Here is the exact data flow:

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRED MACRO DATA                              │
│  macro_features_trading_days_clean.csv                          │
│  6,288 trading days × 49 features                               │
│  (VIX z-scores, yield spreads, regime flags, rates, etc.)       │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │  NOT fed directly to the GNN
                            │  NOT concatenated with stock features
                            │
                            v
┌─────────────────────────────────────────────────────────────────┐
│              REGIME DETECTION MODULE (MTGNN)                    │
│                                                                 │
│  Step 1: Build graph from temporal + text embeddings            │
│          (This captures how stocks relate to each other)        │
│                                                                 │
│  Step 2: Extract graph properties:                              │
│          - Graph density                                        │
│          - Modularity                                           │
│          - Average degree                                       │
│          - Clustering coefficient                               │
│                                                                 │
│  Step 3: FEED GRAPH PROPERTIES + FRED FEATURES → CLASSIFIER     │
│                                                                 │
│          ┌──────────────────────────────────────────┐           │
│          │  MLP Classifier                          │           │
│          │                                          │           │
│          │  Input: [graph_density,                  │           │
│          │          modularity,                      │           │
│          │          avg_degree,                      │           │
│          │          clustering_coef,                 │           │
│          │          yield_spread_10y2y,  ← FRED!    │           │
│          │          vix_z_20d,          ← FRED!    │           │
│          │          regime_yield_inverted, ← FRED!  │           │
│          │          ted_z_60d,          ← FRED!    │           │
│          │          credit_spread,      ← FRED!    │           │
│          │          ...]                            │           │
│          │                                          │           │
│          │  Output: [calm/volatile/crisis/rotation] │           │
│          └──────────────────────────────────────────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## In Plain English

The Regime Detection Module works in TWO stages:

### Stage 1: Graph Building (No FRED Data)
The MTGNN graph builder takes **per-stock features** (temporal embeddings + FinBERT text embeddings) and learns how stocks are connected to each other. This produces a graph.

**FRED data is NOT per-stock.** Yield spreads, VIX, and recession indicators are the same for ALL stocks on a given day. You can't use them to build per-stock connections.

### Stage 2: Regime Classification (FRED Data Enters HERE)
Once the graph is built, we extract **graph-level properties**:
- How dense is the graph? (Are all stocks moving together?)
- How modular is it? (Are there distinct sector clusters?)
- How connected is the average stock?

These graph properties are **market-level**, not per-stock. NOW we can combine them with FRED macro features (which are also market-level):

```python
# THIS is where FRED data enters
classifier_input = [
    graph_density,           # From MTGNN graph
    modularity,              # From MTGNN graph
    avg_degree,              # From MTGNN graph
    yield_spread_10y2y,      # ← FROM FRED
    vix_z_20d,               # ← FROM FRED
    regime_yield_inverted,   # ← FROM FRED
    credit_spread_baa_aaa,   # ← FROM FRED
    ted_z_60d,               # ← FROM FRED
]

regime = mlp_classifier(classifier_input)
# → "calm", "volatile", "crisis", or "rotation"
```

---

## Why FRED Data Matters for Regime Detection

The graph properties alone can be misleading:

| Scenario | Graph Density | FRED Context | True Regime |
|----------|--------------|--------------|-------------|
| Earnings season, sector rotation | Medium | VIX low, yield curve normal | **Rotation** (normal) |
| Market-wide selloff | High | VIX spiking, yield curve inverting | **Crisis** (panic) |
| Holiday week, low volume | Low | VIX low, no credit stress | **Calm** (just quiet) |

**Without FRED data**, the classifier might see "high density" and scream "CRISIS!" when it's actually just earnings season.

**With FRED data**, the classifier can distinguish:
- High density + high VIX + inverted yield curve = REAL CRISIS
- High density + low VIX + normal yield curve = EARNINGS SEASON (normal)

---

## Summary: Where FRED Data Goes

| Module | Uses FRED Data? | How? |
|--------|----------------|------|
| **Temporal Encoder** | ❌ No | Only market price data |
| **FinBERT** | ❌ No | Only SEC text |
| **Fundamental Encoder** | ❌ No | Only SEC fundamentals |
| **Technical Analyst** | ❌ No | Consumes temporal embeddings |
| **Sentiment Analyst** | ❌ No | Consumes text embeddings |
| **News Analyst** | ❌ No | Consumes text embeddings |
| **Fundamental Analyst** | ❌ No | Consumes fundamental embeddings |
| **Volatility Model** | ❌ No | Consumes temporal embeddings |
| **Drawdown Model** | ❌ No | Consumes temporal embeddings |
| **VaR/CVaR** | ❌ No | Statistical from returns |
| **StemGNN Contagion** | ❌ No | Market returns + graph only |
| **MTGNN Regime** | ✅ YES | **Graph properties + FRED features → Classifier** |
| **Liquidity** | ❌ No | Volume data only |
| **Position Sizing** | ⚠️ Indirectly | Receives regime label (which used FRED) |
| **Fusion** | ⚠️ Indirectly | Receives all scores including regime |

---

## The Key Insight

**FRED data is MARKET-LEVEL, not STOCK-LEVEL.**

This is why it cannot be used to build per-stock embeddings or per-stock graphs. It can only help classify the **overall market state** after the stock-level graph has been built.

The Regime Detection Module's classifier is the **single point** where macro data enters the neural pipeline. Everything else in the system either:
- Works at the stock level (encoders, analysts, risk models)
- Receives the regime output indirectly (position sizing, fusion)