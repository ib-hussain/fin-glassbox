# Downstream Modules — Implementation Plan

## Current Progress Summary

| Module | Code | HPO | Trained | Embeddings | Status |
|--------|------|-----|---------|------------|--------|
| **Temporal Encoder** | ✅ | ✅ Ch1-3 | ✅ Ch1-2, ⏳ Ch3 | ✅ Ch1 | Training Ch3 |
| **FinBERT** | ✅ | ✅ | ✅ Ch1-3 | ✅ Ch1-3 | Complete |
| **Sentiment Analyst** | ✅ | ✅ | ✅ Ch1-3 | N/A | Complete |
| **News Analyst** | ✅ | ✅ | ✅ Ch1-3 | N/A | Complete |
| **StemGNN Contagion** | ✅ | ✅ Ch1, ⏳ Ch2-3 | ✅ Ch1 | N/A | HPO Ch2 running |
| **Volatility Model** | ✅ | ⏳ Ch1 | ❌ | N/A | Debugging NaN |
| **VaR/CVaR/Liquidity** | ✅ | N/A | N/A | N/A | Ready to run |
| **Technical Analyst** | ❌ | ❌ | ❌ | N/A | **NEXT** |
| **Drawdown Model** | ❌ | ❌ | ❌ | N/A | Pending |
| **Regime Detection** | ❌ | ❌ | ❌ | N/A | Pending |
| **Position Sizing** | ❌ | ❌ | ❌ | N/A | After risk modules |
| **Fusion Engine** | ❌ | ❌ | ❌ | N/A | After all modules |

---

## Available Data for Downstream Modules

```
outputs/embeddings/TemporalEncoder/
├── chunk1_train_embeddings.npy    (3,065,000 × 256)
├── chunk1_train_manifest.csv      (3,065,000 × 2: ticker, date)
├── chunk1_val_embeddings.npy      (555,000 × 256)
├── chunk1_val_manifest.csv        (555,000 × 2)
├── chunk1_test_embeddings.npy     (552,500 × 256)
├── chunk1_test_manifest.csv       (552,500 × 2)
├── chunk2_*_embeddings.npy        (pending)
└── chunk3_*_embeddings.npy        (pending)

data/yFinance/processed/
├── returns_panel_wide.csv         (6,285 days × 2,500 tickers)
├── returns_long.csv               (ticker, date, log_return)
├── ohlcv_final.csv                (open, high, low, close, volume per ticker per day)
└── features_temporal.csv          (10 engineered features per ticker per day)
```

---

## Module 1: Technical Analyst (BiLSTM)

### Specification

| Aspect | Detail |
|--------|--------|
| **Model** | BiLSTM (1 layer, hidden=64, bidirectional) with Attention Pooling |
| **Input** | 256-dim temporal embeddings (from Temporal Encoder) |
| **Input Shape** | `(batch, seq_len=30, 256)` — sequence of 30 daily embeddings |
| **Output** | `trend_score` (0-1), `momentum_score` (0-1), `timing_confidence` (0-1) |
| **Architecture** | BiLSTM(256→64) → Attention Pooling → Linear(64→3) → Sigmoid |
| **Training** | Supervised — targets derived from future returns |
| **HPO** | TPE, 30-50 trials |

### Target Construction (Labels)

Targets are computed from forward returns using rule-based scoring:

```python
# For each embedding at time t (which encodes days [t-29, t]):
# Trend: sign of 20-day forward return
#   - If 20d forward return > 0.5%: trend=1.0
#   - If 20d forward return < -0.5%: trend=0.0
#   - Else: trend=0.5

# Momentum: strength of 5-day forward return relative to volatility
#   - momentum = |5d_forward_return| / (5d_volatility + eps)
#   - Clamp to [0, 1]

# Timing: based on RSI and MACD divergence
#   - If RSI < 30 and price > 20d MA: timing=1.0 (oversold bounce)
#   - If RSI > 70 and price < 20d MA: timing=0.0 (overbought)
#   - Else: timing=0.5 + 0.5 * MACD_histogram_normalized
```

### XAI Requirements

| Level | Method | Output |
|-------|--------|--------|
| L1 | Attention weights | Which timesteps mattered most |
| L2 | Gradient feature importance | Which embedding dimensions drove scores |
| L3 | Counterfactuals | What would change trend/momentum call |

### CLI Commands

```bash
python code/analysts/technical_analyst.py inspect
python code/analysts/technical_analyst.py hpo --chunk 1 --trials 40 --device cuda
python code/analysts/technical_analyst.py train-best --chunk 1 --device cuda
python code/analysts/technical_analyst.py train-best-all --device cuda
python code/analysts/technical_analyst.py predict --chunk 1 --split test --device cuda
```

### Data Dependencies

- `outputs/embeddings/TemporalEncoder/chunk*_*_embeddings.npy` — 256-dim temporal embeddings
- `outputs/embeddings/TemporalEncoder/chunk*_*_manifest.csv` — ticker/date per embedding row
- `data/yFinance/processed/returns_panel_wide.csv` — for target computation
- `data/yFinance/processed/features_temporal.csv` — for RSI/MACD data (or compute from returns)

---

## Module 2: Drawdown Risk Model (BiLSTM Dual Horizon)

### Specification

| Aspect | Detail |
|--------|--------|
| **Model** | BiLSTM (1 layer, hidden=64) with Dual Horizon Heads |
| **Input** | 256-dim temporal embeddings (sequence of 30 days) |
| **Input Shape** | `(batch, seq_len=30, 256)` |
| **Output** | 10d: `expected_drawdown_pct`, `drawdown_probability`, `recovery_days_estimate` |
| | 30d: same three outputs |
| **Architecture** | BiLSTM(256→64) → Shared Features → Head_10d(64→3) + Head_30d(64→3) |
| **Training** | Supervised — targets are future maximum drawdowns |
| **HPO** | TPE, 30-50 trials |

### Target Construction (Labels)

```python
# For each embedding at time t:
# Compute the maximum drawdown over the next h days:
#   peak = running_max(price[t : t+h])
#   drawdown = (peak - price) / peak  # at each step
#   max_drawdown_h = max(drawdown over horizon)

# 10-day target:
#   expected_drawdown_10d = max_drawdown over [t, t+10]
#   drawdown_probability_10d = 1 if max_drawdown_10d > 5% else 0 (or regression)
#   recovery_days_10d = days until price recovers to pre-drawdown level

# 30-day target:
#   Same, over [t, t+30]
```

Drawdown data is computed from `ohlcv_final.csv` close prices.

### XAI Requirements

| Level | Method | Output |
|-------|--------|--------|
| L1 | Attention weights | Which timesteps warned of drawdown |
| L2 | Gradient feature importance | Key embedding dimensions |
| L3 | Counterfactuals | What conditions would reduce drawdown risk |

### CLI Commands

```bash
python code/riskEngine/drawdown.py inspect
python code/riskEngine/drawdown.py hpo --chunk 1 --trials 40 --device cuda
python code/riskEngine/drawdown.py train-best --chunk 1 --device cuda
python code/riskEngine/drawdown.py train-best-all --device cuda
python code/riskEngine/drawdown.py predict --chunk 1 --split test --device cuda
```

### Data Dependencies

- Same as Technical Analyst plus:
- `data/yFinance/processed/ohlcv_final.csv` — close prices for drawdown calculation

---

## Module 3: Regime Detection (MTGNN Graph Builder + Classifier)

### Specification

| Aspect | Detail |
|--------|--------|
| **Model** | MTGNN Graph Learning Layer → Graph Properties → MLP Classifier |
| **Input 1** | 256-dim temporal embeddings per stock |
| **Input 2** | 256-dim FinBERT embeddings per stock (aggregated per period) |
| **Combined Input** | `(N_stocks, 512)` — concatenated temporal + text |
| **Graph Building** | MTGNN learns N×N adjacency via self-attention (K=66 edges/node) |
| **Graph Properties** | density, modularity, avg_degree, clustering_coef |
| **Output** | `regime_label` (calm/volatile/crisis/rotation), `regime_confidence` (0-1) |
| **Classifier** | MLP(4 graph properties → 32 → 4) |
| **Training** | Weekly; classifier supervised or rule-based fallback |

### Target Construction

The regime detection module has **two modes**:

**Rule-based fallback (no training needed):**
```python
# Graph density thresholds (from GNN_Pre_Specifications.md):
#   density < 0.02 AND modularity > 0.4  → CALM
#   density > 0.08                        → CRISIS
#   density 0.02-0.08, modularity 0.2-0.4 → VOLATILE
#   modularity > 0.3, density 0.03-0.06   → ROTATION
```

**Supervised (if labels available):**
- VIX-based regime labels: VIX < 15 = calm, 15-25 = volatile, 25-35 = crisis, >35 = extreme
- Or use FRED macro data for regime labeling

### XAI Requirements

| Level | Method | Output |
|-------|--------|--------|
| L1 | Graph property values | Density, modularity, avg_degree |
| L2 | Graph diff | What changed from previous period |
| L3 | GNNExplainer | Key edges defining regime clusters |

### CLI Commands

```bash
python code/gnn/mtgnn_regime.py inspect
python code/gnn/mtgnn_regime.py build-graph --chunk 1 --split train --device cuda
python code/gnn/mtgnn_regime.py classify --chunk 1 --split test --device cuda
python code/gnn/mtgnn_regime.py train-classifier --chunk 1 --device cuda
```

### Data Dependencies

- `outputs/embeddings/TemporalEncoder/chunk*_*_embeddings.npy` — temporal embeddings
- `outputs/embeddings/FinBERT/chunk*_*_embeddings.npy` — text embeddings
- `data/graphs/static/edges.csv` & `nodes.csv` — optional static graph for initialization

---

## Common Code Requirements (All Modules Must Follow)

1. **Single file, dual-purpose:** Importable module + executable CLI
2. **CLI commands:** `inspect`, `hpo`, `train-best`, `train-best-all`, `predict`
3. **HPO:** Optuna TPE, SQLite persistence, resume-capable, `--fresh` flag
4. **XAI:** Every module exports explanations alongside predictions (3 levels minimum)
5. **Paths:** Configurable via `--repo-root`, defaults from `.env` or relative paths
6. **No data leakage:** Chronological splits only, train-fitted normalizers
7. **GPU/CPU:** `--device cuda` or `--device cpu`
8. **Checkpointing:** `best_model.pt` + `latest_model.pt` per chunk
9. **Manifest-based:** Load `manifest.csv` + `.npy` embeddings; auto-generate manifest if missing
10. **Run instructions:** Comment block at end of file with exact CLI commands

---

## Development Order

```
1. Technical Analyst    ← Needs temporal embeddings (available)
2. Drawdown Model       ← Needs temporal embeddings (available)
3. Regime Detection     ← Needs temporal + FinBERT embeddings (both available)
```

All three can be developed in parallel since they share no dependencies with each other — only the upstream embeddings which are already generated for Chunk 1.

---


1. **Write Technical Analyst** — start with this since labels are simplest (rule-based from returns)
2. **Write Drawdown Model** — same embedding input, different labels
3. **Write MTGNN Regime** — needs both embedding types plus graph building

