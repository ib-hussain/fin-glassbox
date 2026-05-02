# Explainable Multimodal Neural Framework for Financial Risk Management

## UPDATED ARCHITECTURE SPECIFICATION

**Version: 2.0 (Finalized Model Specifications)**  
**Date: 22 April 2026**

---

## 1. SYSTEM OVERVIEW

### Core Philosophy

This framework integrates **three intelligence streams** into a unified decision system:

1. **Technical Market Stream** — Temporal patterns in price/volume data
2. **Text Stream** — Sentiment, news, and event understanding
3. **Fundamental Stream** — Company financial health and valuation

These streams feed into a **Risk Engine** (the control layer), then a **Fusion Layer** synthesizes all signals into a final trading decision with comprehensive explainability.

### Architecture Diagram

```text
INPUTS (5 Data Families)
├── Time-Series Market Data (4,428 tickers, daily OHLCV)
├── Financial Text Data (SEC filings, news)
├── Fundamental Data (70+ features per company)
├── Macro/Regime Data (FRED series)
└── Cross-Asset Relation Data (Graph structures)

ENCODERS (Produce 512-dim Unified Embeddings)
├── Shared Temporal Attention Encoder → 128-dim temporal embedding
├── FinBERT Financial Text Encoder → 256-dim text embedding
└── Fundamental Encoder (XGBoost→MLP) → 128-dim fundamental embedding

ANALYST MODULES
├── Technical Analyst (BiLSTM) → trend, momentum, timing scores
├── Sentiment Analyst (MLP) → sentiment polarity, confidence
├── News Analyst (Multi-Head Attention Pooling) → event impact, relevance
└── Fundamental Analyst (LightGBM) → value, quality, growth scores

RISK ENGINE
├── Volatility Estimation (GARCH + MLP Hybrid) → 10-day/30-day forecasts
├── Drawdown Risk (BiLSTM, Dual Horizon) → 10-day/30-day expected drawdown
├── Historical VaR (Non-parametric) → 95%, 99% thresholds
├── CVaR Expected Shortfall (Non-parametric) → tail risk severity
├── GNN Contagion Risk (StemGNN) → cross-asset spillover scores
├── Liquidity Risk (Rule-based) → execution feasibility
├── Regime Detection (MTGNN Graph Builder + Classifier) → market state
└── Position Sizing Engine (Rule-based, User-adjustable) → capital allocation

SYNTHESIS
├── Qualitative Analysis → sentiment + news + fundamental
├── Quantitative Analysis → technical + all risk modules
└── Fusion Engine (MLP Layer 1 + Rule-based Layer 2)

DECISION
└── Final Trade Approver → BUY/HOLD/SELL + confidence + size

EXPLAINABILITY (XAI)
└── Module-level + System-level (SHAP, LIME, Attention, Counterfactuals)

OUTPUT
├── Trading Decision (Buy/Hold/Sell)
├── Confidence Score
├── Position Size Recommendation
├── Risk Summary Dashboard
└── Comprehensive Explanation Report
```

---

## 2. ENCODER LAYER — PRODUCING 512-DIM UNIFIED EMBEDDINGS

### 2A. Shared Temporal Attention Encoder

| Specification | Value |
|--------------|-------|
| **Model** | Transformer Encoder (4 layers, 4 attention heads) |
| **Input** | 30-90 days OHLCV + derived indicators (returns, RSI, MACD, etc.) |
| **Input Shape** | `(batch, seq_len=30, features=10)` |
| **Output** | 128-dim temporal embedding |
| **Activation** | GELU |
| **Regularization** | Dropout=0.1, Attention Dropout=0.1, Weight Decay=1e-5 |
| **Training** | From scratch on 26.7M data points |
| **Hyperparameter Optimization** | TPE (Bayesian), 50-100 trials |
| **Anti-Overfitting** | Early Stopping (patience=20), Gradient Clipping=1.0, Label Smoothing=0.05, Cosine LR Schedule with Warmup |

**Output Usage:** Technical Analyst, Volatility Model, Drawdown Model, Regime Model

---

### 2B. FinBERT Financial Text Encoder

| Specification | Value |
|--------------|-------|
| **Model** | FinBERT (base, 110M parameters) + Projection Layer |
| **Input** | SEC filings text, news headlines, earnings call transcripts |
| **Input Processing** | Max 512 tokens per document, mean pooling across documents |
| **Output** | 256-dim text embedding (projected from FinBERT's 768-dim) |
| **Fine-tuning** | Chunked chronological: Train on 2000-2004, 2007-2014, 2017-2022; Test on 2005-2006, 2015-2016, 2023-2024 |
| **Regularization** | Dropout=0.1, Weight Decay=0.01 |
| **Training** | 3 epochs per chunk, LR=2e-5, batch_size=16 |
| **Anti-Overfitting** | Early Stopping (patience=5), Gradient Clipping=1.0 |

**Output Usage:** Sentiment Analyst, News Analyst, Regime Model

---

### 2C. Fundamental Encoder

| Specification | Value |
|--------------|-------|
| **Model** | XGBoost (feature extraction) → MLP Projection (2 layers) |
| **Input** | 70 fundamental features: 36 raw financials + 34 derived ratios |
| **Input Shape** | `(batch, 70)` |
| **Output** | 128-dim fundamental embedding |
| **XGBoost Params** | max_depth=4, learning_rate=0.01, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0 |
| **MLP Architecture** | 70 → 256 → 128 (with LayerNorm, ReLU, Dropout=0.2) |
| **Training** | XGBoost trained first; MLP trained on XGBoost leaf embeddings |
| **Hyperparameter Optimization** | Grid Search for XGBoost (27 combinations); TPE for MLP (30 trials) |
| **Anti-Overfitting** | Early Stopping (rounds=50 for XGBoost, patience=15 for MLP) |

**Output Usage:** Fundamental Analyst

---

### Unified Asset Embedding Assembly

```python
def get_asset_embedding(ticker, date):
    temporal_emb = temporal_encoder(market_data[ticker])           # 128-dim
    text_emb = finbert_encoder(text_data[ticker])                  # 256-dim
    fundamental_emb = fundamental_encoder(fundamentals[ticker])    # 128-dim
    
    # Concatenate into 512-dim unified embedding
    return torch.cat([temporal_emb, text_emb, fundamental_emb])    # 512-dim
```

---

## 3. ANALYST LAYER

### 3A. Technical Analyst

| Specification | Value |
|--------------|-------|
| **Model** | BiLSTM (1 layer, hidden=64) |
| **Input** | 128-dim temporal embedding (sequence of 30 days) |
| **Input Shape** | `(batch, seq_len=30, 128)` |
| **Output** | `trend_score` (0-1), `momentum_score` (0-1), `timing_confidence` (0-1) |
| **Architecture** | BiLSTM(128→64) → Attention Pooling → Linear(64→3) → Sigmoid |
| **Regularization** | Dropout=0.3, Weight Decay=1e-4 |
| **Training** | From scratch on temporal embeddings |
| **Hyperparameter Optimization** | TPE (Bayesian), 30-50 trials |
| **Anti-Overfitting** | Early Stopping (patience=20), Gradient Clipping=1.0 |

---

### 3B. Sentiment Analyst

| Specification | Value |
|--------------|-------|
| **Model** | MLP (3 layers) |
| **Input** | 256-dim text embedding (aggregated across documents) |
| **Output** | `sentiment_polarity` (-1 to +1), `sentiment_confidence` (0-1) |
| **Architecture** | 256 → 128 → 64 → 2 (Tanh for polarity, Sigmoid for confidence) |
| **Regularization** | Dropout=0.2, Weight Decay=1e-5 |
| **Training** | From scratch on text embeddings |
| **Hyperparameter Optimization** | Grid Search (coarse) + TPE fine-tuning (20-30 trials) |

---

### 3C. News Analyst

| Specification | Value |
|--------------|-------|
| **Model** | Multi-Head Attention Pooling (4 heads) |
| **Input** | Multiple 256-dim text embeddings (one per news item/filing) |
| **Input Shape** | `(batch, num_documents, 256)` |
| **Output** | `event_impact_score` (-1 to +1), `relevance_score` (0-1) |
| **Architecture** | Multi-Head Attention (4 heads) → Weighted Mean Pooling → Linear(256→2) |
| **Regularization** | Attention Dropout=0.1, Dropout=0.1 |
| **Training** | From scratch on document sequences |
| **Hyperparameter Optimization** | TPE (Bayesian), 20-30 trials |

---

### 3D. Fundamental Analyst

| Specification | Value |
|--------------|-------|
| **Model** | LightGBM Classifier |
| **Input** | 128-dim fundamental embedding |
| **Output** | `value_score` (0-1), `quality_score` (0-1), `growth_score` (0-1) |
| **LightGBM Params** | num_leaves=31, learning_rate=0.01, min_child_samples=20, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0 |
| **Training** | 425K company-quarters, chunked chronologically |
| **Hyperparameter Optimization** | Grid Search (27 combinations) + Optuna fine-tuning |
| **Anti-Overfitting** | Early Stopping (rounds=50), Cross-validation (5-fold chronological) |

---

## 4. RISK ENGINE

### 4A. Volatility Estimation Model

| Specification | Value |
|--------------|-------|
| **Model** | GARCH(1,1) + MLP Hybrid |
| **Input** | 128-dim temporal embedding + historical returns |
| **Output** | `volatility_10d` (annualized %), `volatility_30d` (annualized %), `volatility_regime` (low/medium/high), `confidence` (0-1) |
| **Architecture** | GARCH statistical baseline + MLP(128→64→4) adjustment factor |
| **Training** | GARCH fitted per stock; MLP trained on residuals |
| **Hyperparameter Optimization** | TPE for MLP (30-50 trials) |
| **Anti-Overfitting** | Early Stopping (patience=15), Weight Decay=1e-5 |

---

### 4B. Drawdown Risk Model

| Specification | Value |
|--------------|-------|
| **Model** | BiLSTM (1 layer, hidden=64) with Dual Horizon Heads |
| **Input** | 128-dim temporal embedding (sequence of 30-90 days) |
| **Output** | 10-day: `expected_drawdown_pct`, `drawdown_probability`, `recovery_days_estimate`; 30-day: same structure |
| **Architecture** | BiLSTM(128→64) → Shared Features → Head_10d(64→3) + Head_30d(64→3) |
| **Regularization** | Dropout=0.3, Weight Decay=1e-4 |
| **Training** | From scratch on temporal embeddings |
| **Hyperparameter Optimization** | TPE (Bayesian), 30-50 trials |
| **Anti-Overfitting** | Early Stopping (patience=20), Gradient Clipping=1.0 |

---

### 4C. Historical VaR Module

| Specification | Value |
|--------------|-------|
| **Model** | Non-parametric (Empirical Distribution) |
| **Input** | 2-year rolling window of daily returns |
| **Output** | `var_95` (threshold loss at 95% confidence), `var_99` (threshold at 99% confidence) |
| **Calculation** | `np.percentile(returns, 5)` and `np.percentile(returns, 1)` |
| **Update Frequency** | Daily (recalculated each trading day) |
| **Training** | None (statistical calculation) |

---

### 4D. CVaR / Expected Shortfall Module

| Specification | Value |
|--------------|-------|
| **Model** | Non-parametric (Empirical Distribution) |
| **Input** | 2-year rolling window of daily returns |
| **Output** | `cvar_95` (average loss beyond VaR 95), `cvar_99` (average loss beyond VaR 99), `tail_risk_ratio` (CVaR/VaR) |
| **Calculation** | Mean of returns below VaR threshold |
| **Update Frequency** | Daily |
| **Training** | None |

---

### 4E. GNN Contagion Risk Module

| Specification | Value |
|--------------|-------|
| **Model** | StemGNN (Full Spectral-Temporal GNN) |
| **Input** | Returns matrix for all 4,428 tickers: `(N_stocks, T=30)` |
| **Graph Structure** | Learned adjacency via GRU + Self-Attention (K=66 edges per node) |
| **Output** | `contagion_score` (0-1 per stock), `network_centrality` (0-1), `cluster_id`, `top_influencers` (list) |
| **Architecture** | Latent Correlation Layer → GFT → 13 Spectral-Temporal Blocks → Output MLP |
| **Regularization** | Dropout=0.75, Weight Decay=1e-5 (from baseline) |
| **Training** | Monthly retraining, 45-60 min on T4 GPU |
| **Hyperparameter Optimization** | TPE (Bayesian), 50-100 trials |
| **Anti-Overfitting** | Early Stopping (patience=20), Gradient Clipping=1.0 |
| **Relationship Vector** | 8-dim: correlation_30d, sector_similarity, etf_overlap, index_membership, market_cap_ratio, volume_correlation, beta_similarity, partial_correlation |

---

### 4F. Liquidity Risk Module

| Specification | Value |
|--------------|-------|
| **Model** | Rule-based (Deterministic) |
| **Input** | Average daily volume, bid-ask spread (proxy), market cap, turnover |
| **Output** | `liquidity_score` (0-1, 1=highly liquid), `slippage_estimate_pct`, `days_to_liquidate`, `tradable` (boolean) |
| **Rules** | Score based on volume percentile, spread threshold, market cap tier |
| **Training** | None (configurable thresholds) |
| **Update Frequency** | Daily |

---

### 4G. Regime Detection Module

| Specification | Value |
|--------------|-------|
| **Model** | MTGNN Graph Builder + MLP Classifier |
| **Input** | 128-dim temporal + 256-dim text (combined 384-dim per stock) |
| **Input Shape** | `(N_stocks=4428, 384)` |
| **Graph Building** | MTGNN Graph Learning Layer (K=66 edges per node) |
| **Output** | `regime_label` (calm/volatile/crisis/rotation), `regime_confidence` (0-1), `graph_density`, `modularity`, `transition_probability` |
| **Classifier** | MLP(5 graph properties → 32 → 4) |
| **Training** | Weekly retraining (graph building: 30-60 sec) |
| **Hyperparameter Optimization** | TPE for classifier (20-30 trials) |
| **Anti-Overfitting** | Dropout=0.2, Early Stopping (patience=10) |

---

### 4H. Position Sizing Engine

| Specification | Value |
|--------------|-------|
| **Model** | Rule-based with User-adjustable Weights |
| **Input** | All 7 risk scores: volatility, drawdown, VaR, CVaR, contagion, liquidity, regime |
| **Output** | `position_size_pct` (0-100% of max allowed), `size_reduction_reasons` (list), `risk_budget_used` (%) |
| **Default Weights** | volatility=0.20, drawdown=0.15, VaR_CVaR=0.15, contagion=0.25, liquidity=0.15, regime=0.10 |
| **Rules** | Weighted average → Threshold mapping: <0.3→100%, <0.5→75%, <0.7→50%, <0.85→25%, ≥0.85→0% |
| **User Control** | Weights adjustable via dashboard; rules modifiable |
| **Training** | None |

---

## 5. SYNTHESIS LAYER

### 5A. Qualitative Analysis

| Specification | Value |
|--------------|-------|
| **Inputs** | Sentiment Analyst (polarity, confidence), News Analyst (impact, relevance), Fundamental Analyst (value, quality, growth) |
| **Output** | `qualitative_score` (-1 to +1), `qualitative_confidence` (0-1) |
| **Method** | Weighted average (learned weights or equal) |

### 5B. Quantitative Analysis

| Specification | Value |
|--------------|-------|
| **Inputs** | Technical Analyst (trend, momentum, timing), All 7 Risk Module Outputs |
| **Output** | `quantitative_score` (-1 to +1), `quantitative_confidence` (0-1) |
| **Method** | Attention-weighted pooling across risk scores |

---

## 6. FUSION ENGINE (HYBRID)

| Specification | Value |
|--------------|-------|
| **Model** | Layer 1: MLP (Learned) + Layer 2: Rule-based Override |
| **Input** | Qualitative score + Quantitative score + All individual module outputs |
| **Layer 1 Architecture** | MLP: (2 + N_modules) → 64 → 32 → 3 (Buy/Hold/Sell logits) |
| **Layer 2 Rules** | Hard overrides: reject if liquidity < 0.3, cap size if drawdown > 0.8, veto if contagion > 0.9 |
| **Output** | `final_decision` (Buy/Hold/Sell), `fusion_confidence` (0-1), `contributing_factors` (list) |
| **Regularization** | Dropout=0.2, Weight Decay=1e-5 |
| **Training** | End-to-end on fusion dataset (all module outputs from training chunks) |
| **Hyperparameter Optimization** | TPE (Bayesian), 50-100 trials |
| **Anti-Overfitting** | Early Stopping (patience=15), Label Smoothing=0.05 |

---

## 7. FINAL DECISION LAYER

### Final Trade Approver

| Specification | Value |
|--------------|-------|
| **Input** | Fusion decision, confidence, position size, all risk summaries |
| **Output** | Final executable trade: `action` (BUY/HOLD/SELL), `size` (shares or %), `limit_price` (optional) |
| **Logic** | Pass-through with final safety checks (e.g., no trades on earnings day, circuit breaker) |

---

## 8. XAI LAYER (SPECIFICATION RESERVED)

### Output Requirements

Each module **MUST** produce an `explanation` dictionary containing:

```python
{
    "primary_score": float,
    "confidence": float,
    "explanation": {
        "top_positive_factors": list,
        "top_negative_factors": list,
        "thresholds_exceeded": list,
        "percentile_vs_history": float,
        "feature_importance": dict,  # SHAP values
        "counterfactuals": dict,
        "similar_historical_periods": list,
        "attention_weights": optional,  # For attention-based models
        "lime_explanation": optional    # For local interpretability
    }
}
# More will be added to the above list, this list is an "atleast" version
```

### XAI Methods to be Integrated

| Method | Applied To |
|--------|-----------|
| **SHAP** | Fundamental Analyst, Sentiment Analyst, Fusion Layer |
| **LIME** | All MLP-based modules, Volatility Hybrid |
| **Attention Visualization** | Temporal Encoder, News Analyst, StemGNN |
| **GNNExplainer** | Contagion Module (StemGNN), Regime Module (MTGNN) |
| **Counterfactual Analysis** | All risk modules |
| **Feature Importance** | XGBoost, LightGBM |

*Full XAI specifications to be added in a separate document.*

---

## 9. TRAINING PROTOCOL — CHUNKED CHRONOLOGICAL VALIDATION

| Chunk | Training Years | Testing Years | Purpose |
|-------|---------------|---------------|---------|
| 1 | 2000-2004 | 2005-2006 | Initial training, dot-com recovery |
| 2 | 2007-2014 | 2015-2016 | Financial crisis + recovery |
| 3 | 2017-2022 | 2023-2024 | COVID + bull market |

**All models use this exact same split** to ensure:
- No lookahead bias (training always before testing)
- Testing on unseen market regimes
- Fair comparison across modules

---

## 10. REGULARIZATION SUMMARY BY MODEL

| Model | Dropout | Attention Dropout | Weight Decay | Label Smoothing | Early Stop | Gradient Clip |
|-------|---------|-------------------|--------------|-----------------|------------|---------------|
| Temporal Encoder | 0.1 | 0.1 | 1e-5 | 0.05 | 20 | 1.0 |
| FinBERT | 0.1 | 0.1 | 0.01 | - | 5 | 1.0 |
| Fundamental Encoder | 0.2 | - | - | - | 50/15 | - |
| Technical Analyst (BiLSTM) | 0.3 | - | 1e-4 | - | 20 | 1.0 |
| Sentiment Analyst (MLP) | 0.2 | - | 1e-5 | 0.05 | 15 | 1.0 |
| News Analyst (Attention) | 0.1 | 0.1 | 1e-5 | - | 15 | 1.0 |
| Fundamental Analyst (LightGBM) | - | - | reg_alpha=0.1, reg_lambda=1.0 | - | 50 | - |
| Volatility Hybrid | 0.2 | - | 1e-5 | - | 15 | 1.0 |
| Drawdown (BiLSTM) | 0.3 | - | 1e-4 | - | 20 | 1.0 |
| StemGNN (Contagion) | 0.75 | - | 1e-5 | - | 20 | 1.0 |
| MTGNN Regime Classifier | 0.2 | - | 1e-5 | - | 10 | 1.0 |
| Fusion MLP | 0.2 | - | 1e-5 | 0.05 | 15 | 1.0 |

---

## 11. HYPERPARAMETER OPTIMIZATION STRATEGY
Hyperparameters for each model will individually be found and used
| Model | Method | Trials | Key Parameters Optimized |
|-------|--------|--------|-------------------------|
| Temporal Encoder | TPE (Bayesian) | 50-100 | lr, layers, heads, dropout |
| FinBERT | TPE (light) | 20-30 | lr, epochs |
| Fundamental Encoder | Grid + TPE | 27 + 30 | XGBoost params, MLP hidden size |
| Technical Analyst | TPE | 30-50 | lr, hidden_size, dropout |
| Sentiment Analyst | Grid + TPE | 9 + 20 | lr, layers, dropout |
| News Analyst | TPE | 20-30 | lr, heads, dropout |
| Fundamental Analyst | Grid + Optuna | 27 + 30 | num_leaves, lr, subsample |
| Volatility Hybrid | TPE | 30-50 | lr, hidden_size |
| Drawdown BiLSTM | TPE | 30-50 | lr, hidden_size, dropout |
| StemGNN | TPE | 50-100 | lr, multi_layer, decay_rate, dropout |
| MTGNN Regime | TPE | 20-30 | lr, hidden_size |
| Fusion MLP | TPE | 50-100 | lr, layers, dropout |

---

## 12. INFERENCE PIPELINE (DAILY)

```python
# this code is for concept only, the real inference code will be much different than this
def run_daily_inference(date, tickers):
    # 1. Fetch today's data
    market_data = fetch_market_data(tickers, lookback=90)
    text_data = fetch_text_data(tickers, lookback=7)
    fundamental_data = fetch_fundamentals(tickers, latest_quarter=True)
    
    # 2. Generate embeddings (frozen models)
    embeddings = {
        'temporal': temporal_encoder(market_data),
        'text': finbert_encoder(text_data),
        'fundamental': fundamental_encoder(fundamental_data)
    }
    
    # 3. Run analysts (frozen)
    analyst_scores = {
        'technical': technical_analyst(embeddings['temporal']),
        'sentiment': sentiment_analyst(embeddings['text']),
        'news': news_analyst(embeddings['text']),
        'fundamental': fundamental_analyst(embeddings['fundamental'])
    }
    
    # 4. Run risk modules (frozen where applicable)
    risk_scores = {
        'volatility': volatility_model(embeddings['temporal']),
        'drawdown': drawdown_model(embeddings['temporal']),
        'var_cvar': calculate_var_cvar(market_data['returns']),
        'contagion': stemgnn_contagion(all_returns_matrix),  # Cached daily
        'liquidity': liquidity_rules(market_data['volume']),
        'regime': mtgnn_regime(embeddings['temporal'], embeddings['text'])  # Cached weekly
    }
    
    # 5. Position sizing (rule-based)
    position_size = position_sizing(risk_scores)
    
    # 6. Fusion (frozen)
    decision = fusion_engine(analyst_scores, risk_scores)
    
    # 7. Final approval
    final = final_approver(decision, position_size, risk_scores)
    
    # 8. Generate explanations
    explanations = xai_layer.generate_all(analyst_scores, risk_scores, decision)
    
    return final, explanations
```

---

## 13. FILE STRUCTURE

```
fin-glassbox/
├── code/
│   ├── encoders/
│   │   ├── temporal_encoder.py      # Transformer (4 layers)
│   │   ├── finbert_encoder.py       # FinBERT + Projection
│   │   └── fundamental_encoder.py   # XGBoost + MLP
│   ├── analysts/
│   │   ├── technical_analyst.py     # BiLSTM
│   │   ├── sentiment_analyst.py     # MLP
│   │   ├── news_analyst.py          # Multi-Head Attention
│   │   └── fundamental_analyst.py   # LightGBM
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
│   │   ├── GNN_Pre_Specifications.md 
|   |   (the files of code here may be deleted but these files are placed here on the basis of the GNN_Pre_Specifications.md file)    
│   ├── riskEngine/
│   │   ├── volatility.py            # GARCH + MLP Hybrid
│   │   ├── drawdown.py              # BiLSTM Dual Horizon
│   │   ├── var_cvar.py              # Non-parametric
│   │   ├── contagion_gnn.py         # StemGNN(it's working will be here but the header file code will be in gnn folder)
│   │   ├── liquidity.py             # Rule-based
│   │   ├── regime_gnn.py            # MTGNN Graph(it's working will be here but the header file code will be in gnn folder) + Classifier
│   │   └── position_sizing.py       # Rule-based
│   ├── fusion/
│   │   └── fusion_engine.py         # MLP + Rules
│   └── xai/
│   │   └── explainability.py        # SHAP, LIME, Attention
│   │   ... and other files for different XAI methods
│   └── config/
│   │   ├── hyperparameters.yaml         # All HP configs
│   │   ├── training_chunks.yaml         # 2000-2004, etc.
│   │   └── regularization.yaml          # Dropout, WD values
│   ├── train_all.py                 # Master training script
│   ├── train_encoder.py
│   ├── train_analyst.py
│   ├── train_risk.py
│   ├── hyperparameter_search.py     # Optuna/TPE scripts
│   └── daily_inference.py           # Production inference
├── data/
│   ├── market_data/                 # yfinance output
│   ├── sec_edgar/processed/cleaned/ # Fundamentals & text
│   └── graphs/                      # Cached graph snapshots
└── researchPapers/
    ├── UpdatedWorkflow.md                  # THIS FILE
    ├── Hyperparameter_Config.md     # To be added
    └── XAI_Specifications.md        # To be added
```

---

## 14. FINAL NOTES

### What is NOT in this workflow (intentionally excluded):
- ❌ Residual connections (not allowed)
- ❌ LangChain/LangGraph (not needed for this architecture)
- ❌ LLM-based agents (TradingAgents approach — we use neural analysts instead)
- ❌ Walk-forward analysis for hyperparameter search (use chunked validation instead to avoid leakage concerns)

### What IS included:
- ✅ Complete model specifications for all 17 modules
- ✅ Input/output shapes and dimensions
- ✅ Regularization strategies per model
- ✅ Hyperparameter optimization methods and trial budgets
- ✅ Training chunk strategy (2000-2004, 2007-2014, 2017-2022)
- ✅ Anti-overfitting measures (Dropout, Weight Decay, Early Stopping, Gradient Clipping, Label Smoothing, LR Scheduling)
- ✅ Inference pipeline structure
- ✅ XAI output requirements (more methods to be added later, reserved for later specification)

---

**Document Version:** 2.0 — Finalized Model Specifications  
**Status:** APPROVED FOR IMPLEMENTATION  
