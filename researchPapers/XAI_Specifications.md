# xAI Specifications

## Document Purpose

This document contains **all specifications, decisions, and requirements** for the Explainability (XAI) Layer of the **Explainable Distributed Deep Learning Framework for Financial Risk Management**.

Explainability is one of the **core design pillars**. Every module in the system must produce human-interpretable explanations for its outputs. The XAI layer aggregates these explanations into a coherent, stakeholder-facing narrative.

---

## Table of Contents

1. [Why XAI Is Central to This Project](#1-why-xai-is-central-to-this-project)
2. [XAI Architecture Overview](#2-xai-architecture-overview)
3. [Standardized Explanation Output Format](#3-standardized-explanation-output-format)
4. [XAI Methods by Module](#4-xai-methods-by-module)
5. [Method Details](#5-method-details)
6. [GNN-Specific XAI (GNNExplainer)](#6-gnn-specific-xai-gnnexplainer)
7. [Implementation Specifications](#7-implementation-specifications)
8. [XAI Output Examples](#8-xai-output-examples)
9. [Evaluation of Explanation Quality](#9-evaluation-of-explanation-quality)
10. [File Structure](#10-file-structure)
11. [Acknowledgment: Living Document](#11-acknowledgment-living-document)

---

## 1. Why XAI Is Central to This Project

### The Core Problem

Deep learning models for financial trading are typically **black boxes**. They produce predictions (Buy/Hold/Sell) but cannot explain WHY. This is unacceptable for:

| Concern | Why It Matters |
|---------|---------------|
| **Trust** | Traders won't act on predictions they don't understand |
| **Regulation** | Financial regulators increasingly require explainable AI |
| **Debugging** | Without explanations, you can't fix model errors |
| **Thesis Defense** | "Explainable" is in the project title — XAI is non-negotiable |
| **Risk Management** | Understanding risk drivers is as important as the risk score itself |

### The Contrast with TradingAgents

The TradingAgents paper (one of our baseline references) uses LLM-based agents that explain decisions in natural language. Our approach is different — we use **learned neural modules** that must be made explainable through post-hoc XAI methods. This is more novel, more reproducible, and more computationally feasible.

### Our XAI Philosophy

> **Every module produces an explanation. Every explanation is auditable. Every decision is traceable.**

The XAI layer operates at **two levels**:
1. **Module-level explanations:** Each module explains its own output
2. **System-level explanations:** The fusion layer explains how module outputs were combined

---

## 2. XAI Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT DATA                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            v
┌─────────────────────────────────────────────────────────────┐
│  ENCODERS                                                   │
│  ├── Temporal Encoder ───→ Attention Visualization          │
│  ├── FinBERT ────────────→ Attention Visualization          │
│  └── Fundamental Encoder ─→ Feature Importance (XGBoost)    │
└─────────────────────────────────────────────────────────────┘
                            │
                            v
┌─────────────────────────────────────────────────────────────┐
│  ANALYSTS                                                   │
│  ├── Technical Analyst ──→ LIME + Attention                 │
│  ├── Sentiment Analyst ──→ SHAP                             │
│  ├── News Analyst ───────→ Attention Weights                │
│  └── Fundamental Analyst → SHAP + Feature Importance        │
└─────────────────────────────────────────────────────────────┘
                            │
                            v
┌─────────────────────────────────────────────────────────────┐
│  RISK ENGINE                                                │
│  ├── Volatility ─────────→ Counterfactuals                  │
│  ├── Drawdown ───────────→ Counterfactuals + LIME           │
│  ├── VaR/CVaR ───────────→ Historical Distribution Plot     │
│  ├── Contagion (StemGNN) → GNNExplainer + Attention         │
│  ├── Liquidity ──────────→ Rule Trace                       │
│  ├── Regime (MTGNN) ─────→ GNNExplainer + Graph Properties  │
│  └── Position Sizing ────→ Weight Breakdown                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            v
┌─────────────────────────────────────────────────────────────┐
│  FUSION LAYER                                               │
│  └── Fusion Engine ──────→ SHAP + Decision Breakdown        │
└─────────────────────────────────────────────────────────────┘
                            │
                            v
┌─────────────────────────────────────────────────────────────┐
│  XAI AGGREGATOR                                             │
│  Collects all module explanations                           │
│  Produces unified stakeholder-facing report                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            v
┌─────────────────────────────────────────────────────────────┐
│  FINAL OUTPUT                                               │
│  ├── Buy / Hold / Sell                                      │
│  ├── Confidence Score                                       │
│  ├── Position Size                                          │
│  ├── Risk Summary                                           │
│  └── COMPREHENSIVE EXPLANATION REPORT                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Standardized Explanation Output Format

### Every Module MUST Return This Structure

```python
{
    "module_name": str,              # e.g., "VolatilityRiskModule"
    "primary_score": float,          # Main output score (0-1 or -1 to +1)
    "confidence": float,             # How confident the module is (0-1)
    "raw_value": float,              # Unnormalized value (e.g., 35.2% volatility)
    
    "explanation": {
        # === What contributed (positive drivers) ===
        "top_positive_factors": [
            {
                "factor": str,       # e.g., "20-day volatility below median"
                "weight": float,     # Contribution magnitude (0-1)
                "direction": str,    # "positive" or "negative"
            },
            ...
        ],
        
        # === What contributed (negative drivers) ===
        "top_negative_factors": [
            {
                "factor": str,
                "weight": float,
                "direction": str,
            },
            ...
        ],
        
        # === Thresholds and benchmarks ===
        "thresholds_exceeded": [
            {
                "threshold": str,    # e.g., "90-day average"
                "current_value": float,
                "limit": float,
                "severity": str,     # "warning", "critical"
            },
            ...
        ],
        
        # === Context and comparisons ===
        "percentile_vs_history": float,     # 0-100
        "percentile_vs_sector": float,      # 0-100
        "percentile_vs_market": float,      # 0-100
        "trend": str,                       # "increasing", "decreasing", "stable"
        "trend_strength": float,            # 0-1
        
        # === Model-specific interpretability ===
        "feature_importance": dict,         # SHAP values: {feature_name: shap_value}
        "attention_weights": dict,          # For attention models: {time_step: weight}
        "lime_explanation": dict,           # LIME local explanation
        "gnn_explanation": dict,            # GNNExplainer output (for GNN modules)
        
        # === Counterfactuals ===
        "counterfactuals": {
            "what_if_scenarios": [
                {
                    "condition": str,       # e.g., "If VIX were 5 points lower"
                    "resulting_score": float,
                },
                ...
            ]
        },
        
        # === Similar historical periods ===
        "similar_historical_periods": [
            {
                "date": str,                # e.g., "2020-03-15"
                "similarity": float,        # 0-1
                "outcome": str,             # What happened
            },
            ...
        ],
        
        # === Confidence breakdown ===
        "confidence_factors": {
            "data_quality": float,          # 0-1
            "model_agreement": float,       # 0-1 (if ensemble)
            "historical_accuracy": float,   # 0-1
            "regime_compatibility": float,  # 0-1
        },
    },
    
    "metadata": {
        "model_version": str,
        "timestamp": str,
        "computation_time_ms": float,
    }
}
```

---

## 4. XAI Methods by Module

| Module | Primary XAI Method | Secondary Method | What It Explains |
|--------|-------------------|------------------|------------------|
| **Temporal Encoder** | Attention Visualization | LIME | Which time steps mattered most |
| **FinBERT Encoder** | Attention Visualization | SHAP | Which words/phrases drove sentiment |
| **Technical Analyst** | LIME | Attention Pooling Weights | Why trend/momentum scores |
| **Sentiment Analyst** | SHAP | — | Which text features drove polarity |
| **News Analyst** | Attention Weights | LIME | Which news items mattered most |
| **Volatility Model** | Counterfactuals | LIME | What would change the volatility score |
| **Drawdown Model** | Counterfactuals | LIME | What would reduce drawdown risk |
| **VaR/CVaR** | Historical Distribution Plot | — | Where current risk sits in history |
| **Contagion (StemGNN)** | **GNNExplainer** | Attention Weights | Which stocks influence contagion score |
| **Liquidity** | Rule Trace | — | Which rule thresholds were triggered |
| **Regime (MTGNN)** | **GNNExplainer** | Graph Properties | Why this regime was classified |
| **Position Sizing** | Weight Breakdown | Rule Trace | How each risk contributed to size |
| **Fusion Engine** | SHAP | Decision Breakdown | How analysts/risk scores were combined |

---

## 5. Method Details

### 5.1 SHAP (SHapley Additive exPlanations)

**What it does:** Assigns each input feature an importance value for a particular prediction. Based on cooperative game theory — each feature gets a "fair share" of the prediction.

**Applied to:**
- **Sentiment Analyst:** Which text embedding dimensions drove polarity
- **Fusion Layer:** Which module outputs most influenced the final decision

**Implementation:**
```python
import shap

# For neural models
explainer = shap.DeepExplainer(model, background_data)
shap_values = explainer.shap_values(input_data)

# Output format
{
    "feature_importance": {
        "revenue_growth_yoy": 0.23,
        "net_margin": 0.18,
        "debt_to_equity": -0.12,
        ...
    }
}
```

### 5.2 LIME (Local Interpretable Model-agnostic Explanations)

**What it does:** Creates a simple, interpretable model (like linear regression) that approximates the complex model's behavior around a specific prediction.

**Applied to:**
- **Technical Analyst:** Which parts of the temporal embedding influenced trend/momentum
- **News Analyst:** Which news items most affected the impact score
- **Volatility Model (MLP part):** Which input features drove volatility estimate
- **Drawdown Model:** Which temporal features indicated drawdown risk

**Implementation:**
```python
import lime
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data, feature_names=feature_names, mode='regression'
)
explanation = explainer.explain_instance(input_instance, model.predict)

# Output format
{
    "lime_explanation": {
        "top_features": [
            {"feature": "volatility_20d", "weight": 0.35},
            {"feature": "rsi_14", "weight": -0.22},
            ...
        ]
    }
}
```

### 5.3 Attention Visualization

**What it does:** Extracts attention weights from transformer/attention layers to show which parts of the input the model focused on.

**Applied to:**
- **Temporal Encoder:** Which days in the 30-90 day window received most attention
- **FinBERT:** Which words/tokens drove the text embedding
- **News Analyst:** Which news articles received highest attention weights
- **StemGNN:** Which other stocks received highest attention in correlation learning

**Implementation:**
```python
def extract_attention_weights(model, input_data):
    """Extract attention weights from transformer layers."""
    with torch.no_grad():
        outputs = model(input_data, output_attentions=True)
        attentions = outputs.attentions  # Tuple of (batch, heads, seq, seq)
    
    # Average across heads and layers
    avg_attention = torch.stack(attentions).mean(dim=(0, 2))
    
    return {
        "attention_weights": avg_attention.tolist()
    }
```

### 5.4 Counterfactual Analysis

**What it does:** Answers "what if" questions — what would need to change for the model to give a different output.

**Applied to:**
- **Volatility Model:** "If VIX were 5 points lower, volatility risk would be 0.45 instead of 0.72"
- **Drawdown Model:** "If the stock hadn't dropped 3% last Tuesday, drawdown risk would be 0.30"
- **All risk modules:** What conditions would change the risk score

**Implementation:**
```python
def generate_counterfactuals(model, input_data, target_score, n_scenarios=5):
    """Generate what-if scenarios that would change the output."""
    scenarios = []
    
    # Perturb each feature and observe output change
    for feature in input_data.columns:
        perturbed = input_data.copy()
        perturbed[feature] = perturbed[feature] * 0.8  # 20% reduction
        new_score = model.predict(perturbed)
        
        if abs(new_score - target_score) > 0.1:
            scenarios.append({
                "condition": f"If {feature} were 20% lower",
                "resulting_score": new_score,
                "change": new_score - target_score,
            })
    
    return {"counterfactuals": {"what_if_scenarios": scenarios[:n_scenarios]}}
```

### 5.5 Rule Trace (for Rule-Based Modules)

**What it does:** Simply logs which rules were triggered and why.

**Applied to:**
- **Liquidity Module:** Which thresholds were exceeded
- **Position Sizing:** Which risk factors caused size reduction

**Implementation:**
```python
def trace_rules(rule_results):
    """Log which rules fired."""
    triggered = []
    for rule_name, (condition_met, details) in rule_results.items():
        if condition_met:
            triggered.append({
                "rule": rule_name,
                "details": details,
            })
    return {"triggered_rules": triggered}
```

---

## 6. GNN-Specific XAI (GNNExplainer)

### Why GNNExplainer?

Both the **StemGNN Contagion Module** and the **MTGNN Regime Module** use graph neural networks. Explaining GNNs is uniquely challenging because predictions depend on both:
- **Node features** (stock characteristics)
- **Graph structure** (relationships between stocks)

Standard XAI methods (SHAP, LIME) do not natively handle graph-structured data. **GNNExplainer** (Ying et al., 2019) is specifically designed for this purpose.

### What GNNExplainer Does

GNNExplainer identifies:
1. A **compact subgraph** of the computation graph that is most influential for a prediction
2. A **small subset of node features** that are most important

For a **node** (stock) whose contagion score we want to explain, GNNExplainer finds:
- Which connected stocks (edges) most influenced the score
- Which features of those stocks (returns, sector, etc.) mattered most

### GNNExplainer for StemGNN (Contagion Module)

**Question:** "Why does AAPL have a contagion score of 0.72?"

**What GNNExplainer produces:**
```python
{
    "gnn_explanation": {
        "important_subgraph": {
            "nodes": ["AAPL", "MSFT", "NVDA", "QQQ"],
            "edges": [
                {"source": "MSFT", "target": "AAPL", "importance": 0.85},
                {"source": "NVDA", "target": "AAPL", "importance": 0.62},
                {"source": "QQQ", "target": "AAPL", "importance": 0.58},
            ]
        },
        "important_features": ["correlation_30d", "sector_similarity", "beta_similarity"],
        "explanation_text": "AAPL's contagion score is driven by strong correlations with MSFT (0.85) and NVDA (0.62), both in the Technology sector. ETF membership in QQQ also contributes."
    }
}
```

### GNNExplainer for MTGNN (Regime Module)

**Question:** "Why is the market classified as 'volatile'?"

**What GNNExplainer produces:**
```python
{
    "gnn_explanation": {
        "graph_properties": {
            "density": 0.042,
            "modularity": 0.31,
            "avg_degree": 186.4,
        },
        "key_structures": "High cross-sector connectivity with emerging clusters in defensive sectors",
        "explanation_text": "Market classified as VOLATILE because graph density (0.042) indicates moderate connectivity, modularity (0.31) shows sector clustering, and average degree (186.4) suggests widespread but not panic-level connections."
    }
}
```

### How GNNExplainer Works (Simplified)

1. **Input:** Trained GNN model + specific prediction to explain
2. **Learn a mask:** Optimize a subgraph mask `M` that selects important edges
3. **Learn a feature mask:** Optimize a feature selector `F` for important node features
4. **Objective:** Maximize mutual information between the masked input and the prediction
5. **Output:** Compact subgraph + feature subset that best explains the prediction

The key insight: GNNExplainer learns what to **remove** from the computation graph. If removing an edge drastically changes the prediction, that edge is important.

---

## 7. Implementation Specifications

### Code Structure

```
code/xai/
├── __init__.py
├── explainability.py          # Main XAI orchestrator
├── shap_explainer.py          # SHAP-based explanations
├── lime_explainer.py          # LIME-based explanations
├── attention_viz.py           # Attention visualization
├── counterfactuals.py         # Counterfactual analysis
├── feature_importance.py      # Tree-based feature importance
├── rule_trace.py              # Rule-based module explanations
├── gnn_explainer.py           # GNNExplainer wrapper
├── explanation_aggregator.py  # Combines all module explanations
└── report_generator.py        # Generates stakeholder-facing reports
```

### Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| **XAI computation** | Offline (after inference) | XAI methods are computationally expensive |
| **Explanation storage** | JSON alongside predictions | Auditable, replayable |
| **GNNExplainer integration** | Post-hoc, model-agnostic | Works with any GNN architecture |
| **Attention extraction** | Hook into model forward pass | Requires model support for `output_attentions=True` |
| **SHAP background data** | Sample of 100 training instances | Balance speed vs accuracy |

### Performance Considerations

| Method | Compute Cost | Frequency |
|--------|-------------|-----------|
| SHAP (tree) | Low (milliseconds) | Every prediction |
| SHAP (deep) | Medium (seconds) | Every prediction |
| LIME | Medium (seconds) | Every prediction |
| Attention Viz | Low (extraction only) | Every prediction |
| Counterfactuals | High (multiple forward passes) | On demand |
| **GNNExplainer** | **High (optimization loop)** | **Daily or on demand** |

GNNExplainer is the most expensive XAI method because it runs an optimization loop. It should be cached and updated daily, not run for every single inference.

---

## 8. XAI Output Examples

### Example 1: Module-Level Explanation (Volatility Risk)

```json
{
  "module_name": "VolatilityRiskModule",
  "primary_score": 0.72,
  "confidence": 0.85,
  "raw_value": 35.2,
  "explanation": {
    "top_positive_factors": [
      {"factor": "20-day realized volatility above 90th percentile", "weight": 0.35, "direction": "positive"},
      {"factor": "VIX correlation increased to 0.82", "weight": 0.25, "direction": "positive"}
    ],
    "top_negative_factors": [
      {"factor": "GARCH forecast declining", "weight": 0.15, "direction": "negative"}
    ],
    "thresholds_exceeded": [
      {"threshold": "90-day average volatility", "current_value": 35.2, "limit": 28.0, "severity": "warning"}
    ],
    "percentile_vs_history": 82,
    "trend": "increasing",
    "trend_strength": 0.67,
    "counterfactuals": {
      "what_if_scenarios": [
        {"condition": "If VIX were 5 points lower", "resulting_score": 0.45},
        {"condition": "If 20-day vol returned to median", "resulting_score": 0.38}
      ]
    },
    "similar_historical_periods": [
      {"date": "2020-03-15", "similarity": 0.82, "outcome": "Volatility remained elevated for 3 weeks"}
    ],
    "confidence_factors": {
      "data_quality": 0.95,
      "model_agreement": 0.82,
      "historical_accuracy": 0.78
    }
  }
}
```

### Example 2: System-Level Explanation (Final Decision)

```json
{
  "final_decision": "SELL",
  "confidence": 0.78,
  "position_size": 0.25,
  "decision_drivers": {
    "primary_reasons": [
      "Contagion risk elevated (0.72) due to Tech sector correlation spike",
      "Volatility above 90th percentile at 35.2%",
      "Fundamental analysis shows overvaluation (P/E 37.8 vs sector 22.1)"
    ],
    "mitigating_factors": [
      "Technical momentum still positive (0.65)",
      "Sentiment remains bullish (0.58)"
    ],
    "veto_triggers": [],
    "module_contributions": {
      "contagion_risk": 0.25,
      "volatility_risk": 0.20,
      "fundamental_analyst": 0.18,
      "technical_analyst": -0.10,
      "sentiment_analyst": -0.08,
      "drawdown_risk": 0.15,
      "liquidity_risk": 0.05,
      "regime_risk": 0.10
    }
  }
}
```

---

## 9. Evaluation of Explanation Quality

### How We Know Explanations Are Good

| Metric | Target | Method |
|--------|--------|--------|
| **Fidelity** | Explanation predicts model output accurately | Compare explanation-based prediction vs actual |
| **Sparsity** | Explanations use few features/edges | Count features/edges in explanation |
| **Stability** | Similar inputs get similar explanations | Cosine similarity of explanations for nearby inputs |
| **Completeness** | Explanation covers all major drivers | % of output variance explained by top factors |
| **Human evaluation** | Domain experts agree with explanations | User study (thesis defense) |

### For GNN Explanations Specifically

| Metric | Target | Method |
|--------|--------|--------|
| **Explanation accuracy** | >80% edge identification | Compare to known causal structures |
| **Subgraph size** | <20% of computation graph | GNNExplainer sparsity constraint |
| **Feature selectivity** | <50% of available features | Feature mask sparsity |

---

## 10. File Structure

```
fin-glassbox/
├── code/
│   └── xai/
│       ├── __init__.py
│       ├── explainability.py           # Main orchestrator
│       ├── shap_explainer.py           # SHAP wrapper
│       ├── lime_explainer.py           # LIME wrapper
│       ├── attention_viz.py            # Attention extraction & visualization
│       ├── counterfactuals.py          # Counterfactual generation
│       ├── feature_importance.py       # Tree-based importance
│       ├── rule_trace.py               # Rule tracing
│       ├── gnn_explainer.py            # GNNExplainer integration
│       ├── explanation_aggregator.py   # Combine module explanations
│       └── report_generator.py         # Generate final report
├── data/
│   └── explanations/                   # Cached explanations (JSON)
└── researchPapers/
    └── XAI_Specifications.md           # THIS FILE
```

---

## 11. Acknowledgment: Living Document

### ⚠️ These Specifications Will Evolve

This document represents the **initial XAI design** based on architecture discussions during the planning phase. As the project progresses through implementation, the following will be refined:

1. **Exact feature lists** for SHAP/LIME depend on final model architectures
2. **GNNExplainer integration** details depend on how StemGNN and MTGNN are adapted
3. **Attention extraction** depends on whether models support `output_attentions=True`
4. **Counterfactual generation** logic depends on the final feature engineering
5. **Explanation report format** will be refined based on stakeholder feedback

### Reference Documents

For the most current overall architecture and model specifications, refer to:
- **`UpdatedWorkflow.md`** — Complete architecture with all finalized model assignments
- **`Hyperparameter_Config.md`** — Current HP values
- **`GNN_Pre_Specifications.md`** — GNN module details
- **`CrossAssetRelationData.md`** — Graph construction (input to GNNs that GNNExplainer explains)

### Key References

- **GNNExplainer:** Ying, R., Bourgeois, D., You, J., Zitnik, M., & Leskovec, J. (2019). GNNExplainer: Generating Explanations for Graph Neural Networks. *Advances in Neural Information Processing Systems, 32.*
- **SHAP:** Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30.*
- **LIME:** Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. *Proceedings of the 22nd ACM SIGKDD.*

---

**Document Version:** 1.0  
**Status:** Pre-Implementation Specification 
**Last Updated:** 2026-04-26  
**Prepared for:** Explainable Distributed Deep Learning Framework for Financial Risk Management