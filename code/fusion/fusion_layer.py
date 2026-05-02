#!/usr/bin/env python3
"""
code/fusion/fusion_layer.py

Hybrid Fusion Layer
===================

Project:
    fin-glassbox — Explainable Distributed Deep Learning Framework for Financial Risk Management

Purpose:
    Implements the reusable Fusion Engine internals.

Architecture:
    Layer 1:
        Learned fusion model.
        Learns:
            - quantitative branch weight
            - qualitative branch weight
            - fused signal
            - fused risk
            - fused confidence
            - learned position multiplier
            - learned Buy/Hold/Sell logits

    Layer 2:
        User-defined rule barrier.
        This is the final safety layer.
        It is deliberately not learned.

Design principle:
    The learned layer proposes.
    The rule barrier approves, caps, vetoes, or modifies.

This preserves:
    - explainability
    - user control
    - risk-engine authority
    - thesis defensibility

Inputs:
    Quantitative Analyst:
        outputs/results/QuantitativeAnalyst/quantitative_analysis_chunk{chunk}_{split}.csv

    Qualitative Analyst:
        outputs/results/QualitativeAnalyst/qualitative_daily_chunk{chunk}_{split}.csv

Outputs are written by:
    code/fusion/final_fusion.py
"""

from __future__ import annotations

import json
import math
import random
import shutil
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import optuna
except Exception:
    optuna = None

warnings.filterwarnings("ignore", category=FutureWarning)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

ACTION_TO_ID = {
    "SELL": 0,
    "HOLD": 1,
    "BUY": 2,
}

ID_TO_ACTION = {
    0: "SELL",
    1: "HOLD",
    2: "BUY",
}

RISK_ATTENTION_COLUMNS = [
    "risk_attention_volatility",
    "risk_attention_drawdown",
    "risk_attention_var_cvar",
    "risk_attention_contagion",
    "risk_attention_liquidity",
    "risk_attention_regime",
]

REQUIRED_QUANT_COLUMNS = [
    "ticker",
    "date",
    "quantitative_recommendation",
    "risk_adjusted_quantitative_signal",
    "technical_direction_score",
    "quantitative_risk_score",
    "quantitative_confidence",
    "quantitative_action_strength",
    "recommended_capital_fraction",
    "recommended_capital_pct",
    "position_fraction_of_max",
    "max_single_stock_exposure",
    "attention_pooled_risk_score",
    "top_attention_risk_driver",
    *RISK_ATTENTION_COLUMNS,
]

OPTIONAL_QUANT_COLUMNS = [
    "trend_score",
    "momentum_score",
    "timing_confidence",
    "technical_confidence",
    "volatility_risk_score",
    "drawdown_risk_score",
    "var_cvar_risk_score",
    "contagion_risk_score",
    "liquidity_risk_score",
    "regime_risk_score",
    "combined_risk_score",
    "binding_cap_source",
    "hard_cap_applied",
    "size_bucket",
    "regime_label",
    "regime_confidence",
    "liquidity_score",
    "tradable",
    "contagion_5d",
    "contagion_20d",
    "contagion_60d",
    "expected_drawdown_10d",
    "expected_drawdown_30d",
    "drawdown_risk_10d",
    "drawdown_risk_30d",
    "vol_10d",
    "vol_30d",
    "var_95",
    "var_99",
    "cvar_95",
    "cvar_99",
    "prob_calm",
    "prob_volatile",
    "prob_crisis",
    "prob_rotation",
    "xai_summary",
]

QUAL_COLUMNS = [
    "ticker",
    "date",
    "event_count",
    "sentiment_event_count",
    "news_event_count",
    "qualitative_score",
    "qualitative_risk_score",
    "qualitative_confidence",
    "qualitative_recommendation",
    "max_event_risk_score",
    "mean_event_risk_score",
    "mean_sentiment_score",
    "mean_news_impact_score",
    "mean_news_importance",
    "dominant_qualitative_driver",
    "xai_summary",
]

BASE_NUMERIC_FEATURES = [
    # Quantitative branch summary
    "risk_adjusted_quantitative_signal",
    "technical_direction_score",
    "quantitative_risk_score",
    "quantitative_confidence",
    "quantitative_action_strength",
    "recommended_capital_fraction",
    "recommended_capital_pct",
    "position_fraction_of_max",
    "max_single_stock_exposure",
    "attention_pooled_risk_score",

    # Risk attention features
    "risk_attention_volatility",
    "risk_attention_drawdown",
    "risk_attention_var_cvar",
    "risk_attention_contagion",
    "risk_attention_liquidity",
    "risk_attention_regime",

    # Technical features
    "trend_score",
    "momentum_score",
    "timing_confidence",
    "technical_confidence",

    # Risk engine features
    "volatility_risk_score",
    "drawdown_risk_score",
    "var_cvar_risk_score",
    "contagion_risk_score",
    "liquidity_risk_score",
    "regime_risk_score",
    "combined_risk_score",
    "regime_confidence",
    "liquidity_score",
    "tradable",
    "contagion_5d",
    "contagion_20d",
    "contagion_60d",
    "expected_drawdown_10d",
    "expected_drawdown_30d",
    "drawdown_risk_10d",
    "drawdown_risk_30d",
    "vol_10d",
    "vol_30d",
    "var_95",
    "var_99",
    "cvar_95",
    "cvar_99",
    "prob_calm",
    "prob_volatile",
    "prob_crisis",
    "prob_rotation",

    # Qualitative branch
    "event_count",
    "sentiment_event_count",
    "news_event_count",
    "event_count_log",
    "sentiment_event_count_log",
    "news_event_count_log",
    "qualitative_score",
    "qualitative_risk_score",
    "qualitative_confidence",
    "max_event_risk_score",
    "mean_event_risk_score",
    "mean_sentiment_score",
    "mean_news_impact_score",
    "mean_news_importance",

    # Agreement / conflict / availability
    "text_available",
    "branch_signal_disagreement",
    "branch_signal_agreement",
    "confidence_product",
    "high_risk_flag",
    "crisis_regime_flag",
    "low_liquidity_flag",
]

QUANT_ACTION_CATEGORIES = ["BUY", "HOLD", "SELL"]
QUAL_ACTION_CATEGORIES = ["BUY", "HOLD", "SELL"]
RISK_DRIVER_CATEGORIES = ["volatility", "drawdown", "var_cvar", "contagion", "liquidity", "regime", "unknown"]
QUAL_DRIVER_CATEGORIES = ["risk_relevance", "news_uncertainty", "sentiment", "news_impact", "event_count", "no_text_event", "unknown"]
REGIME_CATEGORIES = ["calm", "volatile", "crisis", "rotation", "unknown"]


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class UserRuleBarrierConfig:
    """
    User-controlled final rule barrier.

    These values are hard-coded into the code by default because the user asked
    not to create an external config file in outputs/.
    """

    # Exposure caps
    max_single_stock_default: float = 0.10
    conservative_max_single_stock: float = 0.05
    moderate_max_single_stock: float = 0.10
    aggressive_max_single_stock: float = 0.15

    crisis_short_horizon_cap: float = 0.05
    crisis_long_horizon_cap: float = 0.03

    high_drawdown_cap: float = 0.03
    high_contagion_cap: float = 0.02
    high_quant_risk_cap: float = 0.03
    low_liquidity_cap: float = 0.0
    not_tradable_cap: float = 0.0

    # Barrier thresholds
    min_liquidity_score: float = 0.30
    contagion_veto_threshold: float = 0.90
    drawdown_cap_threshold: float = 0.80
    quant_risk_buy_veto_threshold: float = 0.85
    severe_risk_sell_threshold: float = 0.92

    # Action thresholds
    buy_signal_threshold: float = 0.18
    sell_signal_threshold: float = -0.25
    min_confidence_for_buy: float = 0.35
    min_position_fraction: float = 0.0001

    # Fusion target construction
    base_quantitative_target_weight: float = 0.70
    base_qualitative_target_weight: float = 0.30

    # Behaviour
    allow_sell_signal: bool = True
    allow_buy_in_crisis: bool = True
    final_position_never_exceeds_position_sizing: bool = True


@dataclass
class FusionConfig:
    repo_root: str = ""

    quantitative_dir: str = "outputs/results/QuantitativeAnalyst"
    qualitative_dir: str = "outputs/results/QualitativeAnalyst"
    results_dir: str = "outputs/results/FusionEngine"
    model_dir: str = "outputs/models/FusionEngine"
    code_results_dir: str = "outputs/codeResults/FusionEngine"

    device: str = "cuda"
    seed: int = 42

    hidden_dim: int = 96
    n_layers: int = 2
    dropout: float = 0.15
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 2048
    epochs: int = 40
    early_stop_patience: int = 6
    num_workers: int = 0

    hpo_trials: int = 30
    hpo_epochs: int = 10
    hpo_max_train_rows: int = 350_000
    hpo_max_val_rows: int = 120_000
    max_train_rows: int = 0

    allow_missing_qualitative: bool = True
    strict_quantitative_attention_schema: bool = True

    exposure_mode: str = "moderate"
    horizon_mode: str = "short"

    rule_config: UserRuleBarrierConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.rule_config is None:
            self.rule_config = UserRuleBarrierConfig()

    def resolve_paths(self) -> "FusionConfig":
        if self.repo_root:
            root = Path(self.repo_root)
            for attr in ["quantitative_dir", "qualitative_dir", "results_dir", "model_dir", "code_results_dir"]:
                value = getattr(self, attr)
                if value and not Path(value).is_absolute():
                    setattr(self, attr, str(root / value))
        return self

    def to_dict(self) -> Dict[str, Any]:
        obj = asdict(self)
        obj["rule_config"] = asdict(self.rule_config)
        return obj


# ═══════════════════════════════════════════════════════════════════════════════
# GENERAL UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(json_safe(k)): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return max(sum(1 for _ in f) - 1, 0)


def safe_numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=np.float32)
    s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
    return s.astype(np.float32)


def clip01(x: Any) -> Any:
    return np.clip(x, 0.0, 1.0)


def clip11(x: Any) -> Any:
    return np.clip(x, -1.0, 1.0)


def normalise_action(value: Any) -> str:
    s = str(value).upper().strip()
    if s not in {"BUY", "HOLD", "SELL"}:
        return "HOLD"
    return s


def one_hot_series(series: pd.Series, categories: Sequence[str], prefix: str) -> pd.DataFrame:
    values = series.astype(str).str.lower().fillna("unknown")
    out = {}
    for cat in categories:
        out[f"{prefix}_{cat}"] = (values == str(cat).lower()).astype(np.float32)
    return pd.DataFrame(out, index=series.index)


def read_csv_selected(path: Path, wanted_columns: Sequence[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    available = list(header.columns)
    wanted = [c for c in wanted_columns if c in available]
    return pd.read_csv(path, usecols=wanted, low_memory=False, dtype={"ticker": str})


def model_dir(config: FusionConfig, chunk: int) -> Path:
    return Path(config.model_dir) / f"chunk{chunk}"


def best_model_path(config: FusionConfig, chunk: int) -> Path:
    return model_dir(config, chunk) / "best_model.pt"


def final_model_path(config: FusionConfig, chunk: int) -> Path:
    return model_dir(config, chunk) / "final_model.pt"


def scaler_path(config: FusionConfig, chunk: int) -> Path:
    return model_dir(config, chunk) / "scaler.npz"


def quantitative_path(config: FusionConfig, chunk: int, split: str) -> Path:
    return Path(config.quantitative_dir) / f"quantitative_analysis_chunk{chunk}_{split}.csv"


def qualitative_path(config: FusionConfig, chunk: int, split: str) -> Path:
    return Path(config.qualitative_dir) / f"qualitative_daily_chunk{chunk}_{split}.csv"


def prediction_path(config: FusionConfig, chunk: int, split: str) -> Path:
    return Path(config.results_dir) / f"fused_decisions_chunk{chunk}_{split}.csv"


def xai_path(config: FusionConfig, chunk: int, split: str) -> Path:
    return Path(config.results_dir) / "xai" / f"fused_decisions_chunk{chunk}_{split}_xai_summary.json"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_quantitative(config: FusionConfig, chunk: int, split: str) -> pd.DataFrame:
    config.resolve_paths()
    path = quantitative_path(config, chunk, split)

    if not path.exists():
        raise FileNotFoundError(f"Missing Quantitative Analyst output: {path}")

    df_head = pd.read_csv(path, nrows=0)
    columns = set(df_head.columns)

    missing_required = [c for c in REQUIRED_QUANT_COLUMNS if c not in columns]
    if missing_required:
        if config.strict_quantitative_attention_schema:
            raise RuntimeError(
                f"Quantitative output is missing required trained-attention columns: {missing_required}\n"
                f"Path: {path}\n"
                f"This usually means the file is still the old rule-based Quantitative Analyst output. "
                f"Rerun the trained attention-based Quantitative Analyst for chunk{chunk}_{split}."
            )

    old_schema = "top_risk_driver" in columns and "top_attention_risk_driver" not in columns
    if old_schema and config.strict_quantitative_attention_schema:
        raise RuntimeError(
            f"Old Quantitative Analyst schema detected at {path}.\n"
            f"Fusion requires the trained attention schema with top_attention_risk_driver and risk_attention_* columns."
        )

    wanted = list(dict.fromkeys(REQUIRED_QUANT_COLUMNS + OPTIONAL_QUANT_COLUMNS))
    df = read_csv_selected(path, wanted)

    df["ticker"] = df["ticker"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["ticker", "date"]).copy()

    if "xai_summary" in df.columns:
        df = df.rename(columns={"xai_summary": "quantitative_xai_summary"})

    df = df.drop_duplicates(["ticker", "date"], keep="last").reset_index(drop=True)

    return df


def neutral_qualitative_frame(index_df: pd.DataFrame) -> pd.DataFrame:
    q = index_df[["ticker", "date"]].copy()

    q["event_count"] = 0.0
    q["sentiment_event_count"] = 0.0
    q["news_event_count"] = 0.0
    q["qualitative_score"] = 0.0
    q["qualitative_risk_score"] = 0.5
    q["qualitative_confidence"] = 0.0
    q["qualitative_recommendation"] = "HOLD"
    q["max_event_risk_score"] = 0.5
    q["mean_event_risk_score"] = 0.5
    q["mean_sentiment_score"] = 0.0
    q["mean_news_impact_score"] = 0.0
    q["mean_news_importance"] = 0.0
    q["dominant_qualitative_driver"] = "no_text_event"
    q["qualitative_xai_summary"] = "No qualitative text event matched this ticker-date; qualitative branch kept neutral."

    return q


def load_qualitative(config: FusionConfig, chunk: int, split: str, base_index_df: pd.DataFrame) -> pd.DataFrame:
    config.resolve_paths()
    path = qualitative_path(config, chunk, split)

    if not path.exists():
        if not config.allow_missing_qualitative:
            raise FileNotFoundError(f"Missing Qualitative Analyst output: {path}")
        print(f"  Qualitative output missing for chunk{chunk}_{split}; using neutral no-text branch.")
        return neutral_qualitative_frame(base_index_df)

    df = read_csv_selected(path, QUAL_COLUMNS)

    if "ticker" not in df.columns or "date" not in df.columns:
        raise RuntimeError(f"Qualitative file must contain ticker and date: {path}")

    df["ticker"] = df["ticker"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["ticker", "date"]).copy()

    if "xai_summary" in df.columns:
        df = df.rename(columns={"xai_summary": "qualitative_xai_summary"})

    for col in QUAL_COLUMNS:
        if col in ["ticker", "date", "xai_summary"]:
            continue
        if col not in df.columns:
            if col in ["qualitative_recommendation", "dominant_qualitative_driver"]:
                df[col] = "unknown"
            else:
                df[col] = 0.0

    if "qualitative_xai_summary" not in df.columns:
        df["qualitative_xai_summary"] = ""

    # Qualitative is daily and sparse. Deduplicate safely.
    df = df.drop_duplicates(["ticker", "date"], keep="last").reset_index(drop=True)

    return df


def merge_branches(config: FusionConfig, chunk: int, split: str) -> pd.DataFrame:
    quant = load_quantitative(config, chunk, split)
    qual = load_qualitative(config, chunk, split, quant[["ticker", "date"]])

    merged = quant.merge(
        qual,
        on=["ticker", "date"],
        how="left",
        suffixes=("", "_qual"),
    )

    # If qualitative exists but did not match a quantitative ticker-date, neutralise missing values.
    neutral = neutral_qualitative_frame(merged[["ticker", "date"]])
    for col in neutral.columns:
        if col in ["ticker", "date"]:
            continue
        if col not in merged.columns:
            merged[col] = neutral[col].values
        else:
            if merged[col].dtype == object:
                merged[col] = merged[col].fillna(neutral[col])
            else:
                merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(neutral[col])

    return merged.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING AND TARGET CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_fusion_dataframe(df: pd.DataFrame, config: FusionConfig) -> pd.DataFrame:
    out = df.copy()

    # Standardise basic columns.
    out["quantitative_recommendation"] = out.get("quantitative_recommendation", "HOLD")
    out["quantitative_recommendation"] = out["quantitative_recommendation"].map(normalise_action)

    out["qualitative_recommendation"] = out.get("qualitative_recommendation", "HOLD")
    out["qualitative_recommendation"] = out["qualitative_recommendation"].map(normalise_action)

    out["top_attention_risk_driver"] = out.get("top_attention_risk_driver", "unknown").astype(str).str.lower()
    out["dominant_qualitative_driver"] = out.get("dominant_qualitative_driver", "unknown").astype(str).str.lower()
    out["regime_label"] = out.get("regime_label", "unknown").astype(str).str.lower()

    # Numeric defaults.
    numeric_defaults = {
        "risk_adjusted_quantitative_signal": 0.0,
        "technical_direction_score": 0.0,
        "quantitative_risk_score": 0.5,
        "quantitative_confidence": 0.0,
        "quantitative_action_strength": 0.0,
        "recommended_capital_fraction": 0.0,
        "recommended_capital_pct": 0.0,
        "position_fraction_of_max": 0.0,
        "max_single_stock_exposure": config.rule_config.max_single_stock_default,
        "attention_pooled_risk_score": 0.5,

        "trend_score": 0.5,
        "momentum_score": 0.5,
        "timing_confidence": 0.5,
        "technical_confidence": 0.5,

        "volatility_risk_score": 0.5,
        "drawdown_risk_score": 0.5,
        "var_cvar_risk_score": 0.5,
        "contagion_risk_score": 0.5,
        "liquidity_risk_score": 0.5,
        "regime_risk_score": 0.5,
        "combined_risk_score": 0.5,
        "regime_confidence": 0.0,
        "liquidity_score": 0.5,
        "tradable": 1.0,

        "contagion_5d": 0.5,
        "contagion_20d": 0.5,
        "contagion_60d": 0.5,
        "expected_drawdown_10d": 0.0,
        "expected_drawdown_30d": 0.0,
        "drawdown_risk_10d": 0.5,
        "drawdown_risk_30d": 0.5,
        "vol_10d": 0.0,
        "vol_30d": 0.0,
        "var_95": 0.0,
        "var_99": 0.0,
        "cvar_95": 0.0,
        "cvar_99": 0.0,

        "prob_calm": 0.0,
        "prob_volatile": 0.0,
        "prob_crisis": 0.0,
        "prob_rotation": 0.0,

        "event_count": 0.0,
        "sentiment_event_count": 0.0,
        "news_event_count": 0.0,
        "qualitative_score": 0.0,
        "qualitative_risk_score": 0.5,
        "qualitative_confidence": 0.0,
        "max_event_risk_score": 0.5,
        "mean_event_risk_score": 0.5,
        "mean_sentiment_score": 0.0,
        "mean_news_impact_score": 0.0,
        "mean_news_importance": 0.0,
    }

    for col in RISK_ATTENTION_COLUMNS:
        numeric_defaults[col] = 1.0 / len(RISK_ATTENTION_COLUMNS)

    for col, default in numeric_defaults.items():
        out[col] = safe_numeric(out, col, default)

    # Clip where meaningful.
    bounded_01 = [
        "quantitative_risk_score",
        "quantitative_confidence",
        "quantitative_action_strength",
        "recommended_capital_fraction",
        "position_fraction_of_max",
        "max_single_stock_exposure",
        "attention_pooled_risk_score",
        *RISK_ATTENTION_COLUMNS,
        "trend_score",
        "momentum_score",
        "timing_confidence",
        "technical_confidence",
        "volatility_risk_score",
        "drawdown_risk_score",
        "var_cvar_risk_score",
        "contagion_risk_score",
        "liquidity_risk_score",
        "regime_risk_score",
        "combined_risk_score",
        "regime_confidence",
        "liquidity_score",
        "tradable",
        "contagion_5d",
        "contagion_20d",
        "contagion_60d",
        "drawdown_risk_10d",
        "drawdown_risk_30d",
        "prob_calm",
        "prob_volatile",
        "prob_crisis",
        "prob_rotation",
        "qualitative_risk_score",
        "qualitative_confidence",
        "max_event_risk_score",
        "mean_event_risk_score",
        "mean_news_importance",
    ]

    for col in bounded_01:
        if col in out.columns:
            out[col] = clip01(out[col].values).astype(np.float32)

    out["risk_adjusted_quantitative_signal"] = clip11(out["risk_adjusted_quantitative_signal"].values).astype(np.float32)
    out["technical_direction_score"] = clip11(out["technical_direction_score"].values).astype(np.float32)
    out["qualitative_score"] = clip11(out["qualitative_score"].values).astype(np.float32)
    out["mean_sentiment_score"] = clip11(out["mean_sentiment_score"].values).astype(np.float32)
    out["mean_news_impact_score"] = clip11(out["mean_news_impact_score"].values).astype(np.float32)

    # Derived features.
    out["event_count_log"] = np.log1p(np.maximum(0.0, out["event_count"].values)).astype(np.float32)
    out["sentiment_event_count_log"] = np.log1p(np.maximum(0.0, out["sentiment_event_count"].values)).astype(np.float32)
    out["news_event_count_log"] = np.log1p(np.maximum(0.0, out["news_event_count"].values)).astype(np.float32)

    out["text_available"] = (
        (out["event_count"].values > 0)
        & (out["qualitative_confidence"].values > 0.0)
    ).astype(np.float32)

    out["branch_signal_disagreement"] = np.abs(
        out["risk_adjusted_quantitative_signal"].values - out["qualitative_score"].values
    ).astype(np.float32)

    out["branch_signal_agreement"] = (1.0 - np.minimum(1.0, out["branch_signal_disagreement"].values)).astype(np.float32)

    out["confidence_product"] = (
        out["quantitative_confidence"].values * out["qualitative_confidence"].values
    ).astype(np.float32)

    out["high_risk_flag"] = (
        out["quantitative_risk_score"].values >= config.rule_config.quant_risk_buy_veto_threshold
    ).astype(np.float32)

    out["crisis_regime_flag"] = (
        (out["regime_label"].astype(str).str.lower() == "crisis")
        | (out["prob_crisis"].values >= 0.50)
    ).astype(np.float32)

    out["low_liquidity_flag"] = (
        out["liquidity_score"].values < config.rule_config.min_liquidity_score
    ).astype(np.float32)

    return out


def derive_target_branch_weights(df: pd.DataFrame, config: FusionConfig) -> np.ndarray:
    """
    Target branch weights for auxiliary training.

    Quantitative branch should dominate when:
        - qualitative data is missing,
        - qualitative confidence is low,
        - event count is very small.

    Qualitative branch receives more weight when:
        - there are actual text events,
        - qualitative confidence is high,
        - event_count is meaningful.
    """

    q_conf = clip01(df["quantitative_confidence"].values)
    qual_conf = clip01(df["qualitative_confidence"].values)
    text_available = clip01(df["text_available"].values)

    event_factor = np.log1p(np.maximum(0.0, df["event_count"].values)) / np.log1p(250.0)
    event_factor = clip01(event_factor)

    qual_weight = (
        config.rule_config.base_qualitative_target_weight
        * qual_conf
        * event_factor
        * text_available
    )

    # Increase quant weight when quant confidence is strong.
    quant_weight = config.rule_config.base_quantitative_target_weight * (0.50 + 0.50 * q_conf)

    total = np.maximum(quant_weight + qual_weight, 1e-8)
    quant_weight = quant_weight / total
    qual_weight = qual_weight / total

    return np.stack([quant_weight, qual_weight], axis=1).astype(np.float32)


def construct_fusion_targets(df: pd.DataFrame, config: FusionConfig) -> pd.DataFrame:
    out = df.copy()

    branch_w = derive_target_branch_weights(out, config)
    q_w = branch_w[:, 0]
    qual_w = branch_w[:, 1]

    q_signal = out["risk_adjusted_quantitative_signal"].values.astype(np.float32)
    qual_signal = out["qualitative_score"].values.astype(np.float32)

    raw_signal = q_w * q_signal + qual_w * qual_signal

    q_risk = out["quantitative_risk_score"].values.astype(np.float32)
    qual_risk = out["qualitative_risk_score"].values.astype(np.float32)
    qual_conf = out["qualitative_confidence"].values.astype(np.float32)

    target_risk = clip01(
        0.75 * q_risk
        + 0.25 * ((qual_conf * qual_risk) + ((1.0 - qual_conf) * q_risk))
    )

    # Risk penalty should suppress bullishness more than bearishness.
    risk_penalty = clip01((target_risk - 0.50) / 0.50)
    adjusted_signal = raw_signal.copy()
    bullish_mask = adjusted_signal > 0
    adjusted_signal[bullish_mask] = adjusted_signal[bullish_mask] * (1.0 - 0.50 * risk_penalty[bullish_mask])

    # If risk is very high and signal is weak, push toward neutral/sell.
    severe = target_risk >= config.rule_config.severe_risk_sell_threshold
    adjusted_signal[severe & (adjusted_signal > 0)] *= 0.25

    target_signal = clip11(adjusted_signal)

    q_conf = out["quantitative_confidence"].values.astype(np.float32)
    confidence_product = out["confidence_product"].values.astype(np.float32)
    target_conf = clip01(
        0.65 * q_conf
        + 0.25 * qual_conf
        + 0.10 * confidence_product
    )

    recommended = clip01(out["recommended_capital_fraction"].values.astype(np.float32))
    position_target = recommended * clip01((target_signal + 1.0) / 2.0) * clip01(1.0 - target_risk)
    position_target = np.minimum(position_target, recommended)
    position_target = clip01(position_target)

    action = np.full(len(out), "HOLD", dtype=object)

    buy = (
        (target_signal > config.rule_config.buy_signal_threshold)
        & (target_risk < config.rule_config.quant_risk_buy_veto_threshold)
        & (target_conf >= config.rule_config.min_confidence_for_buy)
        & (recommended > config.rule_config.min_position_fraction)
    )

    sell = (
        (target_signal < config.rule_config.sell_signal_threshold)
        | ((target_risk >= config.rule_config.severe_risk_sell_threshold) & (target_signal < 0.0))
    )

    action[buy] = "BUY"
    action[sell] = "SELL"

    out["target_fusion_signal"] = target_signal.astype(np.float32)
    out["target_fusion_risk"] = target_risk.astype(np.float32)
    out["target_fusion_confidence"] = target_conf.astype(np.float32)
    out["target_position_fraction"] = position_target.astype(np.float32)
    out["target_quantitative_weight"] = branch_w[:, 0].astype(np.float32)
    out["target_qualitative_weight"] = branch_w[:, 1].astype(np.float32)
    out["target_action"] = action
    out["target_action_id"] = [ACTION_TO_ID[str(a)] for a in action]

    return out


def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    numeric = df[BASE_NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

    onehot_parts = [
        one_hot_series(df["quantitative_recommendation"], QUANT_ACTION_CATEGORIES, "q_action"),
        one_hot_series(df["qualitative_recommendation"], QUAL_ACTION_CATEGORIES, "qual_action"),
        one_hot_series(df["top_attention_risk_driver"], RISK_DRIVER_CATEGORIES, "risk_driver"),
        one_hot_series(df["dominant_qualitative_driver"], QUAL_DRIVER_CATEGORIES, "qual_driver"),
        one_hot_series(df["regime_label"], REGIME_CATEGORIES, "regime"),
    ]

    feat_df = pd.concat([numeric] + onehot_parts, axis=1)
    feature_names = list(feat_df.columns)

    x = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    return x, feature_names


def prepare_training_arrays(
    df: pd.DataFrame,
    config: FusionConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    df = prepare_fusion_dataframe(df, config)
    df = construct_fusion_targets(df, config)

    x, feature_names = build_feature_matrix(df)

    aux = df[["risk_adjusted_quantitative_signal", "qualitative_score"]].values.astype(np.float32)

    y_action = df["target_action_id"].values.astype(np.int64)

    y_reg = df[
        [
            "target_fusion_signal",
            "target_fusion_risk",
            "target_fusion_confidence",
            "target_position_fraction",
        ]
    ].values.astype(np.float32)

    y_weights = df[["target_quantitative_weight", "target_qualitative_weight"]].values.astype(np.float32)

    return x, aux, y_action, y_reg, y_weights, np.arange(len(df)), df, feature_names


def sample_arrays(
    x: np.ndarray,
    aux: np.ndarray,
    y_action: np.ndarray,
    y_reg: np.ndarray,
    y_weights: np.ndarray,
    df: pd.DataFrame,
    max_rows: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    if max_rows <= 0 or len(df) <= max_rows:
        return x, aux, y_action, y_reg, y_weights, df

    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(df), size=int(max_rows), replace=False)
    idx = np.sort(idx)

    return (
        x[idx],
        aux[idx],
        y_action[idx],
        y_reg[idx],
        y_weights[idx],
        df.iloc[idx].reset_index(drop=True),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SCALER
# ═══════════════════════════════════════════════════════════════════════════════

class FusionScaler:
    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.feature_names: List[str] = []

    def fit(self, x: np.ndarray, feature_names: List[str]) -> None:
        self.mean = x.mean(axis=0, keepdims=True).astype(np.float32)
        self.std = x.std(axis=0, keepdims=True).astype(np.float32)
        self.std[self.std < 1e-8] = 1.0
        self.feature_names = list(feature_names)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return x.astype(np.float32)
        return ((x - self.mean) / self.std).astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(path),
            mean=self.mean,
            std=self.std,
            feature_names=np.array(self.feature_names, dtype=object),
        )

    @classmethod
    def load(cls, path: Path) -> "FusionScaler":
        data = np.load(str(path), allow_pickle=True)
        obj = cls()
        obj.mean = data["mean"].astype(np.float32)
        obj.std = data["std"].astype(np.float32)
        obj.feature_names = [str(x) for x in data["feature_names"].tolist()]
        return obj


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class HybridFusionModel(nn.Module):
    """
    Learned fusion model.

    It does not directly have final authority.
    Its output must pass through the user rule barrier.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        n_layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        dim = int(input_dim)

        for _ in range(int(n_layers)):
            layers.append(nn.Linear(dim, int(hidden_dim)))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(float(dropout)))
            dim = int(hidden_dim)

        self.backbone = nn.Sequential(*layers)

        self.branch_weight_head = nn.Linear(dim, 2)
        self.action_head = nn.Linear(dim, 3)

        self.signal_head = nn.Linear(dim, 1)
        self.risk_head = nn.Linear(dim, 1)
        self.confidence_head = nn.Linear(dim, 1)
        self.position_head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, aux_signals: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        aux_signals = torch.nan_to_num(aux_signals.float(), nan=0.0, posinf=0.0, neginf=0.0)

        h = self.backbone(x)

        branch_weights = torch.softmax(self.branch_weight_head(h), dim=1)

        q_signal = aux_signals[:, 0]
        qual_signal = aux_signals[:, 1]

        blended_signal = branch_weights[:, 0] * q_signal + branch_weights[:, 1] * qual_signal
        residual_signal = torch.tanh(self.signal_head(h)).squeeze(1)

        learned_signal = torch.tanh(0.75 * blended_signal + 0.25 * residual_signal)
        learned_risk = torch.sigmoid(self.risk_head(h)).squeeze(1)
        learned_confidence = torch.sigmoid(self.confidence_head(h)).squeeze(1)
        learned_position_multiplier = torch.sigmoid(self.position_head(h)).squeeze(1)

        action_logits = self.action_head(h)

        outputs = torch.stack(
            [
                learned_signal,
                learned_risk,
                learned_confidence,
                learned_position_multiplier,
            ],
            dim=1,
        )

        return {
            "outputs": outputs,
            "action_logits": action_logits,
            "branch_weights": branch_weights,
            "hidden": h,
        }


def fusion_loss(
    output: Dict[str, torch.Tensor],
    y_action: torch.Tensor,
    y_reg: torch.Tensor,
    y_weights: torch.Tensor,
) -> torch.Tensor:
    logits = output["action_logits"]
    reg = output["outputs"]
    weights = output["branch_weights"]

    ce = nn.functional.cross_entropy(logits, y_action)
    reg_loss = torch.mean((reg - y_reg) ** 2)
    weight_loss = torch.mean((weights - y_weights) ** 2)

    # Position is important, but class + signal/risk should dominate.
    return 0.45 * ce + 0.45 * reg_loss + 0.10 * weight_loss


def save_fusion_model(
    model: HybridFusionModel,
    scaler: FusionScaler,
    config: FusionConfig,
    chunk: int,
    best_val_loss: float,
    feature_names: List[str],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": model.state_dict(),
        "config": config.to_dict(),
        "input_dim": int(len(feature_names)),
        "feature_names": feature_names,
        "best_val_loss": float(best_val_loss),
        "action_to_id": ACTION_TO_ID,
        "id_to_action": ID_TO_ACTION,
    }

    torch.save(payload, path)


def load_fusion_model(
    config: FusionConfig,
    chunk: int,
) -> Tuple[HybridFusionModel, FusionScaler, Dict[str, Any]]:
    config.resolve_paths()

    path = best_model_path(config, chunk)
    if not path.exists():
        raise FileNotFoundError(f"Missing Fusion model: {path}")

    payload = torch.load(path, map_location=config.device)
    model_cfg = payload.get("config", {})

    model = HybridFusionModel(
        input_dim=int(payload["input_dim"]),
        hidden_dim=int(model_cfg.get("hidden_dim", config.hidden_dim)),
        n_layers=int(model_cfg.get("n_layers", config.n_layers)),
        dropout=float(model_cfg.get("dropout", config.dropout)),
    ).to(config.device)

    model.load_state_dict(payload["state_dict"])
    model.eval()

    scaler = FusionScaler.load(scaler_path(config, chunk))

    return model, scaler, payload


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def make_loader(
    x: np.ndarray,
    aux: np.ndarray,
    y_action: np.ndarray,
    y_reg: np.ndarray,
    y_weights: np.ndarray,
    batch_size: int,
    shuffle: bool,
    device: str,
    num_workers: int,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(aux.astype(np.float32)),
        torch.from_numpy(y_action.astype(np.int64)),
        torch.from_numpy(y_reg.astype(np.float32)),
        torch.from_numpy(y_weights.astype(np.float32)),
    )

    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=str(device).startswith("cuda"),
        drop_last=False,
    )


def train_epoch(
    model: HybridFusionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()
    total = 0.0
    n = 0

    for x, aux, y_action, y_reg, y_weights in loader:
        x = x.to(device, non_blocking=True)
        aux = aux.to(device, non_blocking=True)
        y_action = y_action.to(device, non_blocking=True)
        y_reg = y_reg.to(device, non_blocking=True)
        y_weights = y_weights.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        out = model(x, aux)
        loss = fusion_loss(out, y_action, y_reg, y_weights)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite Fusion training loss detected: {float(loss.detach().cpu())}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        bs = x.size(0)
        total += float(loss.detach().cpu()) * bs
        n += bs

    return total / max(n, 1)


@torch.no_grad()
def validate_epoch(
    model: HybridFusionModel,
    loader: DataLoader,
    device: str,
) -> Tuple[float, float]:
    model.eval()
    total = 0.0
    correct = 0
    n = 0

    for x, aux, y_action, y_reg, y_weights in loader:
        x = x.to(device, non_blocking=True)
        aux = aux.to(device, non_blocking=True)
        y_action = y_action.to(device, non_blocking=True)
        y_reg = y_reg.to(device, non_blocking=True)
        y_weights = y_weights.to(device, non_blocking=True)

        out = model(x, aux)
        loss = fusion_loss(out, y_action, y_reg, y_weights)

        pred = out["action_logits"].argmax(dim=1)
        correct += int((pred == y_action).sum().detach().cpu())
        bs = x.size(0)
        total += float(loss.detach().cpu()) * bs
        n += bs

    return total / max(n, 1), correct / max(n, 1)


def train_fusion_model(
    config: FusionConfig,
    chunk: int,
    *,
    fresh: bool = False,
    hpo_mode: bool = False,
    run_tag: str = "",
) -> Tuple[HybridFusionModel, float, Dict[str, Any]]:
    config.resolve_paths()
    set_seed(config.seed)

    out_dir = model_dir(config, chunk)

    if fresh and out_dir.exists() and not hpo_mode:
        print(f"  Fresh run requested. Removing: {out_dir}")
        shutil.rmtree(out_dir)

    train_df = merge_branches(config, chunk, "train")
    val_df = merge_branches(config, chunk, "val")

    x_train, aux_train, y_action_train, y_reg_train, y_weights_train, _, train_df, feature_names = prepare_training_arrays(train_df, config)
    x_val, aux_val, y_action_val, y_reg_val, y_weights_val, _, val_df, feature_names_val = prepare_training_arrays(val_df, config)

    if feature_names != feature_names_val:
        raise RuntimeError("Train/val feature schemas do not match.")

    max_train = int(config.hpo_max_train_rows if hpo_mode else config.max_train_rows)
    max_val = int(config.hpo_max_val_rows if hpo_mode else 0)

    x_train, aux_train, y_action_train, y_reg_train, y_weights_train, train_df = sample_arrays(
        x_train, aux_train, y_action_train, y_reg_train, y_weights_train, train_df, max_train, config.seed
    )

    x_val, aux_val, y_action_val, y_reg_val, y_weights_val, val_df = sample_arrays(
        x_val, aux_val, y_action_val, y_reg_val, y_weights_val, val_df, max_val, config.seed + 1
    )

    scaler = FusionScaler()
    scaler.fit(x_train, feature_names)

    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)

    model = HybridFusionModel(
        input_dim=x_train.shape[1],
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        dropout=config.dropout,
    ).to(config.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )

    train_loader = make_loader(
        x_train,
        aux_train,
        y_action_train,
        y_reg_train,
        y_weights_train,
        config.batch_size,
        shuffle=True,
        device=config.device,
        num_workers=config.num_workers,
    )

    val_loader = make_loader(
        x_val,
        aux_val,
        y_action_val,
        y_reg_val,
        y_weights_val,
        config.batch_size,
        shuffle=False,
        device=config.device,
        num_workers=config.num_workers,
    )

    epochs = int(config.hpo_epochs if hpo_mode else config.epochs)

    print(f"  Train rows: {len(train_df):,} | Val rows: {len(val_df):,}")
    print(f"  Features: {x_train.shape[1]:,}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"  Config: hidden={config.hidden_dim}, layers={config.n_layers}, "
        f"dropout={config.dropout:.3f}, batch={config.batch_size}"
    )

    print("  Train target actions:", pd.Series(y_action_train).map(ID_TO_ACTION).value_counts().to_dict())
    print("  Val target actions:  ", pd.Series(y_action_val).map(ID_TO_ACTION).value_counts().to_dict())

    best_val = float("inf")
    best_state = None
    no_improve = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, config.device)
        val_loss, val_acc = validate_epoch(model, val_loader, config.device)

        history.append({
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_action_accuracy": float(val_acc),
            "lr": float(optimizer.param_groups[0]["lr"]),
        })

        prefix = f"[{run_tag}]" if run_tag else f"[chunk{chunk}]"
        print(f"  {prefix} E{epoch:03d}/{epochs} | train={train_loss:.6f} | val={val_loss:.6f} | acc={val_acc:.4f}")

        if val_loss < best_val - 1e-8:
            best_val = float(val_loss)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= int(config.early_stop_patience):
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if not hpo_mode:
        out_dir.mkdir(parents=True, exist_ok=True)

        save_fusion_model(model, scaler, config, chunk, best_val, feature_names, best_model_path(config, chunk))
        save_fusion_model(model, scaler, config, chunk, best_val, feature_names, final_model_path(config, chunk))
        scaler.save(scaler_path(config, chunk))

        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

        freeze_dir = out_dir / "model_freezed"
        unfreeze_dir = out_dir / "model_unfreezed"
        freeze_dir.mkdir(parents=True, exist_ok=True)
        unfreeze_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(best_model_path(config, chunk), freeze_dir / "model.pt")
        shutil.copy2(best_model_path(config, chunk), unfreeze_dir / "model.pt")

        summary_path = out_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(json_safe({
                "chunk": int(chunk),
                "best_val_loss": float(best_val),
                "history": history,
                "config": config.to_dict(),
                "feature_names": feature_names,
            }), f, indent=2)

    summary = {
        "chunk": int(chunk),
        "best_val_loss": float(best_val),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "features": int(x_train.shape[1]),
        "history": history,
        "config": config.to_dict(),
    }

    return model, float(best_val), summary


# ═══════════════════════════════════════════════════════════════════════════════
# HPO
# ═══════════════════════════════════════════════════════════════════════════════

def hpo_objective(trial: Any, base_config: FusionConfig, chunk: int) -> float:
    cfg = FusionConfig(**{
        k: v for k, v in base_config.to_dict().items()
        if k != "rule_config"
    })
    cfg.rule_config = UserRuleBarrierConfig(**base_config.to_dict()["rule_config"])
    cfg.resolve_paths()

    cfg.hidden_dim = trial.suggest_categorical("hidden_dim", [48, 64, 96, 128, 192])
    cfg.n_layers = trial.suggest_int("n_layers", 1, 3)
    cfg.dropout = trial.suggest_float("dropout", 0.05, 0.40)
    cfg.lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    cfg.weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
    cfg.batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048, 4096])
    cfg.hpo_epochs = int(base_config.hpo_epochs)
    cfg.early_stop_patience = 3

    try:
        _, val_loss, _ = train_fusion_model(
            cfg,
            chunk,
            fresh=False,
            hpo_mode=True,
            run_tag=f"hpo_trial_{trial.number:04d}",
        )

        if not np.isfinite(val_loss):
            return 1e9

        return float(val_loss)

    except Exception as exc:
        print(f"  Trial {trial.number} failed safely: {exc}")
        return 1e9


def run_hpo(config: FusionConfig, chunk: int, trials: int, fresh: bool = False) -> Dict[str, Any]:
    if optuna is None:
        raise RuntimeError("Optuna is not installed, but HPO was requested.")

    config.resolve_paths()

    out_dir = Path(config.code_results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = out_dir / f"hpo_chunk{chunk}.db"
    if fresh and db_path.exists():
        db_path.unlink()
        print(f"  Deleted old HPO DB: {db_path}")

    storage = f"sqlite:///{db_path}"
    study_name = f"fusion_engine_chunk{chunk}"

    sampler = optuna.samplers.TPESampler(seed=config.seed)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        direction="minimize",
        load_if_exists=not fresh,
    )

    study.optimize(
        lambda trial: hpo_objective(trial, config, chunk),
        n_trials=int(trials),
        show_progress_bar=True,
    )

    valid_trials = [t for t in study.trials if t.value is not None and np.isfinite(t.value) and t.value < 1e8]
    if not valid_trials:
        raise RuntimeError("All Fusion HPO trials failed.")

    best = study.best_trial

    result = {
        "study_name": study_name,
        "best_value": float(best.value),
        "best_params": best.params,
        "trials": len(study.trials),
        "storage": storage,
    }

    path = out_dir / f"best_params_chunk{chunk}.json"
    with open(path, "w") as f:
        json.dump(json_safe(result), f, indent=2)

    print(f"  Best HPO: {best.params} (val_loss={best.value:.6f})")
    print(f"  Saved: {path}")

    return result


def load_best_params(config: FusionConfig, chunk: int) -> Optional[Dict[str, Any]]:
    path = Path(config.code_results_dir) / f"best_params_chunk{chunk}.json"
    if not path.exists():
        return None
    with open(path) as f:
        obj = json.load(f)
    return obj.get("best_params", obj)


def apply_best_params(config: FusionConfig, params: Dict[str, Any]) -> FusionConfig:
    for key, value in params.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION AND RULE BARRIER
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_batches(
    model: HybridFusionModel,
    x: np.ndarray,
    aux: np.ndarray,
    config: FusionConfig,
) -> Dict[str, np.ndarray]:
    ds = TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(aux.astype(np.float32)),
    )

    loader = DataLoader(
        ds,
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        pin_memory=str(config.device).startswith("cuda"),
        drop_last=False,
    )

    outputs = []
    logits = []
    weights = []

    model.eval()

    for xb, ab in loader:
        xb = xb.to(config.device, non_blocking=True)
        ab = ab.to(config.device, non_blocking=True)

        out = model(xb, ab)

        outputs.append(out["outputs"].detach().cpu().numpy())
        logits.append(out["action_logits"].detach().cpu().numpy())
        weights.append(out["branch_weights"].detach().cpu().numpy())

    return {
        "outputs": np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 4), dtype=np.float32),
        "action_logits": np.concatenate(logits, axis=0) if logits else np.zeros((0, 3), dtype=np.float32),
        "branch_weights": np.concatenate(weights, axis=0) if weights else np.zeros((0, 2), dtype=np.float32),
    }


def apply_user_rule_barrier(df: pd.DataFrame, config: FusionConfig) -> pd.DataFrame:
    """
    Final rule barrier.

    The learned model proposes an action and position.
    This layer applies user-defined constraints.
    """

    out = df.copy()
    rules = config.rule_config

    recommended = clip01(safe_numeric(out, "recommended_capital_fraction", 0.0).values)
    learned_multiplier = clip01(safe_numeric(out, "learned_position_multiplier", 0.0).values)
    learned_position = recommended * learned_multiplier

    base_cap = safe_numeric(out, "max_single_stock_exposure", rules.max_single_stock_default).values
    base_cap = np.where(np.isfinite(base_cap), base_cap, rules.max_single_stock_default)
    base_cap = clip01(base_cap)

    # Enforce selected exposure profile.
    if config.exposure_mode == "conservative":
        profile_cap = rules.conservative_max_single_stock
    elif config.exposure_mode == "aggressive":
        profile_cap = rules.aggressive_max_single_stock
    else:
        profile_cap = rules.moderate_max_single_stock

    cap = np.minimum(base_cap, profile_cap).astype(np.float32)

    regime_label = out.get("regime_label", pd.Series("unknown", index=out.index)).astype(str).str.lower()
    crisis = (
        (regime_label == "crisis")
        | (safe_numeric(out, "prob_crisis", 0.0).values >= 0.50)
    )

    crisis_cap = rules.crisis_long_horizon_cap if config.horizon_mode == "long" else rules.crisis_short_horizon_cap
    cap = np.where(crisis, np.minimum(cap, crisis_cap), cap)

    drawdown_risk = clip01(safe_numeric(out, "drawdown_risk_score", 0.5).values)
    contagion_risk = clip01(safe_numeric(out, "contagion_risk_score", 0.5).values)
    quant_risk = clip01(safe_numeric(out, "learned_fusion_risk_score", 0.5).values)
    liquidity_score = clip01(safe_numeric(out, "liquidity_score", 0.5).values)
    tradable = clip01(safe_numeric(out, "tradable", 1.0).values)

    high_drawdown = drawdown_risk >= rules.drawdown_cap_threshold
    high_contagion = contagion_risk >= rules.contagion_veto_threshold
    high_quant_risk = quant_risk >= rules.quant_risk_buy_veto_threshold
    low_liquidity = liquidity_score < rules.min_liquidity_score
    not_tradable = tradable < 0.5

    cap = np.where(high_drawdown, np.minimum(cap, rules.high_drawdown_cap), cap)
    cap = np.where(high_contagion, np.minimum(cap, rules.high_contagion_cap), cap)
    cap = np.where(high_quant_risk, np.minimum(cap, rules.high_quant_risk_cap), cap)
    cap = np.where(low_liquidity, np.minimum(cap, rules.low_liquidity_cap), cap)
    cap = np.where(not_tradable, np.minimum(cap, rules.not_tradable_cap), cap)

    if rules.final_position_never_exceeds_position_sizing:
        final_position = np.minimum.reduce([recommended, learned_position, cap])
    else:
        final_position = np.minimum(learned_position, cap)

    signal = safe_numeric(out, "learned_fusion_signal", 0.0).values
    conf = clip01(safe_numeric(out, "learned_fusion_confidence", 0.0).values)

    learned_action = out["learned_recommendation"].astype(str).str.upper().values
    final_action = learned_action.copy()

    # First normalise weak learned decisions using signal/confidence thresholds.
    weak_buy = (
        (final_action == "BUY")
        & (
            (signal < rules.buy_signal_threshold)
            | (conf < rules.min_confidence_for_buy)
            | (final_position <= rules.min_position_fraction)
        )
    )
    final_action[weak_buy] = "HOLD"

    buy_condition = (
        (signal > rules.buy_signal_threshold)
        & (conf >= rules.min_confidence_for_buy)
        & (final_position > rules.min_position_fraction)
        & (quant_risk < rules.quant_risk_buy_veto_threshold)
    )
    final_action[(final_action == "HOLD") & buy_condition] = "BUY"

    sell_condition = (
        (signal < rules.sell_signal_threshold)
        | ((quant_risk >= rules.severe_risk_sell_threshold) & (signal < 0.0))
    )
    if rules.allow_sell_signal:
        final_action[sell_condition] = "SELL"

    # Hard vetoes.
    final_action[not_tradable] = "HOLD"
    final_action[low_liquidity] = "HOLD"

    # Risk vetoes disallow BUY, but do not necessarily force SELL unless the signal is negative.
    risky_buy = (final_action == "BUY") & (high_contagion | high_quant_risk | high_drawdown)
    final_action[risky_buy] = "HOLD"

    if not rules.allow_buy_in_crisis:
        final_action[(final_action == "BUY") & crisis] = "HOLD"

    # SELL means exit/reduce exposure in this framework.
    final_position = np.where(final_action == "SELL", 0.0, final_position)
    final_position = clip01(final_position)

    out["user_rule_cap_fraction"] = cap.astype(np.float32)
    out["pre_rule_learned_position_fraction"] = learned_position.astype(np.float32)
    out["final_position_fraction"] = final_position.astype(np.float32)
    out["final_position_pct"] = (100.0 * final_position).astype(np.float32)
    out["final_recommendation"] = final_action
    out["rule_changed_action"] = (out["learned_recommendation"].astype(str).str.upper().values != final_action).astype(np.int32)

    reasons: List[str] = []
    for i in range(len(out)):
        r: List[str] = []

        if not_tradable[i]:
            r.append("not_tradable")
        if low_liquidity[i]:
            r.append("low_liquidity_block")
        if crisis[i]:
            r.append(f"crisis_cap_{crisis_cap:.2f}")
        if high_contagion[i]:
            r.append("high_contagion_cap_or_veto")
        if high_drawdown[i]:
            r.append("high_drawdown_cap")
        if high_quant_risk[i]:
            r.append("high_quantitative_risk_cap_or_veto")
        if final_position[i] < recommended[i] - 1e-9:
            r.append("position_reduced_by_rule_barrier")
        if out["rule_changed_action"].iloc[i] == 1:
            r.append("action_changed_by_rule_barrier")

        reasons.append("; ".join(r) if r else "no_rule_override")

    out["rule_barrier_reasons"] = reasons

    return out


def build_fusion_xai_row(row: pd.Series) -> str:
    return (
        f"{row['final_recommendation']}: final_signal={row['final_fusion_signal']:.3f}, "
        f"risk={row['final_fusion_risk_score']:.3f}, confidence={row['final_fusion_confidence']:.3f}, "
        f"q_weight={row['learned_quantitative_weight']:.3f}, qual_weight={row['learned_qualitative_weight']:.3f}, "
        f"top_quant_risk={row.get('top_attention_risk_driver', 'unknown')}, "
        f"text_available={int(row.get('text_available', 0))}, "
        f"position={row['final_position_pct']:.2f}%, rules={row['rule_barrier_reasons']}."
    )


def predict_fusion(config: FusionConfig, chunk: int, split: str) -> Dict[str, Any]:
    config.resolve_paths()

    model, scaler, payload = load_fusion_model(config, chunk)

    df_raw = merge_branches(config, chunk, split)
    df_prepared = prepare_fusion_dataframe(df_raw, config)

    x, feature_names = build_feature_matrix(df_prepared)

    if list(feature_names) != list(payload["feature_names"]):
        raise RuntimeError(
            "Prediction feature schema does not match trained Fusion model.\n"
            f"Expected {len(payload['feature_names'])} features, got {len(feature_names)}."
        )

    x_scaled = scaler.transform(x)
    aux = df_prepared[["risk_adjusted_quantitative_signal", "qualitative_score"]].values.astype(np.float32)

    pred = predict_batches(model, x_scaled, aux, config)

    outputs = pred["outputs"]
    logits = pred["action_logits"]
    branch_weights = pred["branch_weights"]

    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    pred_id = probs.argmax(axis=1)

    out = df_prepared.copy()

    out["learned_sell_prob"] = probs[:, ACTION_TO_ID["SELL"]].astype(np.float32)
    out["learned_hold_prob"] = probs[:, ACTION_TO_ID["HOLD"]].astype(np.float32)
    out["learned_buy_prob"] = probs[:, ACTION_TO_ID["BUY"]].astype(np.float32)

    out["learned_recommendation"] = [ID_TO_ACTION[int(i)] for i in pred_id]

    out["learned_fusion_signal"] = clip11(outputs[:, 0]).astype(np.float32)
    out["learned_fusion_risk_score"] = clip01(outputs[:, 1]).astype(np.float32)
    out["learned_fusion_confidence"] = clip01(outputs[:, 2]).astype(np.float32)
    out["learned_position_multiplier"] = clip01(outputs[:, 3]).astype(np.float32)

    out["learned_quantitative_weight"] = branch_weights[:, 0].astype(np.float32)
    out["learned_qualitative_weight"] = branch_weights[:, 1].astype(np.float32)

    out = apply_user_rule_barrier(out, config)

    out["final_fusion_signal"] = out["learned_fusion_signal"]
    out["final_fusion_risk_score"] = out["learned_fusion_risk_score"]
    out["final_fusion_confidence"] = out["learned_fusion_confidence"]

    out["branch_weight_dominance"] = np.where(
        out["learned_quantitative_weight"].values >= out["learned_qualitative_weight"].values,
        "quantitative",
        "qualitative",
    )

    out["fusion_xai_summary"] = [build_fusion_xai_row(row) for _, row in out.iterrows()]

    preferred = [
        "ticker",
        "date",

        "final_recommendation",
        "final_fusion_signal",
        "final_fusion_risk_score",
        "final_fusion_confidence",
        "final_position_fraction",
        "final_position_pct",

        "learned_recommendation",
        "learned_sell_prob",
        "learned_hold_prob",
        "learned_buy_prob",
        "learned_fusion_signal",
        "learned_fusion_risk_score",
        "learned_fusion_confidence",
        "learned_position_multiplier",

        "learned_quantitative_weight",
        "learned_qualitative_weight",
        "branch_weight_dominance",

        "rule_changed_action",
        "user_rule_cap_fraction",
        "pre_rule_learned_position_fraction",
        "rule_barrier_reasons",

        "quantitative_recommendation",
        "risk_adjusted_quantitative_signal",
        "quantitative_risk_score",
        "quantitative_confidence",
        "quantitative_action_strength",
        "recommended_capital_fraction",
        "recommended_capital_pct",
        "attention_pooled_risk_score",
        "top_attention_risk_driver",

        *RISK_ATTENTION_COLUMNS,

        "qualitative_recommendation",
        "qualitative_score",
        "qualitative_risk_score",
        "qualitative_confidence",
        "event_count",
        "sentiment_event_count",
        "news_event_count",
        "dominant_qualitative_driver",

        "text_available",
        "branch_signal_disagreement",
        "branch_signal_agreement",
        "confidence_product",

        "regime_label",
        "regime_confidence",
        "liquidity_score",
        "tradable",
        "drawdown_risk_score",
        "contagion_risk_score",
        "combined_risk_score",

        "quantitative_xai_summary",
        "qualitative_xai_summary",
        "fusion_xai_summary",
    ]

    cols = [c for c in preferred if c in out.columns]
    extras = [c for c in out.columns if c not in cols and not c.startswith("_")]

    out_df = out[cols + extras].copy()
    out_df["chunk"] = int(chunk)
    out_df["split"] = split

    results_dir = Path(config.results_dir)
    xai_dir = results_dir / "xai"
    results_dir.mkdir(parents=True, exist_ok=True)
    xai_dir.mkdir(parents=True, exist_ok=True)

    pred_path = prediction_path(config, chunk, split)
    xai_out_path = xai_path(config, chunk, split)

    out_df.to_csv(pred_path, index=False)

    xai_report = build_fusion_xai_report(out_df, config, chunk, split, payload)
    with open(xai_out_path, "w") as f:
        json.dump(json_safe(xai_report), f, indent=2)

    print(f"  saved: {pred_path} rows={len(out_df):,}")
    print(f"  xai:   {xai_out_path}")
    print("  final recommendation counts:")
    print(out_df["final_recommendation"].value_counts().to_string())
    print("  branch dominance:")
    print(out_df["branch_weight_dominance"].value_counts().to_string())

    return {
        "predictions": out_df,
        "xai": xai_report,
        "paths": {
            "predictions": str(pred_path),
            "xai": str(xai_out_path),
        },
    }


def build_fusion_xai_report(
    df: pd.DataFrame,
    config: FusionConfig,
    chunk: int,
    split: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    report = {
        "module": "HybridFusionEngine",
        "chunk": int(chunk),
        "split": split,
        "rows": int(len(df)),
        "config": config.to_dict(),
        "model_best_val_loss": payload.get("best_val_loss"),
        "plain_english": (
            "The Fusion Engine combines quantitative and qualitative branch outputs using a learned fusion layer, "
            "then applies a user-defined rule barrier as the final line of defence. "
            "The learned layer proposes branch weights and a decision signal. "
            "The rule barrier enforces liquidity, drawdown, contagion, regime, and exposure constraints."
        ),
        "final_recommendation_counts": df["final_recommendation"].value_counts().to_dict(),
        "learned_recommendation_counts": df["learned_recommendation"].value_counts().to_dict(),
        "rule_changed_action_counts": df["rule_changed_action"].value_counts().to_dict(),
        "branch_dominance_counts": df["branch_weight_dominance"].value_counts().to_dict(),
        "summary_stats": {
            "final_signal_mean": float(df["final_fusion_signal"].mean()),
            "final_risk_mean": float(df["final_fusion_risk_score"].mean()),
            "final_confidence_mean": float(df["final_fusion_confidence"].mean()),
            "final_position_pct_mean": float(df["final_position_pct"].mean()),
            "learned_quantitative_weight_mean": float(df["learned_quantitative_weight"].mean()),
            "learned_qualitative_weight_mean": float(df["learned_qualitative_weight"].mean()),
            "text_available_rate": float(df["text_available"].mean()) if "text_available" in df.columns else 0.0,
        },
    }

    example_cols = [
        "ticker",
        "date",
        "final_recommendation",
        "final_fusion_signal",
        "final_fusion_risk_score",
        "final_fusion_confidence",
        "final_position_pct",
        "learned_quantitative_weight",
        "learned_qualitative_weight",
        "top_attention_risk_driver",
        "dominant_qualitative_driver",
        "rule_barrier_reasons",
        "fusion_xai_summary",
    ]
    example_cols = [c for c in example_cols if c in df.columns]

    report["strongest_buy_examples"] = (
        df[df["final_recommendation"] == "BUY"]
        .sort_values("final_fusion_signal", ascending=False)
        .head(50)[example_cols]
        .to_dict(orient="records")
    )

    report["strongest_sell_examples"] = (
        df[df["final_recommendation"] == "SELL"]
        .sort_values("final_fusion_signal", ascending=True)
        .head(50)[example_cols]
        .to_dict(orient="records")
    )

    report["highest_risk_examples"] = (
        df.sort_values("final_fusion_risk_score", ascending=False)
        .head(50)[example_cols]
        .to_dict(orient="records")
    )

    report["most_rule_changed_examples"] = (
        df[df["rule_changed_action"] == 1]
        .head(50)[example_cols]
        .to_dict(orient="records")
    )

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# INSPECTION / VALIDATION / SMOKE
# ═══════════════════════════════════════════════════════════════════════════════

def inspect_inputs(config: FusionConfig) -> None:
    config.resolve_paths()

    print("=" * 100)
    print("FUSION ENGINE INPUT INSPECTION")
    print("=" * 100)

    for module, pattern in [
        ("QuantitativeAnalyst", "quantitative_analysis_chunk{c}_{s}.csv"),
        ("QualitativeAnalyst", "qualitative_daily_chunk{c}_{s}.csv"),
    ]:
        base_dir = Path(config.quantitative_dir if module == "QuantitativeAnalyst" else config.qualitative_dir)

        print("\n" + "=" * 80)
        print(module)
        print("=" * 80)

        for c in [1, 2, 3]:
            for s in ["train", "val", "test"]:
                p = base_dir / pattern.format(c=c, s=s)
                rows = count_rows(p) if p.exists() else 0

                if module == "QuantitativeAnalyst" and p.exists():
                    try:
                        cols = pd.read_csv(p, nrows=0).columns
                        attention_schema = "top_attention_risk_driver" in cols
                        old_schema = "top_risk_driver" in cols and "top_attention_risk_driver" not in cols
                        schema_msg = f"attention_schema={attention_schema}, old_schema={old_schema}"
                    except Exception:
                        schema_msg = "schema_unreadable"
                else:
                    schema_msg = ""

                print(f"chunk{c}_{s}: {'OK' if p.exists() else 'MISSING'} rows={rows:,} {schema_msg} {p}")


def validate_predictions(config: FusionConfig, chunk: int, split: str) -> None:
    config.resolve_paths()

    path = prediction_path(config, chunk, split)

    print("=" * 100)
    print(f"FUSION ENGINE VALIDATION — chunk{chunk}_{split}")
    print("=" * 100)

    if not path.exists():
        raise FileNotFoundError(f"Missing Fusion prediction file: {path}")

    df = pd.read_csv(path, low_memory=False)

    required = [
        "ticker",
        "date",
        "final_recommendation",
        "final_fusion_signal",
        "final_fusion_risk_score",
        "final_fusion_confidence",
        "final_position_fraction",
        "final_position_pct",
        "learned_quantitative_weight",
        "learned_qualitative_weight",
        "rule_barrier_reasons",
        "fusion_xai_summary",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Fusion output missing required columns: {missing}")

    numeric = df.select_dtypes(include="number")
    finite_ratio = float(np.isfinite(numeric.values).mean()) if len(numeric.columns) else 1.0

    invalid_signal = int((df["final_fusion_signal"].abs() > 1.00001).sum())
    invalid_risk = int(((df["final_fusion_risk_score"] < -1e-9) | (df["final_fusion_risk_score"] > 1.00001)).sum())
    invalid_conf = int(((df["final_fusion_confidence"] < -1e-9) | (df["final_fusion_confidence"] > 1.00001)).sum())
    invalid_position = int(((df["final_position_fraction"] < -1e-9) | (df["final_position_fraction"] > 1.00001)).sum())

    weight_sum = df["learned_quantitative_weight"] + df["learned_qualitative_weight"]
    bad_weights = int((np.abs(weight_sum - 1.0) > 1e-4).sum())

    print(f"rows={len(df):,}")
    print(f"numeric finite ratio={finite_ratio:.6f}")
    print(f"invalid_signal={invalid_signal}")
    print(f"invalid_risk={invalid_risk}")
    print(f"invalid_confidence={invalid_conf}")
    print(f"invalid_position={invalid_position}")
    print(f"bad_branch_weight_sum={bad_weights}")

    print("\nfinal recommendation counts:")
    print(df["final_recommendation"].value_counts().to_string())

    print("\nlearned recommendation counts:")
    print(df["learned_recommendation"].value_counts().to_string())

    print("\nrule changes:")
    print(df["rule_changed_action"].value_counts().to_string())

    print("\nmean branch weights:")
    print(df[["learned_quantitative_weight", "learned_qualitative_weight"]].mean().to_string())

    if invalid_signal or invalid_risk or invalid_conf or invalid_position or bad_weights:
        raise RuntimeError("Fusion validation failed.")

    print("\nVALIDATION PASSED")


def smoke_test(config: FusionConfig) -> None:
    print("=" * 100)
    print("FUSION ENGINE SMOKE TEST")
    print("=" * 100)

    set_seed(config.seed)

    n = 512
    rng = np.random.default_rng(config.seed)

    df = pd.DataFrame({
        "ticker": [f"T{i % 32:03d}" for i in range(n)],
        "date": pd.to_datetime("2024-01-01") + pd.to_timedelta(np.arange(n) % 30, unit="D"),

        "quantitative_recommendation": rng.choice(["BUY", "HOLD", "SELL"], n, p=[0.15, 0.75, 0.10]),
        "risk_adjusted_quantitative_signal": rng.uniform(-0.5, 0.6, n),
        "technical_direction_score": rng.uniform(-0.5, 0.6, n),
        "quantitative_risk_score": rng.uniform(0.0, 1.0, n),
        "quantitative_confidence": rng.uniform(0.0, 1.0, n),
        "quantitative_action_strength": rng.uniform(0.0, 1.0, n),
        "recommended_capital_fraction": rng.uniform(0.0, 0.10, n),
        "recommended_capital_pct": rng.uniform(0.0, 10.0, n),
        "position_fraction_of_max": rng.uniform(0.0, 1.0, n),
        "max_single_stock_exposure": 0.10,
        "attention_pooled_risk_score": rng.uniform(0.0, 1.0, n),
        "top_attention_risk_driver": rng.choice(RISK_DRIVER_CATEGORIES[:-1], n),

        "risk_attention_volatility": rng.uniform(0.0, 1.0, n),
        "risk_attention_drawdown": rng.uniform(0.0, 1.0, n),
        "risk_attention_var_cvar": rng.uniform(0.0, 1.0, n),
        "risk_attention_contagion": rng.uniform(0.0, 1.0, n),
        "risk_attention_liquidity": rng.uniform(0.0, 1.0, n),
        "risk_attention_regime": rng.uniform(0.0, 1.0, n),

        "trend_score": rng.uniform(0.0, 1.0, n),
        "momentum_score": rng.uniform(0.0, 1.0, n),
        "timing_confidence": rng.uniform(0.0, 1.0, n),
        "technical_confidence": rng.uniform(0.0, 1.0, n),

        "volatility_risk_score": rng.uniform(0.0, 1.0, n),
        "drawdown_risk_score": rng.uniform(0.0, 1.0, n),
        "var_cvar_risk_score": rng.uniform(0.0, 1.0, n),
        "contagion_risk_score": rng.uniform(0.0, 1.0, n),
        "liquidity_risk_score": rng.uniform(0.0, 1.0, n),
        "regime_risk_score": rng.uniform(0.0, 1.0, n),
        "combined_risk_score": rng.uniform(0.0, 1.0, n),
        "regime_label": rng.choice(REGIME_CATEGORIES[:-1], n),
        "regime_confidence": rng.uniform(0.0, 1.0, n),
        "liquidity_score": rng.uniform(0.0, 1.0, n),
        "tradable": rng.choice([0.0, 1.0], n, p=[0.05, 0.95]),

        "qualitative_recommendation": rng.choice(["BUY", "HOLD", "SELL"], n, p=[0.10, 0.80, 0.10]),
        "event_count": rng.integers(0, 200, n),
        "sentiment_event_count": rng.integers(0, 200, n),
        "news_event_count": rng.integers(0, 200, n),
        "qualitative_score": rng.uniform(-0.4, 0.4, n),
        "qualitative_risk_score": rng.uniform(0.0, 1.0, n),
        "qualitative_confidence": rng.uniform(0.0, 1.0, n),
        "max_event_risk_score": rng.uniform(0.0, 1.0, n),
        "mean_event_risk_score": rng.uniform(0.0, 1.0, n),
        "mean_sentiment_score": rng.uniform(-0.4, 0.4, n),
        "mean_news_impact_score": rng.uniform(-0.4, 0.4, n),
        "mean_news_importance": rng.uniform(0.0, 1.0, n),
        "dominant_qualitative_driver": rng.choice(QUAL_DRIVER_CATEGORIES[:-1], n),
    })

    # Normalise random risk attention rows.
    att = df[RISK_ATTENTION_COLUMNS].values
    att = att / np.maximum(att.sum(axis=1, keepdims=True), 1e-8)
    df[RISK_ATTENTION_COLUMNS] = att

    x, aux, y_action, y_reg, y_weights, _, df_prepared, feature_names = prepare_training_arrays(df, config)

    scaler = FusionScaler()
    scaler.fit(x, feature_names)
    x = scaler.transform(x)

    model = HybridFusionModel(input_dim=x.shape[1], hidden_dim=64, n_layers=2, dropout=0.10).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    loader = make_loader(x, aux, y_action, y_reg, y_weights, 128, True, config.device, 0)

    for _ in range(3):
        loss = train_epoch(model, loader, optimizer, config.device)

    val_loss, val_acc = validate_epoch(model, loader, config.device)

    with torch.no_grad():
        xb = torch.from_numpy(x[:32]).to(config.device)
        ab = torch.from_numpy(aux[:32]).to(config.device)
        out = model(xb, ab)

    assert np.isfinite(val_loss)
    assert out["outputs"].shape == (32, 4)
    assert out["action_logits"].shape == (32, 3)
    assert out["branch_weights"].shape == (32, 2)
    assert torch.allclose(out["branch_weights"].sum(dim=1), torch.ones(32, device=config.device), atol=1e-5)

    print(f"SMOKE TEST PASSED | loss={val_loss:.6f} | acc={val_acc:.4f} | features={x.shape[1]}")