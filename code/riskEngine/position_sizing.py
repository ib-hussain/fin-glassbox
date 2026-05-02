#!/usr/bin/env python3
"""
code/riskEngine/position_sizing.py

Position Sizing Engine
======================

Project:
    fin-glassbox — Explainable Multimodal Neural Framework for Financial Risk Management

Purpose:
    Convert risk-engine outputs into an interpretable position-size recommendation.

Core principle:
    This module does NOT decide Buy/Hold/Sell.
    It decides how much exposure is allowed given the current risk state.

Inputs:
    outputs/results/TechnicalAnalyst/predictions_chunk{c}_{split}.csv
    outputs/results/Volatility/predictions_chunk{c}_{split}.csv
    outputs/results/Drawdown/predictions_chunk{c}_{split}.csv
    outputs/results/StemGNN/contagion_scores_chunk{c}_{split}.csv
    outputs/results/MTGNNRegime/predictions_chunk{c}_{split}.csv
    outputs/results/risk/chunks/var_cvar_chunk{c}_{split}.csv
    outputs/results/risk/chunks/liquidity_chunk{c}_{split}.csv

Outputs:
    outputs/results/PositionSizing/position_sizing_chunk{c}_{split}.csv
    outputs/results/PositionSizing/xai/position_sizing_chunk{c}_{split}_xai_summary.json

CLI:
    python code/riskEngine/position_sizing.py inspect --repo-root .
    python code/riskEngine/position_sizing.py smoke --repo-root .
    python code/riskEngine/position_sizing.py run --repo-root . --chunk 1 --split test --exposure-mode moderate --horizon-mode short
    python code/riskEngine/position_sizing.py run-all --repo-root . --chunks 1 2 3 --splits val test --exposure-mode moderate --horizon-mode short
    python code/riskEngine/position_sizing.py validate --repo-root . --chunk 1 --split test
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

EXPOSURE_MODES = {
    "conservative": 0.05,
    "moderate": 0.10,
    "aggressive": 0.15,
}

DEFAULT_RISK_WEIGHTS = {
    "volatility": 0.20,
    "drawdown": 0.15,
    "var_cvar": 0.15,
    "contagion": 0.25,
    "liquidity": 0.15,
    "regime": 0.10,
}

REGIME_BASE_RISK = {
    "calm": 0.10,
    "volatile": 0.65,
    "crisis": 1.00,
    "rotation": 0.45,
    "unknown": 0.50,
}

SIZE_BUCKETS = [
    (0.30, 1.00, "full"),
    (0.50, 0.75, "three_quarters"),
    (0.70, 0.50, "half"),
    (0.85, 0.25, "quarter"),
    (float("inf"), 0.00, "zero"),
]


@dataclass
class PositionSizingConfig:
    repo_root: str = ""
    output_dir: str = "outputs"

    technical_dir: str = "outputs/results/TechnicalAnalyst"
    volatility_dir: str = "outputs/results/Volatility"
    drawdown_dir: str = "outputs/results/Drawdown"
    stemgnn_dir: str = "outputs/results/StemGNN"
    regime_dir: str = "outputs/results/MTGNNRegime"
    simple_risk_dir: str = "outputs/results/risk/chunks"

    results_dir: str = "outputs/results/PositionSizing"

    exposure_mode: str = "moderate"
    horizon_mode: str = "short"

    # Approved default weights
    weight_volatility: float = 0.20
    weight_drawdown: float = 0.15
    weight_var_cvar: float = 0.15
    weight_contagion: float = 0.25
    weight_liquidity: float = 0.15
    weight_regime: float = 0.10

    # Hard caps, absolute portfolio fractions
    volatile_cap: float = 0.06
    rotation_cap: float = 0.05
    crisis_cap_short: float = 0.05
    crisis_cap_long: float = 0.03

    # Module hard-cap thresholds
    severe_module_risk_threshold: float = 0.85
    high_module_risk_threshold: float = 0.75
    severe_module_cap: float = 0.02
    high_module_cap: float = 0.05

    # Liquidity hard caps
    low_liquidity_threshold: float = 0.35
    severe_liquidity_threshold: float = 0.20
    low_liquidity_cap: float = 0.05
    severe_liquidity_cap: float = 0.02

    # Technical confidence scaling
    use_technical_confidence: bool = True
    technical_multiplier_min: float = 0.75
    technical_multiplier_max: float = 1.10

    # Regime alignment
    regime_tolerance_days: int = 90

    # Risk normalisation anchors
    vol_10d_anchor: float = 0.50
    vol_30d_anchor: float = 0.60
    cvar95_anchor: float = 0.08
    cvar99_anchor: float = 0.12
    max_reason_examples: int = 100

    def resolve_paths(self) -> "PositionSizingConfig":
        if self.repo_root:
            root = Path(self.repo_root)
            for attr in [
                "output_dir",
                "technical_dir",
                "volatility_dir",
                "drawdown_dir",
                "stemgnn_dir",
                "regime_dir",
                "simple_risk_dir",
                "results_dir",
            ]:
                value = getattr(self, attr)
                if value and not Path(value).is_absolute():
                    setattr(self, attr, str(root / value))
        return self

    @property
    def max_single_stock_exposure(self) -> float:
        if self.exposure_mode not in EXPOSURE_MODES:
            raise ValueError(f"Unknown exposure_mode={self.exposure_mode}. Choose from {list(EXPOSURE_MODES)}")
        return float(EXPOSURE_MODES[self.exposure_mode])

    @property
    def crisis_cap(self) -> float:
        if self.horizon_mode == "long":
            return float(self.crisis_cap_long)
        if self.horizon_mode == "short":
            return float(self.crisis_cap_short)
        raise ValueError("horizon_mode must be 'short' or 'long'")

    def risk_weights(self) -> Dict[str, float]:
        weights = {
            "volatility": float(self.weight_volatility),
            "drawdown": float(self.weight_drawdown),
            "var_cvar": float(self.weight_var_cvar),
            "contagion": float(self.weight_contagion),
            "liquidity": float(self.weight_liquidity),
            "regime": float(self.weight_regime),
        }
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Risk weights must sum to a positive value.")
        return {k: v / total for k, v in weights.items()}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(json_safe(k)): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def read_csv_if_exists(path: Path, *, parse_dates: bool = True) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None

    if parse_dates:
        df = pd.read_csv(path, dtype={"ticker": str})
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    return pd.read_csv(path, dtype={"ticker": str})


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return max(sum(1 for _ in f) - 1, 0)


def clip01(x: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return np.clip(x, 0.0, 1.0)


def safe_numeric(df: pd.DataFrame, col: str, default: float = 0.5) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=np.float32)
    s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return s.fillna(default).astype(np.float32)


def dedupe_key(df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    existing = [c for c in key_cols if c in df.columns]
    if not existing:
        return df
    return df.drop_duplicates(existing, keep="last").reset_index(drop=True)


def ensure_key_columns(df: pd.DataFrame, source_name: str) -> None:
    if "ticker" not in df.columns or "date" not in df.columns:
        raise ValueError(f"{source_name} must contain ticker and date columns.")


def path_for(config: PositionSizingConfig, module: str, chunk: int, split: str) -> Path:
    if module == "technical":
        return Path(config.technical_dir) / f"predictions_chunk{chunk}_{split}.csv"
    if module == "volatility":
        return Path(config.volatility_dir) / f"predictions_chunk{chunk}_{split}.csv"
    if module == "drawdown":
        return Path(config.drawdown_dir) / f"predictions_chunk{chunk}_{split}.csv"
    if module == "stemgnn":
        return Path(config.stemgnn_dir) / f"contagion_scores_chunk{chunk}_{split}.csv"
    if module == "regime":
        return Path(config.regime_dir) / f"predictions_chunk{chunk}_{split}.csv"
    if module == "liquidity":
        return Path(config.simple_risk_dir) / f"liquidity_chunk{chunk}_{split}.csv"
    if module == "var_cvar":
        return Path(config.simple_risk_dir) / f"var_cvar_chunk{chunk}_{split}.csv"
    raise ValueError(f"Unknown module: {module}")


# ═══════════════════════════════════════════════════════════════════════════════
# INPUT LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_technical(config: PositionSizingConfig, chunk: int, split: str) -> Optional[pd.DataFrame]:
    path = path_for(config, "technical", chunk, split)
    df = read_csv_if_exists(path)
    if df is None:
        return None

    ensure_key_columns(df, "TechnicalAnalyst")
    keep = ["ticker", "date", "trend_score", "momentum_score", "timing_confidence"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = dedupe_key(df, ["ticker", "date"])
    return df


def load_volatility(config: PositionSizingConfig, chunk: int, split: str) -> Optional[pd.DataFrame]:
    path = path_for(config, "volatility", chunk, split)
    df = read_csv_if_exists(path)
    if df is None:
        return None

    ensure_key_columns(df, "Volatility")
    keep = [
        "ticker", "date", "vol_10d", "vol_30d",
        "regime_label", "regime_probs_low", "regime_probs_medium", "regime_probs_high",
        "confidence", "garch_vol", "recent_vol",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.rename(columns={
        "regime_label": "volatility_regime_label",
        "confidence": "volatility_confidence",
    })
    df = dedupe_key(df, ["ticker", "date"])
    return df


def load_drawdown(config: PositionSizingConfig, chunk: int, split: str) -> Optional[pd.DataFrame]:
    path = path_for(config, "drawdown", chunk, split)
    df = read_csv_if_exists(path)
    if df is None:
        return None

    ensure_key_columns(df, "Drawdown")
    keep = [
        "ticker", "date",
        "expected_drawdown_10d", "drawdown_risk_10d", "recovery_days_10d", "confidence_10d",
        "expected_drawdown_30d", "drawdown_risk_30d", "recovery_days_30d", "confidence_30d",
        "drawdown_risk_score",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = dedupe_key(df, ["ticker", "date"])
    return df


def load_stemgnn(config: PositionSizingConfig, chunk: int, split: str) -> Optional[pd.DataFrame]:
    path = path_for(config, "stemgnn", chunk, split)
    df = read_csv_if_exists(path, parse_dates=False)
    if df is None:
        return None

    if "ticker" not in df.columns:
        raise ValueError("StemGNN contagion output must contain ticker column.")

    keep = ["ticker", "contagion_5d", "contagion_20d", "contagion_60d"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["ticker"] = df["ticker"].astype(str)
    df = dedupe_key(df, ["ticker"])
    return df


def load_var_cvar(config: PositionSizingConfig, chunk: int, split: str) -> Optional[pd.DataFrame]:
    path = path_for(config, "var_cvar", chunk, split)
    df = read_csv_if_exists(path)
    if df is None:
        return None

    ensure_key_columns(df, "VaR/CVaR")
    keep = [
        "ticker", "date",
        "var_95", "var_99", "cvar_95", "cvar_99",
        "tail_ratio_95", "tail_ratio_99", "window_size",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = dedupe_key(df, ["ticker", "date"])
    return df


def load_liquidity(config: PositionSizingConfig, chunk: int, split: str) -> Optional[pd.DataFrame]:
    path = path_for(config, "liquidity", chunk, split)
    df = read_csv_if_exists(path)
    if df is None:
        return None

    ensure_key_columns(df, "Liquidity")
    keep = [
        "ticker", "date", "liquidity_score", "slippage_estimate_pct",
        "days_to_liquidate_1M", "tradable", "dv_score", "vr_score", "to_score",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = dedupe_key(df, ["ticker", "date"])
    return df


def load_regime(config: PositionSizingConfig, chunk: int, split: str) -> Optional[pd.DataFrame]:
    path = path_for(config, "regime", chunk, split)
    df = read_csv_if_exists(path)
    if df is None:
        return None

    if "date" not in df.columns:
        raise ValueError("MTGNNRegime output must contain date column.")

    keep = [
        "date",
        "pred_regime_id", "pred_regime_label",
        "true_regime_id", "true_regime_label",
        "confidence", "transition_probability",
        "prob_calm", "prob_volatile", "prob_crisis", "prob_rotation",
        "graph_density", "avg_degree_norm", "std_degree_norm",
        "mean_edge_weight", "max_edge_weight", "graph_entropy",
        "learned_graph_stress", "macro_stress_score", "label_graph_stress_score",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.rename(columns={
        "pred_regime_id": "regime_id",
        "pred_regime_label": "regime_label",
        "true_regime_id": "true_regime_id",
        "true_regime_label": "true_regime_label",
        "confidence": "regime_confidence",
        "transition_probability": "regime_transition_probability",
    })
    df = dedupe_key(df, ["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MERGING
# ═══════════════════════════════════════════════════════════════════════════════

def choose_base_frame(
    technical: Optional[pd.DataFrame],
    drawdown: Optional[pd.DataFrame],
    volatility: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Choose the row universe.

    Prefer TechnicalAnalyst because sizing may use technical confidence.
    If missing, fall back to Drawdown, then Volatility.
    """
    if technical is not None:
        base = technical.copy()
        base["_base_source"] = "technical"
        return base

    if drawdown is not None:
        base = drawdown[["ticker", "date"]].copy()
        base["_base_source"] = "drawdown"
        return base

    if volatility is not None:
        base = volatility[["ticker", "date"]].copy()
        base["_base_source"] = "volatility"
        return base

    raise FileNotFoundError("No usable base frame found. Need at least TechnicalAnalyst, Drawdown, or Volatility output.")


def merge_inputs(config: PositionSizingConfig, chunk: int, split: str) -> Tuple[pd.DataFrame, Dict[str, bool]]:
    technical = load_technical(config, chunk, split)
    volatility = load_volatility(config, chunk, split)
    drawdown = load_drawdown(config, chunk, split)
    stemgnn = load_stemgnn(config, chunk, split)
    var_cvar = load_var_cvar(config, chunk, split)
    liquidity = load_liquidity(config, chunk, split)
    regime = load_regime(config, chunk, split)

    availability = {
        "technical": technical is not None,
        "volatility": volatility is not None,
        "drawdown": drawdown is not None,
        "stemgnn": stemgnn is not None,
        "var_cvar": var_cvar is not None,
        "liquidity": liquidity is not None,
        "regime": regime is not None,
    }

    df = choose_base_frame(technical, drawdown, volatility)
    ensure_key_columns(df, "base")

    df["ticker"] = df["ticker"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "ticker"]).copy()
    df = dedupe_key(df, ["ticker", "date"])

    if technical is not None and "_base_source" in df.columns and df["_base_source"].iloc[0] != "technical":
        df = df.merge(technical, on=["ticker", "date"], how="left")

    if volatility is not None:
        df = df.merge(volatility, on=["ticker", "date"], how="left")

    if drawdown is not None:
        existing = set(df.columns)
        draw_cols = [c for c in drawdown.columns if c not in existing or c in {"ticker", "date"}]
        df = df.merge(drawdown[draw_cols], on=["ticker", "date"], how="left")

    if var_cvar is not None:
        df = df.merge(var_cvar, on=["ticker", "date"], how="left")

    if liquidity is not None:
        df = df.merge(liquidity, on=["ticker", "date"], how="left")

    if stemgnn is not None:
        df = df.merge(stemgnn, on="ticker", how="left")

    if regime is not None and len(regime) > 0:
        df = df.sort_values("date").reset_index(drop=True)
        regime_sorted = regime.sort_values("date").reset_index(drop=True)

        df = pd.merge_asof(
            df,
            regime_sorted,
            on="date",
            direction="backward",
            tolerance=pd.Timedelta(days=int(config.regime_tolerance_days)),
        )

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df, availability


# ═══════════════════════════════════════════════════════════════════════════════
# RISK SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_technical_confidence(df: pd.DataFrame) -> pd.Series:
    trend = safe_numeric(df, "trend_score", 0.5)
    momentum = safe_numeric(df, "momentum_score", 0.5)
    timing = safe_numeric(df, "timing_confidence", 0.5)

    # Technical confidence is not direction. It is confidence/quality of signal.
    # Strong trend/momentum and high timing confidence preserve size; weak values reduce it.
    conf = 0.35 * trend + 0.35 * momentum + 0.30 * timing
    return pd.Series(clip01(conf), index=df.index, dtype=np.float32)


def compute_volatility_risk(df: pd.DataFrame, config: PositionSizingConfig) -> pd.Series:
    vol10 = safe_numeric(df, "vol_10d", np.nan)
    vol30 = safe_numeric(df, "vol_30d", np.nan)
    high_prob = safe_numeric(df, "regime_probs_high", 0.33)

    risk10 = pd.Series(clip01(vol10 / max(config.vol_10d_anchor, 1e-8)), index=df.index)
    risk30 = pd.Series(clip01(vol30 / max(config.vol_30d_anchor, 1e-8)), index=df.index)

    risk = 0.30 * risk10 + 0.45 * risk30 + 0.25 * high_prob
    risk = risk.replace([np.inf, -np.inf], np.nan).fillna(0.5)
    return pd.Series(clip01(risk), index=df.index, dtype=np.float32)


def compute_drawdown_risk(df: pd.DataFrame) -> pd.Series:
    if "drawdown_risk_score" in df.columns:
        return pd.Series(clip01(safe_numeric(df, "drawdown_risk_score", 0.5)), index=df.index, dtype=np.float32)

    r10 = safe_numeric(df, "drawdown_risk_10d", 0.5)
    r30 = safe_numeric(df, "drawdown_risk_30d", 0.5)
    dd10 = safe_numeric(df, "expected_drawdown_10d", 0.05)
    dd30 = safe_numeric(df, "expected_drawdown_30d", 0.08)

    sev10 = pd.Series(clip01(dd10 / 0.10), index=df.index)
    sev30 = pd.Series(clip01(dd30 / 0.16), index=df.index)

    risk = 0.20 * sev10 + 0.30 * sev30 + 0.20 * r10 + 0.30 * r30
    return pd.Series(clip01(risk), index=df.index, dtype=np.float32)


def compute_var_cvar_risk(df: pd.DataFrame, config: PositionSizingConfig) -> pd.Series:
    cvar95 = safe_numeric(df, "cvar_95", np.nan).abs()
    cvar99 = safe_numeric(df, "cvar_99", np.nan).abs()
    tail95 = safe_numeric(df, "tail_ratio_95", 1.0)
    tail99 = safe_numeric(df, "tail_ratio_99", 1.0)

    c95 = pd.Series(clip01(cvar95 / max(config.cvar95_anchor, 1e-8)), index=df.index)
    c99 = pd.Series(clip01(cvar99 / max(config.cvar99_anchor, 1e-8)), index=df.index)

    tail_score = 0.5 * pd.Series(clip01((tail95 - 1.0) / 1.0), index=df.index) + 0.5 * pd.Series(clip01((tail99 - 1.0) / 1.0), index=df.index)

    risk = 0.40 * c95 + 0.40 * c99 + 0.20 * tail_score
    risk = risk.replace([np.inf, -np.inf], np.nan).fillna(0.5)
    return pd.Series(clip01(risk), index=df.index, dtype=np.float32)


def compute_contagion_risk(df: pd.DataFrame) -> pd.Series:
    c5 = safe_numeric(df, "contagion_5d", 0.5)
    c20 = safe_numeric(df, "contagion_20d", 0.5)
    c60 = safe_numeric(df, "contagion_60d", 0.5)

    risk = 0.20 * c5 + 0.40 * c20 + 0.40 * c60
    return pd.Series(clip01(risk), index=df.index, dtype=np.float32)


def compute_liquidity_risk(df: pd.DataFrame) -> pd.Series:
    liquidity = safe_numeric(df, "liquidity_score", 0.5)
    slippage = safe_numeric(df, "slippage_estimate_pct", 0.0)
    days = safe_numeric(df, "days_to_liquidate_1M", 0.0)
    tradable = safe_numeric(df, "tradable", 1.0)

    base = 1.0 - liquidity
    slip_risk = pd.Series(clip01(slippage / 0.02), index=df.index)
    days_risk = pd.Series(clip01(days / 10.0), index=df.index)

    risk = 0.65 * base + 0.20 * slip_risk + 0.15 * days_risk
    risk = pd.Series(clip01(risk), index=df.index, dtype=np.float32)
    risk = np.where(tradable < 0.5, 1.0, risk)
    return pd.Series(clip01(risk), index=df.index, dtype=np.float32)


def compute_regime_risk(df: pd.DataFrame) -> pd.Series:
    labels = df.get("regime_label", pd.Series("unknown", index=df.index)).astype(str).str.lower()

    label_risk = labels.map(REGIME_BASE_RISK).fillna(0.5).astype(np.float32)

    prob_calm = safe_numeric(df, "prob_calm", np.nan)
    prob_volatile = safe_numeric(df, "prob_volatile", np.nan)
    prob_crisis = safe_numeric(df, "prob_crisis", np.nan)
    prob_rotation = safe_numeric(df, "prob_rotation", np.nan)

    prob_available = (
        prob_calm.notna()
        & prob_volatile.notna()
        & prob_crisis.notna()
        & prob_rotation.notna()
    )

    prob_risk = (
        0.10 * prob_calm.fillna(0.0)
        + 0.65 * prob_volatile.fillna(0.0)
        + 1.00 * prob_crisis.fillna(0.0)
        + 0.45 * prob_rotation.fillna(0.0)
    )

    macro = safe_numeric(df, "macro_stress_score", np.nan)
    learned = safe_numeric(df, "learned_graph_stress", np.nan)

    stress = 0.5 * macro.fillna(label_risk) + 0.5 * learned.fillna(label_risk)

    risk = np.where(
        prob_available,
        0.50 * prob_risk + 0.30 * stress + 0.20 * label_risk,
        0.70 * label_risk + 0.30 * stress,
    )

    return pd.Series(clip01(risk), index=df.index, dtype=np.float32)


def risk_to_bucket(combined_risk: pd.Series) -> Tuple[pd.Series, pd.Series]:
    fractions = np.zeros(len(combined_risk), dtype=np.float32)
    labels = np.empty(len(combined_risk), dtype=object)

    values = combined_risk.values
    for threshold, fraction, label in SIZE_BUCKETS:
        mask = (fractions == 0.0) & (labels == None)  # noqa: E711
        eligible = values < threshold
        selected = mask & eligible
        fractions[selected] = float(fraction)
        labels[selected] = label

    # Handle exact zero bucket correctly
    empty = pd.isna(pd.Series(labels))
    labels[empty.values] = "zero"
    return pd.Series(fractions, index=combined_risk.index), pd.Series(labels, index=combined_risk.index)


def technical_multiplier(technical_confidence: pd.Series, config: PositionSizingConfig) -> pd.Series:
    if not config.use_technical_confidence:
        return pd.Series(1.0, index=technical_confidence.index, dtype=np.float32)

    lo = float(config.technical_multiplier_min)
    hi = float(config.technical_multiplier_max)
    mult = lo + (hi - lo) * clip01(technical_confidence)
    return pd.Series(mult, index=technical_confidence.index, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# HARD CAPS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_regime_hard_cap(df: pd.DataFrame, config: PositionSizingConfig) -> pd.Series:
    max_exp = float(config.max_single_stock_exposure)
    labels = df.get("regime_label", pd.Series("unknown", index=df.index)).astype(str).str.lower()

    cap = pd.Series(max_exp, index=df.index, dtype=np.float32)
    cap = np.where(labels == "volatile", np.minimum(max_exp, float(config.volatile_cap)), cap)
    cap = np.where(labels == "rotation", np.minimum(max_exp, float(config.rotation_cap)), cap)
    cap = np.where(labels == "crisis", np.minimum(max_exp, float(config.crisis_cap)), cap)

    return pd.Series(cap, index=df.index, dtype=np.float32)


def cap_from_module_risk(risk: pd.Series, config: PositionSizingConfig) -> pd.Series:
    max_exp = float(config.max_single_stock_exposure)
    cap = pd.Series(max_exp, index=risk.index, dtype=np.float32)
    cap = np.where(risk >= float(config.high_module_risk_threshold), np.minimum(max_exp, float(config.high_module_cap)), cap)
    cap = np.where(risk >= float(config.severe_module_risk_threshold), np.minimum(max_exp, float(config.severe_module_cap)), cap)
    return pd.Series(cap, index=risk.index, dtype=np.float32)


def compute_liquidity_hard_cap(df: pd.DataFrame, liquidity_risk: pd.Series, config: PositionSizingConfig) -> pd.Series:
    max_exp = float(config.max_single_stock_exposure)
    liquidity = safe_numeric(df, "liquidity_score", 0.5)
    tradable = safe_numeric(df, "tradable", 1.0)

    cap = pd.Series(max_exp, index=df.index, dtype=np.float32)
    cap = np.where(liquidity < float(config.low_liquidity_threshold), np.minimum(max_exp, float(config.low_liquidity_cap)), cap)
    cap = np.where(liquidity < float(config.severe_liquidity_threshold), np.minimum(max_exp, float(config.severe_liquidity_cap)), cap)
    cap = np.where(tradable < 0.5, 0.0, cap)

    return pd.Series(cap, index=df.index, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPLANATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def top_contributors(row: pd.Series, weights: Dict[str, float], n: int = 3) -> List[str]:
    weighted = {
        "volatility": float(row["volatility_risk_score"]) * weights["volatility"],
        "drawdown": float(row["drawdown_risk_score"]) * weights["drawdown"],
        "var_cvar": float(row["var_cvar_risk_score"]) * weights["var_cvar"],
        "contagion": float(row["contagion_risk_score"]) * weights["contagion"],
        "liquidity": float(row["liquidity_risk_score"]) * weights["liquidity"],
        "regime": float(row["regime_risk_score"]) * weights["regime"],
    }
    return [k for k, _ in sorted(weighted.items(), key=lambda kv: kv[1], reverse=True)[:n]]


def build_reduction_reason(row: pd.Series, weights: Dict[str, float]) -> str:
    reasons: List[str] = []

    top = top_contributors(row, weights, 3)
    reasons.append("Top weighted risks: " + ", ".join(top))

    if bool(row.get("hard_cap_applied", False)):
        reasons.append(f"Hard cap applied: {row.get('binding_cap_source', 'unknown')}")

    regime_label = str(row.get("regime_label", "unknown"))
    if regime_label.lower() == "crisis":
        reasons.append("Crisis regime constrained exposure")

    if float(row.get("liquidity_risk_score", 0.0)) >= 0.75:
        reasons.append("Liquidity risk is high")

    if float(row.get("contagion_risk_score", 0.0)) >= 0.75:
        reasons.append("Contagion risk is high")

    if float(row.get("drawdown_risk_score", 0.0)) >= 0.75:
        reasons.append("Drawdown risk is high")

    if float(row.get("technical_confidence", 0.5)) < 0.45:
        reasons.append("Technical confidence reduced position")

    return "; ".join(reasons)


def compact_xai_summary(row: pd.Series) -> str:
    return (
        f"Position={float(row['recommended_capital_pct']):.2f}% | "
        f"risk={float(row['combined_risk_score']):.3f} | "
        f"bucket={row['size_bucket']} | "
        f"cap={row['binding_cap_source']} | "
        f"regime={row.get('regime_label', 'unknown')}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CORE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_position_sizing(config: PositionSizingConfig, chunk: int, split: str) -> Dict[str, Any]:
    config.resolve_paths()

    weights = config.risk_weights()
    max_exp = float(config.max_single_stock_exposure)

    print("=" * 80)
    print(f"POSITION SIZING — chunk{chunk}_{split}")
    print("=" * 80)
    print(f"  exposure_mode={config.exposure_mode}, max_single_stock_exposure={max_exp:.2%}")
    print(f"  horizon_mode={config.horizon_mode}, crisis_cap={config.crisis_cap:.2%}")
    print(f"  weights={weights}")

    df, availability = merge_inputs(config, chunk, split)

    print(f"  merged rows: {len(df):,}")
    print(f"  availability: {availability}")

    if len(df) == 0:
        raise ValueError(f"No rows available after merging inputs for chunk{chunk}_{split}")

    # Module risk scores
    df["technical_confidence"] = compute_technical_confidence(df)
    df["technical_multiplier"] = technical_multiplier(df["technical_confidence"], config)

    df["volatility_risk_score"] = compute_volatility_risk(df, config)
    df["drawdown_risk_score"] = compute_drawdown_risk(df)
    df["var_cvar_risk_score"] = compute_var_cvar_risk(df, config)
    df["contagion_risk_score"] = compute_contagion_risk(df)
    df["liquidity_risk_score"] = compute_liquidity_risk(df)
    df["regime_risk_score"] = compute_regime_risk(df)

    df["combined_risk_score"] = (
        weights["volatility"] * df["volatility_risk_score"]
        + weights["drawdown"] * df["drawdown_risk_score"]
        + weights["var_cvar"] * df["var_cvar_risk_score"]
        + weights["contagion"] * df["contagion_risk_score"]
        + weights["liquidity"] * df["liquidity_risk_score"]
        + weights["regime"] * df["regime_risk_score"]
    ).astype(np.float32)

    df["combined_risk_score"] = clip01(df["combined_risk_score"])

    bucket_fraction, bucket_label = risk_to_bucket(df["combined_risk_score"])
    df["risk_bucket_fraction"] = bucket_fraction
    df["size_bucket"] = bucket_label

    df["max_single_stock_exposure"] = max_exp

    # Candidate size before hard caps
    df["pre_cap_position_fraction_of_max"] = clip01(df["risk_bucket_fraction"] * df["technical_multiplier"])
    df["pre_cap_capital_fraction"] = df["pre_cap_position_fraction_of_max"] * max_exp

    # Hard caps
    df["regime_hard_cap"] = compute_regime_hard_cap(df, config)
    df["volatility_hard_cap"] = cap_from_module_risk(df["volatility_risk_score"], config)
    df["drawdown_hard_cap"] = cap_from_module_risk(df["drawdown_risk_score"], config)
    df["var_cvar_hard_cap"] = cap_from_module_risk(df["var_cvar_risk_score"], config)
    df["contagion_hard_cap"] = cap_from_module_risk(df["contagion_risk_score"], config)
    df["liquidity_hard_cap"] = compute_liquidity_hard_cap(df, df["liquidity_risk_score"], config)

    cap_cols = [
        "regime_hard_cap",
        "volatility_hard_cap",
        "drawdown_hard_cap",
        "var_cvar_hard_cap",
        "contagion_hard_cap",
        "liquidity_hard_cap",
    ]

    cap_matrix = df[cap_cols].values.astype(np.float32)
    min_cap_idx = np.argmin(cap_matrix, axis=1)
    min_cap_values = cap_matrix[np.arange(len(df)), min_cap_idx]
    binding_sources = np.array(cap_cols, dtype=object)[min_cap_idx]

    df["binding_hard_cap"] = min_cap_values
    df["binding_cap_source"] = binding_sources

    df["recommended_capital_fraction"] = np.minimum(
        df["pre_cap_capital_fraction"].values.astype(np.float32),
        df["binding_hard_cap"].values.astype(np.float32),
    )
    df["recommended_capital_fraction"] = clip01(df["recommended_capital_fraction"])

    df["position_fraction_of_max"] = np.where(
        max_exp > 0,
        df["recommended_capital_fraction"] / max_exp,
        0.0,
    )
    df["position_fraction_of_max"] = clip01(df["position_fraction_of_max"])

    df["recommended_capital_pct"] = df["recommended_capital_fraction"] * 100.0
    df["risk_budget_used"] = df["position_fraction_of_max"]

    df["hard_cap_applied"] = df["binding_hard_cap"] < (df["pre_cap_capital_fraction"] - 1e-9)

    # Final explanation strings
    reason_rows = []
    xai_rows = []
    for _, row in df.iterrows():
        reason_rows.append(build_reduction_reason(row, weights))
        xai_rows.append(compact_xai_summary(row))

    df["size_reduction_reasons"] = reason_rows
    df["xai_summary"] = xai_rows

    df["chunk"] = int(chunk)
    df["split"] = split
    df["exposure_mode"] = config.exposure_mode
    df["horizon_mode"] = config.horizon_mode

    preferred_cols = [
        "ticker", "date", "chunk", "split",
        "exposure_mode", "horizon_mode",

        "volatility_risk_score",
        "drawdown_risk_score",
        "var_cvar_risk_score",
        "contagion_risk_score",
        "liquidity_risk_score",
        "regime_risk_score",
        "technical_confidence",
        "technical_multiplier",

        "combined_risk_score",
        "size_bucket",
        "risk_bucket_fraction",

        "max_single_stock_exposure",
        "pre_cap_position_fraction_of_max",
        "pre_cap_capital_fraction",
        "position_fraction_of_max",
        "recommended_capital_fraction",
        "recommended_capital_pct",
        "risk_budget_used",

        "regime_label",
        "regime_confidence",
        "prob_calm",
        "prob_volatile",
        "prob_crisis",
        "prob_rotation",

        "regime_hard_cap",
        "volatility_hard_cap",
        "drawdown_hard_cap",
        "var_cvar_hard_cap",
        "contagion_hard_cap",
        "liquidity_hard_cap",
        "binding_hard_cap",
        "binding_cap_source",
        "hard_cap_applied",

        "trend_score",
        "momentum_score",
        "timing_confidence",

        "vol_10d",
        "vol_30d",
        "expected_drawdown_10d",
        "expected_drawdown_30d",
        "drawdown_risk_10d",
        "drawdown_risk_30d",
        "var_95",
        "var_99",
        "cvar_95",
        "cvar_99",
        "tail_ratio_95",
        "tail_ratio_99",
        "contagion_5d",
        "contagion_20d",
        "contagion_60d",
        "liquidity_score",
        "slippage_estimate_pct",
        "days_to_liquidate_1M",
        "tradable",

        "size_reduction_reasons",
        "xai_summary",
    ]

    cols = [c for c in preferred_cols if c in df.columns]
    extra = [c for c in df.columns if c not in cols and not c.startswith("_")]
    out_df = df[cols + extra].copy()

    results_dir = Path(config.results_dir)
    xai_dir = results_dir / "xai"
    results_dir.mkdir(parents=True, exist_ok=True)
    xai_dir.mkdir(parents=True, exist_ok=True)

    pred_path = results_dir / f"position_sizing_chunk{chunk}_{split}.csv"
    xai_path = xai_dir / f"position_sizing_chunk{chunk}_{split}_xai_summary.json"

    out_df.to_csv(pred_path, index=False)

    xai = build_xai_report(out_df, config, chunk, split, availability, weights)
    with open(xai_path, "w") as f:
        json.dump(json_safe(xai), f, indent=2)

    print(f"  saved: {pred_path} rows={len(out_df):,}")
    print(f"  xai:   {xai_path}")

    return {
        "predictions": out_df,
        "xai": xai,
        "paths": {
            "predictions": str(pred_path),
            "xai": str(xai_path),
        },
    }


def build_xai_report(
    df: pd.DataFrame,
    config: PositionSizingConfig,
    chunk: int,
    split: str,
    availability: Dict[str, bool],
    weights: Dict[str, float],
) -> Dict[str, Any]:
    risk_cols = [
        "volatility_risk_score",
        "drawdown_risk_score",
        "var_cvar_risk_score",
        "contagion_risk_score",
        "liquidity_risk_score",
        "regime_risk_score",
    ]

    summary = {
        "module": "PositionSizingEngine",
        "chunk": int(chunk),
        "split": split,
        "config": config.to_dict(),
        "availability": availability,
        "risk_weights": weights,
        "rows": int(len(df)),
        "date_min": str(pd.to_datetime(df["date"]).min().date()) if len(df) else None,
        "date_max": str(pd.to_datetime(df["date"]).max().date()) if len(df) else None,
        "ticker_count": int(df["ticker"].nunique()) if "ticker" in df.columns else 0,
        "position_summary": {
            "recommended_capital_fraction_mean": float(df["recommended_capital_fraction"].mean()),
            "recommended_capital_fraction_median": float(df["recommended_capital_fraction"].median()),
            "recommended_capital_fraction_p95": float(df["recommended_capital_fraction"].quantile(0.95)),
            "zero_position_rate": float((df["recommended_capital_fraction"] <= 1e-9).mean()),
            "hard_cap_applied_rate": float(df["hard_cap_applied"].mean()),
        },
        "risk_score_summary": {},
        "binding_cap_counts": df["binding_cap_source"].value_counts().to_dict() if "binding_cap_source" in df.columns else {},
        "size_bucket_counts": df["size_bucket"].value_counts().to_dict() if "size_bucket" in df.columns else {},
        "regime_counts": df["regime_label"].astype(str).value_counts().to_dict() if "regime_label" in df.columns else {},
        "plain_english": (
            "The Position Sizing Engine converts module-level risk outputs into a recommended capital fraction. "
            "It first computes weighted risk from volatility, drawdown, VaR/CVaR, contagion, liquidity, and regime risk. "
            "The weighted risk is mapped to a size bucket. Technical confidence may scale the position within the allowed range, "
            "but hard caps from regime, liquidity, drawdown, VaR/CVaR, contagion, and volatility can only reduce exposure. "
            "Risk caps always override optimistic signals."
        ),
    }

    for c in risk_cols:
        if c in df.columns:
            summary["risk_score_summary"][c] = {
                "mean": float(df[c].mean()),
                "median": float(df[c].median()),
                "p90": float(df[c].quantile(0.90)),
                "p95": float(df[c].quantile(0.95)),
                "max": float(df[c].max()),
            }

    example_cols = [
        "ticker", "date", "recommended_capital_pct", "combined_risk_score",
        "size_bucket", "binding_cap_source", "regime_label",
        "size_reduction_reasons", "xai_summary",
    ]
    example_cols = [c for c in example_cols if c in df.columns]

    examples = df.sort_values("recommended_capital_fraction", ascending=False).head(config.max_reason_examples)
    summary["top_position_examples"] = examples[example_cols].to_dict(orient="records")

    capped = df[df["hard_cap_applied"]].head(config.max_reason_examples)
    summary["hard_cap_examples"] = capped[example_cols].to_dict(orient="records")

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# INSPECT / VALIDATE / SMOKE
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_inspect(config: PositionSizingConfig) -> None:
    config.resolve_paths()

    print("=" * 100)
    print("POSITION SIZING INPUT INSPECTION")
    print("=" * 100)

    modules = ["technical", "volatility", "drawdown", "stemgnn", "regime", "liquidity", "var_cvar"]

    for chunk in [1, 2, 3]:
        for split in ["train", "val", "test"]:
            print(f"\n========== chunk{chunk}_{split} ==========")
            for module in modules:
                path = path_for(config, module, chunk, split)
                exists = path.exists()
                rows = count_rows(path) if exists else 0
                print(f"{module:12s} {'OK' if exists else 'MISSING':8s} rows={rows:,} path={path}")
                if exists:
                    try:
                        df = pd.read_csv(path, nrows=1)
                        print(f"{'':12s} columns={list(df.columns)[:20]}")
                    except Exception as exc:
                        print(f"{'':12s} failed to read columns: {exc}")


def cmd_validate(config: PositionSizingConfig, chunk: int, split: str) -> None:
    config.resolve_paths()
    path = Path(config.results_dir) / f"position_sizing_chunk{chunk}_{split}.csv"

    print("=" * 100)
    print(f"POSITION SIZING VALIDATION — chunk{chunk}_{split}")
    print("=" * 100)

    if not path.exists():
        raise FileNotFoundError(f"Missing output: {path}")

    df = pd.read_csv(path, parse_dates=["date"], dtype={"ticker": str})

    required = [
        "ticker", "date",
        "combined_risk_score",
        "position_fraction_of_max",
        "recommended_capital_fraction",
        "recommended_capital_pct",
        "binding_hard_cap",
        "hard_cap_applied",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required output columns: {missing}")

    numeric = df.select_dtypes(include="number")
    finite_ratio = float(np.isfinite(numeric.values).mean()) if len(numeric.columns) else 1.0

    max_exposure = float(df["max_single_stock_exposure"].dropna().max())
    cap_violations = int((df["recommended_capital_fraction"] > df["binding_hard_cap"] + 1e-8).sum())
    max_violations = int((df["recommended_capital_fraction"] > max_exposure + 1e-8).sum())
    negative_violations = int((df["recommended_capital_fraction"] < -1e-9).sum())

    print(f"rows={len(df):,}")
    print(f"tickers={df['ticker'].nunique():,}")
    print(f"date range={df['date'].min().date()} → {df['date'].max().date()}")
    print(f"numeric finite ratio={finite_ratio:.6f}")
    print(f"max exposure={max_exposure:.2%}")
    print(f"cap violations={cap_violations}")
    print(f"max exposure violations={max_violations}")
    print(f"negative size violations={negative_violations}")
    print("\nposition summary:")
    print(df[["recommended_capital_fraction", "position_fraction_of_max", "combined_risk_score"]].describe().to_string())
    print("\nbinding cap counts:")
    print(df["binding_cap_source"].value_counts().to_string())
    print("\nsize bucket counts:")
    print(df["size_bucket"].value_counts().to_string())

    if cap_violations or max_violations or negative_violations:
        raise RuntimeError("Position sizing validation failed due to cap/size violations.")

    print("\nVALIDATION PASSED")


def cmd_smoke(config: PositionSizingConfig) -> None:
    print("=" * 100)
    print("POSITION SIZING SMOKE TEST")
    print("=" * 100)

    df = pd.DataFrame({
        "ticker": ["AAA", "BBB", "CCC"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"]),
        "trend_score": [0.8, 0.5, 0.3],
        "momentum_score": [0.8, 0.5, 0.3],
        "timing_confidence": [0.9, 0.5, 0.2],
        "vol_10d": [0.15, 0.30, 0.70],
        "vol_30d": [0.20, 0.45, 0.90],
        "regime_probs_high": [0.1, 0.5, 0.9],
        "drawdown_risk_score": [0.2, 0.6, 0.9],
        "cvar_95": [-0.02, -0.08, -0.18],
        "cvar_99": [-0.03, -0.12, -0.25],
        "tail_ratio_95": [1.2, 1.6, 2.3],
        "tail_ratio_99": [1.1, 1.8, 2.6],
        "contagion_5d": [0.2, 0.6, 0.9],
        "contagion_20d": [0.2, 0.7, 0.9],
        "contagion_60d": [0.2, 0.7, 0.9],
        "liquidity_score": [0.9, 0.5, 0.1],
        "slippage_estimate_pct": [0.001, 0.005, 0.03],
        "days_to_liquidate_1M": [0.1, 2.0, 20.0],
        "tradable": [1, 1, 0],
        "regime_label": ["calm", "volatile", "crisis"],
        "prob_calm": [0.9, 0.1, 0.0],
        "prob_volatile": [0.1, 0.8, 0.1],
        "prob_crisis": [0.0, 0.1, 0.9],
        "prob_rotation": [0.0, 0.0, 0.0],
        "macro_stress_score": [0.1, 0.5, 1.0],
        "learned_graph_stress": [0.1, 0.5, 1.0],
        "regime_confidence": [0.9, 0.8, 0.9],
    })

    weights = config.risk_weights()

    df["technical_confidence"] = compute_technical_confidence(df)
    df["technical_multiplier"] = technical_multiplier(df["technical_confidence"], config)
    df["volatility_risk_score"] = compute_volatility_risk(df, config)
    df["drawdown_risk_score"] = compute_drawdown_risk(df)
    df["var_cvar_risk_score"] = compute_var_cvar_risk(df, config)
    df["contagion_risk_score"] = compute_contagion_risk(df)
    df["liquidity_risk_score"] = compute_liquidity_risk(df)
    df["regime_risk_score"] = compute_regime_risk(df)

    df["combined_risk_score"] = (
        weights["volatility"] * df["volatility_risk_score"]
        + weights["drawdown"] * df["drawdown_risk_score"]
        + weights["var_cvar"] * df["var_cvar_risk_score"]
        + weights["contagion"] * df["contagion_risk_score"]
        + weights["liquidity"] * df["liquidity_risk_score"]
        + weights["regime"] * df["regime_risk_score"]
    )

    bucket_fraction, bucket_label = risk_to_bucket(df["combined_risk_score"])
    df["risk_bucket_fraction"] = bucket_fraction
    df["size_bucket"] = bucket_label

    max_exp = config.max_single_stock_exposure
    df["pre_cap_capital_fraction"] = bucket_fraction * technical_multiplier(df["technical_confidence"], config) * max_exp
    df["regime_hard_cap"] = compute_regime_hard_cap(df, config)
    df["liquidity_hard_cap"] = compute_liquidity_hard_cap(df, df["liquidity_risk_score"], config)
    df["binding_hard_cap"] = df[["regime_hard_cap", "liquidity_hard_cap"]].min(axis=1)
    df["recommended_capital_fraction"] = np.minimum(df["pre_cap_capital_fraction"], df["binding_hard_cap"])
    df["recommended_capital_pct"] = df["recommended_capital_fraction"] * 100.0

    print(df[[
        "ticker", "regime_label", "combined_risk_score", "size_bucket",
        "pre_cap_capital_fraction", "binding_hard_cap", "recommended_capital_pct"
    ]].to_string(index=False))

    assert df["recommended_capital_fraction"].between(0.0, max_exp).all()
    assert df.loc[df["ticker"] == "CCC", "recommended_capital_fraction"].iloc[0] == 0.0

    print("\nSMOKE TEST PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Position Sizing Engine")
    sub = parser.add_subparsers(dest="command")

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--repo-root", type=str, default="")
        p.add_argument("--output-dir", type=str, default="")
        p.add_argument("--exposure-mode", type=str, default="moderate", choices=["conservative", "moderate", "aggressive"])
        p.add_argument("--horizon-mode", type=str, default="short", choices=["short", "long"])

    p = sub.add_parser("inspect", help="Inspect available module outputs")
    add_common(p)

    p = sub.add_parser("smoke", help="Run synthetic smoke test")
    add_common(p)

    p = sub.add_parser("run", help="Run position sizing for one chunk/split")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    add_common(p)

    p = sub.add_parser("run-all", help="Run position sizing for multiple chunks/splits")
    p.add_argument("--chunks", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    p.add_argument("--splits", type=str, nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    add_common(p)

    p = sub.add_parser("validate", help="Validate one position sizing output")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    add_common(p)

    return parser


def config_from_args(args: argparse.Namespace) -> PositionSizingConfig:
    cfg = PositionSizingConfig()

    if getattr(args, "repo_root", ""):
        cfg.repo_root = args.repo_root
    if getattr(args, "output_dir", ""):
        cfg.output_dir = args.output_dir
        cfg.results_dir = str(Path(args.output_dir) / "results" / "PositionSizing")

    cfg.exposure_mode = getattr(args, "exposure_mode", "moderate")
    cfg.horizon_mode = getattr(args, "horizon_mode", "short")

    return cfg.resolve_paths()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    config = config_from_args(args)

    if args.command == "inspect":
        cmd_inspect(config)

    elif args.command == "smoke":
        cmd_smoke(config)

    elif args.command == "run":
        run_position_sizing(config, args.chunk, args.split)

    elif args.command == "run-all":
        for chunk in args.chunks:
            for split in args.splits:
                run_position_sizing(config, chunk, split)

    elif args.command == "validate":
        cmd_validate(config, args.chunk, args.split)


if __name__ == "__main__":
    main()


# ── Run instructions ─────────────────────────────────────────────────────────
# Compile:
# python -m py_compile code/riskEngine/position_sizing.py
#
# Inspect:
# python code/riskEngine/position_sizing.py inspect --repo-root .
#
# Smoke:
# python code/riskEngine/position_sizing.py smoke --repo-root .
#
# Run one:
# python code/riskEngine/position_sizing.py run     --repo-root . --chunk 1     --split test --exposure-mode moderate --horizon-mode short
#
# Run all val/test:
# python code/riskEngine/position_sizing.py run-all --repo-root . --chunks 1 2 3 --splits val test --exposure-mode moderate --horizon-mode short
#
# Validate:
# python code/riskEngine/position_sizing.py validate --repo-root . --chunk 1 --split test