#!/usr/bin/env python3
"""
code/analysts/quantitative_analyst.py

Quantitative Analyst
====================

Project:
    fin-glassbox — Explainable Distributed Deep Learning Framework for Financial Risk Management

Purpose:
    Convert technical signals + risk-engine outputs + position sizing output into a
    quantitative market recommendation.

Important:
    This module does NOT produce the final system decision.
    It produces the quantitative branch signal for the later Fusion Engine.

Consumes primarily:
    outputs/results/PositionSizing/position_sizing_chunk{chunk}_{split}.csv

Optionally consumes:
    outputs/results/TechnicalAnalyst/predictions_chunk{chunk}_{split}.csv
    only if technical columns are missing from PositionSizing output.

Produces:
    outputs/results/QuantitativeAnalyst/quantitative_analysis_chunk{chunk}_{split}.csv
    outputs/results/QuantitativeAnalyst/xai/quantitative_analysis_chunk{chunk}_{split}_xai_summary.json

CLI:
    python code/analysts/quantitative_analyst.py inspect --repo-root .
    python code/analysts/quantitative_analyst.py smoke --repo-root .
    python code/analysts/quantitative_analyst.py run --repo-root . --chunk 1 --split test
    python code/analysts/quantitative_analyst.py run-all --repo-root . --chunks 1 2 3 --splits val test
    python code/analysts/quantitative_analyst.py validate --repo-root . --chunk 1 --split test
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

RISK_WEIGHTS = {
    "volatility_risk_score": 0.20,
    "drawdown_risk_score": 0.15,
    "var_cvar_risk_score": 0.15,
    "contagion_risk_score": 0.25,
    "liquidity_risk_score": 0.15,
    "regime_risk_score": 0.10,
}


@dataclass
class QuantitativeAnalystConfig:
    repo_root: str = ""

    position_sizing_dir: str = "outputs/results/PositionSizing"
    technical_dir: str = "outputs/results/TechnicalAnalyst"
    results_dir: str = "outputs/results/QuantitativeAnalyst"

    buy_threshold: float = 0.18
    sell_threshold: float = -0.25
    severe_risk_threshold: float = 0.90
    high_risk_threshold: float = 0.75
    min_position_fraction: float = 0.0001

    # Direction weights from Technical Analyst
    trend_weight: float = 0.40
    momentum_weight: float = 0.35
    timing_weight: float = 0.25

    # Confidence weights
    confidence_technical_weight: float = 0.40
    confidence_risk_weight: float = 0.20
    confidence_position_weight: float = 0.25
    confidence_regime_weight: float = 0.15

    max_xai_examples: int = 100

    def resolve_paths(self) -> "QuantitativeAnalystConfig":
        if self.repo_root:
            root = Path(self.repo_root)
            for attr in ["position_sizing_dir", "technical_dir", "results_dir"]:
                value = getattr(self, attr)
                if value and not Path(value).is_absolute():
                    setattr(self, attr, str(root / value))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
# BASIC UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(json_safe(k)): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
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


def clip01(x: Any) -> Any:
    return np.clip(x, 0.0, 1.0)


def clip11(x: Any) -> Any:
    return np.clip(x, -1.0, 1.0)


def safe_numeric(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=np.float32)

    s = pd.to_numeric(df[col], errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).fillna(default)
    return s.astype(np.float32)


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return max(sum(1 for _ in f) - 1, 0)


def dedupe(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    keys = [k for k in keys if k in df.columns]
    if not keys:
        return df
    return df.drop_duplicates(keys, keep="last").reset_index(drop=True)


def require_cols(df: pd.DataFrame, cols: List[str], source_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{source_name} missing required columns: {missing}")


def position_path(config: QuantitativeAnalystConfig, chunk: int, split: str) -> Path:
    return Path(config.position_sizing_dir) / f"position_sizing_chunk{chunk}_{split}.csv"


def technical_path(config: QuantitativeAnalystConfig, chunk: int, split: str) -> Path:
    return Path(config.technical_dir) / f"predictions_chunk{chunk}_{split}.csv"


# ═══════════════════════════════════════════════════════════════════════════════
# INPUT LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_position_sizing(config: QuantitativeAnalystConfig, chunk: int, split: str) -> pd.DataFrame:
    path = position_path(config, chunk, split)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing Position Sizing output: {path}\n"
            f"Run: python code/riskEngine/position_sizing.py run --repo-root . --chunk {chunk} --split {split}"
        )

    df = pd.read_csv(path, dtype={"ticker": str})
    require_cols(df, ["ticker", "date"], "PositionSizing")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["ticker", "date"]).copy()
    df["ticker"] = df["ticker"].astype(str)
    df = dedupe(df, ["ticker", "date"])

    return df


def load_technical_optional(config: QuantitativeAnalystConfig, chunk: int, split: str) -> Optional[pd.DataFrame]:
    path = technical_path(config, chunk, split)
    if not path.exists():
        return None

    df = pd.read_csv(path, dtype={"ticker": str})
    if "ticker" not in df.columns or "date" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["ticker", "date"]).copy()
    df["ticker"] = df["ticker"].astype(str)

    keep = ["ticker", "date", "trend_score", "momentum_score", "timing_confidence"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = dedupe(df, ["ticker", "date"])

    return df


def ensure_technical_columns(
    df: pd.DataFrame,
    config: QuantitativeAnalystConfig,
    chunk: int,
    split: str,
) -> pd.DataFrame:
    needed = ["trend_score", "momentum_score", "timing_confidence"]
    if all(c in df.columns for c in needed):
        return df

    tech = load_technical_optional(config, chunk, split)
    if tech is None:
        for c in needed:
            if c not in df.columns:
                df[c] = 0.5
        return df

    for c in needed:
        if c in df.columns:
            continue
        if c in tech.columns:
            df = df.merge(tech[["ticker", "date", c]], on=["ticker", "date"], how="left")
        else:
            df[c] = 0.5

    for c in needed:
        df[c] = safe_numeric(df, c, 0.5)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTITATIVE SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_technical_direction(df: pd.DataFrame, config: QuantitativeAnalystConfig) -> pd.Series:
    trend = 2.0 * safe_numeric(df, "trend_score", 0.5) - 1.0
    momentum = 2.0 * safe_numeric(df, "momentum_score", 0.5) - 1.0
    timing = 2.0 * safe_numeric(df, "timing_confidence", 0.5) - 1.0

    score = (
        config.trend_weight * trend
        + config.momentum_weight * momentum
        + config.timing_weight * timing
    )

    return pd.Series(clip11(score), index=df.index, dtype=np.float32)


def compute_weighted_risk_driver(df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    contrib = pd.DataFrame(index=df.index)

    for col, weight in RISK_WEIGHTS.items():
        if col in df.columns:
            contrib[col.replace("_risk_score", "")] = safe_numeric(df, col, 0.5) * float(weight)
        else:
            contrib[col.replace("_risk_score", "")] = 0.5 * float(weight)

    arr = contrib.values
    idx = np.argmax(arr, axis=1)
    names = np.array(contrib.columns, dtype=object)

    return pd.Series(names[idx], index=df.index), contrib


def compute_quantitative_confidence(df: pd.DataFrame, config: QuantitativeAnalystConfig) -> pd.Series:
    technical_confidence = safe_numeric(df, "technical_confidence", 0.5)
    combined_risk = safe_numeric(df, "combined_risk_score", 0.5)
    position_fraction = safe_numeric(df, "position_fraction_of_max", 0.0)
    regime_confidence = safe_numeric(df, "regime_confidence", 0.5)

    risk_confidence = 1.0 - combined_risk

    confidence = (
        config.confidence_technical_weight * technical_confidence
        + config.confidence_risk_weight * risk_confidence
        + config.confidence_position_weight * position_fraction
        + config.confidence_regime_weight * regime_confidence
    )

    return pd.Series(clip01(confidence), index=df.index, dtype=np.float32)


def classify_risk_state(combined_risk: pd.Series) -> pd.Series:
    values = combined_risk.values
    labels = np.full(len(values), "low", dtype=object)

    labels[values >= 0.30] = "moderate"
    labels[values >= 0.50] = "elevated"
    labels[values >= 0.75] = "high"
    labels[values >= 0.90] = "severe"

    return pd.Series(labels, index=combined_risk.index)


def classify_action(
    direction: pd.Series,
    risk_adjusted_signal: pd.Series,
    combined_risk: pd.Series,
    recommended_capital_fraction: pd.Series,
    confidence: pd.Series,
    config: QuantitativeAnalystConfig,
) -> pd.Series:
    action = np.full(len(direction), "HOLD", dtype=object)

    direction_v = direction.values
    signal_v = risk_adjusted_signal.values
    risk_v = combined_risk.values
    pos_v = recommended_capital_fraction.values
    conf_v = confidence.values

    sell_mask = direction_v <= float(config.sell_threshold)
    buy_mask = (
        (signal_v >= float(config.buy_threshold))
        & (risk_v < float(config.severe_risk_threshold))
        & (pos_v > float(config.min_position_fraction))
        & (conf_v >= 0.35)
    )

    severe_risk_mask = risk_v >= float(config.severe_risk_threshold)
    no_position_mask = pos_v <= float(config.min_position_fraction)

    action[sell_mask] = "SELL"
    action[buy_mask] = "BUY"

    # If risk is severe but direction is not strongly negative, quantitative branch becomes HOLD, not BUY.
    action[severe_risk_mask & ~sell_mask] = "HOLD"
    action[no_position_mask & ~sell_mask] = "HOLD"

    return pd.Series(action, index=direction.index)


def build_risk_summary(df: pd.DataFrame) -> pd.Series:
    summaries = []

    for (
        top_driver,
        combined_risk,
        risk_state,
        cap_source,
        regime_label,
        recommended_pct,
    ) in zip(
        df["top_risk_driver"].astype(str),
        df["combined_risk_score"],
        df["quantitative_risk_state"].astype(str),
        df.get("binding_cap_source", pd.Series("unknown", index=df.index)).astype(str),
        df.get("regime_label", pd.Series("unknown", index=df.index)).astype(str),
        df["recommended_capital_pct"],
    ):
        summaries.append(
            f"Risk state is {risk_state}; top risk driver is {top_driver}; "
            f"binding cap is {cap_source}; regime={regime_label}; "
            f"recommended exposure={recommended_pct:.2f}%."
        )

    return pd.Series(summaries, index=df.index)


def build_xai_summary(df: pd.DataFrame) -> pd.Series:
    summaries = []

    for (
        action,
        signal,
        direction,
        risk,
        confidence,
        top_driver,
        pos_pct,
    ) in zip(
        df["quantitative_recommendation"].astype(str),
        df["risk_adjusted_quantitative_signal"],
        df["technical_direction_score"],
        df["quantitative_risk_score"],
        df["quantitative_confidence"],
        df["top_risk_driver"].astype(str),
        df["recommended_capital_pct"],
    ):
        summaries.append(
            f"{action}: risk-adjusted signal={signal:.3f}, "
            f"technical direction={direction:.3f}, risk={risk:.3f}, "
            f"confidence={confidence:.3f}, top risk={top_driver}, "
            f"size={pos_pct:.2f}%."
        )

    return pd.Series(summaries, index=df.index)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_quantitative_analysis(
    config: QuantitativeAnalystConfig,
    chunk: int,
    split: str,
) -> Dict[str, Any]:
    config.resolve_paths()

    print("=" * 90)
    print(f"QUANTITATIVE ANALYST — chunk{chunk}_{split}")
    print("=" * 90)

    df = load_position_sizing(config, chunk, split)
    df = ensure_technical_columns(df, config, chunk, split)

    required_risk_cols = [
        "combined_risk_score",
        "position_fraction_of_max",
        "recommended_capital_fraction",
        "recommended_capital_pct",
    ]
    for col in required_risk_cols:
        if col not in df.columns:
            raise ValueError(
                f"PositionSizing output is missing required column: {col}. "
                f"Re-run code/riskEngine/position_sizing.py for chunk{chunk}_{split}."
            )

    print(f"  input rows: {len(df):,}")
    print(f"  tickers: {df['ticker'].nunique():,}")
    print(f"  dates: {pd.to_datetime(df['date']).min().date()} → {pd.to_datetime(df['date']).max().date()}")

    df["technical_direction_score"] = compute_technical_direction(df, config)

    df["quantitative_risk_score"] = safe_numeric(df, "combined_risk_score", 0.5)
    df["quantitative_risk_state"] = classify_risk_state(df["quantitative_risk_score"])

    df["position_fraction_of_max"] = safe_numeric(df, "position_fraction_of_max", 0.0)
    df["recommended_capital_fraction"] = safe_numeric(df, "recommended_capital_fraction", 0.0)
    df["recommended_capital_pct"] = safe_numeric(df, "recommended_capital_pct", 0.0)

    df["top_risk_driver"], risk_contrib = compute_weighted_risk_driver(df)

    risk_gate = 1.0 - df["quantitative_risk_score"]
    risk_gate = pd.Series(clip01(risk_gate), index=df.index, dtype=np.float32)

    position_gate = np.where(
        df["position_fraction_of_max"].values <= float(config.min_position_fraction),
        0.0,
        0.50 + 0.50 * df["position_fraction_of_max"].values,
    )
    position_gate = pd.Series(clip01(position_gate), index=df.index, dtype=np.float32)

    df["risk_gate"] = risk_gate
    df["position_gate"] = position_gate

    # Final quantitative signal is directional technical signal scaled by risk safety and allowed size.
    df["risk_adjusted_quantitative_signal"] = (
        df["technical_direction_score"]
        * (0.30 + 0.70 * df["risk_gate"])
        * df["position_gate"]
    )
    df["risk_adjusted_quantitative_signal"] = clip11(df["risk_adjusted_quantitative_signal"])

    df["quantitative_confidence"] = compute_quantitative_confidence(df, config)

    df["quantitative_recommendation"] = classify_action(
        df["technical_direction_score"],
        df["risk_adjusted_quantitative_signal"],
        df["quantitative_risk_score"],
        df["recommended_capital_fraction"],
        df["quantitative_confidence"],
        config,
    )

    df["quantitative_action_strength"] = (
        np.abs(df["risk_adjusted_quantitative_signal"]) * df["quantitative_confidence"]
    ).astype(np.float32)

    df["risk_summary"] = build_risk_summary(df)
    df["xai_summary"] = build_xai_summary(df)

    df["chunk"] = int(chunk)
    df["split"] = split

    preferred_cols = [
        "ticker", "date", "chunk", "split",

        "quantitative_recommendation",
        "risk_adjusted_quantitative_signal",
        "technical_direction_score",
        "quantitative_risk_score",
        "quantitative_risk_state",
        "quantitative_confidence",
        "quantitative_action_strength",

        "recommended_capital_fraction",
        "recommended_capital_pct",
        "position_fraction_of_max",
        "max_single_stock_exposure",

        "top_risk_driver",
        "binding_cap_source",
        "hard_cap_applied",
        "size_bucket",
        "regime_label",
        "regime_confidence",

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

        "vol_10d",
        "vol_30d",
        "expected_drawdown_10d",
        "expected_drawdown_30d",
        "var_95",
        "var_99",
        "cvar_95",
        "cvar_99",
        "contagion_5d",
        "contagion_20d",
        "contagion_60d",
        "liquidity_score",

        "risk_summary",
        "xai_summary",
    ]

    cols = [c for c in preferred_cols if c in df.columns]
    extras = [c for c in df.columns if c not in cols and not c.startswith("_")]
    out_df = df[cols + extras].copy()

    results_dir = Path(config.results_dir)
    xai_dir = results_dir / "xai"
    results_dir.mkdir(parents=True, exist_ok=True)
    xai_dir.mkdir(parents=True, exist_ok=True)

    pred_path = results_dir / f"quantitative_analysis_chunk{chunk}_{split}.csv"
    xai_path = xai_dir / f"quantitative_analysis_chunk{chunk}_{split}_xai_summary.json"

    out_df.to_csv(pred_path, index=False)

    xai_report = build_xai_report(out_df, risk_contrib, config, chunk, split)

    with open(xai_path, "w") as f:
        json.dump(json_safe(xai_report), f, indent=2)

    print(f"  saved: {pred_path} rows={len(out_df):,}")
    print(f"  xai:   {xai_path}")
    print("  recommendation counts:")
    print(out_df["quantitative_recommendation"].value_counts().to_string())

    return {
        "predictions": out_df,
        "xai": xai_report,
        "paths": {
            "predictions": str(pred_path),
            "xai": str(xai_path),
        },
    }


def build_xai_report(
    df: pd.DataFrame,
    risk_contrib: pd.DataFrame,
    config: QuantitativeAnalystConfig,
    chunk: int,
    split: str,
) -> Dict[str, Any]:
    report = {
        "module": "QuantitativeAnalyst",
        "chunk": int(chunk),
        "split": split,
        "config": config.to_dict(),
        "rows": int(len(df)),
        "ticker_count": int(df["ticker"].nunique()) if "ticker" in df.columns else 0,
        "date_min": str(pd.to_datetime(df["date"]).min().date()) if len(df) else None,
        "date_max": str(pd.to_datetime(df["date"]).max().date()) if len(df) else None,
        "plain_english": (
            "The Quantitative Analyst combines the Technical Analyst signal with risk-engine and "
            "position-sizing outputs. Technical trend, momentum, and timing form the directional score. "
            "Risk score and allowed position size then reduce or suppress the signal. The module outputs "
            "a quantitative Buy/Hold/Sell recommendation for the later Fusion Engine, not the final trade decision."
        ),
        "recommendation_counts": df["quantitative_recommendation"].value_counts().to_dict(),
        "risk_state_counts": df["quantitative_risk_state"].value_counts().to_dict(),
        "top_risk_driver_counts": df["top_risk_driver"].value_counts().to_dict(),
        "summary_stats": {
            "risk_adjusted_signal_mean": float(df["risk_adjusted_quantitative_signal"].mean()),
            "risk_adjusted_signal_median": float(df["risk_adjusted_quantitative_signal"].median()),
            "risk_adjusted_signal_std": float(df["risk_adjusted_quantitative_signal"].std()),
            "quantitative_risk_mean": float(df["quantitative_risk_score"].mean()),
            "quantitative_confidence_mean": float(df["quantitative_confidence"].mean()),
            "recommended_capital_pct_mean": float(df["recommended_capital_pct"].mean()),
            "recommended_capital_pct_p95": float(df["recommended_capital_pct"].quantile(0.95)),
        },
        "risk_contribution_means": risk_contrib.mean().to_dict(),
    }

    example_cols = [
        "ticker", "date", "quantitative_recommendation",
        "risk_adjusted_quantitative_signal", "technical_direction_score",
        "quantitative_risk_score", "quantitative_confidence",
        "recommended_capital_pct", "top_risk_driver",
        "regime_label", "risk_summary", "xai_summary",
    ]
    example_cols = [c for c in example_cols if c in df.columns]

    report["strongest_buy_examples"] = (
        df[df["quantitative_recommendation"] == "BUY"]
        .sort_values("risk_adjusted_quantitative_signal", ascending=False)
        .head(config.max_xai_examples)[example_cols]
        .to_dict(orient="records")
    )

    report["strongest_sell_examples"] = (
        df[df["quantitative_recommendation"] == "SELL"]
        .sort_values("risk_adjusted_quantitative_signal", ascending=True)
        .head(config.max_xai_examples)[example_cols]
        .to_dict(orient="records")
    )

    report["highest_risk_examples"] = (
        df.sort_values("quantitative_risk_score", ascending=False)
        .head(config.max_xai_examples)[example_cols]
        .to_dict(orient="records")
    )

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# INSPECT / VALIDATE / SMOKE
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_inspect(config: QuantitativeAnalystConfig) -> None:
    config.resolve_paths()

    print("=" * 100)
    print("QUANTITATIVE ANALYST INPUT INSPECTION")
    print("=" * 100)

    for chunk in [1, 2, 3]:
        for split in ["train", "val", "test"]:
            print(f"\n========== chunk{chunk}_{split} ==========")

            p_pos = position_path(config, chunk, split)
            p_tech = technical_path(config, chunk, split)

            for label, path in [("PositionSizing", p_pos), ("TechnicalAnalyst", p_tech)]:
                exists = path.exists()
                rows = count_rows(path) if exists else 0
                print(f"{label:18s} {'OK' if exists else 'MISSING':8s} rows={rows:,} path={path}")

                if exists:
                    try:
                        df = pd.read_csv(path, nrows=1)
                        print(f"{'':18s} columns={list(df.columns)[:25]}")
                    except Exception as exc:
                        print(f"{'':18s} could not read columns: {exc}")


def cmd_validate(config: QuantitativeAnalystConfig, chunk: int, split: str) -> None:
    config.resolve_paths()

    path = Path(config.results_dir) / f"quantitative_analysis_chunk{chunk}_{split}.csv"

    print("=" * 100)
    print(f"QUANTITATIVE ANALYST VALIDATION — chunk{chunk}_{split}")
    print("=" * 100)

    if not path.exists():
        raise FileNotFoundError(f"Missing output: {path}")

    df = pd.read_csv(path, dtype={"ticker": str})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    required = [
        "ticker",
        "date",
        "quantitative_recommendation",
        "risk_adjusted_quantitative_signal",
        "technical_direction_score",
        "quantitative_risk_score",
        "quantitative_confidence",
        "recommended_capital_fraction",
        "recommended_capital_pct",
        "xai_summary",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    numeric = df.select_dtypes(include="number")
    finite_ratio = float(np.isfinite(numeric.values).mean()) if len(numeric.columns) else 1.0

    invalid_signal = int((df["risk_adjusted_quantitative_signal"].abs() > 1.000001).sum())
    invalid_direction = int((df["technical_direction_score"].abs() > 1.000001).sum())
    invalid_risk = int(((df["quantitative_risk_score"] < -1e-9) | (df["quantitative_risk_score"] > 1.000001)).sum())
    invalid_conf = int(((df["quantitative_confidence"] < -1e-9) | (df["quantitative_confidence"] > 1.000001)).sum())
    negative_position = int((df["recommended_capital_fraction"] < -1e-9).sum())

    print(f"rows={len(df):,}")
    print(f"tickers={df['ticker'].nunique():,}")
    print(f"date range={df['date'].min().date()} → {df['date'].max().date()}")
    print(f"numeric finite ratio={finite_ratio:.6f}")
    print(f"invalid_signal={invalid_signal}")
    print(f"invalid_direction={invalid_direction}")
    print(f"invalid_risk={invalid_risk}")
    print(f"invalid_confidence={invalid_conf}")
    print(f"negative_position={negative_position}")

    print("\nrecommendation counts:")
    print(df["quantitative_recommendation"].value_counts().to_string())

    print("\nrisk state counts:")
    print(df["quantitative_risk_state"].value_counts().to_string())

    print("\nsummary:")
    print(df[[
        "risk_adjusted_quantitative_signal",
        "technical_direction_score",
        "quantitative_risk_score",
        "quantitative_confidence",
        "recommended_capital_pct",
    ]].describe().to_string())

    if invalid_signal or invalid_direction or invalid_risk or invalid_conf or negative_position:
        raise RuntimeError("Quantitative Analyst validation failed.")

    print("\nVALIDATION PASSED")


def cmd_smoke(config: QuantitativeAnalystConfig) -> None:
    print("=" * 100)
    print("QUANTITATIVE ANALYST SMOKE TEST")
    print("=" * 100)

    df = pd.DataFrame({
        "ticker": ["AAA", "BBB", "CCC", "DDD"],
        "date": pd.to_datetime(["2024-01-01"] * 4),
        "trend_score": [0.90, 0.60, 0.20, 0.80],
        "momentum_score": [0.85, 0.55, 0.25, 0.80],
        "timing_confidence": [0.80, 0.50, 0.20, 0.70],
        "technical_confidence": [0.85, 0.55, 0.25, 0.80],
        "combined_risk_score": [0.20, 0.55, 0.50, 0.95],
        "position_fraction_of_max": [1.00, 0.50, 0.50, 0.00],
        "recommended_capital_fraction": [0.10, 0.05, 0.05, 0.00],
        "recommended_capital_pct": [10.0, 5.0, 5.0, 0.0],
        "regime_confidence": [0.90, 0.60, 0.50, 0.95],
        "regime_label": ["calm", "volatile", "volatile", "crisis"],
        "volatility_risk_score": [0.2, 0.5, 0.5, 0.9],
        "drawdown_risk_score": [0.2, 0.5, 0.5, 0.9],
        "var_cvar_risk_score": [0.2, 0.5, 0.5, 0.9],
        "contagion_risk_score": [0.2, 0.6, 0.5, 0.9],
        "liquidity_risk_score": [0.1, 0.3, 0.5, 0.9],
        "regime_risk_score": [0.1, 0.6, 0.6, 1.0],
        "binding_cap_source": ["regime_hard_cap", "drawdown_hard_cap", "regime_hard_cap", "liquidity_hard_cap"],
    })

    df["technical_direction_score"] = compute_technical_direction(df, config)
    df["quantitative_risk_score"] = safe_numeric(df, "combined_risk_score", 0.5)
    df["quantitative_risk_state"] = classify_risk_state(df["quantitative_risk_score"])
    df["top_risk_driver"], risk_contrib = compute_weighted_risk_driver(df)

    risk_gate = 1.0 - df["quantitative_risk_score"]
    position_gate = np.where(df["position_fraction_of_max"] <= config.min_position_fraction, 0.0, 0.50 + 0.50 * df["position_fraction_of_max"])

    df["risk_adjusted_quantitative_signal"] = (
        df["technical_direction_score"] * (0.30 + 0.70 * risk_gate) * position_gate
    )

    df["quantitative_confidence"] = compute_quantitative_confidence(df, config)

    df["quantitative_recommendation"] = classify_action(
        df["technical_direction_score"],
        df["risk_adjusted_quantitative_signal"],
        df["quantitative_risk_score"],
        df["recommended_capital_fraction"],
        df["quantitative_confidence"],
        config,
    )

    df["quantitative_action_strength"] = np.abs(df["risk_adjusted_quantitative_signal"]) * df["quantitative_confidence"]
    df["risk_summary"] = build_risk_summary(df)
    df["xai_summary"] = build_xai_summary(df)

    print(df[[
        "ticker",
        "technical_direction_score",
        "quantitative_risk_score",
        "risk_adjusted_quantitative_signal",
        "quantitative_confidence",
        "recommended_capital_pct",
        "quantitative_recommendation",
        "top_risk_driver",
    ]].to_string(index=False))

    assert df["risk_adjusted_quantitative_signal"].between(-1.0, 1.0).all()
    assert df["quantitative_risk_score"].between(0.0, 1.0).all()
    assert df["quantitative_confidence"].between(0.0, 1.0).all()
    assert set(df["quantitative_recommendation"]).issubset({"BUY", "HOLD", "SELL"})

    print("\nSMOKE TEST PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantitative Analyst")
    sub = parser.add_subparsers(dest="command")

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--repo-root", type=str, default="")
        p.add_argument("--position-sizing-dir", type=str, default="")
        p.add_argument("--technical-dir", type=str, default="")
        p.add_argument("--results-dir", type=str, default="")

    p = sub.add_parser("inspect", help="Inspect input files")
    add_common(p)

    p = sub.add_parser("smoke", help="Run synthetic smoke test")
    add_common(p)

    p = sub.add_parser("run", help="Run quantitative analyst for one chunk/split")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    add_common(p)

    p = sub.add_parser("run-all", help="Run quantitative analyst for multiple chunks/splits")
    p.add_argument("--chunks", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    p.add_argument("--splits", type=str, nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    add_common(p)

    p = sub.add_parser("validate", help="Validate one quantitative analyst output")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    add_common(p)

    return parser


def config_from_args(args: argparse.Namespace) -> QuantitativeAnalystConfig:
    cfg = QuantitativeAnalystConfig()

    if getattr(args, "repo_root", ""):
        cfg.repo_root = args.repo_root

    if getattr(args, "position_sizing_dir", ""):
        cfg.position_sizing_dir = args.position_sizing_dir

    if getattr(args, "technical_dir", ""):
        cfg.technical_dir = args.technical_dir

    if getattr(args, "results_dir", ""):
        cfg.results_dir = args.results_dir

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
        run_quantitative_analysis(config, args.chunk, args.split)

    elif args.command == "run-all":
        for chunk in args.chunks:
            for split in args.splits:
                run_quantitative_analysis(config, chunk, split)

    elif args.command == "validate":
        cmd_validate(config, args.chunk, args.split)


if __name__ == "__main__":
    main()


# ── Run instructions ─────────────────────────────────────────────────────────
# Compile:
# python -m py_compile code/analysts/quantitative_analyst.py
#
# Inspect:
# python code/analysts/quantitative_analyst.py inspect --repo-root .
#
# Smoke:
# python code/analysts/quantitative_analyst.py smoke --repo-root .
#
# Run one:
# python code/analysts/quantitative_analyst.py run --repo-root . --chunk 1 --split test
#
# Run all val/test:
# python code/analysts/quantitative_analyst.py run-all --repo-root . --chunks 1 2 3 --splits val test
#
# Validate:
# python code/analysts/quantitative_analyst.py validate --repo-root . --chunk 1 --split test