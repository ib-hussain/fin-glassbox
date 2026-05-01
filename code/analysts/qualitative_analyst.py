#!/usr/bin/env python3
"""
code/analysts/qualitative_analyst.py

Qualitative Analyst
===================

Project:
    fin-glassbox — Explainable Distributed Deep Learning Framework for Financial Risk Management

Purpose:
    Train a lightweight qualitative synthesis model that combines:
        - Sentiment Analyst outputs
        - News Analyst outputs

    into a qualitative branch signal for the later Fusion Engine.

Important:
    This module does NOT use fundamentals.
    This module does NOT make the final trade decision.
    It produces the qualitative branch output.

Inputs:
    Sentiment Analyst prediction CSVs under:
        outputs/results/analysts/sentiment/
        outputs/results/SentimentAnalyst/

    News Analyst prediction CSVs under:
        outputs/results/analysts/news/
        outputs/results/NewsAnalyst/

Outputs:
    outputs/models/QualitativeAnalyst/chunk{chunk}/best_model.pt
    outputs/models/QualitativeAnalyst/chunk{chunk}/scaler.npz
    outputs/codeResults/QualitativeAnalyst/best_params_chunk{chunk}.json

    outputs/results/QualitativeAnalyst/qualitative_events_chunk{chunk}_{split}.csv
    outputs/results/QualitativeAnalyst/qualitative_daily_chunk{chunk}_{split}.csv
    outputs/results/QualitativeAnalyst/xai/qualitative_chunk{chunk}_{split}_xai_summary.json

CLI:
    python code/analysts/qualitative_analyst.py inspect --repo-root .
    python code/analysts/qualitative_analyst.py smoke --repo-root . --device cuda
    python code/analysts/qualitative_analyst.py hpo --repo-root . --chunk 1 --trials 30 --device cuda --fresh
    python code/analysts/qualitative_analyst.py train-best --repo-root . --chunk 1 --device cuda --fresh
    python code/analysts/qualitative_analyst.py predict --repo-root . --chunk 1 --split test --device cuda
    python code/analysts/qualitative_analyst.py predict-all --repo-root . --chunks 1 2 3 --splits val test --device cuda
    python code/analysts/qualitative_analyst.py validate --repo-root . --chunk 1 --split test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    "sentiment_score",
    "sentiment_confidence",
    "sentiment_uncertainty",
    "sentiment_magnitude",
    "sentiment_event_present",
    "news_event_impact",
    "news_importance",
    "risk_relevance",
    "volatility_spike",
    "drawdown_risk",
    "news_uncertainty",
    "news_event_present",
    "has_both_sources",
]

TARGET_NAMES = [
    "qualitative_score_target",
    "qualitative_risk_target",
    "qualitative_confidence_target",
]

SENTIMENT_ALIASES = {
    "sentiment_score": [
        "predicted_sentiment_score",
        "sentiment_score",
        "sentiment_score_pred",
        "pred_sentiment_score",
        "score",
    ],
    "sentiment_class": [
        "predicted_sentiment_class",
        "sentiment_class",
        "pred_sentiment_class",
        "class",
    ],
    "sentiment_confidence": [
        "predicted_sentiment_confidence",
        "sentiment_confidence",
        "confidence",
        "pred_confidence",
    ],
    "sentiment_uncertainty": [
        "predicted_sentiment_uncertainty",
        "sentiment_uncertainty",
        "uncertainty",
    ],
    "sentiment_magnitude": [
        "predicted_sentiment_magnitude",
        "sentiment_magnitude",
        "magnitude",
    ],
}

NEWS_ALIASES = {
    "news_event_impact": [
        "predicted_news_event_impact",
        "news_event_impact",
        "event_impact",
        "predicted_event_impact",
    ],
    "news_importance": [
        "predicted_news_importance",
        "news_importance",
        "importance",
        "event_importance",
    ],
    "risk_relevance": [
        "predicted_risk_relevance",
        "risk_relevance",
        "risk_relevance_score",
    ],
    "volatility_spike": [
        "predicted_volatility_spike",
        "volatility_spike",
        "volatility_spike_risk",
    ],
    "drawdown_risk": [
        "predicted_drawdown_risk",
        "drawdown_risk",
        "drawdown_risk_score",
    ],
    "news_uncertainty": [
        "predicted_news_uncertainty",
        "news_uncertainty",
        "uncertainty",
    ],
}

KEY_ALIASES = {
    "ticker": ["ticker", "symbol", "stock", "company_ticker"],
    "date": ["date", "filing_date", "published_at", "timestamp", "event_date"],
    "doc_id": ["doc_id", "document_id", "chunk_id"],
    "accession": ["accession", "accession_number"],
    "form_type": ["form_type", "form"],
    "source_name": ["source_name", "section", "source"],
}


@dataclass
class QualitativeAnalystConfig:
    repo_root: str = ""

    output_dir: str = "outputs"
    sentiment_dirs: Tuple[str, ...] = (
        "outputs/results/analysts/sentiment",
        "outputs/results/SentimentAnalyst",
    )
    news_dirs: Tuple[str, ...] = (
        "outputs/results/analysts/news",
        "outputs/results/NewsAnalyst",
    )

    model_dir: str = "outputs/models/QualitativeAnalyst"
    results_dir: str = "outputs/results/QualitativeAnalyst"
    code_results_dir: str = "outputs/codeResults/QualitativeAnalyst"

    device: str = "cuda"

    hidden_dim: int = 64
    n_layers: int = 2
    dropout: float = 0.15
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    epochs: int = 40
    early_stop_patience: int = 6
    num_workers: int = 0

    hpo_trials: int = 30
    hpo_epochs: int = 10
    hpo_max_train_rows: int = 200_000
    hpo_max_val_rows: int = 75_000

    max_train_rows: int = 0
    seed: int = 42

    buy_threshold: float = 0.20
    sell_threshold: float = -0.20
    high_risk_sell_threshold: float = 0.85
    max_risk_for_buy: float = 0.70
    min_confidence_for_buy: float = 0.40

    xai_sample_size: int = 512
    max_xai_examples: int = 100

    def resolve_paths(self) -> "QualitativeAnalystConfig":
        if self.repo_root:
            root = Path(self.repo_root)

            for attr in ["output_dir", "model_dir", "results_dir", "code_results_dir"]:
                value = getattr(self, attr)
                if value and not Path(value).is_absolute():
                    setattr(self, attr, str(root / value))

            self.sentiment_dirs = tuple(
                str(root / p) if not Path(p).is_absolute() else str(Path(p))
                for p in self.sentiment_dirs
            )
            self.news_dirs = tuple(
                str(root / p) if not Path(p).is_absolute() else str(Path(p))
                for p in self.news_dirs
            )

        return self

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["sentiment_dirs"] = list(self.sentiment_dirs)
        d["news_dirs"] = list(self.news_dirs)
        return d


# ═══════════════════════════════════════════════════════════════════════════════
# GENERAL UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def find_first_col(df: pd.DataFrame, aliases: Iterable[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in lower_map:
            return lower_map[a.lower()]
    return None


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return max(sum(1 for _ in f) - 1, 0)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"ticker": str, "cik": str}, low_memory=False)


def is_prediction_file(path: Path) -> bool:
    s = str(path).lower()
    bad = ["hpo", "history", "training_history", "metrics", "summary", "xai"]
    if any(b in s for b in bad):
        return False
    if path.suffix.lower() != ".csv":
        return False
    return any(k in path.name.lower() for k in ["prediction", "predictions", "scores", "output"])


def discover_prediction_file(
    dirs: Iterable[str],
    chunk: int,
    split: str,
    kind: str,
) -> Optional[Path]:
    candidates: List[Path] = []

    exact_names = []
    if kind == "sentiment":
        exact_names = [
            f"chunk{chunk}_{split}_predictions.csv",
            f"sentiment_predictions_chunk{chunk}_{split}.csv",
            f"predictions_chunk{chunk}_{split}.csv",
        ]
    elif kind == "news":
        exact_names = [
            f"chunk{chunk}_{split}_news_predictions.csv",
            f"news_predictions_chunk{chunk}_{split}.csv",
            f"predictions_chunk{chunk}_{split}.csv",
        ]

    for d in dirs:
        root = Path(d)
        if not root.exists():
            continue

        for name in exact_names:
            p = root / name
            if p.exists():
                return p

        for p in root.rglob("*.csv"):
            s = str(p).lower()
            if f"chunk{chunk}" not in s or split.lower() not in s:
                continue
            if not is_prediction_file(p):
                continue
            if kind == "news" and "news" not in s and "analysts/news" not in s:
                continue
            if kind == "sentiment" and "sentiment" not in s and "analysts/sentiment" not in s:
                continue
            candidates.append(p)

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda p: (len(str(p)), str(p)))
    return candidates[0]


def dedupe(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    keys = [k for k in keys if k in df.columns]
    if not keys:
        return df.reset_index(drop=True)
    return df.drop_duplicates(keys, keep="last").reset_index(drop=True)


def recommendation_from_scores(
    score: pd.Series,
    risk: pd.Series,
    confidence: pd.Series,
    config: QualitativeAnalystConfig,
) -> pd.Series:
    rec = np.full(len(score), "HOLD", dtype=object)

    buy = (
        (score > float(config.buy_threshold))
        & (risk < float(config.max_risk_for_buy))
        & (confidence >= float(config.min_confidence_for_buy))
    )
    sell = (
        (score < float(config.sell_threshold))
        | (risk > float(config.high_risk_sell_threshold))
    )

    rec[buy.values] = "BUY"
    rec[sell.values] = "SELL"
    return pd.Series(rec, index=score.index)


def driver_from_row(row: pd.Series) -> str:
    drivers = {
        "sentiment": abs(float(row.get("sentiment_direction_score", 0.0))),
        "news_impact": abs(float(row.get("news_direction_score", 0.0))),
        "risk_relevance": float(row.get("risk_relevance", 0.0)),
        "volatility_spike": float(row.get("volatility_spike", 0.0)),
        "drawdown_risk": float(row.get("drawdown_risk", 0.0)),
        "news_uncertainty": float(row.get("news_uncertainty", 0.0)),
    }
    return max(drivers.items(), key=lambda kv: kv[1])[0]


# ═══════════════════════════════════════════════════════════════════════════════
# INPUT STANDARDISATION
# ═══════════════════════════════════════════════════════════════════════════════

def normalise_sentiment_class(values: pd.Series) -> pd.Series:
    if values.empty:
        return values.astype(np.float32)

    if pd.api.types.is_numeric_dtype(values):
        v = pd.to_numeric(values, errors="coerce").fillna(0.0)
        # Common encodings: 0/1/2 or -1/0/1.
        if v.min() >= 0 and v.max() <= 2:
            return (v - 1.0).clip(-1.0, 1.0).astype(np.float32)
        return v.clip(-1.0, 1.0).astype(np.float32)

    mapping = {
        "negative": -1.0,
        "neg": -1.0,
        "bearish": -1.0,
        "sell": -1.0,
        "neutral": 0.0,
        "hold": 0.0,
        "mixed": 0.0,
        "positive": 1.0,
        "pos": 1.0,
        "bullish": 1.0,
        "buy": 1.0,
    }

    return values.astype(str).str.lower().map(mapping).fillna(0.0).astype(np.float32)


def standardise_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for canonical, aliases in KEY_ALIASES.items():
        if canonical in out.columns:
            continue
        col = find_first_col(out, aliases)
        if col is not None:
            out[canonical] = out[col]

    if "ticker" not in out.columns:
        out["ticker"] = "UNKNOWN"

    if "date" not in out.columns:
        out["date"] = pd.NaT

    out["ticker"] = out["ticker"].astype(str)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    return out


def load_sentiment(config: QualitativeAnalystConfig, chunk: int, split: str) -> Optional[pd.DataFrame]:
    path = discover_prediction_file(config.sentiment_dirs, chunk, split, "sentiment")
    if path is None:
        return None

    raw = read_csv(path)
    df = standardise_keys(raw)

    if "sentiment_score" not in df.columns:
        col_score = find_first_col(df, SENTIMENT_ALIASES["sentiment_score"])
        if col_score:
            df["sentiment_score"] = pd.to_numeric(df[col_score], errors="coerce")

    if "sentiment_score" not in df.columns or df["sentiment_score"].isna().all():
        col_class = find_first_col(df, SENTIMENT_ALIASES["sentiment_class"])
        if col_class:
            df["sentiment_score"] = normalise_sentiment_class(df[col_class])
        else:
            df["sentiment_score"] = 0.0

    for canonical in ["sentiment_confidence", "sentiment_uncertainty", "sentiment_magnitude"]:
        if canonical not in df.columns:
            col = find_first_col(df, SENTIMENT_ALIASES[canonical])
            df[canonical] = pd.to_numeric(df[col], errors="coerce") if col else np.nan

    df["sentiment_score"] = clip11(safe_numeric(df, "sentiment_score", 0.0))
    df["sentiment_confidence"] = clip01(safe_numeric(df, "sentiment_confidence", 0.5))
    df["sentiment_uncertainty"] = clip01(safe_numeric(df, "sentiment_uncertainty", 1.0 - df["sentiment_confidence"].mean()))
    df["sentiment_magnitude"] = clip01(safe_numeric(df, "sentiment_magnitude", df["sentiment_score"].abs().mean()))
    df["sentiment_event_present"] = 1.0

    keep = [
        "ticker", "date", "doc_id", "accession", "form_type", "source_name",
        "sentiment_score", "sentiment_confidence", "sentiment_uncertainty",
        "sentiment_magnitude", "sentiment_event_present",
    ]

    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    df["_sentiment_path"] = str(path)

    return df


def load_news(config: QualitativeAnalystConfig, chunk: int, split: str) -> Optional[pd.DataFrame]:
    path = discover_prediction_file(config.news_dirs, chunk, split, "news")
    if path is None:
        return None

    raw = read_csv(path)
    df = standardise_keys(raw)

    for canonical, aliases in NEWS_ALIASES.items():
        if canonical not in df.columns:
            col = find_first_col(df, aliases)
            df[canonical] = pd.to_numeric(df[col], errors="coerce") if col else np.nan

    df["news_event_impact"] = clip11(safe_numeric(df, "news_event_impact", 0.0))
    df["news_importance"] = clip01(safe_numeric(df, "news_importance", 0.0))
    df["risk_relevance"] = clip01(safe_numeric(df, "risk_relevance", 0.0))
    df["volatility_spike"] = clip01(safe_numeric(df, "volatility_spike", 0.0))
    df["drawdown_risk"] = clip01(safe_numeric(df, "drawdown_risk", 0.0))
    df["news_uncertainty"] = clip01(safe_numeric(df, "news_uncertainty", 0.5))
    df["news_event_present"] = 1.0

    keep = [
        "ticker", "date", "doc_id", "accession", "form_type", "source_name",
        "news_event_impact", "news_importance", "risk_relevance",
        "volatility_spike", "drawdown_risk", "news_uncertainty",
        "news_event_present",
    ]

    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    df["_news_path"] = str(path)

    return df


def merge_sentiment_news(
    sentiment: Optional[pd.DataFrame],
    news: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if sentiment is None and news is None:
        raise FileNotFoundError("No sentiment or news prediction files were found.")

    if sentiment is not None and news is None:
        df = sentiment.copy()
        df["news_event_present"] = 0.0

    elif sentiment is None and news is not None:
        df = news.copy()
        df["sentiment_event_present"] = 0.0

    else:
        assert sentiment is not None and news is not None

        common_keys = [k for k in ["ticker", "date", "doc_id", "accession"] if k in sentiment.columns and k in news.columns]

        # Use exact document merge only when doc_id or accession exists.
        # Otherwise concatenate to avoid accidental many-to-many ticker/date explosions.
        if "doc_id" in common_keys or "accession" in common_keys:
            df = sentiment.merge(news, on=common_keys, how="outer", suffixes=("_sent", "_news"))
            for c in ["form_type", "source_name"]:
                sent_c = f"{c}_sent"
                news_c = f"{c}_news"
                if c not in df.columns:
                    if sent_c in df.columns and news_c in df.columns:
                        df[c] = df[sent_c].fillna(df[news_c])
                    elif sent_c in df.columns:
                        df[c] = df[sent_c]
                    elif news_c in df.columns:
                        df[c] = df[news_c]
        else:
            df = pd.concat([sentiment, news], ignore_index=True, sort=False)

    defaults = {
        "sentiment_score": 0.0,
        "sentiment_confidence": 0.0,
        "sentiment_uncertainty": 1.0,
        "sentiment_magnitude": 0.0,
        "sentiment_event_present": 0.0,
        "news_event_impact": 0.0,
        "news_importance": 0.0,
        "risk_relevance": 0.0,
        "volatility_spike": 0.0,
        "drawdown_risk": 0.0,
        "news_uncertainty": 0.5,
        "news_event_present": 0.0,
    }

    for c, default in defaults.items():
        if c not in df.columns:
            df[c] = default
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(default)

    if "ticker" not in df.columns:
        df["ticker"] = "UNKNOWN"
    if "date" not in df.columns:
        df["date"] = pd.NaT

    df["ticker"] = df["ticker"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    df["has_both_sources"] = (
        (df["sentiment_event_present"] > 0.5)
        & (df["news_event_present"] > 0.5)
    ).astype(np.float32)

    if "doc_id" not in df.columns:
        df["doc_id"] = ""
    if "accession" not in df.columns:
        df["accession"] = ""
    if "form_type" not in df.columns:
        df["form_type"] = ""
    if "source_name" not in df.columns:
        df["source_name"] = ""

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TARGETS AND FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def construct_weak_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    sentiment_direction = clip11(out["sentiment_score"].values.astype(np.float32))
    news_direction = clip11(
        out["news_event_impact"].values.astype(np.float32)
        * out["news_importance"].values.astype(np.float32)
    )

    qualitative_score = clip11(0.55 * sentiment_direction + 0.45 * news_direction)

    qualitative_risk = clip01(
        0.35 * out["risk_relevance"].values.astype(np.float32)
        + 0.30 * out["volatility_spike"].values.astype(np.float32)
        + 0.25 * out["drawdown_risk"].values.astype(np.float32)
        + 0.10 * out["news_uncertainty"].values.astype(np.float32)
    )

    qualitative_confidence = clip01(
        0.40 * out["sentiment_confidence"].values.astype(np.float32)
        + 0.25 * (1.0 - out["sentiment_uncertainty"].values.astype(np.float32))
        + 0.20 * out["news_importance"].values.astype(np.float32)
        + 0.15 * (1.0 - out["news_uncertainty"].values.astype(np.float32))
    )

    # If there is no source evidence, force low confidence.
    no_source = (
        (out["sentiment_event_present"].values.astype(np.float32) < 0.5)
        & (out["news_event_present"].values.astype(np.float32) < 0.5)
    )
    qualitative_confidence[no_source] = 0.0

    out["qualitative_score_target"] = qualitative_score.astype(np.float32)
    out["qualitative_risk_target"] = qualitative_risk.astype(np.float32)
    out["qualitative_confidence_target"] = qualitative_confidence.astype(np.float32)

    return out


def prepare_features_and_targets(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = construct_weak_targets(df)

    for c in FEATURE_NAMES:
        if c not in df.columns:
            df[c] = 0.0

    x = df[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32)
    y = df[TARGET_NAMES].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32)

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    y[:, 0] = clip11(y[:, 0])
    y[:, 1] = clip01(y[:, 1])
    y[:, 2] = clip01(y[:, 2])

    return x, y, df


def load_qualitative_frame(config: QualitativeAnalystConfig, chunk: int, split: str) -> pd.DataFrame:
    config.resolve_paths()
    sent = load_sentiment(config, chunk, split)
    news = load_news(config, chunk, split)
    df = merge_sentiment_news(sent, news)

    df["chunk"] = int(chunk)
    df["split"] = split

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SCALER
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureScaler:
    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> None:
        self.mean = x.mean(axis=0, keepdims=True).astype(np.float32)
        self.std = x.std(axis=0, keepdims=True).astype(np.float32)
        self.std[self.std < 1e-8] = 1.0

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return x.astype(np.float32)
        return ((x - self.mean) / self.std).astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path), mean=self.mean, std=self.std, feature_names=np.array(FEATURE_NAMES, dtype=object))

    @classmethod
    def load(cls, path: Path) -> "FeatureScaler":
        data = np.load(str(path), allow_pickle=True)
        obj = cls()
        obj.mean = data["mean"].astype(np.float32)
        obj.std = data["std"].astype(np.float32)
        return obj


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class QualitativeMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        in_dim = input_dim

        for _ in range(int(n_layers)):
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(float(dropout)))
            in_dim = int(hidden_dim)

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 3)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        h = self.backbone(x)
        raw = self.head(h)

        qualitative_score = torch.tanh(raw[:, 0])
        qualitative_risk = torch.sigmoid(raw[:, 1])
        qualitative_confidence = torch.sigmoid(raw[:, 2])

        out = torch.stack([qualitative_score, qualitative_risk, qualitative_confidence], dim=1)

        return {
            "outputs": out,
            "qualitative_score": qualitative_score,
            "qualitative_risk": qualitative_risk,
            "qualitative_confidence": qualitative_confidence,
            "hidden": h,
        }


def model_path(config: QualitativeAnalystConfig, chunk: int) -> Path:
    return Path(config.model_dir) / f"chunk{chunk}" / "best_model.pt"


def scaler_path(config: QualitativeAnalystConfig, chunk: int) -> Path:
    return Path(config.model_dir) / f"chunk{chunk}" / "scaler.npz"


def final_model_path(config: QualitativeAnalystConfig, chunk: int) -> Path:
    return Path(config.model_dir) / f"chunk{chunk}" / "final_model.pt"


def save_model(
    model: QualitativeMLP,
    config: QualitativeAnalystConfig,
    chunk: int,
    best_val_loss: float,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": config.to_dict(),
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "best_val_loss": float(best_val_loss),
    }
    torch.save(payload, path)


def load_model(config: QualitativeAnalystConfig, chunk: int) -> Tuple[QualitativeMLP, FeatureScaler, Dict[str, Any]]:
    config.resolve_paths()
    path = model_path(config, chunk)
    if not path.exists():
        raise FileNotFoundError(f"Missing qualitative model: {path}")

    payload = torch.load(path, map_location=config.device)

    model_cfg = payload.get("config", {})
    hidden_dim = int(model_cfg.get("hidden_dim", config.hidden_dim))
    n_layers = int(model_cfg.get("n_layers", config.n_layers))
    dropout = float(model_cfg.get("dropout", config.dropout))

    model = QualitativeMLP(
        input_dim=len(FEATURE_NAMES),
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
    ).to(config.device)

    model.load_state_dict(payload["state_dict"])
    model.eval()

    scaler = FeatureScaler.load(scaler_path(config, chunk))
    return model, scaler, payload


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    device: str,
    num_workers: int = 0,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
    )

    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=str(device).startswith("cuda"),
        drop_last=False,
    )


def qualitative_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    score_loss = torch.mean((pred[:, 0] - target[:, 0]) ** 2)
    risk_loss = torch.mean((pred[:, 1] - target[:, 1]) ** 2)
    conf_loss = torch.mean((pred[:, 2] - target[:, 2]) ** 2)

    return 0.40 * score_loss + 0.35 * risk_loss + 0.25 * conf_loss


def train_epoch(
    model: QualitativeMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()
    total = 0.0
    n = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(xb)["outputs"]
        loss = qualitative_loss(out, yb)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite qualitative training loss detected: {float(loss.detach().cpu())}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        bs = xb.size(0)
        total += float(loss.detach().cpu()) * bs
        n += bs

    return total / max(n, 1)


@torch.no_grad()
def validate_epoch(model: QualitativeMLP, loader: DataLoader, device: str) -> float:
    model.eval()
    total = 0.0
    n = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        out = model(xb)["outputs"]
        loss = qualitative_loss(out, yb)

        bs = xb.size(0)
        total += float(loss.detach().cpu()) * bs
        n += bs

    return total / max(n, 1)


def maybe_sample(x: np.ndarray, y: np.ndarray, max_rows: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if max_rows <= 0 or len(x) <= max_rows:
        return x, y

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=int(max_rows), replace=False)
    idx = np.sort(idx)
    return x[idx], y[idx]


def train_qualitative_model(
    config: QualitativeAnalystConfig,
    chunk: int,
    *,
    fresh: bool = False,
    hpo_mode: bool = False,
    run_tag: str = "",
) -> Tuple[QualitativeMLP, float, Dict[str, Any]]:
    config.resolve_paths()
    set_seed(config.seed)

    out_dir = Path(config.model_dir) / f"chunk{chunk}"
    if fresh and out_dir.exists() and not hpo_mode:
        print(f"  Fresh run requested. Removing: {out_dir}")
        shutil.rmtree(out_dir)

    train_df = load_qualitative_frame(config, chunk, "train")
    val_df = load_qualitative_frame(config, chunk, "val")

    x_train, y_train, train_df = prepare_features_and_targets(train_df)
    x_val, y_val, val_df = prepare_features_and_targets(val_df)

    max_train = int(config.hpo_max_train_rows if hpo_mode else config.max_train_rows)
    max_val = int(config.hpo_max_val_rows if hpo_mode else 0)

    x_train, y_train = maybe_sample(x_train, y_train, max_train, config.seed)
    x_val, y_val = maybe_sample(x_val, y_val, max_val, config.seed + 1)

    scaler = FeatureScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)

    model = QualitativeMLP(
        input_dim=len(FEATURE_NAMES),
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
        x_train, y_train,
        batch_size=config.batch_size,
        shuffle=True,
        device=config.device,
        num_workers=config.num_workers,
    )

    val_loader = make_loader(
        x_val, y_val,
        batch_size=config.batch_size,
        shuffle=False,
        device=config.device,
        num_workers=config.num_workers,
    )

    epochs = int(config.hpo_epochs if hpo_mode else config.epochs)

    print(f"  Train events: {len(x_train):,} | Val events: {len(x_val):,}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Config: hidden={config.hidden_dim}, layers={config.n_layers}, dropout={config.dropout:.3f}, batch={config.batch_size}")

    best_val = float("inf")
    best_state = None
    no_improve = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, config.device)
        val_loss = validate_epoch(model, val_loader, config.device)

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "lr": float(optimizer.param_groups[0]["lr"]),
        })

        prefix = f"[{run_tag}]" if run_tag else f"[chunk{chunk}]"
        print(f"  {prefix} E{epoch:03d}/{epochs} | train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_val - 1e-6:
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
        save_model(model, config, chunk, best_val, model_path(config, chunk))
        save_model(model, config, chunk, best_val, final_model_path(config, chunk))
        scaler.save(scaler_path(config, chunk))

        history_path = out_dir / "training_history.csv"
        pd.DataFrame(history).to_csv(history_path, index=False)

        freeze_dir = out_dir / "model_freezed"
        unfreeze_dir = out_dir / "model_unfreezed"
        freeze_dir.mkdir(parents=True, exist_ok=True)
        unfreeze_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(model_path(config, chunk), freeze_dir / "model.pt")
        shutil.copy2(model_path(config, chunk), unfreeze_dir / "model.pt")

    summary = {
        "chunk": int(chunk),
        "best_val_loss": float(best_val),
        "train_events": int(len(x_train)),
        "val_events": int(len(x_val)),
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "config": config.to_dict(),
        "history": history,
    }

    return model, float(best_val), summary


# ═══════════════════════════════════════════════════════════════════════════════
# HPO
# ═══════════════════════════════════════════════════════════════════════════════

def hpo_objective(trial: Any, base_config: QualitativeAnalystConfig, chunk: int) -> float:
    config = QualitativeAnalystConfig(**base_config.to_dict())
    config.resolve_paths()

    config.hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 96, 128])
    config.n_layers = trial.suggest_int("n_layers", 1, 3)
    config.dropout = trial.suggest_float("dropout", 0.05, 0.40)
    config.lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    config.weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
    config.batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    config.hpo_epochs = int(base_config.hpo_epochs)
    config.early_stop_patience = 3

    try:
        _, val_loss, _ = train_qualitative_model(
            config,
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


def run_hpo(config: QualitativeAnalystConfig, chunk: int, trials: int, fresh: bool = False) -> Dict[str, Any]:
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
    study_name = f"qualitative_analyst_chunk{chunk}"

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
        raise RuntimeError("All Qualitative Analyst HPO trials failed.")

    best = study.best_trial

    result = {
        "study_name": study_name,
        "best_value": float(best.value),
        "best_params": best.params,
        "trials": len(study.trials),
        "storage": storage,
    }

    out_path = out_dir / f"best_params_chunk{chunk}.json"
    with open(out_path, "w") as f:
        json.dump(json_safe(result), f, indent=2)

    print(f"  Best HPO: {best.params} (val_loss={best.value:.6f})")
    print(f"  Saved: {out_path}")

    return result


def load_best_params(config: QualitativeAnalystConfig, chunk: int) -> Optional[Dict[str, Any]]:
    path = Path(config.code_results_dir) / f"best_params_chunk{chunk}.json"
    if not path.exists():
        return None
    with open(path) as f:
        obj = json.load(f)
    return obj.get("best_params", obj)


def apply_best_params(config: QualitativeAnalystConfig, params: Dict[str, Any]) -> QualitativeAnalystConfig:
    for key, value in params.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION + AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_events(
    config: QualitativeAnalystConfig,
    chunk: int,
    split: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    config.resolve_paths()
    model, scaler, payload = load_model(config, chunk)

    raw_df = load_qualitative_frame(config, chunk, split)
    x, _, event_df = prepare_features_and_targets(raw_df)
    x_scaled = scaler.transform(x)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_scaled.astype(np.float32))),
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        pin_memory=str(config.device).startswith("cuda"),
    )

    preds: List[np.ndarray] = []
    for (xb,) in loader:
        xb = xb.to(config.device, non_blocking=True)
        out = model(xb)["outputs"].detach().float().cpu().numpy()
        preds.append(out)

    pred = np.concatenate(preds, axis=0) if preds else np.zeros((0, 3), dtype=np.float32)

    event_df["sentiment_direction_score"] = clip11(event_df["sentiment_score"].values)
    event_df["news_direction_score"] = clip11(event_df["news_event_impact"].values * event_df["news_importance"].values)

    event_df["qualitative_score"] = clip11(pred[:, 0])
    event_df["qualitative_risk_score"] = clip01(pred[:, 1])
    event_df["qualitative_confidence"] = clip01(pred[:, 2])

    event_df["qualitative_recommendation"] = recommendation_from_scores(
        event_df["qualitative_score"],
        event_df["qualitative_risk_score"],
        event_df["qualitative_confidence"],
        config,
    )

    event_df["dominant_qualitative_driver"] = event_df.apply(driver_from_row, axis=1)

    event_df["xai_summary"] = [
        (
            f"{rec}: qualitative_score={score:.3f}, risk={risk:.3f}, confidence={conf:.3f}; "
            f"driver={driver}; sentiment={sent:.3f}; news_impact={news:.3f}."
        )
        for rec, score, risk, conf, driver, sent, news in zip(
            event_df["qualitative_recommendation"],
            event_df["qualitative_score"],
            event_df["qualitative_risk_score"],
            event_df["qualitative_confidence"],
            event_df["dominant_qualitative_driver"],
            event_df["sentiment_direction_score"],
            event_df["news_direction_score"],
        )
    ]

    meta = {
        "model_best_val_loss": payload.get("best_val_loss"),
        "rows": int(len(event_df)),
    }

    return event_df, meta


def weighted_mean(values: np.ndarray, weights: np.ndarray, default: float = 0.0) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.nan_to_num(values, nan=default, posinf=default, neginf=default)

    total = weights.sum()
    if total <= 1e-12:
        return float(np.mean(values)) if len(values) else float(default)
    return float(np.sum(values * weights) / total)


def aggregate_daily(event_df: pd.DataFrame, config: QualitativeAnalystConfig, chunk: int, split: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    group_cols = ["ticker", "date"]

    for (ticker, date), g in event_df.groupby(group_cols, sort=True):
        conf = g["qualitative_confidence"].values
        q_score = weighted_mean(g["qualitative_score"].values, conf, default=0.0)

        weighted_risk = weighted_mean(g["qualitative_risk_score"].values, conf, default=0.0)
        max_risk = float(np.nanmax(g["qualitative_risk_score"].values)) if len(g) else 0.0
        q_risk = float(clip01(0.60 * max_risk + 0.40 * weighted_risk))

        mean_conf = float(np.nanmean(conf)) if len(conf) else 0.0
        event_count = int(len(g))
        count_boost = min(1.0, math.log1p(event_count) / math.log(5.0))
        q_conf = float(clip01(mean_conf * count_boost))

        rec = recommendation_from_scores(
            pd.Series([q_score]),
            pd.Series([q_risk]),
            pd.Series([q_conf]),
            config,
        ).iloc[0]

        driver_counts = g["dominant_qualitative_driver"].value_counts()
        driver = str(driver_counts.index[0]) if len(driver_counts) else "unknown"

        rows.append({
            "ticker": str(ticker),
            "date": pd.to_datetime(date),
            "chunk": int(chunk),
            "split": split,

            "event_count": event_count,
            "sentiment_event_count": int((g["sentiment_event_present"] > 0.5).sum()),
            "news_event_count": int((g["news_event_present"] > 0.5).sum()),

            "qualitative_score": q_score,
            "qualitative_risk_score": q_risk,
            "qualitative_confidence": q_conf,
            "qualitative_recommendation": rec,

            "max_event_risk_score": max_risk,
            "mean_event_risk_score": float(np.nanmean(g["qualitative_risk_score"].values)),
            "mean_sentiment_score": float(np.nanmean(g["sentiment_score"].values)),
            "mean_news_impact_score": float(np.nanmean(g["news_event_impact"].values)),
            "mean_news_importance": float(np.nanmean(g["news_importance"].values)),

            "dominant_qualitative_driver": driver,
            "xai_summary": (
                f"{rec}: daily qualitative_score={q_score:.3f}, risk={q_risk:.3f}, "
                f"confidence={q_conf:.3f}, events={event_count}, dominant_driver={driver}."
            ),
        })

    return pd.DataFrame(rows)


def gradient_xai(
    model: QualitativeMLP,
    scaler: FeatureScaler,
    x_raw: np.ndarray,
    device: str,
    max_rows: int,
) -> Dict[str, Any]:
    if len(x_raw) == 0:
        return {"feature_importance": {}}

    x_raw = x_raw[:max_rows]
    x_scaled = scaler.transform(x_raw)

    x = torch.from_numpy(x_scaled.astype(np.float32)).to(device)
    x.requires_grad_(True)

    model.eval()
    out = model(x)["outputs"]
    score = out[:, 0].mean() + out[:, 1].mean() + out[:, 2].mean()
    score.backward()

    grad = x.grad.detach().abs().mean(dim=0).cpu().numpy()
    if grad.sum() > 0:
        grad = grad / grad.sum()

    importance = {name: float(val) for name, val in zip(FEATURE_NAMES, grad)}
    importance = dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))

    return {
        "method": "input_gradient_importance",
        "feature_importance": importance,
    }


def build_xai_report(
    event_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    grad_xai: Dict[str, Any],
    config: QualitativeAnalystConfig,
    chunk: int,
    split: str,
    model_meta: Dict[str, Any],
) -> Dict[str, Any]:
    report = {
        "module": "QualitativeAnalyst",
        "chunk": int(chunk),
        "split": split,
        "config": config.to_dict(),
        "model_meta": model_meta,
        "event_rows": int(len(event_df)),
        "daily_rows": int(len(daily_df)),
        "ticker_count": int(daily_df["ticker"].nunique()) if len(daily_df) else 0,
        "date_min": str(pd.to_datetime(daily_df["date"]).min().date()) if len(daily_df) else None,
        "date_max": str(pd.to_datetime(daily_df["date"]).max().date()) if len(daily_df) else None,
        "event_recommendation_counts": event_df["qualitative_recommendation"].value_counts().to_dict() if len(event_df) else {},
        "daily_recommendation_counts": daily_df["qualitative_recommendation"].value_counts().to_dict() if len(daily_df) else {},
        "dominant_driver_counts": event_df["dominant_qualitative_driver"].value_counts().to_dict() if len(event_df) else {},
        "gradient_xai": grad_xai,
        "plain_english": (
            "The Qualitative Analyst is a trained synthesis model. It learns a calibrated mapping from "
            "Sentiment Analyst and News Analyst outputs into qualitative direction, qualitative risk, and "
            "qualitative confidence. Event-level outputs preserve document evidence. Daily outputs aggregate "
            "ticker-date evidence for the Fusion Engine. Missing qualitative evidence should be interpreted "
            "as no text signal, not as a positive signal."
        ),
        "summary_stats": {
            "mean_qualitative_score": float(daily_df["qualitative_score"].mean()) if len(daily_df) else None,
            "mean_qualitative_risk": float(daily_df["qualitative_risk_score"].mean()) if len(daily_df) else None,
            "mean_qualitative_confidence": float(daily_df["qualitative_confidence"].mean()) if len(daily_df) else None,
        },
    }

    example_cols = [
        "ticker", "date", "qualitative_recommendation", "qualitative_score",
        "qualitative_risk_score", "qualitative_confidence",
        "dominant_qualitative_driver", "xai_summary",
    ]

    report["top_positive_events"] = (
        event_df.sort_values("qualitative_score", ascending=False)
        .head(config.max_xai_examples)[[c for c in example_cols if c in event_df.columns]]
        .to_dict(orient="records")
    )

    report["top_negative_events"] = (
        event_df.sort_values("qualitative_score", ascending=True)
        .head(config.max_xai_examples)[[c for c in example_cols if c in event_df.columns]]
        .to_dict(orient="records")
    )

    report["highest_risk_events"] = (
        event_df.sort_values("qualitative_risk_score", ascending=False)
        .head(config.max_xai_examples)[[c for c in example_cols if c in event_df.columns]]
        .to_dict(orient="records")
    )

    return report


def predict_qualitative(config: QualitativeAnalystConfig, chunk: int, split: str) -> Dict[str, Any]:
    config.resolve_paths()

    print("=" * 90)
    print(f"QUALITATIVE ANALYST PREDICT — chunk{chunk}_{split}")
    print("=" * 90)

    event_df, meta = predict_events(config, chunk, split)
    daily_df = aggregate_daily(event_df, config, chunk, split)

    model, scaler, _ = load_model(config, chunk)
    x_raw, _, _ = prepare_features_and_targets(event_df.copy())
    grad = gradient_xai(model, scaler, x_raw, config.device, config.xai_sample_size)

    results_dir = Path(config.results_dir)
    xai_dir = results_dir / "xai"
    results_dir.mkdir(parents=True, exist_ok=True)
    xai_dir.mkdir(parents=True, exist_ok=True)

    event_path = results_dir / f"qualitative_events_chunk{chunk}_{split}.csv"
    daily_path = results_dir / f"qualitative_daily_chunk{chunk}_{split}.csv"
    xai_path = xai_dir / f"qualitative_chunk{chunk}_{split}_xai_summary.json"

    event_df.to_csv(event_path, index=False)
    daily_df.to_csv(daily_path, index=False)

    xai = build_xai_report(event_df, daily_df, grad, config, chunk, split, meta)
    with open(xai_path, "w") as f:
        json.dump(json_safe(xai), f, indent=2)

    print(f"  event output: {event_path} rows={len(event_df):,}")
    print(f"  daily output: {daily_path} rows={len(daily_df):,}")
    print(f"  xai output:   {xai_path}")
    if len(daily_df):
        print("  daily recommendation counts:")
        print(daily_df["qualitative_recommendation"].value_counts().to_string())

    return {
        "events": event_df,
        "daily": daily_df,
        "xai": xai,
        "paths": {
            "events": str(event_path),
            "daily": str(daily_path),
            "xai": str(xai_path),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# INSPECT / VALIDATE / SMOKE
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_inspect(config: QualitativeAnalystConfig) -> None:
    config.resolve_paths()

    print("=" * 100)
    print("QUALITATIVE ANALYST INPUT INSPECTION")
    print("=" * 100)

    for chunk in [1, 2, 3]:
        for split in ["train", "val", "test"]:
            sent = discover_prediction_file(config.sentiment_dirs, chunk, split, "sentiment")
            news = discover_prediction_file(config.news_dirs, chunk, split, "news")

            print(f"\n========== chunk{chunk}_{split} ==========")

            for label, path in [("sentiment", sent), ("news", news)]:
                if path is None:
                    print(f"{label:10s} MISSING")
                    continue

                rows = count_rows(path)
                print(f"{label:10s} OK rows={rows:,} path={path}")

                try:
                    df = pd.read_csv(path, nrows=2)
                    print(f"{'':10s} columns={list(df.columns)}")
                    print(df.head(2).to_string(index=False))
                except Exception as exc:
                    print(f"{'':10s} could not read: {exc}")


def cmd_validate(config: QualitativeAnalystConfig, chunk: int, split: str) -> None:
    config.resolve_paths()

    event_path = Path(config.results_dir) / f"qualitative_events_chunk{chunk}_{split}.csv"
    daily_path = Path(config.results_dir) / f"qualitative_daily_chunk{chunk}_{split}.csv"
    xai_path = Path(config.results_dir) / "xai" / f"qualitative_chunk{chunk}_{split}_xai_summary.json"

    print("=" * 100)
    print(f"QUALITATIVE ANALYST VALIDATION — chunk{chunk}_{split}")
    print("=" * 100)

    if not event_path.exists():
        raise FileNotFoundError(f"Missing event output: {event_path}")
    if not daily_path.exists():
        raise FileNotFoundError(f"Missing daily output: {daily_path}")
    if not xai_path.exists():
        raise FileNotFoundError(f"Missing XAI output: {xai_path}")

    event_df = pd.read_csv(event_path)
    daily_df = pd.read_csv(daily_path)

    required_event = [
        "ticker", "date", "qualitative_score", "qualitative_risk_score",
        "qualitative_confidence", "qualitative_recommendation", "xai_summary",
    ]

    required_daily = [
        "ticker", "date", "event_count", "qualitative_score",
        "qualitative_risk_score", "qualitative_confidence",
        "qualitative_recommendation", "xai_summary",
    ]

    missing_event = [c for c in required_event if c not in event_df.columns]
    missing_daily = [c for c in required_daily if c not in daily_df.columns]

    if missing_event:
        raise RuntimeError(f"Event output missing columns: {missing_event}")
    if missing_daily:
        raise RuntimeError(f"Daily output missing columns: {missing_daily}")

    for name, df in [("event", event_df), ("daily", daily_df)]:
        numeric = df.select_dtypes(include="number")
        finite_ratio = float(np.isfinite(numeric.values).mean()) if len(numeric.columns) else 1.0

        bad_score = int((df["qualitative_score"].abs() > 1.00001).sum())
        bad_risk = int(((df["qualitative_risk_score"] < -1e-9) | (df["qualitative_risk_score"] > 1.00001)).sum())
        bad_conf = int(((df["qualitative_confidence"] < -1e-9) | (df["qualitative_confidence"] > 1.00001)).sum())

        print(f"\n{name.upper()} rows={len(df):,}")
        print(f"finite_ratio={finite_ratio:.6f}")
        print(f"bad_score={bad_score}, bad_risk={bad_risk}, bad_conf={bad_conf}")
        print("recommendation counts:")
        print(df["qualitative_recommendation"].value_counts().to_string())

        if bad_score or bad_risk or bad_conf:
            raise RuntimeError(f"{name} output failed bounds validation.")

    print("\nVALIDATION PASSED")


def cmd_smoke(config: QualitativeAnalystConfig) -> None:
    print("=" * 100)
    print("QUALITATIVE ANALYST SMOKE TEST")
    print("=" * 100)

    set_seed(config.seed)

    n = 512
    rng = np.random.default_rng(config.seed)

    df = pd.DataFrame({
        "ticker": [f"T{i % 16:03d}" for i in range(n)],
        "date": pd.to_datetime("2024-01-01") + pd.to_timedelta(np.arange(n) % 20, unit="D"),
        "sentiment_score": rng.uniform(-1, 1, n),
        "sentiment_confidence": rng.uniform(0.2, 1.0, n),
        "sentiment_uncertainty": rng.uniform(0.0, 0.8, n),
        "sentiment_magnitude": rng.uniform(0.0, 1.0, n),
        "sentiment_event_present": np.ones(n),
        "news_event_impact": rng.uniform(-1, 1, n),
        "news_importance": rng.uniform(0.0, 1.0, n),
        "risk_relevance": rng.uniform(0.0, 1.0, n),
        "volatility_spike": rng.uniform(0.0, 1.0, n),
        "drawdown_risk": rng.uniform(0.0, 1.0, n),
        "news_uncertainty": rng.uniform(0.0, 1.0, n),
        "news_event_present": np.ones(n),
    })
    df["has_both_sources"] = 1.0

    x, y, df = prepare_features_and_targets(df)

    scaler = FeatureScaler()
    scaler.fit(x)
    x = scaler.transform(x)

    model = QualitativeMLP(len(FEATURE_NAMES), hidden_dim=32, n_layers=2, dropout=0.1).to(config.device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    loader = make_loader(x, y, batch_size=128, shuffle=True, device=config.device)

    for _ in range(3):
        loss = train_epoch(model, loader, opt, config.device)

    val_loss = validate_epoch(model, loader, config.device)

    assert np.isfinite(val_loss)

    with torch.no_grad():
        pred = model(torch.from_numpy(x[:32]).to(config.device))["outputs"].cpu().numpy()

    assert pred.shape == (32, 3)
    assert np.all(np.isfinite(pred))

    print(f"SMOKE TEST PASSED | loss={val_loss:.6f} | pred_shape={pred.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_hpo(config: QualitativeAnalystConfig, chunk: int, trials: int, fresh: bool) -> None:
    print("=" * 90)
    print(f"QUALITATIVE ANALYST HPO — chunk{chunk} ({trials} trials)")
    print("=" * 90)
    run_hpo(config, chunk, trials, fresh=fresh)


def cmd_train_best(config: QualitativeAnalystConfig, chunk: int, fresh: bool) -> None:
    config.resolve_paths()

    best = load_best_params(config, chunk)
    if best is not None:
        print(f"Loaded best params for chunk{chunk}: {best}")
        config = apply_best_params(config, best)
    else:
        print(f"No HPO params found for chunk{chunk}; using default config.")

    print("=" * 90)
    print(f"QUALITATIVE ANALYST TRAINING — chunk{chunk}")
    print("=" * 90)

    _, best_val, summary = train_qualitative_model(config, chunk, fresh=fresh, hpo_mode=False)
    print(f"  Complete. Best val loss: {best_val:.6f}")


def cmd_train_best_all(config: QualitativeAnalystConfig, chunks: List[int], fresh: bool) -> None:
    for c in chunks:
        cmd_train_best(config, c, fresh=fresh)


def cmd_predict_all(config: QualitativeAnalystConfig, chunks: List[int], splits: List[str]) -> None:
    for c in chunks:
        for s in splits:
            predict_qualitative(config, c, s)


# ═══════════════════════════════════════════════════════════════════════════════
# ARGPARSE
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qualitative Analyst")
    sub = parser.add_subparsers(dest="command")

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--repo-root", type=str, default="")
        p.add_argument("--device", type=str, default="cuda")
        p.add_argument("--batch-size", type=int, default=None)
        p.add_argument("--epochs", type=int, default=None)
        p.add_argument("--lr", type=float, default=None)
        p.add_argument("--num-workers", type=int, default=None)
        p.add_argument("--max-train-rows", type=int, default=None)

    p = sub.add_parser("inspect")
    add_common(p)

    p = sub.add_parser("smoke")
    add_common(p)

    p = sub.add_parser("hpo")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--fresh", action="store_true")
    add_common(p)

    p = sub.add_parser("train-best")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--fresh", action="store_true")
    add_common(p)

    p = sub.add_parser("train-best-all")
    p.add_argument("--chunks", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    p.add_argument("--fresh", action="store_true")
    add_common(p)

    p = sub.add_parser("predict")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    add_common(p)

    p = sub.add_parser("predict-all")
    p.add_argument("--chunks", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    p.add_argument("--splits", type=str, nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    add_common(p)

    p = sub.add_parser("validate")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    add_common(p)

    return parser


def config_from_args(args: argparse.Namespace) -> QualitativeAnalystConfig:
    config = QualitativeAnalystConfig()

    if getattr(args, "repo_root", ""):
        config.repo_root = args.repo_root

    config.device = getattr(args, "device", "cuda")
    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        config.device = "cpu"

    if getattr(args, "batch_size", None) is not None:
        config.batch_size = int(args.batch_size)

    if getattr(args, "epochs", None) is not None:
        config.epochs = int(args.epochs)

    if getattr(args, "lr", None) is not None:
        config.lr = float(args.lr)

    if getattr(args, "num_workers", None) is not None:
        config.num_workers = int(args.num_workers)

    if getattr(args, "max_train_rows", None) is not None:
        config.max_train_rows = int(args.max_train_rows)

    return config.resolve_paths()


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

    elif args.command == "hpo":
        cmd_hpo(config, args.chunk, args.trials, args.fresh)

    elif args.command == "train-best":
        cmd_train_best(config, args.chunk, args.fresh)

    elif args.command == "train-best-all":
        cmd_train_best_all(config, args.chunks, args.fresh)

    elif args.command == "predict":
        predict_qualitative(config, args.chunk, args.split)

    elif args.command == "predict-all":
        cmd_predict_all(config, args.chunks, args.splits)

    elif args.command == "validate":
        cmd_validate(config, args.chunk, args.split)


if __name__ == "__main__":
    main()


# ── Run instructions ─────────────────────────────────────────────────────────
# Compile:
# python -m py_compile code/analysts/qualitative_analyst.py
#
# Inspect:
# python code/analysts/qualitative_analyst.py inspect --repo-root .
#
# Smoke:
# python code/analysts/qualitative_analyst.py smoke --repo-root . --device cuda
#
# HPO one chunk:
# python code/analysts/qualitative_analyst.py hpo --repo-root . --chunk 1 --trials 30 --device cuda --fresh
#
# Train one chunk:
# python code/analysts/qualitative_analyst.py train-best --repo-root . --chunk 1 --device cuda --fresh
#
# Predict one chunk/split:
# python code/analysts/qualitative_analyst.py predict --repo-root . --chunk 1 --split test --device cuda
#
# Full qualitative run after upstream sentiment/news models are completely rerun:
# cd ~/fin-glassbox && python code/analysts/qualitative_analyst.py hpo --repo-root . --chunk 1 --trials 30 --device cuda --fresh && python code/analysts/qualitative_analyst.py train-best --repo-root . --chunk 1 --device cuda --fresh && python code/analysts/qualitative_analyst.py predict-all --repo-root . --chunks 1 --splits val test --device cuda && python code/analysts/qualitative_analyst.py hpo --repo-root . --chunk 2 --trials 30 --device cuda --fresh && python code/analysts/qualitative_analyst.py train-best --repo-root . --chunk 2 --device cuda --fresh && python code/analysts/qualitative_analyst.py predict-all --repo-root . --chunks 2 --splits val test --device cuda && python code/analysts/qualitative_analyst.py hpo --repo-root . --chunk 3 --trials 30 --device cuda --fresh && python code/analysts/qualitative_analyst.py train-best --repo-root . --chunk 3 --device cuda --fresh && python code/analysts/qualitative_analyst.py predict-all --repo-root . --chunks 3 --splits val test --device cuda
