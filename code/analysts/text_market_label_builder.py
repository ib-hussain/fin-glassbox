#!/usr/bin/env python3
"""
Build real supervised market labels for FinBERT text analyst training.

This script joins row-aligned FinBERT metadata with SEC CIK-to-ticker mapping and
market return data. It creates real forward-looking supervised targets for the
Sentiment Analyst and News Analyst. It does not generate dummy data.

Outputs are CSV label files plus JSON manifests/configuration files.

Design rules:
- Inputs are SEC text metadata available at filing time.
- Targets are future market outcomes after filing_date.
- Event start is the first trading day strictly after filing_date.
- Train-only thresholds are fitted per chronological chunk.
- Validation/test thresholds are inherited from that chunk's training split.
- No Parquet is used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


APPROVED_SPLITS: Dict[int, Dict[str, List[int]]] = {
    1: {"train": [2000, 2001, 2002, 2003, 2004], "val": [2005], "test": [2006]},
    2: {"train": list(range(2007, 2015)), "val": [2015], "test": [2016]},
    3: {"train": list(range(2017, 2023)), "val": [2023], "test": [2024]},
}

REQUIRED_METADATA_COLUMNS = [
    "chunk_id",
    "doc_id",
    "year",
    "form_type",
    "cik",
    "filing_date",
    "accession",
    "source_name",
    "chunk_index",
    "word_count",
]

DEFAULT_HORIZONS = [1, 5, 10, 20, 30]


def now_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_repo_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def normalise_cik(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        if "." in text:
            text = str(int(float(text)))
        else:
            text = str(int(text))
    except Exception:
        text = text.lstrip("0") or text
    return text


def parse_horizons(text: str) -> List[int]:
    out: List[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"Horizons must be positive integers, got {value}")
        out.append(value)
    if not out:
        raise ValueError("At least one horizon is required")
    return sorted(set(out))


@dataclass
class LabelBuilderConfig:
    repo_root: Path = Path(".")
    env_file: Path = Path(".env")

    embeddings_dir: Path = Path("outputs/embeddings/FinBERT")
    market_dir: Path = Path("data/yFinance/processed")
    returns_wide_csv: Path = Path("data/yFinance/processed/returns_panel_wide.csv")
    cik_ticker_map_csv: Path = Path("data/sec_edgar/processed/cleaned/cik_ticker_map_cleaned.csv")

    output_dir: Path = Path("outputs/results/analysts/labels")
    code_output_dir: Path = Path("outputs/codeResults/analysts/labels")

    benchmark_ticker: str = "SPY"
    horizons: List[int] = field(default_factory=lambda: list(DEFAULT_HORIZONS))
    primary_sentiment_horizon: int = 10
    primary_news_horizon: int = 10
    risk_horizon: int = 30
    trailing_vol_window: int = 60

    positive_quantile: float = 0.67
    negative_quantile: float = 0.33
    high_quantile: float = 0.80
    scale_quantile: float = 0.95
    min_scale: float = 1e-6

    chunks: List[int] = field(default_factory=lambda: [1, 2, 3])
    splits: List[str] = field(default_factory=lambda: ["train", "val", "test"])

    overwrite: bool = False
    write_combined: bool = False
    drop_missing_targets: bool = False
    strict_metadata_years: bool = True

    def resolve(self) -> "LabelBuilderConfig":
        self.repo_root = Path(self.repo_root).resolve()
        env = read_env_file(self.repo_root / self.env_file)

        if env.get("FinBERTembeddingsPath"):
            self.embeddings_dir = Path(env["FinBERTembeddingsPath"])
        elif env.get("embeddingsPathGlobal"):
            self.embeddings_dir = Path(env["embeddingsPathGlobal"]) / "FinBERT"

        if env.get("yFinDataPath"):
            self.market_dir = Path(env["yFinDataPath"]) / "processed"
            self.returns_wide_csv = self.market_dir / "returns_panel_wide.csv"

        if env.get("resultsPathGlobal"):
            self.output_dir = Path(env["resultsPathGlobal"]) / "analysts" / "labels"

        if env.get("codeOutputsPathGlobal"):
            self.code_output_dir = Path(env["codeOutputsPathGlobal"]) / "analysts" / "labels"

        self.embeddings_dir = resolve_repo_path(self.repo_root, self.embeddings_dir)
        self.market_dir = resolve_repo_path(self.repo_root, self.market_dir)
        self.returns_wide_csv = resolve_repo_path(self.repo_root, self.returns_wide_csv)
        self.cik_ticker_map_csv = resolve_repo_path(self.repo_root, self.cik_ticker_map_csv)
        self.output_dir = resolve_repo_path(self.repo_root, self.output_dir)
        self.code_output_dir = resolve_repo_path(self.repo_root, self.code_output_dir)

        self.benchmark_ticker = self.benchmark_ticker.upper().strip()
        self.horizons = sorted(set(int(h) for h in self.horizons))

        required = {self.primary_sentiment_horizon, self.primary_news_horizon, self.risk_horizon}
        self.horizons = sorted(set(self.horizons).union(required))

        return self

    def serialisable(self) -> Dict[str, object]:
        out = asdict(self)
        for key, value in list(out.items()):
            if isinstance(value, Path):
                out[key] = str(value)
        return out


class MarketPanel:
    def __init__(self, returns_wide_csv: Path, benchmark_ticker: str = "SPY"):
        self.returns_wide_csv = Path(returns_wide_csv)
        self.benchmark_ticker = benchmark_ticker.upper().strip()

        if not self.returns_wide_csv.exists():
            raise FileNotFoundError(f"Missing market returns file: {self.returns_wide_csv}")

        print(f"[market] loading {self.returns_wide_csv}")
        df = pd.read_csv(self.returns_wide_csv)

        if "date" not in df.columns:
            raise ValueError(f"returns_panel_wide.csv must contain a 'date' column. Columns found: {list(df.columns)[:20]}")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        tickers = [c for c in df.columns if c != "date"]
        if not tickers:
            raise ValueError("returns_panel_wide.csv has no ticker columns")

        rename = {c: str(c).upper().strip() for c in tickers}
        df = df.rename(columns=rename)
        tickers = [rename[c] for c in tickers]

        self.dates = pd.to_datetime(df["date"]).to_numpy(dtype="datetime64[ns]")
        self.tickers = tickers
        self.ticker_set = set(tickers)
        self.returns_df = df

        if self.benchmark_ticker in self.ticker_set:
            self.benchmark_returns = df[self.benchmark_ticker].to_numpy(dtype=np.float64)
            self.benchmark_source = self.benchmark_ticker
        else:
            values = df[tickers].to_numpy(dtype=np.float64)
            self.benchmark_returns = np.nanmean(values, axis=1)
            self.benchmark_source = "cross_sectional_mean"
            print(f"[market-warning] benchmark ticker {self.benchmark_ticker} not found. Using cross-sectional mean benchmark.")

        self.benchmark_returns = np.nan_to_num(self.benchmark_returns, nan=0.0, posinf=0.0, neginf=0.0)
        self.benchmark_cumsum = self._prep_cumsum(self.benchmark_returns)

        print(f"[market] rows={len(self.dates):,} tickers={len(self.tickers):,} date_min={str(self.dates[0])[:10]} date_max={str(self.dates[-1])[:10]} benchmark={self.benchmark_source}")

    @staticmethod
    def _prep_cumsum(values: np.ndarray) -> np.ndarray:
        clean = np.nan_to_num(values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return np.concatenate([[0.0], np.cumsum(clean)])

    @staticmethod
    def _forward_sum_from_cumsum(cumsum: np.ndarray, start_idx: np.ndarray, horizon: int) -> np.ndarray:
        out = np.full(start_idx.shape[0], np.nan, dtype=np.float64)
        valid = (start_idx >= 0) & ((start_idx + horizon) < len(cumsum))
        if np.any(valid):
            i = start_idx[valid]
            out[valid] = cumsum[i + horizon] - cumsum[i]
        return out

    @staticmethod
    def _forward_vol(values: np.ndarray, start_idx: np.ndarray, horizon: int) -> np.ndarray:
        clean = np.nan_to_num(values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        c1 = np.concatenate([[0.0], np.cumsum(clean)])
        c2 = np.concatenate([[0.0], np.cumsum(clean * clean)])
        out = np.full(start_idx.shape[0], np.nan, dtype=np.float64)
        valid = (start_idx >= 0) & ((start_idx + horizon) < len(c1))
        if np.any(valid):
            i = start_idx[valid]
            sums = c1[i + horizon] - c1[i]
            sums2 = c2[i + horizon] - c2[i]
            mean = sums / float(horizon)
            var = np.maximum((sums2 / float(horizon)) - (mean * mean), 0.0)
            out[valid] = np.sqrt(var) * math.sqrt(252.0)
        return out

    @staticmethod
    def _trailing_vol(values: np.ndarray, start_idx: np.ndarray, window: int) -> np.ndarray:
        clean = np.nan_to_num(values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        c1 = np.concatenate([[0.0], np.cumsum(clean)])
        c2 = np.concatenate([[0.0], np.cumsum(clean * clean)])
        out = np.full(start_idx.shape[0], np.nan, dtype=np.float64)
        valid = start_idx >= window
        if np.any(valid):
            i = start_idx[valid]
            sums = c1[i] - c1[i - window]
            sums2 = c2[i] - c2[i - window]
            mean = sums / float(window)
            var = np.maximum((sums2 / float(window)) - (mean * mean), 0.0)
            out[valid] = np.sqrt(var) * math.sqrt(252.0)
        return out

    @staticmethod
    def _forward_max_drawdown(values: np.ndarray, start_idx: np.ndarray, horizon: int) -> np.ndarray:
        clean = np.nan_to_num(values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        out = np.full(start_idx.shape[0], np.nan, dtype=np.float64)
        n = clean.shape[0]
        for row_i, idx in enumerate(start_idx):
            if idx < 0 or idx + horizon > n:
                continue
            path = np.exp(np.cumsum(clean[idx : idx + horizon]))
            path = np.concatenate([[1.0], path])
            peak = np.maximum.accumulate(path)
            dd = path / peak - 1.0
            out[row_i] = abs(float(np.min(dd)))
        return out

    def event_start_indices(self, filing_dates: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        fdates = pd.to_datetime(filing_dates, errors="coerce").to_numpy(dtype="datetime64[ns]")
        idx = np.searchsorted(self.dates, fdates, side="right")
        valid = (~pd.isna(fdates)) & (idx >= 0) & (idx < len(self.dates))
        out_dates = np.full(idx.shape[0], np.datetime64("NaT"), dtype="datetime64[ns]")
        out_dates[valid] = self.dates[idx[valid]]
        return idx.astype(np.int64), out_dates

    def label_group(self, ticker: str, start_idx: np.ndarray, horizons: Sequence[int], trailing_vol_window: int, risk_horizon: int) -> Dict[str, np.ndarray]:
        ticker = str(ticker).upper().strip()
        if ticker not in self.ticker_set:
            raise KeyError(ticker)

        r = self.returns_df[ticker].to_numpy(dtype=np.float64)
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        cumsum = self._prep_cumsum(r)

        out: Dict[str, np.ndarray] = {}
        for h in horizons:
            stock_log = self._forward_sum_from_cumsum(cumsum, start_idx, h)
            bench_log = self._forward_sum_from_cumsum(self.benchmark_cumsum, start_idx, h)
            out[f"future_log_return_{h}d"] = stock_log
            out[f"benchmark_log_return_{h}d"] = bench_log
            out[f"future_simple_return_{h}d"] = np.expm1(stock_log)
            out[f"benchmark_simple_return_{h}d"] = np.expm1(bench_log)
            out[f"future_excess_log_return_{h}d"] = stock_log - bench_log
            out[f"future_excess_simple_return_{h}d"] = np.expm1(stock_log) - np.expm1(bench_log)

        out[f"future_realised_vol_{risk_horizon}d"] = self._forward_vol(r, start_idx, risk_horizon)
        out[f"trailing_realised_vol_{trailing_vol_window}d"] = self._trailing_vol(r, start_idx, trailing_vol_window)
        denom = np.maximum(out[f"trailing_realised_vol_{trailing_vol_window}d"], 1e-9)
        out[f"future_vol_ratio_{risk_horizon}d_vs_trailing_{trailing_vol_window}d"] = out[f"future_realised_vol_{risk_horizon}d"] / denom
        out[f"future_max_drawdown_{risk_horizon}d"] = self._forward_max_drawdown(r, start_idx, risk_horizon)

        return out


def load_cik_ticker_map(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CIK→ticker mapping file: {path}")

    df = pd.read_csv(path)
    candidates = ["primary_ticker", "ticker", "symbol"]
    ticker_col = next((c for c in candidates if c in df.columns), None)
    if ticker_col is None:
        raise ValueError(f"Mapping file must contain one of {candidates}. Columns: {list(df.columns)}")
    if "cik" not in df.columns:
        raise ValueError(f"Mapping file must contain 'cik'. Columns: {list(df.columns)}")

    out = df[["cik", ticker_col]].copy()
    out["cik_norm"] = out["cik"].map(normalise_cik)
    out["ticker"] = out[ticker_col].astype(str).str.upper().str.strip()
    out = out[(out["cik_norm"] != "") & (out["ticker"] != "")]
    out = out.drop_duplicates(subset=["cik_norm"], keep="first")
    return out[["cik_norm", "ticker"]]


def metadata_path(cfg: LabelBuilderConfig, chunk_id: int, split: str) -> Path:
    return cfg.embeddings_dir / f"chunk{chunk_id}_{split}_metadata.csv"


def embedding_path(cfg: LabelBuilderConfig, chunk_id: int, split: str) -> Path:
    return cfg.embeddings_dir / f"chunk{chunk_id}_{split}_embeddings.npy"


def manifest_path(cfg: LabelBuilderConfig, chunk_id: int, split: str) -> Path:
    return cfg.embeddings_dir / f"chunk{chunk_id}_{split}_manifest.json"


def label_output_path(cfg: LabelBuilderConfig, chunk_id: int, split: str) -> Path:
    return cfg.output_dir / f"text_market_labels_chunk{chunk_id}_{split}.csv"


def require_metadata_file(cfg: LabelBuilderConfig, chunk_id: int, split: str) -> Path:
    path = metadata_path(cfg, chunk_id, split)
    if not path.exists():
        raise FileNotFoundError(f"Missing FinBERT metadata file: {path}. Run the FinBERT embedding pipeline first.")
    return path


def validate_embedding_contract_if_present(cfg: LabelBuilderConfig, chunk_id: int, split: str, metadata_rows: int) -> Dict[str, object]:
    emb = embedding_path(cfg, chunk_id, split)
    man = manifest_path(cfg, chunk_id, split)
    result: Dict[str, object] = {
        "embedding_file": str(emb),
        "embedding_exists": emb.exists(),
        "manifest_file": str(man),
        "manifest_exists": man.exists(),
        "embedding_shape": None,
        "embedding_dtype": None,
        "embedding_row_match": None,
    }
    if emb.exists():
        arr = np.load(emb, mmap_mode="r")
        result["embedding_shape"] = tuple(int(x) for x in arr.shape)
        result["embedding_dtype"] = str(arr.dtype)
        result["embedding_row_match"] = bool(arr.shape[0] == metadata_rows)
        if len(arr.shape) != 2 or arr.shape[1] != 256:
            raise ValueError(f"Expected embedding shape (N, 256), got {arr.shape} for {emb}")
        if arr.shape[0] != metadata_rows:
            raise ValueError(f"Embedding rows ({arr.shape[0]}) do not match metadata rows ({metadata_rows}) for chunk{chunk_id}_{split}")
    return result


def read_metadata(cfg: LabelBuilderConfig, chunk_id: int, split: str) -> pd.DataFrame:
    path = require_metadata_file(cfg, chunk_id, split)
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_METADATA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Metadata file {path} is missing required columns: {missing}")

    df = df.copy()
    df["metadata_row_index"] = np.arange(len(df), dtype=np.int64)
    df["chunk_file_id"] = int(chunk_id)
    df["split"] = split
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df["year"] = df["year"].astype(int)
    df["cik_norm"] = df["cik"].map(normalise_cik)

    expected_years = set(APPROVED_SPLITS[chunk_id][split])
    observed_years = set(df["year"].dropna().astype(int).unique().tolist())
    if cfg.strict_metadata_years and observed_years and not observed_years.issubset(expected_years):
        raise ValueError(f"Metadata years for chunk{chunk_id}_{split} are {sorted(observed_years)}, expected subset of {sorted(expected_years)}")

    return df


def add_market_labels_to_metadata(df: pd.DataFrame, market: MarketPanel, horizons: Sequence[int], trailing_vol_window: int, risk_horizon: int) -> pd.DataFrame:
    out = df.copy()
    out["event_start_idx"], event_dates = market.event_start_indices(out["filing_date"])
    out["event_start_date"] = pd.to_datetime(event_dates)
    out["market_benchmark"] = market.benchmark_source

    for h in horizons:
        for col in [
            f"future_log_return_{h}d",
            f"benchmark_log_return_{h}d",
            f"future_simple_return_{h}d",
            f"benchmark_simple_return_{h}d",
            f"future_excess_log_return_{h}d",
            f"future_excess_simple_return_{h}d",
        ]:
            out[col] = np.nan

    for col in [
        f"future_realised_vol_{risk_horizon}d",
        f"trailing_realised_vol_{trailing_vol_window}d",
        f"future_vol_ratio_{risk_horizon}d_vs_trailing_{trailing_vol_window}d",
        f"future_max_drawdown_{risk_horizon}d",
    ]:
        out[col] = np.nan

    out["ticker_in_market_panel"] = False

    grouped = out.groupby("ticker", sort=False).indices
    for ticker, idx_obj in grouped.items():
        ticker = str(ticker).upper().strip()
        idx = np.asarray(idx_obj, dtype=np.int64)
        if ticker not in market.ticker_set:
            continue

        start_idx = out.loc[idx, "event_start_idx"].to_numpy(dtype=np.int64)
        label_block = market.label_group(
            ticker=ticker,
            start_idx=start_idx,
            horizons=horizons,
            trailing_vol_window=trailing_vol_window,
            risk_horizon=risk_horizon,
        )
        for col, values in label_block.items():
            out.loc[idx, col] = values
        out.loc[idx, "ticker_in_market_panel"] = True

    out["label_available"] = out[f"future_excess_log_return_{risk_horizon}d"].notna() & out["ticker_in_market_panel"]
    return out


def safe_quantile(values: pd.Series, q: float, default: float = np.nan) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) == 0:
        return float(default)
    return float(clean.quantile(q))


def fit_chunk_thresholds(train_df: pd.DataFrame, cfg: LabelBuilderConfig) -> Dict[str, float]:
    h_sent = cfg.primary_sentiment_horizon
    h_news = cfg.primary_news_horizon
    h_risk = cfg.risk_horizon

    excess_col = f"future_excess_log_return_{h_sent}d"
    news_col = f"future_excess_log_return_{h_news}d"
    vol_ratio_col = f"future_vol_ratio_{h_risk}d_vs_trailing_{cfg.trailing_vol_window}d"
    dd_col = f"future_max_drawdown_{h_risk}d"

    train = train_df[train_df["label_available"]].copy()
    abs_excess = pd.to_numeric(train[excess_col], errors="coerce").abs()
    abs_news = pd.to_numeric(train[news_col], errors="coerce").abs()

    thresholds = {
        "negative_excess_return_quantile": safe_quantile(train[excess_col], cfg.negative_quantile, default=-0.01),
        "positive_excess_return_quantile": safe_quantile(train[excess_col], cfg.positive_quantile, default=0.01),
        "sentiment_scale_abs_excess": max(safe_quantile(abs_excess, cfg.scale_quantile, default=0.05), cfg.min_scale),
        "news_importance_scale_abs_excess": max(safe_quantile(abs_news, cfg.scale_quantile, default=0.05), cfg.min_scale),
        "volatility_spike_ratio_threshold": safe_quantile(train[vol_ratio_col], cfg.high_quantile, default=1.5),
        "drawdown_risk_threshold": safe_quantile(train[dd_col], cfg.high_quantile, default=0.05),
        "risk_scale_drawdown": max(safe_quantile(train[dd_col], cfg.scale_quantile, default=0.10), cfg.min_scale),
        "fitted_rows": int(len(train)),
        "negative_quantile": float(cfg.negative_quantile),
        "positive_quantile": float(cfg.positive_quantile),
        "high_quantile": float(cfg.high_quantile),
        "scale_quantile": float(cfg.scale_quantile),
    }
    return thresholds


def apply_chunk_targets(df: pd.DataFrame, thresholds: Dict[str, float], cfg: LabelBuilderConfig) -> pd.DataFrame:
    out = df.copy()
    h_sent = cfg.primary_sentiment_horizon
    h_news = cfg.primary_news_horizon
    h_risk = cfg.risk_horizon

    sent_col = f"future_excess_log_return_{h_sent}d"
    news_col = f"future_excess_log_return_{h_news}d"
    vol_ratio_col = f"future_vol_ratio_{h_risk}d_vs_trailing_{cfg.trailing_vol_window}d"
    dd_col = f"future_max_drawdown_{h_risk}d"

    neg_thr = float(thresholds["negative_excess_return_quantile"])
    pos_thr = float(thresholds["positive_excess_return_quantile"])
    sent_scale = float(thresholds["sentiment_scale_abs_excess"])
    news_scale = float(thresholds["news_importance_scale_abs_excess"])
    vol_thr = float(thresholds["volatility_spike_ratio_threshold"])
    dd_thr = float(thresholds["drawdown_risk_threshold"])
    dd_scale = float(thresholds["risk_scale_drawdown"])

    sent_values = pd.to_numeric(out[sent_col], errors="coerce").to_numpy(dtype=np.float64)
    news_values = pd.to_numeric(out[news_col], errors="coerce").to_numpy(dtype=np.float64)
    vol_ratio = pd.to_numeric(out[vol_ratio_col], errors="coerce").to_numpy(dtype=np.float64)
    drawdown = pd.to_numeric(out[dd_col], errors="coerce").to_numpy(dtype=np.float64)

    out["sentiment_score_target"] = np.tanh(sent_values / sent_scale)
    out["sentiment_class_target"] = np.where(sent_values <= neg_thr, -1, np.where(sent_values >= pos_thr, 1, 0))
    out.loc[pd.isna(out[sent_col]), "sentiment_class_target"] = np.nan

    out["news_event_impact_target"] = np.tanh(news_values / news_scale)
    out["news_importance_target"] = np.clip(np.abs(news_values) / news_scale, 0.0, 1.0)

    out[f"volatility_spike_{h_risk}d_target"] = np.where(vol_ratio >= vol_thr, 1.0, 0.0)
    out.loc[pd.isna(out[vol_ratio_col]), f"volatility_spike_{h_risk}d_target"] = np.nan

    out[f"drawdown_risk_{h_risk}d_target"] = np.where(drawdown >= dd_thr, 1.0, 0.0)
    out.loc[pd.isna(out[dd_col]), f"drawdown_risk_{h_risk}d_target"] = np.nan

    vol_score = np.clip(vol_ratio / max(vol_thr, cfg.min_scale), 0.0, 1.0)
    dd_score = np.clip(drawdown / max(dd_scale, cfg.min_scale), 0.0, 1.0)
    out["risk_relevance_target"] = np.nanmax(np.vstack([vol_score, dd_score]), axis=0)
    out.loc[pd.isna(out[vol_ratio_col]) & pd.isna(out[dd_col]), "risk_relevance_target"] = np.nan

    # For direct analyst training convenience.
    out["primary_excess_return_target"] = out[sent_col]
    out["primary_abs_excess_return_target"] = np.abs(news_values)
    out["target_schema_version"] = "text_market_labels_v1"

    if cfg.drop_missing_targets:
        out = out[out["label_available"]].copy()

    return out


def write_json(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def build_labels(cfg: LabelBuilderConfig) -> Dict[str, object]:
    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.code_output_dir)

    print("========== TEXT MARKET LABEL BUILDER ==========")
    print(json.dumps(cfg.serialisable(), indent=2))

    cik_map = load_cik_ticker_map(cfg.cik_ticker_map_csv)
    print(f"[map] loaded CIK→ticker rows={len(cik_map):,} from {cfg.cik_ticker_map_csv}")

    market = MarketPanel(cfg.returns_wide_csv, benchmark_ticker=cfg.benchmark_ticker)

    all_manifests: List[Dict[str, object]] = []
    combined_parts: List[pd.DataFrame] = []

    for chunk_id in cfg.chunks:
        if chunk_id not in APPROVED_SPLITS:
            raise ValueError(f"Unsupported chunk_id={chunk_id}. Approved chunks: {sorted(APPROVED_SPLITS)}")

        print(f"\n========== CHUNK {chunk_id}: RAW LABEL CONSTRUCTION ==========")
        raw_by_split: Dict[str, pd.DataFrame] = {}

        for split in cfg.splits:
            if split not in APPROVED_SPLITS[chunk_id]:
                raise ValueError(f"Unsupported split={split}. Use train, val, or test.")

            out_csv = label_output_path(cfg, chunk_id, split)
            if out_csv.exists() and not cfg.overwrite:
                print(f"[skip] exists: {out_csv}")
                existing = pd.read_csv(out_csv)
                raw_by_split[split] = existing
                continue

            meta = read_metadata(cfg, chunk_id, split)
            emb_contract = validate_embedding_contract_if_present(cfg, chunk_id, split, metadata_rows=len(meta))

            meta = meta.merge(cik_map, how="left", on="cik_norm")
            meta["ticker"] = meta["ticker"].astype("string")

            labelled = add_market_labels_to_metadata(
                df=meta,
                market=market,
                horizons=cfg.horizons,
                trailing_vol_window=cfg.trailing_vol_window,
                risk_horizon=cfg.risk_horizon,
            )
            raw_by_split[split] = labelled

            print(f"[chunk{chunk_id}_{split}] rows={len(labelled):,} mapped_ticker={labelled['ticker'].notna().sum():,} in_market={int(labelled['ticker_in_market_panel'].sum()):,} label_available={int(labelled['label_available'].sum()):,}")

            contract_path = cfg.code_output_dir / f"chunk{chunk_id}_{split}_embedding_contract.json"
            write_json(contract_path, emb_contract)

        if "train" not in raw_by_split:
            raise RuntimeError(f"Chunk {chunk_id} train split is required to fit thresholds")

        thresholds = fit_chunk_thresholds(raw_by_split["train"], cfg)
        thresholds_path = cfg.output_dir / f"text_market_label_thresholds_chunk{chunk_id}.json"
        write_json(thresholds_path, thresholds)
        print(f"[thresholds] chunk={chunk_id} saved={thresholds_path}")

        for split, raw_df in raw_by_split.items():
            out_csv = label_output_path(cfg, chunk_id, split)
            if out_csv.exists() and not cfg.overwrite and "target_schema_version" in raw_df.columns:
                final_df = raw_df
            else:
                final_df = apply_chunk_targets(raw_df, thresholds, cfg)
                ensure_dir(out_csv.parent)
                final_df.to_csv(out_csv, index=False)

            if cfg.write_combined:
                combined_parts.append(final_df)

            manifest = {
                "created_at": now_stamp(),
                "schema_version": "text_market_labels_v1",
                "chunk_id": int(chunk_id),
                "split": split,
                "approved_years": APPROVED_SPLITS[chunk_id][split],
                "rows": int(len(final_df)),
                "label_available_rows": int(final_df["label_available"].sum()) if "label_available" in final_df.columns else None,
                "mapped_ticker_rows": int(final_df["ticker"].notna().sum()) if "ticker" in final_df.columns else None,
                "ticker_in_market_panel_rows": int(final_df["ticker_in_market_panel"].sum()) if "ticker_in_market_panel" in final_df.columns else None,
                "metadata_file": str(metadata_path(cfg, chunk_id, split)),
                "embedding_file": str(embedding_path(cfg, chunk_id, split)),
                "label_file": str(out_csv),
                "thresholds_file": str(thresholds_path),
                "benchmark_source": market.benchmark_source,
                "target_columns": [
                    "sentiment_score_target",
                    "sentiment_class_target",
                    "news_event_impact_target",
                    "news_importance_target",
                    f"volatility_spike_{cfg.risk_horizon}d_target",
                    f"drawdown_risk_{cfg.risk_horizon}d_target",
                    "risk_relevance_target",
                ],
                "anti_leakage_rules": [
                    "Event start is the first trading day strictly after filing_date.",
                    "Forward returns are used only as supervised targets, not as model inputs.",
                    "Classification and scaling thresholds are fitted on the chunk train split only.",
                    "Validation and test splits reuse train-fitted thresholds.",
                    "No validation/test distribution statistics are used for threshold fitting.",
                ],
            }
            manifest_path_out = cfg.output_dir / f"text_market_labels_chunk{chunk_id}_{split}_manifest.json"
            write_json(manifest_path_out, manifest)
            all_manifests.append(manifest)
            print(f"[write] {out_csv}")

    combined_file = None
    if cfg.write_combined and combined_parts:
        combined = pd.concat(combined_parts, axis=0, ignore_index=True)
        combined_file = cfg.output_dir / "text_market_labels_all.csv"
        combined.to_csv(combined_file, index=False)
        print(f"[write] {combined_file}")

    run_manifest = {
        "created_at": now_stamp(),
        "config": cfg.serialisable(),
        "combined_file": str(combined_file) if combined_file else None,
        "files": all_manifests,
    }
    write_json(cfg.output_dir / "text_market_label_builder_manifest.json", run_manifest)
    write_json(cfg.code_output_dir / "text_market_label_builder_run_config.json", cfg.serialisable())

    print("========== LABEL BUILD COMPLETE ==========")
    print(json.dumps({"output_dir": str(cfg.output_dir), "files_written": len(all_manifests), "combined_file": str(combined_file) if combined_file else None}, indent=2))
    return run_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build real supervised text-market labels from FinBERT metadata and market returns.")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--embeddings-dir", default=None)
    p.add_argument("--returns-wide-csv", default=None)
    p.add_argument("--cik-ticker-map-csv", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--code-output-dir", default=None)
    p.add_argument("--benchmark-ticker", default="SPY")
    p.add_argument("--horizons", default="1,5,10,20,30")
    p.add_argument("--primary-sentiment-horizon", type=int, default=10)
    p.add_argument("--primary-news-horizon", type=int, default=10)
    p.add_argument("--risk-horizon", type=int, default=30)
    p.add_argument("--trailing-vol-window", type=int, default=60)
    p.add_argument("--chunks", default="1,2,3")
    p.add_argument("--splits", default="train,val,test")
    p.add_argument("--negative-quantile", type=float, default=0.33)
    p.add_argument("--positive-quantile", type=float, default=0.67)
    p.add_argument("--high-quantile", type=float, default=0.80)
    p.add_argument("--scale-quantile", type=float, default=0.95)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--write-combined", action="store_true")
    p.add_argument("--drop-missing-targets", action="store_true")
    p.add_argument("--no-strict-metadata-years", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = LabelBuilderConfig()
    cfg.repo_root = Path(args.repo_root)
    cfg.env_file = Path(args.env_file)
    cfg.resolve()

    if args.embeddings_dir is not None:
        cfg.embeddings_dir = resolve_repo_path(cfg.repo_root, Path(args.embeddings_dir))
    if args.returns_wide_csv is not None:
        cfg.returns_wide_csv = resolve_repo_path(cfg.repo_root, Path(args.returns_wide_csv))
    if args.cik_ticker_map_csv is not None:
        cfg.cik_ticker_map_csv = resolve_repo_path(cfg.repo_root, Path(args.cik_ticker_map_csv))
    if args.output_dir is not None:
        cfg.output_dir = resolve_repo_path(cfg.repo_root, Path(args.output_dir))
    if args.code_output_dir is not None:
        cfg.code_output_dir = resolve_repo_path(cfg.repo_root, Path(args.code_output_dir))

    cfg.benchmark_ticker = args.benchmark_ticker
    cfg.horizons = parse_horizons(args.horizons)
    cfg.primary_sentiment_horizon = int(args.primary_sentiment_horizon)
    cfg.primary_news_horizon = int(args.primary_news_horizon)
    cfg.risk_horizon = int(args.risk_horizon)
    cfg.trailing_vol_window = int(args.trailing_vol_window)
    cfg.chunks = [int(x.strip()) for x in args.chunks.split(",") if x.strip()]
    cfg.splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    cfg.negative_quantile = float(args.negative_quantile)
    cfg.positive_quantile = float(args.positive_quantile)
    cfg.high_quantile = float(args.high_quantile)
    cfg.scale_quantile = float(args.scale_quantile)
    cfg.overwrite = bool(args.overwrite)
    cfg.write_combined = bool(args.write_combined)
    cfg.drop_missing_targets = bool(args.drop_missing_targets)
    cfg.strict_metadata_years = not bool(args.no_strict_metadata_years)
    cfg.resolve()

    build_labels(cfg)


if __name__ == "__main__":
    main()


# python code/analysts/text_market_label_builder.py --repo-root ~/fin-glassbox \
#        --chunks 1,2,3 --splits train,val,test --benchmark-ticker SPY --overwrite
# optional combined output:
# python code/analysts/text_market_label_builder.py --repo-root ~/fin-glassbox \
#        --chunks 1,2,3 --splits train,val,test --benchmark-ticker SPY --overwrite --write-combined
