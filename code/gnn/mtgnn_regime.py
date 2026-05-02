#!/usr/bin/env python3
"""
code/gnn/mtgnn_regime.py

MTGNN Regime Detection Module
=============================

Project:
    fin-glassbox — Explainable Multimodal Neural Framework for Financial Risk Management

Purpose:
    Detect market regime using:
        - Temporal Encoder embeddings per ticker/date
        - FinBERT text embeddings aggregated per ticker/date window
        - FRED macro/regime data
        - Existing cross-asset graph snapshots

Outputs:
    - regime_label: calm / volatile / crisis / rotation
    - regime_confidence
    - transition_probability
    - graph properties
    - macro stress score
    - graph stress score
    - XAI explanation dictionary

Design:
    This is an MTGNN-style graph-builder + graph-property classifier.

    Node features:
        [TemporalEncoder 256-dim] + [FinBERT aggregate 256-dim] = 512-dim

    Graph learner:
        Feature-aware Top-K learned adjacency

    Classifier:
        graph_properties + FRED macro features -> regime class

CLI:
    python code/gnn/mtgnn_regime.py inspect --repo-root .
    python code/gnn/mtgnn_regime.py smoke --repo-root . --device cuda
    python code/gnn/mtgnn_regime.py build-graph --repo-root . --chunk 1 --split train --device cuda
    python code/gnn/mtgnn_regime.py hpo --repo-root . --chunk 1 --trials 30 --device cuda --fresh
    python code/gnn/mtgnn_regime.py train-best --repo-root . --chunk 1 --device cuda --fresh
    python code/gnn/mtgnn_regime.py predict --repo-root . --chunk 1 --split test --device cuda
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import pickle
import random
import shutil
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field, fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

CHUNK_CONFIG = {
    1: {"train": (2000, 2004), "val": (2005, 2005), "test": (2006, 2006), "label": "chunk1"},
    2: {"train": (2007, 2014), "val": (2015, 2015), "test": (2016, 2016), "label": "chunk2"},
    3: {"train": (2017, 2022), "val": (2023, 2023), "test": (2024, 2024), "label": "chunk3"},
}

REGIME_LABELS = {
    0: "calm",
    1: "volatile",
    2: "crisis",
    3: "rotation",
}

REGIME_TO_ID = {v: k for k, v in REGIME_LABELS.items()}

HPO_FAILURE_VALUE = 1_000_000_000.0

DEFAULT_MACRO_COLS = [
    "yield_spread_10y2y",
    "yield_spread_10y3m",
    "credit_spread_baa_aaa",
    "regime_yield_inverted",
    "DFF",
    "DGS10",
    "DGS2",
    "DGS3MO",
    "BAA10Y",
    "AAA10Y",
]


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MTGNNRegimeConfig:
    repo_root: str = ""
    output_dir: str = "outputs"

    temporal_dir: str = "outputs/embeddings/TemporalEncoder"
    finbert_dir: str = "outputs/embeddings/FinBERT"

    fred_path: str = "data/FRED_data/outputs/macro_features_trading_days_clean.csv"
    returns_path: str = "data/yFinance/processed/returns_panel_wide.csv"
    graph_snapshot_dir: str = "data/graphs/snapshots"
    sector_map_path: str = "data/graphs/metadata/sector_map.csv"

    cik_ticker_map_path: str = "data/sec_edgar/processed/cleaned/cik_ticker_map_cleaned.csv"

    # Model dimensions
    temporal_dim: int = 256
    text_dim: int = 256
    node_feature_dim: int = 512
    node_hidden_dim: int = 128
    graph_hidden_dim: int = 64
    classifier_hidden_dim: int = 64
    dropout: float = 0.20

    # Graph
    top_k: int = 66
    node_limit: int = 2500
    text_lookback_days: int = 30
    use_existing_graph_features: bool = True

    # Regime labels and features
    macro_cols: List[str] = field(default_factory=lambda: list(DEFAULT_MACRO_COLS))
    market_lookback_days: int = 21
    drawdown_lookback_days: int = 63

    # Training
    batch_size: int = 1
    epochs: int = 80
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stop_patience: int = 15
    gradient_clip: float = 1.0

    # HPO
    hpo_trials: int = 30
    hpo_n_startup: int = 8
    hpo_epochs: int = 20
    hpo_node_limit: int = 768

    # XAI
    xai_top_edges: int = 50
    xai_counterfactuals: int = 8

    # System
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 0
    cpu_threads: int = 6
    deterministic: bool = False
    persistent_workers: bool = False

    # Runtime
    max_snapshots: int = 0
    run_tag: str = ""
    save_checkpoints: bool = True

    def resolve_paths(self) -> "MTGNNRegimeConfig":
        if self.repo_root:
            root = Path(self.repo_root)
            for attr in [
                "output_dir",
                "temporal_dir",
                "finbert_dir",
                "fred_path",
                "returns_path",
                "graph_snapshot_dir",
                "sector_map_path",
                "cik_ticker_map_path",
            ]:
                value = getattr(self, attr)
                if value and not Path(value).is_absolute():
                    setattr(self, attr, str(root / value))

        self.node_feature_dim = int(self.temporal_dim) + int(self.text_dim)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
# RUNTIME UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def configure_torch_runtime(config: MTGNNRegimeConfig) -> None:
    torch.set_num_threads(max(1, int(config.cpu_threads)))
    torch.set_num_interop_threads(max(1, min(2, int(config.cpu_threads))))

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = not bool(config.deterministic)
    torch.backends.cudnn.deterministic = bool(config.deterministic)

    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(device: str) -> torch.device:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("  CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def shutdown_dataloader(loader: Optional[DataLoader]) -> None:
    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None and hasattr(iterator, "_shutdown_workers"):
        try:
            iterator._shutdown_workers()
        except Exception:
            pass
    loader._iterator = None


def finite_ratio_np(array: np.ndarray) -> float:
    arr = np.asarray(array)
    if arr.size == 0:
        return 1.0
    return float(np.isfinite(arr).mean())


def sanitize_np_array(
    array: np.ndarray,
    *,
    nan: float = 0.0,
    posinf: float = 0.0,
    neginf: float = 0.0,
    dtype=np.float32,
) -> np.ndarray:
    arr = np.asarray(array)
    arr = np.nan_to_num(arr, nan=nan, posinf=posinf, neginf=neginf)
    return arr.astype(dtype, copy=False)
def json_safe(obj: Any) -> Any:
    """Convert NumPy/Pandas/PyTorch values into JSON-serialisable Python types."""
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
        if not np.isfinite(value):
            return None
        return value

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return json_safe(obj.detach().cpu().item())
        return json_safe(obj.detach().cpu().tolist())

    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, float):
        if not np.isfinite(obj):
            return None
        return obj

    return obj

def load_config_from_checkpoint_dict(raw: Dict[str, Any], fallback: Optional[MTGNNRegimeConfig] = None) -> MTGNNRegimeConfig:
    base = fallback.to_dict() if fallback is not None else MTGNNRegimeConfig().to_dict()
    ckpt_cfg = raw.get("config", {})
    if isinstance(ckpt_cfg, dict):
        base.update(ckpt_cfg)

    valid = {f.name for f in dataclass_fields(MTGNNRegimeConfig)}
    filtered = {k: v for k, v in base.items() if k in valid}
    return MTGNNRegimeConfig(**filtered).resolve_paths()


# ═══════════════════════════════════════════════════════════════════════════════
# PATH VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def temporal_paths(config: MTGNNRegimeConfig, chunk_id: int, split: str) -> Tuple[Path, Path]:
    label = CHUNK_CONFIG[chunk_id]["label"]
    base = Path(config.temporal_dir)
    return base / f"{label}_{split}_embeddings.npy", base / f"{label}_{split}_manifest.csv"


def finbert_paths(config: MTGNNRegimeConfig, chunk_id: int, split: str) -> Tuple[Path, Path]:
    label = CHUNK_CONFIG[chunk_id]["label"]
    base = Path(config.finbert_dir)
    return base / f"{label}_{split}_embeddings.npy", base / f"{label}_{split}_metadata.csv"


def validate_required_inputs(config: MTGNNRegimeConfig, chunk_id: int, splits: Tuple[str, ...]) -> None:
    missing: List[Path] = []

    for split in splits:
        t_emb, t_man = temporal_paths(config, chunk_id, split)
        f_emb, f_meta = finbert_paths(config, chunk_id, split)
        for path in [t_emb, t_man, f_emb, f_meta]:
            if not path.exists():
                missing.append(path)

    for path in [Path(config.fred_path), Path(config.returns_path), Path(config.graph_snapshot_dir)]:
        if not path.exists():
            missing.append(path)

    if missing:
        label = CHUNK_CONFIG[chunk_id]["label"]
        missing_text = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            f"Missing required inputs for MTGNN Regime {label}.\n"
            f"This module needs TemporalEncoder embeddings, FinBERT embeddings, FRED macro data, returns, and graph snapshots.\n"
            f"Missing:\n{missing_text}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# BASIC LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_returns_frame(config: MTGNNRegimeConfig) -> pd.DataFrame:
    df = pd.read_csv(config.returns_path, index_col=0, parse_dates=True)
    df = df.sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df.astype(np.float32)


def load_fred_frame(config: MTGNNRegimeConfig) -> pd.DataFrame:
    df = pd.read_csv(config.fred_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    return df


def select_macro_cols(config: MTGNNRegimeConfig, fred_df: pd.DataFrame) -> List[str]:
    cols = [c for c in config.macro_cols if c in fred_df.columns]
    if not cols:
        cols = [c for c in DEFAULT_MACRO_COLS if c in fred_df.columns]
    if not cols:
        cols = [c for c in fred_df.columns if c != "date"][:12]
    return cols


def load_sector_map(config: MTGNNRegimeConfig) -> Dict[str, str]:
    path = Path(config.sector_map_path)
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    ticker_col = cols.get("ticker") or cols.get("symbol") or cols.get("node")
    sector_col = cols.get("sector") or cols.get("sic_sector") or cols.get("industry")

    if ticker_col is None or sector_col is None:
        return {}

    df[ticker_col] = df[ticker_col].astype(str)
    df[sector_col] = df[sector_col].astype(str)
    return dict(zip(df[ticker_col], df[sector_col]))


def load_cik_ticker_map(config: MTGNNRegimeConfig) -> Dict[str, str]:
    candidates = [
        Path(config.cik_ticker_map_path),
        Path(config.repo_root) / "data/sec_edgar/processed/cleaned/cik_ticker_map_cleaned.csv" if config.repo_root else Path("data/sec_edgar/processed/cleaned/cik_ticker_map_cleaned.csv"),
        Path(config.repo_root) / "data/sec_edgar/processed/cik_ticker_map_cleaned.csv" if config.repo_root else Path("data/sec_edgar/processed/cik_ticker_map_cleaned.csv"),
    ]

    for path in candidates:
        if not path.exists():
            continue

        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}

        cik_col = cols.get("cik") or cols.get("cik_str") or cols.get("central_index_key")
        ticker_col = cols.get("ticker") or cols.get("symbol")

        if cik_col is None or ticker_col is None:
            continue

        df[cik_col] = df[cik_col].astype(str).str.replace(".0", "", regex=False).str.zfill(10)
        df[ticker_col] = df[ticker_col].astype(str)
        return dict(zip(df[cik_col], df[ticker_col]))

    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# SNAPSHOT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def parse_snapshot_date(path: Path) -> str:
    # edges_2000-01-04.csv
    stem = path.stem
    return stem.replace("edges_", "")


def snapshot_paths_for_split(config: MTGNNRegimeConfig, chunk_id: int, split: str) -> List[Path]:
    years = CHUNK_CONFIG[chunk_id][split]
    paths = sorted(Path(config.graph_snapshot_dir).glob("edges_*.csv"))

    out: List[Path] = []
    for path in paths:
        date_str = parse_snapshot_date(path)
        year = int(date_str[:4])
        if years[0] <= year <= years[1]:
            out.append(path)

    if config.max_snapshots and config.max_snapshots > 0:
        out = out[: int(config.max_snapshots)]

    return out


def build_temporal_index(manifest: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    manifest = manifest.copy()
    manifest["date"] = pd.to_datetime(manifest["date"])
    manifest["date_ord"] = manifest["date"].map(pd.Timestamp.toordinal)
    manifest["row_id"] = np.arange(len(manifest), dtype=np.int64)

    index: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for ticker, group in tqdm(manifest.groupby("ticker", sort=True), desc="  Temporal index", leave=False):
        group = group.sort_values("date_ord")
        index[str(ticker)] = (
            group["date_ord"].values.astype(np.int64),
            group["row_id"].values.astype(np.int64),
        )
    return index


def find_latest_row(index: Dict[str, Tuple[np.ndarray, np.ndarray]], ticker: str, date_ord: int) -> Optional[int]:
    item = index.get(ticker)
    if item is None:
        return None
    dates, rows = item
    pos = np.searchsorted(dates, date_ord, side="right") - 1
    if pos < 0:
        return None
    return int(rows[pos])


def select_node_universe(
    config: MTGNNRegimeConfig,
    manifest: pd.DataFrame,
    snapshot_paths: List[Path],
) -> List[str]:
    manifest_tickers = set(manifest["ticker"].astype(str).unique().tolist())

    graph_counts: Dict[str, int] = {}
    probe_paths = snapshot_paths[: min(12, len(snapshot_paths))]

    for path in probe_paths:
        try:
            edge_df = pd.read_csv(path, usecols=["source", "target"])
        except Exception:
            continue
        for col in ["source", "target"]:
            vals = edge_df[col].astype(str).values
            for ticker in vals:
                if ticker in manifest_tickers:
                    graph_counts[ticker] = graph_counts.get(ticker, 0) + 1

    if graph_counts:
        ordered = sorted(graph_counts.keys(), key=lambda t: graph_counts[t], reverse=True)
    else:
        ordered = sorted(manifest_tickers)

    if config.node_limit and config.node_limit > 0:
        ordered = ordered[: int(config.node_limit)]

    return ordered


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

class FinBERTTextContext:
    """Ticker/date-window FinBERT aggregation for regime node features.

    Metadata has CIK but not ticker, so this class uses CIK->ticker mapping if available.
    If mapping is unavailable, it falls back to global market-level text context.
    """

    def __init__(self, config: MTGNNRegimeConfig, chunk_id: int, split: str) -> None:
        self.config = config
        self.chunk_id = chunk_id
        self.split = split

        emb_path, meta_path = finbert_paths(config, chunk_id, split)
        self.embeddings = np.load(emb_path, mmap_mode="r")
        self.metadata = pd.read_csv(meta_path)

        n = min(len(self.metadata), len(self.embeddings))
        self.metadata = self.metadata.iloc[:n].reset_index(drop=True)
        self.embeddings = self.embeddings[:n]

        self.cik_ticker = load_cik_ticker_map(config)

        self.has_ticker_mapping = False
        self.metadata["date"] = pd.to_datetime(self.metadata.get("filing_date", self.metadata.get("date")), errors="coerce")
        self.metadata["date_ord"] = self.metadata["date"].map(lambda x: x.toordinal() if pd.notna(x) else -1)

        if "ticker" in self.metadata.columns:
            self.metadata["ticker_resolved"] = self.metadata["ticker"].astype(str)
            self.has_ticker_mapping = True
        elif "cik" in self.metadata.columns and self.cik_ticker:
            ciks = self.metadata["cik"].astype(str).str.replace(".0", "", regex=False).str.zfill(10)
            self.metadata["ticker_resolved"] = ciks.map(self.cik_ticker)
            self.has_ticker_mapping = self.metadata["ticker_resolved"].notna().any()
        else:
            self.metadata["ticker_resolved"] = None

        self.metadata = self.metadata[self.metadata["date_ord"] > 0].reset_index(drop=True)

    def aggregate_for_date(self, node_tickers: List[str], date_str: str, lookback_days: int) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        n_nodes = len(node_tickers)
        out = np.zeros((n_nodes, int(self.config.text_dim)), dtype=np.float32)

        if len(self.metadata) == 0:
            return out, 0.0, {"mode": "none", "rows": 0, "coverage": 0.0}

        date_ord = pd.Timestamp(date_str).toordinal()
        start_ord = date_ord - int(lookback_days)

        mask = (self.metadata["date_ord"].values >= start_ord) & (self.metadata["date_ord"].values <= date_ord)
        idx = np.where(mask)[0]

        if len(idx) == 0:
            return out, 0.0, {"mode": "empty_window", "rows": 0, "coverage": 0.0}

        if not self.has_ticker_mapping:
            global_vec = sanitize_np_array(np.asarray(self.embeddings[idx]).mean(axis=0))
            out[:] = global_vec[None, :]
            return out, 1.0, {"mode": "global_text_broadcast", "rows": int(len(idx)), "coverage": 1.0}

        sub = self.metadata.iloc[idx].copy()
        sub = sub.dropna(subset=["ticker_resolved"])

        if len(sub) == 0:
            global_vec = sanitize_np_array(np.asarray(self.embeddings[idx]).mean(axis=0))
            out[:] = global_vec[None, :]
            return out, 1.0, {"mode": "global_text_broadcast_no_ticker_rows", "rows": int(len(idx)), "coverage": 1.0}

        ticker_to_node = {ticker: i for i, ticker in enumerate(node_tickers)}
        covered = 0

        for ticker, group in sub.groupby("ticker_resolved", sort=False):
            ticker = str(ticker)
            node_idx = ticker_to_node.get(ticker)
            if node_idx is None:
                continue
            emb_rows = group.index.values.astype(np.int64)
            if len(emb_rows) == 0:
                continue
            out[node_idx] = sanitize_np_array(np.asarray(self.embeddings[emb_rows]).mean(axis=0))
            covered += 1

        coverage = covered / max(n_nodes, 1)
        return out, float(coverage), {"mode": "ticker_text", "rows": int(len(idx)), "coverage": float(coverage), "covered_nodes": int(covered)}


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH / MARKET / MACRO FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_existing_graph_features(
    edge_path: Path,
    node_tickers: List[str],
    sector_map: Dict[str, str],
) -> Dict[str, float]:
    n = len(node_tickers)
    node_set = set(node_tickers)

    try:
        edges = pd.read_csv(edge_path)
    except Exception:
        return {
            "existing_edges": 0.0,
            "existing_density": 0.0,
            "existing_avg_degree_norm": 0.0,
            "existing_mean_abs_corr": 0.0,
            "existing_max_abs_corr": 0.0,
            "sector_concentration": 0.0,
        }

    if not {"source", "target", "correlation"}.issubset(edges.columns):
        return {
            "existing_edges": 0.0,
            "existing_density": 0.0,
            "existing_avg_degree_norm": 0.0,
            "existing_mean_abs_corr": 0.0,
            "existing_max_abs_corr": 0.0,
            "sector_concentration": 0.0,
        }

    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)
    edges = edges[edges["source"].isin(node_set) & edges["target"].isin(node_set)].copy()

    m = len(edges)
    denom = max(n * (n - 1), 1)
    density = m / denom
    avg_degree_norm = (m / max(n, 1)) / max(n - 1, 1)

    corr = pd.to_numeric(edges["correlation"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().values
    if len(corr):
        mean_abs_corr = float(np.mean(np.abs(corr)))
        max_abs_corr = float(np.max(np.abs(corr)))
    else:
        mean_abs_corr = 0.0
        max_abs_corr = 0.0

    sector_concentration = 0.0
    if sector_map and len(edges):
        edge_sectors = []
        for s, t in zip(edges["source"].values, edges["target"].values):
            ss = sector_map.get(str(s), "UNKNOWN")
            ts = sector_map.get(str(t), "UNKNOWN")
            edge_sectors.append(f"{ss}->{ts}")
        if edge_sectors:
            vc = pd.Series(edge_sectors).value_counts(normalize=True)
            sector_concentration = float(vc.iloc[0])

    return {
        "existing_edges": float(m),
        "existing_density": float(density),
        "existing_avg_degree_norm": float(avg_degree_norm),
        "existing_mean_abs_corr": float(mean_abs_corr),
        "existing_max_abs_corr": float(max_abs_corr),
        "sector_concentration": float(sector_concentration),
    }


def compute_market_features(
    returns_df: pd.DataFrame,
    date_str: str,
    lookback: int,
    drawdown_lookback: int,
) -> Dict[str, float]:
    date = pd.Timestamp(date_str)
    if date not in returns_df.index:
        pos = returns_df.index.searchsorted(date, side="right") - 1
        if pos < 0:
            return {
                "market_vol_21d": 0.0,
                "market_ret_21d": 0.0,
                "market_drawdown_63d": 0.0,
                "xsec_dispersion": 0.0,
            }
        date = returns_df.index[pos]

    end_pos = returns_df.index.get_loc(date)
    if isinstance(end_pos, slice):
        end_pos = end_pos.stop - 1
    end_pos = int(end_pos)

    start = max(0, end_pos - int(lookback) + 1)
    window = returns_df.iloc[start:end_pos + 1].values.astype(np.float64)
    window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)

    ew_ret = window.mean(axis=1) if window.size else np.array([0.0])
    market_vol = float(np.std(ew_ret) * math.sqrt(252)) if len(ew_ret) > 1 else 0.0
    market_ret = float(np.sum(ew_ret)) if len(ew_ret) else 0.0
    xsec_disp = float(np.mean(np.std(window, axis=1))) if window.size else 0.0

    dd_start = max(0, end_pos - int(drawdown_lookback) + 1)
    dd_window = returns_df.iloc[dd_start:end_pos + 1].values.astype(np.float64)
    dd_window = np.nan_to_num(dd_window, nan=0.0, posinf=0.0, neginf=0.0)
    dd_ew = dd_window.mean(axis=1) if dd_window.size else np.array([0.0])
    equity = np.exp(np.cumsum(dd_ew))
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.maximum(peak, 1e-12)
    max_dd = float(np.min(dd)) if len(dd) else 0.0

    return {
        "market_vol_21d": float(market_vol),
        "market_ret_21d": float(market_ret),
        "market_drawdown_63d": float(max_dd),
        "xsec_dispersion": float(xsec_disp),
    }


def get_fred_row(fred_df: pd.DataFrame, date_str: str) -> pd.Series:
    date_ord = pd.Timestamp(date_str).toordinal()
    ords = pd.to_datetime(fred_df["date"]).map(pd.Timestamp.toordinal).values
    pos = np.searchsorted(ords, date_ord, side="right") - 1
    if pos < 0:
        pos = 0
    return fred_df.iloc[int(pos)]


def fit_macro_stats(records: List[Dict[str, Any]], macro_cols: List[str]) -> Dict[str, Dict[str, float]]:
    values = np.array([[r["macro_raw"].get(c, 0.0) for c in macro_cols] for r in records], dtype=np.float64)
    stats: Dict[str, Dict[str, float]] = {}

    for j, col in enumerate(macro_cols):
        x = values[:, j]
        mean = float(np.nanmean(x))
        std = float(np.nanstd(x))
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0
        stats[col] = {"mean": mean, "std": std}

    return stats


def normalise_macro(macro_raw: Dict[str, float], macro_cols: List[str], macro_stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    out = []
    for col in macro_cols:
        v = float(macro_raw.get(col, 0.0))
        st = macro_stats.get(col, {"mean": 0.0, "std": 1.0})
        z = (v - float(st["mean"])) / max(float(st["std"]), 1e-8)
        z = float(np.clip(z, -5.0, 5.0))
        out.append(z)
    return np.array(out, dtype=np.float32)


def fit_label_stats(records: List[Dict[str, Any]]) -> Dict[str, float]:
    def arr(key: str) -> np.ndarray:
        return np.array([float(r["raw_metrics"].get(key, 0.0)) for r in records], dtype=np.float64)

    vol = arr("market_vol_21d")
    credit = arr("credit_spread_baa_aaa")
    density = arr("existing_density")
    abs_corr = arr("existing_mean_abs_corr")
    sector = arr("sector_concentration")

    return {
        "vol_p50": float(np.nanpercentile(vol, 50)),
        "vol_p75": float(np.nanpercentile(vol, 75)),
        "vol_p90": float(np.nanpercentile(vol, 90)),
        "credit_p75": float(np.nanpercentile(credit, 75)),
        "credit_p90": float(np.nanpercentile(credit, 90)),
        "density_p50": float(np.nanpercentile(density, 50)),
        "density_p75": float(np.nanpercentile(density, 75)),
        "density_p90": float(np.nanpercentile(density, 90)),
        "abs_corr_p50": float(np.nanpercentile(abs_corr, 50)),
        "abs_corr_p75": float(np.nanpercentile(abs_corr, 75)),
        "sector_p75": float(np.nanpercentile(sector, 75)),
    }


def assign_regime_label(raw: Dict[str, float], label_stats: Dict[str, float]) -> int:
    vol = float(raw.get("market_vol_21d", 0.0))
    ret = float(raw.get("market_ret_21d", 0.0))
    drawdown = float(raw.get("market_drawdown_63d", 0.0))
    credit = float(raw.get("credit_spread_baa_aaa", 0.0))
    inverted = float(raw.get("regime_yield_inverted", 0.0)) >= 0.5
    density = float(raw.get("existing_density", 0.0))
    abs_corr = float(raw.get("existing_mean_abs_corr", 0.0))
    sector = float(raw.get("sector_concentration", 0.0))

    crisis = (
        vol >= label_stats["vol_p90"]
        or credit >= label_stats["credit_p90"]
        or (inverted and ret < -0.03)
        or drawdown < -0.08
        or (density >= label_stats["density_p90"] and abs_corr >= label_stats["abs_corr_p75"])
    )
    if crisis:
        return 2

    volatile = (
        vol >= label_stats["vol_p75"]
        or credit >= label_stats["credit_p75"]
        or inverted
        or density >= label_stats["density_p75"]
        or drawdown < -0.04
    )
    if volatile:
        return 1

    rotation = (
        sector >= label_stats["sector_p75"]
        and density >= label_stats["density_p50"]
        and abs_corr <= label_stats["abs_corr_p75"]
    )
    if rotation:
        return 3

    return 0


def macro_stress_score(raw: Dict[str, float], label_stats: Dict[str, float]) -> float:
    credit = float(raw.get("credit_spread_baa_aaa", 0.0))
    vol = float(raw.get("market_vol_21d", 0.0))
    inverted = float(raw.get("regime_yield_inverted", 0.0))

    credit_component = credit / max(label_stats.get("credit_p90", credit + 1e-6), 1e-6)
    vol_component = vol / max(label_stats.get("vol_p90", vol + 1e-6), 1e-6)

    score = 0.45 * credit_component + 0.45 * vol_component + 0.10 * inverted
    return float(np.clip(score, 0.0, 1.0))


def graph_stress_score(raw: Dict[str, float], label_stats: Dict[str, float]) -> float:
    density = float(raw.get("existing_density", 0.0))
    abs_corr = float(raw.get("existing_mean_abs_corr", 0.0))

    density_component = density / max(label_stats.get("density_p90", density + 1e-6), 1e-6)
    corr_component = abs_corr / max(label_stats.get("abs_corr_p75", abs_corr + 1e-6), 1e-6)

    score = 0.5 * density_component + 0.5 * corr_component
    return float(np.clip(score, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class RegimeSnapshotDataset(Dataset):
    def __init__(
        self,
        config: MTGNNRegimeConfig,
        chunk_id: int,
        split: str,
        *,
        fit_stats: bool = False,
        macro_stats: Optional[Dict[str, Dict[str, float]]] = None,
        label_stats: Optional[Dict[str, float]] = None,
        node_tickers: Optional[List[str]] = None,
    ) -> None:
        self.config = config
        self.chunk_id = int(chunk_id)
        self.split = split
        self.chunk_label = CHUNK_CONFIG[chunk_id]["label"]

        emb_path, manifest_path = temporal_paths(config, chunk_id, split)
        self.temporal_embeddings = np.load(emb_path, mmap_mode="r")
        self.temporal_manifest = pd.read_csv(manifest_path, dtype={"ticker": str})
        self.temporal_manifest["date"] = pd.to_datetime(self.temporal_manifest["date"])

        n = min(len(self.temporal_manifest), len(self.temporal_embeddings))
        self.temporal_manifest = self.temporal_manifest.iloc[:n].reset_index(drop=True)
        self.temporal_embeddings = self.temporal_embeddings[:n]

        self.snapshot_paths = snapshot_paths_for_split(config, chunk_id, split)
        if not self.snapshot_paths:
            raise ValueError(f"No graph snapshots found for {self.chunk_label}_{split}")

        if node_tickers is None:
            self.node_tickers = select_node_universe(config, self.temporal_manifest, self.snapshot_paths)
        else:
            self.node_tickers = list(node_tickers)

        if not self.node_tickers:
            raise ValueError("No node tickers selected for regime dataset.")

        self.node_to_idx = {t: i for i, t in enumerate(self.node_tickers)}

        self.temporal_index = build_temporal_index(self.temporal_manifest)
        self.text_context = FinBERTTextContext(config, chunk_id, split)

        self.returns_df = load_returns_frame(config)
        self.fred_df = load_fred_frame(config)
        self.macro_cols = select_macro_cols(config, self.fred_df)
        self.sector_map = load_sector_map(config)

        print(
            f"  Building {self.chunk_label}_{split}: "
            f"snapshots={len(self.snapshot_paths)}, nodes={len(self.node_tickers)}, macro_cols={len(self.macro_cols)}"
        )

        self.raw_records = self._build_raw_records()

        if fit_stats:
            self.macro_stats = fit_macro_stats(self.raw_records, self.macro_cols)
            self.label_stats = fit_label_stats(self.raw_records)
        else:
            if macro_stats is None or label_stats is None:
                raise ValueError("macro_stats and label_stats are required when fit_stats=False")
            self.macro_stats = macro_stats
            self.label_stats = label_stats

        self.records = self._finalise_records()
        self.audit = self._audit()

        print(
            f"  {self.chunk_label}_{split}: {len(self.records):,} samples | "
            f"label_counts={self.audit['label_counts']} | "
            f"node_feature_finite={self.audit['node_feature_finite_ratio']:.6f}"
        )

    def _build_raw_records(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        date_min = self.temporal_manifest["date"].min()

        for edge_path in tqdm(self.snapshot_paths, desc=f"  Snapshots {self.chunk_label}_{self.split}", leave=False):
            date_str = parse_snapshot_date(edge_path)
            date_ts = pd.Timestamp(date_str)

            if date_ts < date_min:
                continue

            date_ord = date_ts.toordinal()

            temporal_features = np.zeros((len(self.node_tickers), int(self.config.temporal_dim)), dtype=np.float32)

            for i, ticker in enumerate(self.node_tickers):
                row = find_latest_row(self.temporal_index, ticker, date_ord)
                if row is None:
                    continue
                temporal_features[i] = sanitize_np_array(np.asarray(self.temporal_embeddings[row]), dtype=np.float32)

            text_features, text_coverage, text_meta = self.text_context.aggregate_for_date(
                self.node_tickers,
                date_str,
                int(self.config.text_lookback_days),
            )

            node_features = np.concatenate([temporal_features, text_features], axis=1).astype(np.float32)

            fred_row = get_fred_row(self.fred_df, date_str)
            macro_raw = {
                col: float(fred_row[col]) if col in fred_row.index and pd.notna(fred_row[col]) else 0.0
                for col in self.macro_cols
            }

            graph_raw = compute_existing_graph_features(edge_path, self.node_tickers, self.sector_map)
            market_raw = compute_market_features(
                self.returns_df,
                date_str,
                int(self.config.market_lookback_days),
                int(self.config.drawdown_lookback_days),
            )

            raw_metrics: Dict[str, float] = {}
            raw_metrics.update(graph_raw)
            raw_metrics.update(market_raw)
            raw_metrics.update({
                "credit_spread_baa_aaa": macro_raw.get("credit_spread_baa_aaa", 0.0),
                "yield_spread_10y2y": macro_raw.get("yield_spread_10y2y", 0.0),
                "yield_spread_10y3m": macro_raw.get("yield_spread_10y3m", 0.0),
                "regime_yield_inverted": macro_raw.get("regime_yield_inverted", 0.0),
                "text_coverage": text_coverage,
            })

            records.append({
                "date": date_str,
                "edge_path": str(edge_path),
                "node_features": node_features,
                "macro_raw": macro_raw,
                "raw_metrics": raw_metrics,
                "text_meta": text_meta,
            })

        if not records:
            raise ValueError(f"No usable regime snapshots built for {self.chunk_label}_{self.split}")

        return records

    def _finalise_records(self) -> List[Dict[str, Any]]:
        out = []

        for record in self.raw_records:
            macro_vec = normalise_macro(record["macro_raw"], self.macro_cols, self.macro_stats)
            label = assign_regime_label(record["raw_metrics"], self.label_stats)
            macro_stress = macro_stress_score(record["raw_metrics"], self.label_stats)
            graph_stress = graph_stress_score(record["raw_metrics"], self.label_stats)

            out.append({
                "date": record["date"],
                "node_features": sanitize_np_array(record["node_features"]),
                "macro_features": macro_vec,
                "label": int(label),
                "label_name": REGIME_LABELS[int(label)],
                "macro_raw": record["macro_raw"],
                "raw_metrics": record["raw_metrics"],
                "text_meta": record["text_meta"],
                "macro_stress_score": float(macro_stress),
                "graph_stress_score": float(graph_stress),
            })

        return out

    def _audit(self) -> Dict[str, Any]:
        if not self.records:
            return {"samples": 0, "label_counts": {}, "node_feature_finite_ratio": 1.0}

        labels = [r["label_name"] for r in self.records]
        # label_counts = dict(pd.Series(labels).value_counts().sort_index())
        label_counts = {
            str(k): int(v)
            for k, v in pd.Series(labels).value_counts().sort_index().items()
        }

        sample_n = min(5, len(self.records))
        feats = np.concatenate([self.records[i]["node_features"].reshape(-1) for i in range(sample_n)])

        return {
            "samples": int(len(self.records)),
            "nodes": int(len(self.node_tickers)),
            "label_counts": label_counts,
            "node_feature_finite_ratio": finite_ratio_np(feats),
            "macro_cols": list(self.macro_cols),
        }

    def __len__(self) -> int:
        return int(len(self.records))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]

        return {
            "node_features": torch.from_numpy(r["node_features"].astype(np.float32)),
            "macro_features": torch.from_numpy(r["macro_features"].astype(np.float32)),
            "label": torch.tensor(int(r["label"]), dtype=torch.long),
            "date": r["date"],
            "macro_stress_score": torch.tensor(float(r["macro_stress_score"]), dtype=torch.float32),
            "graph_stress_score": torch.tensor(float(r["graph_stress_score"]), dtype=torch.float32),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class MTGNNGraphLearner(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, graph_dim: int, top_k: int, dropout: float) -> None:
        super().__init__()
        self.top_k = int(top_k)

        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, graph_dim),
            nn.LayerNorm(graph_dim),
            nn.GELU(),
        )

        self.query = nn.Linear(graph_dim, graph_dim, bias=False)
        self.key = nn.Linear(graph_dim, graph_dim, bias=False)

    def forward(self, node_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # node_features: [B, N, F]
        h = self.node_encoder(node_features)
        q = self.query(h)
        k = self.key(h)

        score = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(max(q.size(-1), 1))
        n = score.size(1)

        eye = torch.eye(n, device=score.device, dtype=torch.bool).unsqueeze(0)
        score = score.masked_fill(eye, -1e9)

        weights = torch.sigmoid(score)

        k_top = min(max(1, self.top_k), max(1, n - 1))
        top_values, top_indices = torch.topk(weights, k=k_top, dim=-1)

        adj = torch.zeros_like(weights)
        adj.scatter_(dim=-1, index=top_indices, src=top_values)

        return adj, h


def graph_properties_from_adjacency(adj: torch.Tensor) -> torch.Tensor:
    # adj: [B, N, N]
    b, n, _ = adj.shape
    eps = 1e-8

    mask = adj > 0
    edge_count = mask.float().sum(dim=(1, 2))
    possible = max(n * (n - 1), 1)

    density = edge_count / possible

    degree = mask.float().sum(dim=-1)
    mean_degree_norm = degree.mean(dim=1) / max(n - 1, 1)
    std_degree_norm = degree.std(dim=1) / max(n - 1, 1)

    weight_sum = adj.sum(dim=(1, 2))
    mean_weight = weight_sum / (edge_count + eps)
    max_weight = adj.reshape(b, -1).max(dim=1).values

    row_sum = adj.sum(dim=-1, keepdim=True)
    p = adj / (row_sum + eps)
    entropy = -(p * torch.log(p + eps)).sum(dim=-1).mean(dim=1) / math.log(max(n, 2))

    graph_stress = 0.5 * density + 0.5 * mean_weight

    props = torch.stack(
        [
            density,
            mean_degree_norm,
            std_degree_norm,
            mean_weight,
            max_weight,
            entropy,
            graph_stress,
        ],
        dim=1,
    )
    return props


class MTGNNRegimeModel(nn.Module):
    def __init__(self, config: MTGNNRegimeConfig, macro_dim: int) -> None:
        super().__init__()
        self.config = config
        self.macro_dim = int(macro_dim)

        self.graph_learner = MTGNNGraphLearner(
            input_dim=int(config.node_feature_dim),
            hidden_dim=int(config.node_hidden_dim),
            graph_dim=int(config.graph_hidden_dim),
            top_k=int(config.top_k),
            dropout=float(config.dropout),
        )

        classifier_in = 7 + self.macro_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, int(config.classifier_hidden_dim)),
            nn.LayerNorm(int(config.classifier_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(int(config.classifier_hidden_dim), 4),
        )

        self.transition_head = nn.Sequential(
            nn.Linear(classifier_in, int(config.classifier_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(int(config.classifier_hidden_dim), 1),
            nn.Sigmoid(),
        )

    def forward(self, node_features: torch.Tensor, macro_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
        if macro_features.dim() == 1:
            macro_features = macro_features.unsqueeze(0)

        node_features = torch.nan_to_num(node_features.float(), nan=0.0, posinf=0.0, neginf=0.0)
        macro_features = torch.nan_to_num(macro_features.float(), nan=0.0, posinf=0.0, neginf=0.0)

        adj, node_hidden = self.graph_learner(node_features)
        graph_props = graph_properties_from_adjacency(adj)

        clf_in = torch.cat([graph_props, macro_features], dim=1)
        logits = self.classifier(clf_in)
        probs = torch.softmax(logits, dim=-1)
        transition_probability = self.transition_head(clf_in).squeeze(-1)

        return {
            "logits": logits,
            "probs": probs,
            "regime_id": probs.argmax(dim=-1),
            "confidence": probs.max(dim=-1).values,
            "transition_probability": transition_probability,
            "adjacency": adj,
            "graph_properties": graph_props,
            "node_hidden": node_hidden,
        }

    def save(
        self,
        path: Union[str, Path],
        macro_stats: Dict[str, Dict[str, float]],
        label_stats: Dict[str, float],
        macro_cols: List[str],
        node_tickers: List[str],
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": self.config.to_dict(),
                "macro_dim": self.macro_dim,
                "macro_stats": macro_stats,
                "label_stats": label_stats,
                "macro_cols": macro_cols,
                "node_tickers": node_tickers,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        config: Optional[MTGNNRegimeConfig] = None,
        device: str = "cpu",
    ) -> Tuple["MTGNNRegimeModel", Dict[str, Any]]:
        path = Path(path)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = load_config_from_checkpoint_dict(ckpt, fallback=config)
        model = cls(cfg, macro_dim=int(ckpt["macro_dim"]))
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        return model, ckpt


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def make_loader(dataset: Dataset, config: MTGNNRegimeConfig, train: bool) -> DataLoader:
    workers = max(0, int(config.num_workers))
    kwargs: Dict[str, Any] = {
        "batch_size": int(config.batch_size),
        "shuffle": bool(train),
        "drop_last": False,
        "num_workers": workers,
        "pin_memory": str(config.device).startswith("cuda"),
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(config.persistent_workers)
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def compute_class_weights(dataset: RegimeSnapshotDataset, device: torch.device) -> torch.Tensor:
    labels = np.array([r["label"] for r in dataset.records], dtype=np.int64)
    counts = np.bincount(labels, minlength=4).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (4.0 * counts)
    weights = np.clip(weights, 0.25, 5.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def model_dir_for(config: MTGNNRegimeConfig, chunk_label: str, run_tag: str = "") -> Path:
    base = Path(config.output_dir) / "models" / "MTGNNRegime"
    if run_tag:
        return base / "_hpo_trials" / run_tag / chunk_label
    return base / chunk_label


def train_epoch(
    model: MTGNNRegimeModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    config: MTGNNRegimeConfig,
) -> float:
    model.train()
    total = 0.0
    batches = 0

    bar = tqdm(loader, desc=f"    train bs={loader.batch_size}", leave=False, unit="batch")

    for batch in bar:
        node_features = batch["node_features"].to(device, non_blocking=True)
        macro_features = batch["macro_features"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        out = model(node_features, macro_features)
        loss = loss_fn(out["logits"], labels)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite MTGNN Regime training loss: {float(loss.detach().cpu())}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.gradient_clip))
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        total += loss_value
        batches += 1
        bar.set_postfix(loss=f"{loss_value:.5f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    return total / max(batches, 1)


@torch.no_grad()
def validate_epoch(
    model: MTGNNRegimeModel,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total = 0.0
    batches = 0
    correct = 0
    n = 0

    all_probs = []
    all_labels = []

    bar = tqdm(loader, desc=f"    val   bs={loader.batch_size}", leave=False, unit="batch")

    for batch in bar:
        node_features = batch["node_features"].to(device, non_blocking=True)
        macro_features = batch["macro_features"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        out = model(node_features, macro_features)
        loss = loss_fn(out["logits"], labels)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite MTGNN Regime validation loss: {float(loss.detach().cpu())}")

        preds = out["probs"].argmax(dim=-1)
        correct += int((preds == labels).sum().item())
        n += int(labels.numel())

        all_probs.append(out["probs"].detach().cpu())
        all_labels.append(labels.detach().cpu())

        loss_value = float(loss.detach().cpu())
        total += loss_value
        batches += 1
        bar.set_postfix(loss=f"{loss_value:.5f}")

    acc = correct / max(n, 1)

    probs = torch.cat(all_probs, dim=0) if all_probs else torch.empty(0, 4)
    labels = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0, dtype=torch.long)

    brier = 0.0
    if len(labels):
        onehot = F.one_hot(labels, num_classes=4).float()
        brier = float(torch.mean(torch.sum((probs - onehot) ** 2, dim=1)).item())

    return {
        "loss": total / max(batches, 1),
        "accuracy": float(acc),
        "brier": float(brier),
    }


def _build_train_val_datasets(
    config: MTGNNRegimeConfig,
    chunk_id: int,
) -> Tuple[RegimeSnapshotDataset, RegimeSnapshotDataset]:
    train_ds = RegimeSnapshotDataset(config, chunk_id, "train", fit_stats=True)

    val_ds = RegimeSnapshotDataset(
        config,
        chunk_id,
        "val",
        fit_stats=False,
        macro_stats=train_ds.macro_stats,
        label_stats=train_ds.label_stats,
        node_tickers=train_ds.node_tickers,
    )

    return train_ds, val_ds


def _train_model(
    config: MTGNNRegimeConfig,
    chunk_id: int,
    device: torch.device,
    *,
    run_tag: str = "",
    save_checkpoints: bool = True,
) -> Tuple[MTGNNRegimeModel, Dict[str, Any]]:
    label = CHUNK_CONFIG[chunk_id]["label"]

    train_ds, val_ds = _build_train_val_datasets(config, chunk_id)

    train_loader = make_loader(train_ds, config, train=True)
    val_loader = make_loader(val_ds, config, train=False)

    model = MTGNNRegimeModel(config, macro_dim=len(train_ds.macro_cols)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )

    class_weights = compute_class_weights(train_ds, device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    out_dir = model_dir_for(config, label, "" if save_checkpoints else run_tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_path = out_dir / "best_model.pt"
    latest_path = out_dir / "latest_model.pt"
    history_path = out_dir / "training_history.csv"
    summary_path = out_dir / "training_summary.json"

    best_val = float("inf")
    no_improve = 0
    history = []
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(
        f"  Train snapshots: {len(train_ds):,} | Val snapshots: {len(val_ds):,} | "
        f"nodes={len(train_ds.node_tickers):,} | params={total_params:,}"
    )
    print(f"  Class weights: {class_weights.detach().cpu().numpy().round(3).tolist()}")
    print(f"  Train labels: {train_ds.audit['label_counts']}")
    print(f"  Val labels: {val_ds.audit['label_counts']}")

    epoch = 0

    try:
        for epoch in range(1, int(config.epochs) + 1):
            start = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, config)
            val_metrics = validate_epoch(model, val_loader, loss_fn, device)
            val_loss = float(val_metrics["loss"])

            row = {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": val_loss,
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_brier": float(val_metrics["brier"]),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "seconds": round(time.time() - start, 2),
            }
            history.append(row)

            print(
                f"  [{label}] E{epoch:03d}/{config.epochs} | "
                f"train={train_loss:.6f} | val={val_loss:.6f} | "
                f"acc={val_metrics['accuracy']:.3f} | brier={val_metrics['brier']:.4f} | "
                f"{row['seconds']:.1f}s"
            )

            if val_loss < best_val:
                best_val = val_loss
                no_improve = 0
                if save_checkpoints:
                    model.save(
                        best_path,
                        train_ds.macro_stats,
                        train_ds.label_stats,
                        train_ds.macro_cols,
                        train_ds.node_tickers,
                    )
            else:
                no_improve += 1

            if save_checkpoints:
                model.save(
                    latest_path,
                    train_ds.macro_stats,
                    train_ds.label_stats,
                    train_ds.macro_cols,
                    train_ds.node_tickers,
                )

            if no_improve >= int(config.early_stop_patience):
                print(f"  Early stopping at epoch {epoch}")
                break

    finally:
        shutdown_dataloader(train_loader)
        shutdown_dataloader(val_loader)
        cleanup_memory()

    if not np.isfinite(best_val):
        raise RuntimeError("MTGNN Regime training failed: non-finite best validation loss.")

    if save_checkpoints:
        pd.DataFrame(history).to_csv(history_path, index=False)
        if best_path.exists():
            model, _ = MTGNNRegimeModel.load(best_path, config=config, device=str(device))

    summary = {
        "chunk": label,
        "run_tag": run_tag,
        "best_val_loss": float(best_val),
        "epochs_trained": int(epoch),
        "total_params": int(total_params),
        "train_snapshots": int(len(train_ds)),
        "val_snapshots": int(len(val_ds)),
        "nodes": int(len(train_ds.node_tickers)),
        "macro_cols": train_ds.macro_cols,
        "train_audit": train_ds.audit,
        "val_audit": val_ds.audit,
        "label_stats": train_ds.label_stats,
        "saved_checkpoints": bool(save_checkpoints),
    }

    if save_checkpoints:
        with open(summary_path, "w") as f:
            json.dump(json_safe(summary), f, indent=2)

    print(f"  Best val loss: {best_val:.6f}")
    return model, summary


def train_regime_model(config: MTGNNRegimeConfig, chunk_id: int, fresh: bool = False) -> Dict[str, Any]:
    config.resolve_paths()
    configure_torch_runtime(config)
    seed_everything(config.seed)

    validate_required_inputs(config, chunk_id, ("train", "val"))

    device = resolve_device(config.device)
    label = CHUNK_CONFIG[chunk_id]["label"]

    print(f"\n{'=' * 72}")
    print(f"  MTGNN REGIME TRAINING — {label}")
    print(f"{'=' * 72}")
    print(f"  Device: {device}")

    out_dir = model_dir_for(config, label, "")
    if fresh and out_dir.exists():
        print(f"  Fresh run requested. Removing: {out_dir}")
        shutil.rmtree(out_dir)

    model, summary = _train_model(config, chunk_id, device, run_tag="", save_checkpoints=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    # model.save(
    #     out_dir / "final_model.pt",
    #     macro_stats=summary.get("macro_stats", {}),
    #     label_stats=summary.get("label_stats", {}),
    #     macro_cols=summary.get("macro_cols", []),
    #     node_tickers=[],
    # )
    best_path = out_dir / "best_model.pt"
    if best_path.exists(): shutil.copy2(best_path, out_dir / "final_model.pt")
    else:  raise FileNotFoundError(f"Expected best checkpoint not found: {best_path}")

    frozen = out_dir / "model_freezed"
    frozen.mkdir(parents=True, exist_ok=True)

    best_path = out_dir / "best_model.pt"
    # if best_path.exists():
    #     shutil.copy2(best_path, frozen / "model.pt")
    shutil.copy2(best_path, frozen / "model.pt")

    print(f"\n  Complete. Best val loss: {summary['best_val_loss']:.6f}")
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# HPO
# ═══════════════════════════════════════════════════════════════════════════════

def _hpo_objective(
    trial: "optuna.trial.Trial",
    base_config: MTGNNRegimeConfig,
    chunk_id: int,
) -> float:
    try:
        cfg = MTGNNRegimeConfig(**base_config.to_dict()).resolve_paths()

        cfg.node_hidden_dim = trial.suggest_categorical("node_hidden_dim", [64, 128, 192])
        cfg.graph_hidden_dim = trial.suggest_categorical("graph_hidden_dim", [32, 64, 96])
        cfg.classifier_hidden_dim = trial.suggest_categorical("classifier_hidden_dim", [32, 64, 128])
        cfg.dropout = trial.suggest_float("dropout", 0.05, 0.35)
        cfg.top_k = trial.suggest_categorical("top_k", [32, 48, 66, 96])
        cfg.learning_rate = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        cfg.weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)

        cfg.epochs = int(base_config.hpo_epochs)
        cfg.early_stop_patience = min(8, int(base_config.early_stop_patience))
        cfg.node_limit = int(base_config.hpo_node_limit)
        cfg.num_workers = 0
        cfg.persistent_workers = False
        cfg.batch_size = 1
        cfg.run_tag = f"trial_{trial.number:04d}"
        cfg.save_checkpoints = False

        device = resolve_device(cfg.device)
        _, summary = _train_model(cfg, chunk_id, device, run_tag=cfg.run_tag, save_checkpoints=False)

        value = float(summary["best_val_loss"])
        if not np.isfinite(value):
            return HPO_FAILURE_VALUE
        return value

    except Exception as exc:
        print(f"  Trial {trial.number} failed safely: {exc}")
        cleanup_memory()
        return HPO_FAILURE_VALUE


def run_hpo(config: MTGNNRegimeConfig, chunk_id: int, trials: int, fresh: bool = False) -> Dict[str, Any]:
    if not HAS_OPTUNA:
        raise ImportError("optuna required: pip install optuna")

    config.resolve_paths()
    configure_torch_runtime(config)
    seed_everything(config.seed)
    validate_required_inputs(config, chunk_id, ("train", "val"))

    label = CHUNK_CONFIG[chunk_id]["label"]

    print(f"\n{'=' * 72}")
    print(f"  MTGNN REGIME HPO — {label} ({trials} trials)")
    print(f"{'=' * 72}")

    study_dir = Path(config.output_dir) / "codeResults" / "MTGNNRegime"
    study_dir.mkdir(parents=True, exist_ok=True)

    db_path = study_dir / f"hpo_{label}.db"
    study_name = f"mtgnn_regime_{label}"

    if fresh and db_path.exists():
        db_path.unlink()
        print(f"  Deleted old HPO database: {db_path}")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=int(config.seed),
            n_startup_trials=min(int(config.hpo_n_startup), int(trials)),
        ),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
        storage=f"sqlite:///{db_path}",
        study_name=study_name,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: _hpo_objective(trial, config, chunk_id),
        n_trials=int(trials),
        show_progress_bar=True,
    )

    usable = [
        t for t in study.trials
        if t.value is not None
        and np.isfinite(float(t.value))
        and float(t.value) < HPO_FAILURE_VALUE
    ]

    if not usable:
        raise RuntimeError("All MTGNN Regime HPO trials failed. No best params saved.")

    best_trial = min(usable, key=lambda t: float(t.value))
    best_params = dict(best_trial.params)
    best_value = float(best_trial.value)

    result = {"params": best_params, "value": best_value}
    path = study_dir / f"best_params_{label}.json"

    with open(path, "w") as f:
        json.dump(json_safe(result), f, indent=2)

    print(f"\n  Best HPO: {best_params} (val_loss={best_value:.6f})")
    print(f"  Saved: {path}")

    return result


def apply_hpo_params_if_available(config: MTGNNRegimeConfig, chunk_id: int) -> MTGNNRegimeConfig:
    label = CHUNK_CONFIG[chunk_id]["label"]
    path = Path(config.output_dir) / "codeResults" / "MTGNNRegime" / f"best_params_{label}.json"

    if not path.exists():
        print("  No HPO params found. Using current/default config.")
        return config

    with open(path) as f:
        result = json.load(f)

    value = float(result.get("value", HPO_FAILURE_VALUE))
    if not np.isfinite(value) or value >= HPO_FAILURE_VALUE:
        raise RuntimeError(f"Invalid HPO file: {path}")

    params = result.get("params", {})

    for key in [
        "node_hidden_dim",
        "graph_hidden_dim",
        "classifier_hidden_dim",
        "dropout",
        "top_k",
        "lr",
        "weight_decay",
    ]:
        if key not in params:
            continue

        if key == "lr":
            config.learning_rate = float(params[key])
        elif key == "weight_decay":
            config.weight_decay = float(params[key])
        elif key in {"dropout"}:
            setattr(config, key, float(params[key]))
        else:
            setattr(config, key, int(params[key]))

    print(f"Loaded HPO params: {params} (val_loss={value:.6f})")
    return config


# ═══════════════════════════════════════════════════════════════════════════════
# XAI / PREDICT
# ═══════════════════════════════════════════════════════════════════════════════

def top_edges_from_adjacency(adj: np.ndarray, tickers: List[str], top_n: int) -> List[Dict[str, Any]]:
    n = adj.shape[0]
    flat = adj.reshape(-1)
    top_n = min(int(top_n), flat.size)

    idx = np.argpartition(flat, -top_n)[-top_n:]
    edges = []

    for flat_idx in idx:
        src = int(flat_idx // n)
        dst = int(flat_idx % n)
        if src == dst:
            continue
        weight = float(adj[src, dst])
        if weight <= 0:
            continue
        edges.append({"source": tickers[src], "target": tickers[dst], "weight": weight})

    edges.sort(key=lambda x: x["weight"], reverse=True)
    return edges[:top_n]


def counterfactual_xai(
    model: MTGNNRegimeModel,
    node_features: torch.Tensor,
    macro_features: torch.Tensor,
    device: torch.device,
) -> List[Dict[str, Any]]:
    model.eval()

    scenarios = []

    with torch.no_grad():
        base = model(node_features, macro_features)

        neutral_macro = torch.zeros_like(macro_features)
        out_neutral_macro = model(node_features, neutral_macro)

        no_text = node_features.clone()
        if no_text.size(-1) >= 512:
            no_text[..., 256:] = 0.0
        out_no_text = model(no_text, macro_features)

        damp_recent = node_features * 0.85
        out_damp = model(damp_recent, macro_features)

    def pack(out: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        probs = out["probs"][0].detach().cpu().numpy()
        rid = int(np.argmax(probs))
        return {
            "regime_id": rid,
            "regime_label": REGIME_LABELS[rid],
            "confidence": float(np.max(probs)),
            "probs": {REGIME_LABELS[i]: float(probs[i]) for i in range(4)},
            "transition_probability": float(out["transition_probability"][0].detach().cpu().item()),
        }

    scenarios.append({"condition": "Original", "output": pack(base)})
    scenarios.append({"condition": "Macro features neutralised", "output": pack(out_neutral_macro)})
    scenarios.append({"condition": "FinBERT/text half of node features removed", "output": pack(out_no_text)})
    scenarios.append({"condition": "All node features dampened by 15%", "output": pack(out_damp)})

    return scenarios


def predict_with_xai(config: MTGNNRegimeConfig, chunk_id: int, split: str) -> Dict[str, Any]:
    config.resolve_paths()
    configure_torch_runtime(config)
    seed_everything(config.seed)

    validate_required_inputs(config, chunk_id, (split,))

    device = resolve_device(config.device)
    label = CHUNK_CONFIG[chunk_id]["label"]

    model_dir = model_dir_for(config, label, "")
    model_path = model_dir / "best_model.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"No trained MTGNN Regime model found: {model_path}")

    model, ckpt = MTGNNRegimeModel.load(model_path, config=config, device=str(device))
    config = model.config.resolve_paths()
    model.eval()

    dataset = RegimeSnapshotDataset(
        config,
        chunk_id,
        split,
        fit_stats=False,
        macro_stats=ckpt["macro_stats"],
        label_stats=ckpt["label_stats"],
        node_tickers=ckpt["node_tickers"],
    )

    loader = make_loader(dataset, config, train=False)

    result_rows = []
    xai_records = []
    prev_edge_set: Optional[set] = None

    with torch.no_grad():
        bar = tqdm(loader, desc=f"  Predict {label}_{split} bs={config.batch_size}", leave=False, unit="batch")

        for batch_idx, batch in enumerate(bar):
            node_features = batch["node_features"].to(device, non_blocking=True)
            macro_features = batch["macro_features"].to(device, non_blocking=True)
            true_label = batch["label"].cpu().numpy()

            out = model(node_features, macro_features)
            probs = out["probs"].detach().cpu().numpy()
            graph_props = out["graph_properties"].detach().cpu().numpy()
            adj = out["adjacency"].detach().cpu().numpy()

            dates = batch["date"]

            for i in range(len(dates)):
                rid = int(np.argmax(probs[i]))
                confidence = float(np.max(probs[i]))
                true_id = int(true_label[i])
                date_str = str(dates[i])

                top_edges = top_edges_from_adjacency(adj[i], dataset.node_tickers, int(config.xai_top_edges))
                edge_set = {(e["source"], e["target"]) for e in top_edges}

                if prev_edge_set is None:
                    added_edges = []
                    removed_edges = []
                else:
                    added_edges = sorted(list(edge_set - prev_edge_set))[:50]
                    removed_edges = sorted(list(prev_edge_set - edge_set))[:50]

                prev_edge_set = edge_set

                result_rows.append({
                    "date": date_str,
                    "pred_regime_id": rid,
                    "pred_regime_label": REGIME_LABELS[rid],
                    "true_regime_id": true_id,
                    "true_regime_label": REGIME_LABELS[true_id],
                    "confidence": confidence,
                    "transition_probability": float(out["transition_probability"][i].detach().cpu().item()),
                    "prob_calm": float(probs[i, 0]),
                    "prob_volatile": float(probs[i, 1]),
                    "prob_crisis": float(probs[i, 2]),
                    "prob_rotation": float(probs[i, 3]),
                    "graph_density": float(graph_props[i, 0]),
                    "avg_degree_norm": float(graph_props[i, 1]),
                    "std_degree_norm": float(graph_props[i, 2]),
                    "mean_edge_weight": float(graph_props[i, 3]),
                    "max_edge_weight": float(graph_props[i, 4]),
                    "graph_entropy": float(graph_props[i, 5]),
                    "learned_graph_stress": float(graph_props[i, 6]),
                    "macro_stress_score": float(batch["macro_stress_score"][i].cpu().item()),
                    "label_graph_stress_score": float(batch["graph_stress_score"][i].cpu().item()),
                })

                xai_records.append({
                    "date": date_str,
                    "regime_label": REGIME_LABELS[rid],
                    "confidence": confidence,
                    "level1_graph_properties": {
                        "density": float(graph_props[i, 0]),
                        "avg_degree_norm": float(graph_props[i, 1]),
                        "mean_edge_weight": float(graph_props[i, 3]),
                        "graph_entropy": float(graph_props[i, 5]),
                        "learned_graph_stress": float(graph_props[i, 6]),
                    },
                    "level2_top_edges": top_edges,
                    "level2_graph_diff": {
                        "added_edges": [{"source": a, "target": b} for a, b in added_edges],
                        "removed_edges": [{"source": a, "target": b} for a, b in removed_edges],
                    },
                })

    predictions = pd.DataFrame(result_rows)

    # Counterfactuals on a small sample.
    cf_records = []
    n_cf = min(int(config.xai_counterfactuals), len(dataset))

    for idx in range(n_cf):
        sample = dataset[idx]
        node_features = sample["node_features"].unsqueeze(0).to(device)
        macro_features = sample["macro_features"].unsqueeze(0).to(device)
        cf_records.append({
            "date": sample["date"],
            "counterfactuals": counterfactual_xai(model, node_features, macro_features, device),
        })

    xai = {
        "module": "MTGNNRegime",
        "chunk": label,
        "split": split,
        "regime_labels": REGIME_LABELS,
        "macro_cols": dataset.macro_cols,
        "dataset_audit": dataset.audit,
        "level1_level2_records": xai_records,
        "level3_counterfactuals": cf_records,
        "explanation_summary": {
            "plain_english": (
                "The regime module builds a learned cross-asset graph from temporal and FinBERT node features, "
                "extracts graph properties, combines them with FRED macro features, and classifies the current "
                "market state as calm, volatile, crisis, or rotation. XAI reports graph properties, strongest "
                "learned edges, graph changes between snapshots, and counterfactual behaviour under neutralised "
                "macro or removed text context."
            )
        },
    }

    results_dir = Path(config.output_dir) / "results" / "MTGNNRegime"
    xai_dir = results_dir / "xai"
    results_dir.mkdir(parents=True, exist_ok=True)
    xai_dir.mkdir(parents=True, exist_ok=True)

    pred_path = results_dir / f"predictions_{label}_{split}.csv"
    xai_path = xai_dir / f"{label}_{split}_xai.json"
    compact_xai_path = xai_dir / f"{label}_{split}_xai_summary.json"

    predictions.to_csv(pred_path, index=False)

    with open(xai_path, "w") as f:
        json.dump(json_safe(xai), f, indent=2)

    compact = {
        "module": xai["module"],
        "chunk": xai["chunk"],
        "split": xai["split"],
        "dataset_audit": xai["dataset_audit"],
        "sample_records": xai_records[: min(10, len(xai_records))],
        "counterfactuals": cf_records,
        "explanation_summary": xai["explanation_summary"],
    }

    with open(compact_xai_path, "w") as f:
        json.dump(json_safe(compact), f, indent=2)

    print(f"  Predictions saved: {pred_path} ({len(predictions):,} rows)")
    print(f"  XAI saved: {xai_path}")

    shutdown_dataloader(loader)
    cleanup_memory()

    return {
        "predictions": predictions,
        "xai": xai,
        "paths": {
            "predictions": str(pred_path),
            "xai": str(xai_path),
            "xai_summary": str(compact_xai_path),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD GRAPH SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph_summary(config: MTGNNRegimeConfig, chunk_id: int, split: str) -> pd.DataFrame:
    config.resolve_paths()
    validate_required_inputs(config, chunk_id, (split,))

    if split == "train":
        dataset = RegimeSnapshotDataset(config, chunk_id, split, fit_stats=True)
    else:
        train_dataset = RegimeSnapshotDataset(config, chunk_id, "train", fit_stats=True)
        dataset = RegimeSnapshotDataset(
            config,
            chunk_id,
            split,
            fit_stats=False,
            macro_stats=train_dataset.macro_stats,
            label_stats=train_dataset.label_stats,
            node_tickers=train_dataset.node_tickers,
        )

    rows = []
    for record in dataset.records:
        row = {
            "date": record["date"],
            "label": record["label_name"],
            "macro_stress_score": record["macro_stress_score"],
            "graph_stress_score": record["graph_stress_score"],
        }
        row.update(record["raw_metrics"])
        rows.append(row)

    df = pd.DataFrame(rows)

    out_dir = Path(config.output_dir) / "results" / "MTGNNRegime"
    out_dir.mkdir(parents=True, exist_ok=True)

    label = CHUNK_CONFIG[chunk_id]["label"]
    path = out_dir / f"graph_summary_{label}_{split}.csv"
    df.to_csv(path, index=False)

    print(f"  Saved graph summary: {path} ({len(df):,} rows)")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# INSPECT / SMOKE
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_inspect(config: MTGNNRegimeConfig) -> None:
    config.resolve_paths()

    print("=" * 72)
    print("MTGNN REGIME — DATA INSPECTION")
    print("=" * 72)

    for chunk_id in [1, 2, 3]:
        label = CHUNK_CONFIG[chunk_id]["label"]
        for split in ["train", "val", "test"]:
            t_emb, t_man = temporal_paths(config, chunk_id, split)
            f_emb, f_meta = finbert_paths(config, chunk_id, split)

            status = {
                "temporal_emb": t_emb.exists(),
                "temporal_manifest": t_man.exists(),
                "finbert_emb": f_emb.exists(),
                "finbert_meta": f_meta.exists(),
            }
            print(f"  {label}_{split}: {status}")

            if t_emb.exists():
                arr = np.load(t_emb, mmap_mode="r")
                print(f"    temporal shape={arr.shape}")
            if f_emb.exists():
                arr = np.load(f_emb, mmap_mode="r")
                print(f"    finbert shape={arr.shape}")

    fred = load_fred_frame(config)
    macro_cols = select_macro_cols(config, fred)
    print(f"\n  FRED: {config.fred_path}")
    print(f"  FRED shape: {fred.shape}")
    print(f"  selected macro cols: {macro_cols}")

    graph_paths = sorted(Path(config.graph_snapshot_dir).glob("edges_*.csv"))
    print(f"\n  Graph snapshots: {len(graph_paths)}")
    if graph_paths:
        print(f"  first: {graph_paths[0]}")
        print(f"  last : {graph_paths[-1]}")
        sample = pd.read_csv(graph_paths[0], nrows=5)
        print(f"  sample columns: {list(sample.columns)}")

    cik_map = load_cik_ticker_map(config)
    print(f"\n  CIK->ticker mappings loaded: {len(cik_map):,}")


def cmd_smoke(config: MTGNNRegimeConfig) -> None:
    config.resolve_paths()
    configure_torch_runtime(config)
    seed_everything(config.seed)

    device = resolve_device(config.device)

    print("=" * 72)
    print("MTGNN REGIME — SMOKE TEST")
    print("=" * 72)

    b = 2
    n = 64
    node_features = torch.randn(b, n, int(config.node_feature_dim), device=device)
    macro_features = torch.randn(b, len(DEFAULT_MACRO_COLS), device=device)
    labels = torch.tensor([0, 2], dtype=torch.long, device=device)

    model = MTGNNRegimeModel(config, macro_dim=len(DEFAULT_MACRO_COLS)).to(device)
    out = model(node_features, macro_features)

    loss = F.cross_entropy(out["logits"], labels)
    if not torch.isfinite(loss):
        raise RuntimeError("Smoke failed: non-finite loss.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
    optimizer.step()

    print("SMOKE TEST PASSED")
    print(f"  loss={float(loss.detach().cpu()):.6f}")
    print(f"  logits_shape={tuple(out['logits'].shape)}")
    print(f"  adjacency_shape={tuple(out['adjacency'].shape)}")
    print(f"  graph_properties_shape={tuple(out['graph_properties'].shape)}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--node-limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--max-snapshots", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")


def config_from_args(args: argparse.Namespace) -> MTGNNRegimeConfig:
    config = MTGNNRegimeConfig()

    if getattr(args, "repo_root", ""):
        config.repo_root = args.repo_root

    config.resolve_paths()

    if getattr(args, "device", None):
        config.device = args.device
    # if getattr(args, "node_limit", None) is not None:
    #     config.node_limit = int(args.node_limit)
    if getattr(args, "node_limit", None) is not None:
        config.node_limit = int(args.node_limit)
        if getattr(args, "command", "") == "hpo":
            config.hpo_node_limit = int(args.node_limit)
    if getattr(args, "top_k", None) is not None:
        config.top_k = int(args.top_k)
    if getattr(args, "epochs", None) is not None:
        config.epochs = int(args.epochs)
    if getattr(args, "batch_size", None) is not None:
        config.batch_size = int(args.batch_size)
    if getattr(args, "num_workers", None) is not None:
        config.num_workers = int(args.num_workers)
    if getattr(args, "cpu_threads", None) is not None:
        config.cpu_threads = int(args.cpu_threads)
    if getattr(args, "max_snapshots", None) is not None:
        config.max_snapshots = int(args.max_snapshots)
    if getattr(args, "deterministic", False):
        config.deterministic = True
    if getattr(args, "trials", None) is not None:
        config.hpo_trials = int(args.trials)

    return config.resolve_paths()


def main() -> None:
    parser = argparse.ArgumentParser(description="MTGNN Regime Detection Module")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("inspect", help="Inspect required input files")
    add_common_args(p)

    p = sub.add_parser("smoke", help="Synthetic forward/backward smoke test")
    add_common_args(p)

    p = sub.add_parser("build-graph", help="Build graph/regime summary table")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    add_common_args(p)

    p = sub.add_parser("hpo", help="Run Optuna TPE HPO")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--fresh", action="store_true")
    add_common_args(p)

    p = sub.add_parser("train-best", help="Train with best HPO params if available")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--fresh", action="store_true")
    add_common_args(p)

    p = sub.add_parser("train-best-all", help="Train all available chunks")
    p.add_argument("--fresh", action="store_true")
    add_common_args(p)

    p = sub.add_parser("predict", help="Predict regime and generate XAI")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    add_common_args(p)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    config = config_from_args(args)

    if args.command == "inspect":
        cmd_inspect(config)

    elif args.command == "smoke":
        cmd_smoke(config)

    elif args.command == "build-graph":
        build_graph_summary(config, args.chunk, args.split)

    elif args.command == "hpo":
        run_hpo(config, args.chunk, int(args.trials), fresh=bool(args.fresh))

    elif args.command == "train-best":
        config = apply_hpo_params_if_available(config, args.chunk)
        train_regime_model(config, args.chunk, fresh=bool(args.fresh))

    elif args.command == "train-best-all":
        for chunk_id in [1, 2, 3]:
            validate_required_inputs(config, chunk_id, ("train", "val"))
            chunk_config = apply_hpo_params_if_available(config, chunk_id)
            train_regime_model(chunk_config, chunk_id, fresh=bool(args.fresh))

    elif args.command == "predict":
        result = predict_with_xai(config, args.chunk, args.split)
        print(f"  Returned keys: {list(result.keys())}")


if __name__ == "__main__":
    main()


# ── Run instructions ─────────────────────────────────────────────────────────
# Compile:
# python -m py_compile code/gnn/mtgnn_regime.py
#
# Inspect:
# python code/gnn/mtgnn_regime.py inspect --repo-root .
#
# Smoke:
# python code/gnn/mtgnn_regime.py smoke --repo-root . --device cuda
#
# Build graph/regime summary:
# python code/gnn/mtgnn_regime.py build-graph --repo-root . --chunk 1 --split train --device cuda --node-limit 512
#
# Small HPO:
# python code/gnn/mtgnn_regime.py hpo --repo-root . --chunk 1 --trials 3 --device cuda --fresh
#
# Full HPO:
# python code/gnn/mtgnn_regime.py hpo --repo-root . --chunk 1 --trials 30 --device cuda --fresh
#
# Train:
# python code/gnn/mtgnn_regime.py train-best --repo-root . --chunk 1 --device cuda --fresh
#
# Predict:
# python code/gnn/mtgnn_regime.py predict --repo-root . --chunk 1 --split test --device cuda

