#!/usr/bin/env python3
"""
code/analysts/technical_analyst.py

Technical Analyst — BiLSTM + Attention Pooling
==============================================

Project:
    fin-glassbox — Explainable Multimodal Neural Framework for Financial Risk Management

Purpose:
    The Technical Analyst consumes Temporal Encoder embeddings and produces:

        1. trend_score        — directional technical signal, range [0, 1]
        2. momentum_score     — strength of recent/future momentum, range [0, 1]
        3. timing_confidence  — timing quality / entry-confidence style score, range [0, 1]

Input:
    Manifest-aligned temporal embeddings:

        outputs/embeddings/TemporalEncoder/chunk1_train_embeddings.npy
        outputs/embeddings/TemporalEncoder/chunk1_train_manifest.csv

    The model builds a second-level sequence over the temporal embeddings:

        input shape = [batch, analyst_seq_len=30, embedding_dim=256]

    Each row-level Temporal Encoder embedding already summarises a price-feature window.
    This Technical Analyst then looks at a sequence of these embeddings to model higher-level
    technical behaviour.

Targets:
    Targets are derived from forward returns and technical indicators:

        trend target:
            20d forward return > +0.5%  -> 1.0
            20d forward return < -0.5%  -> 0.0
            otherwise                   -> 0.5

        momentum target:
            |5d forward return| / recent 5d volatility, clamped to [0, 1]

        timing target:
            uses RSI, MACD histogram, and price_pos from features_temporal.csv
            RSI < 30 and price_pos > 0.5 -> 1.0
            RSI > 70 and price_pos < 0.5 -> 0.0
            otherwise -> 0.5 + 0.5 * normalised MACD histogram, clamped to [0, 1]

XAI:
    Level 1:
        attention weights over the 30 embedding timesteps

    Level 2:
        gradient-based embedding-dimension and timestep importance

    Level 3:
        counterfactual scenarios showing how scores change if recent sequence information is perturbed

CLI:
    python code/analysts/technical_analyst.py inspect --repo-root .
    python code/analysts/technical_analyst.py smoke --repo-root . --device cuda
    python code/analysts/technical_analyst.py hpo --repo-root . --chunk 1 --trials 40 --device cuda --fresh
    python code/analysts/technical_analyst.py train-best --repo-root . --chunk 1 --device cuda --fresh
    python code/analysts/technical_analyst.py predict --repo-root . --chunk 1 --split test --device cuda
"""

from __future__ import annotations

import argparse
import gc
import json
from contextlib import nullcontext
import math
import random
import shutil
import time
import warnings
from dataclasses import asdict, dataclass, field, fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
# GLOBAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

CHUNK_CONFIG = {
    1: {"train": (2000, 2004), "val": (2005, 2005), "test": (2006, 2006), "label": "chunk1"},
    2: {"train": (2007, 2014), "val": (2015, 2015), "test": (2016, 2016), "label": "chunk2"},
    3: {"train": (2017, 2022), "val": (2023, 2023), "test": (2024, 2024), "label": "chunk3"},
}

FEATURE_NAMES = [
    "log_return",
    "vol_5d",
    "vol_21d",
    "rsi_14",
    "macd_hist",
    "bb_pos",
    "volume_ratio",
    "hl_ratio",
    "price_pos",
    "spy_corr_63d",
]

REQUIRED_FEATURE_COLUMNS = ["ticker", "date", "rsi_14", "macd_hist", "price_pos"]

# Optuna SQLite/RDB storage must never receive inf/nan values.
HPO_FAILURE_VALUE = 1_000_000_000.0


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TechnicalAnalystConfig:
    repo_root: str = ""
    output_dir: str = "outputs"
    embeddings_dir: str = "outputs/embeddings/TemporalEncoder"
    returns_path: str = "data/yFinance/processed/returns_panel_wide.csv"
    features_path: str = "data/yFinance/processed/features_temporal.csv"

    # Input / architecture
    input_dim: int = 256
    analyst_seq_len: int = 30
    lstm_hidden: int = 64
    lstm_layers: int = 1
    bidirectional: bool = True
    dropout: float = 0.20
    attention_dim: int = 64

    # Training
    batch_size: int = 512
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stop_patience: int = 10
    gradient_clip: float = 1.0
    mixed_precision: bool = True

    # Target construction
    trend_horizon: int = 20
    momentum_horizon: int = 5
    trend_threshold: float = 0.005
    momentum_vol_lookback: int = 5
    neutral_target: float = 0.5

    # HPO
    hpo_trials: int = 40
    hpo_n_startup: int = 10
    hpo_epochs: int = 8
    hpo_max_train_samples: int = 250_000
    hpo_max_val_samples: int = 75_000

    # XAI
    xai_sample_size: int = 500
    xai_counterfactual_scenarios: int = 8

    # System
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 6
    cpu_threads: int = 6
    deterministic: bool = False
    persistent_workers: bool = True

    # Runtime control
    max_train_samples: int = 0
    max_val_samples: int = 0
    max_predict_samples: int = 0
    run_tag: str = ""
    save_checkpoints: bool = True

    def resolve_paths(self) -> "TechnicalAnalystConfig":
        if self.repo_root:
            root = Path(self.repo_root)
            for attr in ["output_dir", "embeddings_dir", "returns_path", "features_path"]:
                value = getattr(self, attr)
                if value and not Path(value).is_absolute():
                    setattr(self, attr, str(root / value))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
# RUNTIME UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def configure_torch_runtime(config: TechnicalAnalystConfig) -> None:
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
def cudnn_disabled_if_cuda(device: torch.device):
    """Disable cuDNN only for eval-mode RNN/LSTM gradient XAI.

    cuDNN RNN backward fails if the RNN forward pass was executed in eval mode.
    For XAI we want eval mode because dropout should not affect explanations.
    Therefore, we keep model.eval() but disable cuDNN only during attribution
    forward/backward passes.
    """
    if device.type == "cuda" and torch.backends.cudnn.enabled:
        return torch.backends.cudnn.flags(enabled=False)
    return nullcontext()

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


def finite_ratio_np(array: np.ndarray) -> float:
    arr = np.asarray(array)
    if arr.size == 0:
        return 1.0
    return float(np.isfinite(arr).mean())


def load_config_from_checkpoint_dict(raw: Dict[str, Any], fallback: Optional[TechnicalAnalystConfig] = None) -> TechnicalAnalystConfig:
    base = fallback.to_dict() if fallback is not None else TechnicalAnalystConfig().to_dict()
    ckpt_cfg = raw.get("config", {})
    if isinstance(ckpt_cfg, dict):
        base.update(ckpt_cfg)

    valid = {f.name for f in dataclass_fields(TechnicalAnalystConfig)}
    filtered = {k: v for k, v in base.items() if k in valid}
    return TechnicalAnalystConfig(**filtered).resolve_paths()


def required_embedding_files(
    config: TechnicalAnalystConfig,
    chunk_id: int,
    splits: Tuple[str, ...] = ("train", "val", "test"),
) -> List[Path]:
    label = CHUNK_CONFIG[chunk_id]["label"]
    emb_dir = Path(config.embeddings_dir)
    paths: List[Path] = []
    for split in splits:
        paths.append(emb_dir / f"{label}_{split}_embeddings.npy")
        paths.append(emb_dir / f"{label}_{split}_manifest.csv")
    return paths


def validate_required_embeddings(
    config: TechnicalAnalystConfig,
    chunk_id: int,
    splits: Tuple[str, ...] = ("train", "val", "test"),
) -> None:
    missing = [p for p in required_embedding_files(config, chunk_id, splits) if not p.exists()]
    if missing:
        label = CHUNK_CONFIG[chunk_id]["label"]
        missing_text = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            f"Temporal Encoder embeddings/manifests are missing for {label}.\n"
            f"The Technical Analyst depends on TemporalEncoder embeddings.\n"
            f"Missing files:\n{missing_text}\n\n"
            f"Finish TemporalEncoder embedding generation for {label}, then rerun this module."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_returns_frame(config: TechnicalAnalystConfig) -> pd.DataFrame:
    path = Path(config.returns_path)
    if not path.exists():
        raise FileNotFoundError(f"Returns file not found: {path}")

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df.astype(np.float32)


def load_features_frame(config: TechnicalAnalystConfig) -> pd.DataFrame:
    path = Path(config.features_path)
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")

    df = pd.read_csv(path, dtype={"ticker": str}, parse_dates=["date"])
    missing = [col for col in REQUIRED_FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"features_temporal.csv missing required columns: {missing}")

    keep_cols = list(dict.fromkeys(REQUIRED_FEATURE_COLUMNS))
    df = df[keep_cols].copy()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df.replace([np.inf, -np.inf], np.nan)

    for col in ["rsi_14", "macd_hist", "price_pos"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def split_date_index_bounds(dates: pd.DatetimeIndex, year_pair: Tuple[int, int]) -> Tuple[int, int]:
    mask = (dates.year >= year_pair[0]) & (dates.year <= year_pair[1])
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise ValueError(f"No dates found for years {year_pair}")
    return int(idx[0]), int(idx[-1])


def manifest_and_embedding_paths(config: TechnicalAnalystConfig, chunk_id: int, split: str) -> Tuple[Path, Path]:
    label = CHUNK_CONFIG[chunk_id]["label"]
    emb_dir = Path(config.embeddings_dir)
    manifest_path = emb_dir / f"{label}_{split}_manifest.csv"
    emb_path = emb_dir / f"{label}_{split}_embeddings.npy"
    return manifest_path, emb_path


# ═══════════════════════════════════════════════════════════════════════════════
# TARGET STATS
# ═══════════════════════════════════════════════════════════════════════════════

# def fit_target_stats(features_aligned: pd.DataFrame) -> Dict[str, float]:
#     macd = pd.to_numeric(features_aligned["macd_hist"], errors="coerce").replace([np.inf, -np.inf], np.nan)
#     macd_values = macd.dropna().values.astype(np.float64)

#     if len(macd_values) == 0:
#         mean = 0.0
#         std = 1.0
#     else:
#         mean = float(np.mean(macd_values))
#         std = float(np.std(macd_values))
#         if not np.isfinite(std) or std < 1e-8:
#             std = 1.0

#     return {
#         "macd_mean": float(mean),
#         "macd_std": float(std),
#     }
def fit_target_stats(features_aligned: pd.DataFrame) -> Dict[str, float]:
    """Robust MACD statistics for timing-target normalisation.

    Raw MACD can contain huge outliers after adjusted/unadjusted source stitching.
    Mean/std is too sensitive and can collapse the timing target toward 0.5.
    We use median and MAD after percentile clipping.
    """
    macd = pd.to_numeric(features_aligned["macd_hist"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = macd.dropna().values.astype(np.float64)
    if len(values) == 0: return {"macd_mean": 0.0, "macd_std": 1.0}

    lo, hi = np.nanpercentile(values, [1.0, 99.0])
    clipped = np.clip(values, lo, hi)
    median = float(np.nanmedian(clipped))
    mad = float(np.nanmedian(np.abs(clipped - median)))
    robust_std = 1.4826 * mad

    if not np.isfinite(robust_std) or robust_std < 1e-8:
        robust_std = float(np.nanstd(clipped))
    if not np.isfinite(robust_std) or robust_std < 1e-8:
        robust_std = 1.0

    return {
        "macd_mean": float(median),
        "macd_std": float(robust_std),
        "macd_clip_low": float(lo),
        "macd_clip_high": float(hi),
    }

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class TechnicalAnalystDataset(Dataset):
    """Manifest-based sequence dataset for the Technical Analyst.

    Each sample is a sequence of analyst_seq_len Temporal Encoder embeddings for one ticker.

    For a manifest row at date t:
        input  = embeddings[t-29 : t]
        target = rule-derived technical labels using future returns and current indicators
    """

    def __init__(
        self,
        config: TechnicalAnalystConfig,
        split: str,
        chunk_id: int,
        returns_df: pd.DataFrame,
        features_df: pd.DataFrame,
        *,
        target_stats: Optional[Dict[str, float]] = None,
        fit_stats: bool = False,
        max_samples: int = 0,
        label_suffix: str = "",
    ) -> None:
        self.config = config
        self.split = split
        self.chunk_id = int(chunk_id)
        self.chunk_label = CHUNK_CONFIG[chunk_id]["label"]
        self.label_suffix = label_suffix

        manifest_path, emb_path = manifest_and_embedding_paths(config, chunk_id, split)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        if not emb_path.exists():
            raise FileNotFoundError(f"Embeddings not found: {emb_path}")

        self.manifest = pd.read_csv(manifest_path, dtype={"ticker": str})
        self.manifest["date"] = pd.to_datetime(self.manifest["date"]).dt.strftime("%Y-%m-%d")
        self.manifest["row_id"] = np.arange(len(self.manifest), dtype=np.int64)

        self.embeddings = np.load(emb_path, mmap_mode="r")

        n = min(len(self.manifest), len(self.embeddings))
        self.manifest = self.manifest.iloc[:n].reset_index(drop=True)
        self.embeddings = self.embeddings[:n]
        self.n_rows = n

        if self.n_rows == 0:
            raise ValueError(f"No rows available for {self.chunk_label}_{split}")

        first_emb = np.asarray(self.embeddings[0])
        if first_emb.shape[0] != int(config.input_dim):
            raise ValueError(
                f"Embedding dimension mismatch for {self.chunk_label}_{split}: "
                f"expected {config.input_dim}, got {first_emb.shape[0]}"
            )

        returns_df = returns_df.sort_index()
        self.returns_df = returns_df
        self.return_dates = returns_df.index
        self.date_to_idx = {str(d)[:10]: i for i, d in enumerate(returns_df.index)}

        split_start_idx, split_end_idx = split_date_index_bounds(
            returns_df.index,
            CHUNK_CONFIG[chunk_id][split],
        )
        self.split_start_idx = split_start_idx
        self.split_end_idx = split_end_idx

        # Align features to manifest rows.
        features_subset = features_df.copy()
        features_subset["date"] = pd.to_datetime(features_subset["date"]).dt.strftime("%Y-%m-%d")

        aligned = self.manifest[["ticker", "date", "row_id"]].merge(
            features_subset,
            how="left",
            on=["ticker", "date"],
            sort=False,
        )
        aligned = aligned.sort_values("row_id").reset_index(drop=True)

        for col, default in [("rsi_14", 50.0), ("macd_hist", 0.0), ("price_pos", 0.5)]:
            aligned[col] = pd.to_numeric(aligned[col], errors="coerce")
            aligned[col] = aligned[col].replace([np.inf, -np.inf], np.nan).fillna(default)

        if fit_stats:
            self.target_stats = fit_target_stats(aligned)
        elif target_stats is not None:
            self.target_stats = dict(target_stats)
        else:
            self.target_stats = {"macd_mean": 0.0, "macd_std": 1.0}

        self.features_aligned = aligned

        print(f"  Building {self.chunk_label}_{split}{label_suffix}: rows={self.n_rows:,}")

        self.sequence_rows, self.targets, self.sample_tickers, self.sample_dates = self._build_sequences_and_targets()

        if max_samples and max_samples > 0 and len(self.sequence_rows) > int(max_samples):
            rng = np.random.default_rng(int(config.seed))
            idx = rng.choice(len(self.sequence_rows), size=int(max_samples), replace=False)
            idx = np.sort(idx)

            self.sequence_rows = self.sequence_rows[idx]
            self.targets = self.targets[idx]
            self.sample_tickers = self.sample_tickers[idx]
            self.sample_dates = self.sample_dates[idx]

        self.audit = self._audit_dataset()

        print(
            f"  {self.chunk_label}_{split}{label_suffix}: "
            f"{len(self.sequence_rows):,} samples, "
            f"{len(np.unique(self.sample_tickers)):,} tickers | "
            f"target_finite={self.audit['target_finite_ratio']:.6f}, "
            f"emb_sample_finite={self.audit['embedding_sample_finite_ratio']:.6f}"
        )

    def _build_sequences_and_targets(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        seq_len = int(self.config.analyst_seq_len)
        max_forward = max(int(self.config.trend_horizon), int(self.config.momentum_horizon))

        all_seq_rows: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []
        all_tickers: List[np.ndarray] = []
        all_dates: List[np.ndarray] = []

        macd_mean = float(self.target_stats.get("macd_mean", 0.0))
        macd_std = float(self.target_stats.get("macd_std", 1.0))
        if not np.isfinite(macd_std) or macd_std < 1e-8:
            macd_std = 1.0

        grouped = self.manifest.groupby("ticker", sort=True)

        for ticker, group in tqdm(grouped, desc=f"  Samples {self.chunk_label}_{self.split}", leave=False):
            ticker = str(ticker)

            if ticker not in self.returns_df.columns:
                continue

            group = group.sort_values("row_id")
            row_ids = group["row_id"].values.astype(np.int64)
            dates = group["date"].values.astype(str)

            if len(row_ids) < seq_len:
                continue

            returns = self.returns_df[ticker].values.astype(np.float64)
            returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

            ret_csum = np.concatenate([[0.0], np.cumsum(returns, dtype=np.float64)])

            positions = np.arange(seq_len - 1, len(row_ids), dtype=np.int64)
            end_row_ids = row_ids[positions]
            end_dates = dates[positions]

            end_return_idx = np.array([self.date_to_idx.get(str(d)[:10], -1) for d in end_dates], dtype=np.int64)

            # Strict split-local target rule:
            # target horizon must remain inside the same split's date range.
            valid_mask = (
                (end_return_idx >= self.split_start_idx) &
                ((end_return_idx + max_forward) <= self.split_end_idx)
            )

            if not np.any(valid_mask):
                continue

            positions = positions[valid_mask]
            end_row_ids = end_row_ids[valid_mask]
            end_dates = end_dates[valid_mask]
            end_return_idx = end_return_idx[valid_mask]

            offsets = np.arange(seq_len - 1, -1, -1, dtype=np.int64)
            seq_rows = row_ids[positions[:, None] - offsets[None, :]]

            # Forward returns: strictly after date t.
            idx = end_return_idx
            h20 = int(self.config.trend_horizon)
            h5 = int(self.config.momentum_horizon)

            fwd20 = ret_csum[idx + h20 + 1] - ret_csum[idx + 1]
            fwd5 = ret_csum[idx + h5 + 1] - ret_csum[idx + 1]

            trend = np.full(len(idx), float(self.config.neutral_target), dtype=np.float32)
            trend[fwd20 > float(self.config.trend_threshold)] = 1.0
            trend[fwd20 < -float(self.config.trend_threshold)] = 0.0

            # Recent realized volatility up to current day. Scale to 5-day horizon.
            recent_vol = np.zeros(len(idx), dtype=np.float64)
            lookback = int(self.config.momentum_vol_lookback)

            for j, t_idx in enumerate(idx):
                lo = max(0, int(t_idx) - lookback + 1)
                hi = int(t_idx) + 1
                window = returns[lo:hi]
                if len(window) < 2:
                    recent_vol[j] = 1e-6
                else:
                    recent_vol[j] = max(float(np.std(window) * math.sqrt(max(1, h5))), 1e-6)

            momentum = np.abs(fwd5) / (recent_vol + 1e-8)
            momentum = np.clip(momentum, 0.0, 1.0).astype(np.float32)

            # Timing from aligned current-row indicators.
            rsi = self.features_aligned.loc[end_row_ids, "rsi_14"].values.astype(np.float64)
            macd = self.features_aligned.loc[end_row_ids, "macd_hist"].values.astype(np.float64)
            price_pos = self.features_aligned.loc[end_row_ids, "price_pos"  ].values.astype(np.float64)

            rsi = np.nan_to_num(rsi, nan=50.0, posinf=50.0, neginf=50.0)
            macd = np.nan_to_num(macd, nan=0.0, posinf=0.0, neginf=0.0)
            price_pos = np.nan_to_num(price_pos, nan=0.5, posinf=0.5, neginf=0.5)

            # macd_z = np.clip((macd - macd_mean) / macd_std, -3.0, 3.0)
            macd_clip_low = self.target_stats.get("macd_clip_low", None)
            macd_clip_high = self.target_stats.get("macd_clip_high", None)
            if macd_clip_low is not None and macd_clip_high is not None: macd = np.clip(macd, float(macd_clip_low), float(macd_clip_high))
            macd_z = np.clip((macd - macd_mean) / macd_std, -3.0, 3.0)
            macd_norm = 1.0 / (1.0 + np.exp(-macd_z))

            timing = 0.5 + 0.5 * (macd_norm - 0.5) * 2.0
            timing = np.clip(timing, 0.0, 1.0)

            timing[(rsi < 30.0) & (price_pos > 0.5)] = 1.0
            timing[(rsi > 70.0) & (price_pos < 0.5)] = 0.0
            timing = timing.astype(np.float32)

            targets = np.stack([trend, momentum, timing], axis=1).astype(np.float32)
            targets = np.nan_to_num(targets, nan=0.5, posinf=1.0, neginf=0.0)
            targets = np.clip(targets, 0.0, 1.0).astype(np.float32)

            all_seq_rows.append(seq_rows.astype(np.int64))
            all_targets.append(targets)
            all_tickers.append(np.array([ticker] * len(targets), dtype=object))
            all_dates.append(end_dates.astype(object))

        if not all_seq_rows:
            raise ValueError(
                f"No usable Technical Analyst samples for {self.chunk_label}_{self.split}. "
                f"Check analyst_seq_len, manifests, and forward horizon availability."
            )

        sequence_rows = np.concatenate(all_seq_rows, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        tickers = np.concatenate(all_tickers, axis=0)
        dates = np.concatenate(all_dates, axis=0)

        return sequence_rows, targets, tickers, dates

    def _audit_dataset(self) -> Dict[str, float]:
        sample_n = min(2048, len(self.sequence_rows))
        if sample_n > 0:
            idx = np.linspace(0, len(self.sequence_rows) - 1, sample_n).astype(np.int64)
            rows = self.sequence_rows[idx].reshape(-1)
            emb_sample = np.asarray(self.embeddings[rows])
        else:
            emb_sample = np.array([], dtype=np.float32)

        return {
            "samples": int(len(self.sequence_rows)),
            "embedding_sample_finite_ratio": finite_ratio_np(emb_sample),
            "target_finite_ratio": finite_ratio_np(self.targets),
            "target_trend_mean": float(np.mean(self.targets[:, 0])) if len(self.targets) else 0.0,
            "target_momentum_mean": float(np.mean(self.targets[:, 1])) if len(self.targets) else 0.0,
            "target_timing_mean": float(np.mean(self.targets[:, 2])) if len(self.targets) else 0.0,
        }

    def __len__(self) -> int:
        return int(len(self.sequence_rows))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rows = self.sequence_rows[idx]
        seq = np.asarray(self.embeddings[rows], dtype=np.float32).copy()
        seq = sanitize_np_array(seq, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "x": torch.from_numpy(seq),
            "target": torch.from_numpy(self.targets[idx].astype(np.float32)),
            "ticker": str(self.sample_tickers[idx]),
            "date": str(self.sample_dates[idx])[:10],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class AttentionPooling(nn.Module):
    """Additive attention pooling over BiLSTM timesteps."""

    def __init__(self, input_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, S, D]
        h = torch.tanh(self.proj(x))
        scores = self.score(h).squeeze(-1)  # [B, S]
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return context, weights


class TechnicalAnalystModel(nn.Module):
    """BiLSTM + attention model for technical signal analysis."""

    def __init__(self, config: TechnicalAnalystConfig) -> None:
        super().__init__()
        self.config = config

        lstm_dropout = float(config.dropout) if int(config.lstm_layers) > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=int(config.input_dim),
            hidden_size=int(config.lstm_hidden),
            num_layers=int(config.lstm_layers),
            batch_first=True,
            bidirectional=bool(config.bidirectional),
            dropout=lstm_dropout,
        )

        lstm_out_dim = int(config.lstm_hidden) * (2 if config.bidirectional else 1)
        self.attention = AttentionPooling(lstm_out_dim, int(config.attention_dim))

        self.head = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Dropout(float(config.dropout)),
            nn.Linear(lstm_out_dim, max(32, lstm_out_dim // 2)),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(max(32, lstm_out_dim // 2), 3),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [B, S, 256]
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)

        lstm_out, _ = self.lstm(x)
        context, attention_weights = self.attention(lstm_out)

        logits = self.head(context)
        scores = torch.sigmoid(logits)

        return {
            "logits": logits,
            "scores": scores,
            "trend_score": scores[:, 0],
            "momentum_score": scores[:, 1],
            "timing_confidence": scores[:, 2],
            "attention_weights": attention_weights,
            "context": context,
        }

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": self.config.to_dict(),
        }, path)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        config: Optional[TechnicalAnalystConfig] = None,
        device: str = "cpu",
    ) -> "TechnicalAnalystModel":
        path = Path(path)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = load_config_from_checkpoint_dict(ckpt, fallback=config)
        model = cls(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        return model


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS / DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def technical_loss(output: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    target = torch.nan_to_num(target.float(), nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    scores = torch.nan_to_num(output["scores"].float(), nan=0.5, posinf=1.0, neginf=0.0).clamp(1e-5, 1.0 - 1e-5)

    trend_loss = F.smooth_l1_loss(scores[:, 0], target[:, 0])
    momentum_loss = F.smooth_l1_loss(scores[:, 1], target[:, 1])
    timing_loss = F.smooth_l1_loss(scores[:, 2], target[:, 2])

    return trend_loss + momentum_loss + timing_loss


def tensors_are_finite(*tensors: torch.Tensor) -> bool:
    return all(torch.isfinite(t).all().item() for t in tensors)


def make_loader(dataset: Dataset, config: TechnicalAnalystConfig, train: bool) -> DataLoader:
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
        kwargs["prefetch_factor"] = 4

    return DataLoader(dataset, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def model_dir_for(config: TechnicalAnalystConfig, chunk_label: str, run_tag: str = "") -> Path:
    base = Path(config.output_dir) / "models" / "TechnicalAnalyst"
    if run_tag:
        return base / "_hpo_trials" / run_tag / chunk_label
    return base / chunk_label


def build_train_val_datasets(
    config: TechnicalAnalystConfig,
    chunk_id: int,
    returns_df: pd.DataFrame,
    features_df: pd.DataFrame,
    *,
    run_tag: str = "",
) -> Tuple[TechnicalAnalystDataset, TechnicalAnalystDataset]:
    train_limit = int(config.max_train_samples) if config.max_train_samples else 0
    val_limit = int(config.max_val_samples) if config.max_val_samples else 0

    train_ds = TechnicalAnalystDataset(
        config,
        "train",
        chunk_id,
        returns_df,
        features_df,
        fit_stats=True,
        max_samples=train_limit,
        label_suffix=f"_{run_tag}" if run_tag else "",
    )

    val_ds = TechnicalAnalystDataset(
        config,
        "val",
        chunk_id,
        returns_df,
        features_df,
        target_stats=train_ds.target_stats,
        fit_stats=False,
        max_samples=val_limit,
        label_suffix=f"_{run_tag}" if run_tag else "",
    )

    return train_ds, val_ds


def _train_model(
    config: TechnicalAnalystConfig,
    chunk_id: int,
    returns_df: pd.DataFrame,
    features_df: pd.DataFrame,
    device: torch.device,
    *,
    run_tag: str = "",
    save_checkpoints: bool = True,
) -> Tuple[TechnicalAnalystModel, Dict[str, Any]]:
    label = CHUNK_CONFIG[chunk_id]["label"]

    train_ds, val_ds = build_train_val_datasets(
        config,
        chunk_id,
        returns_df,
        features_df,
        run_tag=run_tag,
    )

    train_loader = make_loader(train_ds, config, train=True)
    val_loader = make_loader(val_ds, config, train=False)

    model = TechnicalAnalystModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )

    use_amp = bool(config.mixed_precision and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out_dir = model_dir_for(config, label, "" if save_checkpoints else run_tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_path = out_dir / "best_model.pt"
    latest_path = out_dir / "latest_model.pt"
    history_path = out_dir / "training_history.csv"
    summary_path = out_dir / "training_summary.json"
    stats_path = out_dir / "target_stats.json"

    best_val_loss = float("inf")
    no_improve = 0
    history: List[Dict[str, Any]] = []
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(
        f"  Train: {len(train_ds):,} | Val: {len(val_ds):,} | "
        f"batch_size={config.batch_size} | params={total_params:,}"
    )
    print(f"  Target stats: {train_ds.target_stats}")

    epoch = 0

    try:
        for epoch in range(1, int(config.epochs) + 1):
            epoch_start = time.time()

            model.train()
            train_loss_sum = 0.0
            train_batches = 0

            train_bar = tqdm(
                train_loader,
                desc=f"  [{label}] E{epoch:03d}/{config.epochs} train bs={config.batch_size}",
                leave=False,
                unit="batch",
            )

            for batch in train_bar:
                x = batch["x"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)

                if not tensors_are_finite(x, target):
                    raise RuntimeError("Non-finite input/target detected before forward pass.")

                optimizer.zero_grad(set_to_none=True)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        output = model(x)
                        loss = technical_loss(output, target)

                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite training loss at epoch {epoch}")

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.gradient_clip))
                    scaler.step(optimizer)
                    scaler.update()

                else:
                    output = model(x)
                    loss = technical_loss(output, target)

                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite training loss at epoch {epoch}")

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.gradient_clip))
                    optimizer.step()

                batch_loss = float(loss.detach().cpu())
                train_loss_sum += batch_loss
                train_batches += 1
                train_bar.set_postfix(loss=f"{batch_loss:.5f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

            train_loss = train_loss_sum / max(train_batches, 1)

            model.eval()
            val_loss_sum = 0.0
            val_batches = 0

            val_bar = tqdm(
                val_loader,
                desc=f"  [{label}] E{epoch:03d}/{config.epochs} val   bs={config.batch_size}",
                leave=False,
                unit="batch",
            )

            with torch.no_grad():
                for batch in val_bar:
                    x = batch["x"].to(device, non_blocking=True)
                    target = batch["target"].to(device, non_blocking=True)

                    if not tensors_are_finite(x, target):
                        raise RuntimeError("Non-finite validation input/target detected.")

                    output = model(x)
                    loss = technical_loss(output, target)

                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite validation loss at epoch {epoch}")

                    batch_loss = float(loss.detach().cpu())
                    val_loss_sum += batch_loss
                    val_batches += 1
                    val_bar.set_postfix(loss=f"{batch_loss:.5f}")

            val_loss = val_loss_sum / max(val_batches, 1)

            row = {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "seconds": round(time.time() - epoch_start, 2),
            }
            history.append(row)

            print(
                f"  [{label}] E{epoch:03d}/{config.epochs} | "
                f"train={train_loss:.6f} | val={val_loss:.6f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} | {row['seconds']:.1f}s"
            )

            if val_loss < best_val_loss:
                best_val_loss = float(val_loss)
                no_improve = 0
                if save_checkpoints:
                    model.save(best_path)
            else:
                no_improve += 1

            if save_checkpoints:
                model.save(latest_path)

            if no_improve >= int(config.early_stop_patience):
                print(f"  Early stopping at epoch {epoch}")
                break

    finally:
        shutdown_dataloader(train_loader)
        shutdown_dataloader(val_loader)
        cleanup_memory()

    if not np.isfinite(best_val_loss):
        raise RuntimeError("Training failed: best validation loss is not finite.")

    if save_checkpoints:
        pd.DataFrame(history).to_csv(history_path, index=False)
        with open(stats_path, "w") as f:
            json.dump(train_ds.target_stats, f, indent=2)

        if best_path.exists():
            model = TechnicalAnalystModel.load(best_path, config=config, device=str(device))

    summary = {
        "chunk": label,
        "run_tag": run_tag,
        "best_val_loss": float(best_val_loss),
        "epochs_trained": int(epoch),
        "total_params": int(total_params),
        "train_samples": int(len(train_ds)),
        "val_samples": int(len(val_ds)),
        "batch_size": int(config.batch_size),
        "lstm_hidden": int(config.lstm_hidden),
        "lstm_layers": int(config.lstm_layers),
        "dropout": float(config.dropout),
        "learning_rate": float(config.learning_rate),
        "weight_decay": float(config.weight_decay),
        "train_audit": train_ds.audit,
        "val_audit": val_ds.audit,
        "target_stats": train_ds.target_stats,
        "saved_checkpoints": bool(save_checkpoints),
    }

    if save_checkpoints:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    print(f"  Best val loss: {best_val_loss:.6f}")
    return model, summary


def train_technical_analyst(config: TechnicalAnalystConfig, chunk_id: int, fresh: bool = False) -> Dict[str, Any]:
    config.resolve_paths()
    configure_torch_runtime(config)
    seed_everything(config.seed)

    validate_required_embeddings(config, chunk_id, ("train", "val"))

    device = resolve_device(config.device)
    label = CHUNK_CONFIG[chunk_id]["label"]

    print(f"\n{'=' * 72}")
    print(f"  TECHNICAL ANALYST — Chunk {chunk_id}")
    print(f"{'=' * 72}")
    print(f"  Device: {device}")

    model_dir = model_dir_for(config, label, "")
    if fresh and model_dir.exists():
        print(f"  Fresh run requested. Removing old model directory: {model_dir}")
        shutil.rmtree(model_dir)

    returns_df = load_returns_frame(config)
    features_df = load_features_frame(config)

    model, summary = _train_model(
        config,
        chunk_id,
        returns_df,
        features_df,
        device,
        run_tag="",
        save_checkpoints=True,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "final_model.pt")

    freezed_dir = model_dir / "model_freezed"
    freezed_dir.mkdir(parents=True, exist_ok=True)
    model.save(freezed_dir / "model.pt")

    print(f"\n  Complete. Best val loss: {summary['best_val_loss']:.6f}")
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# HPO
# ═══════════════════════════════════════════════════════════════════════════════

def _hpo_objective(
    trial: "optuna.trial.Trial",
    base_config: TechnicalAnalystConfig,
    chunk_id: int,
    returns_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> float:
    try:
        trial_config = TechnicalAnalystConfig(**base_config.to_dict()).resolve_paths()

        trial_config.lstm_hidden = trial.suggest_categorical("lstm_hidden", [32, 64, 96, 128])
        trial_config.lstm_layers = trial.suggest_categorical("lstm_layers", [1, 2])
        trial_config.dropout = trial.suggest_float("dropout", 0.05, 0.35)
        trial_config.attention_dim = trial.suggest_categorical("attention_dim", [32, 64, 128])
        trial_config.learning_rate = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        trial_config.weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)
        trial_config.batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])

        trial_config.epochs = int(base_config.hpo_epochs)
        trial_config.early_stop_patience = min(5, int(base_config.early_stop_patience))
        trial_config.max_train_samples = int(base_config.hpo_max_train_samples)
        trial_config.max_val_samples = int(base_config.hpo_max_val_samples)

        # HPO stability: no multiprocessing worker accumulation.
        trial_config.num_workers = 0
        trial_config.persistent_workers = False
        trial_config.run_tag = f"trial_{trial.number:04d}"
        trial_config.save_checkpoints = False

        device = resolve_device(trial_config.device)

        _, summary = _train_model(
            trial_config,
            chunk_id,
            returns_df,
            features_df,
            device,
            run_tag=trial_config.run_tag,
            save_checkpoints=False,
        )

        value = float(summary["best_val_loss"])

        if not np.isfinite(value):
            print(f"  Trial {trial.number} produced non-finite val_loss={value}. Returning finite HPO penalty.")
            return HPO_FAILURE_VALUE

        return value

    except Exception as exc:
        print(f"  Trial {trial.number} failed safely: {exc}")
        cleanup_memory()
        return HPO_FAILURE_VALUE


def run_hpo(config: TechnicalAnalystConfig, chunk_id: int, fresh: bool = False) -> Dict[str, Any]:
    if not HAS_OPTUNA:
        raise ImportError("optuna required: pip install optuna")

    config.resolve_paths()
    configure_torch_runtime(config)
    seed_everything(config.seed)

    validate_required_embeddings(config, chunk_id, ("train", "val"))

    label = CHUNK_CONFIG[chunk_id]["label"]

    print(f"\n{'=' * 72}")
    print(f"  TECHNICAL ANALYST HPO — {label}")
    print(f"{'=' * 72}")

    returns_df = load_returns_frame(config)
    features_df = load_features_frame(config)

    study_dir = Path(config.output_dir) / "codeResults" / "TechnicalAnalyst"
    study_dir.mkdir(parents=True, exist_ok=True)

    db_path = study_dir / "hpo.db"
    study_name = f"technical_analyst_{label}"

    if fresh and db_path.exists():
        db_path.unlink()
        print(f"  Deleted old HPO database: {db_path}")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=int(config.seed),
            n_startup_trials=min(int(config.hpo_n_startup), int(config.hpo_trials)),
        ),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
        storage=f"sqlite:///{db_path}",
        study_name=study_name,
        load_if_exists=True,
    )

    objective = lambda trial: _hpo_objective(trial, config, chunk_id, returns_df, features_df)

    study.optimize(
        objective,
        n_trials=int(config.hpo_trials),
        show_progress_bar=True,
    )

    usable_trials = [
        trial for trial in study.trials
        if trial.value is not None
        and np.isfinite(float(trial.value))
        and float(trial.value) < HPO_FAILURE_VALUE
    ]

    if not usable_trials:
        raise RuntimeError(
            "All Technical Analyst HPO trials failed or produced non-finite validation losses. "
            "No usable best params saved."
        )

    best_trial = min(usable_trials, key=lambda t: float(t.value))
    best_params = dict(best_trial.params)
    best_value = float(best_trial.value)

    if (not np.isfinite(best_value)) or best_value >= HPO_FAILURE_VALUE:
        raise RuntimeError("Best HPO result is invalid. Refusing to save best params.")

    best = {"params": best_params, "value": best_value}
    best_path = study_dir / f"best_params_{label}.json"

    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)

    print(f"\n  Best HPO: {best_params} (val_loss={best_value:.6f})")
    print(f"  Saved: {best_path}")

    return best


def apply_hpo_params_if_available(config: TechnicalAnalystConfig, chunk_id: int) -> TechnicalAnalystConfig:
    label = CHUNK_CONFIG[chunk_id]["label"]
    hpo_path = Path(config.output_dir) / "codeResults" / "TechnicalAnalyst" / f"best_params_{label}.json"

    if not hpo_path.exists():
        print("  No HPO params found. Training with current/default config.")
        return config

    with open(hpo_path) as f:
        hpo = json.load(f)

    value = float(hpo.get("value", float("inf")))
    if not np.isfinite(value) or value >= HPO_FAILURE_VALUE:
        raise RuntimeError(f"Invalid HPO file: {hpo_path}")

    params = hpo.get("params", {})

    if "lstm_hidden" in params:
        config.lstm_hidden = int(params["lstm_hidden"])
    if "lstm_layers" in params:
        config.lstm_layers = int(params["lstm_layers"])
    if "dropout" in params:
        config.dropout = float(params["dropout"])
    if "attention_dim" in params:
        config.attention_dim = int(params["attention_dim"])
    if "lr" in params:
        config.learning_rate = float(params["lr"])
    if "weight_decay" in params:
        config.weight_decay = float(params["weight_decay"])
    if "batch_size" in params:
        config.batch_size = int(params["batch_size"])

    print(f"Loaded HPO params: {params} (val_loss={value:.6f})")
    return config


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION + XAI
# ═══════════════════════════════════════════════════════════════════════════════

def extract_attention_xai(
    model: TechnicalAnalystModel,
    loader: DataLoader,
    config: TechnicalAnalystConfig,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()

    attention_sum = np.zeros(int(config.analyst_seq_len), dtype=np.float64)
    count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  XAI-L1 attention bs={loader.batch_size}", leave=False, unit="batch"):
            x = batch["x"].to(device, non_blocking=True)
            out = model(x)
            weights = out["attention_weights"].detach().cpu().numpy()
            attention_sum += weights.sum(axis=0)
            count += weights.shape[0]

    avg_attention = attention_sum / max(count, 1)
    avg_attention = avg_attention.astype(np.float32)

    top_timesteps = [
        {
            "relative_timestep": int(i - int(config.analyst_seq_len) + 1),
            "sequence_index": int(i),
            "attention": float(avg_attention[i]),
        }
        for i in np.argsort(avg_attention)[::-1][:10]
    ]

    return {
        "average_attention": avg_attention.tolist(),
        "top_timesteps": top_timesteps,
        "n_samples": int(count),
    }


def extract_gradient_xai(
    model: TechnicalAnalystModel,
    loader: DataLoader,
    config: TechnicalAnalystConfig,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()

    dim_importance_chunks = []
    time_importance_chunks = []
    n_xai = 0

    for batch in tqdm(loader, desc=f"  XAI-L2 gradients bs={loader.batch_size}", leave=False, unit="batch"):
        if n_xai >= int(config.xai_sample_size):
            break

        x = batch["x"].to(device, non_blocking=True)

        take = min(len(x), int(config.xai_sample_size) - n_xai)
        x_xai = x[:take].detach().clone().requires_grad_(True)

        # model.zero_grad(set_to_none=True)
        # out = model(x_xai)
        # objective = (
        #     out["trend_score"].mean()
        #     + out["momentum_score"].mean()
        #     + out["timing_confidence"].mean()
        # )
        # objective.backward()

        model.zero_grad(set_to_none=True)
        with cudnn_disabled_if_cuda(device):
            out = model(x_xai)
            objective = (
                out["trend_score"].mean()
                + out["momentum_score"].mean()
                + out["timing_confidence"].mean()
            )
            objective.backward()

        if x_xai.grad is not None:
            grad = x_xai.grad.detach().abs()
            dim_importance_chunks.append(grad.mean(dim=(0, 1)).cpu().numpy())
            time_importance_chunks.append(grad.mean(dim=2).mean(dim=0).cpu().numpy())

        n_xai += take

    if not dim_importance_chunks:
        return {
            "embedding_dim_importance": [],
            "timestep_importance": [],
            "n_samples": 0,
        }

    dim_mean = np.stack(dim_importance_chunks).mean(axis=0)
    time_mean = np.stack(time_importance_chunks).mean(axis=0)

    dim_total = float(dim_mean.sum())
    if not np.isfinite(dim_total) or dim_total <= 0:
        dim_total = 1.0

    time_total = float(time_mean.sum())
    if not np.isfinite(time_total) or time_total <= 0:
        time_total = 1.0

    dim_importance = [
        {
            "dim": int(i),
            "importance": float(v),
            "importance_pct": float((v / dim_total) * 100.0),
        }
        for i, v in enumerate(dim_mean)
    ]
    dim_importance.sort(key=lambda x: x["importance"], reverse=True)

    timestep_importance = [
        {
            "relative_timestep": int(i - int(config.analyst_seq_len) + 1),
            "sequence_index": int(i),
            "importance": float(v),
            "importance_pct": float((v / time_total) * 100.0),
        }
        for i, v in enumerate(time_mean)
    ]
    timestep_importance.sort(key=lambda x: x["importance"], reverse=True)

    return {
        "embedding_dim_importance": dim_importance,
        "timestep_importance": timestep_importance,
        "n_samples": int(n_xai),
    }


def generate_counterfactual_xai(
    model: TechnicalAnalystModel,
    dataset: TechnicalAnalystDataset,
    config: TechnicalAnalystConfig,
    device: torch.device,
) -> List[Dict[str, Any]]:
    model.eval()

    n = min(int(config.xai_counterfactual_scenarios), len(dataset))
    if n <= 0:
        return []

    rng = np.random.default_rng(int(config.seed))
    indices = rng.choice(len(dataset), size=n, replace=False)

    scenarios: List[Dict[str, Any]] = []

    for idx in indices:
        sample = dataset[int(idx)]
        x = sample["x"].unsqueeze(0).to(device)

        with torch.no_grad():
            original = model(x)

            # Scenario 1: remove the most recent embedding.
            no_last = x.clone()
            no_last[:, -1, :] = 0.0
            out_no_last = model(no_last)

            # Scenario 2: dampen recent 5 embedding vectors.
            damp_recent = x.clone()
            damp_recent[:, -5:, :] *= 0.8
            out_damp = model(damp_recent)

            # Scenario 3: amplify recent 5 embedding vectors.
            amp_recent = x.clone()
            amp_recent[:, -5:, :] *= 1.2
            out_amp = model(amp_recent)

        def pack(out: Dict[str, torch.Tensor]) -> Dict[str, float]:
            return {
                "trend_score": float(out["trend_score"].cpu().item()),
                "momentum_score": float(out["momentum_score"].cpu().item()),
                "timing_confidence": float(out["timing_confidence"].cpu().item()),
            }

        scenarios.append({
            "ticker": sample["ticker"],
            "date": sample["date"],
            "original": pack(original),
            "counterfactuals": [
                {
                    "condition": "Most recent embedding removed",
                    "scores": pack(out_no_last),
                },
                {
                    "condition": "Recent 5 embeddings dampened by 20%",
                    "scores": pack(out_damp),
                },
                {
                    "condition": "Recent 5 embeddings amplified by 20%",
                    "scores": pack(out_amp),
                },
            ],
        })

    return scenarios


def predict_with_xai(config: TechnicalAnalystConfig, chunk_id: int, split: str) -> Dict[str, Any]:
    config.resolve_paths()
    configure_torch_runtime(config)
    seed_everything(config.seed)

    validate_required_embeddings(config, chunk_id, (split,))

    device = resolve_device(config.device)
    label = CHUNK_CONFIG[chunk_id]["label"]

    model_dir = model_dir_for(config, label, "")
    model_path = model_dir / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"No trained Technical Analyst model found: {model_path}")

    model = TechnicalAnalystModel.load(model_path, config=config, device=str(device))
    config = model.config.resolve_paths()
    model.eval()

    stats_path = model_dir / "target_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            target_stats = json.load(f)
    else:
        target_stats = {"macd_mean": 0.0, "macd_std": 1.0}

    returns_df = load_returns_frame(config)
    features_df = load_features_frame(config)

    dataset = TechnicalAnalystDataset(
        config,
        split,
        chunk_id,
        returns_df,
        features_df,
        target_stats=target_stats,
        fit_stats=False,
        max_samples=int(config.max_predict_samples),
    )

    loader = make_loader(dataset, config, train=False)

    results_dir = Path(config.output_dir) / "results" / "TechnicalAnalyst"
    xai_dir = results_dir / "xai"
    results_dir.mkdir(parents=True, exist_ok=True)
    xai_dir.mkdir(parents=True, exist_ok=True)

    pred_chunks: List[pd.DataFrame] = []

    with torch.no_grad():
        pred_bar = tqdm(
            loader,
            desc=f"  Predict {label}_{split} bs={config.batch_size}",
            leave=False,
            unit="batch",
        )

        for batch in pred_bar:
            x = batch["x"].to(device, non_blocking=True)
            out = model(x)
            scores = out["scores"].detach().cpu().numpy()
            targets = batch["target"].cpu().numpy()

            pred_chunks.append(pd.DataFrame({
                "ticker": list(batch["ticker"]),
                "date": list(batch["date"]),
                "trend_score": scores[:, 0],
                "momentum_score": scores[:, 1],
                "timing_confidence": scores[:, 2],
                "target_trend": targets[:, 0],
                "target_momentum": targets[:, 1],
                "target_timing": targets[:, 2],
            }))

    predictions = pd.concat(pred_chunks, ignore_index=True) if pred_chunks else pd.DataFrame()

    # XAI Level 1: attention.
    xai_loader_attention = make_loader(dataset, config, train=False)
    try:
        attention_xai = extract_attention_xai(model, xai_loader_attention, config, device)
    finally:
        shutdown_dataloader(xai_loader_attention)

    # XAI Level 2: gradients.
    xai_loader_grad = make_loader(dataset, config, train=False)
    try:
        gradient_xai = extract_gradient_xai(model, xai_loader_grad, config, device)
    finally:
        shutdown_dataloader(xai_loader_grad)

    # XAI Level 3: counterfactuals.
    counterfactuals = generate_counterfactual_xai(model, dataset, config, device)

    xai = {
        "module": "TechnicalAnalyst",
        "chunk": label,
        "split": split,
        "level1_attention": attention_xai,
        "level2_gradient_importance": gradient_xai,
        "level3_counterfactuals": counterfactuals,
        "dataset_audit": dataset.audit,
        "target_stats": target_stats,
        "explanation_summary": {
            "plain_english": (
                "The Technical Analyst uses a BiLSTM over recent Temporal Encoder embeddings. "
                "Attention XAI shows which embedding timesteps mattered most. Gradient XAI shows "
                "which embedding dimensions and sequence positions most influenced the technical scores. "
                "Counterfactual XAI shows how scores change when recent technical information is removed, "
                "dampened, or amplified."
            )
        },
    }

    pred_path = results_dir / f"predictions_{label}_{split}.csv"
    attention_path = xai_dir / f"{label}_{split}_attention.json"
    dim_importance_path = xai_dir / f"{label}_{split}_embedding_dim_importance.csv"
    timestep_importance_path = xai_dir / f"{label}_{split}_timestep_importance.csv"
    counterfactual_path = xai_dir / f"{label}_{split}_counterfactuals.json"
    xai_summary_path = xai_dir / f"{label}_{split}_xai_summary.json"

    predictions.to_csv(pred_path, index=False)

    with open(attention_path, "w") as f:
        json.dump(attention_xai, f, indent=2)

    if gradient_xai["embedding_dim_importance"]:
        pd.DataFrame(gradient_xai["embedding_dim_importance"]).to_csv(dim_importance_path, index=False)

    if gradient_xai["timestep_importance"]:
        pd.DataFrame(gradient_xai["timestep_importance"]).to_csv(timestep_importance_path, index=False)

    with open(counterfactual_path, "w") as f:
        json.dump(counterfactuals, f, indent=2)

    compact_xai = {
        "module": xai["module"],
        "chunk": xai["chunk"],
        "split": xai["split"],
        "level1_attention_top_timesteps": attention_xai["top_timesteps"],
        "level2_top_embedding_dims": gradient_xai["embedding_dim_importance"][:20],
        "level2_top_timesteps": gradient_xai["timestep_importance"][:10],
        "level3_counterfactuals": counterfactuals[: min(5, len(counterfactuals))],
        "dataset_audit": dataset.audit,
        "explanation_summary": xai["explanation_summary"],
    }

    with open(xai_summary_path, "w") as f:
        json.dump(compact_xai, f, indent=2)

    print(f"  Predictions saved: {pred_path} ({len(predictions):,} rows)")
    print(f"  XAI saved: {xai_dir}")

    shutdown_dataloader(loader)
    cleanup_memory()

    return {
        "predictions": predictions,
        "xai": xai,
        "paths": {
            "predictions": str(pred_path),
            "attention": str(attention_path),
            "embedding_dim_importance": str(dim_importance_path),
            "timestep_importance": str(timestep_importance_path),
            "counterfactuals": str(counterfactual_path),
            "xai_summary": str(xai_summary_path),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# INSPECT AND SMOKE
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_inspect(config: TechnicalAnalystConfig) -> None:
    config.resolve_paths()

    print("=" * 72)
    print("TECHNICAL ANALYST — DATA INSPECTION")
    print("=" * 72)

    emb_dir = Path(config.embeddings_dir)
    if not emb_dir.exists():
        print(f"  Missing embeddings directory: {emb_dir}")
    else:
        for file in sorted(emb_dir.glob("chunk*_embeddings.npy")):
            emb = np.load(file, mmap_mode="r")
            sample = np.asarray(emb[: min(2048, len(emb))])
            print(f"  {file.name}: {emb.shape}, sample_finite={finite_ratio_np(sample):.6f}")

        for file in sorted(emb_dir.glob("chunk*_manifest.csv")):
            manifest = pd.read_csv(file)
            if len(manifest):
                print(
                    f"  {file.name}: {len(manifest):,} rows, "
                    f"{manifest['ticker'].nunique():,} tickers, "
                    f"dates {manifest['date'].min()} → {manifest['date'].max()}"
                )

    returns_df = pd.read_csv(config.returns_path, index_col=0)
    returns_values = returns_df.apply(pd.to_numeric, errors="coerce").values
    print(f"  Returns: {returns_df.shape[0]:,} days × {returns_df.shape[1]:,} tickers")
    print(f"  Returns finite ratio: {finite_ratio_np(returns_values):.6f}")

    features_path = Path(config.features_path)
    if features_path.exists():
        features_df = pd.read_csv(features_path, nrows=5)
        print(f"  features_temporal exists: {features_path}")
        print(f"  feature columns sample: {list(features_df.columns)}")
    else:
        print(f"  Missing features file: {features_path}")


def cmd_smoke(config: TechnicalAnalystConfig, device_str: str) -> None:
    config.resolve_paths()
    configure_torch_runtime(config)
    seed_everything(config.seed)

    device = resolve_device(device_str)

    print("=" * 72)
    print("TECHNICAL ANALYST — SMOKE TEST")
    print("=" * 72)

    batch = 128
    x = torch.randn(batch, int(config.analyst_seq_len), int(config.input_dim), device=device)
    target = torch.rand(batch, 3, device=device)

    model = TechnicalAnalystModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    out = model(x)
    loss = technical_loss(out, target)

    if not torch.isfinite(loss):
        raise RuntimeError("Smoke test failed: non-finite loss.")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
    optimizer.step()

    # model.eval()
    # x_xai = x[:16].detach().clone().requires_grad_(True)
    # out_xai = model(x_xai)
    # score = out_xai["trend_score"].mean() + out_xai["momentum_score"].mean() + out_xai["timing_confidence"].mean()
    # model.zero_grad(set_to_none=True)
    # score.backward()
    model.eval()
    x_xai = x[:16].detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    with cudnn_disabled_if_cuda(device):
        out_xai = model(x_xai)
        score = (
            out_xai["trend_score"].mean()
            + out_xai["momentum_score"].mean()
            + out_xai["timing_confidence"].mean()
        )
        score.backward()
    
    if x_xai.grad is None or not torch.isfinite(x_xai.grad).all():
        raise RuntimeError("Smoke test failed: invalid XAI gradient.")

    print("SMOKE TEST PASSED")
    print(f"  loss={float(loss.detach().cpu()):.6f}")
    print(f"  scores_shape={tuple(out['scores'].shape)}")
    print(f"  attention_shape={tuple(out['attention_weights'].shape)}")
    print(f"  xai_grad_shape={tuple(x_xai.grad.shape)}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-predict-samples", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")


def config_from_args(args: argparse.Namespace) -> TechnicalAnalystConfig:
    config = TechnicalAnalystConfig()

    if getattr(args, "repo_root", ""):
        config.repo_root = args.repo_root

    config.resolve_paths()

    if getattr(args, "device", None):
        config.device = args.device
    if getattr(args, "batch_size", None) is not None:
        config.batch_size = int(args.batch_size)
    if getattr(args, "epochs", None) is not None:
        config.epochs = int(args.epochs)
    if getattr(args, "num_workers", None) is not None:
        config.num_workers = int(args.num_workers)
    if getattr(args, "cpu_threads", None) is not None:
        config.cpu_threads = int(args.cpu_threads)
    if getattr(args, "max_train_samples", None) is not None:
        config.max_train_samples = int(args.max_train_samples)
    if getattr(args, "max_val_samples", None) is not None:
        config.max_val_samples = int(args.max_val_samples)
    if getattr(args, "max_predict_samples", None) is not None:
        config.max_predict_samples = int(args.max_predict_samples)
    if getattr(args, "no_amp", False):
        config.mixed_precision = False
    if getattr(args, "deterministic", False):
        config.deterministic = True
    if getattr(args, "trials", None) is not None:
        config.hpo_trials = int(args.trials)

    return config.resolve_paths()


def main() -> None:
    parser = argparse.ArgumentParser(description="Technical Analyst — BiLSTM + Attention + XAI")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("inspect", help="Inspect embeddings, manifests, returns, and features")
    add_common_args(p)

    p = sub.add_parser("smoke", help="Synthetic forward/backward/XAI smoke test")
    add_common_args(p)

    p = sub.add_parser("hpo", help="Run Optuna TPE HPO")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--fresh", action="store_true")
    add_common_args(p)

    p = sub.add_parser("train-best", help="Train using best HPO params if available")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--fresh", action="store_true")
    add_common_args(p)

    p = sub.add_parser("train-best-all", help="Train all available chunks")
    p.add_argument("--fresh", action="store_true")
    add_common_args(p)

    p = sub.add_parser("predict", help="Predict and generate XAI")
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
        cmd_smoke(config, config.device)

    elif args.command == "hpo":
        config.hpo_trials = int(args.trials)
        run_hpo(config, args.chunk, fresh=bool(args.fresh))

    elif args.command == "train-best":
        config = apply_hpo_params_if_available(config, args.chunk)
        train_technical_analyst(config, args.chunk, fresh=bool(args.fresh))

    elif args.command == "train-best-all":
        for chunk_id in [1, 2, 3]:
            validate_required_embeddings(config, chunk_id, ("train", "val"))
            chunk_config = apply_hpo_params_if_available(config, chunk_id)
            train_technical_analyst(chunk_config, chunk_id, fresh=bool(args.fresh))

    elif args.command == "predict":
        result = predict_with_xai(config, args.chunk, args.split)
        print(f"  Returned keys: {list(result.keys())}")


if __name__ == "__main__":
    main()


# ── Run instructions ─────────────────────────────────────────────────────────
# Compile:
# python -m py_compile code/analysts/technical_analyst.py
#
# Inspect available data:
# python code/analysts/technical_analyst.py inspect --repo-root .
#
# Smoke test:
# python code/analysts/technical_analyst.py smoke --repo-root . --device cuda
#
# HPO for Chunk 1:
# python code/analysts/technical_analyst.py hpo --repo-root . --chunk 1 --trials 40 --device cuda --fresh
#
# Train Chunk 1:
# python code/analysts/technical_analyst.py train-best --repo-root . --chunk 1 --device cuda --fresh
#
# Predict Chunk 1 test split with XAI:
# python code/analysts/technical_analyst.py predict --repo-root . --chunk 1 --split test --device cuda
#
# Small debug run:
# python code/analysts/technical_analyst.py hpo --repo-root . --chunk 1 --trials 3 --device cuda --fresh
# python code/analysts/technical_analyst.py train-best --repo-root . --chunk 1 --device cuda --epochs 2 --max-train-samples 50000 --max-val-samples 10000 --fresh
