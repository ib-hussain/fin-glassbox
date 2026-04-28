#!/usr/bin/env python3
"""
StemGNN Contagion Risk Module — resource-safe, resume-safe implementation.

Project: fin-glassbox — Explainable Distributed Deep Learning Framework for Financial Risk Management
File:    code/gnn/stemgnn_contagion.py

This module implements the StemGNN-based contagion risk component with:
  - chronological chunked training and validation,
  - Optuna TPE HPO before final training,
  - architecture-safe checkpoint resume,
  - HPO-safe DataLoader behaviour to avoid "Too many open files",
  - smoke tests using synthetic or real data,
  - module-level XAI outputs returned as Python objects and optionally saved to disk,
  - CLI commands for inspect, smoke, hpo, train-best, train-best-all, and predict.

Input shape used by ContagionStemGNN.forward:
    x: [batch, nodes, window]

Returned forward dict:
    contagion_logits: [batch, nodes, horizons]
    contagion_scores: [batch, nodes, horizons]
    attention:         [nodes, nodes]
    stemgnn_output:    [batch, nodes, window]

High-level integration functions return explicit `xai` dictionaries so explanations can be
passed through the integrated system rather than only saved as files.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import random
import shutil
import sys
import time
import warnings
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields as dataclass_fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:  # pragma: no cover
    HAS_OPTUNA = False

warnings.filterwarnings("ignore", category=FutureWarning)

# Local imports. This keeps the file runnable directly and importable as a module.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from stemgnn_base_model import Model as StemGNNBase


# =============================================================================
# RUNTIME / HARDWARE UTILITIES
# =============================================================================

def configure_torch_runtime(num_threads: int = 6, deterministic: bool = False) -> None:
    """Configure PyTorch for the target RTX 3090 Ti machine without harming CPU runs."""
    num_threads = max(1, int(num_threads))
    try:
        torch.set_num_threads(num_threads)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(max(1, min(2, num_threads // 2)))
    except RuntimeError:
        pass

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic

    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def raise_open_file_limit(min_soft_limit: int = 4096) -> None:
    """Best-effort increase of Linux open-file soft limit.

    HPO repeatedly creates loaders, queues, pipes and SQLite handles. This helper is not
    a substitute for proper cleanup, but it gives the process enough headroom on Linux.
    """
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        desired = min(max(int(min_soft_limit), soft), hard)
        if desired > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (desired, hard))
            print(f"  Raised open-file soft limit: {soft} -> {desired}")
    except Exception as exc:  # pragma: no cover - platform dependent
        print(f"  Open-file limit unchanged: {exc}")


def resolve_device(device: str) -> torch.device:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False on this machine.")
    return torch.device(device)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def cudnn_disabled_if_cuda(device: torch.device):
    """Disable cuDNN only for gradient-based XAI through GRU/RNN layers.

    cuDNN RNN backward requires training mode. XAI attribution should stay in
    eval mode to avoid stochastic dropout explanations, so this context forces
    PyTorch to use the non-cuDNN autograd path for the small XAI backward pass.

    Normal training still uses cuDNN.
    """
    if device.type == "cuda" and torch.backends.cudnn.enabled:
        return torch.backends.cudnn.flags(enabled=False)
    return nullcontext()

def shutdown_dataloader(loader: Optional[DataLoader]) -> None:
    """Close persistent multiprocessing workers created by a DataLoader.

    PyTorch closes workers when the iterator is destroyed, but in Optuna loops with
    exceptions this can lag behind and exhaust file descriptors. This explicit cleanup is
    intentionally defensive.
    """
    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass
        try:
            loader._iterator = None
        except Exception:
            pass


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ContagionConfig:
    """Configuration for StemGNN contagion training, inference, and XAI."""

    # Paths
    repo_root: str = ""
    returns_path: str = "data/yFinance/processed/returns_panel_wide.csv"
    output_dir: str = "outputs"

    # Architecture
    window_size: int = 30
    multi_layer: int = 13
    dropout_rate: float = 0.75
    leaky_rate: float = 0.2
    stack_cnt: int = 2

    # Contagion targets
    contagion_horizons: List[int] = field(default_factory=lambda: [5, 20, 60])
    extreme_quantile: float = 0.05
    excess_threshold_std: float = 2.0
    history_days: int = 504
    recent_days: int = 60
    min_history_days: int = 100

    # Training
    batch_size: int = 8
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    decay_rate: float = 0.5
    exponential_decay_step: int = 13
    gradient_clip: float = 1.0
    early_stop_patience: int = 20
    validate_freq: int = 1
    optimizer: str = "RMSProp"
    use_pos_weight: bool = True
    max_pos_weight: float = 50.0

    # HPO
    hpo_trials: int = 50
    hpo_n_startup: int = 10
    hpo_epochs: int = 10
    hpo_num_workers: int = 0
    hpo_max_train_windows: int = 500
    hpo_max_eval_windows: int = 150

    # System
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 6
    cpu_threads: int = 6
    amp: bool = True
    compile_model: bool = False
    deterministic: bool = False
    persistent_workers: bool = True

    # XAI
    xai_sample_size: int = 32
    xai_top_influencers: int = 10
    enable_gnnexplainer: bool = False
    gnnexplainer_epochs: int = 50
    save_xai_to_disk: bool = True

    # Data controls
    max_train_windows: int = 0
    max_eval_windows: int = 0
    ticker_limit: int = 0
    chunk_id: int = 1

    # Resume / run behaviour
    fresh_train: bool = False

    def resolve_paths(self) -> "ContagionConfig":
        if self.repo_root:
            root = Path(self.repo_root)
            if not Path(self.returns_path).is_absolute():
                self.returns_path = str(root / self.returns_path)
            if not Path(self.output_dir).is_absolute():
                self.output_dir = str(root / self.output_dir)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict_safe(cls, payload: Dict[str, Any]) -> "ContagionConfig":
        valid = {f.name for f in dataclass_fields(cls)}
        filtered = {k: v for k, v in dict(payload).items() if k in valid}
        return cls(**filtered)


CHUNK_CONFIG = {
    1: {"train": (2000, 2004), "val": (2005, 2005), "test": (2006, 2006), "label": "chunk1"},
    2: {"train": (2007, 2014), "val": (2015, 2015), "test": (2016, 2016), "label": "chunk2"},
    3: {"train": (2017, 2022), "val": (2023, 2023), "test": (2024, 2024), "label": "chunk3"},
}

ARCHITECTURE_KEYS = ("window_size", "multi_layer", "stack_cnt", "contagion_horizons")


def architecture_signature(config: ContagionConfig, num_nodes: int) -> Dict[str, Any]:
    return {
        "num_nodes": int(num_nodes),
        "window_size": int(config.window_size),
        "multi_layer": int(config.multi_layer),
        "stack_cnt": int(config.stack_cnt),
        "contagion_horizons": list(map(int, config.contagion_horizons)),
    }


def checkpoint_signature(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    cfg = ContagionConfig.from_dict_safe(checkpoint.get("config", {}))
    return architecture_signature(cfg, int(checkpoint.get("num_nodes", 0)))


def checkpoint_is_compatible(checkpoint: Dict[str, Any], config: ContagionConfig, num_nodes: int) -> Tuple[bool, str]:
    expected = architecture_signature(config, num_nodes)
    found = checkpoint_signature(checkpoint)
    mismatches = []
    for key, expected_value in expected.items():
        if found.get(key) != expected_value:
            mismatches.append(f"{key}: checkpoint={found.get(key)} current={expected_value}")
    if mismatches:
        return False, "; ".join(mismatches)
    return True, "compatible"


def archive_existing_run(out_dir: Path, reason: str) -> Optional[Path]:
    """Move stale checkpoints/metrics aside so a fresh architecture cannot append to old logs."""
    if not out_dir.exists():
        return None
    files_to_archive = [
        out_dir / "latest_model.pt",
        out_dir / "best_model.pt",
        out_dir / "training_metrics.jsonl",
        out_dir / "training_summary.json",
    ]
    if not any(p.exists() for p in files_to_archive):
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = out_dir / f"archived_{stamp}"
    backup.mkdir(parents=True, exist_ok=True)
    for p in files_to_archive:
        if p.exists():
            shutil.move(str(p), str(backup / p.name))
    with open(backup / "archive_reason.txt", "w", encoding="utf-8") as f:
        f.write(reason + "\n")
    print(f"  Archived stale run files to: {backup}")
    return backup


# =============================================================================
# DATASET
# =============================================================================

class ContagionDataset(Dataset):
    """Vectorized contagion dataset with precomputed windows and targets.

    Each sample contains all stocks for one time window.

    `x`:      [nodes, window]
    `target`: [nodes, horizons]
    """

    def __init__(
        self,
        returns_matrix: np.ndarray,
        tickers: List[str],
        config: ContagionConfig,
        start_idx: int,
        end_idx_exclusive: int,
        fit_stats: bool = True,
        norm_stats: Optional[Dict[str, np.ndarray]] = None,
        max_windows: int = 0,
        label: str = "dataset",
    ) -> None:
        if returns_matrix.ndim != 2:
            raise ValueError(f"returns_matrix must be [dates, tickers], got {returns_matrix.shape}")
        if len(tickers) != returns_matrix.shape[1]:
            raise ValueError("Ticker count does not match returns matrix columns.")

        self.config = config
        self.tickers = list(tickers)
        self.num_nodes = len(tickers)
        self.window_size = int(config.window_size)
        self.horizons = list(map(int, config.contagion_horizons))
        self.max_horizon = max(self.horizons)
        self.label = label

        returns = np.asarray(returns_matrix, dtype=np.float32)
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        self.returns = returns
        self.n_dates = returns.shape[0]

        start_idx = max(int(start_idx), self.window_size)
        end_idx_exclusive = min(int(end_idx_exclusive), self.n_dates)
        last_valid_t_exclusive = end_idx_exclusive - self.max_horizon
        candidate_t = np.arange(start_idx, last_valid_t_exclusive, dtype=np.int64)

        if candidate_t.size == 0:
            raise ValueError(
                f"No usable windows for {label}: start={start_idx}, end={end_idx_exclusive}, "
                f"window={self.window_size}, max_horizon={self.max_horizon}"
            )
        if max_windows and int(max_windows) > 0 and candidate_t.size > int(max_windows):
            candidate_t = candidate_t[: int(max_windows)]

        self.sample_t = candidate_t
        n_samples = int(candidate_t.size)
        self.windows = np.empty((n_samples, self.num_nodes, self.window_size), dtype=np.float32)
        self.targets = np.zeros((n_samples, self.num_nodes, len(self.horizons)), dtype=np.float32)

        # Cumulative sum of daily log returns. Forward h-day return at t is csum[t+h]-csum[t].
        csum = np.vstack([
            np.zeros((1, self.num_nodes), dtype=np.float32),
            np.cumsum(returns, axis=0, dtype=np.float32),
        ])
        horizon_returns = {h: (csum[h:] - csum[:-h]).astype(np.float32) for h in self.horizons}

        iterator = tqdm(candidate_t, desc=f"  Building {label}", leave=False)
        for out_idx, t in enumerate(iterator):
            self.windows[out_idx] = returns[t - self.window_size:t].T

            hist_start = max(0, int(t) - int(config.history_days))
            for h_idx, h in enumerate(self.horizons):
                hret = horizon_returns[h]
                forward_ret = hret[int(t)]

                hist_end = max(hist_start, int(t) - h + 1)
                hist = hret[hist_start:hist_end]
                if hist.shape[0] < int(config.min_history_days):
                    continue

                recent_start = max(hist_start, int(t) - int(config.recent_days) - h + 1)
                recent = hret[recent_start:hist_end]
                if recent.shape[0] < 5:
                    recent = hist[-min(hist.shape[0], int(config.recent_days)):]

                thresholds = np.quantile(hist, float(config.extreme_quantile), axis=0)
                expected = np.mean(recent, axis=0)
                sigma = np.std(recent, axis=0)
                sigma = np.where(sigma < 1e-8, 1e-8, sigma)

                below_threshold = forward_ret < thresholds
                excess_negative = (forward_ret - expected) < (-float(config.excess_threshold_std) * sigma)
                self.targets[out_idx, :, h_idx] = (below_threshold & excess_negative).astype(np.float32)

        if fit_stats:
            mean = self.windows.mean(axis=(0, 2), keepdims=True).astype(np.float32)
            std = self.windows.std(axis=(0, 2), keepdims=True).astype(np.float32)
            std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
            self.norm_stats = {"mean": mean, "std": std}
        elif norm_stats is not None:
            mean = np.asarray(norm_stats["mean"], dtype=np.float32)
            std = np.asarray(norm_stats["std"], dtype=np.float32)
            if mean.ndim == 1:
                mean = mean.reshape(1, -1, 1)
            if std.ndim == 1:
                std = std.reshape(1, -1, 1)
            std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
            self.norm_stats = {"mean": mean, "std": std}
        else:
            self.norm_stats = {
                "mean": np.zeros((1, self.num_nodes, 1), dtype=np.float32),
                "std": np.ones((1, self.num_nodes, 1), dtype=np.float32),
            }

        self.windows = (self.windows - self.norm_stats["mean"]) / self.norm_stats["std"]
        self.windows = np.ascontiguousarray(self.windows, dtype=np.float32)
        self.targets = np.ascontiguousarray(self.targets, dtype=np.float32)

        pos = self.targets.sum(axis=(0, 1))
        total = self.targets.shape[0] * self.targets.shape[1]
        neg = total - pos
        pos_weight = neg / np.maximum(pos, 1.0)
        self.pos_weight = np.clip(pos_weight, 1.0, float(config.max_pos_weight)).astype(np.float32)
        self.positive_rate = (pos / max(total, 1)).astype(np.float32)

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.windows[idx]),
            "target": torch.from_numpy(self.targets[idx]),
            "t": torch.tensor(int(self.sample_t[idx]), dtype=torch.long),
        }


# =============================================================================
# MODEL
# =============================================================================

class ContagionStemGNN(nn.Module):
    """StemGNN adapted for multi-horizon contagion classification."""

    def __init__(self, config: ContagionConfig, num_nodes: int, norm_stats: Optional[Dict[str, np.ndarray]] = None) -> None:
        super().__init__()
        self.config = copy.deepcopy(config)
        self.num_nodes = int(num_nodes)
        self.norm_stats = norm_stats

        self.stemgnn = StemGNNBase(
            units=self.num_nodes,
            stack_cnt=int(config.stack_cnt),
            time_step=int(config.window_size),
            multi_layer=int(config.multi_layer),
            horizon=int(config.window_size),
            dropout_rate=float(config.dropout_rate),
            leaky_rate=float(config.leaky_rate),
            device="cpu",
        )

        hidden = max(4, int(config.window_size) // 2)
        self.contagion_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(int(config.window_size), hidden),
                nn.LeakyReLU(0.2),
                nn.Dropout(float(config.dropout_rate)),
                nn.Linear(hidden, 1),
            )
            for _ in config.contagion_horizons
        ])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: [batch, nodes, window]
        """
        if x.dim() != 3:
            raise ValueError(f"Expected [batch, nodes, window], got {tuple(x.shape)}")
        if x.size(1) != self.num_nodes:
            raise ValueError(f"Expected nodes={self.num_nodes}, got {x.size(1)}")
        if x.size(2) != int(self.config.window_size):
            raise ValueError(f"Expected window_size={self.config.window_size}, got {x.size(2)}")

        stem_out, attention = self.stemgnn(x.permute(0, 2, 1).contiguous())
        if stem_out.dim() == 3 and stem_out.size(-1) == self.num_nodes:
            stem_features = stem_out.permute(0, 2, 1).contiguous()
        else:
            stem_features = stem_out.contiguous()

        logits = [head(stem_features).squeeze(-1) for head in self.contagion_heads]
        contagion_logits = torch.stack(logits, dim=-1)
        contagion_scores = torch.sigmoid(contagion_logits)
        return {
            "contagion_logits": contagion_logits,
            "contagion_scores": contagion_scores,
            "attention": attention,
            "stemgnn_output": stem_features,
        }

    def explain_forward(self, x: torch.Tensor, tickers: Optional[List[str]] = None, top_k: int = 10) -> Dict[str, Any]:
        """Forward pass plus a compact explanation object for integration-time use."""
        output = self.forward(x)
        explanation = build_batch_explanation(output, tickers=tickers, top_k=top_k)
        return {**output, "xai": explanation}

    @classmethod
    def load(cls, path: Union[str, Path], device: str = "cpu") -> "ContagionStemGNN":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = ContagionConfig.from_dict_safe(checkpoint.get("config", {}))
        config.device = str(device)
        norm_stats = _deserialize_norm_stats(checkpoint.get("norm_stats"))
        model = cls(config, int(checkpoint["num_nodes"]), norm_stats=norm_stats)
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state)
        model.to(device)
        return model


def _serialize_norm_stats(stats: Optional[Dict[str, np.ndarray]]) -> Optional[Dict[str, Any]]:
    if stats is None:
        return None
    return {
        "mean": np.asarray(stats["mean"], dtype=np.float32).tolist(),
        "std": np.asarray(stats["std"], dtype=np.float32).tolist(),
    }


def _deserialize_norm_stats(stats: Optional[Dict[str, Any]]) -> Optional[Dict[str, np.ndarray]]:
    if stats is None:
        return None
    return {
        "mean": np.asarray(stats["mean"], dtype=np.float32),
        "std": np.asarray(stats["std"], dtype=np.float32),
    }


def save_training_checkpoint(
    path: Union[str, Path],
    model: ContagionStemGNN,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    epoch: int = 0,
    best_val_loss: float = float("inf"),
    no_improve: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    payload = {
        "model_state_dict": raw_model.state_dict(),
        "config": raw_model.config.to_dict(),
        "num_nodes": raw_model.num_nodes,
        "norm_stats": _serialize_norm_stats(raw_model.norm_stats),
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "no_improve": int(no_improve),
        "architecture_signature": architecture_signature(raw_model.config, raw_model.num_nodes),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None and hasattr(scaler, "state_dict"):
        payload["scaler_state_dict"] = scaler.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_returns_frame(config: ContagionConfig) -> pd.DataFrame:
    fp = Path(config.returns_path)
    if not fp.exists():
        raise FileNotFoundError(f"Returns file not found: {fp}")
    df = pd.read_csv(fp, index_col=0, parse_dates=True).sort_index()
    if config.ticker_limit and int(config.ticker_limit) > 0:
        df = df.iloc[:, : int(config.ticker_limit)]
    return df.astype(np.float32)


def split_indices(dates: pd.DatetimeIndex, year_pair: Tuple[int, int]) -> Tuple[int, int]:
    mask = (dates.year >= int(year_pair[0])) & (dates.year <= int(year_pair[1]))
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise ValueError(f"No dates found for years {year_pair}")
    return int(idx[0]), int(idx[-1] + 1)


def make_loader(dataset: Dataset, config: ContagionConfig, train: bool, override_workers: Optional[int] = None) -> DataLoader:
    workers = config.num_workers if override_workers is None else int(override_workers)
    workers = max(0, int(workers))
    use_cuda = str(config.device).startswith("cuda") and torch.cuda.is_available()
    kwargs = {
        "batch_size": int(config.batch_size),
        "shuffle": bool(train),
        "drop_last": bool(train and len(dataset) >= int(config.batch_size)),
        "num_workers": workers,
        "pin_memory": bool(use_cuda),
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 4
        kwargs["persistent_workers"] = bool(config.persistent_workers)
    return DataLoader(dataset, **kwargs)


def build_train_val_datasets(
    config: ContagionConfig,
    chunk_id: int,
    df: pd.DataFrame,
    norm_stats_override: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[ContagionDataset, ContagionDataset, List[str]]:
    chunk_cfg = CHUNK_CONFIG[int(chunk_id)]
    returns = df.values.astype(np.float32)
    tickers = list(df.columns)
    train_start, train_end = split_indices(df.index, chunk_cfg["train"])
    val_start, val_end = split_indices(df.index, chunk_cfg["val"])

    train_ds = ContagionDataset(
        returns, tickers, config,
        start_idx=train_start,
        end_idx_exclusive=train_end,
        fit_stats=norm_stats_override is None,
        norm_stats=norm_stats_override,
        max_windows=int(config.max_train_windows),
        label=f"{chunk_cfg['label']}_train",
    )
    val_ds = ContagionDataset(
        returns, tickers, config,
        start_idx=val_start,
        end_idx_exclusive=val_end,
        fit_stats=False,
        norm_stats=train_ds.norm_stats,
        max_windows=int(config.max_eval_windows),
        label=f"{chunk_cfg['label']}_val",
    )
    return train_ds, val_ds, tickers


def build_split_dataset(
    config: ContagionConfig,
    chunk_id: int,
    split: str,
    df: pd.DataFrame,
    norm_stats: Dict[str, np.ndarray],
) -> Tuple[ContagionDataset, List[str]]:
    chunk_cfg = CHUNK_CONFIG[int(chunk_id)]
    start, end = split_indices(df.index, chunk_cfg[split])
    tickers = list(df.columns)
    ds = ContagionDataset(
        df.values.astype(np.float32), tickers, config,
        start_idx=start,
        end_idx_exclusive=end,
        fit_stats=False,
        norm_stats=norm_stats,
        max_windows=int(config.max_eval_windows),
        label=f"{chunk_cfg['label']}_{split}",
    )
    return ds, tickers


# =============================================================================
# TRAINING
# =============================================================================

def make_loss_fn(train_ds: ContagionDataset, config: ContagionConfig, device: torch.device) -> nn.Module:
    if bool(config.use_pos_weight):
        pos_weight = torch.tensor(train_ds.pos_weight, dtype=torch.float32, device=device).view(1, 1, -1)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return nn.BCEWithLogitsLoss()


def build_optimizer(model: nn.Module, config: ContagionConfig) -> torch.optim.Optimizer:
    if config.optimizer.lower() == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=float(config.learning_rate), eps=1e-8, weight_decay=float(config.weight_decay))
    return torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate), weight_decay=max(float(config.weight_decay), 1e-5))


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    config: ContagionConfig,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    use_amp = scaler is not None and device.type == "cuda" and bool(config.amp)

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(x)
                loss = loss_fn(output["contagion_logits"], target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.gradient_clip))
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(x)
            loss = loss_fn(output["contagion_logits"], target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.gradient_clip))
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    pred_sum = None
    target_sum = None
    count = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True).float()
        output = model(x)
        logits = output["contagion_logits"]
        probs = torch.sigmoid(logits)
        loss = loss_fn(logits, target)

        total_loss += float(loss.detach().cpu())
        n_batches += 1

        batch_pred = probs.sum(dim=(0, 1)).detach().cpu()
        batch_target = target.sum(dim=(0, 1)).detach().cpu()
        pred_sum = batch_pred if pred_sum is None else pred_sum + batch_pred
        target_sum = batch_target if target_sum is None else target_sum + batch_target
        count += int(probs.shape[0] * probs.shape[1])

    metrics = {"loss": total_loss / max(n_batches, 1)}
    if count > 0 and pred_sum is not None and target_sum is not None:
        pred_mean = (pred_sum / count).numpy()
        target_mean = (target_sum / count).numpy()
        for i, (p, t) in enumerate(zip(pred_mean, target_mean)):
            metrics[f"pred_mean_h{i}"] = float(p)
            metrics[f"target_rate_h{i}"] = float(t)
    return metrics


def train_contagion_model(
    config: ContagionConfig,
    chunk_id: int,
    df: pd.DataFrame,
    resume: bool = True,
    save_checkpoints: bool = True,
    run_tag: Optional[str] = None,
) -> Tuple[ContagionStemGNN, float, Dict[str, Any]]:
    """Train the contagion model.

    Returns:
        (model, best_val_loss, training_summary)
    """
    chunk_id = int(chunk_id)
    chunk_cfg = CHUNK_CONFIG[chunk_id]
    label = chunk_cfg["label"] if run_tag is None else str(run_tag)
    device = resolve_device(config.device)
    seed_everything(config.seed)

    tickers_probe = list(df.columns)
    num_nodes = len(tickers_probe)
    out_dir = Path(config.output_dir) / "models" / "StemGNN" / label
    latest_path = out_dir / "latest_model.pt"
    best_path = out_dir / "best_model.pt"
    metrics_path = out_dir / "training_metrics.jsonl"
    summary_path = out_dir / "training_summary.json"

    checkpoint = None
    norm_stats_override = None
    start_epoch = 0
    best_val_loss = float("inf")
    no_improve = 0
    allow_resume = bool(resume and save_checkpoints and latest_path.exists() and not config.fresh_train)

    if allow_resume:
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        compatible, reason = checkpoint_is_compatible(checkpoint, config, num_nodes)
        if compatible:
            print(f"  Resume checkpoint compatible: {latest_path}")
            start_epoch = int(checkpoint.get("epoch", 0))
            best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
            no_improve = int(checkpoint.get("no_improve", 0))
            norm_stats_override = _deserialize_norm_stats(checkpoint.get("norm_stats"))
        else:
            print("  Resume checkpoint is incompatible with current architecture; starting fresh.")
            print(f"  Reason: {reason}")
            archive_existing_run(out_dir, f"Architecture mismatch: {reason}")
            checkpoint = None
    elif save_checkpoints and config.fresh_train:
        archive_existing_run(out_dir, "User requested --fresh training.")

    train_loader = None
    val_loader = None
    try:
        train_ds, val_ds, tickers = build_train_val_datasets(config, chunk_id, df, norm_stats_override=norm_stats_override)
        train_loader = make_loader(train_ds, config, train=True)
        val_loader = make_loader(val_ds, config, train=False)

        if checkpoint is not None:
            model = ContagionStemGNN(config, len(tickers), norm_stats=train_ds.norm_stats).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model = ContagionStemGNN(config, len(tickers), norm_stats=train_ds.norm_stats).to(device)
            start_epoch = 0
            best_val_loss = float("inf")
            no_improve = 0

        if bool(config.compile_model) and hasattr(torch, "compile"):
            model = torch.compile(model)

        optimizer = build_optimizer(model, config)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(config.decay_rate))
        scaler = torch.cuda.amp.GradScaler(enabled=(bool(config.amp) and device.type == "cuda"))

        if checkpoint is not None:
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if "scaler_state_dict" in checkpoint and bool(config.amp) and device.type == "cuda":
                try:
                    scaler.load_state_dict(checkpoint["scaler_state_dict"])
                except Exception:
                    pass
            print(f"  Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")

        loss_fn = make_loss_fn(train_ds, config, device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model params: {total_params:,}")
        print(f"  Architecture: window={config.window_size}, multi_layer={config.multi_layer}, stack={config.stack_cnt}, horizons={config.contagion_horizons}")
        print(f"  Train windows: {len(train_ds):,} | Val: {len(val_ds):,} | Tickers: {len(tickers):,}")
        print(f"  Positive rates: {train_ds.positive_rate.tolist()} | pos_weight={train_ds.pos_weight.tolist()}")
        print(f"  Starting from epoch {start_epoch + 1}/{config.epochs}")

        if save_checkpoints:
            out_dir.mkdir(parents=True, exist_ok=True)

        training_start = time.time()
        last_epoch = start_epoch
        for epoch in range(start_epoch + 1, int(config.epochs) + 1):
            epoch_start = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, config, scaler)
            if epoch % int(config.validate_freq) == 0:
                val_metrics = validate_epoch(model, val_loader, loss_fn, device)
                val_loss = float(val_metrics["loss"])
            else:
                val_metrics = {"loss": float("nan")}
                val_loss = float("nan")

            row = {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "seconds": round(time.time() - epoch_start, 2),
                "architecture": architecture_signature(config, len(tickers)),
                **val_metrics,
            }

            print(
                f"  [{label}] E{epoch:03d}/{config.epochs} | "
                f"train={train_loss:.5f} | val={val_loss:.5f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} | {row['seconds']:.1f}s"
            )

            improved = math.isfinite(val_loss) and val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                no_improve = 0
            else:
                no_improve += 1

            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            if save_checkpoints:
                with open(metrics_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                save_training_checkpoint(latest_path, raw_model, optimizer, scheduler, scaler, epoch, best_val_loss, no_improve)
                if improved:
                    save_training_checkpoint(best_path, raw_model, None, None, None, epoch, best_val_loss, no_improve)

            if epoch % int(config.exponential_decay_step) == 0:
                scheduler.step()

            last_epoch = epoch
            if no_improve >= int(config.early_stop_patience):
                print(f"  Early stopping at epoch {epoch}; no_improve={no_improve}.")
                break

        elapsed_min = (time.time() - training_start) / 60.0
        print(f"  Best val loss: {best_val_loss:.6f} | Total: {elapsed_min:.2f} min")

        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        if save_checkpoints and best_path.exists():
            best_model = ContagionStemGNN.load(best_path, device=str(device))
        else:
            best_model = raw_model

        summary = {
            "chunk": CHUNK_CONFIG[chunk_id]["label"],
            "run_tag": label,
            "best_val_loss": float(best_val_loss),
            "epochs_trained": int(last_epoch),
            "total_params": int(total_params),
            "elapsed_min": round(elapsed_min, 3),
            "tickers": len(tickers),
            "architecture": architecture_signature(config, len(tickers)),
            "saved_checkpoints": bool(save_checkpoints),
        }
        if save_checkpoints:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        return best_model, best_val_loss, summary
    finally:
        shutdown_dataloader(train_loader)
        shutdown_dataloader(val_loader)
        cleanup_cuda()


# =============================================================================
# XAI
# =============================================================================

def build_batch_explanation(output: Dict[str, torch.Tensor], tickers: Optional[List[str]] = None, top_k: int = 10) -> Dict[str, Any]:
    """Build a compact explanation from a forward output.

    This is intentionally lightweight so it can be used inside an integrated daily
    inference system without writing files.
    """
    attention = output["attention"].detach().float().cpu().numpy()
    scores = output["contagion_scores"].detach().float().cpu().numpy()
    n = attention.shape[0]
    if tickers is None:
        tickers = [f"node_{i}" for i in range(n)]
    top_k = min(int(top_k), max(1, n - 1))
    top_influencers = {}
    for i, ticker in enumerate(tickers):
        row = attention[i].copy()
        row[i] = -np.inf
        idx = np.argsort(row)[-top_k:][::-1]
        top_influencers[ticker] = [
            {"ticker": tickers[j], "weight": float(row[j])}
            for j in idx if np.isfinite(row[j])
        ]
    return {
        "type": "stemgnn_contagion_batch_explanation",
        "attention_shape": list(attention.shape),
        "top_influencers": top_influencers,
        "batch_mean_contagion": scores.mean(axis=0).tolist(),
    }


@torch.no_grad()
def _online_mean_update(current: Optional[torch.Tensor], count: int, new_value: torch.Tensor) -> Tuple[torch.Tensor, int]:
    if current is None:
        return new_value.detach().cpu().float(), 1
    count_new = count + 1
    current += (new_value.detach().cpu().float() - current) / count_new
    return current, count_new


def extract_xai_level1(model: ContagionStemGNN, dataloader: DataLoader, config: ContagionConfig, tickers: List[str], device: torch.device) -> Dict[str, Any]:
    """Level 1 XAI: learned adjacency + top influencers + average split contagion."""
    model.eval()
    attention_mean = None
    attention_count = 0
    score_sum = None
    score_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  XAI-L1", leave=False):
            x = batch["x"].to(device, non_blocking=True)
            output = model(x)
            attention_mean, attention_count = _online_mean_update(attention_mean, attention_count, output["attention"])
            scores = output["contagion_scores"].detach().cpu().float().sum(dim=0)
            score_sum = scores if score_sum is None else score_sum + scores
            score_count += int(x.size(0))

    avg_attention = attention_mean.numpy().astype(np.float32)
    avg_contagion = (score_sum / max(score_count, 1)).numpy().astype(np.float32)
    top_influencers = {}
    k = min(int(config.xai_top_influencers), max(1, len(tickers) - 1))
    for i, ticker in enumerate(tickers):
        row = avg_attention[i].copy()
        row[i] = -np.inf
        top_idx = np.argsort(row)[-k:][::-1]
        top_influencers[ticker] = [
            {"ticker": tickers[j], "weight": float(row[j])}
            for j in top_idx if np.isfinite(row[j])
        ]
    return {"adjacency": avg_attention, "top_influencers": top_influencers, "avg_contagion": avg_contagion}


def extract_xai_level2(
    model: ContagionStemGNN,
    dataset: ContagionDataset,
    config: ContagionConfig,
    tickers: List[str],
    device: torch.device,
) -> Dict[str, Any]:
    """Level 2 XAI: gradient-based node/time importance and edge importance approximation.

    StemGNN contains a GRU in the latent-correlation layer. On CUDA, cuDNN does
    not allow backward through an RNN if the forward pass was executed in eval
    mode. For XAI we still want eval-mode behaviour because dropout should not
    make explanations stochastic.

    The fix is to keep the model in eval mode, but disable cuDNN only during
    the small XAI attribution forward/backward pass. Normal training remains
    cuDNN-accelerated.
    """
    was_training = model.training
    model.eval()

    n_samples = min(int(config.xai_sample_size), len(dataset))
    node_importance = np.zeros((len(tickers), int(config.window_size)), dtype=np.float32)
    processed = 0

    try:
        for i in tqdm(range(n_samples), desc="  XAI-L2", leave=False):
            sample = dataset[i]
            x = sample["x"].unsqueeze(0).to(device)
            x.requires_grad_(True)

            model.zero_grad(set_to_none=True)

            with cudnn_disabled_if_cuda(device):
                output = model(x)
                objective = output["contagion_logits"].sum()
                objective.backward()

            if x.grad is None:
                raise RuntimeError("XAI Level 2 failed: input gradient is None.")

            grad = x.grad.detach().abs().cpu().numpy().squeeze(0)
            node_importance += grad.astype(np.float32)
            processed += 1

            del x, output, objective

    finally:
        model.train(was_training)

    node_importance /= max(processed, 1)

    l1_loader = make_loader(dataset, config, train=False, override_workers=0)
    try:
        l1 = extract_xai_level1(model, l1_loader, config, tickers, device)
    finally:
        shutdown_dataloader(l1_loader)

    edge_importance = l1["adjacency"] * node_importance.mean(axis=1)[:, None]

    return {
        "node_temporal_importance": node_importance,
        "edge_importance": edge_importance.astype(np.float32),
    }


# def extract_xai_level3_gnnexplainer(model: ContagionStemGNN, dataset: ContagionDataset, config: ContagionConfig, tickers: List[str], device: torch.device) -> Dict[str, Any]:
#     """Level 3 XAI: opt-in approximate GNNExplainer-style edge mask.

#     This is intentionally not used by default because full 2,500x2,500 masks are expensive.
#     The returned JSON metadata contains top edges only; raw masks are returned as numpy arrays
#     and can be saved as `.npy` if needed.
#     """
#     model.eval()
#     n_explain = min(1, len(dataset))
#     results = []
#     masks = []
#     n_nodes = len(tickers)

#     for idx in range(n_explain):
#         sample = dataset[idx]
#         x_orig = sample["x"].unsqueeze(0).to(device)
#         with torch.no_grad():
#             ref_score = model(x_orig)["contagion_scores"].mean().detach()

#         mask = nn.Parameter(torch.zeros(n_nodes, n_nodes, device=device))
#         optimizer = torch.optim.Adam([mask], lr=0.05)

#         for _ in tqdm(range(int(config.gnnexplainer_epochs)), desc="  XAI-L3", leave=False):
#             edge_mask = torch.sigmoid(mask)
#             node_gate = edge_mask.mean(dim=0).view(1, n_nodes, 1)
#             masked_x = x_orig * node_gate
#             pred_score = model(masked_x)["contagion_scores"].mean()
#             loss = F.mse_loss(pred_score, ref_score) + 0.01 * edge_mask.mean()
#             optimizer.zero_grad(set_to_none=True)
#             loss.backward()
#             optimizer.step()

#         final_mask = torch.sigmoid(mask).detach().cpu().numpy().astype(np.float32)
#         masks.append(final_mask)
#         flat = final_mask.reshape(-1)
#         top_count = min(50, flat.size)
#         top_flat_idx = np.argpartition(flat, -top_count)[-top_count:]
#         edges = []
#         for flat_i in top_flat_idx:
#             src = int(flat_i // n_nodes)
#             dst = int(flat_i % n_nodes)
#             if src == dst:
#                 continue
#             edges.append({"source": tickers[src], "target": tickers[dst], "importance": float(final_mask[src, dst])})
#         edges.sort(key=lambda x: x["importance"], reverse=True)
#         results.append({"sample_idx": int(idx), "important_edges": edges[:50], "num_returned_edges": len(edges[:50])})

#     return {"gnnexplainer_results": results, "edge_masks": masks}
def extract_xai_level3_gnnexplainer(
    model: ContagionStemGNN,
    dataset: ContagionDataset,
    config: ContagionConfig,
    tickers: List[str],
    device: torch.device,
) -> Dict[str, Any]:
    """Level 3 XAI: opt-in approximate GNNExplainer-style edge mask.

    This is intentionally not used by default because full 2,500x2,500 masks
    are expensive. The returned JSON metadata contains top edges only; raw
    masks are returned as numpy arrays and can be saved as `.npy` if needed.

    This function also disables cuDNN only during the attribution backward pass
    because StemGNN contains a GRU.
    """
    n_explain = min(1, len(dataset))
    results = []
    masks = []
    n_nodes = len(tickers)

    was_training = model.training
    model.eval()

    try:
        for idx in range(n_explain):
            sample = dataset[idx]
            x_orig = sample["x"].unsqueeze(0).to(device)

            with torch.no_grad():
                ref_score = model(x_orig)["contagion_scores"].mean().detach()

            mask = nn.Parameter(torch.zeros(n_nodes, n_nodes, device=device))
            optimizer = torch.optim.Adam([mask], lr=0.05)

            for _ in tqdm(range(int(config.gnnexplainer_epochs)), desc="  XAI-L3", leave=False):
                edge_mask = torch.sigmoid(mask)
                node_gate = edge_mask.mean(dim=0).view(1, n_nodes, 1)
                masked_x = x_orig * node_gate

                optimizer.zero_grad(set_to_none=True)

                with cudnn_disabled_if_cuda(device):
                    pred_score = model(masked_x)["contagion_scores"].mean()
                    loss = F.mse_loss(pred_score, ref_score) + 0.01 * edge_mask.mean()
                    loss.backward()

                optimizer.step()

            final_mask = torch.sigmoid(mask).detach().cpu().numpy().astype(np.float32)
            masks.append(final_mask)

            flat = final_mask.reshape(-1)
            top_count = min(50, flat.size)
            top_flat_idx = np.argpartition(flat, -top_count)[-top_count:]

            edges = []
            for flat_i in top_flat_idx:
                src = int(flat_i // n_nodes)
                dst = int(flat_i % n_nodes)
                if src == dst:
                    continue
                edges.append({
                    "source": tickers[src],
                    "target": tickers[dst],
                    "importance": float(final_mask[src, dst]),
                })

            edges.sort(key=lambda x: x["importance"], reverse=True)

            results.append({
                "sample_idx": int(idx),
                "important_edges": edges[:50],
                "num_returned_edges": len(edges[:50]),
            })

    finally:
        model.train(was_training)

    return {
        "gnnexplainer_results": results,
        "edge_masks": masks,
    }

def save_xai_outputs(xai_dir: Path, split_label: str, xai: Dict[str, Any]) -> None:
    xai_dir.mkdir(parents=True, exist_ok=True)
    if "level1" in xai:
        np.save(xai_dir / f"{split_label}_adjacency.npy", xai["level1"]["adjacency"])
        np.save(xai_dir / f"{split_label}_avg_contagion.npy", xai["level1"]["avg_contagion"])
        with open(xai_dir / f"{split_label}_top_influencers.json", "w", encoding="utf-8") as f:
            json.dump(xai["level1"]["top_influencers"], f, indent=2)
    if "level2" in xai:
        np.save(xai_dir / f"{split_label}_node_temporal_importance.npy", xai["level2"]["node_temporal_importance"])
        np.save(xai_dir / f"{split_label}_edge_importance.npy", xai["level2"]["edge_importance"])
    if "level3" in xai and xai["level3"]:
        level3 = xai["level3"]
        for i, mask in enumerate(level3.get("edge_masks", [])):
            np.save(xai_dir / f"{split_label}_gnnexplainer_mask_{i}.npy", mask)
        metadata = level3.get("gnnexplainer_results", [])
        with open(xai_dir / f"{split_label}_gnnexplainer_top_edges.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


def run_full_xai(model: ContagionStemGNN, dataset: ContagionDataset, config: ContagionConfig, tickers: List[str], device: torch.device, split_label: str) -> Dict[str, Any]:
    """Run XAI and return explanations. Level 1 and 2 are always run; Level 3 is opt-in."""
    xai_loader = make_loader(dataset, config, train=False, override_workers=min(int(config.num_workers), 2))
    try:
        print("\n" + "=" * 60 + "\n  XAI Level 1: Adjacency + Top Influencers\n" + "=" * 60)
        level1 = extract_xai_level1(model, xai_loader, config, tickers, device)
    finally:
        shutdown_dataloader(xai_loader)

    print("\n" + "=" * 60 + "\n  XAI Level 2: Gradient Node/Edge Importance\n" + "=" * 60)
    level2 = extract_xai_level2(model, dataset, config, tickers, device)

    xai = {"level1": level1, "level2": level2}
    if bool(config.enable_gnnexplainer):
        print("\n" + "=" * 60 + "\n  XAI Level 3: GNNExplainer Approximation\n" + "=" * 60)
        xai["level3"] = extract_xai_level3_gnnexplainer(model, dataset, config, tickers, device)

    if bool(config.save_xai_to_disk):
        xai_dir = Path(config.output_dir) / "results" / "StemGNN" / "xai"
        save_xai_outputs(xai_dir, split_label, xai)
        print(f"  XAI saved to: {xai_dir}")
    return xai


# =============================================================================
# PREDICTION
# =============================================================================

@torch.no_grad()
def generate_contagion_predictions(model: ContagionStemGNN, dataset: ContagionDataset, config: ContagionConfig, tickers: List[str], device: torch.device, split_label: str) -> pd.DataFrame:
    """Generate average per-stock contagion scores for a split."""
    model.eval()
    loader = make_loader(dataset, config, train=False, override_workers=min(int(config.num_workers), 2))
    all_scores = []
    try:
        for batch in tqdm(loader, desc=f"  Predict {split_label}", leave=False):
            x = batch["x"].to(device, non_blocking=True)
            output = model(x)
            all_scores.append(output["contagion_scores"].detach().cpu().numpy())
    finally:
        shutdown_dataloader(loader)

    scores = np.concatenate(all_scores, axis=0)
    avg_scores = scores.mean(axis=0)
    df = pd.DataFrame(avg_scores, index=tickers, columns=[f"contagion_{h}d" for h in config.contagion_horizons])
    df.index.name = "ticker"
    return df


# =============================================================================
# HPO
# =============================================================================

def _run_hpo_objective(trial: Any, base_config: ContagionConfig, chunk_id: int, df: pd.DataFrame) -> float:
    trial_config = ContagionConfig.from_dict_safe(base_config.to_dict())
    trial_config.window_size = trial.suggest_categorical("window_size", [15, 30, 60])
    trial_config.multi_layer = trial.suggest_categorical("multi_layer", [5, 8, 13])
    trial_config.dropout_rate = trial.suggest_categorical("dropout_rate", [0.5, 0.6, 0.75])
    trial_config.batch_size = trial.suggest_categorical("batch_size", [4, 8])
    trial_config.learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    trial_config.decay_rate = trial.suggest_categorical("decay_rate", [0.5, 0.7, 0.9])
    trial_config.exponential_decay_step = trial.suggest_categorical("exponential_decay_step", [5, 8, 13])
    trial_config.optimizer = trial.suggest_categorical("optimizer", ["RMSProp", "AdamW"])

    # HPO must not leave worker processes or chunk-level training checkpoints behind.
    trial_config.epochs = int(base_config.hpo_epochs)
    trial_config.max_train_windows = int(base_config.max_train_windows or base_config.hpo_max_train_windows)
    trial_config.max_eval_windows = int(base_config.max_eval_windows or base_config.hpo_max_eval_windows)
    trial_config.num_workers = int(base_config.hpo_num_workers)
    trial_config.persistent_workers = False
    trial_config.compile_model = False
    trial_config.enable_gnnexplainer = False
    trial_config.save_xai_to_disk = False
    trial_config.early_stop_patience = min(5, int(base_config.early_stop_patience))

    model = None
    try:
        model, val_loss, _ = train_contagion_model(
            trial_config,
            chunk_id,
            df,
            resume=False,
            save_checkpoints=False,
            run_tag=f"hpo_{CHUNK_CONFIG[int(chunk_id)]['label']}_trial_{trial.number}",
        )
        return float(val_loss)
    finally:
        del model
        cleanup_cuda()


def run_hpo(config: ContagionConfig, chunk_id: int, n_trials: int, fresh: bool = False) -> Dict[str, Any]:
    """Run Optuna TPE HPO. Returns and saves the best params."""
    if not HAS_OPTUNA:
        raise ImportError("Optuna is required for HPO. Install with: pip install optuna")
    raise_open_file_limit()
    df = load_returns_frame(config)
    label = CHUNK_CONFIG[int(chunk_id)]["label"]
    storage_dir = Path(config.output_dir) / "codeResults" / "StemGNN"
    storage_dir.mkdir(parents=True, exist_ok=True)
    db_path = storage_dir / "hpo.db"
    study_name = f"stemgnn_contagion_{label}"

    if fresh and db_path.exists():
        db_path.unlink()
        print("  Deleted existing HPO database for fresh start.")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=int(config.seed), n_startup_trials=int(config.hpo_n_startup)),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        storage=f"sqlite:///{db_path}",
        study_name=study_name,
        load_if_exists=True,
    )
    study.optimize(lambda trial: _run_hpo_objective(trial, config, chunk_id, df), n_trials=int(n_trials), show_progress_bar=True)

    result = {"params": study.best_params, "value": float(study.best_value), "study_name": study_name}
    params_path = storage_dir / f"best_params_{label}.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Best params ({label}): {study.best_params}")
    print(f"  Best val loss: {study.best_value:.6f}")
    print(f"  Saved to: {params_path}")
    return result


# =============================================================================
# HIGH-LEVEL INTEGRATION API
# =============================================================================

def apply_hpo_params_if_available(config: ContagionConfig, chunk_id: int) -> bool:
    label = CHUNK_CONFIG[int(chunk_id)]["label"]
    hpo_path = Path(config.output_dir) / "codeResults" / "StemGNN" / f"best_params_{label}.json"
    if not hpo_path.exists():
        return False
    with open(hpo_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    params = payload.get("params", {})
    for key, value in params.items():
        if hasattr(config, key):
            setattr(config, key, value)
    print(f"Loaded HPO params from {hpo_path}: {params}")
    return True


def train_and_predict(config: Union[ContagionConfig, Dict[str, Any]], chunk_id: int = 1, load_hpo: bool = True, run_xai: bool = True) -> Dict[str, Any]:
    """Train -> predict test split -> optional XAI. This is the main integration entrypoint."""
    if isinstance(config, dict):
        config = ContagionConfig.from_dict_safe(config)
    config.resolve_paths()
    configure_torch_runtime(config.cpu_threads, config.deterministic)
    device = resolve_device(config.device)
    if load_hpo:
        apply_hpo_params_if_available(config, chunk_id)
    df = load_returns_frame(config)
    model, best_val_loss, summary = train_contagion_model(config, chunk_id, df, resume=True, save_checkpoints=True)
    test_ds, tickers = build_split_dataset(config, chunk_id, "test", df, model.norm_stats)
    label = CHUNK_CONFIG[int(chunk_id)]["label"]
    predictions = generate_contagion_predictions(model, test_ds, config, tickers, device, f"{label}_test")
    pred_path = Path(config.output_dir) / "results" / "StemGNN" / f"contagion_scores_{label}_test.csv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(pred_path, index_label="ticker")
    xai = run_full_xai(model, test_ds, config, tickers, device, f"{label}_test") if run_xai else {}
    return {"model": model, "predictions": predictions, "xai": xai, "summary": summary, "best_val_loss": best_val_loss}


def load_and_predict(model_path: Union[str, Path], split: str = "test", chunk_id: int = 1, device: str = "cuda", run_xai: bool = True) -> Dict[str, Any]:
    """Load a trained checkpoint and run prediction plus optional XAI."""
    device_obj = resolve_device(device)
    model = ContagionStemGNN.load(model_path, device=str(device_obj))
    config = model.config.resolve_paths()
    config.device = str(device_obj)
    configure_torch_runtime(config.cpu_threads, config.deterministic)
    df = load_returns_frame(config)
    ds, tickers = build_split_dataset(config, chunk_id, split, df, model.norm_stats)
    label = CHUNK_CONFIG[int(chunk_id)]["label"]
    predictions = generate_contagion_predictions(model, ds, config, tickers, device_obj, f"{label}_{split}")
    xai = run_full_xai(model, ds, config, tickers, device_obj, f"{label}_{split}") if run_xai else {}
    return {"model": model, "predictions": predictions, "xai": xai}


# =============================================================================
# SMOKE TEST DATA
# =============================================================================

def make_synthetic_returns_frame(n_tickers: int = 32, start: str = "2000-01-03", end: str = "2006-12-29", seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    t = len(dates)
    n = int(n_tickers)
    market = rng.normal(0.0002, 0.01, size=(t, 1)).astype(np.float32)
    sectors = rng.normal(0.0, 0.006, size=(t, 4)).astype(np.float32)
    loadings = rng.normal(0.0, 0.35, size=(4, n)).astype(np.float32)
    idio = rng.normal(0.0, 0.012, size=(t, n)).astype(np.float32)
    returns = 0.7 * market + sectors @ loadings + idio
    # Add a few market-wide shock windows so contagion labels are not always empty.
    for shock_start in [350, 900, 1400]:
        if shock_start + 5 < t:
            returns[shock_start:shock_start + 5] += rng.normal(-0.035, 0.01, size=(5, n)).astype(np.float32)
    cols = [f"TK{i:03d}" for i in range(n)]
    return pd.DataFrame(returns.astype(np.float32), index=dates, columns=cols)


# =============================================================================
# CLI COMMANDS
# =============================================================================

def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", type=str, default="")
    parser.add_argument("--returns-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-eval-windows", type=int, default=None)
    parser.add_argument("--ticker-limit", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--multi-layer", type=int, default=None)
    parser.add_argument("--dropout-rate", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--amp", action="store_true", default=None)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--enable-gnnexplainer", action="store_true", default=True)
    parser.add_argument("--disable-xai-save", action="store_true")


def config_from_args(args: argparse.Namespace) -> ContagionConfig:
    config = ContagionConfig()
    direct_fields = [
        "repo_root", "returns_path", "output_dir", "device", "batch_size", "epochs", "num_workers",
        "cpu_threads", "max_train_windows", "max_eval_windows", "ticker_limit", "window_size",
        "multi_layer", "dropout_rate", "learning_rate",
    ]
    for attr in direct_fields:
        value = getattr(args, attr, None)
        if value is not None and (not isinstance(value, str) or value != ""):
            setattr(config, attr, value)
    if getattr(args, "amp", None) is not None:
        config.amp = bool(args.amp)
    if getattr(args, "compile_model", False):
        config.compile_model = True
    if getattr(args, "deterministic", False):
        config.deterministic = True
    if getattr(args, "enable_gnnexplainer", None) is not None:
        config.enable_gnnexplainer = bool(args.enable_gnnexplainer)
    if getattr(args, "disable_xai_save", False):
        config.save_xai_to_disk = False
    if getattr(args, "fresh", False):
        config.fresh_train = True
    if getattr(args, "chunk", None) is not None:
        config.chunk_id = int(args.chunk)
    config.resolve_paths()
    return config


def cmd_inspect(config: ContagionConfig) -> None:
    print("=" * 72)
    print("STEMGNN CONTAGION — SYSTEM INSPECTION")
    print("=" * 72)
    print(f"Device: {config.device} | Workers: {config.num_workers} | Threads: {config.cpu_threads}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name} | VRAM: {p.total_memory / 1024 ** 3:.1f} GB")
    df = load_returns_frame(config)
    print(f"\nReturns: {df.shape[0]:,} days × {df.shape[1]:,} tickers")
    print(f"Range: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"NaN rate: {df.isna().mean().mean() * 100:.6f}%")
    print("\nSplits:")
    for cid, cfg in CHUNK_CONFIG.items():
        for split, years in [("train", cfg["train"]), ("val", cfg["val"]), ("test", cfg["test"])]:
            mask = (df.index.year >= years[0]) & (df.index.year <= years[1])
            usable = max(0, int(mask.sum()) - int(config.window_size) - max(config.contagion_horizons))
            print(f"  Chunk {cid} {split:5s} {years[0]}-{years[1]}: {int(mask.sum()):5d} days | ~{usable:5d} windows")
    adj_mb = df.shape[1] ** 2 * 4 / 1024 ** 2
    print(f"\nAdjacency matrix: {adj_mb:.1f} MB ({df.shape[1]}×{df.shape[1]})")


def cmd_hpo(config: ContagionConfig, chunk_id: int, trials: int, fresh: bool = False) -> None:
    print("=" * 72)
    print(f"STEMGNN HPO — Chunk {chunk_id} ({trials} trials)")
    print("=" * 72)
    configure_torch_runtime(config.cpu_threads, config.deterministic)
    run_hpo(config, chunk_id, trials, fresh=fresh)


def cmd_train(config: ContagionConfig, chunk_id: int) -> None:
    print("=" * 72)
    print(f"STEMGNN TRAINING — Chunk {chunk_id}")
    print("=" * 72)
    configure_torch_runtime(config.cpu_threads, config.deterministic)
    apply_hpo_params_if_available(config, chunk_id)
    df = load_returns_frame(config)
    train_contagion_model(config, chunk_id, df, resume=True, save_checkpoints=True)


def cmd_predict(config: ContagionConfig, chunk_id: int, split: str) -> None:
    print("=" * 72)
    print(f"STEMGNN PREDICTION — Chunk {chunk_id} / {split}")
    print("=" * 72)
    configure_torch_runtime(config.cpu_threads, config.deterministic)
    label = CHUNK_CONFIG[int(chunk_id)]["label"]
    model_path = Path(config.output_dir) / "models" / "StemGNN" / label / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"No checkpoint found: {model_path}")
    result = load_and_predict(model_path, split=split, chunk_id=chunk_id, device=config.device, run_xai=True)
    pred_path = Path(config.output_dir) / "results" / "StemGNN" / f"contagion_scores_{label}_{split}.csv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    result["predictions"].to_csv(pred_path, index_label="ticker")
    print(f"\nPredictions saved: {pred_path}")
    for col in result["predictions"].columns:
        print(f"  Mean {col}: {result['predictions'][col].mean():.6f}")
    print(f"  XAI levels returned: {list(result['xai'].keys())}")


def cmd_smoke(config: ContagionConfig, chunk_id: int, real: bool = False) -> None:
    print("=" * 72)
    print(f"STEMGNN SMOKE TEST — {'real data' if real else 'synthetic data'}")
    print("=" * 72)
    configure_torch_runtime(config.cpu_threads, config.deterministic)
    config.device = str(resolve_device(config.device))
    config.ticker_limit = int(config.ticker_limit or 32)
    config.batch_size = int(config.batch_size or 2)
    config.num_workers = int(config.num_workers if config.num_workers is not None else 0)
    config.persistent_workers = False
    config.epochs = int(config.epochs or 1)
    config.max_train_windows = int(config.max_train_windows or 4)
    config.max_eval_windows = int(config.max_eval_windows or 2)
    config.xai_sample_size = min(int(config.xai_sample_size), 1)
    config.enable_gnnexplainer = True
    config.save_xai_to_disk = False
    config.min_history_days = min(int(config.min_history_days), 20)

    if real:
        df = load_returns_frame(config)
    else:
        df = make_synthetic_returns_frame(config.ticker_limit, seed=config.seed)

    model, best_loss, summary = train_contagion_model(
        config,
        chunk_id,
        df,
        resume=False,
        save_checkpoints=False,
        run_tag=f"smoke_{CHUNK_CONFIG[int(chunk_id)]['label']}",
    )
    device = resolve_device(config.device)
    val_ds, tickers = build_split_dataset(config, chunk_id, "val", df, model.norm_stats)
    preds = generate_contagion_predictions(model, val_ds, config, tickers, device, "smoke_val")
    xai = run_full_xai(model, val_ds, config, tickers, device, "smoke_val")
    print("\nSMOKE TEST PASSED")
    print(f"  best_loss={best_loss:.6f}")
    print(f"  predictions_shape={preds.shape}")
    print(f"  xai_levels={list(xai.keys())}")
    print(f"  summary={summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description="StemGNN Contagion Risk Module")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("inspect", help="Verify input data and system resources")
    add_common_args(p)

    p = sub.add_parser("smoke", help="Run a minimum smoke test")
    p.add_argument("--chunk", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--real", action="store_true", help="Use real returns_panel_wide.csv instead of synthetic data")
    add_common_args(p)

    p = sub.add_parser("hpo", help="Run Optuna TPE hyperparameter optimisation")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--fresh", action="store_true", help="Delete existing HPO database before running")
    add_common_args(p)

    p = sub.add_parser("train-best", help="Train with best HPO params if available")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--fresh", action="store_true", help="Archive old checkpoint/metrics and start fresh")
    add_common_args(p)

    p = sub.add_parser("train-best-all", help="Train all chunks sequentially")
    p.add_argument("--fresh", action="store_true", help="Archive old checkpoints/metrics and start fresh")
    add_common_args(p)

    p = sub.add_parser("predict", help="Generate predictions and XAI")
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
        cmd_smoke(config, args.chunk, real=bool(args.real))
    elif args.command == "hpo":
        cmd_hpo(config, args.chunk, args.trials, fresh=bool(args.fresh))
    elif args.command == "train-best":
        cmd_train(config, args.chunk)
    elif args.command == "train-best-all":
        for cid in [1, 2, 3]:
            print(f"\n{'=' * 72}\nCHUNK {cid}\n{'=' * 72}")
            cmd_train(config, cid)
    elif args.command == "predict":
        cmd_predict(config, args.chunk, args.split)


if __name__ == "__main__":
    main()

# Single-line run commands:
# python code/gnn/stemgnn_contagion.py inspect --repo-root . --device cuda --num-workers 6 --cpu-threads 6
# python code/gnn/stemgnn_contagion.py smoke --repo-root . --device cuda --ticker-limit 32 --batch-size 2 --num-workers 0 --cpu-threads 6 --epochs 1 --max-train-windows 4 --max-eval-windows 2
# python code/gnn/stemgnn_contagion.py smoke --repo-root . --device cuda --real --ticker-limit 64 --batch-size 2 --num-workers 0 --cpu-threads 6 --epochs 1 --max-train-windows 4 --max-eval-windows 2
# python code/gnn/stemgnn_contagion.py hpo --repo-root . --chunk 2 --trials 50 --device cuda --fresh
# python code/gnn/stemgnn_contagion.py train-best --repo-root . --chunk 2 --device cuda --fresh
# python code/gnn/stemgnn_contagion.py train-best --repo-root . --chunk 2 --device cuda
# python code/gnn/stemgnn_contagion.py predict --repo-root . --chunk 2 --split test --device cuda
# python code/gnn/stemgnn_contagion.py predict --repo-root . --chunk 2 --split test --device cuda --enable-gnnexplainer
