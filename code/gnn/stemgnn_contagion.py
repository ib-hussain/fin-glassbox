#!/usr/bin/env python3
"""StemGNN Contagion Risk Module for fin-glassbox.

This file is a drop-in replacement for:
    code/gnn/stemgnn_contagion.py

It fixes the uploaded implementation and adds hardware-aware execution for:
    - 6-core / 12-thread CPU
    - NVMe-backed data loading
    - 64 GB system RAM
    - RTX 3090 Ti 24 GB GPU

Main commands:
    python code/gnn/stemgnn_contagion.py inspect --repo-root .
    python code/gnn/stemgnn_contagion.py smoke --repo-root . --device cuda --real --ticker-limit 64 --max-train-windows 8 --epochs 1 --batch-size 2 --num-workers 0
    python code/gnn/stemgnn_contagion.py train-best --repo-root . --chunk 1 --device cuda
    python code/gnn/stemgnn_contagion.py predict --repo-root . --chunk 1 --split test --device cuda

Important design decisions:
    - `returns_panel_wide.csv` is treated as DAILY LOG RETURNS.
    - h-day forward returns are computed as SUM of daily log returns.
    - Targets are binary contagion events: extreme negative h-day return and
      worse than recent expectation by `excess_threshold_std` standard deviations.
    - Loss is BCEWithLogitsLoss with optional positive-class weighting.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Local imports when this file lives in code/gnn/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stemgnn_base_model import Model as StemGNNBase  # noqa: E402


# -----------------------------------------------------------------------------
# Hardware setup
# -----------------------------------------------------------------------------

def configure_torch_runtime(num_threads: int = 6, deterministic: bool = False) -> None:
    """Configure PyTorch for RTX 30-series + limited CPU threads.

    Determinism is slower. For fastest research runs on RTX 3090 Ti, leave it off.
    """
    num_threads = max(1, int(num_threads))
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(max(1, min(2, num_threads // 2)))

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic

    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def resolve_device(device: str) -> torch.device:
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False on this machine.")
        return torch.device(device)
    return torch.device(device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class ContagionConfig:
    # Paths
    repo_root: str = ""
    returns_path: str = "data/yFinance/processed/returns_panel_wide.csv"
    output_dir: str = "outputs"

    # Architecture
    window_size: int = 30
    horizon: int = 1
    multi_layer: int = 13
    dropout_rate: float = 0.75
    leaky_rate: float = 0.2
    stack_cnt: int = 2

    # Contagion target construction
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

    # Data controls
    max_train_windows: int = 0
    max_eval_windows: int = 0
    ticker_limit: int = 0
    chunk_id: int = 1

    def resolve_paths(self) -> "ContagionConfig":
        if self.repo_root:
            root = Path(self.repo_root)
            if not Path(self.returns_path).is_absolute():
                self.returns_path = str(root / self.returns_path)
            if not Path(self.output_dir).is_absolute():
                self.output_dir = str(root / self.output_dir)
        return self


CHUNK_CONFIG = {
    1: {"train": (2000, 2004), "val": (2005, 2005), "test": (2006, 2006), "label": "chunk1"},
    2: {"train": (2007, 2014), "val": (2015, 2015), "test": (2016, 2016), "label": "chunk2"},
    3: {"train": (2017, 2022), "val": (2023, 2023), "test": (2024, 2024), "label": "chunk3"},
}


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class ContagionDataset(Dataset):
    """Contagion dataset built from a full daily log-return matrix.

    The dataset precomputes all windows and targets once. This is faster for GPU
    training because DataLoader workers only slice already-normalised arrays.
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

        self.config = config
        self.tickers = list(tickers)
        self.num_nodes = len(tickers)
        self.window_size = int(config.window_size)
        self.horizons = list(config.contagion_horizons)
        self.max_horizon = max(self.horizons)
        self.label = label

        returns = np.asarray(returns_matrix, dtype=np.float32)
        if not np.isfinite(returns).all():
            returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        self.returns = returns
        self.n_dates = returns.shape[0]

        start_idx = max(int(start_idx), self.window_size)
        end_idx_exclusive = min(int(end_idx_exclusive), self.n_dates)
        last_valid_t = end_idx_exclusive - self.max_horizon
        candidate_t = np.arange(start_idx, last_valid_t, dtype=np.int64)
        if len(candidate_t) == 0:
            raise ValueError(
                f"No usable windows for {label}: start={start_idx}, end={end_idx_exclusive}, "
                f"window={self.window_size}, max_horizon={self.max_horizon}"
            )

        if max_windows and len(candidate_t) > max_windows:
            # For smoke tests, use the earliest windows. This is deterministic and avoids
            # random disk/cache patterns. For full training, leave max_windows=0.
            candidate_t = candidate_t[: int(max_windows)]

        self.sample_t = candidate_t
        n_samples = len(candidate_t)

        self.windows = np.empty((n_samples, self.num_nodes, self.window_size), dtype=np.float32)
        self.targets = np.zeros((n_samples, self.num_nodes, len(self.horizons)), dtype=np.float32)

        csum = np.vstack([
            np.zeros((1, self.num_nodes), dtype=np.float32),
            np.cumsum(returns, axis=0, dtype=np.float32),
        ])

        horizon_returns = {
            h: (csum[h:] - csum[:-h]).astype(np.float32) for h in self.horizons
        }

        iterator = tqdm(candidate_t, desc=f"  Building {label}", leave=False)
        for out_idx, t in enumerate(iterator):
            self.windows[out_idx] = returns[t - self.window_size:t].T

            hist_start = max(0, t - config.history_days)
            for h_idx, h in enumerate(self.horizons):
                hret = horizon_returns[h]
                forward_ret = hret[t]

                # Historical h-day returns whose h-day period ends before t.
                hist_end = max(hist_start, t - h + 1)
                hist = hret[hist_start:hist_end]
                if hist.shape[0] < config.min_history_days:
                    continue

                recent_start = max(hist_start, t - config.recent_days - h + 1)
                recent = hret[recent_start:hist_end]
                if recent.shape[0] < 5:
                    recent = hist[-min(hist.shape[0], config.recent_days):]

                thresholds = np.quantile(hist, config.extreme_quantile, axis=0)
                expected = np.mean(recent, axis=0)
                sigma = np.std(recent, axis=0)
                sigma = np.where(sigma < 1e-8, 1e-8, sigma)

                below_threshold = forward_ret < thresholds
                excess_negative = (forward_ret - expected) < (-config.excess_threshold_std * sigma)
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

        # Normalise once, in-place, so __getitem__ stays cheap.
        self.windows -= self.norm_stats["mean"]
        self.windows /= self.norm_stats["std"]
        self.windows = np.ascontiguousarray(self.windows, dtype=np.float32)
        self.targets = np.ascontiguousarray(self.targets, dtype=np.float32)

        pos = self.targets.sum(axis=(0, 1))
        total = self.targets.shape[0] * self.targets.shape[1]
        neg = total - pos
        pos_weight = neg / np.maximum(pos, 1.0)
        self.pos_weight = np.clip(pos_weight, 1.0, config.max_pos_weight).astype(np.float32)
        self.positive_rate = (pos / max(total, 1)).astype(np.float32)

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.windows[idx]),
            "target": torch.from_numpy(self.targets[idx]),
            "t": torch.tensor(int(self.sample_t[idx]), dtype=torch.long),
        }


def norm_stats_to_serialisable(stats: Dict[str, np.ndarray]) -> Dict[str, List]:
    return {"mean": np.asarray(stats["mean"]).tolist(), "std": np.asarray(stats["std"]).tolist()}


def norm_stats_from_checkpoint(stats: Dict) -> Dict[str, np.ndarray]:
    return {"mean": np.asarray(stats["mean"], dtype=np.float32), "std": np.asarray(stats["std"], dtype=np.float32)}


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class ContagionStemGNN(nn.Module):
    """StemGNN adapted for multi-horizon contagion classification."""

    def __init__(self, config: ContagionConfig, num_nodes: int, norm_stats: Optional[Dict] = None) -> None:
        super().__init__()
        self.config = config
        self.num_nodes = int(num_nodes)
        self.norm_stats = norm_stats

        self.stemgnn = StemGNNBase(
            units=self.num_nodes,
            stack_cnt=config.stack_cnt,
            time_step=config.window_size,
            multi_layer=config.multi_layer,
            horizon=config.window_size,
            dropout_rate=config.dropout_rate,
            leaky_rate=config.leaky_rate,
            device="cpu",  # final placement is handled by caller
        )

        n_horizons = len(config.contagion_horizons)
        hidden = max(4, config.window_size // 2)
        self.contagion_heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(config.window_size, hidden),
                nn.LeakyReLU(0.2),
                nn.Dropout(config.dropout_rate),
                nn.Linear(hidden, 1),
            )
            for _ in range(n_horizons)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [B, N, W]
        if x.dim() != 3:
            raise ValueError(f"Expected x [batch, nodes, window], got {tuple(x.shape)}")
        if x.size(1) != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {x.size(1)}")

        stem_out, attention = self.stemgnn(x.permute(0, 2, 1).contiguous())
        # Base output is [B, W, N] when horizon=window_size.
        if stem_out.dim() == 3 and stem_out.size(-1) == self.num_nodes:
            stem_features = stem_out.permute(0, 2, 1).contiguous()
        else:
            stem_features = stem_out.contiguous()

        logits = []
        for head in self.contagion_heads:
            logits.append(head(stem_features).squeeze(-1))
        contagion_logits = torch.stack(logits, dim=-1)       # [B, N, H]
        contagion_scores = torch.sigmoid(contagion_logits)  # [B, N, H]

        return {
            "contagion_logits": contagion_logits,
            "contagion_scores": contagion_scores,
            "attention": attention,
            "stemgnn_output": stem_features,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": asdict(self.config),
                "num_nodes": self.num_nodes,
                "norm_stats": norm_stats_to_serialisable(self.norm_stats) if self.norm_stats is not None else None,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "ContagionStemGNN":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        cfg_dict = checkpoint["config"]
        cfg_dict["device"] = str(device)
        config = ContagionConfig(**cfg_dict)
        norm_stats = checkpoint.get("norm_stats")
        if norm_stats is not None:
            norm_stats = norm_stats_from_checkpoint(norm_stats)
        model = cls(config, checkpoint["num_nodes"], norm_stats=norm_stats)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model


def maybe_compile_model(model: nn.Module, config: ContagionConfig) -> nn.Module:
    if not config.compile_model:
        return model
    if not hasattr(torch, "compile"):
        print("  [compile] torch.compile not available; continuing without compilation.")
        return model
    print("  [compile] Compiling model. First epoch may be slower due to compilation overhead.")
    return torch.compile(model, mode="reduce-overhead")


# -----------------------------------------------------------------------------
# Data loading and dataloaders
# -----------------------------------------------------------------------------

def load_returns_frame(config: ContagionConfig) -> pd.DataFrame:
    fp = Path(config.returns_path)
    if not fp.exists():
        raise FileNotFoundError(f"Returns file not found: {fp}")
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    df = df.sort_index()
    if config.ticker_limit and config.ticker_limit > 0:
        df = df.iloc[:, : int(config.ticker_limit)]
    return df.astype(np.float32)


def split_indices(dates: pd.DatetimeIndex, year_pair: Tuple[int, int]) -> Tuple[int, int]:
    mask = (dates.year >= year_pair[0]) & (dates.year <= year_pair[1])
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise ValueError(f"No dates found for years {year_pair}")
    return int(idx[0]), int(idx[-1] + 1)


def make_loader(dataset: Dataset, config: ContagionConfig, train: bool, override_workers: Optional[int] = None) -> DataLoader:
    workers = config.num_workers if override_workers is None else int(override_workers)
    workers = max(0, int(workers))
    kwargs = {
        "batch_size": int(config.batch_size),
        "shuffle": bool(train),
        "drop_last": bool(train and len(dataset) >= config.batch_size),
        "num_workers": workers,
        "pin_memory": str(config.device).startswith("cuda"),
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 4
        kwargs["persistent_workers"] = bool(config.persistent_workers)
    return DataLoader(dataset, **kwargs)


def build_train_val_datasets(config: ContagionConfig, chunk_id: int, df: pd.DataFrame):
    chunk_cfg = CHUNK_CONFIG[chunk_id]
    returns = df.values.astype(np.float32)
    tickers = list(df.columns)
    train_start, train_end = split_indices(df.index, chunk_cfg["train"])
    val_start, val_end = split_indices(df.index, chunk_cfg["val"])

    train_ds = ContagionDataset(
        returns,
        tickers,
        config,
        start_idx=train_start,
        end_idx_exclusive=train_end,
        fit_stats=True,
        max_windows=config.max_train_windows,
        label=f"{chunk_cfg['label']}_train",
    )
    val_ds = ContagionDataset(
        returns,
        tickers,
        config,
        start_idx=val_start,
        end_idx_exclusive=val_end,
        fit_stats=False,
        norm_stats=train_ds.norm_stats,
        max_windows=config.max_eval_windows,
        label=f"{chunk_cfg['label']}_val",
    )
    return train_ds, val_ds, tickers


# -----------------------------------------------------------------------------
# Training / validation
# -----------------------------------------------------------------------------

def make_loss_fn(train_ds: ContagionDataset, config: ContagionConfig, device: torch.device):
    if config.use_pos_weight:
        pos_weight = torch.tensor(train_ds.pos_weight, dtype=torch.float32, device=device).view(1, 1, -1)
        print(f"  Positive rates by horizon: {train_ds.positive_rate.tolist()}")
        print(f"  BCE pos_weight by horizon: {train_ds.pos_weight.tolist()}")
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return nn.BCEWithLogitsLoss()


def train_epoch(model, loader, optimizer, loss_fn, device, config, scaler=None) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    use_amp = scaler is not None and device.type == "cuda"

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(x)
            loss = loss_fn(output["contagion_logits"], target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate_epoch(model, loader, loss_fn, device) -> Dict[str, float]:
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
        reduce_dims = (0, 1)
        batch_pred = probs.sum(dim=reduce_dims).detach().cpu()
        batch_target = target.sum(dim=reduce_dims).detach().cpu()
        pred_sum = batch_pred if pred_sum is None else pred_sum + batch_pred
        target_sum = batch_target if target_sum is None else target_sum + batch_target
        count += probs.shape[0] * probs.shape[1]

    metrics = {"loss": total_loss / max(n_batches, 1)}
    if count > 0:
        pred_mean = (pred_sum / count).numpy().tolist()
        target_mean = (target_sum / count).numpy().tolist()
        for i, value in enumerate(pred_mean):
            metrics[f"pred_mean_h{i}"] = float(value)
        for i, value in enumerate(target_mean):
            metrics[f"target_rate_h{i}"] = float(value)
    return metrics


def build_optimizer(model: nn.Module, config: ContagionConfig):
    if config.optimizer.lower() == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=config.learning_rate, eps=1e-8, weight_decay=config.weight_decay)
    if config.optimizer.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=max(config.weight_decay, 1e-5))
    return torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


def train_contagion_model(config: ContagionConfig, chunk_id: int, df: pd.DataFrame):
    chunk_cfg = CHUNK_CONFIG[chunk_id]
    label = chunk_cfg["label"]
    device = resolve_device(config.device)
    seed_everything(config.seed)

    train_ds, val_ds, tickers = build_train_val_datasets(config, chunk_id, df)
    train_loader = make_loader(train_ds, config, train=True)
    val_loader = make_loader(val_ds, config, train=False)

    model = ContagionStemGNN(config, len(tickers), norm_stats=train_ds.norm_stats).to(device)
    model = maybe_compile_model(model, config)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {total_params:,}")
    print(f"  Train windows: {len(train_ds):,} | Val windows: {len(val_ds):,} | Tickers: {len(tickers):,}")

    optimizer = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.decay_rate)
    loss_fn = make_loss_fn(train_ds, config, device)
    scaler = torch.cuda.amp.GradScaler(enabled=(config.amp and device.type == "cuda"))

    out_dir = Path(config.output_dir) / "models" / "StemGNN" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_model.pt"
    latest_path = out_dir / "latest_model.pt"
    metrics_path = out_dir / "training_metrics.jsonl"

    best_val_loss = math.inf
    no_improve = 0
    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, config, scaler=scaler)
        val_metrics = validate_epoch(model, val_loader, loss_fn, device)
        val_loss = val_metrics["loss"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - epoch_start,
            **val_metrics,
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        print(
            f"  [{label}] Epoch {epoch:03d}/{config.epochs} | "
            f"train={train_loss:.5f} | val={val_loss:.5f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | {row['seconds']:.1f}s"
        )

        # Need the real model object for saving if torch.compile wraps it.
        save_model_obj = model._orig_mod if hasattr(model, "_orig_mod") else model
        save_model_obj.save(latest_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            save_model_obj.save(best_path)
        else:
            no_improve += 1

        if epoch % config.exponential_decay_step == 0:
            scheduler.step()

        if no_improve >= config.early_stop_patience:
            print(f"  Early stopping at epoch {epoch}.")
            break

    print(f"  Best val loss: {best_val_loss:.6f} | elapsed: {(time.time() - start_time) / 60:.2f} min")
    return model, best_val_loss


# -----------------------------------------------------------------------------
# XAI / prediction
# -----------------------------------------------------------------------------

@torch.no_grad()
def _online_mean_update(current, count, new_value):
    if current is None:
        return new_value.detach().cpu().float(), 1
    count_new = count + 1
    current += (new_value.detach().cpu().float() - current) / count_new
    return current, count_new


def extract_xai_and_scores(model, dataloader, config, tickers, split_label: str):
    """Extract adjacency, top influencers, gradient-based importance, and scores."""
    device = resolve_device(config.device)
    model.eval()
    xai_dir = Path(config.output_dir) / "results" / "StemGNN" / "xai"
    xai_dir.mkdir(parents=True, exist_ok=True)

    attention_mean = None
    attention_count = 0
    score_sum = None
    score_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"  Predict/XAI {split_label}"):
            x = batch["x"].to(device, non_blocking=True)
            output = model(x)
            attention_mean, attention_count = _online_mean_update(attention_mean, attention_count, output["attention"])
            scores = output["contagion_scores"].detach().cpu().float().sum(dim=0)
            score_sum = scores if score_sum is None else score_sum + scores
            score_count += x.size(0)

    avg_attention = attention_mean.numpy()
    avg_contagion = (score_sum / max(score_count, 1)).numpy()

    # Remove self-influence for reporting.
    top_influencers = {}
    for i, ticker in enumerate(tickers):
        row = avg_attention[i].copy()
        row[i] = -np.inf
        top_idx = np.argsort(row)[-config.xai_top_influencers:][::-1]
        top_influencers[ticker] = [
            {"ticker": tickers[j], "weight": float(row[j])}
            for j in top_idx if np.isfinite(row[j])
        ]

    np.save(xai_dir / f"{split_label}_adjacency.npy", avg_attention.astype(np.float32))
    with open(xai_dir / f"{split_label}_top_influencers.json", "w") as f:
        json.dump(top_influencers, f, indent=2)

    # Gradient-based node/edge importance approximation.
    n_samples = min(config.xai_sample_size, len(dataloader.dataset))
    if n_samples > 0:
        node_importance = np.zeros((len(tickers), config.window_size), dtype=np.float32)
        processed = 0
        for i in range(n_samples):
            sample = dataloader.dataset[i]
            x = sample["x"].unsqueeze(0).to(device)
            x.requires_grad_(True)
            output = model(x)
            score = output["contagion_logits"].sum()
            model.zero_grad(set_to_none=True)
            score.backward()
            grad = x.grad.detach().abs().cpu().numpy().squeeze(0)  # [N, W]
            node_importance += grad.astype(np.float32)
            processed += 1
        node_importance /= max(processed, 1)
        edge_importance = avg_attention.astype(np.float32) * node_importance.mean(axis=1)[None, :]
        np.save(xai_dir / f"{split_label}_node_temporal_importance.npy", node_importance)
        np.save(xai_dir / f"{split_label}_edge_importance.npy", edge_importance)

    if config.enable_gnnexplainer:
        # Approximate input-node mask explainer. StemGNN learns graph internally,
        # so a true edge-mask explainer needs a deeper model hook. This opt-in
        # routine is safe and will not crash, but it is intentionally labelled
        # approximate in its output.
        explain_count = min(3, n_samples)
        explainer_results = []
        for idx in range(explain_count):
            x_orig = dataloader.dataset[idx]["x"].unsqueeze(0).to(device)
            with torch.no_grad():
                orig_score = model(x_orig)["contagion_scores"].mean()
            mask = nn.Parameter(torch.zeros(len(tickers), 1, device=device))
            opt = torch.optim.Adam([mask], lr=0.05)
            for _ in range(25):
                masked_x = x_orig * torch.sigmoid(mask).unsqueeze(0)
                pred = model(masked_x)["contagion_scores"].mean()
                loss = F.mse_loss(pred, orig_score) + 0.01 * torch.sigmoid(mask).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            final_mask = torch.sigmoid(mask).detach().cpu().numpy().squeeze()
            top_nodes = np.argsort(final_mask)[-20:][::-1]
            explainer_results.append({
                "sample_idx": int(idx),
                "method": "approximate_input_node_mask_not_true_edge_mask",
                "important_nodes": [
                    {"ticker": tickers[j], "importance": float(final_mask[j])}
                    for j in top_nodes
                ],
            })
        with open(xai_dir / f"{split_label}_gnnexplainer_approx.json", "w") as f:
            json.dump(explainer_results, f, indent=2)

    print(f"  [xai] Saved explanations to: {xai_dir}")
    return avg_contagion, top_influencers


# -----------------------------------------------------------------------------
# CLI commands
# -----------------------------------------------------------------------------

def cmd_inspect(config: ContagionConfig) -> None:
    print("=" * 72)
    print("STEMGNN CONTAGION — DATA + HARDWARE INSPECTION")
    print("=" * 72)
    print(f"Device requested: {config.device}")
    print(f"CPU threads configured: {config.cpu_threads}")
    print(f"DataLoader workers: {config.num_workers}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"CUDA device 0: {props.name} | VRAM: {props.total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA available: False")

    df = load_returns_frame(config)
    print(f"\nReturns matrix: {df.shape[0]:,} days × {df.shape[1]:,} tickers")
    print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"NaN rate: {df.isna().mean().mean() * 100:.6f}%")
    print(f"Approx file memory as float32: {df.shape[0] * df.shape[1] * 4 / 1024**2:.1f} MB")

    print("\nChronological splits:")
    for cid, cfg in CHUNK_CONFIG.items():
        for split, years in [("train", cfg["train"]), ("val", cfg["val"]), ("test", cfg["test"] )]:
            mask = (df.index.year >= years[0]) & (df.index.year <= years[1])
            usable = max(0, int(mask.sum()) - max(config.contagion_horizons) - 1)
            print(f"  Chunk {cid} {split:5s} {years[0]}-{years[1]}: {int(mask.sum()):5d} days | ~{usable:5d} label windows")

    n = df.shape[1]
    adjacency_mb = n * n * 4 / 1024**2
    print(f"\nDense adjacency per matrix: {adjacency_mb:.1f} MB for {n:,}×{n:,}")
    print("RTX 3090 Ti 24GB is sufficient for the default batch size, but batch size is still architecture-limited by dense N×N graph ops.")


def cmd_smoke(config: ContagionConfig, use_real: bool) -> None:
    """Minimum-time smoke test. Uses tiny synthetic data unless --real is passed."""
    print("=" * 72)
    print("STEMGNN CONTAGION — MINIMUM SMOKE TEST")
    print("=" * 72)
    device = resolve_device(config.device)
    seed_everything(config.seed)

    original_epochs = config.epochs
    # Smoke test must be minimum-time by default: exactly one epoch.
    # Use train-best for real training.
    config.epochs = 1
    config.max_train_windows = max(1, int(config.max_train_windows or 8))
    config.max_eval_windows = max(1, int(config.max_eval_windows or 4))

    if use_real:
        df = load_returns_frame(config)
        if config.ticker_limit <= 0:
            # Avoid accidentally running a 2,500-node smoke test.
            config.ticker_limit = 64
            df = df.iloc[:, :64]
        print(f"  Real smoke data: {df.shape[0]} days × {df.shape[1]} tickers")
    else:
        n_days = 180
        n_tickers = config.ticker_limit if config.ticker_limit > 0 else 32
        dates = pd.bdate_range("2020-01-01", periods=n_days)
        rng = np.random.default_rng(config.seed)
        values = rng.normal(0.0002, 0.02, size=(n_days, n_tickers)).astype(np.float32)
        # Inject a tiny market-wide crash pattern so targets are non-trivial.
        values[120:125] -= 0.06
        df = pd.DataFrame(values, index=dates, columns=[f"T{i:03d}" for i in range(n_tickers)])
        print(f"  Synthetic smoke data: {df.shape[0]} days × {df.shape[1]} tickers")

    # Use chunk 3 if synthetic years are 2020; otherwise requested chunk.
    chunk_id = config.chunk_id
    if not use_real:
        chunk_id = 3
        CHUNK_CONFIG[3] = {"train": (2020, 2020), "val": (2020, 2020), "test": (2020, 2020), "label": "smoke"}

    config.batch_size = max(1, int(config.batch_size))
    config.num_workers = int(config.num_workers)
    model, loss = train_contagion_model(config, chunk_id, df)
    print(f"Smoke test completed. Best validation loss: {loss:.6f}")
    config.epochs = original_epochs


def cmd_hpo(config: ContagionConfig, chunk_id: int, n_trials: int) -> None:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("optuna is required for HPO. Install with: pip install optuna") from exc

    df = load_returns_frame(config)
    chunk_cfg = CHUNK_CONFIG[chunk_id]
    label = chunk_cfg["label"]

    def objective(trial):
        trial_config = ContagionConfig(**asdict(config))
        trial_config.window_size = trial.suggest_categorical("window_size", [15, 30, 60])
        trial_config.multi_layer = trial.suggest_categorical("multi_layer", [5, 8, 13])
        trial_config.dropout_rate = trial.suggest_categorical("dropout_rate", [0.5, 0.6, 0.75])
        trial_config.learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
        trial_config.decay_rate = trial.suggest_categorical("decay_rate", [0.5, 0.7, 0.9])
        trial_config.exponential_decay_step = trial.suggest_categorical("exponential_decay_step", [5, 8, 13])
        trial_config.batch_size = trial.suggest_categorical("batch_size", [4, 8, 12])
        trial_config.epochs = min(config.epochs, 15)
        trial_config.max_train_windows = config.max_train_windows or 384
        trial_config.max_eval_windows = config.max_eval_windows or 128
        trial_config.compile_model = False

        _, val_loss = train_contagion_model(trial_config, chunk_id, df)
        return float(val_loss)

    storage_dir = Path(config.output_dir) / "codeResults" / "StemGNN"
    storage_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.seed, n_startup_trials=config.hpo_n_startup),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        storage=f"sqlite:///{storage_dir / 'hpo.db'}",
        study_name=f"stemgnn_contagion_{label}",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=True)

    with open(storage_dir / f"best_params_{label}.json", "w") as f:
        json.dump({"params": study.best_params, "value": study.best_value}, f, indent=2)
    print(f"Best params: {study.best_params}")
    print(f"Best val loss: {study.best_value:.6f}")


def cmd_train_best(config: ContagionConfig, chunk_id: int) -> None:
    chunk_cfg = CHUNK_CONFIG[chunk_id]
    label = chunk_cfg["label"]
    hpo_path = Path(config.output_dir) / "codeResults" / "StemGNN" / f"best_params_{label}.json"
    if hpo_path.exists():
        with open(hpo_path) as f:
            best = json.load(f)
        for k, v in best.get("params", {}).items():
            setattr(config, k, v)
        print(f"Loaded HPO params from {hpo_path}")
    else:
        print("No HPO params found; using current/default config.")
    df = load_returns_frame(config)
    train_contagion_model(config, chunk_id, df)


def cmd_predict(config: ContagionConfig, chunk_id: int, split: str) -> None:
    chunk_cfg = CHUNK_CONFIG[chunk_id]
    label = chunk_cfg["label"]
    model_path = Path(config.output_dir) / "models" / "StemGNN" / label / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    device = resolve_device(config.device)
    model = ContagionStemGNN.load(model_path, device=str(device))
    model.eval()

    df = load_returns_frame(config)
    tickers = list(df.columns)
    returns = df.values.astype(np.float32)
    start, end = split_indices(df.index, chunk_cfg[split])
    ds = ContagionDataset(
        returns,
        tickers,
        config,
        start_idx=start,
        end_idx_exclusive=end,
        fit_stats=False,
        norm_stats=model.norm_stats,
        max_windows=config.max_eval_windows,
        label=f"{label}_{split}",
    )
    loader = make_loader(ds, config, train=False, override_workers=max(0, min(config.num_workers, 2)))
    avg_contagion, _ = extract_xai_and_scores(model, loader, config, tickers, f"{label}_{split}")

    scores_df = pd.DataFrame(
        avg_contagion,
        index=tickers,
        columns=[f"contagion_{h}d" for h in config.contagion_horizons],
    )
    out_path = Path(config.output_dir) / "results" / "StemGNN" / f"contagion_scores_{label}_{split}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(out_path, index_label="ticker")
    print(f"Contagion scores saved: {out_path}")
    for i, h in enumerate(config.contagion_horizons):
        print(f"Mean contagion {h}d: {avg_contagion[:, i].mean():.6f}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

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
    parser.add_argument("--amp", action="store_true", default=None)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--enable-gnnexplainer", action="store_true")


def config_from_args(args) -> ContagionConfig:
    config = ContagionConfig()
    if getattr(args, "repo_root", ""):
        config.repo_root = args.repo_root
    if getattr(args, "returns_path", ""):
        config.returns_path = args.returns_path
    if getattr(args, "output_dir", ""):
        config.output_dir = args.output_dir
    if getattr(args, "device", None):
        config.device = args.device
    if getattr(args, "batch_size", None) is not None:
        config.batch_size = args.batch_size
    if getattr(args, "epochs", None) is not None:
        config.epochs = args.epochs
    if getattr(args, "num_workers", None) is not None:
        config.num_workers = args.num_workers
    if getattr(args, "cpu_threads", None) is not None:
        config.cpu_threads = args.cpu_threads
    if getattr(args, "max_train_windows", None) is not None:
        config.max_train_windows = args.max_train_windows
    if getattr(args, "max_eval_windows", None) is not None:
        config.max_eval_windows = args.max_eval_windows
    if getattr(args, "ticker_limit", None) is not None:
        config.ticker_limit = args.ticker_limit
    if getattr(args, "amp", None) is not None:
        config.amp = bool(args.amp)
    if getattr(args, "compile_model", False):
        config.compile_model = True
    if getattr(args, "deterministic", False):
        config.deterministic = True
    if getattr(args, "enable_gnnexplainer", False):
        config.enable_gnnexplainer = True
    if getattr(args, "chunk", None) is not None:
        config.chunk_id = args.chunk
    config.resolve_paths()
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="StemGNN Contagion Risk Module")
    sub = parser.add_subparsers(dest="command")

    p_inspect = sub.add_parser("inspect")
    add_common_args(p_inspect)

    p_smoke = sub.add_parser("smoke")
    p_smoke.add_argument("--real", action="store_true", help="Use real returns file instead of synthetic tiny data.")
    p_smoke.add_argument("--chunk", type=int, default=1, choices=[1, 2, 3])
    add_common_args(p_smoke)

    p_hpo = sub.add_parser("hpo")
    p_hpo.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p_hpo.add_argument("--trials", type=int, default=50)
    add_common_args(p_hpo)

    p_train = sub.add_parser("train-best")
    p_train.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    add_common_args(p_train)

    p_train_all = sub.add_parser("train-best-all")
    add_common_args(p_train_all)

    p_predict = sub.add_parser("predict")
    p_predict.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p_predict.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    add_common_args(p_predict)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    config = config_from_args(args)
    configure_torch_runtime(config.cpu_threads, deterministic=config.deterministic)

    if args.command == "inspect":
        cmd_inspect(config)
    elif args.command == "smoke":
        cmd_smoke(config, use_real=args.real)
    elif args.command == "hpo":
        cmd_hpo(config, args.chunk, args.trials)
    elif args.command == "train-best":
        cmd_train_best(config, args.chunk)
    elif args.command == "train-best-all":
        for cid in [1, 2, 3]:
            print(f"\n{'=' * 72}\nTRAINING CHUNK {cid}\n{'=' * 72}")
            cmd_train_best(config, cid)
    elif args.command == "predict":
        cmd_predict(config, args.chunk, args.split)


if __name__ == "__main__":
    main()
