#!/usr/bin/env python3
"""
code/riskEngine/drawdown.py

Drawdown Risk Model — Continuous Dual-Horizon Downside Path Risk
================================================================

Project:
    fin-glassbox — Explainable Distributed Deep Learning Framework for Financial Risk Management

Purpose:
    Estimate future downside path risk from Temporal Encoder embeddings.

Inputs:
    outputs/embeddings/TemporalEncoder/chunk*_train_embeddings.npy
    outputs/embeddings/TemporalEncoder/chunk*_train_manifest.csv
    data/yFinance/processed/ohlcv_final.csv

Model:
    BiLSTM + Attention Pooling + Dual Horizon Heads

Outputs:
    10-day:
        expected_drawdown_10d
        drawdown_risk_10d
        recovery_days_10d
        confidence_10d

    30-day:
        expected_drawdown_30d
        drawdown_risk_30d
        recovery_days_30d
        confidence_30d

    combined:
        drawdown_risk_score

Target philosophy:
    This is intentionally not a binary classifier.
    Drawdown is a continuous path-dependent risk. The model predicts real-valued
    drawdown severity, soft threshold risk, and recovery delay.

XAI:
    L1: Attention weights over input embedding timesteps.
    L2: Gradient feature/timestep importance.
    L3: Counterfactual downside-risk scenarios.

CLI:
    python code/riskEngine/drawdown.py inspect --repo-root .
    python code/riskEngine/drawdown.py smoke --repo-root . --device cuda
    python code/riskEngine/drawdown.py hpo --repo-root . --chunk 1 --trials 40 --device cuda --fresh
    python code/riskEngine/drawdown.py train-best --repo-root . --chunk 1 --device cuda --fresh
    python code/riskEngine/drawdown.py predict --repo-root . --chunk 1 --split test --device cuda
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import shutil
import time
import warnings
from contextlib import nullcontext
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
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

CHUNK_CONFIG = {
    1: {"train": (2000, 2004), "val": (2005, 2005), "test": (2006, 2006), "label": "chunk1"},
    2: {"train": (2007, 2014), "val": (2015, 2015), "test": (2016, 2016), "label": "chunk2"},
    3: {"train": (2017, 2022), "val": (2023, 2023), "test": (2024, 2024), "label": "chunk3"},
}

TARGET_COLUMNS = [
    "expected_drawdown_10d",
    "drawdown_risk_10d",
    "recovery_norm_10d",
    "confidence_target_10d",
    "expected_drawdown_30d",
    "drawdown_risk_30d",
    "recovery_norm_30d",
    "confidence_target_30d",
]

HPO_FAILURE_VALUE = 1_000_000_000.0


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DrawdownConfig:
    repo_root: str = ""
    output_dir: str = "outputs"

    temporal_dir: str = "outputs/embeddings/TemporalEncoder"
    ohlcv_path: str = "data/yFinance/processed/ohlcv_final.csv"

    # Input / architecture
    input_dim: int = 256
    seq_len: int = 30
    lstm_hidden: int = 64
    lstm_layers: int = 1
    bidirectional: bool = True
    attention_dim: int = 64
    dropout: float = 0.20

    # Drawdown horizons
    horizon_10d: int = 10
    horizon_30d: int = 30

    # Soft risk thresholds
    drawdown_threshold_10d: float = 0.05
    drawdown_threshold_30d: float = 0.08
    softness_10d: float = 0.02
    softness_30d: float = 0.03
    max_drawdown_clip: float = 0.80
    min_drawdown_for_recovery: float = 0.005

    # Training
    batch_size: int = 512
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stop_patience: int = 10
    gradient_clip: float = 1.0
    mixed_precision: bool = True

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

    def resolve_paths(self) -> "DrawdownConfig":
        if self.repo_root:
            root = Path(self.repo_root)
            for attr in ["output_dir", "temporal_dir", "ohlcv_path"]:
                value = getattr(self, attr)
                if value and not Path(value).is_absolute():
                    setattr(self, attr, str(root / value))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
# RUNTIME UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def configure_torch_runtime(config: DrawdownConfig) -> None:
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
    """Disable cuDNN only for eval-mode LSTM gradient XAI."""
    if device.type == "cuda" and torch.backends.cudnn.enabled:
        return torch.backends.cudnn.flags(enabled=False)
    return nullcontext()


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
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return json_safe(obj.detach().cpu().item())
        return json_safe(obj.detach().cpu().tolist())
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def load_config_from_checkpoint_dict(raw: Dict[str, Any], fallback: Optional[DrawdownConfig] = None) -> DrawdownConfig:
    base = fallback.to_dict() if fallback is not None else DrawdownConfig().to_dict()
    ckpt_cfg = raw.get("config", {})
    if isinstance(ckpt_cfg, dict):
        base.update(ckpt_cfg)

    valid = {f.name for f in dataclass_fields(DrawdownConfig)}
    filtered = {k: v for k, v in base.items() if k in valid}
    return DrawdownConfig(**filtered).resolve_paths()


# ═══════════════════════════════════════════════════════════════════════════════
# PATHS / INPUT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def temporal_paths(config: DrawdownConfig, chunk_id: int, split: str) -> Tuple[Path, Path]:
    label = CHUNK_CONFIG[chunk_id]["label"]
    base = Path(config.temporal_dir)
    return base / f"{label}_{split}_embeddings.npy", base / f"{label}_{split}_manifest.csv"


def validate_required_inputs(config: DrawdownConfig, chunk_id: int, splits: Tuple[str, ...]) -> None:
    missing: List[Path] = []

    for split in splits:
        emb_path, manifest_path = temporal_paths(config, chunk_id, split)
        if not emb_path.exists():
            missing.append(emb_path)
        if not manifest_path.exists():
            missing.append(manifest_path)

    if not Path(config.ohlcv_path).exists():
        missing.append(Path(config.ohlcv_path))

    if missing:
        label = CHUNK_CONFIG[chunk_id]["label"]
        missing_text = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            f"Missing required inputs for Drawdown Risk Model {label}.\n"
            f"Missing files:\n{missing_text}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CLOSE PANEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_close_panel(config: DrawdownConfig) -> pd.DataFrame:
    """Load close prices as date × ticker panel.

    Uses a local NumPy cache because ohlcv_final.csv is large.
    """
    config.resolve_paths()
    cache_dir = Path(config.output_dir) / "cache" / "Drawdown"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "close_panel_wide_float32.npz"

    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        dates = pd.to_datetime(data["dates"].astype(str))
        tickers = data["tickers"].astype(str)
        close = data["close"].astype(np.float32, copy=False)
        return pd.DataFrame(close, index=dates, columns=tickers)

    print("  Building close-price panel cache from ohlcv_final.csv...")
    df = pd.read_csv(
        config.ohlcv_path,
        usecols=["date", "ticker", "close"],
        dtype={"ticker": str},
        parse_dates=["date"],
    )
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df["close"] = df["close"].ffill().bfill().fillna(0.0)
    df["close"] = np.maximum(df["close"].astype(np.float32), 1e-6)

    panel = df.pivot(index="date", columns="ticker", values="close").sort_index()
    panel = panel.ffill().bfill().fillna(0.0).astype(np.float32)

    np.savez(
        cache_path,
        dates=panel.index.strftime("%Y-%m-%d").values.astype(str),
        tickers=panel.columns.values.astype(str),
        close=panel.values.astype(np.float32),
    )

    print(f"  Saved close panel cache: {cache_path}")
    return panel


def split_date_index_bounds(dates: pd.DatetimeIndex, year_pair: Tuple[int, int]) -> Tuple[int, int]:
    mask = (dates.year >= year_pair[0]) & (dates.year <= year_pair[1])
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise ValueError(f"No close-price dates found for years {year_pair}")
    return int(idx[0]), int(idx[-1])


# ═══════════════════════════════════════════════════════════════════════════════
# DRAWDOWN TARGET CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def soft_drawdown_risk(drawdown: float, threshold: float, softness: float) -> float:
    z = (float(drawdown) - float(threshold)) / max(float(softness), 1e-8)
    z = float(np.clip(z, -20.0, 20.0))
    return float(1.0 / (1.0 + math.exp(-z)))


def compute_drawdown_metrics(
    prices: np.ndarray,
    horizon: int,
    threshold: float,
    softness: float,
    max_drawdown_clip: float,
    min_drawdown_for_recovery: float,
) -> Tuple[float, float, float, float]:
    """Compute continuous drawdown targets from a future price window.

    prices:
        close prices from t through t+h inclusive.

    Returns:
        expected_drawdown, soft_risk, recovery_norm, confidence_target
    """
    h = int(horizon)
    window = np.asarray(prices[: h + 1], dtype=np.float64)
    window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
    window = np.maximum(window, 1e-8)

    if len(window) < 2:
        return 0.0, soft_drawdown_risk(0.0, threshold, softness), 0.0, 1.0

    running_peak = np.maximum.accumulate(window)
    drawdowns = (running_peak - window) / np.maximum(running_peak, 1e-8)

    trough_idx = int(np.argmax(drawdowns))
    max_dd = float(drawdowns[trough_idx])
    max_dd = float(np.clip(max_dd, 0.0, max_drawdown_clip))

    risk = soft_drawdown_risk(max_dd, threshold, softness)

    if max_dd < min_drawdown_for_recovery:
        recovery_days = 0.0
    else:
        reference_peak = float(running_peak[trough_idx])
        after_trough = window[trough_idx:]
        recovered = np.where(after_trough >= reference_peak)[0]
        if len(recovered) > 0:
            recovery_days = float(recovered[0])
        else:
            recovery_days = float(h)

    recovery_norm = float(np.clip(recovery_days / max(h, 1), 0.0, 1.0))

    # Confidence target is high when the risk target is clearly low or clearly high.
    confidence_target = float(np.clip(abs(risk - 0.5) * 2.0, 0.0, 1.0))

    return max_dd, risk, recovery_norm, confidence_target


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class DrawdownDataset(Dataset):
    """Sequence dataset for continuous dual-horizon drawdown risk."""

    def __init__(
        self,
        config: DrawdownConfig,
        split: str,
        chunk_id: int,
        close_panel: pd.DataFrame,
        *,
        max_samples: int = 0,
        label_suffix: str = "",
    ) -> None:
        self.config = config
        self.split = split
        self.chunk_id = int(chunk_id)
        self.chunk_label = CHUNK_CONFIG[chunk_id]["label"]
        self.label_suffix = label_suffix

        emb_path, manifest_path = temporal_paths(config, chunk_id, split)
        if not emb_path.exists():
            raise FileNotFoundError(f"Missing embeddings: {emb_path}")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")

        self.embeddings = np.load(emb_path, mmap_mode="r")
        self.manifest = pd.read_csv(manifest_path, dtype={"ticker": str})
        self.manifest["date"] = pd.to_datetime(self.manifest["date"])
        self.manifest["date_str"] = self.manifest["date"].dt.strftime("%Y-%m-%d")
        self.manifest["row_id"] = np.arange(len(self.manifest), dtype=np.int64)

        n = min(len(self.manifest), len(self.embeddings))
        self.manifest = self.manifest.iloc[:n].reset_index(drop=True)
        self.embeddings = self.embeddings[:n]
        self.n_rows = n

        if self.n_rows == 0:
            raise ValueError(f"No rows available for {self.chunk_label}_{split}")

        first_emb = np.asarray(self.embeddings[0])
        if first_emb.shape[0] != int(config.input_dim):
            raise ValueError(
                f"Embedding dimension mismatch: expected {config.input_dim}, got {first_emb.shape[0]}"
            )

        self.close_panel = close_panel.sort_index()
        self.close_panel = self.close_panel.astype(np.float32)
        self.close_dates = self.close_panel.index
        self.date_to_idx = {str(d)[:10]: i for i, d in enumerate(self.close_dates)}

        split_start, split_end = split_date_index_bounds(self.close_dates, CHUNK_CONFIG[chunk_id][split])
        self.split_start_idx = split_start
        self.split_end_idx = split_end

        print(f"  Building {self.chunk_label}_{split}{label_suffix}: rows={self.n_rows:,}")

        (
            self.end_rows,
            self.targets,
            self.sample_tickers,
            self.sample_dates,
        ) = self._build_samples()

        if max_samples and max_samples > 0 and len(self.end_rows) > int(max_samples):
            rng = np.random.default_rng(int(config.seed))
            idx = rng.choice(len(self.end_rows), size=int(max_samples), replace=False)
            idx = np.sort(idx)
            self.end_rows = self.end_rows[idx]
            self.targets = self.targets[idx]
            self.sample_tickers = self.sample_tickers[idx]
            self.sample_dates = self.sample_dates[idx]

        self.audit = self._audit_dataset()

        print(
            f"  {self.chunk_label}_{split}{label_suffix}: {len(self.end_rows):,} samples, "
            f"{len(np.unique(self.sample_tickers)):,} tickers | "
            f"target_finite={self.audit['target_finite_ratio']:.6f}, "
            f"emb_sample_finite={self.audit['embedding_sample_finite_ratio']:.6f}"
        )

    def _build_samples(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        seq_len = int(self.config.seq_len)
        max_h = max(int(self.config.horizon_10d), int(self.config.horizon_30d))

        all_end_rows: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []
        all_tickers: List[np.ndarray] = []
        all_dates: List[np.ndarray] = []

        grouped = self.manifest.groupby("ticker", sort=True)

        for ticker, group in tqdm(grouped, desc=f"  Samples {self.chunk_label}_{self.split}", leave=False):
            ticker = str(ticker)

            if ticker not in self.close_panel.columns:
                continue

            group = group.sort_values("row_id")
            row_ids = group["row_id"].values.astype(np.int64)
            dates = group["date_str"].values.astype(str)

            if len(row_ids) < seq_len:
                continue

            close = self.close_panel[ticker].values.astype(np.float64)
            close = np.nan_to_num(close, nan=0.0, posinf=0.0, neginf=0.0)
            close = np.maximum(close, 1e-8)

            positions = np.arange(seq_len - 1, len(row_ids), dtype=np.int64)
            end_rows = row_ids[positions]
            end_dates = dates[positions]

            close_idx = np.array([self.date_to_idx.get(str(d)[:10], -1) for d in end_dates], dtype=np.int64)

            valid = (
                (close_idx >= self.split_start_idx)
                & ((close_idx + max_h) <= self.split_end_idx)
                & (close_idx >= 0)
            )

            if not np.any(valid):
                continue

            end_rows = end_rows[valid]
            end_dates = end_dates[valid]
            close_idx = close_idx[valid]

            targets = np.zeros((len(end_rows), 8), dtype=np.float32)

            for j, idx in enumerate(close_idx):
                w30 = close[int(idx): int(idx) + int(self.config.horizon_30d) + 1]

                dd10, risk10, rec10, conf10 = compute_drawdown_metrics(
                    w30,
                    int(self.config.horizon_10d),
                    float(self.config.drawdown_threshold_10d),
                    float(self.config.softness_10d),
                    float(self.config.max_drawdown_clip),
                    float(self.config.min_drawdown_for_recovery),
                )

                dd30, risk30, rec30, conf30 = compute_drawdown_metrics(
                    w30,
                    int(self.config.horizon_30d),
                    float(self.config.drawdown_threshold_30d),
                    float(self.config.softness_30d),
                    float(self.config.max_drawdown_clip),
                    float(self.config.min_drawdown_for_recovery),
                )

                targets[j] = np.array(
                    [dd10, risk10, rec10, conf10, dd30, risk30, rec30, conf30],
                    dtype=np.float32,
                )

            targets = np.nan_to_num(targets, nan=0.0, posinf=1.0, neginf=0.0)
            targets = np.clip(targets, 0.0, 1.0).astype(np.float32)

            all_end_rows.append(end_rows.astype(np.int64))
            all_targets.append(targets)
            all_tickers.append(np.array([ticker] * len(end_rows), dtype=object))
            all_dates.append(end_dates.astype(object))

        if not all_end_rows:
            raise ValueError(
                f"No usable drawdown samples for {self.chunk_label}_{self.split}. "
                f"Check seq_len, manifest dates, and future close-price coverage."
            )

        return (
            np.concatenate(all_end_rows, axis=0),
            np.concatenate(all_targets, axis=0),
            np.concatenate(all_tickers, axis=0),
            np.concatenate(all_dates, axis=0),
        )

    def _audit_dataset(self) -> Dict[str, Any]:
        sample_n = min(2048, len(self.end_rows))
        if sample_n > 0:
            idx = np.linspace(0, len(self.end_rows) - 1, sample_n).astype(np.int64)
            rows = []
            offsets = np.arange(int(self.config.seq_len) - 1, -1, -1, dtype=np.int64)
            for end_row in self.end_rows[idx]:
                rows.extend((end_row - offsets).tolist())
            emb_sample = np.asarray(self.embeddings[np.array(rows, dtype=np.int64)])
        else:
            emb_sample = np.array([], dtype=np.float32)

        out: Dict[str, Any] = {
            "samples": int(len(self.end_rows)),
            "embedding_sample_finite_ratio": finite_ratio_np(emb_sample),
            "target_finite_ratio": finite_ratio_np(self.targets),
        }

        for i, col in enumerate(TARGET_COLUMNS):
            out[f"{col}_mean"] = float(np.mean(self.targets[:, i])) if len(self.targets) else 0.0
            out[f"{col}_p95"] = float(np.percentile(self.targets[:, i], 95)) if len(self.targets) else 0.0

        return out

    def __len__(self) -> int:
        return int(len(self.end_rows))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        end_row = int(self.end_rows[idx])
        offsets = np.arange(int(self.config.seq_len) - 1, -1, -1, dtype=np.int64)
        rows = end_row - offsets

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
    def __init__(self, input_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.proj(x))
        scores = self.score(h).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return context, weights


class DrawdownRiskModel(nn.Module):
    def __init__(self, config: DrawdownConfig) -> None:
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

        shared_dim = max(32, lstm_out_dim // 2)

        self.shared = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Dropout(float(config.dropout)),
            nn.Linear(lstm_out_dim, shared_dim),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
        )

        self.head_10d = nn.Linear(shared_dim, 4)
        self.head_30d = nn.Linear(shared_dim, 4)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x = x.to(dtype=torch.float32)
        # torch.nan_to_num_(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.nan_to_num(
            x.to(dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        lstm_out, _ = self.lstm(x)
        context, attention_weights = self.attention(lstm_out)
        shared = self.shared(context)

        out10 = torch.sigmoid(self.head_10d(shared))
        out30 = torch.sigmoid(self.head_30d(shared))

        pred = torch.cat([out10, out30], dim=1)

        return {
            "predictions": pred,
            "expected_drawdown_10d": pred[:, 0],
            "drawdown_risk_10d": pred[:, 1],
            "recovery_norm_10d": pred[:, 2],
            "confidence_10d": pred[:, 3],
            "expected_drawdown_30d": pred[:, 4],
            "drawdown_risk_30d": pred[:, 5],
            "recovery_norm_30d": pred[:, 6],
            "confidence_30d": pred[:, 7],
            "attention_weights": attention_weights,
            "context": context,
        }

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": self.config.to_dict(),
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        config: Optional[DrawdownConfig] = None,
        device: str = "cpu",
    ) -> "DrawdownRiskModel":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = load_config_from_checkpoint_dict(ckpt, fallback=config)
        model = cls(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        return model


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS / LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def drawdown_loss(output: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    pred = torch.nan_to_num(output["predictions"].float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    target = torch.nan_to_num(target.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    expected_loss = F.smooth_l1_loss(pred[:, [0, 4]], target[:, [0, 4]])
    risk_loss = F.smooth_l1_loss(pred[:, [1, 5]], target[:, [1, 5]])
    recovery_loss = F.smooth_l1_loss(pred[:, [2, 6]], target[:, [2, 6]])
    confidence_loss = F.smooth_l1_loss(pred[:, [3, 7]], target[:, [3, 7]])

    return 2.0 * expected_loss + 1.25 * risk_loss + 0.75 * recovery_loss + 0.20 * confidence_loss


def tensors_are_finite(*tensors: torch.Tensor) -> bool:
    return all(torch.isfinite(t).all().item() for t in tensors)


def make_loader(dataset: Dataset, config: DrawdownConfig, train: bool) -> DataLoader:
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

def model_dir_for(config: DrawdownConfig, chunk_label: str, run_tag: str = "") -> Path:
    base = Path(config.output_dir) / "models" / "Drawdown"
    if run_tag:
        return base / "_hpo_trials" / run_tag / chunk_label
    return base / chunk_label


def save_latest_checkpoint(
    path: Path,
    model: DrawdownRiskModel,
    optimizer: torch.optim.Optimizer,
    config: DrawdownConfig,
    epoch: int,
    best_val_loss: float,
    history: List[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.to_dict(),
            "epoch": int(epoch),
            "best_val_loss": float(best_val_loss),
            "history": history,
        },
        path,
    )


def build_train_val_datasets(
    config: DrawdownConfig,
    chunk_id: int,
    close_panel: pd.DataFrame,
    *,
    run_tag: str = "",
) -> Tuple[DrawdownDataset, DrawdownDataset]:
    train_limit = int(config.max_train_samples) if config.max_train_samples else 0
    val_limit = int(config.max_val_samples) if config.max_val_samples else 0

    train_ds = DrawdownDataset(
        config,
        "train",
        chunk_id,
        close_panel,
        max_samples=train_limit,
        label_suffix=f"_{run_tag}" if run_tag else "",
    )

    val_ds = DrawdownDataset(
        config,
        "val",
        chunk_id,
        close_panel,
        max_samples=val_limit,
        label_suffix=f"_{run_tag}" if run_tag else "",
    )

    return train_ds, val_ds


def _train_model(
    config: DrawdownConfig,
    chunk_id: int,
    close_panel: pd.DataFrame,
    device: torch.device,
    *,
    run_tag: str = "",
    save_checkpoints: bool = True,
    resume: bool = True,
) -> Tuple[DrawdownRiskModel, Dict[str, Any]]:
    label = CHUNK_CONFIG[chunk_id]["label"]

    train_ds, val_ds = build_train_val_datasets(config, chunk_id, close_panel, run_tag=run_tag)

    train_loader = make_loader(train_ds, config, train=True)
    val_loader = make_loader(val_ds, config, train=False)

    model = DrawdownRiskModel(config).to(device)
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
    latest_path = out_dir / "latest_checkpoint.pt"
    history_path = out_dir / "training_history.csv"
    summary_path = out_dir / "training_summary.json"

    best_val_loss = float("inf")
    start_epoch = 1
    history: List[Dict[str, Any]] = []

    if save_checkpoints and resume and latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        history = list(ckpt.get("history", []))
        print(f"  Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")

    no_improve = 0
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(
        f"  Train: {len(train_ds):,} | Val: {len(val_ds):,} | "
        f"batch_size={config.batch_size} | params={total_params:,}"
    )
    print(f"  Train target means: 10d_dd={train_ds.audit['expected_drawdown_10d_mean']:.4f}, 30d_dd={train_ds.audit['expected_drawdown_30d_mean']:.4f}")
    print(f"  Val target means:   10d_dd={val_ds.audit['expected_drawdown_10d_mean']:.4f}, 30d_dd={val_ds.audit['expected_drawdown_30d_mean']:.4f}")

    epoch = start_epoch - 1

    try:
        for epoch in range(start_epoch, int(config.epochs) + 1):
            epoch_start = time.time()

            model.train()
            train_total = 0.0
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
                    raise RuntimeError("Non-finite train input/target detected.")

                optimizer.zero_grad(set_to_none=True)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        output = model(x)
                        loss = drawdown_loss(output, target)

                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite drawdown train loss at epoch {epoch}")

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.gradient_clip))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(x)
                    loss = drawdown_loss(output, target)

                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite drawdown train loss at epoch {epoch}")

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.gradient_clip))
                    optimizer.step()

                batch_loss = float(loss.detach().cpu())
                train_total += batch_loss
                train_batches += 1
                train_bar.set_postfix(loss=f"{batch_loss:.5f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

                del x, target, output, loss

            train_loss = train_total / max(train_batches, 1)

            model.eval()
            val_total = 0.0
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
                    loss = drawdown_loss(output, target)

                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite drawdown validation loss at epoch {epoch}")

                    batch_loss = float(loss.detach().cpu())
                    val_total += batch_loss
                    val_batches += 1
                    val_bar.set_postfix(loss=f"{batch_loss:.5f}")

                    del x, target, output, loss

            val_loss = val_total / max(val_batches, 1)

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
                save_latest_checkpoint(latest_path, model, optimizer, config, epoch, best_val_loss, history)

            if no_improve >= int(config.early_stop_patience):
                print(f"  Early stopping at epoch {epoch}")
                break

    finally:
        shutdown_dataloader(train_loader)
        shutdown_dataloader(val_loader)
        cleanup_memory()

    if not np.isfinite(best_val_loss):
        raise RuntimeError("Drawdown training failed: best validation loss is not finite.")

    if save_checkpoints:
        pd.DataFrame(history).to_csv(history_path, index=False)
        if best_path.exists():
            model = DrawdownRiskModel.load(best_path, config=config, device=str(device))

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
        "drawdown_threshold_10d": float(config.drawdown_threshold_10d),
        "drawdown_threshold_30d": float(config.drawdown_threshold_30d),
        "train_audit": train_ds.audit,
        "val_audit": val_ds.audit,
        "saved_checkpoints": bool(save_checkpoints),
    }

    if save_checkpoints:
        with open(summary_path, "w") as f:
            json.dump(json_safe(summary), f, indent=2)

    print(f"  Best val loss: {best_val_loss:.6f}")
    return model, summary


def train_drawdown_model(config: DrawdownConfig, chunk_id: int, fresh: bool = False) -> Dict[str, Any]:
    config.resolve_paths()
    configure_torch_runtime(config)
    seed_everything(config.seed)

    validate_required_inputs(config, chunk_id, ("train", "val"))

    device = resolve_device(config.device)
    label = CHUNK_CONFIG[chunk_id]["label"]

    print(f"\n{'=' * 72}")
    print(f"  DRAWDOWN RISK MODEL — {label}")
    print(f"{'=' * 72}")
    print(f"  Device: {device}")

    out_dir = model_dir_for(config, label, "")

    if fresh and out_dir.exists():
        print(f"  Fresh run requested. Removing: {out_dir}")
        shutil.rmtree(out_dir)

    close_panel = load_close_panel(config)

    model, summary = _train_model(
        config,
        chunk_id,
        close_panel,
        device,
        run_tag="",
        save_checkpoints=True,
        resume=not fresh,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_model.pt"

    if best_path.exists():
        shutil.copy2(best_path, out_dir / "final_model.pt")
    else:
        model.save(out_dir / "final_model.pt")

    frozen = out_dir / "model_freezed"
    frozen.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_dir / "final_model.pt", frozen / "model.pt")

    print(f"\n  Complete. Best val loss: {summary['best_val_loss']:.6f}")
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# HPO
# ═══════════════════════════════════════════════════════════════════════════════

def _hpo_objective(
    trial: "optuna.trial.Trial",
    base_config: DrawdownConfig,
    chunk_id: int,
    close_panel: pd.DataFrame,
) -> float:
    try:
        cfg = DrawdownConfig(**base_config.to_dict()).resolve_paths()

        cfg.lstm_hidden = trial.suggest_categorical("lstm_hidden", [32, 64, 96, 128])
        cfg.lstm_layers = trial.suggest_categorical("lstm_layers", [1, 2])
        cfg.attention_dim = trial.suggest_categorical("attention_dim", [32, 64, 128])
        cfg.dropout = trial.suggest_float("dropout", 0.05, 0.35)
        cfg.learning_rate = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        cfg.weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)
        cfg.batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])

        cfg.epochs = int(base_config.hpo_epochs)
        cfg.early_stop_patience = min(5, int(base_config.early_stop_patience))
        cfg.max_train_samples = int(base_config.hpo_max_train_samples)
        cfg.max_val_samples = int(base_config.hpo_max_val_samples)
        cfg.num_workers = 0
        cfg.persistent_workers = False
        cfg.run_tag = f"trial_{trial.number:04d}"
        cfg.save_checkpoints = False

        device = resolve_device(cfg.device)

        _, summary = _train_model(
            cfg,
            chunk_id,
            close_panel,
            device,
            run_tag=cfg.run_tag,
            save_checkpoints=False,
            resume=False,
        )

        value = float(summary["best_val_loss"])
        if not np.isfinite(value):
            return HPO_FAILURE_VALUE
        return value

    except Exception as exc:
        print(f"  Trial {trial.number} failed safely: {exc}")
        cleanup_memory()
        return HPO_FAILURE_VALUE


def run_hpo(config: DrawdownConfig, chunk_id: int, trials: int, fresh: bool = False) -> Dict[str, Any]:
    if not HAS_OPTUNA:
        raise ImportError("optuna required: pip install optuna")

    config.resolve_paths()
    configure_torch_runtime(config)
    seed_everything(config.seed)

    validate_required_inputs(config, chunk_id, ("train", "val"))

    label = CHUNK_CONFIG[chunk_id]["label"]

    print(f"\n{'=' * 72}")
    print(f"  DRAWDOWN HPO — {label} ({trials} trials)")
    print(f"{'=' * 72}")

    close_panel = load_close_panel(config)

    study_dir = Path(config.output_dir) / "codeResults" / "Drawdown"
    study_dir.mkdir(parents=True, exist_ok=True)

    db_path = study_dir / f"hpo_{label}.db"
    study_name = f"drawdown_{label}"

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
        lambda trial: _hpo_objective(trial, config, chunk_id, close_panel),
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
        raise RuntimeError("All Drawdown HPO trials failed. No best params saved.")

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


def apply_hpo_params_if_available(config: DrawdownConfig, chunk_id: int) -> DrawdownConfig:
    label = CHUNK_CONFIG[chunk_id]["label"]
    path = Path(config.output_dir) / "codeResults" / "Drawdown" / f"best_params_{label}.json"

    if not path.exists():
        print("  No HPO params found. Using default/current config.")
        return config

    with open(path) as f:
        result = json.load(f)

    value = float(result.get("value", HPO_FAILURE_VALUE))
    if not np.isfinite(value) or value >= HPO_FAILURE_VALUE:
        raise RuntimeError(f"Invalid HPO result file: {path}")

    params = result.get("params", {})

    if "lstm_hidden" in params:
        config.lstm_hidden = int(params["lstm_hidden"])
    if "lstm_layers" in params:
        config.lstm_layers = int(params["lstm_layers"])
    if "attention_dim" in params:
        config.attention_dim = int(params["attention_dim"])
    if "dropout" in params:
        config.dropout = float(params["dropout"])
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

def compute_drawdown_risk_score(pred: np.ndarray, config: DrawdownConfig) -> np.ndarray:
    dd10 = pred[:, 0]
    risk10 = pred[:, 1]
    rec10 = pred[:, 2]
    dd30 = pred[:, 4]
    risk30 = pred[:, 5]
    rec30 = pred[:, 6]

    sev10 = np.clip(dd10 / max(float(config.drawdown_threshold_10d) * 2.0, 1e-8), 0.0, 1.0)
    sev30 = np.clip(dd30 / max(float(config.drawdown_threshold_30d) * 2.0, 1e-8), 0.0, 1.0)

    score = (
        0.20 * sev10
        + 0.25 * sev30
        + 0.20 * risk10
        + 0.25 * risk30
        + 0.05 * rec10
        + 0.05 * rec30
    )
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def extract_attention_xai(
    model: DrawdownRiskModel,
    loader: DataLoader,
    config: DrawdownConfig,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()

    attention_sum = np.zeros(int(config.seq_len), dtype=np.float64)
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
            "relative_timestep": int(i - int(config.seq_len) + 1),
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
    model: DrawdownRiskModel,
    loader: DataLoader,
    config: DrawdownConfig,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()

    dim_chunks = []
    time_chunks = []
    n_xai = 0

    for batch in tqdm(loader, desc=f"  XAI-L2 gradients bs={loader.batch_size}", leave=False, unit="batch"):
        if n_xai >= int(config.xai_sample_size):
            break

        x = batch["x"].to(device, non_blocking=True)
        take = min(len(x), int(config.xai_sample_size) - n_xai)

        x_xai = x[:take].detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)

        with cudnn_disabled_if_cuda(device):
            out = model(x_xai)
            objective = (
                out["expected_drawdown_10d"].mean()
                + out["drawdown_risk_10d"].mean()
                + out["expected_drawdown_30d"].mean()
                + out["drawdown_risk_30d"].mean()
            )
            objective.backward()

        if x_xai.grad is not None:
            grad = x_xai.grad.detach().abs()
            dim_chunks.append(grad.mean(dim=(0, 1)).cpu().numpy())
            time_chunks.append(grad.mean(dim=2).mean(dim=0).cpu().numpy())

        n_xai += take

    if not dim_chunks:
        return {
            "embedding_dim_importance": [],
            "timestep_importance": [],
            "n_samples": 0,
        }

    dim_mean = np.stack(dim_chunks).mean(axis=0)
    time_mean = np.stack(time_chunks).mean(axis=0)

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
            "relative_timestep": int(i - int(config.seq_len) + 1),
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
    model: DrawdownRiskModel,
    dataset: DrawdownDataset,
    config: DrawdownConfig,
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

            no_last = x.clone()
            no_last[:, -1, :] = 0.0
            out_no_last = model(no_last)

            damp_recent = x.clone()
            damp_recent[:, -5:, :] *= 0.80
            out_damp = model(damp_recent)

            amp_recent = x.clone()
            amp_recent[:, -5:, :] *= 1.20
            out_amp = model(amp_recent)

        def pack(out: Dict[str, torch.Tensor]) -> Dict[str, float]:
            pred = out["predictions"].detach().cpu().numpy()
            risk_score = compute_drawdown_risk_score(pred, config)[0]
            return {
                "expected_drawdown_10d": float(pred[0, 0]),
                "drawdown_risk_10d": float(pred[0, 1]),
                "recovery_days_10d": float(pred[0, 2] * config.horizon_10d),
                "confidence_10d": float(pred[0, 3]),
                "expected_drawdown_30d": float(pred[0, 4]),
                "drawdown_risk_30d": float(pred[0, 5]),
                "recovery_days_30d": float(pred[0, 6] * config.horizon_30d),
                "confidence_30d": float(pred[0, 7]),
                "drawdown_risk_score": float(risk_score),
            }

        scenarios.append({
            "ticker": sample["ticker"],
            "date": sample["date"],
            "original": pack(original),
            "counterfactuals": [
                {"condition": "Most recent embedding removed", "scores": pack(out_no_last)},
                {"condition": "Recent 5 embeddings dampened by 20%", "scores": pack(out_damp)},
                {"condition": "Recent 5 embeddings amplified by 20%", "scores": pack(out_amp)},
            ],
        })

    return scenarios


def predict_with_xai(config: DrawdownConfig, chunk_id: int, split: str) -> Dict[str, Any]:
    config.resolve_paths()
    configure_torch_runtime(config)
    seed_everything(config.seed)

    validate_required_inputs(config, chunk_id, (split,))

    device = resolve_device(config.device)
    cleanup_memory()

    label = CHUNK_CONFIG[chunk_id]["label"]
    model_dir = model_dir_for(config, label, "")
    model_path = model_dir / "best_model.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"No trained Drawdown model found: {model_path}")

    model = DrawdownRiskModel.load(model_path, config=config, device=str(device))
    config = model.config.resolve_paths()
    model.eval()

    close_panel = load_close_panel(config)

    dataset = DrawdownDataset(
        config,
        split,
        chunk_id,
        close_panel,
        max_samples=int(config.max_predict_samples),
    )

    loader = make_loader(dataset, config, train=False)

    results_dir = Path(config.output_dir) / "results" / "Drawdown"
    xai_dir = results_dir / "xai"
    results_dir.mkdir(parents=True, exist_ok=True)
    xai_dir.mkdir(parents=True, exist_ok=True)

    pred_chunks: List[pd.DataFrame] = []

    with torch.no_grad():
        bar = tqdm(loader, desc=f"  Predict {label}_{split} bs={config.batch_size}", leave=False, unit="batch")

        for batch in bar:
            x = batch["x"].to(device, non_blocking=True)
            target = batch["target"].cpu().numpy()

            out = model(x)
            pred = out["predictions"].detach().cpu().numpy()
            risk_score = compute_drawdown_risk_score(pred, config)

            pred_chunks.append(pd.DataFrame({
                "ticker": list(batch["ticker"]),
                "date": list(batch["date"]),

                "expected_drawdown_10d": pred[:, 0],
                "drawdown_risk_10d": pred[:, 1],
                "recovery_days_10d": pred[:, 2] * float(config.horizon_10d),
                "confidence_10d": pred[:, 3],

                "expected_drawdown_30d": pred[:, 4],
                "drawdown_risk_30d": pred[:, 5],
                "recovery_days_30d": pred[:, 6] * float(config.horizon_30d),
                "confidence_30d": pred[:, 7],

                "drawdown_risk_score": risk_score,

                "target_expected_drawdown_10d": target[:, 0],
                "target_drawdown_risk_10d": target[:, 1],
                "target_recovery_days_10d": target[:, 2] * float(config.horizon_10d),

                "target_expected_drawdown_30d": target[:, 4],
                "target_drawdown_risk_30d": target[:, 5],
                "target_recovery_days_30d": target[:, 6] * float(config.horizon_30d),
            }))

            del x, out

    predictions = pd.concat(pred_chunks, ignore_index=True) if pred_chunks else pd.DataFrame()

    attention_loader = make_loader(dataset, config, train=False)
    try:
        attention_xai = extract_attention_xai(model, attention_loader, config, device)
    finally:
        shutdown_dataloader(attention_loader)

    gradient_loader = make_loader(dataset, config, train=False)
    try:
        gradient_xai = extract_gradient_xai(model, gradient_loader, config, device)
    finally:
        shutdown_dataloader(gradient_loader)

    counterfactuals = generate_counterfactual_xai(model, dataset, config, device)

    xai = {
        "module": "DrawdownRisk",
        "chunk": label,
        "split": split,
        "level1_attention": attention_xai,
        "level2_gradient_importance": gradient_xai,
        "level3_counterfactuals": counterfactuals,
        "dataset_audit": dataset.audit,
        "target_columns": TARGET_COLUMNS,
        "explanation_summary": {
            "plain_english": (
                "The Drawdown Risk Model predicts continuous downside path risk over 10-day and 30-day horizons. "
                "It estimates drawdown severity, soft threshold risk, recovery delay, and confidence. Attention XAI "
                "shows which historical embedding timesteps mattered most. Gradient XAI identifies influential "
                "embedding dimensions and timesteps. Counterfactual XAI shows how downside risk changes when recent "
                "technical context is removed, dampened, or amplified."
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
        json.dump(json_safe(attention_xai), f, indent=2)

    if gradient_xai["embedding_dim_importance"]:
        pd.DataFrame(gradient_xai["embedding_dim_importance"]).to_csv(dim_importance_path, index=False)

    if gradient_xai["timestep_importance"]:
        pd.DataFrame(gradient_xai["timestep_importance"]).to_csv(timestep_importance_path, index=False)

    with open(counterfactual_path, "w") as f:
        json.dump(json_safe(counterfactuals), f, indent=2)

    compact_xai = {
        "module": "DrawdownRisk",
        "chunk": label,
        "split": split,
        "level1_attention_top_timesteps": attention_xai["top_timesteps"],
        "level2_top_embedding_dims": gradient_xai["embedding_dim_importance"][:20],
        "level2_top_timesteps": gradient_xai["timestep_importance"][:10],
        "level3_counterfactuals": counterfactuals,
        "dataset_audit": dataset.audit,
        "explanation_summary": xai["explanation_summary"],
    }

    with open(xai_summary_path, "w") as f:
        json.dump(json_safe(compact_xai), f, indent=2)

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
# INSPECT / SMOKE
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_inspect(config: DrawdownConfig) -> None:
    config.resolve_paths()

    print("=" * 72)
    print("DRAWDOWN RISK MODEL — DATA INSPECTION")
    print("=" * 72)

    temporal_dir = Path(config.temporal_dir)

    if not temporal_dir.exists():
        print(f"  Missing temporal directory: {temporal_dir}")
    else:
        for file in sorted(temporal_dir.glob("chunk*_embeddings.npy")):
            arr = np.load(file, mmap_mode="r")
            sample = np.asarray(arr[: min(2048, len(arr))])
            print(f"  {file.name}: {arr.shape}, sample_finite={finite_ratio_np(sample):.6f}")

        for file in sorted(temporal_dir.glob("chunk*_manifest.csv")):
            df = pd.read_csv(file, nrows=5)
            print(f"  {file.name}: sample columns={list(df.columns)}")
            print(df.head(3).to_string(index=False))

    print(f"\n  OHLCV path: {config.ohlcv_path}")
    if Path(config.ohlcv_path).exists():
        sample = pd.read_csv(config.ohlcv_path, nrows=5)
        print(f"  OHLCV sample columns: {list(sample.columns)}")
        print(sample.to_string(index=False))

    close_panel = load_close_panel(config)
    print(f"\n  Close panel: {close_panel.shape[0]:,} dates × {close_panel.shape[1]:,} tickers")
    print(f"  Date range: {close_panel.index.min().date()} → {close_panel.index.max().date()}")
    print(f"  Close finite ratio: {finite_ratio_np(close_panel.values):.6f}")


def cmd_smoke(config: DrawdownConfig) -> None:
    config.resolve_paths()
    configure_torch_runtime(config)
    seed_everything(config.seed)

    device = resolve_device(config.device)

    print("=" * 72)
    print("DRAWDOWN RISK MODEL — SMOKE TEST")
    print("=" * 72)

    batch = 128
    x = torch.randn(batch, int(config.seq_len), int(config.input_dim), device=device)
    target = torch.rand(batch, 8, device=device)

    model = DrawdownRiskModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    out = model(x)
    loss = drawdown_loss(out, target)

    if not torch.isfinite(loss):
        raise RuntimeError("Smoke failed: non-finite loss.")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.gradient_clip))
    optimizer.step()

    model.eval()
    x_xai = x[:16].detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)

    with cudnn_disabled_if_cuda(device):
        out_xai = model(x_xai)
        score = (
            out_xai["expected_drawdown_10d"].mean()
            + out_xai["drawdown_risk_10d"].mean()
            + out_xai["expected_drawdown_30d"].mean()
            + out_xai["drawdown_risk_30d"].mean()
        )
        score.backward()

    if x_xai.grad is None or not torch.isfinite(x_xai.grad).all():
        raise RuntimeError("Smoke failed: invalid XAI gradient.")

    print("SMOKE TEST PASSED")
    print(f"  loss={float(loss.detach().cpu()):.6f}")
    print(f"  predictions_shape={tuple(out['predictions'].shape)}")
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


def config_from_args(args: argparse.Namespace) -> DrawdownConfig:
    config = DrawdownConfig()

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
    parser = argparse.ArgumentParser(description="Drawdown Risk Model — Continuous Dual-Horizon Risk")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("inspect", help="Inspect input files and close-price panel")
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

    p = sub.add_parser("train-best-all", help="Train all chunks with available temporal embeddings")
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
        cmd_smoke(config)

    elif args.command == "hpo":
        run_hpo(config, args.chunk, int(args.trials), fresh=bool(args.fresh))

    elif args.command == "train-best":
        config = apply_hpo_params_if_available(config, args.chunk)
        train_drawdown_model(config, args.chunk, fresh=bool(args.fresh))

    elif args.command == "train-best-all":
        for chunk_id in [1, 2, 3]:
            validate_required_inputs(config, chunk_id, ("train", "val"))
            chunk_config = apply_hpo_params_if_available(config, chunk_id)
            train_drawdown_model(chunk_config, chunk_id, fresh=bool(args.fresh))

    elif args.command == "predict":
        result = predict_with_xai(config, args.chunk, args.split)
        print(f"  Returned keys: {list(result.keys())}")


if __name__ == "__main__":
    main()


# ── Run instructions ─────────────────────────────────────────────────────────
# Compile:
# python -m py_compile code/riskEngine/drawdown.py
#
# Inspect:
# python code/riskEngine/drawdown.py inspect --repo-root .
#
# Smoke:
# python code/riskEngine/drawdown.py smoke --repo-root . --device cuda
#
# Small HPO:
# python code/riskEngine/drawdown.py hpo --repo-root . --chunk 1 --trials 3 --device cuda --fresh
#
# Full HPO:
# python code/riskEngine/drawdown.py hpo --repo-root . --chunk 1 --trials 40 --device cuda --fresh
#
# Train:
# python code/riskEngine/drawdown.py train-best --repo-root . --chunk 1 --device cuda --fresh
#
# Predict validation:
# python code/riskEngine/drawdown.py predict --repo-root . --chunk 1 --split val --device cuda
#
# Predict test:
# python code/riskEngine/drawdown.py predict --repo-root . --chunk 1 --split test --device cuda
#
# Small debug training:
# python code/riskEngine/drawdown.py train-best --repo-root . --chunk 1 --device cuda --epochs 2 --max-train-samples 50000 --max-val-samples 10000 --fresh