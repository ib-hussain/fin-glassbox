#!/usr/bin/env python3
"""
code/riskEngine/volatility.py

Volatility Estimation Model — GARCH(1,1) + MLP Hybrid
=====================================================
Project: fin-glassbox — Explainable Distributed Deep Learning Framework

This file is designed to work both as:
    1. an importable module for the integrated fin-glassbox system, and
    2. an independently runnable CLI file for inspect, smoke, HPO, training, and prediction.

Core guarantees:
    - chronological chunking, no future leakage in input features,
    - robust finite-value cleaning before model training,
    - HPO trial isolation from production checkpoints,
    - progress bars with batch size and running losses,
    - XAI objects returned by prediction functions and saved to disk,
    - production checkpoint contains model config for reliable reloading.

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

REGIME_LABELS = {0: "low", 1: "medium", 2: "high"}

# Optuna RDB/SQLite can behave badly with inf/nan objective values.
# Failed trials must return a large finite penalty instead.
HPO_FAILURE_VALUE = 1_000_000_000.0

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VolatilityConfig:
    repo_root: str = ""
    output_dir: str = "outputs"
    embeddings_dir: str = "outputs/embeddings/TemporalEncoder"
    features_path: str = "data/yFinance/processed/features_temporal.csv"
    returns_path: str = "data/yFinance/processed/returns_panel_wide.csv"

    # Architecture
    input_dim: int = 256
    hidden_dims: List[int] = field(default_factory=lambda: [64])
    dropout: float = 0.2

    # Training
    batch_size: int = 512
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stop_patience: int = 15
    gradient_clip: float = 1.0

    # Volatility target configuration
    vol_horizons: List[int] = field(default_factory=lambda: [10, 30])
    seq_len: int = 30
    min_target_vol: float = 0.01
    max_target_vol: float = 5.0
    fallback_vol: float = 0.30
    recent_vol_lookback: int = 21

    # HPO
    hpo_trials: int = 40
    hpo_n_startup: int = 10
    hpo_epochs: int = 8
    hpo_max_train_samples: int = 200_000
    hpo_max_val_samples: int = 50_000

    # XAI
    xai_sample_size: int = 500
    xai_counterfactual_scenarios: int = 5

    # System
    device: str = "cuda"
    seed: int = 42
    mixed_precision: bool = True
    num_workers: int = 6
    cpu_threads: int = 6
    deterministic: bool = False
    persistent_workers: bool = True

    # Runtime control
    max_train_samples: int = 0
    max_val_samples: int = 0
    save_checkpoints: bool = True
    run_tag: str = ""

    def resolve_paths(self) -> "VolatilityConfig":
        if self.repo_root:
            root = Path(self.repo_root)
            for attr in ["embeddings_dir", "features_path", "returns_path", "output_dir"]:
                val = getattr(self, attr)
                if val and not Path(val).is_absolute():
                    setattr(self, attr, str(root / val))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
# RUNTIME UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def configure_torch_runtime(config: VolatilityConfig) -> None:
    torch.set_num_threads(max(1, int(config.cpu_threads)))
    torch.set_num_interop_threads(max(1, min(2, int(config.cpu_threads))))

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = not config.deterministic
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


def safe_clip_vol(array: np.ndarray, config: VolatilityConfig) -> np.ndarray:
    arr = sanitize_np_array(array, nan=config.fallback_vol, posinf=config.max_target_vol, neginf=config.min_target_vol)
    arr = np.clip(arr, config.min_target_vol, config.max_target_vol)
    return arr.astype(np.float32)


def load_config_from_checkpoint_dict(raw: Dict[str, Any], fallback: Optional[VolatilityConfig] = None) -> VolatilityConfig:
    base = fallback.to_dict() if fallback is not None else VolatilityConfig().to_dict()
    ckpt_cfg = raw.get("config", {})
    if isinstance(ckpt_cfg, dict):
        base.update(ckpt_cfg)

    valid = {f.name for f in dataclass_fields(VolatilityConfig)}
    filtered = {k: v for k, v in base.items() if k in valid}
    return VolatilityConfig(**filtered).resolve_paths()


# ═══════════════════════════════════════════════════════════════════════════════
# MANIFEST GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_manifest_exists(config: VolatilityConfig, chunk_id: int, split: str) -> Path:
    label = CHUNK_CONFIG[chunk_id]["label"]
    emb_dir = Path(config.embeddings_dir)
    manifest_path = emb_dir / f"{label}_{split}_manifest.csv"
    emb_path = emb_dir / f"{label}_{split}_embeddings.npy"

    if manifest_path.exists():
        return manifest_path

    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {emb_path}. Run TemporalEncoder embedding first.")

    print(f"  Manifest missing for {label}_{split}. Auto-generating...")
    _build_manifest(config, chunk_id, split, emb_path, manifest_path)
    return manifest_path


def _build_manifest(config: VolatilityConfig, chunk_id: int, split: str, emb_path: Path, manifest_path: Path) -> None:
    years = CHUNK_CONFIG[chunk_id][split]
    label = CHUNK_CONFIG[chunk_id]["label"]

    embeddings = np.load(emb_path, mmap_mode="r")
    features_df = pd.read_csv(config.features_path, dtype={"ticker": str}, parse_dates=["date"])
    missing = [c for c in FEATURE_NAMES if c not in features_df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns in {config.features_path}: {missing}")

    features_df["year"] = features_df["date"].dt.year
    df = features_df[(features_df["year"] >= years[0]) & (features_df["year"] <= years[1])].copy()
    df = df.sort_values(["ticker", "date"])

    records = []
    grouped = df.groupby("ticker", sort=True)
    for ticker, group in tqdm(grouped, desc=f"  Manifest {label}_{split}", leave=False):
        dates = group["date"].values
        if len(dates) < config.seq_len:
            continue
        for i in range(config.seq_len - 1, len(dates)):
            records.append({"ticker": ticker, "date": str(dates[i])[:10]})

    n = min(len(records), len(embeddings))
    if len(records) != len(embeddings):
        print(f"    WARNING: Manifest sequence count {len(records):,} != embeddings {len(embeddings):,}. Truncating to {n:,}.")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records[:n]).to_csv(manifest_path, index=False)
    print(f"    Saved: {manifest_path.name} ({n:,} rows)")


def required_embedding_files(config: VolatilityConfig, chunk_id: int, splits: Tuple[str, ...] = ("train", "val", "test")) -> List[Path]:
    """Return required TemporalEncoder embedding files for a chunk."""
    label = CHUNK_CONFIG[chunk_id]["label"]
    emb_dir = Path(config.embeddings_dir)
    return [emb_dir / f"{label}_{split}_embeddings.npy" for split in splits]


def validate_required_embeddings(
    config: VolatilityConfig,
    chunk_id: int,
    splits: Tuple[str, ...] = ("train", "val", "test"),
) -> None:
    """Fail before HPO/training starts if required temporal embeddings are missing.

    This prevents Optuna from recording meaningless failed trials and prevents
    command chains from continuing into train/predict with incomplete inputs.
    """
    missing = [p for p in required_embedding_files(config, chunk_id, splits) if not p.exists()]

    if missing:
        label = CHUNK_CONFIG[chunk_id]["label"]
        missing_text = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            f"Temporal Encoder embeddings are missing for {label}.\n"
            f"Volatility depends on TemporalEncoder embeddings, so this chunk cannot run yet.\n"
            f"Missing files:\n{missing_text}\n\n"
            f"Run/finish TemporalEncoder embedding generation for {label} first, then rerun volatility."
        )


def ensure_all_manifests(config: VolatilityConfig, chunk_id: int) -> None:
    """Generate manifests only after confirming all required embeddings exist."""
    validate_required_embeddings(config, chunk_id, ("train", "val", "test"))

    for split in ["train", "val", "test"]:
        ensure_manifest_exists(config, chunk_id, split)


# ═══════════════════════════════════════════════════════════════════════════════
# SIMPLE, STABLE GARCH(1,1)-STYLE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleGARCH:
    """Stable GARCH(1,1)-style volatility forecaster.

    This implementation intentionally prioritizes numerical stability over complex
    likelihood optimization. The previous gradient-like update could produce NaN
    parameters, which poisoned MLP training.
    """

    def __init__(self) -> None:
        self.omega: Optional[float] = None
        self.alpha: Optional[float] = None
        self.beta: Optional[float] = None
        self.mu: Optional[float] = None
        self.fitted: bool = False

    def fit(self, returns: np.ndarray) -> bool:
        rets = np.asarray(returns, dtype=np.float64)
        rets = rets[np.isfinite(rets)]
        if len(rets) < 63:
            return False

        rets = np.clip(rets, -0.75, 0.75)
        self.mu = float(np.mean(rets))
        centered = rets - self.mu
        var = float(np.var(centered))

        if not np.isfinite(var) or var <= 1e-12:
            return False

        # Stable starting parameters. Persistence is high but < 1.
        alpha = 0.08
        beta = 0.90
        omega = max(var * (1.0 - alpha - beta), 1e-10)

        self.omega = float(omega)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.fitted = True
        return True

    def forecast(self, returns: np.ndarray, horizon: int) -> float:
        rets = np.asarray(returns, dtype=np.float64)
        rets = rets[np.isfinite(rets)]
        if len(rets) < 5:
            return 0.30

        if not self.fitted or self.omega is None or self.alpha is None or self.beta is None or self.mu is None:
            recent = rets[-min(21, len(rets)):]
            vol = float(np.std(recent) * math.sqrt(252))
            return float(np.clip(vol if np.isfinite(vol) else 0.30, 0.01, 5.0))

        rets = np.clip(rets[-252:], -0.75, 0.75)
        centered = rets - self.mu
        s2 = float(np.var(centered))
        if len(centered) >= 2:
            s2 = self.omega + self.alpha * float(centered[-1] ** 2) + self.beta * max(s2, 1e-10)

        for _ in range(max(1, int(horizon))):
            s2 = self.omega + (self.alpha + self.beta) * max(s2, 1e-10)
            if not np.isfinite(s2):
                s2 = 0.30 ** 2 / 252.0
                break

        vol = math.sqrt(max(float(s2), 1e-10)) * math.sqrt(252)
        return float(np.clip(vol, 0.01, 5.0))

    def to_dict(self) -> Dict[str, Optional[float]]:
        persistence = None
        if self.fitted and self.alpha is not None and self.beta is not None:
            persistence = float(self.alpha + self.beta)
        return {
            "omega": self.omega,
            "alpha": self.alpha,
            "beta": self.beta,
            "mu": self.mu,
            "persistence": persistence,
            "fitted": self.fitted,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class VolatilityMLP(nn.Module):
    """MLP that adjusts stable GARCH/recent-vol forecasts using temporal embeddings."""

    def __init__(self, config: VolatilityConfig) -> None:
        super().__init__()
        self.config = config

        layers: List[nn.Module] = []
        in_dim = int(config.input_dim) + 2

        for hidden in config.hidden_dims:
            hidden = int(hidden)
            layers.extend([
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(float(config.dropout)),
            ])
            in_dim = hidden

        self.shared = nn.Sequential(*layers) if layers else nn.Identity()
        last_dim = int(config.hidden_dims[-1]) if config.hidden_dims else in_dim

        self.head_vol10 = nn.Linear(last_dim, 1)
        self.head_vol30 = nn.Linear(last_dim, 1)
        self.head_regime = nn.Linear(last_dim, 3)
        self.head_conf = nn.Linear(last_dim, 1)

    def forward(self, emb: torch.Tensor, garch_vol: torch.Tensor, recent_vol: torch.Tensor) -> Dict[str, torch.Tensor]:
        emb = torch.nan_to_num(emb.float(), nan=0.0, posinf=0.0, neginf=0.0)
        garch_vol = torch.nan_to_num(garch_vol.float(), nan=0.30, posinf=5.0, neginf=0.01).clamp(0.01, 5.0)
        recent_vol = torch.nan_to_num(recent_vol.float(), nan=0.30, posinf=5.0, neginf=0.01).clamp(0.01, 5.0)

        x = torch.cat([emb, garch_vol.unsqueeze(-1), recent_vol.unsqueeze(-1)], dim=-1)
        shared = self.shared(x)

        regime_logits = self.head_regime(shared)
        vol_10d = F.softplus(self.head_vol10(shared)).squeeze(-1).clamp(0.001, 10.0)
        vol_30d = F.softplus(self.head_vol30(shared)).squeeze(-1).clamp(0.001, 10.0)

        return {
            "vol_10d": vol_10d,
            "vol_30d": vol_30d,
            "regime_logits": regime_logits,
            "regime_probs": F.softmax(regime_logits, dim=-1),
            "confidence": torch.sigmoid(self.head_conf(shared)).squeeze(-1),
        }

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": self.config.to_dict(),
        }, path)

    @classmethod
    def load(cls, path: Union[str, Path], config: Optional[VolatilityConfig] = None, device: str = "cpu") -> "VolatilityMLP":
        path = Path(path)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = load_config_from_checkpoint_dict(ckpt, fallback=config)
        model = cls(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        return model


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class VolatilityDataset(Dataset):
    """Pairs temporal embeddings with realized volatility targets.

    Input:
        embedding[ticker, date] from TemporalEncoder
        garch_vol from training-period fitted GARCH-style model
        recent_vol from historical returns before or at date

    Target:
        future realized volatility over 10d/30d using returns after the manifest date.
    """

    def __init__(
        self,
        config: VolatilityConfig,
        split: str,
        chunk_id: int,
        returns_df: pd.DataFrame,
        garch_models: Optional[Dict[str, SimpleGARCH]] = None,
        max_samples: int = 0,
        label_suffix: str = "",
    ) -> None:
        self.config = config
        self.split = split
        self.chunk_id = chunk_id
        self.label = CHUNK_CONFIG[chunk_id]["label"]
        self.label_suffix = label_suffix

        manifest_path = ensure_manifest_exists(config, chunk_id, split)
        emb_path = Path(config.embeddings_dir) / f"{self.label}_{split}_embeddings.npy"
        if not emb_path.exists():
            raise FileNotFoundError(f"Embedding file missing: {emb_path}")

        manifest = pd.read_csv(manifest_path, dtype={"ticker": str})
        embeddings = np.load(emb_path, mmap_mode="r")

        n = min(len(manifest), len(embeddings))
        if max_samples and max_samples > 0:
            n = min(n, int(max_samples))

        self.manifest = manifest.iloc[:n].reset_index(drop=True)
        self.embeddings = embeddings
        self.n = n

        returns_df = returns_df.sort_index()
        returns_df = returns_df.apply(pd.to_numeric, errors="coerce")
        returns_df = returns_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.returns_data = returns_df
        self._ticker_ret_cache: Dict[str, np.ndarray] = {}
        self._date_to_idx = {str(d)[:10]: i for i, d in enumerate(returns_df.index)}

        print(f"  {self.label}_{split}{label_suffix}: {self.n:,} samples, {self.manifest['ticker'].nunique()} tickers")

        self.targets = self._compute_targets()
        self.recent_vol = self._compute_recent_vol()
        self.garch_forecasts = self._compute_garch_forecasts(garch_models)

        self.audit = self._audit_arrays()
        print(
            f"    finite ratios | emb(sample)={self.audit['embedding_sample_finite_ratio']:.6f}, "
            f"target={self.audit['target_finite_ratio']:.6f}, "
            f"garch={self.audit['garch_finite_ratio']:.6f}, "
            f"recent={self.audit['recent_finite_ratio']:.6f}"
        )

    def _get_returns(self, ticker: str) -> np.ndarray:
        if ticker not in self._ticker_ret_cache:
            if ticker in self.returns_data.columns:
                arr = self.returns_data[ticker].values.astype(np.float64)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                self._ticker_ret_cache[ticker] = arr
            else:
                self._ticker_ret_cache[ticker] = np.array([], dtype=np.float64)
        return self._ticker_ret_cache[ticker]

    def _realized_vol_from_slice(self, rets: np.ndarray, start: int, end: int) -> float:
        window = rets[start:end]
        window = window[np.isfinite(window)]
        if len(window) < 3:
            return self.config.fallback_vol
        vol = float(np.std(window) * math.sqrt(252))
        if not np.isfinite(vol):
            return self.config.fallback_vol
        return float(np.clip(vol, self.config.min_target_vol, self.config.max_target_vol))

    def _compute_targets(self) -> np.ndarray:
        targets = np.zeros((self.n, len(self.config.vol_horizons)), dtype=np.float32)

        for h_idx, h in enumerate(self.config.vol_horizons):
            iterator = tqdm(range(self.n), desc=f"  Targets {self.label}_{self.split} h={h}", leave=False)
            for i in iterator:
                row = self.manifest.iloc[i]
                ticker = row["ticker"]
                date_str = str(row["date"])[:10]
                rets = self._get_returns(ticker)
                idx = self._date_to_idx.get(date_str)

                # Use strictly future returns as target: idx+1 through idx+h.
                if idx is None or len(rets) == 0 or idx + 1 >= len(rets):
                    targets[i, h_idx] = self.config.fallback_vol
                    continue

                start = idx + 1
                end = min(len(rets), idx + 1 + int(h))
                targets[i, h_idx] = self._realized_vol_from_slice(rets, start, end)

        return safe_clip_vol(targets, self.config)

    def _compute_recent_vol(self) -> np.ndarray:
        recent = np.zeros((self.n,), dtype=np.float32)
        lookback = int(self.config.recent_vol_lookback)

        iterator = tqdm(range(self.n), desc=f"  Recent vol {self.label}_{self.split}", leave=False)
        for i in iterator:
            row = self.manifest.iloc[i]
            ticker = row["ticker"]
            date_str = str(row["date"])[:10]
            rets = self._get_returns(ticker)
            idx = self._date_to_idx.get(date_str)

            if idx is None or len(rets) == 0:
                recent[i] = self.config.fallback_vol
                continue

            start = max(0, idx - lookback + 1)
            end = idx + 1
            recent[i] = self._realized_vol_from_slice(rets, start, end)

        return safe_clip_vol(recent, self.config)

    def _compute_garch_forecasts(self, garch_models: Optional[Dict[str, SimpleGARCH]]) -> np.ndarray:
        forecasts = np.zeros((self.n, len(self.config.vol_horizons)), dtype=np.float32)

        if garch_models is None:
            forecasts[:, :] = self.recent_vol[:, None]
            return safe_clip_vol(forecasts, self.config)

        iterator = tqdm(range(self.n), desc=f"  GARCH forecasts {self.label}_{self.split}", leave=False)
        for i in iterator:
            row = self.manifest.iloc[i]
            ticker = row["ticker"]
            date_str = str(row["date"])[:10]
            rets = self._get_returns(ticker)
            idx = self._date_to_idx.get(date_str)

            if ticker not in garch_models or idx is None or idx < 10 or len(rets) == 0:
                forecasts[i, :] = self.recent_vol[i]
                continue

            history = rets[: idx + 1]
            for j, h in enumerate(self.config.vol_horizons):
                forecasts[i, j] = garch_models[ticker].forecast(history, int(h))

        return safe_clip_vol(forecasts, self.config)

    def _audit_arrays(self) -> Dict[str, float]:
        sample_n = min(self.n, 2048)
        if sample_n > 0:
            emb_sample = np.asarray(self.embeddings[:sample_n])
        else:
            emb_sample = np.array([], dtype=np.float32)

        return {
            "embedding_sample_finite_ratio": finite_ratio_np(emb_sample),
            "target_finite_ratio": finite_ratio_np(self.targets),
            "garch_finite_ratio": finite_ratio_np(self.garch_forecasts),
            "recent_finite_ratio": finite_ratio_np(self.recent_vol),
            "target_min": float(np.nanmin(self.targets)) if self.targets.size else 0.0,
            "target_max": float(np.nanmax(self.targets)) if self.targets.size else 0.0,
            "garch_min": float(np.nanmin(self.garch_forecasts)) if self.garch_forecasts.size else 0.0,
            "garch_max": float(np.nanmax(self.garch_forecasts)) if self.garch_forecasts.size else 0.0,
        }

    def __len__(self) -> int:
        return int(self.n)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        emb = np.asarray(self.embeddings[idx], dtype=np.float32).copy()
        emb = sanitize_np_array(emb, nan=0.0, posinf=0.0, neginf=0.0)

        if emb.shape[0] != self.config.input_dim:
            raise ValueError(f"Embedding dim mismatch: expected {self.config.input_dim}, got {emb.shape[0]} at idx={idx}")

        return {
            "emb": torch.from_numpy(emb),
            "ticker": str(self.manifest.iloc[idx]["ticker"]),
            "date": str(self.manifest.iloc[idx]["date"])[:10],
            "target": torch.from_numpy(self.targets[idx].astype(np.float32)),
            "garch_vol": torch.tensor(float(self.garch_forecasts[idx, 0]), dtype=torch.float32),
            "recent_vol": torch.tensor(float(self.recent_vol[idx]), dtype=torch.float32),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS AND LOSSES
# ═══════════════════════════════════════════════════════════════════════════════

def make_loader(dataset: Dataset, config: VolatilityConfig, train: bool) -> DataLoader:
    workers = max(0, int(config.num_workers))
    kwargs = {
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


def volatility_loss(out: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    target = torch.nan_to_num(target.float(), nan=0.30, posinf=5.0, neginf=0.01).clamp(0.01, 5.0)

    pred10 = torch.nan_to_num(out["vol_10d"].float(), nan=0.30, posinf=5.0, neginf=0.01).clamp(0.001, 10.0)
    pred30 = torch.nan_to_num(out["vol_30d"].float(), nan=0.30, posinf=5.0, neginf=0.01).clamp(0.001, 10.0)

    # Log-space loss is much more stable for volatility because volatility is positive and scale-sensitive.
    loss10 = F.smooth_l1_loss(torch.log(pred10), torch.log(target[:, 0]))
    loss30 = F.smooth_l1_loss(torch.log(pred30), torch.log(target[:, 1]))
    return loss10 + 0.5 * loss30


def tensors_are_finite(*tensors: torch.Tensor) -> bool:
    return all(torch.isfinite(t).all().item() for t in tensors)


# ═══════════════════════════════════════════════════════════════════════════════
# GARCH FITTING
# ═══════════════════════════════════════════════════════════════════════════════

def fit_garch_models(config: VolatilityConfig, chunk_id: int) -> Dict[str, SimpleGARCH]:
    returns_df = pd.read_csv(config.returns_path, index_col=0, parse_dates=True)
    returns_df = returns_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    train_years = CHUNK_CONFIG[chunk_id]["train"]
    train_returns = returns_df[
        (returns_df.index.year >= train_years[0]) &
        (returns_df.index.year <= train_years[1])
    ]

    garch_models: Dict[str, SimpleGARCH] = {}
    tickers = list(train_returns.columns)

    for ticker in tqdm(tickers, desc="  GARCH fit", unit="ticker"):
        rets = train_returns[ticker].values.astype(np.float64)
        rets = rets[np.isfinite(rets)]
        if len(rets) < 252:
            continue
        garch = SimpleGARCH()
        if garch.fit(rets):
            garch_models[ticker] = garch

    print(f"  Fitted: {len(garch_models)}/{len(tickers)}")
    return garch_models


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def model_dir_for(config: VolatilityConfig, chunk_label: str, run_tag: str = "") -> Path:
    base = Path(config.output_dir) / "models" / "Volatility"
    if run_tag:
        return base / "_hpo_trials" / run_tag / chunk_label
    return base / chunk_label


def _train_mlp(
    config: VolatilityConfig,
    chunk_id: int,
    garch_models: Dict[str, SimpleGARCH],
    returns_df: pd.DataFrame,
    device: torch.device,
    *,
    run_tag: str = "",
    save_checkpoints: bool = True,
) -> Tuple[VolatilityMLP, Dict[str, Any]]:
    label = CHUNK_CONFIG[chunk_id]["label"]

    train_limit = int(config.max_train_samples) if config.max_train_samples else 0
    val_limit = int(config.max_val_samples) if config.max_val_samples else 0

    train_ds = VolatilityDataset(
        config,
        "train",
        chunk_id,
        returns_df,
        garch_models,
        max_samples=train_limit,
        label_suffix=f"_{run_tag}" if run_tag else "",
    )
    val_ds = VolatilityDataset(
        config,
        "val",
        chunk_id,
        returns_df,
        garch_models,
        max_samples=val_limit,
        label_suffix=f"_{run_tag}" if run_tag else "",
    )

    train_loader = make_loader(train_ds, config, train=True)
    val_loader = make_loader(val_ds, config, train=False)

    model = VolatilityMLP(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate), weight_decay=float(config.weight_decay))
    use_amp = bool(config.mixed_precision and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out_dir = model_dir_for(config, label, run_tag if not save_checkpoints else "")
    out_dir.mkdir(parents=True, exist_ok=True)

    best_path = out_dir / "best_model.pt"
    latest_path = out_dir / "latest_model.pt"
    history_path = out_dir / "training_history.csv"
    summary_path = out_dir / "training_summary.json"

    best_val_loss = float("inf")
    no_improve = 0
    history: List[Dict[str, Any]] = []
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,} | batch_size={config.batch_size} | params={total_params:,}")

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
                emb = batch["emb"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                garch = batch["garch_vol"].to(device, non_blocking=True)
                recent = batch["recent_vol"].to(device, non_blocking=True)

                if not tensors_are_finite(emb, target, garch, recent):
                    raise RuntimeError("Non-finite tensor detected before forward pass. Dataset sanitization failed.")

                optimizer.zero_grad(set_to_none=True)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        out = model(emb, garch, recent)
                        loss = volatility_loss(out, target)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite training loss detected at epoch {epoch}.")
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.gradient_clip))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out = model(emb, garch, recent)
                    loss = volatility_loss(out, target)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite training loss detected at epoch {epoch}.")
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
                    emb = batch["emb"].to(device, non_blocking=True)
                    target = batch["target"].to(device, non_blocking=True)
                    garch = batch["garch_vol"].to(device, non_blocking=True)
                    recent = batch["recent_vol"].to(device, non_blocking=True)

                    if not tensors_are_finite(emb, target, garch, recent):
                        raise RuntimeError("Non-finite tensor detected during validation.")

                    out = model(emb, garch, recent)
                    loss = volatility_loss(out, target)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite validation loss detected at epoch {epoch}.")

                    batch_loss = float(loss.detach().cpu())
                    val_loss_sum += batch_loss
                    val_batches += 1
                    val_bar.set_postfix(loss=f"{batch_loss:.5f}")

            val_loss = val_loss_sum / max(val_batches, 1)

            row = {
                "epoch": epoch,
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
        if best_path.exists():
            model = VolatilityMLP.load(best_path, config=config, device=str(device))

    summary = {
        "chunk": label,
        "run_tag": run_tag,
        "best_val_loss": float(best_val_loss),
        "epochs_trained": int(epoch),
        "n_tickers_garch": int(len(garch_models)),
        "total_params": int(total_params),
        "train_samples": int(len(train_ds)),
        "val_samples": int(len(val_ds)),
        "batch_size": int(config.batch_size),
        "hidden_dims": list(config.hidden_dims),
        "dropout": float(config.dropout),
        "learning_rate": float(config.learning_rate),
        "weight_decay": float(config.weight_decay),
        "train_audit": train_ds.audit,
        "val_audit": val_ds.audit,
        "saved_checkpoints": bool(save_checkpoints),
    }

    if save_checkpoints:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    print(f"  Best val loss: {best_val_loss:.6f}")
    return model, summary


def train_volatility_model(config: VolatilityConfig, chunk_id: int, fresh: bool = False) -> Dict[str, Any]:
    config.resolve_paths()
    configure_torch_runtime(config)
    seed_everything(config.seed)
    device = resolve_device(config.device)
    label = CHUNK_CONFIG[chunk_id]["label"]

    print(f"\n{'=' * 72}")
    print(f"  VOLATILITY MODEL — Chunk {chunk_id}")
    print(f"{'=' * 72}")
    print(f"  Device: {device}")

    ensure_all_manifests(config, chunk_id)

    returns_df = pd.read_csv(config.returns_path, index_col=0, parse_dates=True)
    returns_df = returns_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    model_dir = model_dir_for(config, label, "")
    if fresh and model_dir.exists():
        print(f"  Fresh run requested. Removing old model directory: {model_dir}")
        shutil.rmtree(model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)

    print("\n  [1/2] Fitting stable GARCH-style volatility baselines...")
    garch_models = fit_garch_models(config, chunk_id)

    with open(model_dir / "garch_models.pkl", "wb") as f:
        pickle.dump(garch_models, f)

    garch_xai = {ticker: model.to_dict() for ticker, model in garch_models.items()}
    with open(model_dir / "garch_params.json", "w") as f:
        json.dump(garch_xai, f, indent=2)

    print("\n  [2/2] Training volatility MLP...")
    model, summary = _train_mlp(
        config,
        chunk_id,
        garch_models,
        returns_df,
        device,
        run_tag="",
        save_checkpoints=True,
    )

    final_path = model_dir / "final_model.pt"
    freezed_dir = model_dir / "model_freezed"
    freezed_dir.mkdir(parents=True, exist_ok=True)

    model.save(final_path)
    model.save(freezed_dir / "model.pt")

    print(f"\n  Complete. Best val loss: {summary['best_val_loss']:.6f}")
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# HPO
# ═══════════════════════════════════════════════════════════════════════════════

# def _hpo_objective(
#     trial: "optuna.trial.Trial",
#     base_config: VolatilityConfig,
#     chunk_id: int,
#     returns_df: pd.DataFrame,
#     garch_models: Dict[str, SimpleGARCH],
# ) -> float:
#     cfg_dict = base_config.to_dict()
#     trial_config = VolatilityConfig(**cfg_dict).resolve_paths()

#     hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
#     n_layers = trial.suggest_categorical("n_layers", [1, 2])
#     if n_layers == 1:
#         trial_config.hidden_dims = [hidden_dim]
#     else:
#         trial_config.hidden_dims = [hidden_dim, max(32, hidden_dim // 2)]

#     trial_config.dropout = trial.suggest_float("dropout", 0.05, 0.35)
#     trial_config.learning_rate = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
#     trial_config.weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)
#     trial_config.batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048])
#     trial_config.epochs = int(base_config.hpo_epochs)
#     trial_config.early_stop_patience = min(5, int(base_config.early_stop_patience))
#     trial_config.max_train_samples = int(base_config.hpo_max_train_samples)
#     trial_config.max_val_samples = int(base_config.hpo_max_val_samples)

#     # HPO should not use persistent workers or many workers. It repeats trials.
#     trial_config.num_workers = 0
#     trial_config.persistent_workers = False
#     trial_config.run_tag = f"trial_{trial.number:04d}"
#     trial_config.save_checkpoints = False

#     device = resolve_device(trial_config.device)

#     try:
#         _, summary = _train_mlp(
#             trial_config,
#             chunk_id,
#             garch_models,
#             returns_df,
#             device,
#             run_tag=trial_config.run_tag,
#             save_checkpoints=False,
#         )
#         value = float(summary["best_val_loss"])
#         if not np.isfinite(value):
#             raise RuntimeError("HPO produced non-finite objective.")
#         return value
#     except Exception as exc:
#         print(f"  Trial {trial.number} failed safely: {exc}")
#         cleanup_memory()
#         return float("inf")
def _hpo_objective(
    trial: "optuna.trial.Trial",
    base_config: VolatilityConfig,
    chunk_id: int,
    returns_df: pd.DataFrame,
    garch_models: Dict[str, SimpleGARCH],
) -> float:
    cfg_dict = base_config.to_dict()
    trial_config = VolatilityConfig(**cfg_dict).resolve_paths()

    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    n_layers = trial.suggest_categorical("n_layers", [1, 2])

    if n_layers == 1:
        trial_config.hidden_dims = [hidden_dim]
    else:
        trial_config.hidden_dims = [hidden_dim, max(32, hidden_dim // 2)]

    trial_config.dropout = trial.suggest_float("dropout", 0.05, 0.35)
    trial_config.learning_rate = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    trial_config.weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)
    trial_config.batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048])

    trial_config.epochs = int(base_config.hpo_epochs)
    trial_config.early_stop_patience = min(5, int(base_config.early_stop_patience))
    trial_config.max_train_samples = int(base_config.hpo_max_train_samples)
    trial_config.max_val_samples = int(base_config.hpo_max_val_samples)

    # HPO repeats many short runs, so avoid worker/file-handle accumulation.
    trial_config.num_workers = 0
    trial_config.persistent_workers = False
    trial_config.run_tag = f"trial_{trial.number:04d}"
    trial_config.save_checkpoints = False

    device = resolve_device(trial_config.device)

    try:
        _, summary = _train_mlp(
            trial_config,
            chunk_id,
            garch_models,
            returns_df,
            device,
            run_tag=trial_config.run_tag,
            save_checkpoints=False,
        )

        value = float(summary["best_val_loss"])

        if not np.isfinite(value):
            print(f"  Trial {trial.number} produced non-finite loss. Returning finite HPO penalty.")
            return HPO_FAILURE_VALUE

        return value

    except Exception as exc:
        # Never return inf/nan to Optuna RDB storage.
        print(f"  Trial {trial.number} failed safely: {exc}")
        cleanup_memory()
        return HPO_FAILURE_VALUE

def run_hpo(config: VolatilityConfig, chunk_id: int, fresh: bool = False) -> Dict[str, Any]:
    if not HAS_OPTUNA:
        raise ImportError("optuna required: pip install optuna")

    config.resolve_paths()
    configure_torch_runtime(config)
    seed_everything(config.seed)

    label = CHUNK_CONFIG[chunk_id]["label"]
    ensure_all_manifests(config, chunk_id)

    returns_df = pd.read_csv(config.returns_path, index_col=0, parse_dates=True)
    returns_df = returns_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    print("\n  Fitting GARCH baselines once for HPO...")
    garch_models = fit_garch_models(config, chunk_id)

    study_dir = Path(config.output_dir) / "codeResults" / "Volatility"
    study_dir.mkdir(parents=True, exist_ok=True)
    db_path = study_dir / "hpo.db"
    study_name = f"volatility_{label}"

    if fresh and db_path.exists():
        db_path.unlink()
        print(f"  Deleted old HPO database: {db_path}")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.seed, n_startup_trials=min(config.hpo_n_startup, config.hpo_trials)),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
        storage=f"sqlite:///{db_path}",
        study_name=study_name,
        load_if_exists=True,
    )

    objective = lambda trial: _hpo_objective(trial, config, chunk_id, returns_df, garch_models)
    study.optimize(objective, n_trials=int(config.hpo_trials), show_progress_bar=True)

    # completed = [t for t in study.trials if t.value is not None and np.isfinite(t.value)]
    # if not completed:
    #     raise RuntimeError("All HPO trials failed or produced non-finite values. No best params saved.")

    # best_params = study.best_params
    # best_value = float(study.best_value)
    # if not np.isfinite(best_value):
    #     raise RuntimeError("Best HPO value is non-finite. Refusing to save invalid HPO params.")
    completed = [
        t for t in study.trials
        if t.value is not None
        and np.isfinite(float(t.value))
        and float(t.value) < HPO_FAILURE_VALUE
    ]

    if not completed:
        raise RuntimeError(
            "All HPO trials failed. No usable best params saved. "
            "Check missing embeddings, dataset construction, or non-finite losses."
        )

    best_params = study.best_params
    best_value = float(study.best_value)

    if (not np.isfinite(best_value)) or best_value >= HPO_FAILURE_VALUE:
        raise RuntimeError(
            "Best HPO value is invalid or only reflects failed penalty trials. "
            "Refusing to save invalid HPO params."
        )
    best = {"params": best_params, "value": best_value}
    best_path = study_dir / f"best_params_{label}.json"
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)

    print(f"\n  Best HPO: {best_params} (val_loss={best_value:.6f})")
    print(f"  Saved: {best_path}")
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION + XAI
# ═══════════════════════════════════════════════════════════════════════════════

def extract_gradient_xai(
    model: VolatilityMLP,
    loader: DataLoader,
    config: VolatilityConfig,
    device: torch.device,
) -> Dict[str, Any]:
    """XAI Level 1: gradient importance over temporal embedding dimensions."""
    model.eval()

    grad_chunks = []
    n_xai = 0

    for batch in tqdm(loader, desc=f"  XAI-L1 gradients bs={loader.batch_size}", leave=False, unit="batch"):
        if n_xai >= config.xai_sample_size:
            break

        emb = batch["emb"].to(device, non_blocking=True)
        garch = batch["garch_vol"].to(device, non_blocking=True)
        recent = batch["recent_vol"].to(device, non_blocking=True)

        take = min(len(emb), int(config.xai_sample_size) - n_xai)
        emb_xai = emb[:take].clone().detach().requires_grad_(True)
        garch_xai = garch[:take]
        recent_xai = recent[:take]

        model.zero_grad(set_to_none=True)
        out = model(emb_xai, garch_xai, recent_xai)
        score = out["vol_10d"].mean() + out["vol_30d"].mean()
        score.backward()

        if emb_xai.grad is not None:
            grad_chunks.append(emb_xai.grad.detach().abs().mean(dim=0).cpu().numpy())

        n_xai += take

    if not grad_chunks:
        return {"feature_importance": [], "n_samples": 0}

    grad_mean = np.stack(grad_chunks).mean(axis=0)
    total = float(grad_mean.sum())
    if not np.isfinite(total) or total <= 0:
        total = 1.0

    importance = [
        {
            "dim": int(i),
            "importance": float(v),
            "importance_pct": float((v / total) * 100.0),
        }
        for i, v in enumerate(grad_mean)
    ]
    importance.sort(key=lambda x: x["importance"], reverse=True)

    return {
        "feature_importance": importance,
        "n_samples": int(n_xai),
    }


def generate_counterfactual_xai(
    model: VolatilityMLP,
    dataset: VolatilityDataset,
    config: VolatilityConfig,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """XAI Level 2: counterfactual what-if scenarios."""
    model.eval()
    n = min(int(config.xai_counterfactual_scenarios), len(dataset))
    if n <= 0:
        return []

    rng = np.random.default_rng(config.seed)
    indices = rng.choice(len(dataset), size=n, replace=False)

    scenarios = []
    for idx in indices:
        sample = dataset[int(idx)]
        emb = sample["emb"].unsqueeze(0).to(device)
        garch = sample["garch_vol"].unsqueeze(0).to(device)
        recent = sample["recent_vol"].unsqueeze(0).to(device)

        with torch.no_grad():
            original = model(emb, garch, recent)
            garch_low = model(emb, garch * 0.8, recent)
            garch_high = model(emb, garch * 1.2, recent)
            recent_high = model(emb, garch, recent * 1.5)

        scenarios.append({
            "ticker": sample["ticker"],
            "date": sample["date"],
            "original": {
                "vol_10d": float(original["vol_10d"].cpu().item()),
                "vol_30d": float(original["vol_30d"].cpu().item()),
                "regime": REGIME_LABELS[int(original["regime_probs"].cpu().argmax().item())],
                "confidence": float(original["confidence"].cpu().item()),
            },
            "counterfactuals": [
                {
                    "condition": "GARCH forecast 20% lower",
                    "vol_10d": float(garch_low["vol_10d"].cpu().item()),
                    "vol_30d": float(garch_low["vol_30d"].cpu().item()),
                },
                {
                    "condition": "GARCH forecast 20% higher",
                    "vol_10d": float(garch_high["vol_10d"].cpu().item()),
                    "vol_30d": float(garch_high["vol_30d"].cpu().item()),
                },
                {
                    "condition": "Recent realized volatility 50% higher",
                    "vol_10d": float(recent_high["vol_10d"].cpu().item()),
                    "vol_30d": float(recent_high["vol_30d"].cpu().item()),
                },
            ],
        })

    return scenarios


def predict_with_xai(config: VolatilityConfig, chunk_id: int, split: str) -> Dict[str, Any]:
    """Generate predictions and return XAI for integrated system use.

    Returns:
        {
            "predictions": pd.DataFrame,
            "xai": dict,
            "paths": dict
        }
    """
    config.resolve_paths()
    configure_torch_runtime(config)
    seed_everything(config.seed)

    device = resolve_device(config.device)
    label = CHUNK_CONFIG[chunk_id]["label"]

    model_dir = model_dir_for(config, label, "")
    model_path = model_dir / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model found: {model_path}")

    model = VolatilityMLP.load(model_path, config=config, device=str(device))
    config = model.config.resolve_paths()
    model.eval()

    garch_path = model_dir / "garch_models.pkl"
    params_path = model_dir / "garch_params.json"

    if not garch_path.exists():
        raise FileNotFoundError(f"Missing GARCH models: {garch_path}")

    with open(garch_path, "rb") as f:
        garch_models = pickle.load(f)

    garch_xai = {}
    if params_path.exists():
        with open(params_path) as f:
            garch_xai = json.load(f)

    returns_df = pd.read_csv(config.returns_path, index_col=0, parse_dates=True)
    returns_df = returns_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    dataset = VolatilityDataset(config, split, chunk_id, returns_df, garch_models)
    loader = make_loader(dataset, config, train=False)

    results_dir = Path(config.output_dir) / "results" / "Volatility"
    xai_dir = results_dir / "xai"
    results_dir.mkdir(parents=True, exist_ok=True)
    xai_dir.mkdir(parents=True, exist_ok=True)

    pred_chunks = []

    with torch.no_grad():
        pred_bar = tqdm(loader, desc=f"  Predict {label}_{split} bs={config.batch_size}", leave=False, unit="batch")
        for batch in pred_bar:
            emb = batch["emb"].to(device, non_blocking=True)
            garch = batch["garch_vol"].to(device, non_blocking=True)
            recent = batch["recent_vol"].to(device, non_blocking=True)

            out = model(emb, garch, recent)
            probs = out["regime_probs"].detach().cpu().numpy()
            regimes = probs.argmax(axis=1)

            pred_chunks.append(pd.DataFrame({
                "ticker": list(batch["ticker"]),
                "date": list(batch["date"]),
                "vol_10d": out["vol_10d"].detach().cpu().numpy(),
                "vol_30d": out["vol_30d"].detach().cpu().numpy(),
                "regime_id": regimes,
                "regime_label": [REGIME_LABELS[int(r)] for r in regimes],
                "regime_probs_low": probs[:, 0],
                "regime_probs_medium": probs[:, 1],
                "regime_probs_high": probs[:, 2],
                "confidence": out["confidence"].detach().cpu().numpy(),
                "garch_vol": garch.detach().cpu().numpy(),
                "recent_vol": recent.detach().cpu().numpy(),
            }))

    predictions = pd.concat(pred_chunks, ignore_index=True) if pred_chunks else pd.DataFrame()

    # XAI requires gradients, so it is intentionally outside no_grad.
    xai_loader = make_loader(dataset, config, train=False)
    try:
        gradient_xai = extract_gradient_xai(model, xai_loader, config, device)
    finally:
        shutdown_dataloader(xai_loader)

    counterfactuals = generate_counterfactual_xai(model, dataset, config, device)

    garch_summary = {
        "n_models": len(garch_xai),
        "avg_omega": float(np.mean([v["omega"] for v in garch_xai.values() if v.get("omega") is not None])) if garch_xai else None,
        "avg_alpha": float(np.mean([v["alpha"] for v in garch_xai.values() if v.get("alpha") is not None])) if garch_xai else None,
        "avg_beta": float(np.mean([v["beta"] for v in garch_xai.values() if v.get("beta") is not None])) if garch_xai else None,
        "avg_persistence": float(np.mean([v["persistence"] for v in garch_xai.values() if v.get("persistence") is not None])) if garch_xai else None,
    }

    xai = {
        "module": "Volatility",
        "chunk": label,
        "split": split,
        "level1_gradient_feature_importance": gradient_xai,
        "level2_counterfactuals": counterfactuals,
        "garch_parameter_summary": garch_summary,
        "dataset_audit": dataset.audit,
        "explanation_summary": {
            "plain_english": (
                "The volatility model combines a stable GARCH-style volatility baseline, recent realized volatility, "
                "and temporal encoder embeddings. Gradient XAI ranks which embedding dimensions most influenced "
                "the predicted volatility, while counterfactual XAI shows how predictions change when GARCH or "
                "recent-volatility inputs are altered."
            )
        },
    }

    pred_path = results_dir / f"predictions_{label}_{split}.csv"
    importance_path = xai_dir / f"{label}_{split}_feature_importance.csv"
    counterfactual_path = xai_dir / f"{label}_{split}_counterfactuals.json"
    garch_summary_path = xai_dir / f"{label}_{split}_garch_summary.json"
    xai_json_path = xai_dir / f"{label}_{split}_xai_summary.json"

    predictions.to_csv(pred_path, index=False)

    if gradient_xai["feature_importance"]:
        pd.DataFrame(gradient_xai["feature_importance"]).to_csv(importance_path, index=False)

    with open(counterfactual_path, "w") as f:
        json.dump(counterfactuals, f, indent=2)

    with open(garch_summary_path, "w") as f:
        json.dump(garch_summary, f, indent=2)

    # Avoid saving huge duplicated structures in the compact summary.
    compact_xai = dict(xai)
    compact_xai["level1_gradient_feature_importance"] = {
        "n_samples": gradient_xai["n_samples"],
        "top_20": gradient_xai["feature_importance"][:20],
    }
    compact_xai["level2_counterfactuals"] = counterfactuals[: min(5, len(counterfactuals))]

    with open(xai_json_path, "w") as f:
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
            "feature_importance": str(importance_path),
            "counterfactuals": str(counterfactual_path),
            "garch_summary": str(garch_summary_path),
            "xai_summary": str(xai_json_path),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# INSPECT AND SMOKE
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_inspect(config: VolatilityConfig) -> None:
    config.resolve_paths()
    print("=" * 72)
    print("VOLATILITY MODEL — DATA INSPECTION")
    print("=" * 72)

    emb_dir = Path(config.embeddings_dir)
    if emb_dir.exists():
        for file in sorted(emb_dir.glob("*_embeddings.npy")):
            emb = np.load(file, mmap_mode="r")
            sample = np.asarray(emb[: min(2048, len(emb))])
            print(f"  {file.name}: {emb.shape}, sample_finite={finite_ratio_np(sample):.6f}")

        for file in sorted(emb_dir.glob("*_manifest.csv")):
            manifest = pd.read_csv(file)
            if len(manifest):
                print(
                    f"  {file.name}: {len(manifest):,} rows, "
                    f"{manifest['ticker'].nunique()} tickers, "
                    f"dates {manifest['date'].min()} → {manifest['date'].max()}"
                )

    returns_df = pd.read_csv(config.returns_path, index_col=0)
    returns_values = returns_df.apply(pd.to_numeric, errors="coerce").values
    print(f"  Returns: {returns_df.shape[0]:,} days × {returns_df.shape[1]:,} tickers")
    print(f"  Date range: {returns_df.index[0]} → {returns_df.index[-1]}")
    print(f"  Returns finite ratio: {finite_ratio_np(returns_values):.6f}")


def cmd_smoke(config: VolatilityConfig, device_str: str) -> None:
    config.resolve_paths()
    configure_torch_runtime(config)
    seed_everything(config.seed)

    device = resolve_device(device_str)

    print("=" * 72)
    print("VOLATILITY MODEL — SMOKE TEST")
    print("=" * 72)

    n = 512
    emb_dim = int(config.input_dim)

    emb = torch.randn(n, emb_dim, device=device)
    garch = torch.rand(n, device=device) * 0.4 + 0.05
    recent = torch.rand(n, device=device) * 0.4 + 0.05
    target = torch.rand(n, 2, device=device) * 0.5 + 0.05

    model = VolatilityMLP(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    out = model(emb, garch, recent)
    loss = volatility_loss(out, target)

    if not torch.isfinite(loss):
        raise RuntimeError("Smoke test failed: non-finite loss.")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
    optimizer.step()

    model.eval()
    emb_xai = emb[:16].detach().clone().requires_grad_(True)
    out_xai = model(emb_xai, garch[:16], recent[:16])
    score = out_xai["vol_10d"].mean() + out_xai["vol_30d"].mean()
    model.zero_grad(set_to_none=True)
    score.backward()

    if emb_xai.grad is None or not torch.isfinite(emb_xai.grad).all():
        raise RuntimeError("Smoke test failed: XAI gradient invalid.")

    print("SMOKE TEST PASSED")
    print(f"  loss={float(loss.detach().cpu()):.6f}")
    print(f"  vol10_mean={float(out['vol_10d'].mean().detach().cpu()):.6f}")
    print(f"  vol30_mean={float(out['vol_30d'].mean().detach().cpu()):.6f}")
    print(f"  xai_grad_shape={tuple(emb_xai.grad.shape)}")


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
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")


def config_from_args(args: argparse.Namespace) -> VolatilityConfig:
    config = VolatilityConfig()

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
    if getattr(args, "no_amp", False):
        config.mixed_precision = False
    if getattr(args, "deterministic", False):
        config.deterministic = True
    if getattr(args, "trials", None) is not None:
        config.hpo_trials = int(args.trials)

    return config.resolve_paths()


def apply_hpo_params_if_available(config: VolatilityConfig, chunk_id: int) -> VolatilityConfig:
    label = CHUNK_CONFIG[chunk_id]["label"]
    hpo_path = Path(config.output_dir) / "codeResults" / "Volatility" / f"best_params_{label}.json"

    if not hpo_path.exists():
        print("  No HPO params found. Training with current/default config.")
        return config

    with open(hpo_path) as f:
        hpo = json.load(f)

    value = float(hpo.get("value", float("inf")))
    if not np.isfinite(value):
        raise RuntimeError(f"Invalid HPO file has non-finite value: {hpo_path}")

    params = hpo.get("params", {})
    hidden_dim = int(params.get("hidden_dim", config.hidden_dims[0]))
    n_layers = int(params.get("n_layers", 1))

    if n_layers == 1:
        config.hidden_dims = [hidden_dim]
    else:
        config.hidden_dims = [hidden_dim, max(32, hidden_dim // 2)]

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Volatility Estimation Model (Stable GARCH + MLP + XAI)")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("inspect", help="Inspect embeddings, manifests, and returns")
    add_common_args(p)

    p = sub.add_parser("smoke", help="Run synthetic forward/backward/XAI smoke test")
    add_common_args(p)

    p = sub.add_parser("hpo", help="Run Optuna TPE HPO")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--fresh", action="store_true")
    add_common_args(p)

    p = sub.add_parser("train-best", help="Train with best HPO params if available")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--fresh", action="store_true")
    add_common_args(p)

    p = sub.add_parser("train-best-all", help="Train all chunks")
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
        run_hpo(config, args.chunk, fresh=bool(args.fresh))

    elif args.command == "train-best":
        config = apply_hpo_params_if_available(config, args.chunk)
        train_volatility_model(config, args.chunk, fresh=bool(args.fresh))

    elif args.command == "train-best-all":
        for chunk_id in [1, 2, 3]:
            config = apply_hpo_params_if_available(config, chunk_id)
            train_volatility_model(config, chunk_id, fresh=bool(args.fresh))

    elif args.command == "predict":
        result = predict_with_xai(config, args.chunk, args.split)
        print(f"  Returned keys: {list(result.keys())}")


if __name__ == "__main__":
    main()

"""
Commands:
    python code/riskEngine/volatility.py inspect --repo-root .
    python code/riskEngine/volatility.py smoke --repo-root . --device cuda
    python code/riskEngine/volatility.py hpo --repo-root . --chunk 1 --trials 35 --device cuda --fresh
    python code/riskEngine/volatility.py train-best --repo-root . --chunk 1 --device cuda --fresh
    python code/riskEngine/volatility.py predict --repo-root . --chunk 1 --split test --device cuda
"""