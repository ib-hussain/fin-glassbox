#!/usr/bin/env python3
"""
code/encoders/temporal_encoder.py

Shared Temporal Attention Encoder for financial time series.
Project: fin-glassbox — Explainable Multimodal Neural Framework

Architecture:
  Transformer Encoder (HPO-optimized per chunk)
  Input: 30-day sequences of 10 market features
  Output: 128-dim temporal embedding (last, mean, or attention pooled)

Training objective:
  Self-supervised masked prediction — randomly mask 15% of time steps
  and predict the masked feature values.

GPU Optimizations:
  - Mixed precision (AMP) with GradScaler
  - cuDNN benchmark auto-tuning
  - Pinned memory + non-blocking transfers
  - Prefetch_factor=4 for continuous GPU feeding
  - num_workers=8 for parallel data loading
  - Batch size 256 (up from 64) for better GPU saturation
  - set_to_none=True for zero_grad (memory efficient)

Usage:
  python code/encoders/temporal_encoder.py inspect
  python code/encoders/temporal_encoder.py train-best --chunk 1 --device cuda
  python code/encoders/temporal_encoder.py train-best-all --device cuda
  python code/encoders/temporal_encoder.py embed --chunk 1 --split val --device cuda
"""

from __future__ import annotations

import argparse, json, math, os, sys, time, warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# GPU OPTIMISATIONS
# ═══════════════════════════════════════════════════════════════════
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
if hasattr(torch.backends.cudnn, 'allow_tf32'):
    torch.backends.cudnn.allow_tf32 = True
if hasattr(torch.backends.cuda, 'matmul'):
    torch.backends.cuda.matmul.allow_tf32 = True

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TemporalEncoderConfig:
    """Configuration for the Shared Temporal Attention Encoder."""

    repo_root: str = ""
    features_path: str = "data/yFinance/processed/features_temporal.csv"
    output_dir: str = "outputs"

    # Model architecture
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation: str = "gelu"
    max_seq_len: int = 90
    n_input_features: int = 10

    # Training — OPTIMIZED for RTX 3090 Ti
    seq_len: int = 30
    batch_size: int = 256          # ↑ from 64 — better GPU saturation
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 4000
    gradient_clip: float = 1.0
    label_smoothing: float = 0.0
    early_stop_patience: int = 20

    # XAI
    xai_sample_size: int = 1000

    # Masked prediction
    mask_prob: float = 0.15
    mask_seed: int = 42

    # HPO
    hpo_trials: int = 75
    hpo_n_startup: int = 20

    # System — OPTIMIZED
    device: str = "cuda"
    seed: int = 42
    mixed_precision: bool = True
    num_workers: int = 8           # ↑ from 6 — max out 12 threads
    prefetch_factor: int = 4       # Preload batches

    # Data
    max_train_rows: int = 0

    def __post_init__(self):
        if self.repo_root:
            self.features_path = str(Path(self.repo_root) / self.features_path)
            self.output_dir = str(Path(self.repo_root) / self.output_dir)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


# ═══════════════════════════════════════════════════════════════════
# POSITIONAL ENCODING
# ═══════════════════════════════════════════════════════════════════

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 90):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


# ═══════════════════════════════════════════════════════════════════
# TEMPORAL ENCODER
# ═══════════════════════════════════════════════════════════════════

class TemporalEncoder(nn.Module):
    """Shared Temporal Attention Encoder — Transformer-based."""

    FEATURE_NAMES = [
        "log_return", "vol_5d", "vol_21d", "rsi_14", "macd_hist",
        "bb_pos", "volume_ratio", "hl_ratio", "price_pos", "spy_corr_63d",
    ]

    CHUNK_CONFIG = {
        1: {"train": (2000, 2004), "val": (2005, 2005), "test": (2006, 2006), "label": "chunk1"},
        2: {"train": (2007, 2014), "val": (2015, 2015), "test": (2016, 2016), "label": "chunk2"},
        3: {"train": (2017, 2022), "val": (2023, 2023), "test": (2024, 2024), "label": "chunk3"},
    }

    def __init__(self, config: TemporalEncoderConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model

        self.input_projection = nn.Linear(config.n_input_features, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, config.max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=config.n_heads, dim_feedforward=config.d_ff,
            dropout=config.dropout, activation=config.activation,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        self.pooling_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=1, batch_first=True, dropout=0.0,
        )
        self.output_projection = nn.Linear(d_model, config.n_input_features)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
                ) -> dict[str, torch.Tensor]:
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.transformer(x, src_key_padding_mask=mask if mask is not None else None)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            x_masked = x * (1 - mask_expanded)
            seq_lengths = (1 - mask_expanded).sum(dim=1).clamp(min=1)
            mean_pooled = x_masked.sum(dim=1) / seq_lengths
            lengths = (1 - mask.int()).sum(dim=1) - 1
            lengths = lengths.clamp(min=0)
            last_hidden = x[torch.arange(x.size(0)), lengths]
        else:
            mean_pooled = x.mean(dim=1)
            last_hidden = x[:, -1, :]

        query = self.pooling_query.expand(x.size(0), -1, -1)
        if mask is not None:
            attn_pooled, _ = self.attention_pooling(query, x, x, key_padding_mask=mask)
        else:
            attn_pooled, _ = self.attention_pooling(query, x, x)
        attn_pooled = attn_pooled.squeeze(1)

        return {
            "sequence": x, "last_hidden": last_hidden,
            "mean_pooled": mean_pooled, "attention_pooled": attn_pooled,
        }

    def get_embedding(self, x: torch.Tensor, pooling: str = "last",
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = self.forward(x, mask)
        return output[f"{pooling}_pooled" if pooling != "last" else "last_hidden"]

    def predict_masked(self, x: torch.Tensor, mask_indices: torch.Tensor) -> torch.Tensor:
        output = self.forward(x)
        return self.output_projection(output["sequence"])

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": self.state_dict(), "config": asdict(self.config)}, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "TemporalEncoder":        
        checkpoint = torch.load(path, map_location=device, weights_only=False)        
        config = TemporalEncoderConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model


# ═══════════════════════════════════════════════════════════════════
# FEATURE NORMALIZER
# ═══════════════════════════════════════════════════════════════════

# class FeatureNormalizer:
#     def __init__(self):
#         self.mean = None
#         self.std = None

#     def fit(self, sequences: list[np.ndarray]):
#         all_data = np.concatenate(sequences, axis=0)
#         self.mean = all_data.mean(axis=0, keepdims=True)
#         self.std = all_data.std(axis=0, keepdims=True)
#         self.std[self.std < 1e-8] = 1.0

#     def transform(self, x: torch.Tensor) -> torch.Tensor:
#         if self.mean is None:
#             return x
#         mean_t = torch.from_numpy(self.mean).float().to(x.device)
#         std_t = torch.from_numpy(self.std).float().to(x.device)
#         return (x - mean_t) / std_t
class FeatureNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None
        self.mean_t = None
        self.std_t = None
        self.device = None

    def fit(self, sequences: list[np.ndarray]):
        all_data = np.concatenate(sequences, axis=0)
        self.mean = all_data.mean(axis=0, keepdims=True).reshape(1, 1, -1).astype(np.float32)
        self.std = all_data.std(axis=0, keepdims=True).reshape(1, 1, -1).astype(np.float32)
        self.std[self.std < 1e-8] = 1.0
        self.mean_t = None
        self.std_t = None
        self.device = None

    def fit_from_dataset(self, dataset: "MarketSequenceDataset"):
        mean, std = dataset.compute_feature_stats()
        self.mean = mean.reshape(1, 1, -1).astype(np.float32)
        self.std = std.reshape(1, 1, -1).astype(np.float32)
        self.std[self.std < 1e-8] = 1.0
        self.mean_t = None
        self.std_t = None
        self.device = None

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path), mean=self.mean.astype(np.float32), std=self.std.astype(np.float32))

    @classmethod
    def load(cls, path: str | Path) -> "FeatureNormalizer":
        data = np.load(str(path))
        obj = cls()
        obj.mean = data["mean"].astype(np.float32)
        obj.std = data["std"].astype(np.float32)
        obj.mean_t = None
        obj.std_t = None
        obj.device = None
        return obj

    def to_device(self, device: torch.device | str):
        if self.mean is None:
            return
        device = torch.device(device)
        if self.device == device and self.mean_t is not None and self.std_t is not None:
            return
        self.mean_t = torch.from_numpy(self.mean).float().to(device)
        self.std_t = torch.from_numpy(self.std).float().to(device)
        self.device = device

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None:
            return x
        self.to_device(x.device)
        return (x - self.mean_t) / self.std_t
    

# ═══════════════════════════════════════════════════════════════════
# DATASET — PARALLELIZED SEQUENCE BUILDING
# ═══════════════════════════════════════════════════════════════════

# class MarketSequenceDataset(Dataset):
#     """Sliding-window dataset with parallel sequence construction."""

#     def __init__(self, features_df: pd.DataFrame, seq_len: int = 30,
#                  years: tuple[int, int] = (2000, 2004),
#                  tickers: Optional[list[str]] = None,
#                  max_rows: int = 0,
#                  normalizer: Optional[FeatureNormalizer] = None,
#                  num_workers: int = 8):
#         df = features_df.copy()
#         df["date"] = pd.to_datetime(df["date"])
#         df["year"] = df["date"].dt.year

#         mask = (df["year"] >= years[0]) & (df["year"] <= years[1])
#         df = df[mask]

#         if tickers is not None:
#             df = df[df["ticker"].isin(tickers)]

#         feature_cols = [c for c in TemporalEncoder.FEATURE_NAMES if c in df.columns]
#         self.feature_names = feature_cols

#         # ── Parallel sequence building ──
#         ticker_groups = [(ticker, group) for ticker, group in df.groupby("ticker")]
        
#         self.sequences = []
        
#         def build_ticker_sequences(args):
#             ticker, group = args
#             vals = group[feature_cols].values.astype(np.float32)
#             vals = np.nan_to_num(vals, nan=0.0)
#             if len(vals) < seq_len:
#                 return []
#             return [vals[i:i + seq_len] for i in range(0, len(vals) - seq_len)]

#         with ThreadPoolExecutor(max_workers=num_workers) as pool:
#             results = list(tqdm(
#                 pool.map(build_ticker_sequences, ticker_groups),
#                 total=len(ticker_groups), desc="  Building sequences", leave=False,
#             ))
        
#         for seqs in results:
#             self.sequences.extend(seqs)

#         self.sequences_array = [s.copy() for s in self.sequences]
#         self.normalizer = normalizer

#         if max_rows > 0 and len(self.sequences) > max_rows:
#             rng = np.random.RandomState(42)
#             indices = rng.choice(len(self.sequences), max_rows, replace=False)            
#             self.sequences = [self.sequences[i] for i in indices]

#     def get_raw_sequences(self) -> list[np.ndarray]:
#         return self.sequences_array if hasattr(self, 'sequences_array') else self.sequences

#     def __len__(self) -> int:
#         return len(self.sequences)

#     def __getitem__(self, idx: int) -> torch.Tensor:
#         return torch.from_numpy(self.sequences[idx].copy())
class MarketSequenceDataset(Dataset):
    """Fast sliding-window dataset.

    Does not materialise millions of sequence copies.
    Stores grouped feature arrays and lightweight window indices.
    Also preserves ticker/date manifest order for downstream embeddings.
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        seq_len: int = 30,
        years: tuple[int, int] = (2000, 2004),
        tickers: Optional[list[str]] = None,
        max_rows: int = 0,
        normalizer: Optional[FeatureNormalizer] = None,
        num_workers: int = 8,
    ):
        df = features_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year

        mask = (df["year"] >= years[0]) & (df["year"] <= years[1])
        df = df[mask]

        if tickers is not None:
            df = df[df["ticker"].isin(tickers)]

        feature_cols = [c for c in TemporalEncoder.FEATURE_NAMES if c in df.columns]
        self.feature_names = feature_cols
        self.seq_len = int(seq_len)
        self.normalizer = normalizer

        self.group_values: list[np.ndarray] = []
        self.group_tickers: list[str] = []
        self.group_dates: list[np.ndarray] = []

        group_ids = []
        start_indices = []
        sample_tickers = []
        sample_dates = []

        grouped = list(df.groupby("ticker", sort=True))

        for ticker, group in tqdm(grouped, desc="  Indexing sequences", leave=False):
            group = group.sort_values("date")

            vals = group[feature_cols].values.astype(np.float32)
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

            dates = group["date"].dt.strftime("%Y-%m-%d").values.astype(str)

            n_windows = len(vals) - self.seq_len
            if n_windows <= 0:
                continue

            gid = len(self.group_values)
            starts = np.arange(n_windows, dtype=np.int32)

            self.group_values.append(vals)
            self.group_tickers.append(str(ticker))
            self.group_dates.append(dates)

            group_ids.append(np.full(n_windows, gid, dtype=np.int32))
            start_indices.append(starts)
            sample_tickers.append(np.array([str(ticker)] * n_windows, dtype=object))
            sample_dates.append(dates[starts + self.seq_len - 1].astype(object))

        if not group_ids:
            raise ValueError(f"No sequences built for years={years}")

        self.group_ids = np.concatenate(group_ids)
        self.start_indices = np.concatenate(start_indices)
        self.sample_tickers = np.concatenate(sample_tickers)
        self.sample_dates = np.concatenate(sample_dates)

        if max_rows > 0 and len(self.group_ids) > max_rows:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(self.group_ids), size=int(max_rows), replace=False)
            idx = np.sort(idx)
            self.group_ids = self.group_ids[idx]
            self.start_indices = self.start_indices[idx]
            self.sample_tickers = self.sample_tickers[idx]
            self.sample_dates = self.sample_dates[idx]

    def compute_feature_stats(self) -> tuple[np.ndarray, np.ndarray]:
        total_sum = None
        total_sq = None
        total_count = 0

        for vals in self.group_values:
            vals64 = vals.astype(np.float64, copy=False)
            if total_sum is None:
                total_sum = vals64.sum(axis=0)
                total_sq = (vals64 ** 2).sum(axis=0)
            else:
                total_sum += vals64.sum(axis=0)
                total_sq += (vals64 ** 2).sum(axis=0)
            total_count += vals64.shape[0]

        mean = total_sum / max(total_count, 1)
        var = total_sq / max(total_count, 1) - mean ** 2
        var = np.maximum(var, 1e-12)
        std = np.sqrt(var)

        std[std < 1e-8] = 1.0
        return mean.astype(np.float32), std.astype(np.float32)

    def get_raw_sequences(self) -> list[np.ndarray]:
        # Compatibility only. Avoid using this for full datasets.
        return [self[i].numpy() for i in range(len(self))]

    def get_manifest(self) -> pd.DataFrame:
        return pd.DataFrame({
            "ticker": self.sample_tickers.astype(str),
            "date": self.sample_dates.astype(str),
        })

    def __len__(self) -> int:
        return int(len(self.group_ids))

    def __getitem__(self, idx: int) -> torch.Tensor:
        gid = int(self.group_ids[idx])
        start = int(self.start_indices[idx])
        vals = self.group_values[gid]
        seq = vals[start:start + self.seq_len]
        return torch.from_numpy(seq.copy())

# ═══════════════════════════════════════════════════════════════════
# MASKING
# ═══════════════════════════════════════════════════════════════════

def apply_masking(x: torch.Tensor, mask_prob: float = 0.15,
                  mask_seed: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
    if mask_seed is not None:
        torch.manual_seed(mask_seed)
    batch, seq_len, n_feat = x.shape
    mask = torch.rand(batch, seq_len, 1, device=x.device) < mask_prob
    mask[:, 0, :] = False
    mask_indices = mask.expand(-1, -1, n_feat)
    masked_x = x.clone()
    masked_x[mask_indices] = 0.0
    return masked_x, mask_indices


# ═══════════════════════════════════════════════════════════════════
# TRAINING — WITH MIXED PRECISION & NON-BLOCKING TRANSFERS
# ═══════════════════════════════════════════════════════════════════

# def train_epoch(model: TemporalEncoder, dataloader: DataLoader,
#                 optimizer: torch.optim.Optimizer,
#                 scaler: Optional[torch.cuda.amp.GradScaler],
#                 device: str) -> float:
def train_epoch(
    model: TemporalEncoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: str,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        x = batch.to(device, non_blocking=True)
        if hasattr(dataloader.dataset, 'normalizer') and dataloader.dataset.normalizer is not None:
            x = dataloader.dataset.normalizer.transform(x)
        masked_x, mask_indices = apply_masking(x, model.config.mask_prob)
        masked_x = masked_x.to(device, non_blocking=True)
        mask_indices = mask_indices.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                pred = model.predict_masked(masked_x, mask_indices)
                loss = F.mse_loss(pred[mask_indices], x[mask_indices])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), model.config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
        else:
            pred = model.predict_masked(masked_x, mask_indices)
            loss = F.mse_loss(pred[mask_indices], x[mask_indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), model.config.gradient_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model: TemporalEncoder, dataloader: DataLoader, device: str) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in dataloader:
        x = batch.to(device, non_blocking=True)
        if hasattr(dataloader.dataset, 'normalizer') and dataloader.dataset.normalizer is not None:
            x = dataloader.dataset.normalizer.transform(x)
        masked_x, mask_indices = apply_masking(x, model.config.mask_prob)
        masked_x = masked_x.to(device, non_blocking=True)
        mask_indices = mask_indices.to(device, non_blocking=True)
        pred = model.predict_masked(masked_x, mask_indices)
        loss = F.mse_loss(pred[mask_indices], x[mask_indices])
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def create_optimizer_and_scheduler(model, total_steps):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=model.config.learning_rate,
        weight_decay=model.config.weight_decay, fused=True if torch.cuda.is_available() else False,
    )
    def lr_lambda(step):
        if step < model.config.warmup_steps:
            return step / max(model.config.warmup_steps, 1)
        progress = (step - model.config.warmup_steps) / max(total_steps - model.config.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


# def train_model(model, train_loader, val_loader, chunk_id, output_dir):
#     config = model.config
#     device = config.device
#     total_steps = len(train_loader) * config.epochs
#     optimizer, scheduler = create_optimizer_and_scheduler(model, total_steps)

#     scaler = torch.cuda.amp.GradScaler() if (config.mixed_precision and device == "cuda") else None

#     best_val_loss = float("inf")
#     no_improve = 0
#     history = {"train_loss": [], "val_loss": [], "lr": []}
#     best_path = output_dir / "best_model.pt"
#     latest_path = output_dir / "latest_model.pt"
#     chunk_label = TemporalEncoder.CHUNK_CONFIG[chunk_id]["label"]

#     for epoch in range(1, config.epochs + 1):
#         train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
#         val_loss = validate(model, val_loader, device)
#         current_lr = optimizer.param_groups[0]["lr"]

#         history["train_loss"].append(train_loss)
#         history["val_loss"].append(val_loss)
#         history["lr"].append(current_lr)

#         tqdm.write(f"  [{chunk_label}] Epoch {epoch:3d}/{config.epochs} | "
#                     f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.2e}")

#         model.save(str(latest_path))
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             no_improve = 0
#             model.save(str(best_path))
#         else:
#             no_improve += 1

#         if no_improve >= config.early_stop_patience:
#             tqdm.write(f"  Early stopping at epoch {epoch}")
#             break
#         scheduler.step()

#     pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
#     summary = {"chunk": chunk_label, "best_val_loss": float(best_val_loss),
#                "epochs_trained": epoch, "config": asdict(config)}
#     with open(output_dir / "training_summary.json", "w") as f:
#         json.dump(summary, f, indent=2)
#     return summary
# Fix 1: In train_model(), add resume logic BEFORE the training loop
# Replace the function starting from line ~450 with this updated version

def train_model(model, train_loader, val_loader, chunk_id, output_dir):
    config = model.config
    device = config.device
    total_steps = len(train_loader) * config.epochs
    optimizer, scheduler = create_optimizer_and_scheduler(model, total_steps)

    scaler = torch.cuda.amp.GradScaler() if (config.mixed_precision and device == "cuda") else None

    best_val_loss = float("inf")
    no_improve = 0
    start_epoch = 1
    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_path = output_dir / "best_model.pt"
    latest_path = output_dir / "latest_model.pt"
    history_path = output_dir / "training_history.csv"
    chunk_label = TemporalEncoder.CHUNK_CONFIG[chunk_id]["label"]

    # ── RESUME SUPPORT ──
    if latest_path.exists():
        print(f"  Found existing checkpoint at {latest_path}")
        # Load previous training history if it exists
        if history_path.exists():
            prev_history = pd.read_csv(history_path)
            if len(prev_history) > 0:
                history["train_loss"] = prev_history["train_loss"].tolist()
                history["val_loss"] = prev_history["val_loss"].tolist()
                history["lr"] = prev_history["lr"].tolist()
                start_epoch = len(prev_history) + 1
                
                # Find best val loss from history
                best_val_loss = min(prev_history["val_loss"])
                # Count consecutive epochs without improvement from the end
                best_idx = prev_history["val_loss"].idxmin()
                no_improve = len(prev_history) - best_idx - 1
                
                print(f"  Resuming from epoch {start_epoch} "
                      f"(best val loss so far: {best_val_loss:.6f}, "
                      f"no_improve: {no_improve})")
        
        # Load model weights
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Reconstruct optimizer with same LR that was at the end
        # Rebuild scheduler from scratch based on completed epochs
        for _ in range(start_epoch - 1):
            scheduler.step()
    else:
        print(f"  Starting fresh training")

    for epoch in range(start_epoch, config.epochs + 1):
        # train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, scheduler=scheduler)
        val_loss = validate(model, val_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        # Save history incrementally (not just at end)
        pd.DataFrame(history).to_csv(history_path, index=False)

        tqdm.write(f"  [{chunk_label}] Epoch {epoch:3d}/{config.epochs} | "
                    f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.2e}")

        model.save(str(latest_path))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            model.save(str(best_path))
        else:
            no_improve += 1

        if no_improve >= config.early_stop_patience:
            tqdm.write(f"  Early stopping at epoch {epoch}")
            break
        # scheduler.step()

    summary = {"chunk": chunk_label, "best_val_loss": float(best_val_loss),
               "epochs_trained": epoch, "config": asdict(config)}
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary

# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_features_df(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"ticker": str}, parse_dates=["date"])
def get_or_fit_train_normalizer(
    config: TemporalEncoderConfig,
    chunk_id: int,
    df: Optional[pd.DataFrame] = None,
) -> FeatureNormalizer:
    chunk_cfg = TemporalEncoder.CHUNK_CONFIG[chunk_id]
    chunk_label = chunk_cfg["label"]

    output_dir = Path(config.output_dir) / "models" / "TemporalEncoder" / chunk_label
    normalizer_path = output_dir / "normalizer.npz"

    if normalizer_path.exists():
        print(f"  Loading train-only normalizer: {normalizer_path}")
        return FeatureNormalizer.load(normalizer_path)

    print("  Train-only normalizer not found. Rebuilding from train split...")
    if df is None:
        df = load_features_df(config.features_path)

    train_dataset = MarketSequenceDataset(
        df,
        seq_len=config.seq_len,
        years=chunk_cfg["train"],
        num_workers=config.num_workers,
    )

    normalizer = FeatureNormalizer()
    normalizer.fit_from_dataset(train_dataset)
    normalizer.save(normalizer_path)
    print(f"  Saved train-only normalizer: {normalizer_path}")
    return normalizer

# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def cmd_inspect(config):
    print("=" * 60)
    print("TEMPORAL ENCODER — DATA INSPECTION")
    print("=" * 60)
    fp = Path(config.features_path)
    if not fp.exists():
        print(f"❌ Features file not found: {fp}")
        return
    df = load_features_df(fp)
    print(f"\n✅ Features file: {fp}")
    print(f"   Rows: {len(df):,}  |  Tickers: {df['ticker'].nunique()}")
    print(f"   Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    missing = [f for f in TemporalEncoder.FEATURE_NAMES if f not in df.columns]
    if missing:
        print(f"❌ Missing features: {missing}")
    else:
        print(f"✅ All {len(TemporalEncoder.FEATURE_NAMES)} features present")    
    print("\nNaN rates:")
    for f in TemporalEncoder.FEATURE_NAMES:
        rate = df[f].isna().mean() * 100
        bar = "█" * int(rate) if rate > 0 else ""
        print(f"  {f:20s}: {rate:5.1f}% {bar}")
    print("\nChronological splits:")
    for chunk_id, cfg in TemporalEncoder.CHUNK_CONFIG.items():
        for split, (y1, y2) in [("train", cfg["train"]), ("val", cfg["val"]), ("test", cfg["test"])]:
            mask = (df["date"].dt.year >= y1) & (df["date"].dt.year <= y2)
            print(f"  Chunk {chunk_id} {split:5s} ({y1}-{y2}): {mask.sum():>10,} rows, {df.loc[mask, 'ticker'].nunique():>5} tickers")
    print(f"\n✅ Inspection complete. Batch size={config.batch_size}, Workers={config.num_workers}\n")


def cmd_hpo(config, chunk_id):
    try:
        import optuna
    except ImportError:
        print("❌ optuna not installed.")
        return

    chunk_cfg = TemporalEncoder.CHUNK_CONFIG[chunk_id]
    chunk_label = chunk_cfg["label"]
    output_dir = Path(config.output_dir) / "codeResults" / "TemporalEncoder" / "hpo"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_features_df(config.features_path)
    train_dataset = MarketSequenceDataset(df, seq_len=config.seq_len, years=chunk_cfg["train"], max_rows=50000, num_workers=4)
    val_dataset = MarketSequenceDataset(df, seq_len=config.seq_len, years=chunk_cfg["val"], max_rows=10000, num_workers=4)

    def objective(trial):
        tc = TemporalEncoderConfig(**{k: v for k, v in asdict(config).items()
            if k not in ("n_layers", "n_heads", "d_model", "dropout", "attention_dropout",
                         "learning_rate", "weight_decay", "warmup_steps", "batch_size")})
        tc.n_layers = trial.suggest_int("n_layers", 2, 6)
        tc.n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
        tc.d_model = trial.suggest_categorical("d_model", [64, 128, 256])
        tc.dropout = trial.suggest_float("dropout", 0.05, 0.3)
        tc.attention_dropout = trial.suggest_float("attention_dropout", 0.05, 0.2)
        tc.learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
        tc.weight_decay = trial.suggest_float("weight_decay", 5e-6, 5e-4, log=True)
        tc.warmup_steps = trial.suggest_categorical("warmup_steps", [1000, 2000, 4000])
        tc.batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        tc.epochs = 15
        tc.device = config.device
        tc.num_workers = 4

        if tc.d_model % tc.n_heads != 0:
            return float("inf")

        # hpo_normalizer = FeatureNormalizer()
        # hpo_normalizer.fit(train_dataset.get_raw_sequences())
        # train_dataset.normalizer = hpo_normalizer
        # val_dataset.normalizer = hpo_normalizer
        hpo_normalizer = FeatureNormalizer()
        hpo_normalizer.fit_from_dataset(train_dataset)
        train_dataset.normalizer = hpo_normalizer
        val_dataset.normalizer = hpo_normalizer

        model = TemporalEncoder(tc).to(tc.device)
        train_loader = DataLoader(train_dataset, batch_size=tc.batch_size, shuffle=True,
                                   num_workers=0, drop_last=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=tc.batch_size, shuffle=False,
                                 num_workers=0, pin_memory=True)
        total_steps = len(train_loader) * tc.epochs
        optimizer, scheduler = create_optimizer_and_scheduler(model, total_steps)
        scaler = torch.cuda.amp.GradScaler() if config.device == "cuda" else None

        for epoch in range(tc.epochs):
            train_epoch(model, train_loader, optimizer, scaler, tc.device)
            scheduler.step()
        val_loss = validate(model, val_loader, tc.device)
        if np.isnan(val_loss) or np.isinf(val_loss):
            return float("inf")
        return float(val_loss)

    study = optuna.create_study(direction="minimize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=config.hpo_n_startup),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        storage=f"sqlite:///{output_dir}/temporal_encoder_hpo.db",
        study_name=f"temporal_encoder_{chunk_label}", load_if_exists=True)

    print(f"Running {config.hpo_trials} HPO trials for Chunk {chunk_id}...")
    study.optimize(objective, n_trials=config.hpo_trials, show_progress_bar=True)

    best = study.best_params
    best_path = output_dir / f"best_params_{chunk_label}.json"
    with open(best_path, "w") as f:
        json.dump({"params": best, "value": study.best_value}, f, indent=2)
    print(f"\nBest params: {best}\nBest val loss: {study.best_value:.6f}")


def cmd_train_best(config, chunk_id):
    chunk_cfg = TemporalEncoder.CHUNK_CONFIG[chunk_id]
    chunk_label = chunk_cfg["label"]

    hpo_dir = Path(config.output_dir) / "codeResults" / "TemporalEncoder" / "hpo"
    best_params_path = hpo_dir / f"best_params_{chunk_label}.json"
    if best_params_path.exists():
        with open(best_params_path) as f:
            best = json.load(f)
        print(f"Loaded best HPO params from {best_params_path}")
        for k, v in best["params"].items():
            setattr(config, k, v)
    else:
        print(f"No HPO results — using default params.")

    # HPO batch size was optimised on tiny sampled data.
    # It must not control full training on millions of windows.
    cli_batch_size = getattr(config, "_cli_batch_size", 0)
    cli_epochs = getattr(config, "_cli_epochs", 0)
    cli_patience = getattr(config, "_cli_early_stop_patience", 0)

    if cli_batch_size and cli_batch_size > 0:
        config.batch_size = int(cli_batch_size)
    elif config.batch_size < 1024:
        print(f"  Overriding HPO batch_size={config.batch_size} -> 2048 for full training")
        config.batch_size = 2048
    if cli_epochs and cli_epochs > 0:
        config.epochs = int(cli_epochs)
    elif config.epochs > 50:
        print(f"  Overriding epochs={config.epochs} -> 40 for full training")
        config.epochs = 40
    if cli_patience and cli_patience > 0:
        config.early_stop_patience = int(cli_patience)
    elif config.early_stop_patience > 10:
        print(f"  Overriding early_stop_patience={config.early_stop_patience} -> 8")
        config.early_stop_patience = 8
    print(
        f"  Runtime training config: batch_size={config.batch_size}, "
        f"epochs={config.epochs}, patience={config.early_stop_patience}, "
        f"workers={config.num_workers}"
    )
    # upppppp
    output_dir = Path(config.output_dir) / "models" / "TemporalEncoder" / chunk_label
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "effective_config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    print(f"Loading features for Chunk {chunk_id}...")
    df = load_features_df(config.features_path)

    train_dataset = MarketSequenceDataset(
        df, seq_len=config.seq_len, years=chunk_cfg["train"],
        max_rows=config.max_train_rows if config.max_train_rows > 0 else 0,
        num_workers=config.num_workers)
    val_dataset = MarketSequenceDataset(
        df, seq_len=config.seq_len, years=chunk_cfg["val"],
        num_workers=config.num_workers)

    print(f"  Train samples: {len(train_dataset):,}  |  Val samples: {len(val_dataset):,}")

    # print("  Fitting feature normalizer...")
    # normalizer = FeatureNormalizer()
    # normalizer.fit(train_dataset.get_raw_sequences())
    # train_dataset.normalizer = normalizer
    # val_dataset.normalizer = normalizer
    print("  Fitting feature normalizer...")
    normalizer = FeatureNormalizer()
    normalizer.fit_from_dataset(train_dataset)
    train_dataset.normalizer = normalizer
    val_dataset.normalizer = normalizer
    normalizer_path = output_dir / "normalizer.npz"
    normalizer.save(normalizer_path)
    print(f"  Saved train-only normalizer: {normalizer_path}")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                               num_workers=config.num_workers, drop_last=True,                               pin_memory=True, prefetch_factor=config.prefetch_factor,
                               persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                             num_workers=config.num_workers, pin_memory=True,
                             prefetch_factor=config.prefetch_factor,
                             persistent_workers=True)

    model = TemporalEncoder(config).to(config.device)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    summary = train_model(model, train_loader, val_loader, chunk_id, output_dir)

    frozen_dir = output_dir / "model_freezed"
    frozen_dir.mkdir(exist_ok=True)
    model.eval()
    model.save(str(frozen_dir / "model.pt"))

    unfrozen_dir = output_dir / "model_unfreezed"
    unfrozen_dir.mkdir(exist_ok=True)
    model.save(str(unfrozen_dir / "model.pt"))
    with open(unfrozen_dir / "UNFREEZE_NOTE.txt", "w") as f:
        f.write("Unfrozen model. Load with TemporalEncoder.load() to continue training.\n")

    print(f"\nTraining complete. Models saved to {output_dir}")
    print(f"Best val loss: {summary['best_val_loss']:.6f}\n")

    for split in ["train", "val", "test"]:
        cmd_embed(config, chunk_id, split)


# @torch.no_grad()
# def cmd_embed(config, chunk_id, split):
#     chunk_cfg = TemporalEncoder.CHUNK_CONFIG[chunk_id]
#     chunk_label = chunk_cfg["label"]

#     model_path = (Path(config.output_dir) / "models" / "TemporalEncoder" /
#                   chunk_label / "model_freezed" / "model.pt")
#     if not model_path.exists():
#         print(f"  Model not found: {model_path}")
#         return

#     print(f"Loading model from {model_path}")
#     model = TemporalEncoder.load(str(model_path), device=config.device)
#     model.eval()

#     df = load_features_df(config.features_path)
#     year_range = chunk_cfg[split]
#     dataset = MarketSequenceDataset(df, seq_len=config.seq_len, years=year_range, num_workers=config.num_workers)

#     normalizer = FeatureNormalizer()
#     normalizer.fit(dataset.get_raw_sequences())

#     loader = DataLoader(dataset, batch_size=min(config.batch_size, 256), shuffle=False,
#                          num_workers=config.num_workers, pin_memory=True)

#     emb_dir = Path(config.output_dir) / "embeddings" / "TemporalEncoder"
#     xai_dir = Path(config.output_dir) / "results" / "TemporalEncoder" / "xai"
#     emb_dir.mkdir(parents=True, exist_ok=True)
#     xai_dir.mkdir(parents=True, exist_ok=True)

#     all_embeddings = []
#     all_attention_weights = []
#     all_gradient_importance = []

#     print(f"Generating XAI-enhanced embeddings for Chunk {chunk_id} {split}...")
#     n_processed = 0
#     max_xai_samples = getattr(config, 'xai_sample_size', 1000)

#     for batch in tqdm(loader, desc="  Embedding + XAI"):
#         x = batch.to(config.device, non_blocking=True)
#         x_norm = normalizer.transform(x) if normalizer.mean is not None else x
#         with torch.no_grad():
#             output = model(x_norm)
#         emb = output["attention_pooled"].cpu().numpy()
#         all_embeddings.append(emb)

#         if n_processed < max_xai_samples:
#             x_xai = x_norm[:min(10, x_norm.size(0))].clone().detach().requires_grad_(True)
#             with torch.enable_grad():
#                 out_xai = model(x_xai)
#                 score = out_xai["attention_pooled"].sum(dim=1).mean()
#                 score.backward()
#             grads = x_xai.grad.abs().mean(dim=(0, 1)).cpu().numpy()
#             all_gradient_importance.append(grads)

#             query = model.pooling_query.expand(x_xai.size(0), -1, -1)
#             _, attn_weights = model.attention_pooling(query, out_xai["sequence"], out_xai["sequence"])
#             attn_weights = attn_weights.squeeze(1).cpu().numpy()
#             all_attention_weights.append(attn_weights)
#             n_processed += x_xai.size(0)
#             x_xai.grad = None

#     embeddings = np.concatenate(all_embeddings, axis=0)
#     emb_path = emb_dir / f"{chunk_label}_{split}_embeddings.npy"
#     np.save(str(emb_path), embeddings)
#     print(f"  Embeddings: {embeddings.shape} → {emb_path}")

#     manifest_path = emb_dir / f"{chunk_label}_{split}_manifest.csv"
#     # Rebuild manifest from the dataset ordering
#     manifest_records = []
#     for idx in range(len(dataset)):
#         seq = dataset.sequences_array[idx]
#         # Get ticker and date from the original dataframe at this index
#         pass

#     if all_attention_weights:
#         attn_array = np.concatenate(all_attention_weights, axis=0)
#         np.save(str(xai_dir / f"{chunk_label}_{split}_attention_weights.npy"), attn_array)
#         attn_df = pd.DataFrame(attn_array, columns=[f"timestep_{i}" for i in range(attn_array.shape[1])])
#         attn_df.to_csv(xai_dir / f"{chunk_label}_{split}_attention_weights.csv", index=False)
#         avg_attention = attn_array.mean(axis=0)
#         print(f"  Attention weights saved. Top timesteps: {list(np.argsort(avg_attention)[-5:][::-1])}")

#     if all_gradient_importance:
#         grad_array = np.stack(all_gradient_importance, axis=0)
#         grad_mean = grad_array.mean(axis=0)
#         np.save(str(xai_dir / f"{chunk_label}_{split}_feature_importance.npy"), grad_mean)
#         importance_df = pd.DataFrame({
#             "feature": TemporalEncoder.FEATURE_NAMES,
#             "importance": grad_mean,
#             "importance_pct": (grad_mean / grad_mean.sum() * 100).round(1)
#         }).sort_values("importance", ascending=False)
#         importance_df.to_csv(xai_dir / f"{chunk_label}_{split}_feature_importance.csv", index=False)
#         print(f"  Feature importance saved. Top: {importance_df.iloc[0]['feature']} ({importance_df.iloc[0]['importance_pct']}%)")

#     print(f"  XAI complete for {chunk_label}_{split}\n")
@torch.no_grad()
def cmd_embed(config, chunk_id, split):
    chunk_cfg = TemporalEncoder.CHUNK_CONFIG[chunk_id]
    chunk_label = chunk_cfg["label"]

    model_dir = Path(config.output_dir) / "models" / "TemporalEncoder" / chunk_label
    model_path = model_dir / "model_freezed" / "model.pt"

    if not model_path.exists():
        fallback = model_dir / "best_model.pt"
        if fallback.exists():
            print(f"  Frozen model not found. Using best checkpoint: {fallback}")
            model_path = fallback
        else:
            print(f"  Model not found: {model_path}")
            return

    print(f"Loading model from {model_path}")
    model = TemporalEncoder.load(str(model_path), device=config.device)
    model.eval()

    # Keep architecture from checkpoint, but use runtime paths/device/batch settings.
    model.config.repo_root = config.repo_root
    model.config.features_path = config.features_path
    model.config.output_dir = config.output_dir
    model.config.device = config.device

    if getattr(config, "_cli_batch_size", 0):
        model.config.batch_size = int(config._cli_batch_size)
    else:
        model.config.batch_size = max(int(config.batch_size), 2048)

    model.config.num_workers = int(config.num_workers)
    model.config.prefetch_factor = int(config.prefetch_factor)

    df = load_features_df(model.config.features_path)
    year_range = chunk_cfg[split]

    dataset = MarketSequenceDataset(
        df,
        seq_len=model.config.seq_len,
        years=year_range,
        num_workers=model.config.num_workers,
    )

    normalizer = get_or_fit_train_normalizer(model.config, chunk_id, df=df)

    loader_kwargs = {
        "batch_size": int(model.config.batch_size),
        "shuffle": False,
        "num_workers": int(model.config.num_workers),
        "pin_memory": str(model.config.device).startswith("cuda"),
        "drop_last": False,
    }

    if int(model.config.num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(model.config.prefetch_factor)

    loader = DataLoader(dataset, **loader_kwargs)

    emb_dir = Path(model.config.output_dir) / "embeddings" / "TemporalEncoder"
    xai_dir = Path(model.config.output_dir) / "results" / "TemporalEncoder" / "xai"
    emb_dir.mkdir(parents=True, exist_ok=True)
    xai_dir.mkdir(parents=True, exist_ok=True)

    emb_path = emb_dir / f"{chunk_label}_{split}_embeddings.npy"
    manifest_path = emb_dir / f"{chunk_label}_{split}_manifest.csv"

    n_samples = len(dataset)
    d_model = int(model.config.d_model)

    print(
        f"Generating FAST embeddings for {chunk_label}_{split}: "
        f"samples={n_samples:,}, batch_size={model.config.batch_size}, d_model={d_model}"
    )

    embeddings = np.lib.format.open_memmap(
        str(emb_path),
        mode="w+",
        dtype=np.float32,
        shape=(n_samples, d_model),
    )

    write_pos = 0

    for batch in tqdm(loader, desc=f"  Embedding {chunk_label}_{split}", unit="batch"):
        x = batch.to(model.config.device, non_blocking=True)
        x = normalizer.transform(x)

        with torch.cuda.amp.autocast(enabled=(model.config.mixed_precision and model.config.device == "cuda")):
            output = model(x)

        emb = output["attention_pooled"].detach().float().cpu().numpy().astype(np.float32)
        n = emb.shape[0]
        embeddings[write_pos:write_pos + n] = emb
        write_pos += n

        del x, output, emb

    embeddings.flush()

    manifest = dataset.get_manifest()
    manifest.to_csv(manifest_path, index=False)

    print(f"  Embeddings saved: {emb_path} shape=({n_samples}, {d_model})")
    print(f"  Manifest saved:   {manifest_path} rows={len(manifest):,}")

    if not getattr(config, "_skip_xai", False):
        print("  Saving lightweight encoder attention XAI sample...")
        attn_rows = []
        max_xai = min(int(model.config.xai_sample_size), len(dataset))

        xai_loader = DataLoader(
            dataset,
            batch_size=min(256, int(model.config.batch_size)),
            shuffle=False,
            num_workers=0,
            pin_memory=str(model.config.device).startswith("cuda"),
        )

        seen = 0
        for batch in tqdm(xai_loader, desc="  Encoder XAI sample", leave=False):
            if seen >= max_xai:
                break

            x = batch.to(model.config.device, non_blocking=True)
            x = normalizer.transform(x)

            output = model(x)
            query = model.pooling_query.expand(x.size(0), -1, -1)
            _, attn_weights = model.attention_pooling(query, output["sequence"], output["sequence"])
            attn = attn_weights.squeeze(1).detach().float().cpu().numpy()

            take = min(attn.shape[0], max_xai - seen)
            attn_rows.append(attn[:take])
            seen += take

        if attn_rows:
            attn_array = np.concatenate(attn_rows, axis=0)
            np.save(str(xai_dir / f"{chunk_label}_{split}_attention_weights.npy"), attn_array)
            attn_df = pd.DataFrame(attn_array, columns=[f"timestep_{i}" for i in range(attn_array.shape[1])])
            attn_df.to_csv(xai_dir / f"{chunk_label}_{split}_attention_weights.csv", index=False)
            avg_attention = attn_array.mean(axis=0)
            print(f"  XAI attention sample saved. Top timesteps: {list(np.argsort(avg_attention)[-5:][::-1])}")

    print(f"  Complete: {chunk_label}_{split}\n")

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Shared Temporal Attention Encoder")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("inspect")
    p = sub.add_parser("hpo")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p = sub.add_parser("train-best")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    sub.add_parser("train-best-all")
    p = sub.add_parser("embed")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    sub.add_parser("embed-all")

    # for sp in [p for p in sub.choices.values() if p is not None]:
    #     sp.add_argument("--repo-root", type=str, default="")
    #     sp.add_argument("--features-path", type=str, default="")
    #     sp.add_argument("--output-dir", type=str, default="")
    #     sp.add_argument("--device", type=str, default="cuda")
    #     sp.add_argument("--max-train-rows", type=int, default=0)
    for sp in [p for p in sub.choices.values() if p is not None]:
        sp.add_argument("--repo-root", type=str, default="")
        sp.add_argument("--features-path", type=str, default="")
        sp.add_argument("--output-dir", type=str, default="")
        sp.add_argument("--device", type=str, default="cuda")
        sp.add_argument("--max-train-rows", type=int, default=0)
        sp.add_argument("--batch-size", type=int, default=0)
        sp.add_argument("--epochs", type=int, default=0)
        sp.add_argument("--early-stop-patience", type=int, default=0)
        sp.add_argument("--num-workers", type=int, default=0)
        sp.add_argument("--prefetch-factor", type=int, default=0)
        sp.add_argument("--skip-xai", action="store_true")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    config = TemporalEncoderConfig()
    if hasattr(args, "repo_root") and args.repo_root:
        config.repo_root = args.repo_root
        config.__post_init__()
    if hasattr(args, "features_path") and args.features_path:
        config.features_path = args.features_path
    if hasattr(args, "output_dir") and args.output_dir:
        config.output_dir = args.output_dir
    if hasattr(args, "device"):
        config.device = args.device
    if hasattr(args, "max_train_rows") and args.max_train_rows > 0:
        config.max_train_rows = args.max_train_rows
    if hasattr(args, "batch_size") and args.batch_size > 0:
        config.batch_size = int(args.batch_size)
        config._cli_batch_size = int(args.batch_size)
    else:
        config._cli_batch_size = 0

    if hasattr(args, "epochs") and args.epochs > 0:
        config.epochs = int(args.epochs)
        config._cli_epochs = int(args.epochs)
    else:
        config._cli_epochs = 0

    if hasattr(args, "early_stop_patience") and args.early_stop_patience > 0:
        config.early_stop_patience = int(args.early_stop_patience)
        config._cli_early_stop_patience = int(args.early_stop_patience)
    else:
        config._cli_early_stop_patience = 0

    if hasattr(args, "num_workers") and args.num_workers > 0:
        config.num_workers = int(args.num_workers)

    if hasattr(args, "prefetch_factor") and args.prefetch_factor > 0:
        config.prefetch_factor = int(args.prefetch_factor)

    config._skip_xai = bool(getattr(args, "skip_xai", False))

    # Fix argparse duplication — handle train-best manually
    cmd = args.command
    if cmd == "inspect":
        cmd_inspect(config)
    elif cmd == "hpo":
        cmd_hpo(config, args.chunk)
    elif cmd == "train-best":
        cmd_train_best(config, args.chunk)
    elif cmd == "train-best-all":
        for cid in [1, 2, 3]:
            print(f"\n{'='*60}\nTRAINING CHUNK {cid}\n{'='*60}")
            cmd_train_best(config, cid)
    elif cmd == "embed":
        cmd_embed(config, args.chunk, args.split)
    elif cmd == "embed-all":
        for cid in [1, 2, 3]:
            for split in ["train", "val", "test"]:
                print(f"\n{'='*60}\nEMBEDDING Chunk {cid} {split}\n{'='*60}")
                cmd_embed(config, cid, split)


if __name__ == "__main__":
    main()

# ── Run instructions ──────────────────────────────────────────
# python code/encoders/temporal_encoder.py inspect
# python code/encoders/temporal_encoder.py train-best --chunk 1 --device cuda
# python code/encoders/temporal_encoder.py train-best-all --device cuda
# python code/encoders/temporal_encoder.py embed-all --device cuda 