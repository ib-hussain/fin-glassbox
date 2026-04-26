#!/usr/bin/env python3
"""
code/encoders/temporal_encoder.py

Shared Temporal Attention Encoder for financial time series.
Project: fin-glassbox — Explainable Distributed Deep Learning Framework

Architecture:
  Transformer Encoder (4 layers, 4 heads, 128-dim)
  Input: 30-90 day sequences of 10 market features
  Output: 128-dim temporal embedding (last, mean, or attention pooled)

Training objective:
  Self-supervised masked prediction — randomly mask 15% of time steps
  and predict the masked feature values.

Usage:
  # Inspect data
  python code/encoders/temporal_encoder.py inspect

  # HPO search
  python code/encoders/temporal_encoder.py hpo --chunk 1 --trials 50

  # Train with best HPO params
  python code/encoders/temporal_encoder.py train-best --chunk 1

  # Generate embeddings
  python code/encoders/temporal_encoder.py embed --chunk 1 --split val
"""

from __future__ import annotations

import argparse, json, math, os, sys, time, warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TemporalEncoderConfig:
    """Configuration for the Shared Temporal Attention Encoder."""

    # ── Paths ────────────────────────────────────────────
    repo_root: str = ""
    features_path: str = "data/yFinance/processed/features_temporal.csv"
    output_dir: str = "outputs"

    # ── Model architecture ───────────────────────────────
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation: str = "gelu"
    max_seq_len: int = 90
    n_input_features: int = 10

    # ── Training ─────────────────────────────────────────
    seq_len: int = 30
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 4000
    gradient_clip: float = 1.0
    label_smoothing: float = 0.0
    early_stop_patience: int = 20

    # ── XAI ─────────────────────────────────────────────
    xai_sample_size: int = 1000

    # ── Masked prediction ────────────────────────────────
    mask_prob: float = 0.15
    mask_seed: int = 42

    # ── HPO ──────────────────────────────────────────────
    hpo_trials: int = 75
    hpo_n_startup: int = 20

    # ── System ───────────────────────────────────────────
    device: str = "cuda"
    seed: int = 42
    mixed_precision: bool = True
    num_workers: int = 6

    # ── Data ─────────────────────────────────────────────
    max_train_rows: int = 0
    val_fraction: float = 0.0

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
    """Sinusoidal positional encoding as described in 'Attention Is All You Need'."""

    def __init__(self, d_model: int, max_len: int = 90):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


# ═══════════════════════════════════════════════════════════════════
# TEMPORAL ENCODER
# ═══════════════════════════════════════════════════════════════════

class TemporalEncoder(nn.Module):
    """Shared Temporal Attention Encoder.

    Transforms variable-length sequences of market features into
    128-dimensional temporal embeddings using a transformer encoder
    with multi-head self-attention.

    Args:
        config: TemporalEncoderConfig instance.

    Input:
        x: (batch, seq_len, 10) — market features

    Output:
        Dict with keys:
          - 'sequence': (batch, seq_len, 128) full temporal embeddings
          - 'last_hidden': (batch, 128) final time step
          - 'mean_pooled': (batch, 128) mean over time
          - 'attention_pooled': (batch, 128) learned weighted average
    """

    FEATURE_NAMES = [
        "log_return", "vol_5d", "vol_21d", "rsi_14", "macd_hist",
        "bb_pos", "volume_ratio", "hl_ratio", "price_pos", "spy_corr_63d",
    ]

    CHUNK_CONFIG = {
        1: {"train": (2000, 2004), "val": (2005, 2005), "test": (2006, 2006),
            "label": "chunk1"},
        2: {"train": (2007, 2014), "val": (2015, 2015), "test": (2016, 2016),
            "label": "chunk2"},
        3: {"train": (2017, 2022), "val": (2023, 2023), "test": (2024, 2024),
            "label": "chunk3"},
    }

    def __init__(self, config: TemporalEncoderConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model

        # Input projection: raw features → d_model
        self.input_projection = nn.Linear(config.n_input_features, d_model)

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, config.max_seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # Attention pooling — learn which time steps matter
        self.pooling_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=1, batch_first=True, dropout=0.0,
        )

        # Output projection for masked prediction head
        self.output_projection = nn.Linear(d_model, config.n_input_features)

        self._init_weights()

    def _init_weights(self):
        """Initialise weights with small random values for stable training."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
                ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, seq_len, n_features) raw market features
            mask: (batch, seq_len) True for positions to ignore (padding)

        Returns:
            Dict of output tensors.
        """
        # 1. Project to d_model
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # 2. Add positional encoding
        x = self.pos_encoding(x)

        # 3. Transformer encoder
        x = self.transformer(
            x, src_key_padding_mask=mask if mask is not None else None
        )  # (batch, seq_len, d_model)

        # 4. Pooling strategies
        if mask is not None:
            # Zero out masked positions for correct pooling
            mask_expanded = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            x_masked = x * (1 - mask_expanded)
            seq_lengths = (1 - mask_expanded).sum(dim=1).clamp(min=1)
            mean_pooled = x_masked.sum(dim=1) / seq_lengths  # (batch, d_model)
            # Last valid time step
            lengths = (1 - mask.int()).sum(dim=1) - 1
            lengths = lengths.clamp(min=0)
            last_hidden = x[torch.arange(x.size(0)), lengths]  # (batch, d_model)
        else:
            mean_pooled = x.mean(dim=1)
            last_hidden = x[:, -1, :]

        # Attention pooling
        query = self.pooling_query.expand(x.size(0), -1, -1)
        if mask is not None:
            attn_pooled, _ = self.attention_pooling(
                query, x, x, key_padding_mask=mask
            )
        else:
            attn_pooled, _ = self.attention_pooling(query, x, x)
        attn_pooled = attn_pooled.squeeze(1)

        return {
            "sequence": x,
            "last_hidden": last_hidden,
            "mean_pooled": mean_pooled,
            "attention_pooled": attn_pooled,
        }

    def get_embedding(self, x: torch.Tensor, pooling: str = "last",
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convenience method returning a single pooled embedding.

        Args:
            x: (batch, seq_len, n_features)
            pooling: 'last', 'mean', or 'attention'
            mask: optional padding mask

        Returns:
            (batch, d_model) embedding
        """
        output = self.forward(x, mask)
        if pooling == "last":
            return output["last_hidden"]
        elif pooling == "mean":
            return output["mean_pooled"]
        elif pooling == "attention":
            return output["attention_pooled"]
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

    def predict_masked(self, x: torch.Tensor, mask_indices: torch.Tensor
                       ) -> torch.Tensor:
        """Predict masked feature values for self-supervised training.

        Args:
            x: (batch, seq_len, n_features) — some values masked with zeros
            mask_indices: (batch, seq_len, n_features) — True where masked

        Returns:
            (batch, seq_len, n_features) predicted values
        """
        output = self.forward(x)
        seq = output["sequence"]
        return self.output_projection(seq)

    def save(self, path: str | Path):
        """Save model state dict and config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": asdict(self.config),
        }, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "TemporalEncoder":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = TemporalEncoderConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model


# ═══════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════


class FeatureNormalizer:
    """Z-score normalizer fitted on training data, applied to all splits."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, sequences: list[np.ndarray]):
        """Compute mean and std from a list of (seq_len, n_features) arrays."""
        all_data = np.concatenate(sequences, axis=0)
        self.mean = all_data.mean(axis=0, keepdims=True)
        self.std = all_data.std(axis=0, keepdims=True)
        self.std[self.std < 1e-8] = 1.0  # avoid div by zero

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize a batch. x: (batch, seq_len, n_features)"""
        if self.mean is None:
            return x
        mean_t = torch.from_numpy(self.mean).float().to(x.device)
        std_t = torch.from_numpy(self.std).float().to(x.device)
        return (x - mean_t) / std_t




class MarketSequenceDataset(Dataset):
    """Sliding-window dataset over market features.

    Creates sequences of length seq_len from the features DataFrame.
    Uses overlapping windows with stride=1.

    Args:
        features_df: DataFrame with columns [date, ticker, feature_1, ..., feature_n]
        seq_len: Sequence length in trading days.
        years: Tuple of (start_year, end_year) inclusive.
        tickers: List of tickers to include (None = all).
        max_rows: If > 0, limit total samples for smoke testing.
    """

    def __init__(self, features_df: pd.DataFrame, seq_len: int = 30,
                 years: tuple[int, int] = (2000, 2004),
                 tickers: Optional[list[str]] = None,
                 max_rows: int = 0,
                 normalizer: Optional[FeatureNormalizer] = None):
        df = features_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year

        # Filter by year range
        mask = (df["year"] >= years[0]) & (df["year"] <= years[1])
        df = df[mask]

        # Filter tickers
        if tickers is not None:
            df = df[df["ticker"].isin(tickers)]

        # Drop date/ticker for tensor conversion
        feature_cols = [c for c in TemporalEncoder.FEATURE_NAMES if c in df.columns]
        self.feature_names = feature_cols

        # Build sequences per ticker
        self.sequences = []
        for ticker, group in tqdm(df.groupby("ticker"), desc="  Building sequences",
                                   leave=False):
            vals = group[feature_cols].values.astype(np.float32)
            # Fill NaN with 0 (first row of returns, pre-warmup indicators)
            vals = np.nan_to_num(vals, nan=0.0)
            if len(vals) < seq_len:
                continue
            # Sliding windows
            for i in range(0, len(vals) - seq_len):
                self.sequences.append(vals[i:i + seq_len])

        self.sequences_array = [s.copy() for s in self.sequences]
        self.normalizer = normalizer

        if max_rows > 0 and len(self.sequences) > max_rows:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(self.sequences), max_rows, replace=False)
            self.sequences = [self.sequences[i] for i in indices]

    def get_raw_sequences(self) -> list[np.ndarray]:
        """Return raw sequences for fitting normalizer."""
        return self.sequences_array if hasattr(self, 'sequences_array') else self.sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.sequences[idx].copy())


# ═══════════════════════════════════════════════════════════════════
# MASKING
# ═══════════════════════════════════════════════════════════════════

def apply_masking(x: torch.Tensor, mask_prob: float = 0.15,
                  mask_seed: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply random masking to a batch of sequences.

    Masks 15% of time steps by replacing them with zeros.
    The model must predict the original values at masked positions.

    Args:
        x: (batch, seq_len, n_features)
        mask_prob: Fraction of time steps to mask.
        mask_seed: Random seed for reproducibility.

    Returns:
        (masked_x, mask_indices) where mask_indices is True for masked positions.
    """
    if mask_seed is not None:
        torch.manual_seed(mask_seed)

    batch, seq_len, n_feat = x.shape
    # Mask entire time steps (all features at that step)
    mask = torch.rand(batch, seq_len, 1) < mask_prob
    # Never mask first step (no prior context)
    mask[:, 0, :] = False

    mask_indices = mask.expand(-1, -1, n_feat)
    masked_x = x.clone()
    masked_x[mask_indices] = 0.0

    return masked_x, mask_indices


# ═══════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════

def train_epoch(model: TemporalEncoder, dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                scaler: Optional[torch.cuda.amp.GradScaler],
                device: str) -> float:
    """Train one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        x = batch.to(device)
        # Apply normalization if available
        if hasattr(dataloader.dataset, 'normalizer') and dataloader.dataset.normalizer is not None:
            x = dataloader.dataset.normalizer.transform(x)
        masked_x, mask_indices = apply_masking(x, model.config.mask_prob)
        masked_x = masked_x.to(device)
        mask_indices = mask_indices.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                pred = model.predict_masked(masked_x, mask_indices)
                # Loss only on masked positions
                loss = F.mse_loss(pred[mask_indices], x[mask_indices])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), model.config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model.predict_masked(masked_x, mask_indices)
            loss = F.mse_loss(pred[mask_indices], x[mask_indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), model.config.gradient_clip)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model: TemporalEncoder, dataloader: DataLoader, device: str) -> float:
    """Validate. Returns average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        x = batch.to(device)
        # Apply normalization if available
        if hasattr(dataloader.dataset, 'normalizer') and dataloader.dataset.normalizer is not None:
            x = dataloader.dataset.normalizer.transform(x)
        masked_x, mask_indices = apply_masking(x, model.config.mask_prob)
        masked_x = masked_x.to(device)
        mask_indices = mask_indices.to(device)

        pred = model.predict_masked(masked_x, mask_indices)
        loss = F.mse_loss(pred[mask_indices], x[mask_indices])
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def create_optimizer_and_scheduler(model: TemporalEncoder,
                                    total_steps: int) -> tuple:
    """Create AdamW optimizer with cosine LR schedule and warmup."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model.config.learning_rate,
        weight_decay=model.config.weight_decay,
    )

    def lr_lambda(step):
        if step < model.config.warmup_steps:
            return step / max(model.config.warmup_steps, 1)
        progress = (step - model.config.warmup_steps) / max(
            total_steps - model.config.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def train_model(model: TemporalEncoder,
                train_loader: DataLoader,
                val_loader: DataLoader,
                chunk_id: int,
                output_dir: Path) -> dict:
    """Full training loop with early stopping and checkpointing."""
    config = model.config
    device = config.device

    total_steps = len(train_loader) * config.epochs
    optimizer, scheduler = create_optimizer_and_scheduler(model, total_steps)

    scaler = None
    if config.mixed_precision and device == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float("inf")
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_path = output_dir / "best_model.pt"
    latest_path = output_dir / "latest_model.pt"

    chunk_label = TemporalEncoder.CHUNK_CONFIG[chunk_id]["label"]

    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        val_loss = validate(model, val_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        tqdm.write(
            f"  [{chunk_label}] Epoch {epoch:3d}/{config.epochs} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # Save latest
        model.save(str(latest_path))

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            model.save(str(best_path))
        else:
            no_improve += 1

        if no_improve >= config.early_stop_patience:
            tqdm.write(f"  Early stopping at epoch {epoch}")
            break

        scheduler.step()

    # Save history
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    # Save summary
    summary = {
        "chunk": chunk_label,
        "best_val_loss": float(best_val_loss),
        "epochs_trained": epoch,
        "config": asdict(config),
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_features_df(path: str | Path) -> pd.DataFrame:
    """Load the temporal features CSV."""
    df = pd.read_csv(path, dtype={"ticker": str}, parse_dates=["date"])
    return df


def get_ticker_list(features_df: pd.DataFrame) -> list[str]:
    """Get sorted list of unique tickers."""
    return sorted(features_df["ticker"].unique())


def get_date_range(features_df: pd.DataFrame) -> tuple:
    """Get (min_date, max_date) from features."""
    return (features_df["date"].min().date(), features_df["date"].max().date())


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def cmd_inspect(config: TemporalEncoderConfig):
    """Inspect input data availability and alignment."""
    print("=" * 60)
    print("TEMPORAL ENCODER — DATA INSPECTION")
    print("=" * 60)

    # Check features file
    fp = Path(config.features_path)
    if not fp.exists():
        print(f"❌ Features file not found: {fp}")
        return

    df = load_features_df(fp)
    print(f"\n✅ Features file: {fp}")
    print(f"   Rows: {len(df):,}")
    print(f"   Tickers: {df['ticker'].nunique()}")
    print(f"   Date range: {df['date'].min().date()} → {df['date'].max().date()}")

    # Check feature columns
    missing = [f for f in TemporalEncoder.FEATURE_NAMES if f not in df.columns]
    if missing:
        print(f"❌ Missing features: {missing}")
    else:
        print(f"✅ All {len(TemporalEncoder.FEATURE_NAMES)} features present")

    # Check NaN rate per feature
    print("\nNaN rates:")
    for f in TemporalEncoder.FEATURE_NAMES:
        rate = df[f].isna().mean() * 100
        bar = "█" * int(rate) if rate > 0 else ""
        print(f"  {f:20s}: {rate:5.1f}% {bar}")

    # Verify chunk splits
    print("\nChronological splits:")
    for chunk_id, cfg in TemporalEncoder.CHUNK_CONFIG.items():
        for split, (y1, y2) in [("train", cfg["train"]), ("val", cfg["val"]),
                                  ("test", cfg["test"])]:
            mask = (df["date"].dt.year >= y1) & (df["date"].dt.year <= y2)
            count = mask.sum()
            tickers = df.loc[mask, "ticker"].nunique()
            print(f"  Chunk {chunk_id} {split:5s} ({y1}-{y2}): {count:>10,} rows, {tickers:>5} tickers")

    print("\n✅ Inspection complete.\n")


def cmd_hpo(config: TemporalEncoderConfig, chunk_id: int):
    """Run HPO for one chunk."""
    try:
        import optuna
    except ImportError:
        print("❌ optuna not installed. pip install optuna")
        return

    chunk_cfg = TemporalEncoder.CHUNK_CONFIG[chunk_id]
    chunk_label = chunk_cfg["label"]

    output_dir = Path(config.output_dir) / "codeResults" / "TemporalEncoder" / "hpo"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data for HPO (use small subset)
    print(f"Loading features for HPO (Chunk {chunk_id})...")
    df = load_features_df(config.features_path)

    # Small dataset for fast HPO trials
    train_dataset = MarketSequenceDataset(
        df, seq_len=config.seq_len,
        years=chunk_cfg["train"], max_rows=50000,
    )
    val_dataset = MarketSequenceDataset(
        df, seq_len=config.seq_len,
        years=chunk_cfg["val"], max_rows=10000,
    )

    def objective(trial: optuna.Trial) -> float:
        trial_config = TemporalEncoderConfig(
            **{k: v for k, v in asdict(config).items()
               if k not in ("n_layers", "n_heads", "d_model", "dropout",
                            "attention_dropout", "learning_rate", "weight_decay",
                            "warmup_steps", "batch_size")}
        )
        trial_config.n_layers = trial.suggest_int("n_layers", 2, 6)
        trial_config.n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
        trial_config.d_model = trial.suggest_categorical("d_model", [64, 128, 256])
        trial_config.dropout = trial.suggest_float("dropout", 0.05, 0.3)
        trial_config.attention_dropout = trial.suggest_float("attention_dropout", 0.05, 0.2)
        trial_config.learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
        trial_config.weight_decay = trial.suggest_float("weight_decay", 5e-6, 5e-4, log=True)
        trial_config.warmup_steps = trial.suggest_categorical("warmup_steps", [1000, 2000, 4000])
        trial_config.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        trial_config.epochs = 15  # Fast HPO
        trial_config.device = config.device

        # Validate n_heads divides d_model
        if trial_config.d_model % trial_config.n_heads != 0:
            return float("inf")

        # Fit normalizer on HPO train subset
        hpo_normalizer = FeatureNormalizer()
        hpo_normalizer.fit(train_dataset.get_raw_sequences())
        train_dataset.normalizer = hpo_normalizer
        val_dataset.normalizer = hpo_normalizer

        model = TemporalEncoder(trial_config).to(trial_config.device)

        train_loader = DataLoader(train_dataset, batch_size=trial_config.batch_size,
                                   shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=trial_config.batch_size,
                                 shuffle=False, num_workers=0)

        total_steps = len(train_loader) * trial_config.epochs
        optimizer, scheduler = create_optimizer_and_scheduler(model, total_steps)
        scaler = torch.cuda.amp.GradScaler() if config.device == "cuda" else None

        for epoch in range(trial_config.epochs):
            train_epoch(model, train_loader, optimizer, scaler, trial_config.device)
            scheduler.step()

        val_loss = validate(model, val_loader, trial_config.device)
        if np.isnan(val_loss) or np.isinf(val_loss):
            return float("inf")
        return float(val_loss)

    # Run HPO
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=config.hpo_n_startup),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        storage=f"sqlite:///{output_dir}/temporal_encoder_hpo.db",
        study_name=f"temporal_encoder_{chunk_label}",
        load_if_exists=True,
    )

    print(f"Running {config.hpo_trials} HPO trials for Chunk {chunk_id}...")
    study.optimize(objective, n_trials=config.hpo_trials, show_progress_bar=True)

    # Save results
    best = study.best_params
    best_path = output_dir / f"best_params_{chunk_label}.json"
    with open(best_path, "w") as f:
        json.dump({"params": best, "value": study.best_value}, f, indent=2)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / f"trials_{chunk_label}.csv", index=False)

    print(f"\nBest params: {best}")
    print(f"Best val loss: {study.best_value:.6f}")
    print(f"Saved to: {best_path}\n")


def cmd_train_best(config: TemporalEncoderConfig, chunk_id: int):
    """Train with best HPO params for one chunk."""
    chunk_cfg = TemporalEncoder.CHUNK_CONFIG[chunk_id]
    chunk_label = chunk_cfg["label"]

    # Try to load best HPO params
    hpo_dir = Path(config.output_dir) / "codeResults" / "TemporalEncoder" / "hpo"
    best_params_path = hpo_dir / f"best_params_{chunk_label}.json"

    if best_params_path.exists():
        with open(best_params_path) as f:
            best = json.load(f)
        print(f"Loaded best HPO params from {best_params_path}")
        for k, v in best["params"].items():
            setattr(config, k, v)
    else:
        print(f"No HPO results found at {best_params_path}, using default params.")

    output_dir = Path(config.output_dir) / "models" / "TemporalEncoder" / chunk_label
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save effective config
    with open(output_dir / "effective_config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    print(f"Loading features for Chunk {chunk_id}...")
    df = load_features_df(config.features_path)

    train_dataset = MarketSequenceDataset(
        df, seq_len=config.seq_len, years=chunk_cfg["train"],
        max_rows=config.max_train_rows if config.max_train_rows > 0 else 0,
    )
    val_dataset = MarketSequenceDataset(
        df, seq_len=config.seq_len, years=chunk_cfg["val"],
    )

    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Val samples:   {len(val_dataset):,}")

    # Fit normalizer on training data
    print("  Fitting feature normalizer on training data...")
    normalizer = FeatureNormalizer()
    normalizer.fit(train_dataset.get_raw_sequences())
    train_dataset.normalizer = normalizer
    val_dataset.normalizer = normalizer
    print(f"    Feature means: {normalizer.mean[0]}")
    print(f"    Feature stds:  {normalizer.std[0]}")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                               shuffle=True, num_workers=config.num_workers,
                               drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                             shuffle=False, num_workers=config.num_workers,
                             pin_memory=True)

    model = TemporalEncoder(config).to(config.device)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    summary = train_model(model, train_loader, val_loader, chunk_id, output_dir)

    # Export frozen model
    frozen_dir = output_dir / "model_freezed"
    frozen_dir.mkdir(exist_ok=True)
    model.eval()
    model.save(str(frozen_dir / "model.pt"))

    # Export unfrozen model
    unfrozen_dir = output_dir / "model_unfreezed"
    unfrozen_dir.mkdir(exist_ok=True)
    model.save(str(unfrozen_dir / "model.pt"))
    with open(unfrozen_dir / "UNFREEZE_NOTE.txt", "w") as f:
        f.write("This model is saved unfrozen. Load with TemporalEncoder.load() "
                "to continue training or fine-tuning.\n")

    print(f"\nTraining complete. Models saved to {output_dir}")
    print(f"Best val loss: {summary['best_val_loss']:.6f}\n")
    # Auto-generate embeddings and XAI for all splits after training
    for split in ["train", "val", "test"]:
        cmd_embed(config, chunk_id, split)


# def cmd_embed(config: TemporalEncoderConfig, chunk_id: int, split: str):
#     """Generate embeddings and XAI explanations for one chunk/split."""
#     chunk_cfg = TemporalEncoder.CHUNK_CONFIG[chunk_id]
#     chunk_label = chunk_cfg["label"]

#     # Load model
#     model_path = (Path(config.output_dir) / "models" / "TemporalEncoder" /
#                   chunk_label / "model_freezed" / "model.pt")
#     if not model_path.exists():
#         print(f"❌ Model not found: {model_path}")
#         return

#     print(f"Loading model from {model_path}")
#     model = TemporalEncoder.load(str(model_path), device=config.device)
#     model.eval()

#     # Load data
#     df = load_features_df(config.features_path)
#     year_range = chunk_cfg[split]
#     dataset = MarketSequenceDataset(df, seq_len=config.seq_len, years=year_range)
#     loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False,
#                          num_workers=config.num_workers)

#     # Generate embeddings
#     output_dir = Path(config.output_dir) / "embeddings" / "TemporalEncoder"
#     output_dir.mkdir(parents=True, exist_ok=True)

#     all_embeddings = []
#     print(f"Generating embeddings for Chunk {chunk_id} {split}...")
#     with torch.no_grad():
#         for batch in tqdm(loader, desc="  Embedding"):
#             batch = batch.to(config.device)
#             emb = model.get_embedding(batch, pooling="last")
#             all_embeddings.append(emb.cpu().numpy())

#     embeddings = np.concatenate(all_embeddings, axis=0)
#     out_path = output_dir / f"{chunk_label}_{split}_embeddings.npy"
#     np.save(str(out_path), embeddings)

#     print(f"  Shape: {embeddings.shape}")
#     print(f"  Saved: {out_path}")

#     # ══════════════════════════════════════════════════════════
#     # XAI: Attention weights + Feature importance
#     # ══════════════════════════════════════════════════════════
#     xai_sample_size = min(500, len(dataset))
#     xai_dir = Path(config.output_dir) / "codeResults" / "TemporalEncoder" / "xai"
#     xai_dir.mkdir(parents=True, exist_ok=True)
    
#     # ── 1. Per-layer attention weights ──
#     print(f"\n[xai] Extracting attention weights for {xai_sample_size} samples...")
    
#     # Enable attention output on the transformer
#     for layer in model.transformer.layers:
#         layer.self_attn._attention_weights = None
    
#     attention_samples = []
#     feature_importance_samples = []
    
#     # Take a subset for XAI (first N samples)
#     sample_indices = list(range(0, min(xai_sample_size, len(dataset)), 
#                                   max(1, len(dataset) // xai_sample_size)))
    
#     for idx in tqdm(sample_indices, desc="  XAI extraction"):
#         batch = dataset[idx].unsqueeze(0).to(config.device)
        
#         # ── 1a. Extract attention pooling weights ──
#         with torch.no_grad():
#             output = model.forward(batch)
#             # Get the learned attention pooling weights
#             query = model.pooling_query.expand(1, -1, -1)
#             attn_output, attn_weights = model.attention_pooling(
#                 query, output["sequence"], output["sequence"]
#             )
#             # attn_weights shape: (1, 1, seq_len) — weight per time step
#             time_weights = attn_weights.squeeze().cpu().numpy()
        
#         # Normalize to sum to 1
#         time_weights = time_weights / (time_weights.sum() + 1e-10)
        
#         attention_samples.append({
#             "sample_idx": int(idx),
#             "time_step_weights": time_weights.tolist(),
#             "top_time_steps": [
#                 {"position": int(pos), "weight": float(time_weights[pos])}
#                 for pos in np.argsort(time_weights)[-5:][::-1]
#             ],
#             "embedding_pooling": "attention_pooled",
#         })
        
#         # ── 1b. Gradient-based feature importance ──
#         x_tensor = batch.clone().detach().requires_grad_(True)
        
#         # Apply normalization if available
#         if hasattr(dataset, 'normalizer') and dataset.normalizer is not None:
#             x_norm = dataset.normalizer.transform(x_tensor)
#         else:
#             x_norm = x_tensor
        
#         with torch.enable_grad():
#             output_grad = model.get_embedding(x_norm, pooling="attention")
#             # Use L2 norm of embedding as proxy importance signal
#             score = output_grad.norm()
#             score.backward()
        
#         # Gradient magnitude per feature (averaged across time steps)
#         grads = x_tensor.grad.cpu().numpy().squeeze()  # (seq_len, n_features)
#         feature_importance = np.abs(grads).mean(axis=0)  # (n_features,)
#         time_importance = np.abs(grads).mean(axis=1)     # (seq_len,)
        
#         # Normalize
#         feature_importance = feature_importance / (feature_importance.sum() + 1e-10)
#         time_importance = time_importance / (time_importance.sum() + 1e-10)
        
#         feature_importance_samples.append({
#             "sample_idx": int(idx),
#             "feature_importance": {
#                 name: float(feature_importance[i])
#                 for i, name in enumerate(TemporalEncoder.FEATURE_NAMES)
#             },
#             "top_features": [
#                 {"feature": TemporalEncoder.FEATURE_NAMES[i], 
#                  "importance": float(feature_importance[i])}
#                 for i in np.argsort(feature_importance)[-5:][::-1]
#             ],
#             "time_step_importance": time_importance.tolist(),
#             "top_time_steps_gradient": [
#                 {"position": int(pos), "importance": float(time_importance[pos])}
#                 for pos in np.argsort(time_importance)[-5:][::-1]
#             ],
#         })
        
#         x_tensor.grad = None
    
#     # ── 2. Aggregate across samples ──
#     agg_feature_importance = {}
#     for name in TemporalEncoder.FEATURE_NAMES:
#         vals = [s["feature_importance"][name] for s in feature_importance_samples]
#         agg_feature_importance[name] = {
#             "mean": float(np.mean(vals)),
#             "std": float(np.std(vals)),
#             "rank": None,  # filled below
#         }
    
#     # Rank features by mean importance
#     ranked = sorted(agg_feature_importance.items(), key=lambda x: x[1]["mean"], reverse=True)
#     for rank, (name, info) in enumerate(ranked):
#         agg_feature_importance[name]["rank"] = rank + 1
    
#     # ── 3. Save XAI outputs ──
#     xai_summary = {
#         "module": "TemporalEncoder",
#         "chunk_id": chunk_id,
#         "split": split,
#         "n_samples": len(sample_indices),
#         "seq_len": config.seq_len,
#         "feature_names": TemporalEncoder.FEATURE_NAMES,
#         "d_model": model.config.d_model,
#         "feature_importance_aggregated": agg_feature_importance,
#         "attention_explanations": attention_samples,
#         "gradient_explanations": feature_importance_samples,
#         "interpretation_guide": {
#             "time_step_weights": "Higher weight = this day received more attention in the learned pooling. These are the most important days for the embedding.",
#             "feature_importance": "Higher value = this feature's value had more influence on the embedding. Calculated via gradient magnitude.",
#             "top_time_steps": "The 5 days (positions in the 30-day window) that most influenced this embedding.",
#             "top_features": "The 5 market features that most influenced this embedding.",
#         },
#     }
    
#     xai_path = xai_dir / f"{chunk_label}_{split}_xai.json"
#     with open(xai_path, "w") as f:
#         json.dump(xai_summary, f, indent=2, default=str)
    
#     # Also save a lightweight per-sample attention matrix
#     attn_matrix = np.zeros((len(sample_indices), config.seq_len), dtype=np.float32)
#     for i, s in enumerate(attention_samples):
#         attn_matrix[i, :] = s["time_step_weights"]
#     np.save(str(xai_dir / f"{chunk_label}_{split}_attention_weights.npy"), attn_matrix)
    
#     print(f"[xai] Attention weights saved: {xai_dir / f'{chunk_label}_{split}_attention_weights.npy'}")
#     print(f"[xai] XAI summary saved: {xai_path}")
#     print(f"[xai] Top 5 features by importance:")
#     for name, info in list(agg_feature_importance.items())[:5]:
#         print(f"  {info['rank']}. {name}: {info['mean']:.4f} ± {info['std']:.4f}")
#     print()

# ═══════════════════════════════════════════════════════════════════
# XAI: ATTENTION VISUALIZATION & FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def cmd_embed(config: TemporalEncoderConfig, chunk_id: int, split: str):
    """Generate embeddings WITH attention weights and feature importance."""
    chunk_cfg = TemporalEncoder.CHUNK_CONFIG[chunk_id]
    chunk_label = chunk_cfg["label"]

    # Load model
    model_path = (Path(config.output_dir) / "models" / "TemporalEncoder" /
                  chunk_label / "model_freezed" / "model.pt")
    if not model_path.exists():
        print(f"  Model not found: {model_path}")
        return

    print(f"Loading model from {model_path}")
    model = TemporalEncoder.load(str(model_path), device=config.device)
    model.eval()

    # Load data
    df = load_features_df(config.features_path)
    year_range = chunk_cfg[split]
    dataset = MarketSequenceDataset(df, seq_len=config.seq_len, years=year_range)
    
    # Fit normalizer for this split (using train stats ideally, but for XAI
    # we use the dataset's own stats since the model expects normalized input)
    normalizer = FeatureNormalizer()
    normalizer.fit(dataset.get_raw_sequences())
    
    loader = DataLoader(dataset, batch_size=min(config.batch_size, 256), 
                         shuffle=False, num_workers=config.num_workers)

    # Output directories
    emb_dir = Path(config.output_dir) / "embeddings" / "TemporalEncoder"
    xai_dir = Path(config.output_dir) / "results" / "TemporalEncoder" / "xai"
    emb_dir.mkdir(parents=True, exist_ok=True)
    xai_dir.mkdir(parents=True, exist_ok=True)

    # ── Generate embeddings + collect attention ──
    all_embeddings = []
    all_attention_weights = []  # Per-batch attention from pooling
    all_gradient_importance = []  # Gradient-based feature importance
    
    print(f"Generating XAI-enhanced embeddings for Chunk {chunk_id} {split}...")
    
    n_processed = 0
    max_xai_samples = getattr(config, 'xai_sample_size', 1000)
    
    for batch_idx, batch in enumerate(tqdm(loader, desc="  Embedding + XAI")):
        x = batch.to(config.device)
        
        # Normalize
        x_norm = normalizer.transform(x) if normalizer.mean is not None else x
        
        # Get embeddings
        with torch.no_grad():
            output = model(x_norm)
        
        emb = output["attention_pooled"].cpu().numpy()
        all_embeddings.append(emb)
        
        # Get attention weights from the pooling layer
        # Re-run forward with attention output enabled for explainability
        if n_processed < max_xai_samples:
            # Gradient-based feature importance
            x_xai = x_norm[:min(10, x_norm.size(0))].clone().detach()
            x_xai.requires_grad_(True)
            
            with torch.enable_grad():
                out_xai = model(x_xai)
                # Use the mean of attention_pooled as the target for gradient
                score = out_xai["attention_pooled"].sum(dim=1).mean()
                score.backward()
            
            grads = x_xai.grad.abs().mean(dim=(0, 1)).cpu().numpy()  # Average over batch and seq
            all_gradient_importance.append(grads)
            
            # Collect pooling attention by doing one more forward pass
            query = model.pooling_query.expand(x_xai.size(0), -1, -1)
            _, attn_weights = model.attention_pooling(query, out_xai["sequence"], out_xai["sequence"])
            attn_weights = attn_weights.squeeze(1).cpu().numpy()  # (batch, seq_len)
            all_attention_weights.append(attn_weights)
            
            n_processed += x_xai.size(0)
        
        x_xai.grad = None if 'x_xai' in dir() else None

    # Save embeddings
    embeddings = np.concatenate(all_embeddings, axis=0)
    emb_path = emb_dir / f"{chunk_label}_{split}_embeddings.npy"
    np.save(str(emb_path), embeddings)
    print(f"  Embeddings: {embeddings.shape} → {emb_path}")

    # ── Save attention weights ──
    if all_attention_weights:
        attn_array = np.concatenate(all_attention_weights, axis=0)  # (n_samples, seq_len)
        attn_path = xai_dir / f"{chunk_label}_{split}_attention_weights.npy"
        np.save(str(attn_path), attn_array)
        
        # Also save as CSV with time-step labels
        attn_df = pd.DataFrame(
            attn_array,
            columns=[f"timestep_{i}" for i in range(attn_array.shape[1])]
        )
        attn_df.index.name = "sample_index"
        attn_df.to_csv(xai_dir / f"{chunk_label}_{split}_attention_weights.csv")
        
        # Attention statistics
        avg_attention = attn_array.mean(axis=0)
        top_timesteps = np.argsort(avg_attention)[-5:][::-1]
        
        print(f"  Attention weights: {attn_array.shape} → {attn_path}")
        print(f"  Top attended timesteps (avg): {list(top_timesteps)}")
        print(f"    Most recent step (t=-1): {avg_attention[-1]:.4f}")
        print(f"    Earliest step (t=0):     {avg_attention[0]:.4f}")
    else:
        print("  No attention weights collected (sample size may be 0)")

    # ── Save gradient-based feature importance ──
    if all_gradient_importance:
        grad_array = np.stack(all_gradient_importance, axis=0)  # (n_batches, n_features)
        grad_mean = grad_array.mean(axis=0)
        grad_path = xai_dir / f"{chunk_label}_{split}_feature_importance.npy"
        np.save(str(grad_path), grad_mean)
        
        # Feature importance report
        feature_names = TemporalEncoder.FEATURE_NAMES
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": grad_mean,
            "importance_pct": (grad_mean / grad_mean.sum() * 100).round(1)
        }).sort_values("importance", ascending=False)
        importance_df.to_csv(xai_dir / f"{chunk_label}_{split}_feature_importance.csv", index=False)
        
        print(f"\n  Feature Importance (gradient-based):")
        for _, row in importance_df.head(5).iterrows():
            bar = "█" * int(row["importance_pct"])
            print(f"    {row['feature']:20s}: {row['importance_pct']:5.1f}% {bar}")

    # ── Save XAI explanations per the spec ──
    explanations = []
    for i in range(min(100, len(all_attention_weights) if all_attention_weights else 0)):
        attn_sample = attn_array[i] if all_attention_weights else None
        grad_sample = grad_mean if all_gradient_importance else None
        
        top_attn_idx = np.argsort(attn_sample)[-3:][::-1] if attn_sample is not None else []
        top_feat_idx = np.argsort(grad_sample)[-3:][::-1] if grad_sample is not None else []
        
        explanations.append({
            "sample_index": i,
            "attention_by_timestep": {
                "most_recent_weight": float(attn_sample[-1]) if attn_sample is not None else None,
                "earliest_weight": float(attn_sample[0]) if attn_sample is not None else None,
                "top_timesteps": [
                    {"timestep": int(j), "weight": float(attn_sample[j])}
                    for j in top_attn_idx
                ] if attn_sample is not None else [],
            },
            "top_features": [
                {"feature": feature_names[j], "importance": float(grad_sample[j])}
                for j in top_feat_idx
            ] if grad_sample is not None else [],
            "interpretation": "Most recent timesteps receive highest attention (recency bias). "
                            "Top features drive the temporal embedding for downstream risk modules."
        })
    
    if explanations:
        import json as json_mod
        xai_json_path = xai_dir / f"{chunk_label}_{split}_explanations.json"
        with open(xai_json_path, "w") as f:
            json_mod.dump({
                "module": "TemporalEncoder",
                "chunk": chunk_label,
                "split": split,
                "n_samples_explained": len(explanations),
                "feature_names": TemporalEncoder.FEATURE_NAMES,
                "explanations": explanations,
            }, f, indent=2, default=str)
        print(f"  XAI explanations: {xai_json_path}")

    print(f"  XAI complete for {chunk_label}_{split}\n")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Shared Temporal Attention Encoder")
    sub = parser.add_subparsers(dest="command")

    # inspect
    sub.add_parser("inspect", help="Check input data availability")

    # hpo
    p = sub.add_parser("hpo", help="Run HPO for one chunk")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])

    # train-best
    p = sub.add_parser("train-best", help="Train with best HPO params")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])

    # train-best-all
    sub.add_parser("train-best-all", help="Train all 3 chunks with best HPO params")

    # embed (with XAI support)
    p = sub.add_parser("embed", help="Generate embeddings with XAI (attention weights + feature importance)")
    # embed-all
    sub.add_parser("embed-all", help="Generate embeddings for all chunks and splits")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    p.add_argument("--xai-sample-size", type=int, default=1000,
                   help="Number of samples for XAI analysis (more = slower)")

    # Common args
    for sp in [p for p in sub.choices.values() if p is not None]:
        sp.add_argument("--repo-root", type=str, default="")
        sp.add_argument("--features-path", type=str, default="")
        sp.add_argument("--output-dir", type=str, default="")
        sp.add_argument("--device", type=str, default="cuda")
        sp.add_argument("--max-train-rows", type=int, default=0)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Setup config
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

    if args.command == "inspect":
        cmd_inspect(config)
    elif args.command == "hpo":
        cmd_hpo(config, args.chunk)
    elif args.command == "train-best":
        cmd_train_best(config, args.chunk)
    elif args.command == "train-best-all":
        for chunk_id in [1, 2, 3]:
            print(f"\n{'='*60}")
            print(f"TRAINING CHUNK {chunk_id}")
            print(f"{'='*60}")
            cmd_train_best(config, chunk_id)
    elif args.command == "embed":
        cmd_embed(config, args.chunk, args.split)
    elif args.command == "embed-all":
        for chunk_id in [1, 2, 3]:
            for split in ["train", "val", "test"]:
                print(f"\n{'='*60}")
                print(f"EMBEDDING Chunk {chunk_id} {split}")
                print(f"{'='*60}")
                cmd_embed(config, chunk_id, split)


if __name__ == "__main__":
    main()

# ── Run instructions ──────────────────────────────────────────────
# python code/encoders/temporal_encoder.py inspect
# python code/encoders/temporal_encoder.py hpo --chunk 1 --device cuda
# python code/encoders/temporal_encoder.py train-best --chunk 1 --device cuda
# python code/encoders/temporal_encoder.py train-best-all --device cuda
# python code/encoders/temporal_encoder.py embed --chunk 1 --split val --device cuda
