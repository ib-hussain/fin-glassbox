#!/usr/bin/env python3
"""
code/gnn/stemgnn_contagion.py

StemGNN Contagion Risk Module for the fin-glassbox project.

Predicts multi-horizon contagion probability per stock:
  "Given recent returns of ALL other stocks, how likely is THIS stock
   to suffer an extreme negative return driven by spillover effects?"

Architecture:
  - Wraps baseline StemGNN Model from code/gnn/stemgnn_base_model.py
  - Replaces price-prediction fc layer with 3-horizon contagion head (5d, 20d, 60d)
  - Uses same 3 chronological chunks as Temporal Encoder
  - XAI: Level 1 (adjacency export), Level 2 (gradient edge importance),
         Level 3 (GNNExplainer subgraph mask, opt-in)

Input:  data/yFinance/processed/returns_panel_wide.csv  (2500 tickers x 6285 days)
Output: outputs/results/StemGNN/contagion_scores_*.csv
        outputs/models/StemGNN/chunk*/best_model.pt
        outputs/results/StemGNN/xai/*.json

Usage:
  python code/gnn/stemgnn_contagion.py inspect
  python code/gnn/stemgnn_contagion.py hpo --chunk 1 --trials 50 --device cuda
  python code/gnn/stemgnn_contagion.py train-best --chunk 1 --device cuda
  python code/gnn/stemgnn_contagion.py train-best-all --device cuda
  python code/gnn/stemgnn_contagion.py predict --chunk 1 --split test --device cuda
"""

from __future__ import annotations

import argparse, json, math, os, sys, time, warnings, pickle
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── GPU optimisations ─────────────────────────────
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# ── Import baseline StemGNN Model ─────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stemgnn_base_model import Model as StemGNNBase
from stemgnn_forecast_dataloader import normalized

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ContagionConfig:
    """Configuration for StemGNN Contagion Risk Module."""

    # ── Paths ────────────────────────────────────────────
    repo_root: str = ""
    returns_path: str = "data/yFinance/processed/returns_panel_wide.csv"
    output_dir: str = "outputs"

    # ── Model architecture ───────────────────────────────
    window_size: int = 30
    horizon: int = 1           # Predict 1 step (contagion probability, not price)
    multi_layer: int = 13      # StemGNN spectral-temporal blocks
    dropout_rate: float = 0.75
    leaky_rate: float = 0.2
    stack_cnt: int = 2         # Number of StemGNN blocks (fixed from baseline)

    # ── Contagion targets ────────────────────────────────
    contagion_horizons: List[int] = field(default_factory=lambda: [5, 20, 60])
    extreme_quantile: float = 0.05
    excess_threshold_std: float = 2.0

    # ── Training ─────────────────────────────────────────
    batch_size: int = 8        # Small — 2500x2500 matrices
    epochs: int = 100
    learning_rate: float = 0.001
    decay_rate: float = 0.5
    exponential_decay_step: int = 13
    gradient_clip: float = 1.0
    early_stop_patience: int = 20
    validate_freq: int = 1

    # ── HPO ──────────────────────────────────────────────
    hpo_trials: int = 50
    hpo_n_startup: int = 10

    # ── System ───────────────────────────────────────────
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 6
    optimizer: str = "RMSProp"
    norm_method: str = "z_score"

    # ── XAI ──────────────────────────────────────────────
    xai_sample_size: int = 500
    xai_top_influencers: int = 10
    enable_gnnexplainer: bool = False

    # ── Data ─────────────────────────────────────────────
    max_train_windows: int = 0  # 0 = use all
    chunk_id: int = 1

    def __post_init__(self):
        if self.repo_root:
            self.returns_path = str(Path(self.repo_root) / self.returns_path)
            self.output_dir = str(Path(self.repo_root) / self.output_dir)


# ═══════════════════════════════════════════════════════════════════
# CONTAGION-SPECIFIC MODEL (wraps baseline StemGNN)
# ═══════════════════════════════════════════════════════════════════

class ContagionStemGNN(nn.Module):
    """
    StemGNN adapted for contagion risk scoring.
    Wraps the baseline Model and adds multi-horizon contagion output heads.
    """

    def __init__(self, config: ContagionConfig, num_nodes: int):
        super().__init__()
        self.config = config
        self.num_nodes = num_nodes

        # Baseline StemGNN core (uses its latent correlation + spectral-temporal blocks)
        self.stemgnn = StemGNNBase(
            units=num_nodes,
            stack_cnt=config.stack_cnt,
            time_step=config.window_size,
            multi_layer=config.multi_layer,
            horizon=config.window_size,  # Output dim from stemgnn = window_size
            dropout_rate=config.dropout_rate,
            leaky_rate=config.leaky_rate,
            device=config.device,
        )

        # Replace the final fc layer with contagion heads
        # The stemgnn outputs (batch, window_size, num_nodes)
        # We pool across time and add per-horizon heads
        n_horizons = len(config.contagion_horizons)
        self.contagion_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.window_size, config.window_size // 2),
                nn.LeakyReLU(0.2),
                nn.Dropout(config.dropout_rate),
                nn.Linear(config.window_size // 2, 1),
            )
            for _ in range(n_horizons)
        ])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, num_nodes, window_size) — returns matrix

        Returns:
            Dict with:
              - contagion_scores: (batch, num_nodes, n_horizons)
              - attention: (num_nodes, num_nodes) learned adjacency
              - stemgnn_output: raw output from baseline model
        """
        # Run baseline StemGNN forward pass
        stemgnn_out, attention = self.stemgnn(x.permute(0, 2, 1))
        # stemgnn_out shape: (batch, num_nodes, window_size) or (batch, window_size, num_nodes)

        if stemgnn_out.dim() == 3 and stemgnn_out.size(-1) == self.num_nodes:
            stemgnn_out = stemgnn_out.permute(0, 2, 1)  # → (batch, num_nodes, window_size)

        # Apply contagion heads per stock
        batch_size = stemgnn_out.size(0)
        n_horizons = len(self.config.contagion_horizons)
        contagion_scores = torch.zeros(batch_size, self.num_nodes, n_horizons,
                                        device=x.device)

        for h in range(n_horizons):
            # Each head: pool temporal features → probability
            scores = self.contagion_heads[h](stemgnn_out)  # (batch, num_nodes, 1)
            contagion_scores[:, :, h] = torch.sigmoid(scores.squeeze(-1))

        return {
            "contagion_scores": contagion_scores,
            "attention": attention,
            "stemgnn_output": stemgnn_out,
        }

    def save(self, path: str | Path):
        """Save model state dict and config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": asdict(self.config),
            "num_nodes": self.num_nodes,
        }, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "ContagionStemGNN":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = ContagionConfig(**checkpoint["config"])
        model = cls(config, checkpoint["num_nodes"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model


# ═══════════════════════════════════════════════════════════════════
# DATASET: Build contagion targets from returns matrix
# ═══════════════════════════════════════════════════════════════════

class ContagionDataset(Dataset):
    """
    Sliding-window dataset over the returns matrix.
    Creates binary contagion targets per stock:
      1 if forward return is extreme AND not explained by own history.
    """

    def __init__(
        self,
        returns_matrix: np.ndarray,   # (n_dates, n_stocks)
        tickers: List[str],
        config: ContagionConfig,
        years: Tuple[int, int],
        max_windows: int = 0,
        fit_stats: bool = True,
        norm_stats: Optional[Dict] = None,
    ):
        self.config = config
        self.tickers = tickers
        self.num_nodes = len(tickers)
        self.window_size = config.window_size

        # Filter to year range using the returns index
        # returns_matrix rows correspond to trading days
        # We need date information — load separately
        self.returns = returns_matrix
        self.n_dates = returns_matrix.shape[0]

        # Build sliding windows
        self.windows = []
        self.targets = []  # Shape: (n_windows, num_nodes, n_horizons)

        # Pre-compute rolling statistics for target construction
        ret_df = pd.DataFrame(returns_matrix)

        max_horizon = max(config.contagion_horizons)

        for t in tqdm(range(config.window_size, self.n_dates - max_horizon),
                       desc="  Building contagion windows", leave=False):
            window = returns_matrix[t - config.window_size:t]  # (window, n_stocks)

            # Build target: for each horizon, is forward return extreme?
            targets_h = np.zeros((self.num_nodes, len(config.contagion_horizons)),
                                  dtype=np.float32)

            # Vectorized target construction (much faster than per-stock loop)
            hist_start = max(0, t - 504)
            hist_data = returns_matrix[hist_start:t]  # (hist_len, n_stocks)
            hist_len = hist_data.shape[0]
            
            if hist_len >= 100:
                # Per-stock historical quantiles (vectorized)
                thresholds = np.percentile(hist_data, config.extreme_quantile * 100, axis=0)  # (n_stocks,)
                expected = np.mean(hist_data[-60:], axis=0) if hist_len >= 60 else np.zeros(self.num_nodes)
                std_60 = np.std(hist_data[-60:], axis=0) if hist_len >= 60 else np.ones(self.num_nodes)
                
                for h_idx, horizon in enumerate(config.contagion_horizons):
                    forward_ret = returns_matrix[t + horizon - 1] - returns_matrix[t - 1]  # (n_stocks,)
                    below_threshold = forward_ret < thresholds
                    excess_negative = (forward_ret - expected) < (config.excess_threshold_std * std_60)
                    targets_h[:, h_idx] = (below_threshold & excess_negative).astype(np.float32)

            self.windows.append(window.T)  # → (n_stocks, window_size) for StemGNN
            self.targets.append(targets_h)

        if max_windows > 0 and len(self.windows) > max_windows:
            rng = np.random.RandomState(config.seed)
            idx = rng.choice(len(self.windows), max_windows, replace=False)
            self.windows = [self.windows[i] for i in idx]
            self.targets = [self.targets[i] for i in idx]

        # Fit normalizer
        if fit_stats and norm_stats is None:
            all_data = np.stack(self.windows, axis=0)
            self.norm_mean = all_data.mean(axis=(0, 2), keepdims=True)
            self.norm_std = all_data.std(axis=(0, 2), keepdims=True)
            self.norm_std[self.norm_std < 1e-8] = 1.0
        elif norm_stats is not None:
            self.norm_mean = norm_stats["mean"]
            self.norm_std = norm_stats["std"]
        else:
            self.norm_mean = 0.0
            self.norm_std = 1.0

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = self.windows[idx].copy()
        return {
            "x": torch.from_numpy(x).float(),
            "target": torch.from_numpy(self.targets[idx].astype(np.float32)),
        }


# ═══════════════════════════════════════════════════════════════════
# CHUNK CONFIG
# ═══════════════════════════════════════════════════════════════════

CHUNK_CONFIG = {
    1: {"train": (2000, 2004), "val": (2005, 2005), "test": (2006, 2006), "label": "chunk1"},
    2: {"train": (2007, 2014), "val": (2015, 2015), "test": (2016, 2016), "label": "chunk2"},
    3: {"train": (2017, 2022), "val": (2023, 2023), "test": (2024, 2024), "label": "chunk3"},
}


# ═══════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, device, clip_grad):
    model.train()
    total_loss = 0.0
    n = 0
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True).clamp(0, 1).float()
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(x)
                loss = F.binary_cross_entropy(output["contagion_scores"], target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(x)
            loss = F.binary_cross_entropy(output["contagion_scores"], target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
        
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def validate_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in loader:
        x = batch["x"].to(device)
        target = batch["target"].to(device).clamp(0, 1).float()
        output = model(x)
        loss = F.binary_cross_entropy(output["contagion_scores"], target)
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def train_contagion_model(config, chunk_id, returns_matrix, tickers, dates_series):
    """Full training loop for one chunk."""
    chunk_cfg = CHUNK_CONFIG[chunk_id]
    label = chunk_cfg["label"]

    out_dir = Path(config.output_dir) / "models" / "StemGNN" / label
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter returns to year ranges
    train_mask = (dates_series.year >= chunk_cfg["train"][0]) & \
                 (dates_series.year <= chunk_cfg["train"][1])
    val_mask = (dates_series.year >= chunk_cfg["val"][0]) & \
               (dates_series.year <= chunk_cfg["val"][1])

    train_dates_idx = np.where(train_mask)[0]
    val_dates_idx = np.where(val_mask)[0]

    if len(train_dates_idx) < config.window_size + max(config.contagion_horizons):
        raise ValueError(f"Not enough training data for chunk {chunk_id}")

    # Build datasets
    train_ret = returns_matrix[train_dates_idx[0]:train_dates_idx[-1]+1]
    val_ret = returns_matrix[val_dates_idx[0]:val_dates_idx[-1]+1]

    train_ds = ContagionDataset(
        train_ret, tickers, config, chunk_cfg["train"],
        max_windows=config.max_train_windows,
    )
    val_ds = ContagionDataset(
        val_ret, tickers, config, chunk_cfg["val"],
        fit_stats=False, norm_stats={"mean": train_ds.norm_mean, "std": train_ds.norm_std},
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                               num_workers=config.num_workers, drop_last=True, pin_memory=True, prefetch_factor=4)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                             num_workers=config.num_workers, pin_memory=True, prefetch_factor=4)

    # Build model
    model = ContagionStemGNN(config, len(tickers)).to(config.device)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    if config.optimizer == "RMSProp":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate, eps=1e-8)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.decay_rate)

    best_val_loss = float("inf")
    no_improve = 0
    best_path = out_dir / "best_model.pt"

    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, config.device,
                                  config.gradient_clip)
        val_loss = validate_epoch(model, val_loader, config.device)

        tqdm.write(f"  [{label}] Epoch {epoch:3d}/{config.epochs} | "
                    f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            model.save(str(best_path))
        else:
            no_improve += 1

        if no_improve >= config.early_stop_patience:
            tqdm.write(f"  Early stopping at epoch {epoch}")
            break

        if (epoch + 1) % config.exponential_decay_step == 0:
            scheduler.step()

    print(f"  Best val loss: {best_val_loss:.6f}")
    return model, best_val_loss


# ═══════════════════════════════════════════════════════════════════
# XAI
# ═══════════════════════════════════════════════════════════════════

def extract_xai(model, dataloader, config, tickers, split_label):
    """
    Extract all 3 levels of XAI from a trained model.
    Level 1: Learned adjacency + top influencers (always)
    Level 2: Gradient-based edge importance (always)
    Level 3: GNNExplainer subgraph mask (if enabled)
    """
    model.eval()
    xai_dir = Path(config.output_dir) / "results" / "StemGNN" / "xai"
    xai_dir.mkdir(parents=True, exist_ok=True)

    # ── Level 1: Adjacency matrix ──
    all_attention = []
    all_contagion = []
    all_inputs = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(config.device)
            output = model(x)
            all_attention.append(output["attention"].cpu().numpy())
            all_contagion.append(output["contagion_scores"].cpu().numpy())
            all_inputs.append(x.cpu().numpy())

    # Aggregate adjacency
    avg_attention = np.mean(all_attention, axis=0)  # (num_nodes, num_nodes)
    avg_contagion = np.concatenate(all_contagion, axis=0).mean(axis=0)  # (num_nodes, n_horizons)

    # Top influencers per stock
    top_influencers = {}
    for i, ticker in enumerate(tickers):
        row = avg_attention[i]
        top_idx = np.argsort(row)[-config.xai_top_influencers:][::-1]
        top_influencers[ticker] = [
            {"ticker": tickers[j], "weight": float(row[j])}
            for j in top_idx if row[j] > 0.01
        ]

    # Save Level 1
    np.save(str(xai_dir / f"{split_label}_adjacency.npy"), avg_attention)
    with open(xai_dir / f"{split_label}_top_influencers.json", "w") as f:
        json.dump(top_influencers, f, indent=2, default=str)

    # ── Level 2: Gradient edge importance ──
    edge_importance = np.zeros_like(avg_attention)
    n_samples = min(config.xai_sample_size, len(dataloader.dataset))

    for idx in range(0, n_samples, config.batch_size):
        batch_data = dataloader.dataset[idx:min(idx + config.batch_size, n_samples)]
        # Simplified: use first element
        if isinstance(batch_data, dict):
            x_tensor = batch_data["x"].unsqueeze(0).to(config.device)
        else:
            x_tensor = batch_data[0]["x"].unsqueeze(0).to(config.device)

        x_tensor.requires_grad_(True)
        output = model(x_tensor)
        score = output["contagion_scores"].sum()
        score.backward()

        # Gradient through stemgnn's attention
        # The attention is from latent_correlation_layer
        # We approximate edge importance via input gradient magnitude
        grad = x_tensor.grad.abs().cpu().numpy().squeeze()
        if grad.ndim == 2:
            edge_importance += grad / n_samples

        x_tensor.grad = None

    np.save(str(xai_dir / f"{split_label}_edge_importance.npy"), edge_importance)

    # ── Level 3: GNNExplainer (opt-in) ──
    if config.enable_gnnexplainer:
        print(f"  [xai] Running GNNExplainer on {min(10, n_samples)} samples...")
        gnnexplainer_results = []

        for idx in range(min(10, n_samples)):
            # Simplified GNNExplainer: learn a subgraph mask
            mask = torch.nn.Parameter(torch.randn(len(tickers), len(tickers),
                                                   device=config.device) * 0.01)
            opt = torch.optim.Adam([mask], lr=0.01)

            if isinstance(dataloader.dataset[idx], dict):
                x_orig = dataloader.dataset[idx]["x"].unsqueeze(0).to(config.device)
            else:
                x_orig = dataloader.dataset[idx][0]["x"].unsqueeze(0).to(config.device)

            with torch.no_grad():
                orig_out = model(x_orig)
                orig_score = orig_out["contagion_scores"].mean()

            for _ in range(50):
                masked_x = x_orig * torch.sigmoid(mask).unsqueeze(0)
                out = model(masked_x)
                loss = -F.mse_loss(out["contagion_scores"].mean(),
                                    orig_score.expand_as(out["contagion_scores"].mean()))
                loss = loss + 0.1 * torch.sigmoid(mask).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

            final_mask = torch.sigmoid(mask).detach().cpu().numpy()
            top_edges = np.argsort(final_mask.flatten())[-20:][::-1]
            n_nodes = len(tickers)

            gnnexplainer_results.append({
                "sample_idx": int(idx),
                "important_edges": [
                    {"source": tickers[e // n_nodes], "target": tickers[e % n_nodes],
                     "importance": float(final_mask.flatten()[e])}
                    for e in top_edges if final_mask.flatten()[e] > 0.1
                ],
            })

        with open(xai_dir / f"{split_label}_gnnexplainer.json", "w") as f:
            json.dump(gnnexplainer_results, f, indent=2, default=str)

    print(f"  [xai] Saved: {xai_dir}/")
    return avg_contagion, top_influencers


# ═══════════════════════════════════════════════════════════════════
# CLI COMMANDS
# ═══════════════════════════════════════════════════════════════════

def cmd_inspect(config):
    print("=" * 60)
    print("STEMGNN CONTAGION — DATA INSPECTION")
    print("=" * 60)

    fp = Path(config.returns_path)
    if not fp.exists():
        print(f"❌ Returns file not found: {fp}")
        return

    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    print(f"\n✅ Returns matrix: {df.shape[0]} days × {df.shape[1]} tickers")
    print(f"   Date range: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"   NaN rate: {df.isna().mean().mean()*100:.2f}%")

    print("\nChronological splits:")
    for cid, cfg in CHUNK_CONFIG.items():
        for split, (y1, y2) in [("train", cfg["train"]), ("val", cfg["val"]), ("test", cfg["test"])]:
            mask = (df.index.year >= y1) & (df.index.year <= y2)
            print(f"  Chunk {cid} {split:5s} ({y1}-{y2}): {mask.sum():>6} days")

    print(f"\nEstimated GPU memory for 2500×2500 adjacency: ~50 MB")
    print(f"With batch_size=8: ~400 MB per forward pass")
    print("✅ RTX 3090 Ti (24GB) is sufficient.\n")


def cmd_hpo(config, chunk_id, n_trials=50):
    try:
        import optuna
    except ImportError:
        print("❌ optuna not installed. pip install optuna")
        return

    chunk_cfg = CHUNK_CONFIG[chunk_id]
    label = chunk_cfg["label"]

    df = pd.read_csv(config.returns_path, index_col=0, parse_dates=True)
    tickers = list(df.columns)
    returns = df.values.astype(np.float32)
    dates = df.index

    # Small subset for HPO
    train_mask = (dates.year >= chunk_cfg["train"][0]) & (dates.year <= chunk_cfg["train"][1])
    train_idx = np.where(train_mask)[0]
    val_mask = (dates.year >= chunk_cfg["val"][0]) & (dates.year <= chunk_cfg["val"][1])
    val_idx = np.where(val_mask)[0]

    train_ret = returns[train_idx[0]:train_idx[-1]+1]
    val_ret = returns[val_idx[0]:val_idx[-1]+1]

    def objective(trial):
        trial_config = ContagionConfig(**{
            k: v for k, v in asdict(config).items()
            if k not in ("window_size", "multi_layer", "dropout_rate", "learning_rate",
                         "decay_rate", "exponential_decay_step", "batch_size")
        })
        trial_config.window_size = trial.suggest_categorical("window_size", [15, 30, 60])
        trial_config.multi_layer = trial.suggest_categorical("multi_layer", [5, 8, 13, 20])
        trial_config.dropout_rate = trial.suggest_categorical("dropout_rate", [0.5, 0.6, 0.75, 0.8])
        trial_config.learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        trial_config.decay_rate = trial.suggest_categorical("decay_rate", [0.3, 0.5, 0.7, 0.9])
        trial_config.exponential_decay_step = trial.suggest_categorical("exponential_decay_step", [5, 8, 13])
        trial_config.batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
        trial_config.epochs = 20
        trial_config.device = config.device
        trial_config.max_train_windows = 2000

        train_ds = ContagionDataset(train_ret, tickers, trial_config, chunk_cfg["train"],
                                     max_windows=trial_config.max_train_windows)
        val_ds = ContagionDataset(val_ret, tickers, trial_config, chunk_cfg["val"],
                                   fit_stats=False,
                                   norm_stats={"mean": train_ds.norm_mean, "std": train_ds.norm_std})

        train_loader = DataLoader(train_ds, batch_size=trial_config.batch_size, shuffle=True,
                                   num_workers=0, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=trial_config.batch_size, shuffle=False,
                                 num_workers=0)

        model = ContagionStemGNN(trial_config, len(tickers)).to(trial_config.device)
        opt = torch.optim.RMSprop(model.parameters(), lr=trial_config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=trial_config.decay_rate)

        for epoch in range(trial_config.epochs):
            train_epoch(model, train_loader, opt, trial_config.device, trial_config.gradient_clip)
            if (epoch + 1) % trial_config.exponential_decay_step == 0:
                scheduler.step()

        val_loss = validate_epoch(model, val_loader, trial_config.device)
        return float(val_loss)

    study = optuna.create_study(direction="minimize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=config.hpo_n_startup),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        storage=f"sqlite:///{config.output_dir}/codeResults/StemGNN/hpo.db",
        study_name=f"stemgnn_contagion_{label}", load_if_exists=True)

    print(f"Running {config.hpo_trials} HPO trials for Chunk {chunk_id}...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    hpo_dir = Path(config.output_dir) / "codeResults" / "StemGNN"
    hpo_dir.mkdir(parents=True, exist_ok=True)
    with open(hpo_dir / f"best_params_{label}.json", "w") as f:
        json.dump({"params": best, "value": study.best_value}, f, indent=2)

    print(f"\nBest params: {best}")
    print(f"Best val loss: {study.best_value:.6f}")


def cmd_train_best(config, chunk_id):
    chunk_cfg = CHUNK_CONFIG[chunk_id]
    label = chunk_cfg["label"]

    hpo_dir = Path(config.output_dir) / "codeResults" / "StemGNN"
    best_path = hpo_dir / f"best_params_{label}.json"
    if best_path.exists():
        with open(best_path) as f:
            best = json.load(f)
        for k, v in best["params"].items():
            setattr(config, k, v)
        print(f"Loaded best HPO params from {best_path}")
    else:
        print(f"No HPO results — using defaults")

    df = pd.read_csv(config.returns_path, index_col=0, parse_dates=True)
    tickers = list(df.columns)
    returns = df.values.astype(np.float32)

    model, best_loss = train_contagion_model(config, chunk_id, returns, tickers, df.index)
    return model


def cmd_predict(config, chunk_id, split):
    chunk_cfg = CHUNK_CONFIG[chunk_id]
    label = chunk_cfg["label"]

    model_path = Path(config.output_dir) / "models" / "StemGNN" / label / "best_model.pt"
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return

    model = ContagionStemGNN.load(str(model_path), device=config.device)
    model.eval()

    df = pd.read_csv(config.returns_path, index_col=0, parse_dates=True)
    tickers = list(df.columns)
    returns = df.values.astype(np.float32)

    mask = (df.index.year >= chunk_cfg[split][0]) & (df.index.year <= chunk_cfg[split][1])
    idx = np.where(mask)[0]
    split_ret = returns[idx[0]:idx[-1]+1]

    ds = ContagionDataset(split_ret, tickers, config, chunk_cfg[split], fit_stats=False,
                           norm_stats={"mean": 0.0, "std": 1.0})
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # Run prediction + XAI
    split_label = f"{label}_{split}"
    avg_contagion, top_influencers = extract_xai(model, loader, config, tickers, split_label)

    # Save contagion scores
    scores_df = pd.DataFrame(avg_contagion, index=tickers,
                              columns=[f"contagion_{h}d" for h in config.contagion_horizons])
    out_path = Path(config.output_dir) / "results" / "StemGNN" / f"contagion_scores_{split_label}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(out_path)
    print(f"  Contagion scores saved: {out_path}")
    print(f"  Mean contagion (5d): {avg_contagion[:, 0].mean():.4f}")
    print(f"  Mean contagion (20d): {avg_contagion[:, 1].mean():.4f}")
    print(f"  Mean contagion (60d): {avg_contagion[:, 2].mean():.4f}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="StemGNN Contagion Risk Module")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("inspect")
    p = sub.add_parser("hpo")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--trials", type=int, default=50)
    p = sub.add_parser("train-best")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    sub.add_parser("train-best-all")
    p = sub.add_parser("predict")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])

    for sp in [p for p in sub.choices.values() if p is not None]:
        sp.add_argument("--repo-root", type=str, default="")
        sp.add_argument("--returns-path", type=str, default="")
        sp.add_argument("--output-dir", type=str, default="")
        sp.add_argument("--device", type=str, default="cuda")
        sp.add_argument("--max-train-windows", type=int, default=0)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    config = ContagionConfig()
    if hasattr(args, "repo_root") and args.repo_root:
        config.repo_root = args.repo_root
        config.__post_init__()
    if hasattr(args, "returns_path") and args.returns_path:
        config.returns_path = args.returns_path
    if hasattr(args, "output_dir") and args.output_dir:
        config.output_dir = args.output_dir
    if hasattr(args, "device"):
        config.device = args.device
    if hasattr(args, "max_train_windows"):
        config.max_train_windows = args.max_train_windows

    if args.command == "inspect":
        cmd_inspect(config)
    elif args.command == "hpo":
        cmd_hpo(config, args.chunk, getattr(args, "trials", 50))
    elif args.command == "train-best":
        cmd_train_best(config, args.chunk)
    elif args.command == "train-best-all":
        for cid in [1, 2, 3]:
            print(f"\n{'='*60}\nTRAINING CHUNK {cid}\n{'='*60}")
            cmd_train_best(config, cid)
    elif args.command == "predict":
        cmd_predict(config, args.chunk, args.split)


if __name__ == "__main__":
    main()

# ── Run instructions ──────────────────────────────────────────
# python code/gnn/stemgnn_contagion.py inspect
# python code/gnn/stemgnn_contagion.py hpo --chunk 1 --trials 50 --device cuda
# python code/gnn/stemgnn_contagion.py train-best --chunk 1 --device cuda
# python code/gnn/stemgnn_contagion.py train-best-all --device cuda
# python code/gnn/stemgnn_contagion.py predict --chunk 1 --split test --device cuda

