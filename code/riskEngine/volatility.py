#!/usr/bin/env python3
"""
code/riskEngine/volatility.py

Volatility Estimation Model — GARCH(1,1) + MLP Hybrid
=====================================================
Project: fin-glassbox — Explainable Distributed Deep Learning Framework

Complete module with:
  - Auto-generating manifests (ticker/date per embedding row)
  - SimpleGARCH(1,1) per ticker (fitted on training returns, no leakage)
  - MLP(256+2→64→4): vol_10d, vol_30d, regime (low/med/high), confidence
  - HPO: Optuna TPE on MLP hyperparameters (40 trials per chunk)
  - XAI Level 1: GARCH parameters + MLP gradient feature importance
  - XAI Level 2: Counterfactual analysis (what-if scenarios)
  - Clean CLI: inspect, hpo, train-best, train-best-all, predict

Training:
  python code/riskEngine/volatility.py train-best --chunk 1 --device cuda
  python code/riskEngine/volatility.py hpo --chunk 1 --trials 40 --device cuda
  python code/riskEngine/volatility.py predict --chunk 1 --split test --device cuda
"""

from __future__ import annotations

import argparse, json, math, os, sys, time, warnings, pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

@dataclass
class VolatilityConfig:
    repo_root: str = ""
    output_dir: str = "outputs"
    embeddings_dir: str = "outputs/embeddings/TemporalEncoder"
    features_path: str = "data/yFinance/processed/features_temporal.csv"
    returns_path: str = "data/yFinance/processed/returns_panel_wide.csv"

    # Architecture
    input_dim: int = 256
    hidden_dims: list = field(default_factory=lambda: [64])
    dropout: float = 0.2

    # Training
    batch_size: int = 512
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stop_patience: int = 15
    gradient_clip: float = 1.0

    # Target horizons
    vol_horizons: list = field(default_factory=lambda: [10, 30])
    seq_len: int = 30

    # HPO
    hpo_trials: int = 40
    hpo_n_startup: int = 10
    hpo_epochs: int = 15

    # XAI
    xai_sample_size: int = 500
    xai_counterfactual_scenarios: int = 5

    # System
    device: str = "cuda"
    seed: int = 42
    mixed_precision: bool = True
    num_workers: int = 6

    def __post_init__(self):
        if self.repo_root:
            root = Path(self.repo_root)
            for attr in ['embeddings_dir', 'features_path', 'returns_path', 'output_dir']:
                val = getattr(self, attr)
                if val and not Path(val).is_absolute():
                    setattr(self, attr, str(root / val))


CHUNK_CONFIG = {
    1: {"train": (2000, 2004), "val": (2005, 2005), "test": (2006, 2006), "label": "chunk1"},
    2: {"train": (2007, 2014), "val": (2015, 2015), "test": (2016, 2016), "label": "chunk2"},
    3: {"train": (2017, 2022), "val": (2023, 2023), "test": (2024, 2024), "label": "chunk3"},
}

FEATURE_NAMES = [
    "log_return", "vol_5d", "vol_21d", "rsi_14", "macd_hist",
    "bb_pos", "volume_ratio", "hl_ratio", "price_pos", "spy_corr_63d",
]

REGIME_LABELS = {0: "low", 1: "medium", 2: "high"}


# ═══════════════════════════════════════════════════════════════════
# MANIFEST GENERATION (auto-detect, build if missing)
# ═══════════════════════════════════════════════════════════════════

def ensure_manifest_exists(config: VolatilityConfig, chunk_id: int, split: str) -> Path:
    label = CHUNK_CONFIG[chunk_id]["label"]
    emb_dir = Path(config.embeddings_dir)
    manifest_path = emb_dir / f"{label}_{split}_manifest.csv"
    emb_path = emb_dir / f"{label}_{split}_embeddings.npy"

    if manifest_path.exists():
        return manifest_path

    if not emb_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {emb_path}. Run temporal_encoder.py embed-all first."
        )

    print(f"  ⚠ Manifest missing for {label}_{split} — auto-generating...")
    _build_manifest(config, chunk_id, split, emb_path, manifest_path)
    return manifest_path


def _build_manifest(config: VolatilityConfig, chunk_id: int, split: str,
                    emb_path: Path, manifest_path: Path) -> None:
    years = CHUNK_CONFIG[chunk_id][split]
    label = CHUNK_CONFIG[chunk_id]["label"]

    embeddings = np.load(emb_path, mmap_mode='r')
    features_df = pd.read_csv(config.features_path, dtype={"ticker": str}, parse_dates=["date"])
    features_df["year"] = features_df["date"].dt.year
    mask = (features_df["year"] >= years[0]) & (features_df["year"] <= years[1])
    df = features_df[mask]

    records = []
    for ticker, group in tqdm(df.groupby("ticker"), desc=f"  Manifest {label}_{split}", leave=False):
        vals = group[FEATURE_NAMES].values.astype(np.float32)
        dates = group["date"].values
        if len(vals) < config.seq_len:
            continue
        for i in range(config.seq_len - 1, len(vals)):
            records.append({"ticker": ticker, "date": str(dates[i])[:10]})

    n = min(len(records), len(embeddings))
    if len(records) != len(embeddings):
        print(f"    Mismatch {len(records):,} vs {len(embeddings):,} — truncating")
    pd.DataFrame(records[:n]).to_csv(manifest_path, index=False)
    print(f"    Saved: {manifest_path.name} ({n:,} rows)")


def ensure_all_manifests(config: VolatilityConfig, chunk_id: int) -> None:
    """Pre-generate manifests for all splits of a chunk."""
    for split in ["train", "val", "test"]:
        try:
            ensure_manifest_exists(config, chunk_id, split)
        except FileNotFoundError:
            pass  # embeddings not yet generated


# ═══════════════════════════════════════════════════════════════════
# SIMPLE GARCH(1,1)
# ═══════════════════════════════════════════════════════════════════

class SimpleGARCH:
    """GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}."""

    def __init__(self):
        self.omega = self.alpha = self.beta = self.mu = None
        self.fitted = False

    def fit(self, returns: np.ndarray, max_iter: int = 500, lr: float = 0.01) -> bool:
        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[~np.isnan(returns)]
        if len(returns) < 63:
            return False

        self.mu = float(returns.mean())
        centered = returns - self.mu
        n = len(centered)
        omega, alpha, beta = np.var(centered) * 0.1, 0.1, 0.8

        for _ in range(max_iter):
            sigma2 = np.zeros(n)
            sigma2[0] = np.var(centered)
            for t in range(1, n):
                sigma2[t] = omega + alpha * centered[t-1]**2 + beta * sigma2[t-1]

            g_w = 0.5 * np.sum(1/sigma2 - centered**2 / sigma2**2)
            g_a = 0.5 * np.sum((1/sigma2 - centered**2/sigma2**2) * np.roll(centered**2, 1))
            g_b = 0.5 * np.sum((1/sigma2 - centered**2/sigma2**2) * np.roll(sigma2, 1))

            omega = max(1e-8, omega - lr * g_w / n)
            alpha = np.clip(alpha - lr * g_a / n, 0.01, 0.3)
            beta = np.clip(beta - lr * g_b / n, 0.5, 0.98)

        self.omega, self.alpha, self.beta = float(omega), float(alpha), float(beta)
        self.fitted = True
        return True

    def forecast(self, returns: np.ndarray, horizon: int) -> float:
        if not self.fitted:
            return np.std(returns[-21:]) * math.sqrt(252) if len(returns) >= 21 else 0.3
        rets = np.asarray(returns, dtype=np.float64)[-252:]
        centered = rets - self.mu
        s2 = np.var(centered) if len(centered) < 2 else self.omega + self.alpha * centered[-1]**2
        for _ in range(horizon):
            s2 = self.omega + (self.alpha + self.beta) * s2
        return math.sqrt(max(s2, 1e-10)) * math.sqrt(252)

    def to_dict(self) -> Dict:
        return {"omega": self.omega, "alpha": self.alpha, "beta": self.beta,
                "mu": self.mu, "persistence": self.alpha + self.beta if self.fitted else None}


# ═══════════════════════════════════════════════════════════════════
# MLP MODEL
# ═══════════════════════════════════════════════════════════════════

class VolatilityMLP(nn.Module):
    """MLP adjusting GARCH forecasts using temporal embeddings."""

    def __init__(self, config: VolatilityConfig):
        super().__init__()
        layers, in_dim = [], config.input_dim + 2
        for h in config.hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(config.dropout)])
            in_dim = h
        self.shared = nn.Sequential(*layers) if layers else nn.Identity()
        last_dim = config.hidden_dims[-1] if config.hidden_dims else in_dim
        self.head_vol10 = nn.Linear(last_dim, 1)
        self.head_vol30 = nn.Linear(last_dim, 1)
        self.head_regime = nn.Linear(last_dim, 3)
        self.head_conf = nn.Linear(last_dim, 1)

    def forward(self, emb: torch.Tensor, garch_vol: torch.Tensor,
                recent_vol: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = torch.cat([emb, garch_vol.unsqueeze(-1), recent_vol.unsqueeze(-1)], dim=-1)
        shared = self.shared(x)
        return {
            "vol_10d": F.softplus(self.head_vol10(shared)).squeeze(-1),
            "vol_30d": F.softplus(self.head_vol30(shared)).squeeze(-1),
            "regime_logits": self.head_regime(shared),
            "regime_probs": F.softmax(self.head_regime(shared), dim=-1),
            "confidence": torch.sigmoid(self.head_conf(shared)).squeeze(-1),
        }

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: Path, config: "VolatilityConfig", device: str = "cpu") -> "VolatilityMLP":
        model = cls(config)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        return model


# ═══════════════════════════════════════════════════════════════════
# DATASET — Manifest-based
# ═══════════════════════════════════════════════════════════════════

class VolatilityDataset(Dataset):
    """Pairs embeddings with realized volatility using manifest."""

    def __init__(self, config: VolatilityConfig, split: str, chunk_id: int,
                 returns_df: pd.DataFrame,
                 garch_models: Optional[Dict[str, SimpleGARCH]] = None):
        self.config = config
        label = CHUNK_CONFIG[chunk_id]["label"]

        manifest_path = ensure_manifest_exists(config, chunk_id, split)
        emb_path = Path(config.embeddings_dir) / f"{label}_{split}_embeddings.npy"

        self.manifest = pd.read_csv(manifest_path)
        self.embeddings = np.load(emb_path, mmap_mode='r')

        n = min(len(self.manifest), len(self.embeddings))
        self.manifest = self.manifest.iloc[:n]
        self.embeddings = self.embeddings[:n] if hasattr(self.embeddings, '__getitem__') else self.embeddings[:n]

        print(f"  {label}_{split}: {n:,} samples, {self.manifest['ticker'].nunique()} tickers")

        self.returns_data = returns_df
        self._ticker_ret_cache = {}
        self._date_to_idx = {str(d)[:10]: i for i, d in enumerate(returns_df.index)}

        self.targets = self._compute_targets()
        self.garch_forecasts = self._compute_garch_forecasts(garch_models)

    def _get_returns(self, ticker: str) -> np.ndarray:
        if ticker not in self._ticker_ret_cache:
            if ticker in self.returns_data.columns:
                self._ticker_ret_cache[ticker] = self.returns_data[ticker].values.astype(np.float64)
            else:
                self._ticker_ret_cache[ticker] = np.array([])
        return self._ticker_ret_cache[ticker]

    def _compute_targets(self) -> np.ndarray:
        targets = np.zeros((len(self.manifest), 2), dtype=np.float32)
        for h_idx, h in enumerate(self.config.vol_horizons):
            for i in tqdm(range(len(self.manifest)), desc=f"  Targets h={h}", leave=False):
                ticker = self.manifest.iloc[i]["ticker"]
                date_str = self.manifest.iloc[i]["date"]
                rets = self._get_returns(ticker)
                idx = self._date_to_idx.get(date_str)
                if idx is None or len(rets) == 0 or idx >= len(rets) - h:
                    targets[i, h_idx] = 0.3
                else:
                    fut = rets[idx:min(len(rets), idx + h)]
                    targets[i, h_idx] = np.std(fut) * math.sqrt(252) if len(fut) >= 3 else 0.3
        return np.clip(targets, 0.01, 5.0)

    def _compute_garch_forecasts(self, garch_models) -> np.ndarray:
        forecasts = np.zeros((len(self.manifest), 2), dtype=np.float32)
        if garch_models is None:
            return forecasts
        for i in tqdm(range(len(self.manifest)), desc="  GARCH forecasts", leave=False):
            ticker = self.manifest.iloc[i]["ticker"]
            date_str = self.manifest.iloc[i]["date"]
            if ticker not in garch_models:
                continue
            rets = self._get_returns(ticker)
            idx = self._date_to_idx.get(date_str)
            if idx is None or idx < 63 or len(rets) == 0:
                continue
            for j, h in enumerate(self.config.vol_horizons):
                forecasts[i, j] = garch_models[ticker].forecast(rets[:idx], h)
        return np.clip(forecasts, 0.01, 5.0)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        emb = self.embeddings[idx].copy()
        return {
            "emb": torch.from_numpy(emb),
            "ticker": self.manifest.iloc[idx]["ticker"],
            "date": self.manifest.iloc[idx]["date"],
            "target": torch.from_numpy(self.targets[idx]),
            "garch_vol": torch.tensor(self.garch_forecasts[idx, 0], dtype=torch.float32),
            "recent_vol": torch.tensor(max(self.targets[idx, 0] * 0.8, 0.05), dtype=torch.float32),
        }


# ═══════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════

def fit_garch_models(config: VolatilityConfig, chunk_id: int) -> Dict[str, SimpleGARCH]:
    returns_df = pd.read_csv(config.returns_path, index_col=0, parse_dates=True)
    train_years = CHUNK_CONFIG[chunk_id]["train"]
    train_returns = returns_df[
        (returns_df.index.year >= train_years[0]) &
        (returns_df.index.year <= train_years[1])
    ]
    garch_models = {}
    tickers = list(train_returns.columns)
    for ticker in tqdm(tickers, desc="  GARCH fit"):
        rets = train_returns[ticker].dropna().values.astype(np.float64)
        if len(rets) < 252:
            continue
        g = SimpleGARCH()
        if g.fit(rets):
            garch_models[ticker] = g
    print(f"  Fitted: {len(garch_models)}/{len(tickers)}")
    return garch_models


def _train_mlp(config: VolatilityConfig, chunk_id: int, garch_models: Dict,
               returns_df: pd.DataFrame, device: torch.device) -> Tuple[VolatilityMLP, Dict]:
    label = CHUNK_CONFIG[chunk_id]["label"]
    train_ds = VolatilityDataset(config, "train", chunk_id, returns_df, garch_models)
    val_ds = VolatilityDataset(config, "val", chunk_id, returns_df, garch_models)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                               num_workers=config.num_workers, pin_memory=True,
                               persistent_workers=config.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                             num_workers=config.num_workers, pin_memory=True)

    model = VolatilityMLP(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(config.mixed_precision and device.type == "cuda"))

    best_val_loss, no_improve = float("inf"), 0
    history, garch_dir = [], Path(config.output_dir) / "models" / "Volatility" / label
    garch_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss, n_b = 0.0, 0
        for batch in train_loader:
            emb = batch["emb"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            garch = batch["garch_vol"].to(device, non_blocking=True)
            recent = batch["recent_vol"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    out = model(emb, garch, recent)
                    loss = F.mse_loss(out["vol_10d"], target[:, 0]) + F.mse_loss(out["vol_30d"], target[:, 1]) * 0.5
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(emb, garch, recent)
                loss = F.mse_loss(out["vol_10d"], target[:, 0])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
            train_loss += loss.item()
            n_b += 1

        train_loss /= max(n_b, 1)
        model.eval()
        val_loss, n_v = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                emb = batch["emb"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                garch = batch["garch_vol"].to(device, non_blocking=True)
                recent = batch["recent_vol"].to(device, non_blocking=True)
                out = model(emb, garch, recent)
                loss = F.mse_loss(out["vol_10d"], target[:, 0]) + F.mse_loss(out["vol_30d"], target[:, 1]) * 0.5
                val_loss += loss.item()
                n_v += 1
        val_loss /= max(n_v, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        print(f"  [{label}] E{epoch:03d} | train={train_loss:.5f} | val={val_loss:.5f}")
        if val_loss < best_val_loss:
            best_val_loss, no_improve = val_loss, 0
            model.save(garch_dir / "best_model.pt")
        else:
            no_improve += 1
        if no_improve >= config.early_stop_patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    pd.DataFrame(history).to_csv(garch_dir / "training_history.csv", index=False)
    summary = {"chunk": label, "best_val_loss": float(best_val_loss), "epochs_trained": epoch,
               "n_tickers_garch": len(garch_models)}
    with open(garch_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Best val loss: {best_val_loss:.6f}")
    return model, summary


def train_volatility_model(config: VolatilityConfig, chunk_id: int) -> Dict:
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)
    label = CHUNK_CONFIG[chunk_id]["label"]

    print(f"\n{'='*60}\n  VOLATILITY MODEL — Chunk {chunk_id}\n{'='*60}")
    print(f"  Device: {device}")

    # Ensure manifests exist
    ensure_all_manifests(config, chunk_id)

    returns_df = pd.read_csv(config.returns_path, index_col=0, parse_dates=True)

    # Fit GARCH
    print(f"\n  [1/2] Fitting GARCH(1,1)...")
    garch_models = fit_garch_models(config, chunk_id)
    garch_dir = Path(config.output_dir) / "models" / "Volatility" / label
    garch_dir.mkdir(parents=True, exist_ok=True)
    with open(garch_dir / "garch_models.pkl", "wb") as f:
        pickle.dump(garch_models, f)

    # Save GARCH XAI
    garch_xai = {t: g.to_dict() for t, g in garch_models.items()}
    with open(garch_dir / "garch_params.json", "w") as f:
        json.dump(garch_xai, f, indent=2)

    # Train MLP
    print(f"\n  [2/2] Training MLP...")
    model, summary = _train_mlp(config, chunk_id, garch_models, returns_df, device)

    # Save final model
    model.save(garch_dir / "final_model.pt")
    model.save(garch_dir / "model_freezed" / "model.pt")
    (garch_dir / "model_freezed").mkdir(exist_ok=True)
    print(f"\n  Complete. Best val loss: {summary['best_val_loss']:.6f}")
    return summary


# ═══════════════════════════════════════════════════════════════════
# HPO
# ═══════════════════════════════════════════════════════════════════

def _hpo_objective(trial, base_config: VolatilityConfig, chunk_id: int,
                   returns_df: pd.DataFrame, garch_models: Dict) -> float:
    tc = VolatilityConfig(**{k: v for k, v in asdict(base_config).items()
        if k not in ("hidden_dims", "dropout", "learning_rate", "weight_decay", "batch_size")})
    tc.hidden_dims = [trial.suggest_categorical("hidden_dim", [32, 64, 128])]
    tc.dropout = trial.suggest_float("dropout", 0.1, 0.4)
    tc.learning_rate = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
    tc.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    tc.batch_size = trial.suggest_categorical("batch_size", [256, 512])
    tc.epochs = base_config.hpo_epochs
    tc.device = base_config.device
    tc.num_workers = 4

    device = torch.device(tc.device if torch.cuda.is_available() else "cpu")
    _, summary = _train_mlp(tc, chunk_id, garch_models, returns_df, device)
    return float(summary["best_val_loss"])


def run_hpo(config: VolatilityConfig, chunk_id: int) -> Dict:
    if not HAS_OPTUNA:
        raise ImportError("optuna required: pip install optuna")

    label = CHUNK_CONFIG[chunk_id]["label"]
    ensure_all_manifests(config, chunk_id)
    returns_df = pd.read_csv(config.returns_path, index_col=0, parse_dates=True)
    garch_models = fit_garch_models(config, chunk_id)

    study_dir = Path(config.output_dir) / "codeResults" / "Volatility"
    study_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.seed, n_startup_trials=config.hpo_n_startup),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        storage=f"sqlite:///{study_dir}/hpo.db",
        study_name=f"volatility_{label}",
        load_if_exists=True,
    )

    objective = lambda t: _hpo_objective(t, config, chunk_id, returns_df, garch_models)
    study.optimize(objective, n_trials=config.hpo_trials, show_progress_bar=True)

    best = {"params": study.best_params, "value": study.best_value}
    with open(study_dir / f"best_params_{label}.json", "w") as f:
        json.dump(best, f, indent=2)
    print(f"\n  Best HPO: {study.best_params} (val_loss={study.best_value:.6f})")
    return best


# ═══════════════════════════════════════════════════════════════════
# PREDICTION + XAI
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_with_xai(config: VolatilityConfig, chunk_id: int, split: str) -> pd.DataFrame:
    """Generate predictions with Level 1 (gradient feature importance) + Level 2 (counterfactuals)."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    label = CHUNK_CONFIG[chunk_id]["label"]

    model_dir = Path(config.output_dir) / "models" / "Volatility" / label
    model = VolatilityMLP.load(model_dir / "best_model.pt", config, str(device))
    model.eval()

    with open(model_dir / "garch_models.pkl", "rb") as f:
        garch_models = pickle.load(f)
    with open(model_dir / "garch_params.json") as f:
        garch_xai = json.load(f)

    returns_df = pd.read_csv(config.returns_path, index_col=0, parse_dates=True)
    ds = VolatilityDataset(config, split, chunk_id, returns_df, garch_models)
    loader = DataLoader(ds, batch_size=min(config.batch_size, 256), shuffle=False,
                         num_workers=min(config.num_workers, 4), pin_memory=True)

    results_dir = Path(config.output_dir) / "results" / "Volatility"
    xai_dir = results_dir / "xai"
    results_dir.mkdir(parents=True, exist_ok=True)
    xai_dir.mkdir(parents=True, exist_ok=True)

    all_preds, grad_importance = [], []
    n_xai = 0

    for batch in tqdm(loader, desc=f"  Predict+{split}"):
        emb = batch["emb"].to(device, non_blocking=True)
        garch = batch["garch_vol"].to(device, non_blocking=True)
        recent = batch["recent_vol"].to(device, non_blocking=True)

        # Forward pass
        if n_xai < config.xai_sample_size:
            emb_xai = emb[:min(10, len(emb))].clone().detach().requires_grad_(True)
            garch_x = garch[:len(emb_xai)]
            recent_x = recent[:len(emb_xai)]
            out_xai = model(emb_xai, garch_x, recent_x)
            score = out_xai["vol_10d"].mean() + out_xai["vol_30d"].mean()
            model.zero_grad(set_to_none=True)
            score.backward()
            g = emb_xai.grad.abs().mean(dim=0).cpu().numpy()
            grad_importance.append(g)
            n_xai += len(emb_xai)

        out = model(emb, garch, recent)
        all_preds.append({
            "ticker": list(batch["ticker"]),
            "date": list(batch["date"]),
            "vol_10d": out["vol_10d"].cpu().numpy(),
            "vol_30d": out["vol_30d"].cpu().numpy(),
            "regime": out["regime_probs"].cpu().numpy().argmax(axis=1),
            "regime_probs_low": out["regime_probs"].cpu().numpy()[:, 0],
            "regime_probs_med": out["regime_probs"].cpu().numpy()[:, 1],
            "regime_probs_high": out["regime_probs"].cpu().numpy()[:, 2],
            "confidence": out["confidence"].cpu().numpy(),
        })

    # Combine predictions
    results = pd.DataFrame({
        "ticker": np.concatenate([p["ticker"] for p in all_preds]),
        "date": np.concatenate([p["date"] for p in all_preds]),
        "vol_10d": np.concatenate([p["vol_10d"] for p in all_preds]),
        "vol_30d": np.concatenate([p["vol_30d"] for p in all_preds]),
        "regime": np.concatenate([p["regime"] for p in all_preds]),
        "regime_probs_low": np.concatenate([p["regime_probs_low"] for p in all_preds]),
        "regime_probs_med": np.concatenate([p["regime_probs_med"] for p in all_preds]),
        "regime_probs_high": np.concatenate([p["regime_probs_high"] for p in all_preds]),
        "confidence": np.concatenate([p["confidence"] for p in all_preds]),
    })

    pred_path = results_dir / f"predictions_{label}_{split}.csv"
    results.to_csv(pred_path, index=False)
    print(f"  Predictions: {pred_path} ({len(results):,} rows)")

    # XAI Level 1: Feature importance
    if grad_importance:
        grad_mean = np.stack(grad_importance).mean(axis=0)
        importance_df = pd.DataFrame({
            "dim": range(len(grad_mean)),
            "importance": grad_mean,
            "importance_pct": (grad_mean / grad_mean.sum() * 100).round(2),
        }).sort_values("importance", ascending=False)
        importance_df.to_csv(xai_dir / f"{label}_{split}_feature_importance.csv", index=False)
        print(f"  XAI-L1 saved: feature importance (top dim: {importance_df.iloc[0]['dim']})")

    # XAI Level 2: Counterfactuals
    cf_results = _generate_counterfactuals(model, ds, config, garch_models, device)
    with open(xai_dir / f"{label}_{split}_counterfactuals.json", "w") as f:
        json.dump(cf_results, f, indent=2)
    print(f"  XAI-L2 saved: {len(cf_results)} counterfactual scenarios")

    # XAI Level 1: GARCH parameter summary
    garch_summary = {
        "n_models": len(garch_xai),
        "avg_omega": np.mean([v["omega"] for v in garch_xai.values() if v["omega"] is not None]),
        "avg_alpha": np.mean([v["alpha"] for v in garch_xai.values() if v["alpha"] is not None]),
        "avg_beta": np.mean([v["beta"] for v in garch_xai.values() if v["beta"] is not None]),
        "avg_persistence": np.mean([v["persistence"] for v in garch_xai.values() if v["persistence"] is not None]),
    }
    with open(xai_dir / f"{label}_{split}_garch_summary.json", "w") as f:
        json.dump(garch_summary, f, indent=2)

    return results


def _generate_counterfactuals(model, dataset, config, garch_models, device) -> List[Dict]:
    """XAI Level 2: Counterfactual what-if analysis."""
    scenarios = []
    indices = np.random.choice(len(dataset), min(config.xai_counterfactual_scenarios, len(dataset)), replace=False)

    for idx in indices:
        sample = dataset[idx]
        emb = sample["emb"].unsqueeze(0).to(device)
        garch = sample["garch_vol"].unsqueeze(0).to(device)
        recent = sample["recent_vol"].unsqueeze(0).to(device)

        with torch.no_grad():
            orig = model(emb, garch, recent)

        # What-if: GARCH vol 20% lower
        garch_low = garch * 0.8
        with torch.no_grad():
            cf_low = model(emb, garch_low, recent)

        # What-if: recent vol 50% higher
        recent_high = recent * 1.5
        with torch.no_grad():
            cf_high = model(emb, garch, recent_high)

        scenarios.append({
            "ticker": sample["ticker"],
            "date": sample["date"],
            "original_vol10": float(orig["vol_10d"].cpu()),
            "original_vol30": float(orig["vol_30d"].cpu()),
            "original_regime": int(orig["regime_probs"].cpu().argmax()),
            "counterfactual_garch_low": {
                "condition": "GARCH forecast 20% lower",
                "vol_10d": float(cf_low["vol_10d"].cpu()),
                "vol_30d": float(cf_low["vol_30d"].cpu()),
            },
            "counterfactual_recent_high": {
                "condition": "Recent volatility 50% higher",
                "vol_10d": float(cf_high["vol_10d"].cpu()),
                "vol_30d": float(cf_high["vol_30d"].cpu()),
            },
        })

    return scenarios


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Volatility Estimation Model (GARCH + MLP)")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("inspect")

    p = sub.add_parser("hpo")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--trials", type=int, default=40)

    p = sub.add_parser("train-best")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])

    sub.add_parser("train-best-all")

    p = sub.add_parser("predict")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])

    for sp in [p for p in sub.choices.values() if p is not None]:
        sp.add_argument("--repo-root", type=str, default="")
        sp.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    config = VolatilityConfig()
    if hasattr(args, "repo_root") and args.repo_root:
        config.repo_root = args.repo_root
        config.__post_init__()
    if hasattr(args, "device"):
        config.device = args.device
    if hasattr(args, "trials") and args.trials:
        config.hpo_trials = args.trials

    if args.command == "inspect":
        print("=" * 60)
        print("VOLATILITY MODEL — DATA INSPECTION")
        print("=" * 60)
        emb_dir = Path(config.embeddings_dir)
        if emb_dir.exists():
            for f in sorted(emb_dir.glob("*_embeddings.npy")):
                emb = np.load(f, mmap_mode='r')
                print(f"  {f.name}: {emb.shape}")
            for f in sorted(emb_dir.glob("*_manifest.csv")):
                m = pd.read_csv(f)
                print(f"  {f.name}: {len(m):,} rows, {m['ticker'].nunique()} tickers, dates {m['date'].min()} → {m['date'].max()}")
        ret_df = pd.read_csv(config.returns_path, index_col=0)
        print(f"  Returns: {ret_df.shape[0]} days × {ret_df.shape[1]} tickers")
        print(f"  Date range: {ret_df.index[0]} → {ret_df.index[-1]}")

    elif args.command == "hpo":
        run_hpo(config, args.chunk)
    elif args.command == "train-best":
        label = CHUNK_CONFIG[args.chunk]["label"]
        hpo_path = Path(config.output_dir) / "codeResults" / "Volatility" / f"best_params_{label}.json"
        if hpo_path.exists():
            with open(hpo_path) as f:
                hpo = json.load(f)
            for k, v in hpo["params"].items():
                if k == "hidden_dim":
                    config.hidden_dims = [v]
                elif hasattr(config, k):
                    setattr(config, k, v)
            print(f"Loaded HPO params: {hpo['params']}")
        train_volatility_model(config, args.chunk)
    elif args.command == "train-best-all":
        for cid in [1, 2, 3]:
            label = CHUNK_CONFIG[cid]["label"]
            hpo_path = Path(config.output_dir) / "codeResults" / "Volatility" / f"best_params_{label}.json"
            if hpo_path.exists():
                with open(hpo_path) as f:
                    hpo = json.load(f)
                for k, v in hpo["params"].items():
                    if k == "hidden_dim":
                        config.hidden_dims = [v]
                    elif hasattr(config, k):
                        setattr(config, k, v)
            train_volatility_model(config, cid)
    elif args.command == "predict":
        predict_with_xai(config, args.chunk, args.split)


if __name__ == "__main__":
    main()

# ── Run instructions ──────────────────────────────────────────
# python code/riskEngine/volatility.py inspect
# python code/riskEngine/volatility.py hpo --chunk 1 --trials 40 --device cuda
# python code/riskEngine/volatility.py train-best --chunk 1 --device cuda
# python code/riskEngine/volatility.py train-best-all --device cuda
# python code/riskEngine/volatility.py predict --chunk 1 --split test --device cuda