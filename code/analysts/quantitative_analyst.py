#!/usr/bin/env python3
"""
code/analysts/quantitative_analyst.py

Trained Quantitative Analyst
============================

Project:
    fin-glassbox — Explainable Distributed Deep Learning Framework for Financial Risk Management

Purpose:
    Train a lightweight quantitative synthesis model that combines:
        - Technical Analyst outputs
        - Risk Engine scores
        - Position Sizing Engine outputs

    into a quantitative branch signal for the Fusion Engine.

Important:
    This module does NOT make the final system decision.
    It produces the quantitative branch output only.

Main design requirement:
    Uses attention-weighted pooling across risk scores.

Inputs:
    outputs/results/PositionSizing/position_sizing_chunk{chunk}_{split}.csv

Outputs:
    outputs/models/QuantitativeAnalyst/chunk{chunk}/best_model.pt
    outputs/models/QuantitativeAnalyst/chunk{chunk}/final_model.pt
    outputs/models/QuantitativeAnalyst/chunk{chunk}/scaler.npz
    outputs/codeResults/QuantitativeAnalyst/best_params_chunk{chunk}.json

    outputs/results/QuantitativeAnalyst/quantitative_analysis_chunk{chunk}_{split}.csv
    outputs/results/QuantitativeAnalyst/xai/quantitative_analysis_chunk{chunk}_{split}_xai_summary.json

CLI:
    python code/analysts/quantitative_analyst.py inspect --repo-root .
    python code/analysts/quantitative_analyst.py smoke --repo-root . --device cuda
    python code/analysts/quantitative_analyst.py hpo --repo-root . --chunk 1 --trials 30 --device cuda --fresh
    python code/analysts/quantitative_analyst.py train-best --repo-root . --chunk 1 --device cuda --fresh
    python code/analysts/quantitative_analyst.py predict --repo-root . --chunk 1 --split test --device cuda
    python code/analysts/quantitative_analyst.py predict-all --repo-root . --chunks 1 2 3 --splits val test --device cuda
    python code/analysts/quantitative_analyst.py validate --repo-root . --chunk 1 --split test
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import optuna
except Exception:
    optuna = None

warnings.filterwarnings("ignore", category=FutureWarning)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

RISK_COLUMNS = [
    "volatility_risk_score",
    "drawdown_risk_score",
    "var_cvar_risk_score",
    "contagion_risk_score",
    "liquidity_risk_score",
    "regime_risk_score",
]

RISK_NAMES = [
    "volatility",
    "drawdown",
    "var_cvar",
    "contagion",
    "liquidity",
    "regime",
]

CONTEXT_COLUMNS = [
    "trend_score",
    "momentum_score",
    "timing_confidence",
    "technical_confidence",
    "position_fraction_of_max",
    "recommended_capital_fraction",
    "recommended_capital_pct",
    "max_single_stock_exposure",
    "regime_confidence",
    "hard_cap_applied",
    "pre_cap_position_fraction_of_max",
    "pre_cap_capital_fraction",
    "risk_bucket_fraction",
]

TARGET_COLUMNS = [
    "target_quantitative_signal",
    "target_quantitative_risk",
    "target_quantitative_confidence",
]


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QuantitativeAnalystConfig:
    repo_root: str = ""

    position_sizing_dir: str = "outputs/results/PositionSizing"
    model_dir: str = "outputs/models/QuantitativeAnalyst"
    results_dir: str = "outputs/results/QuantitativeAnalyst"
    code_results_dir: str = "outputs/codeResults/QuantitativeAnalyst"

    device: str = "cuda"

    attention_dim: int = 32
    hidden_dim: int = 64
    n_layers: int = 2
    dropout: float = 0.15
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 1024
    epochs: int = 40
    early_stop_patience: int = 6
    num_workers: int = 0
    seed: int = 42

    hpo_trials: int = 30
    hpo_epochs: int = 10
    hpo_max_train_rows: int = 300_000
    hpo_max_val_rows: int = 100_000

    max_train_rows: int = 0

    buy_threshold: float = 0.18
    sell_threshold: float = -0.25
    max_risk_for_buy: float = 0.75
    severe_risk_sell_threshold: float = 0.90
    min_confidence_for_buy: float = 0.35
    min_position_fraction: float = 0.0001

    trend_weight: float = 0.40
    momentum_weight: float = 0.35
    timing_weight: float = 0.25

    confidence_technical_weight: float = 0.40
    confidence_risk_weight: float = 0.20
    confidence_position_weight: float = 0.25
    confidence_regime_weight: float = 0.15

    xai_sample_size: int = 1024
    max_xai_examples: int = 100

    def resolve_paths(self) -> "QuantitativeAnalystConfig":
        if self.repo_root:
            root = Path(self.repo_root)
            for attr in ["position_sizing_dir", "model_dir", "results_dir", "code_results_dir"]:
                value = getattr(self, attr)
                if value and not Path(value).is_absolute():
                    setattr(self, attr, str(root / value))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(json_safe(k)): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def clip01(x: Any) -> Any:
    return np.clip(x, 0.0, 1.0)


def clip11(x: Any) -> Any:
    return np.clip(x, -1.0, 1.0)


def safe_numeric(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=np.float32)

    s = pd.to_numeric(df[col], errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).fillna(default)
    return s.astype(np.float32)


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return max(sum(1 for _ in f) - 1, 0)


def dedupe(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    keys = [k for k in keys if k in df.columns]
    if not keys:
        return df.reset_index(drop=True)
    return df.drop_duplicates(keys, keep="last").reset_index(drop=True)


def position_path(config: QuantitativeAnalystConfig, chunk: int, split: str) -> Path:
    return Path(config.position_sizing_dir) / f"position_sizing_chunk{chunk}_{split}.csv"


def model_dir(config: QuantitativeAnalystConfig, chunk: int) -> Path:
    return Path(config.model_dir) / f"chunk{chunk}"


def model_path(config: QuantitativeAnalystConfig, chunk: int) -> Path:
    return model_dir(config, chunk) / "best_model.pt"


def final_model_path(config: QuantitativeAnalystConfig, chunk: int) -> Path:
    return model_dir(config, chunk) / "final_model.pt"


def scaler_path(config: QuantitativeAnalystConfig, chunk: int) -> Path:
    return model_dir(config, chunk) / "scaler.npz"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING AND TARGET CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def load_position_sizing(config: QuantitativeAnalystConfig, chunk: int, split: str) -> pd.DataFrame:
    path = position_path(config, chunk, split)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing Position Sizing output: {path}\n"
            f"Run Position Sizing first for this chunk/split."
        )

    df = pd.read_csv(path, dtype={"ticker": str}, low_memory=False)

    if "ticker" not in df.columns or "date" not in df.columns:
        raise ValueError(f"{path} must contain ticker and date columns.")

    df["ticker"] = df["ticker"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["ticker", "date"]).copy()
    df = dedupe(df, ["ticker", "date"])

    return df.reset_index(drop=True)


def compute_technical_direction(df: pd.DataFrame, config: QuantitativeAnalystConfig) -> pd.Series:
    trend = 2.0 * safe_numeric(df, "trend_score", 0.5) - 1.0
    momentum = 2.0 * safe_numeric(df, "momentum_score", 0.5) - 1.0
    timing = 2.0 * safe_numeric(df, "timing_confidence", 0.5) - 1.0

    direction = (
        config.trend_weight * trend
        + config.momentum_weight * momentum
        + config.timing_weight * timing
    )

    return pd.Series(clip11(direction), index=df.index, dtype=np.float32)


def construct_targets(df: pd.DataFrame, config: QuantitativeAnalystConfig) -> pd.DataFrame:
    out = df.copy()

    for col in RISK_COLUMNS:
        out[col] = clip01(safe_numeric(out, col, 0.5))

    for col in CONTEXT_COLUMNS:
        default = 0.0
        if col in ["trend_score", "momentum_score", "timing_confidence", "technical_confidence", "regime_confidence"]:
            default = 0.5
        out[col] = safe_numeric(out, col, default)

    out["technical_direction_score_rule"] = compute_technical_direction(out, config)

    combined_risk = safe_numeric(out, "combined_risk_score", 0.5)
    combined_risk = pd.Series(clip01(combined_risk), index=out.index, dtype=np.float32)

    position_fraction = safe_numeric(out, "position_fraction_of_max", 0.0)
    position_fraction = pd.Series(clip01(position_fraction), index=out.index, dtype=np.float32)

    technical_confidence = safe_numeric(out, "technical_confidence", 0.5)
    regime_confidence = safe_numeric(out, "regime_confidence", 0.5)

    risk_gate = 1.0 - combined_risk
    position_gate = np.where(
        position_fraction.values <= float(config.min_position_fraction),
        0.0,
        0.50 + 0.50 * position_fraction.values,
    )
    position_gate = pd.Series(clip01(position_gate), index=out.index, dtype=np.float32)

    target_signal = (
        out["technical_direction_score_rule"].values
        * (0.30 + 0.70 * risk_gate.values)
        * position_gate.values
    )
    target_signal = clip11(target_signal)

    target_confidence = (
        config.confidence_technical_weight * technical_confidence
        + config.confidence_risk_weight * risk_gate
        + config.confidence_position_weight * position_fraction
        + config.confidence_regime_weight * regime_confidence
    )
    target_confidence = clip01(target_confidence)

    out["target_quantitative_signal"] = target_signal.astype(np.float32)
    out["target_quantitative_risk"] = combined_risk.astype(np.float32)
    out["target_quantitative_confidence"] = np.asarray(target_confidence, dtype=np.float32)

    return out


def prepare_arrays(df: pd.DataFrame, config: QuantitativeAnalystConfig) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = construct_targets(df, config)

    risk = df[RISK_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.5).values.astype(np.float32)
    context = df[CONTEXT_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32)
    target = df[TARGET_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32)

    risk = np.nan_to_num(risk, nan=0.5, posinf=0.5, neginf=0.5)
    context = np.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
    target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

    target[:, 0] = clip11(target[:, 0])
    target[:, 1] = clip01(target[:, 1])
    target[:, 2] = clip01(target[:, 2])

    return risk, context, target, df


def maybe_sample(
    risk: np.ndarray,
    context: np.ndarray,
    target: np.ndarray,
    df: pd.DataFrame,
    max_rows: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    if max_rows <= 0 or len(df) <= max_rows:
        return risk, context, target, df

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=int(max_rows), replace=False)
    idx = np.sort(idx)

    return risk[idx], context[idx], target[idx], df.iloc[idx].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SCALER
# ═══════════════════════════════════════════════════════════════════════════════

class ContextScaler:
    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, context: np.ndarray) -> None:
        self.mean = context.mean(axis=0, keepdims=True).astype(np.float32)
        self.std = context.std(axis=0, keepdims=True).astype(np.float32)
        self.std[self.std < 1e-8] = 1.0

    def transform(self, context: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return context.astype(np.float32)
        return ((context - self.mean) / self.std).astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(path),
            mean=self.mean,
            std=self.std,
            context_columns=np.array(CONTEXT_COLUMNS, dtype=object),
            risk_columns=np.array(RISK_COLUMNS, dtype=object),
        )

    @classmethod
    def load(cls, path: Path) -> "ContextScaler":
        data = np.load(str(path), allow_pickle=True)
        obj = cls()
        obj.mean = data["mean"].astype(np.float32)
        obj.std = data["std"].astype(np.float32)
        return obj


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class QuantitativeRiskAttentionModel(nn.Module):
    """
    Quantitative model with attention-weighted pooling across risk scores.

    Input:
        risk_scores:    (batch, 6)
        context_feats:  (batch, context_dim)

    Output:
        quantitative_signal:      [-1, 1]
        quantitative_risk:        [0, 1]
        quantitative_confidence:  [0, 1]
        risk_attention_weights:   (batch, 6)
        attention_pooled_risk:    [0, 1]
    """

    def __init__(
        self,
        n_risks: int,
        context_dim: int,
        attention_dim: int = 32,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()

        self.n_risks = int(n_risks)
        self.context_dim = int(context_dim)
        self.attention_dim = int(attention_dim)

        self.risk_value_proj = nn.Linear(1, self.attention_dim)
        self.risk_identity_embedding = nn.Parameter(torch.randn(self.n_risks, self.attention_dim) * 0.02)
        self.context_proj = nn.Linear(self.context_dim, self.attention_dim)
        self.attention_score = nn.Linear(self.attention_dim, 1)

        mlp_input_dim = self.n_risks + self.context_dim + self.attention_dim + 1

        layers: List[nn.Module] = []
        in_dim = mlp_input_dim

        for _ in range(int(n_layers)):
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(float(dropout)))
            in_dim = int(hidden_dim)

        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 3)

    def forward(self, risk_scores: torch.Tensor, context_feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        risk_scores = torch.nan_to_num(risk_scores.float(), nan=0.5, posinf=0.5, neginf=0.5).clamp(0.0, 1.0)
        context_feats = torch.nan_to_num(context_feats.float(), nan=0.0, posinf=0.0, neginf=0.0)

        batch = risk_scores.size(0)

        risk_values = risk_scores.unsqueeze(-1)  # (B, R, 1)
        risk_emb = self.risk_value_proj(risk_values)  # (B, R, A)

        identity = self.risk_identity_embedding.unsqueeze(0).expand(batch, -1, -1)
        context_emb = self.context_proj(context_feats).unsqueeze(1)

        attention_hidden = torch.tanh(risk_emb + identity + context_emb)
        logits = self.attention_score(attention_hidden).squeeze(-1)
        attention_weights = torch.softmax(logits, dim=1)

        pooled_risk_embedding = torch.sum(attention_weights.unsqueeze(-1) * risk_emb, dim=1)
        attention_pooled_risk = torch.sum(attention_weights * risk_scores, dim=1, keepdim=True)

        mlp_in = torch.cat(
            [
                risk_scores,
                context_feats,
                pooled_risk_embedding,
                attention_pooled_risk,
            ],
            dim=1,
        )

        h = self.mlp(mlp_in)
        raw = self.head(h)

        quantitative_signal = torch.tanh(raw[:, 0])
        quantitative_risk = torch.sigmoid(raw[:, 1])
        quantitative_confidence = torch.sigmoid(raw[:, 2])

        outputs = torch.stack(
            [quantitative_signal, quantitative_risk, quantitative_confidence],
            dim=1,
        )

        return {
            "outputs": outputs,
            "quantitative_signal": quantitative_signal,
            "quantitative_risk": quantitative_risk,
            "quantitative_confidence": quantitative_confidence,
            "risk_attention_weights": attention_weights,
            "attention_pooled_risk": attention_pooled_risk.squeeze(1),
            "hidden": h,
        }


def save_model(
    model: QuantitativeRiskAttentionModel,
    config: QuantitativeAnalystConfig,
    chunk: int,
    best_val_loss: float,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": config.to_dict(),
        "risk_columns": RISK_COLUMNS,
        "risk_names": RISK_NAMES,
        "context_columns": CONTEXT_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "best_val_loss": float(best_val_loss),
    }
    torch.save(payload, path)


def load_model(
    config: QuantitativeAnalystConfig,
    chunk: int,
) -> Tuple[QuantitativeRiskAttentionModel, ContextScaler, Dict[str, Any]]:
    config.resolve_paths()

    path = model_path(config, chunk)
    if not path.exists():
        raise FileNotFoundError(f"Missing Quantitative Analyst model: {path}")

    payload = torch.load(path, map_location=config.device)
    model_cfg = payload.get("config", {})

    model = QuantitativeRiskAttentionModel(
        n_risks=len(RISK_COLUMNS),
        context_dim=len(CONTEXT_COLUMNS),
        attention_dim=int(model_cfg.get("attention_dim", config.attention_dim)),
        hidden_dim=int(model_cfg.get("hidden_dim", config.hidden_dim)),
        n_layers=int(model_cfg.get("n_layers", config.n_layers)),
        dropout=float(model_cfg.get("dropout", config.dropout)),
    ).to(config.device)

    model.load_state_dict(payload["state_dict"])
    model.eval()

    scaler = ContextScaler.load(scaler_path(config, chunk))

    return model, scaler, payload


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def make_loader(
    risk: np.ndarray,
    context: np.ndarray,
    target: np.ndarray,
    batch_size: int,
    shuffle: bool,
    device: str,
    num_workers: int,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(risk.astype(np.float32)),
        torch.from_numpy(context.astype(np.float32)),
        torch.from_numpy(target.astype(np.float32)),
    )

    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=str(device).startswith("cuda"),
        drop_last=False,
    )


def quantitative_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    signal_loss = torch.mean((pred[:, 0] - target[:, 0]) ** 2)
    risk_loss = torch.mean((pred[:, 1] - target[:, 1]) ** 2)
    confidence_loss = torch.mean((pred[:, 2] - target[:, 2]) ** 2)

    return 0.40 * signal_loss + 0.35 * risk_loss + 0.25 * confidence_loss


def train_epoch(
    model: QuantitativeRiskAttentionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()
    total = 0.0
    n = 0

    for risk, context, target in loader:
        risk = risk.to(device, non_blocking=True)
        context = context.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(risk, context)["outputs"]
        loss = quantitative_loss(out, target)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite quantitative training loss detected: {float(loss.detach().cpu())}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        bs = risk.size(0)
        total += float(loss.detach().cpu()) * bs
        n += bs

    return total / max(n, 1)


@torch.no_grad()
def validate_epoch(
    model: QuantitativeRiskAttentionModel,
    loader: DataLoader,
    device: str,
) -> float:
    model.eval()
    total = 0.0
    n = 0

    for risk, context, target in loader:
        risk = risk.to(device, non_blocking=True)
        context = context.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        out = model(risk, context)["outputs"]
        loss = quantitative_loss(out, target)

        bs = risk.size(0)
        total += float(loss.detach().cpu()) * bs
        n += bs

    return total / max(n, 1)


def train_quantitative_model(
    config: QuantitativeAnalystConfig,
    chunk: int,
    *,
    fresh: bool = False,
    hpo_mode: bool = False,
    run_tag: str = "",
) -> Tuple[QuantitativeRiskAttentionModel, float, Dict[str, Any]]:
    config.resolve_paths()
    set_seed(config.seed)

    out_dir = model_dir(config, chunk)

    if fresh and out_dir.exists() and not hpo_mode:
        print(f"  Fresh run requested. Removing: {out_dir}")
        shutil.rmtree(out_dir)

    train_df = load_position_sizing(config, chunk, "train")
    val_df = load_position_sizing(config, chunk, "val")

    train_risk, train_context, train_target, train_df = prepare_arrays(train_df, config)
    val_risk, val_context, val_target, val_df = prepare_arrays(val_df, config)

    max_train = int(config.hpo_max_train_rows if hpo_mode else config.max_train_rows)
    max_val = int(config.hpo_max_val_rows if hpo_mode else 0)

    train_risk, train_context, train_target, train_df = maybe_sample(
        train_risk, train_context, train_target, train_df, max_train, config.seed
    )

    val_risk, val_context, val_target, val_df = maybe_sample(
        val_risk, val_context, val_target, val_df, max_val, config.seed + 1
    )

    scaler = ContextScaler()
    scaler.fit(train_context)

    train_context = scaler.transform(train_context)
    val_context = scaler.transform(val_context)

    model = QuantitativeRiskAttentionModel(
        n_risks=len(RISK_COLUMNS),
        context_dim=len(CONTEXT_COLUMNS),
        attention_dim=config.attention_dim,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        dropout=config.dropout,
    ).to(config.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )

    train_loader = make_loader(
        train_risk,
        train_context,
        train_target,
        config.batch_size,
        shuffle=True,
        device=config.device,
        num_workers=config.num_workers,
    )

    val_loader = make_loader(
        val_risk,
        val_context,
        val_target,
        config.batch_size,
        shuffle=False,
        device=config.device,
        num_workers=config.num_workers,
    )

    epochs = int(config.hpo_epochs if hpo_mode else config.epochs)

    print(f"  Train rows: {len(train_df):,} | Val rows: {len(val_df):,}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"  Config: attention_dim={config.attention_dim}, hidden={config.hidden_dim}, "
        f"layers={config.n_layers}, dropout={config.dropout:.3f}, batch={config.batch_size}"
    )

    best_val = float("inf")
    best_state = None
    no_improve = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, config.device)
        val_loss = validate_epoch(model, val_loader, config.device)

        history.append({
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "lr": float(optimizer.param_groups[0]["lr"]),
        })

        prefix = f"[{run_tag}]" if run_tag else f"[chunk{chunk}]"
        print(f"  {prefix} E{epoch:03d}/{epochs} | train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_val - 1e-8:
            best_val = float(val_loss)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= int(config.early_stop_patience):
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if not hpo_mode:
        out_dir.mkdir(parents=True, exist_ok=True)

        save_model(model, config, chunk, best_val, model_path(config, chunk))
        save_model(model, config, chunk, best_val, final_model_path(config, chunk))
        scaler.save(scaler_path(config, chunk))

        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

        freeze_dir = out_dir / "model_freezed"
        unfreeze_dir = out_dir / "model_unfreezed"
        freeze_dir.mkdir(parents=True, exist_ok=True)
        unfreeze_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(model_path(config, chunk), freeze_dir / "model.pt")
        shutil.copy2(model_path(config, chunk), unfreeze_dir / "model.pt")

    summary = {
        "chunk": int(chunk),
        "best_val_loss": float(best_val),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "risk_columns": RISK_COLUMNS,
        "risk_names": RISK_NAMES,
        "context_columns": CONTEXT_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "config": config.to_dict(),
        "history": history,
    }

    return model, float(best_val), summary


# ═══════════════════════════════════════════════════════════════════════════════
# HPO
# ═══════════════════════════════════════════════════════════════════════════════

def hpo_objective(trial: Any, base_config: QuantitativeAnalystConfig, chunk: int) -> float:
    config = QuantitativeAnalystConfig(**base_config.to_dict())
    config.resolve_paths()

    config.attention_dim = trial.suggest_categorical("attention_dim", [16, 32, 64])
    config.hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 96, 128])
    config.n_layers = trial.suggest_int("n_layers", 1, 3)
    config.dropout = trial.suggest_float("dropout", 0.05, 0.40)
    config.lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    config.weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
    config.batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048])
    config.hpo_epochs = int(base_config.hpo_epochs)
    config.early_stop_patience = 3

    try:
        _, val_loss, _ = train_quantitative_model(
            config,
            chunk,
            fresh=False,
            hpo_mode=True,
            run_tag=f"hpo_trial_{trial.number:04d}",
        )

        if not np.isfinite(val_loss):
            return 1e9

        return float(val_loss)

    except Exception as exc:
        print(f"  Trial {trial.number} failed safely: {exc}")
        return 1e9


def run_hpo(config: QuantitativeAnalystConfig, chunk: int, trials: int, fresh: bool = False) -> Dict[str, Any]:
    if optuna is None:
        raise RuntimeError("Optuna is not installed, but HPO was requested.")

    config.resolve_paths()

    out_dir = Path(config.code_results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = out_dir / f"hpo_chunk{chunk}.db"
    if fresh and db_path.exists():
        db_path.unlink()
        print(f"  Deleted old HPO DB: {db_path}")

    storage = f"sqlite:///{db_path}"
    study_name = f"quantitative_analyst_chunk{chunk}"

    sampler = optuna.samplers.TPESampler(seed=config.seed)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        direction="minimize",
        load_if_exists=not fresh,
    )

    study.optimize(
        lambda trial: hpo_objective(trial, config, chunk),
        n_trials=int(trials),
        show_progress_bar=True,
    )

    valid_trials = [t for t in study.trials if t.value is not None and np.isfinite(t.value) and t.value < 1e8]
    if not valid_trials:
        raise RuntimeError("All Quantitative Analyst HPO trials failed.")

    best = study.best_trial

    result = {
        "study_name": study_name,
        "best_value": float(best.value),
        "best_params": best.params,
        "trials": len(study.trials),
        "storage": storage,
    }

    out_path = out_dir / f"best_params_chunk{chunk}.json"
    with open(out_path, "w") as f:
        json.dump(json_safe(result), f, indent=2)

    print(f"  Best HPO: {best.params} (val_loss={best.value:.6f})")
    print(f"  Saved: {out_path}")

    return result


def load_best_params(config: QuantitativeAnalystConfig, chunk: int) -> Optional[Dict[str, Any]]:
    path = Path(config.code_results_dir) / f"best_params_chunk{chunk}.json"
    if not path.exists():
        return None
    with open(path) as f:
        obj = json.load(f)
    return obj.get("best_params", obj)


def apply_best_params(config: QuantitativeAnalystConfig, params: Dict[str, Any]) -> QuantitativeAnalystConfig:
    for key, value in params.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

def classify_action(
    signal: pd.Series,
    risk: pd.Series,
    confidence: pd.Series,
    recommended_capital_fraction: pd.Series,
    config: QuantitativeAnalystConfig,
) -> pd.Series:
    rec = np.full(len(signal), "HOLD", dtype=object)

    buy = (
        (signal > float(config.buy_threshold))
        & (risk < float(config.max_risk_for_buy))
        & (confidence >= float(config.min_confidence_for_buy))
        & (recommended_capital_fraction > float(config.min_position_fraction))
    )

    sell = (
        (signal < float(config.sell_threshold))
        | (risk > float(config.severe_risk_sell_threshold))
    )

    rec[buy.values] = "BUY"
    rec[sell.values] = "SELL"

    return pd.Series(rec, index=signal.index)


def classify_risk_state(risk: pd.Series) -> pd.Series:
    values = risk.values
    labels = np.full(len(values), "low", dtype=object)

    labels[values >= 0.30] = "moderate"
    labels[values >= 0.50] = "elevated"
    labels[values >= 0.75] = "high"
    labels[values >= 0.90] = "severe"

    return pd.Series(labels, index=risk.index)


@torch.no_grad()
def predict_batches(
    model: QuantitativeRiskAttentionModel,
    risk: np.ndarray,
    context: np.ndarray,
    config: QuantitativeAnalystConfig,
) -> Dict[str, np.ndarray]:
    ds = TensorDataset(
        torch.from_numpy(risk.astype(np.float32)),
        torch.from_numpy(context.astype(np.float32)),
    )

    loader = DataLoader(
        ds,
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        pin_memory=str(config.device).startswith("cuda"),
        drop_last=False,
    )

    outputs = []
    attentions = []
    pooled = []

    model.eval()

    for rb, cb in loader:
        rb = rb.to(config.device, non_blocking=True)
        cb = cb.to(config.device, non_blocking=True)

        out = model(rb, cb)

        outputs.append(out["outputs"].detach().cpu().numpy())
        attentions.append(out["risk_attention_weights"].detach().cpu().numpy())
        pooled.append(out["attention_pooled_risk"].detach().cpu().numpy())

    return {
        "outputs": np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 3), dtype=np.float32),
        "attention": np.concatenate(attentions, axis=0) if attentions else np.zeros((0, len(RISK_COLUMNS)), dtype=np.float32),
        "pooled_risk": np.concatenate(pooled, axis=0) if pooled else np.zeros((0,), dtype=np.float32),
    }


def build_row_xai(row: pd.Series) -> str:
    return (
        f"{row['quantitative_recommendation']}: signal={row['risk_adjusted_quantitative_signal']:.3f}, "
        f"risk={row['quantitative_risk_score']:.3f}, confidence={row['quantitative_confidence']:.3f}, "
        f"attention_top_risk={row['top_attention_risk_driver']}, "
        f"position={row['recommended_capital_pct']:.2f}%."
    )


def predict_quantitative(config: QuantitativeAnalystConfig, chunk: int, split: str) -> Dict[str, Any]:
    config.resolve_paths()

    print("=" * 90)
    print(f"QUANTITATIVE ANALYST PREDICT — chunk{chunk}_{split}")
    print("=" * 90)

    model, scaler, payload = load_model(config, chunk)

    df = load_position_sizing(config, chunk, split)
    risk, context, target, df = prepare_arrays(df, config)
    context_scaled = scaler.transform(context)

    pred = predict_batches(model, risk, context_scaled, config)

    outputs = pred["outputs"]
    attention = pred["attention"]
    pooled_risk = pred["pooled_risk"]

    df["risk_adjusted_quantitative_signal"] = clip11(outputs[:, 0])
    df["quantitative_risk_score"] = clip01(outputs[:, 1])
    df["quantitative_confidence"] = clip01(outputs[:, 2])
    df["attention_pooled_risk_score"] = clip01(pooled_risk)

    for i, name in enumerate(RISK_NAMES):
        df[f"risk_attention_{name}"] = attention[:, i]

    top_idx = np.argmax(attention, axis=1)
    df["top_attention_risk_driver"] = [RISK_NAMES[int(i)] for i in top_idx]

    df["technical_direction_score"] = df["technical_direction_score_rule"]
    df["quantitative_risk_state"] = classify_risk_state(df["quantitative_risk_score"])

    df["recommended_capital_fraction"] = safe_numeric(df, "recommended_capital_fraction", 0.0)
    df["recommended_capital_pct"] = safe_numeric(df, "recommended_capital_pct", 0.0)

    df["quantitative_recommendation"] = classify_action(
        df["risk_adjusted_quantitative_signal"],
        df["quantitative_risk_score"],
        df["quantitative_confidence"],
        df["recommended_capital_fraction"],
        config,
    )

    df["quantitative_action_strength"] = (
        np.abs(df["risk_adjusted_quantitative_signal"].values)
        * df["quantitative_confidence"].values
    ).astype(np.float32)

    df["xai_summary"] = df.apply(build_row_xai, axis=1)

    df["chunk"] = int(chunk)
    df["split"] = split

    preferred_cols = [
        "ticker", "date", "chunk", "split",

        "quantitative_recommendation",
        "risk_adjusted_quantitative_signal",
        "technical_direction_score",
        "quantitative_risk_score",
        "quantitative_risk_state",
        "quantitative_confidence",
        "quantitative_action_strength",

        "attention_pooled_risk_score",
        "top_attention_risk_driver",
        "risk_attention_volatility",
        "risk_attention_drawdown",
        "risk_attention_var_cvar",
        "risk_attention_contagion",
        "risk_attention_liquidity",
        "risk_attention_regime",

        "recommended_capital_fraction",
        "recommended_capital_pct",
        "position_fraction_of_max",
        "max_single_stock_exposure",

        "trend_score",
        "momentum_score",
        "timing_confidence",
        "technical_confidence",

        "volatility_risk_score",
        "drawdown_risk_score",
        "var_cvar_risk_score",
        "contagion_risk_score",
        "liquidity_risk_score",
        "regime_risk_score",
        "combined_risk_score",

        "binding_cap_source",
        "hard_cap_applied",
        "size_bucket",
        "regime_label",
        "regime_confidence",

        "xai_summary",
    ]

    cols = [c for c in preferred_cols if c in df.columns]
    extras = [c for c in df.columns if c not in cols and not c.startswith("_")]
    out_df = df[cols + extras].copy()

    results_dir = Path(config.results_dir)
    xai_dir = results_dir / "xai"
    results_dir.mkdir(parents=True, exist_ok=True)
    xai_dir.mkdir(parents=True, exist_ok=True)

    pred_path = results_dir / f"quantitative_analysis_chunk{chunk}_{split}.csv"
    xai_path = xai_dir / f"quantitative_analysis_chunk{chunk}_{split}_xai_summary.json"

    out_df.to_csv(pred_path, index=False)

    xai = build_xai_report(out_df, config, chunk, split, payload)

    with open(xai_path, "w") as f:
        json.dump(json_safe(xai), f, indent=2)

    print(f"  saved: {pred_path} rows={len(out_df):,}")
    print(f"  xai:   {xai_path}")
    print("  recommendation counts:")
    print(out_df["quantitative_recommendation"].value_counts().to_string())

    return {
        "predictions": out_df,
        "xai": xai,
        "paths": {
            "predictions": str(pred_path),
            "xai": str(xai_path),
        },
    }


def build_xai_report(
    df: pd.DataFrame,
    config: QuantitativeAnalystConfig,
    chunk: int,
    split: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    attention_cols = [f"risk_attention_{name}" for name in RISK_NAMES]

    report = {
        "module": "QuantitativeAnalyst",
        "chunk": int(chunk),
        "split": split,
        "config": config.to_dict(),
        "model_best_val_loss": payload.get("best_val_loss"),
        "rows": int(len(df)),
        "ticker_count": int(df["ticker"].nunique()) if "ticker" in df.columns else 0,
        "date_min": str(pd.to_datetime(df["date"]).min().date()) if len(df) else None,
        "date_max": str(pd.to_datetime(df["date"]).max().date()) if len(df) else None,
        "plain_english": (
            "The trained Quantitative Analyst combines technical signals, risk scores, and position sizing outputs. "
            "It uses attention-weighted pooling across volatility, drawdown, VaR/CVaR, contagion, liquidity, and regime risk. "
            "The attention weights identify which risk modules most strongly shaped the quantitative output."
        ),
        "recommendation_counts": df["quantitative_recommendation"].value_counts().to_dict(),
        "risk_state_counts": df["quantitative_risk_state"].value_counts().to_dict(),
        "top_attention_risk_driver_counts": df["top_attention_risk_driver"].value_counts().to_dict(),
        "summary_stats": {
            "signal_mean": float(df["risk_adjusted_quantitative_signal"].mean()),
            "signal_std": float(df["risk_adjusted_quantitative_signal"].std()),
            "risk_mean": float(df["quantitative_risk_score"].mean()),
            "confidence_mean": float(df["quantitative_confidence"].mean()),
            "attention_pooled_risk_mean": float(df["attention_pooled_risk_score"].mean()),
            "recommended_capital_pct_mean": float(df["recommended_capital_pct"].mean()),
        },
        "mean_attention_weights": {
            col.replace("risk_attention_", ""): float(df[col].mean())
            for col in attention_cols
            if col in df.columns
        },
    }

    example_cols = [
        "ticker", "date", "quantitative_recommendation",
        "risk_adjusted_quantitative_signal", "quantitative_risk_score",
        "quantitative_confidence", "recommended_capital_pct",
        "top_attention_risk_driver", "xai_summary",
    ]
    example_cols = [c for c in example_cols if c in df.columns]

    report["strongest_buy_examples"] = (
        df[df["quantitative_recommendation"] == "BUY"]
        .sort_values("risk_adjusted_quantitative_signal", ascending=False)
        .head(config.max_xai_examples)[example_cols]
        .to_dict(orient="records")
    )

    report["strongest_sell_examples"] = (
        df[df["quantitative_recommendation"] == "SELL"]
        .sort_values("risk_adjusted_quantitative_signal", ascending=True)
        .head(config.max_xai_examples)[example_cols]
        .to_dict(orient="records")
    )

    report["highest_risk_examples"] = (
        df.sort_values("quantitative_risk_score", ascending=False)
        .head(config.max_xai_examples)[example_cols]
        .to_dict(orient="records")
    )

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# INSPECT / VALIDATE / SMOKE
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_inspect(config: QuantitativeAnalystConfig) -> None:
    config.resolve_paths()

    print("=" * 100)
    print("QUANTITATIVE ANALYST INPUT INSPECTION")
    print("=" * 100)

    for chunk in [1, 2, 3]:
        for split in ["train", "val", "test"]:
            p = position_path(config, chunk, split)
            rows = count_rows(p) if p.exists() else 0

            print(f"\nchunk{chunk}_{split}")
            print(f"  {'OK' if p.exists() else 'MISSING'} rows={rows:,} path={p}")

            if p.exists():
                try:
                    df = pd.read_csv(p, nrows=2)
                    print(f"  columns={list(df.columns)[:40]}")
                    print(df.head(2).to_string(index=False))
                except Exception as exc:
                    print(f"  could not read: {exc}")


def cmd_validate(config: QuantitativeAnalystConfig, chunk: int, split: str) -> None:
    config.resolve_paths()

    path = Path(config.results_dir) / f"quantitative_analysis_chunk{chunk}_{split}.csv"

    print("=" * 100)
    print(f"QUANTITATIVE ANALYST VALIDATION — chunk{chunk}_{split}")
    print("=" * 100)

    if not path.exists():
        raise FileNotFoundError(f"Missing output: {path}")

    df = pd.read_csv(path, dtype={"ticker": str})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    required = [
        "ticker", "date", "quantitative_recommendation",
        "risk_adjusted_quantitative_signal",
        "quantitative_risk_score",
        "quantitative_confidence",
        "attention_pooled_risk_score",
        "top_attention_risk_driver",
        "recommended_capital_fraction",
        "recommended_capital_pct",
        "xai_summary",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    attention_cols = [f"risk_attention_{name}" for name in RISK_NAMES]
    missing_attention = [c for c in attention_cols if c not in df.columns]
    if missing_attention:
        raise RuntimeError(f"Missing attention columns: {missing_attention}")

    numeric = df.select_dtypes(include="number")
    finite_ratio = float(np.isfinite(numeric.values).mean()) if len(numeric.columns) else 1.0

    invalid_signal = int((df["risk_adjusted_quantitative_signal"].abs() > 1.00001).sum())
    invalid_risk = int(((df["quantitative_risk_score"] < -1e-9) | (df["quantitative_risk_score"] > 1.00001)).sum())
    invalid_conf = int(((df["quantitative_confidence"] < -1e-9) | (df["quantitative_confidence"] > 1.00001)).sum())
    negative_position = int((df["recommended_capital_fraction"] < -1e-9).sum())

    attn_sum = df[attention_cols].sum(axis=1)
    bad_attention_sum = int((np.abs(attn_sum - 1.0) > 1e-4).sum())

    print(f"rows={len(df):,}")
    print(f"tickers={df['ticker'].nunique():,}")
    print(f"date range={df['date'].min().date()} → {df['date'].max().date()}")
    print(f"numeric finite ratio={finite_ratio:.6f}")
    print(f"invalid_signal={invalid_signal}")
    print(f"invalid_risk={invalid_risk}")
    print(f"invalid_confidence={invalid_conf}")
    print(f"negative_position={negative_position}")
    print(f"bad_attention_sum={bad_attention_sum}")

    print("\nrecommendation counts:")
    print(df["quantitative_recommendation"].value_counts().to_string())

    print("\nattention driver counts:")
    print(df["top_attention_risk_driver"].value_counts().to_string())

    print("\nmean attention:")
    print(df[attention_cols].mean().to_string())

    if invalid_signal or invalid_risk or invalid_conf or negative_position or bad_attention_sum:
        raise RuntimeError("Quantitative Analyst validation failed.")

    print("\nVALIDATION PASSED")


def cmd_smoke(config: QuantitativeAnalystConfig) -> None:
    print("=" * 100)
    print("QUANTITATIVE ANALYST SMOKE TEST")
    print("=" * 100)

    set_seed(config.seed)

    n = 512
    rng = np.random.default_rng(config.seed)

    df = pd.DataFrame({
        "ticker": [f"T{i % 16:03d}" for i in range(n)],
        "date": pd.to_datetime("2024-01-01") + pd.to_timedelta(np.arange(n) % 20, unit="D"),

        "trend_score": rng.uniform(0.0, 1.0, n),
        "momentum_score": rng.uniform(0.0, 1.0, n),
        "timing_confidence": rng.uniform(0.0, 1.0, n),
        "technical_confidence": rng.uniform(0.0, 1.0, n),

        "position_fraction_of_max": rng.uniform(0.0, 1.0, n),
        "recommended_capital_fraction": rng.uniform(0.0, 0.10, n),
        "recommended_capital_pct": rng.uniform(0.0, 10.0, n),
        "max_single_stock_exposure": 0.10,
        "regime_confidence": rng.uniform(0.0, 1.0, n),
        "hard_cap_applied": rng.integers(0, 2, n),

        "pre_cap_position_fraction_of_max": rng.uniform(0.0, 1.0, n),
        "pre_cap_capital_fraction": rng.uniform(0.0, 0.10, n),
        "risk_bucket_fraction": rng.choice([0.0, 0.25, 0.50, 0.75, 1.0], n),

        "volatility_risk_score": rng.uniform(0.0, 1.0, n),
        "drawdown_risk_score": rng.uniform(0.0, 1.0, n),
        "var_cvar_risk_score": rng.uniform(0.0, 1.0, n),
        "contagion_risk_score": rng.uniform(0.0, 1.0, n),
        "liquidity_risk_score": rng.uniform(0.0, 1.0, n),
        "regime_risk_score": rng.uniform(0.0, 1.0, n),
        "combined_risk_score": rng.uniform(0.0, 1.0, n),
    })

    risk, context, target, df = prepare_arrays(df, config)

    scaler = ContextScaler()
    scaler.fit(context)
    context = scaler.transform(context)

    model = QuantitativeRiskAttentionModel(
        n_risks=len(RISK_COLUMNS),
        context_dim=len(CONTEXT_COLUMNS),
        attention_dim=16,
        hidden_dim=32,
        n_layers=2,
        dropout=0.10,
    ).to(config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = make_loader(risk, context, target, batch_size=128, shuffle=True, device=config.device, num_workers=0)

    for _ in range(3):
        loss = train_epoch(model, loader, optimizer, config.device)

    val_loss = validate_epoch(model, loader, config.device)

    with torch.no_grad():
        rb = torch.from_numpy(risk[:32]).to(config.device)
        cb = torch.from_numpy(context[:32]).to(config.device)
        out = model(rb, cb)

    assert np.isfinite(val_loss)
    assert out["outputs"].shape == (32, 3)
    assert out["risk_attention_weights"].shape == (32, len(RISK_COLUMNS))
    assert torch.allclose(out["risk_attention_weights"].sum(dim=1), torch.ones(32, device=config.device), atol=1e-5)

    print(f"SMOKE TEST PASSED | loss={val_loss:.6f} | output_shape={tuple(out['outputs'].shape)}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_hpo(config: QuantitativeAnalystConfig, chunk: int, trials: int, fresh: bool) -> None:
    print("=" * 90)
    print(f"QUANTITATIVE ANALYST HPO — chunk{chunk} ({trials} trials)")
    print("=" * 90)
    run_hpo(config, chunk, trials, fresh=fresh)


def cmd_train_best(config: QuantitativeAnalystConfig, chunk: int, fresh: bool) -> None:
    config.resolve_paths()

    best = load_best_params(config, chunk)
    if best is not None:
        print(f"Loaded best params for chunk{chunk}: {best}")
        config = apply_best_params(config, best)
    else:
        print(f"No HPO params found for chunk{chunk}; using default config.")

    print("=" * 90)
    print(f"QUANTITATIVE ANALYST TRAINING — chunk{chunk}")
    print("=" * 90)

    _, best_val, _ = train_quantitative_model(config, chunk, fresh=fresh, hpo_mode=False)
    print(f"  Complete. Best val loss: {best_val:.6f}")


def cmd_predict_all(config: QuantitativeAnalystConfig, chunks: List[int], splits: List[str]) -> None:
    for c in chunks:
        for s in splits:
            predict_quantitative(config, c, s)


# ═══════════════════════════════════════════════════════════════════════════════
# ARGPARSE
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trained Quantitative Analyst with Risk Attention")
    sub = parser.add_subparsers(dest="command")

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--repo-root", type=str, default="")
        p.add_argument("--device", type=str, default="cuda")
        p.add_argument("--batch-size", type=int, default=None)
        p.add_argument("--epochs", type=int, default=None)
        p.add_argument("--lr", type=float, default=None)
        p.add_argument("--num-workers", type=int, default=None)
        p.add_argument("--max-train-rows", type=int, default=None)

    p = sub.add_parser("inspect")
    add_common(p)

    p = sub.add_parser("smoke")
    add_common(p)

    p = sub.add_parser("hpo")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--fresh", action="store_true")
    add_common(p)

    p = sub.add_parser("train-best")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--fresh", action="store_true")
    add_common(p)

    p = sub.add_parser("predict")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    add_common(p)

    p = sub.add_parser("predict-all")
    p.add_argument("--chunks", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    p.add_argument("--splits", type=str, nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    add_common(p)

    p = sub.add_parser("validate")
    p.add_argument("--chunk", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    add_common(p)

    return parser


def config_from_args(args: argparse.Namespace) -> QuantitativeAnalystConfig:
    config = QuantitativeAnalystConfig()

    if getattr(args, "repo_root", ""):
        config.repo_root = args.repo_root

    config.device = getattr(args, "device", "cuda")
    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        config.device = "cpu"

    if getattr(args, "batch_size", None) is not None:
        config.batch_size = int(args.batch_size)

    if getattr(args, "epochs", None) is not None:
        config.epochs = int(args.epochs)

    if getattr(args, "lr", None) is not None:
        config.lr = float(args.lr)

    if getattr(args, "num_workers", None) is not None:
        config.num_workers = int(args.num_workers)

    if getattr(args, "max_train_rows", None) is not None:
        config.max_train_rows = int(args.max_train_rows)

    return config.resolve_paths()


def main() -> None:
    parser = build_parser()
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
        cmd_hpo(config, args.chunk, args.trials, args.fresh)

    elif args.command == "train-best":
        cmd_train_best(config, args.chunk, args.fresh)

    elif args.command == "predict":
        predict_quantitative(config, args.chunk, args.split)

    elif args.command == "predict-all":
        cmd_predict_all(config, args.chunks, args.splits)

    elif args.command == "validate":
        cmd_validate(config, args.chunk, args.split)


if __name__ == "__main__":
    main()


# ── Run instructions ─────────────────────────────────────────────────────────
# Compile:
# python -m py_compile code/analysts/quantitative_analyst.py
#
# Inspect:
# python code/analysts/quantitative_analyst.py inspect --repo-root .
#
# Smoke:
# python code/analysts/quantitative_analyst.py smoke --repo-root . --device cuda
#
# If train position-sizing outputs are missing, generate them first:
# python code/riskEngine/position_sizing.py run-all --repo-root . --chunks 1 2 3 --splits train val test --exposure-mode moderate --horizon-mode short
#
# HPO:
# python code/analysts/quantitative_analyst.py hpo --repo-root . --chunk 1 --trials 30 --device cuda --fresh
#
# Train:
# python code/analysts/quantitative_analyst.py train-best --repo-root . --chunk 1 --device cuda --fresh
#
# Predict:
# python code/analysts/quantitative_analyst.py predict --repo-root . --chunk 1 --split test --device cuda
#
# Full trained quantitative analyst rerun:
# cd ~/fin-glassbox && python code/riskEngine/position_sizing.py run-all --repo-root . --chunks 1 2 3 --splits train val test --exposure-mode moderate --horizon-mode short && python code/analysts/quantitative_analyst.py hpo --repo-root . --chunk 1 --trials 30 --device cuda --fresh && python code/analysts/quantitative_analyst.py train-best --repo-root . --chunk 1 --device cuda --fresh && python code/analysts/quantitative_analyst.py predict-all --repo-root . --chunks 1 --splits val test --device cuda && python code/analysts/quantitative_analyst.py hpo --repo-root . --chunk 2 --trials 30 --device cuda --fresh && python code/analysts/quantitative_analyst.py train-best --repo-root . --chunk 2 --device cuda --fresh && python code/analysts/quantitative_analyst.py predict-all --repo-root . --chunks 2 --splits val test --device cuda && python code/analysts/quantitative_analyst.py hpo --repo-root . --chunk 3 --trials 30 --device cuda --fresh && python code/analysts/quantitative_analyst.py train-best --repo-root . --chunk 3 --device cuda --fresh && python code/analysts/quantitative_analyst.py predict-all --repo-root . --chunks 3 --splits val test --device cuda
