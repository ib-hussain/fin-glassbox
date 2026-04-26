#!/usr/bin/env python3
"""
Supervised Sentiment Analyst for the fin-glassbox project.

This module trains an MLP analyst on real FinBERT text embeddings and real
market-derived labels produced by code/analysts/text_market_label_builder.py.

It does not create dummy data, synthetic labels, or fake embeddings.

Core contract:
- Input embeddings: outputs/embeddings/FinBERT/chunk{N}_{split}_embeddings.npy
- Input labels: outputs/results/analysts/labels/text_market_labels_chunk{N}_{split}.csv
- Embedding dimension: 256 float32 values per row
- Label rows must align row-for-row with embedding rows
- Training uses only the train split of the selected chronological chunk
- Validation/test are never used to fit preprocessing statistics or thresholds

Outputs:
- PyTorch checkpoints: outputs/models/analysts/sentiment/chunk{N}/latest.pt and best.pt
- Training history: outputs/results/analysts/sentiment/chunk{N}_training_history.csv
- Evaluation metrics: outputs/results/analysts/sentiment/chunk{N}_{split}_metrics.json
- Predictions CSV: outputs/results/analysts/sentiment/chunk{N}_{split}_predictions.csv
- 64-dim analyst embeddings: outputs/embeddings/analysts/sentiment/chunk{N}_{split}_sentiment_embeddings.npy

British English is used in documentation and messages.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

try:
    import optuna
except Exception as ebbbbbb:
    print(f"Warning: optuna is not available, HPO functionality will be disabled. Install optuna to enable HPO: pip install optuna.\nError was: {ebbbbbb}")
    optuna = None
    exit(-1)


APPROVED_SPLITS: Dict[int, Dict[str, List[int]]] = {
    1: {"train": [2000, 2001, 2002, 2003, 2004], "val": [2005], "test": [2006]},
    2: {"train": list(range(2007, 2015)), "val": [2015], "test": [2016]},
    3: {"train": list(range(2017, 2023)), "val": [2023], "test": [2024]},
}

REQUIRED_LABEL_COLUMNS = [
    "chunk_id",
    "doc_id",
    "year",
    "form_type",
    "cik",
    "filing_date",
    "accession",
    "source_name",
    "chunk_index",
    "word_count",
    "metadata_row_index",
    "split",
    "ticker",
    "ticker_in_market_panel",
    "label_available",
    "sentiment_score_target",
    "sentiment_class_target",
]

TARGET_SCORE_COLUMN = "sentiment_score_target"
TARGET_CLASS_COLUMN = "sentiment_class_target"
CLASS_TO_INDEX = {-1: 0, 0: 1, 1: 2}
INDEX_TO_CLASS = {0: -1, 1: 0, 2: 1}


def now_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_repo_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise ValueError("At least one integer value is required")
    return out


def parse_split_list(text: str) -> List[str]:
    out = [x.strip() for x in str(text).split(",") if x.strip()]
    allowed = {"train", "val", "test"}
    bad = [x for x in out if x not in allowed]
    if bad:
        raise ValueError(f"Unsupported split(s): {bad}. Allowed: train,val,test")
    return out




def parse_hidden_dims(text: str) -> List[int]:
    dims: List[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            value = int(part)
            if value <= 0:
                raise ValueError("Hidden dimensions must be positive integers")
            dims.append(value)
    if not dims:
        raise ValueError("At least one hidden dimension is required")
    return dims

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device(preferred: str) -> torch.device:
    preferred = preferred.lower().strip()
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


@dataclass
class SentimentConfig:
    repo_root: Path = Path(".")
    env_file: Path = Path(".env")

    embeddings_dir: Path = Path("outputs/embeddings/FinBERT")
    labels_dir: Path = Path("outputs/results/analysts/labels")
    analyst_embeddings_dir: Path = Path("outputs/embeddings/analysts/sentiment")
    models_dir: Path = Path("outputs/models/analysts/sentiment")
    results_dir: Path = Path("outputs/results/analysts/sentiment")
    code_results_dir: Path = Path("outputs/codeResults/analysts/sentiment")

    chunk_id: int = 1
    input_dim: int = 256
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    representation_dim: int = 64
    dropout: float = 0.15
    use_metadata_features: bool = True

    seed: int = 42
    device: str = "cuda"
    num_workers: int = 0
    batch_size: int = 512
    eval_batch_size: int = 1024
    epochs: int = 40
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    early_stop_patience: int = 8
    mixed_precision: bool = True

    polarity_loss_weight: float = 1.0
    class_loss_weight: float = 0.5
    magnitude_loss_weight: float = 0.25

    max_train_rows: Optional[int] = None
    max_val_rows: Optional[int] = None
    max_test_rows: Optional[int] = None
    resume: bool = False
    overwrite_predictions: bool = True

    def resolve(self) -> "SentimentConfig":
        self.repo_root = Path(self.repo_root).resolve()
        env = read_env_file(self.repo_root / self.env_file)

        if env.get("FinBERTembeddingsPath"):
            self.embeddings_dir = Path(env["FinBERTembeddingsPath"])
        elif env.get("embeddingsPathGlobal"):
            self.embeddings_dir = Path(env["embeddingsPathGlobal"]) / "FinBERT"

        if env.get("embeddingsPathGlobal"):
            self.analyst_embeddings_dir = Path(env["embeddingsPathGlobal"]) / "analysts" / "sentiment"

        if env.get("modelsPathGlobal"):
            self.models_dir = Path(env["modelsPathGlobal"]) / "analysts" / "sentiment"

        if env.get("resultsPathGlobal"):
            self.labels_dir = Path(env["resultsPathGlobal"]) / "analysts" / "labels"
            self.results_dir = Path(env["resultsPathGlobal"]) / "analysts" / "sentiment"

        if env.get("codeOutputsPathGlobal"):
            self.code_results_dir = Path(env["codeOutputsPathGlobal"]) / "analysts" / "sentiment"

        self.embeddings_dir = resolve_repo_path(self.repo_root, self.embeddings_dir)
        self.labels_dir = resolve_repo_path(self.repo_root, self.labels_dir)
        self.analyst_embeddings_dir = resolve_repo_path(self.repo_root, self.analyst_embeddings_dir)
        self.models_dir = resolve_repo_path(self.repo_root, self.models_dir)
        self.results_dir = resolve_repo_path(self.repo_root, self.results_dir)
        self.code_results_dir = resolve_repo_path(self.repo_root, self.code_results_dir)
        return self

    def serialisable(self) -> Dict[str, Any]:
        out = asdict(self)
        for key, value in list(out.items()):
            if isinstance(value, Path):
                out[key] = str(value)
        return out

    def chunk_model_dir(self) -> Path:
        return self.models_dir / f"chunk{self.chunk_id}"

    def preprocessor_path(self) -> Path:
        return self.chunk_model_dir() / "metadata_preprocessor.json"

    def latest_checkpoint_path(self) -> Path:
        return self.chunk_model_dir() / "latest.pt"

    def best_checkpoint_path(self) -> Path:
        return self.chunk_model_dir() / "best.pt"


def embedding_path(cfg: SentimentConfig, split: str) -> Path:
    return cfg.embeddings_dir / f"chunk{cfg.chunk_id}_{split}_embeddings.npy"


def label_path(cfg: SentimentConfig, split: str) -> Path:
    return cfg.labels_dir / f"text_market_labels_chunk{cfg.chunk_id}_{split}.csv"


def predictions_path(cfg: SentimentConfig, split: str) -> Path:
    return cfg.results_dir / f"chunk{cfg.chunk_id}_{split}_predictions.csv"


def metrics_path(cfg: SentimentConfig, split: str) -> Path:
    return cfg.results_dir / f"chunk{cfg.chunk_id}_{split}_metrics.json"


def analyst_embedding_path(cfg: SentimentConfig, split: str) -> Path:
    return cfg.analyst_embeddings_dir / f"chunk{cfg.chunk_id}_{split}_sentiment_embeddings.npy"


def history_path(cfg: SentimentConfig) -> Path:
    return cfg.results_dir / f"chunk{cfg.chunk_id}_training_history.csv"


def read_label_file(cfg: SentimentConfig, split: str) -> pd.DataFrame:
    path = label_path(cfg, split)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing supervised label CSV: {path}. Run code/analysts/text_market_label_builder.py before training."
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_LABEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Label CSV {path} is missing required columns: {missing}")
    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    expected_years = set(APPROVED_SPLITS[cfg.chunk_id][split])
    observed_years = set(df["year"].dropna().astype(int).unique().tolist())
    if observed_years and not observed_years.issubset(expected_years):
        raise ValueError(f"Unexpected years in {path}: observed={sorted(observed_years)} expected_subset={sorted(expected_years)}")
    return df


def load_embeddings(cfg: SentimentConfig, split: str, mmap_mode: str = "r") -> np.ndarray:
    path = embedding_path(cfg, split)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing FinBERT embedding file: {path}. Wait for the FinBERT encoder pipeline to finish this split."
        )
    arr = np.load(path, mmap_mode=mmap_mode)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embeddings in {path}, got shape={arr.shape}")
    if arr.shape[1] != cfg.input_dim:
        raise ValueError(f"Expected {cfg.input_dim}-dim FinBERT embeddings in {path}, got shape={arr.shape}")
    return arr


def validate_alignment(cfg: SentimentConfig, split: str, labels: pd.DataFrame, embeddings: np.ndarray) -> None:
    if len(labels) != int(embeddings.shape[0]):
        raise ValueError(
            f"Row alignment failure for chunk{cfg.chunk_id}_{split}: labels rows={len(labels):,}, embeddings rows={embeddings.shape[0]:,}. "
            "The label CSV must be row-aligned with the corresponding FinBERT .npy matrix."
        )
    if "metadata_row_index" in labels.columns:
        idx = pd.to_numeric(labels["metadata_row_index"], errors="coerce")
        expected = np.arange(len(labels), dtype=np.int64)
        if idx.notna().all() and not np.array_equal(idx.astype(np.int64).to_numpy(), expected):
            raise ValueError(f"metadata_row_index is not sequential for chunk{cfg.chunk_id}_{split}")


def valid_supervised_rows(df: pd.DataFrame) -> np.ndarray:
    score = pd.to_numeric(df[TARGET_SCORE_COLUMN], errors="coerce")
    klass = pd.to_numeric(df[TARGET_CLASS_COLUMN], errors="coerce")
    available = df["label_available"].astype(str).str.lower().isin(["true", "1", "yes"])
    valid_class = klass.isin([-1, 0, 1])
    valid = score.notna() & valid_class & available
    return np.flatnonzero(valid.to_numpy())


def deterministic_row_subset(indices: np.ndarray, max_rows: Optional[int], seed: int) -> np.ndarray:
    if max_rows is None or max_rows <= 0 or len(indices) <= max_rows:
        return indices
    rng = np.random.default_rng(seed)
    selected = rng.choice(indices, size=max_rows, replace=False)
    return np.sort(selected.astype(np.int64))


class MetadataPreprocessor:
    """Train-only metadata encoder for allowed filing-time metadata features."""

    numeric_columns = ["year", "word_count", "chunk_index"]

    def __init__(self, use_metadata_features: bool = True):
        self.use_metadata_features = bool(use_metadata_features)
        self.numeric_mean: Dict[str, float] = {}
        self.numeric_std: Dict[str, float] = {}
        self.form_type_vocab: Dict[str, int] = {"<UNK>": 0}
        self.feature_dim: int = 0
        self.fitted: bool = False

    @staticmethod
    def _normalise_form_type(value: Any) -> str:
        if pd.isna(value):
            return "<UNK>"
        text = str(value).strip().upper()
        return text if text else "<UNK>"

    def fit(self, df: pd.DataFrame, row_indices: np.ndarray) -> "MetadataPreprocessor":
        if not self.use_metadata_features:
            self.feature_dim = 0
            self.fitted = True
            return self

        train = df.iloc[row_indices].copy()
        for col in self.numeric_columns:
            values = pd.to_numeric(train[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            mean = float(values.mean()) if values.notna().any() else 0.0
            std = float(values.std(ddof=0)) if values.notna().any() else 1.0
            if not np.isfinite(std) or std < 1e-8:
                std = 1.0
            self.numeric_mean[col] = mean
            self.numeric_std[col] = std

        forms = sorted({self._normalise_form_type(x) for x in train["form_type"].tolist()})
        self.form_type_vocab = {"<UNK>": 0}
        for form in forms:
            if form not in self.form_type_vocab:
                self.form_type_vocab[form] = len(self.form_type_vocab)

        self.feature_dim = len(self.numeric_columns) + len(self.form_type_vocab)
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("MetadataPreprocessor must be fitted before transform")
        if not self.use_metadata_features:
            return np.zeros((len(df), 0), dtype=np.float32)

        numeric_parts: List[np.ndarray] = []
        for col in self.numeric_columns:
            values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(self.numeric_mean[col])
            scaled = (values.to_numpy(dtype=np.float32) - np.float32(self.numeric_mean[col])) / np.float32(self.numeric_std[col])
            numeric_parts.append(scaled.reshape(-1, 1))

        form_mat = np.zeros((len(df), len(self.form_type_vocab)), dtype=np.float32)
        form_values = [self._normalise_form_type(x) for x in df["form_type"].tolist()]
        for i, form in enumerate(form_values):
            j = self.form_type_vocab.get(form, 0)
            form_mat[i, j] = 1.0

        return np.concatenate(numeric_parts + [form_mat], axis=1).astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "use_metadata_features": self.use_metadata_features,
            "numeric_columns": self.numeric_columns,
            "numeric_mean": self.numeric_mean,
            "numeric_std": self.numeric_std,
            "form_type_vocab": self.form_type_vocab,
            "feature_dim": self.feature_dim,
            "fitted": self.fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetadataPreprocessor":
        obj = cls(use_metadata_features=bool(data.get("use_metadata_features", True)))
        obj.numeric_mean = {str(k): float(v) for k, v in data.get("numeric_mean", {}).items()}
        obj.numeric_std = {str(k): float(v) for k, v in data.get("numeric_std", {}).items()}
        obj.form_type_vocab = {str(k): int(v) for k, v in data.get("form_type_vocab", {"<UNK>": 0}).items()}
        obj.feature_dim = int(data.get("feature_dim", 0))
        obj.fitted = bool(data.get("fitted", True))
        return obj

    def save(self, path: Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> "MetadataPreprocessor":
        if not path.exists():
            raise FileNotFoundError(f"Missing metadata preprocessor: {path}")
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


class LabelledEmbeddingDataset(Dataset):
    def __init__(
        self,
        embeddings: np.ndarray,
        labels: pd.DataFrame,
        metadata_features: np.ndarray,
        row_indices: np.ndarray,
        require_targets: bool = True,
    ):
        self.embeddings = embeddings
        self.labels = labels
        self.metadata_features = metadata_features
        self.row_indices = row_indices.astype(np.int64)
        self.require_targets = bool(require_targets)

        if len(labels) != embeddings.shape[0]:
            raise ValueError("labels and embeddings must have the same row count")
        if metadata_features.shape[0] != embeddings.shape[0]:
            raise ValueError("metadata_features and embeddings must have the same row count")

        self.score_values = pd.to_numeric(labels[TARGET_SCORE_COLUMN], errors="coerce").to_numpy(dtype=np.float32)
        class_raw = pd.to_numeric(labels[TARGET_CLASS_COLUMN], errors="coerce")
        class_idx = np.full(len(labels), -1, dtype=np.int64)
        for value, mapped in CLASS_TO_INDEX.items():
            class_idx[class_raw == value] = mapped
        self.class_values = class_idx
        self.magnitude_values = np.clip(np.abs(self.score_values), 0.0, 1.0).astype(np.float32)

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        row = int(self.row_indices[i])
        emb = np.asarray(self.embeddings[row], dtype=np.float32)
        meta = np.asarray(self.metadata_features[row], dtype=np.float32)
        x = np.concatenate([emb, meta], axis=0).astype(np.float32)

        item: Dict[str, torch.Tensor] = {
            "x": torch.from_numpy(x),
            "row_index": torch.tensor(row, dtype=torch.long),
        }
        if self.require_targets:
            item["target_score"] = torch.tensor(self.score_values[row], dtype=torch.float32)
            item["target_class"] = torch.tensor(self.class_values[row], dtype=torch.long)
            item["target_magnitude"] = torch.tensor(self.magnitude_values[row], dtype=torch.float32)
        return item


class SentimentAnalystMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        metadata_dim: int = 0,
        hidden_dims: Sequence[int] = (128, 64),
        representation_dim: int = 64,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.metadata_dim = int(metadata_dim)
        self.total_input_dim = self.input_dim + self.metadata_dim
        self.hidden_dims = [int(x) for x in hidden_dims]
        self.representation_dim = int(representation_dim)
        self.dropout = float(dropout)

        layers: List[nn.Module] = []
        prev = self.total_input_dim
        for hidden in self.hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(self.dropout))
            prev = hidden

        layers.append(nn.Linear(prev, self.representation_dim))
        layers.append(nn.LayerNorm(self.representation_dim))
        layers.append(nn.Tanh())
        layers.append(nn.Dropout(self.dropout))
        self.trunk = nn.Sequential(*layers)

        self.polarity_head = nn.Linear(self.representation_dim, 1)
        self.class_head = nn.Linear(self.representation_dim, 3)
        self.magnitude_head = nn.Linear(self.representation_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        rep = self.trunk(x)
        polarity = torch.tanh(self.polarity_head(rep)).squeeze(-1)
        class_logits = self.class_head(rep)
        # magnitude = self.magnitude_head(rep).squeeze(-1)
        magnitude = self.magnitude_head(rep).squeeze(-1)

        return {
            "sentiment_embedding": rep,
            "sentiment_score": polarity,
            "class_logits": class_logits,
            "magnitude": magnitude,
        }

    def model_config(self) -> Dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "metadata_dim": self.metadata_dim,
            "hidden_dims": self.hidden_dims,
            "representation_dim": self.representation_dim,
            "dropout": self.dropout,
            "activation": "tanh",
            "score_output_activation": "tanh",
            "magnitude_output_activation": "sigmoid",
        }


@dataclass
class SplitBundle:
    split: str
    labels: pd.DataFrame
    embeddings: np.ndarray
    metadata_features: np.ndarray
    valid_indices: np.ndarray


def prepare_split_bundle(
    cfg: SentimentConfig,
    split: str,
    preprocessor: MetadataPreprocessor,
    max_rows: Optional[int] = None,
    require_targets: bool = True,
) -> SplitBundle:
    labels = read_label_file(cfg, split)
    embeddings = load_embeddings(cfg, split, mmap_mode="r")
    validate_alignment(cfg, split, labels, embeddings)
    metadata_features = preprocessor.transform(labels)
    if require_targets:
        indices = valid_supervised_rows(labels)
    else:
        indices = np.arange(len(labels), dtype=np.int64)
    indices = deterministic_row_subset(indices, max_rows=max_rows, seed=cfg.seed + {"train": 11, "val": 17, "test": 23}[split])
    if require_targets and len(indices) == 0:
        raise ValueError(f"No valid supervised rows found for chunk{cfg.chunk_id}_{split}")
    return SplitBundle(split=split, labels=labels, embeddings=embeddings, metadata_features=metadata_features, valid_indices=indices)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, cfg: SentimentConfig, device: torch.device) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


def compute_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], cfg: SentimentConfig) -> Tuple[torch.Tensor, Dict[str, float]]:
    target_score = batch["target_score"]
    target_class = batch["target_class"]
    target_magnitude = batch["target_magnitude"]

    polarity_loss = F.mse_loss(outputs["sentiment_score"], target_score)
    class_loss = F.cross_entropy(outputs["class_logits"], target_class)
    # magnitude_loss = F.binary_cross_entropy_with_logits(outputs["magnitude"], target_magnitude)
    magnitude_loss = F.binary_cross_entropy_with_logits(outputs["magnitude"], target_magnitude)
    
    loss = (
        cfg.polarity_loss_weight * polarity_loss
        + cfg.class_loss_weight * class_loss
        + cfg.magnitude_loss_weight * magnitude_loss
    )
    parts = {
        "loss": float(loss.detach().cpu().item()),
        "polarity_loss": float(polarity_loss.detach().cpu().item()),
        "class_loss": float(class_loss.detach().cpu().item()),
        "magnitude_loss": float(magnitude_loss.detach().cpu().item()),
    }
    return loss, parts


def classification_accuracy(logits: np.ndarray, target_class: np.ndarray) -> float:
    if len(target_class) == 0:
        return float("nan")
    pred = np.argmax(logits, axis=1)
    return float(np.mean(pred == target_class))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    if np.nanstd(a) < 1e-12 or np.nanstd(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    err = pred - target
    return {
        "mse": float(np.mean(err * err)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "corr": safe_corr(pred, target),
    }


def confusion_matrix_3(pred_idx: np.ndarray, target_idx: np.ndarray) -> List[List[int]]:
    mat = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(target_idx, pred_idx):
        if 0 <= int(t) <= 2 and 0 <= int(p) <= 2:
            mat[int(t), int(p)] += 1
    return mat.tolist()


def run_epoch(
    model: SentimentAnalystMLP,
    loader: DataLoader,
    optimiser: Optional[torch.optim.Optimizer],
    scaler: Optional[GradScaler],
    cfg: SentimentConfig,
    device: torch.device,
    train: bool,
) -> Dict[str, float]:
    model.train(mode=train)
    totals = {"loss": 0.0, "polarity_loss": 0.0, "class_loss": 0.0, "magnitude_loss": 0.0}
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        if train:
            assert optimiser is not None
            optimiser.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with autocast(enabled=(cfg.mixed_precision and device.type == "cuda")):
                outputs = model(batch["x"])
                loss, parts = compute_loss(outputs, batch, cfg)

            if train:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                clip_grad_norm_(model.parameters(), cfg.gradient_clip)
                scaler.step(optimiser)
                scaler.update()

        for key in totals:
            totals[key] += parts[key]
        n_batches += 1

    return {key: value / max(1, n_batches) for key, value in totals.items()}


@torch.no_grad()
def evaluate_supervised(
    model: SentimentAnalystMLP,
    loader: DataLoader,
    cfg: SentimentConfig,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    all_scores: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_logits: List[np.ndarray] = []
    all_classes: List[np.ndarray] = []
    all_magnitude: List[np.ndarray] = []
    all_target_magnitude: List[np.ndarray] = []
    losses = {"loss": 0.0, "polarity_loss": 0.0, "class_loss": 0.0, "magnitude_loss": 0.0}
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with autocast(enabled=(cfg.mixed_precision and device.type == "cuda")):
            outputs = model(batch["x"])
            _, parts = compute_loss(outputs, batch, cfg)

        for key in losses:
            losses[key] += parts[key]
        n_batches += 1

        all_scores.append(outputs["sentiment_score"].float().cpu().numpy())
        all_targets.append(batch["target_score"].float().cpu().numpy())
        all_logits.append(outputs["class_logits"].float().cpu().numpy())
        all_classes.append(batch["target_class"].long().cpu().numpy())
        all_magnitude.append(outputs["magnitude"].float().cpu().numpy())
        all_target_magnitude.append(batch["target_magnitude"].float().cpu().numpy())

    pred_score = np.concatenate(all_scores) if all_scores else np.array([], dtype=np.float32)
    target_score = np.concatenate(all_targets) if all_targets else np.array([], dtype=np.float32)
    logits = np.concatenate(all_logits) if all_logits else np.empty((0, 3), dtype=np.float32)
    target_class = np.concatenate(all_classes) if all_classes else np.array([], dtype=np.int64)
    pred_magnitude = np.concatenate(all_magnitude) if all_magnitude else np.array([], dtype=np.float32)
    target_magnitude = np.concatenate(all_target_magnitude) if all_target_magnitude else np.array([], dtype=np.float32)

    pred_class_idx = np.argmax(logits, axis=1) if len(logits) else np.array([], dtype=np.int64)
    out: Dict[str, Any] = {key: value / max(1, n_batches) for key, value in losses.items()}
    out.update({f"polarity_{k}": v for k, v in regression_metrics(pred_score, target_score).items()})
    out.update({f"magnitude_{k}": v for k, v in regression_metrics(pred_magnitude, target_magnitude).items()})
    out["class_accuracy"] = classification_accuracy(logits, target_class)
    out["confusion_matrix_target_rows_pred_cols"] = confusion_matrix_3(pred_class_idx, target_class)
    out["rows_evaluated"] = int(len(target_score))
    return out


def save_checkpoint(
    cfg: SentimentConfig,
    model: SentimentAnalystMLP,
    optimiser: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    history: List[Dict[str, Any]],
    is_best: bool,
) -> None:
    ensure_dir(cfg.chunk_model_dir())
    state = {
        "saved_at": now_stamp(),
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "model_state": model.state_dict(),
        "optimiser_state": optimiser.state_dict(),
        "model_config": model.model_config(),
        "config": cfg.serialisable(),
        "history": history,
        "target_schema": {
            "score": TARGET_SCORE_COLUMN,
            "class": TARGET_CLASS_COLUMN,
            "class_mapping": CLASS_TO_INDEX,
        },
    }
    torch.save(state, cfg.chunk_model_dir() / f"epoch_{epoch:03d}.pt")
    torch.save(state, cfg.latest_checkpoint_path())
    if is_best:
        torch.save(state, cfg.best_checkpoint_path())


def load_checkpoint_for_training(
    cfg: SentimentConfig,
    model: SentimentAnalystMLP,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[int, float, List[Dict[str, Any]]]:
    path = cfg.latest_checkpoint_path()
    if not path.exists():
        return 1, float("inf"), []
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state"])
    optimiser.load_state_dict(state["optimiser_state"])
    start_epoch = int(state.get("epoch", 0)) + 1
    best_val_loss = float(state.get("best_val_loss", float("inf")))
    history = list(state.get("history", []))
    print(f"[resume] loaded {path}, start_epoch={start_epoch}, best_val_loss={best_val_loss:.6f}")
    return start_epoch, best_val_loss, history


def load_model_for_inference(cfg: SentimentConfig, checkpoint: str, device: torch.device) -> SentimentAnalystMLP:
    if checkpoint == "best":
        path = cfg.best_checkpoint_path()
    elif checkpoint == "latest":
        path = cfg.latest_checkpoint_path()
    else:
        path = Path(checkpoint)
        if not path.is_absolute():
            path = cfg.repo_root / path
    if not path.exists():
        raise FileNotFoundError(f"Missing sentiment analyst checkpoint: {path}")
    state = torch.load(path, map_location=device)
    model_cfg = state["model_config"]
    model = SentimentAnalystMLP(
        input_dim=int(model_cfg["input_dim"]),
        metadata_dim=int(model_cfg["metadata_dim"]),
        hidden_dims=list(model_cfg["hidden_dims"]),
        representation_dim=int(model_cfg["representation_dim"]),
        dropout=float(model_cfg["dropout"]),
    )
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()
    # Update config to match actual model dims (HPO may have changed them)
    cfg.representation_dim = model.representation_dim
    cfg.hidden_dims = model.hidden_dims
    return model


def train_one_chunk(cfg: SentimentConfig) -> Dict[str, Any]:
    cfg.resolve()
    if cfg.chunk_id not in APPROVED_SPLITS:
        raise ValueError(f"Unsupported chunk_id={cfg.chunk_id}")

    ensure_dir(cfg.chunk_model_dir())
    ensure_dir(cfg.results_dir)
    ensure_dir(cfg.code_results_dir)
    ensure_dir(cfg.analyst_embeddings_dir)
    write_json(cfg.code_results_dir / f"chunk{cfg.chunk_id}_run_config.json", cfg.serialisable())

    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print("========== SENTIMENT ANALYST TRAINING ==========")
    print(json.dumps(cfg.serialisable(), indent=2))
    print(f"[device] requested={cfg.device} resolved={device} cuda_available={torch.cuda.is_available()}")

    train_labels = read_label_file(cfg, "train")
    train_embeddings = load_embeddings(cfg, "train", mmap_mode="r")
    validate_alignment(cfg, "train", train_labels, train_embeddings)
    train_valid = valid_supervised_rows(train_labels)
    train_valid = deterministic_row_subset(train_valid, cfg.max_train_rows, cfg.seed + 101)
    if len(train_valid) == 0:
        raise ValueError(f"No valid supervised training rows for chunk{cfg.chunk_id}")

    if cfg.resume and cfg.preprocessor_path().exists():
        preprocessor = MetadataPreprocessor.load(cfg.preprocessor_path())
    else:
        preprocessor = MetadataPreprocessor(use_metadata_features=cfg.use_metadata_features).fit(train_labels, train_valid)
        preprocessor.save(cfg.preprocessor_path())

    train_meta = preprocessor.transform(train_labels)
    train_bundle = SplitBundle("train", train_labels, train_embeddings, train_meta, train_valid)
    val_bundle = prepare_split_bundle(cfg, "val", preprocessor, max_rows=cfg.max_val_rows, require_targets=True)

    train_dataset = LabelledEmbeddingDataset(train_bundle.embeddings, train_bundle.labels, train_bundle.metadata_features, train_bundle.valid_indices, require_targets=True)
    val_dataset = LabelledEmbeddingDataset(val_bundle.embeddings, val_bundle.labels, val_bundle.metadata_features, val_bundle.valid_indices, require_targets=True)

    train_loader = make_loader(train_dataset, cfg.batch_size, shuffle=True, cfg=cfg, device=device)
    val_loader = make_loader(val_dataset, cfg.eval_batch_size, shuffle=False, cfg=cfg, device=device)

    model = SentimentAnalystMLP(
        input_dim=cfg.input_dim,
        metadata_dim=preprocessor.feature_dim,
        hidden_dims=cfg.hidden_dims,
        representation_dim=cfg.representation_dim,
        dropout=cfg.dropout,
    ).to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=(cfg.mixed_precision and device.type == "cuda"))

    start_epoch = 1
    best_val_loss = float("inf")
    history: List[Dict[str, Any]] = []
    if cfg.resume:
        start_epoch, best_val_loss, history = load_checkpoint_for_training(cfg, model, optimiser, device)

    no_improve = 0
    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, optimiser, scaler, cfg, device, train=True)
        val_metrics = evaluate_supervised(model, val_loader, cfg, device)
        val_loss = float(val_metrics["loss"])
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1

        row = {
            "epoch": epoch,
            "created_at": now_stamp(),
            "seconds": round(time.time() - t0, 3),
            "train_rows": int(len(train_dataset)),
            "val_rows": int(len(val_dataset)),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items() if not isinstance(v, list)},
            "best_val_loss": best_val_loss,
            "improved": bool(improved),
            "no_improve": int(no_improve),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path(cfg), index=False)
        save_checkpoint(cfg, model, optimiser, epoch, best_val_loss, history, is_best=improved)
        print(f"[epoch {epoch}] train_loss={train_metrics['loss']:.6f} val_loss={val_loss:.6f} val_mae={val_metrics['polarity_mae']:.6f} val_acc={val_metrics['class_accuracy']:.4f} improved={improved}")

        if no_improve >= cfg.early_stop_patience:
            print(f"[early-stop] no validation improvement for {no_improve} epochs")
            break

    summary = {
        "chunk_id": cfg.chunk_id,
        "best_val_loss": best_val_loss,
        "history_file": str(history_path(cfg)),
        "best_checkpoint": str(cfg.best_checkpoint_path()),
        "latest_checkpoint": str(cfg.latest_checkpoint_path()),
        "metadata_preprocessor": str(cfg.preprocessor_path()),
    }
    write_json(cfg.results_dir / f"chunk{cfg.chunk_id}_training_summary.json", summary)
    return summary


@torch.no_grad()
def run_prediction_export(cfg: SentimentConfig, split: str, checkpoint: str = "best", max_rows: Optional[int] = None) -> Dict[str, Any]:
    cfg.resolve()
    device = get_device(cfg.device)
    preprocessor = MetadataPreprocessor.load(cfg.preprocessor_path())
    labels = read_label_file(cfg, split)
    embeddings = load_embeddings(cfg, split, mmap_mode="r")
    validate_alignment(cfg, split, labels, embeddings)
    meta = preprocessor.transform(labels)
    row_indices = np.arange(len(labels), dtype=np.int64)
    if max_rows is not None and max_rows > 0:
        row_indices = deterministic_row_subset(row_indices, max_rows, cfg.seed + 301)

    dataset = LabelledEmbeddingDataset(embeddings, labels, meta, row_indices, require_targets=False)
    loader = make_loader(dataset, cfg.eval_batch_size, shuffle=False, cfg=cfg, device=device)
    model = load_model_for_inference(cfg, checkpoint=checkpoint, device=device)

    ensure_dir(cfg.results_dir)
    ensure_dir(cfg.analyst_embeddings_dir)

    n_rows = len(labels) if max_rows is None else len(row_indices)
    row_to_output_pos = {int(row): i for i, row in enumerate(row_indices.tolist())}
    # Use model's actual representation_dim (HPO may have changed it)
    actual_repr_dim = model.representation_dim
    repr_out = np.lib.format.open_memmap(
        analyst_embedding_path(cfg, split),
        mode="w+",
        dtype=np.float32,
        shape=(n_rows, actual_repr_dim),
    )

    pred_score = np.full(n_rows, np.nan, dtype=np.float32)
    pred_class_idx = np.full(n_rows, -1, dtype=np.int64)
    pred_conf = np.full(n_rows, np.nan, dtype=np.float32)
    pred_mag = np.full(n_rows, np.nan, dtype=np.float32)

    model.eval()
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        rows = batch["row_index"].cpu().numpy().astype(np.int64)
        with autocast(enabled=(cfg.mixed_precision and device.type == "cuda")):
            outputs = model(x)
            probs = torch.softmax(outputs["class_logits"], dim=1)
        scores = outputs["sentiment_score"].float().cpu().numpy()
        classes = torch.argmax(probs, dim=1).long().cpu().numpy()
        conf = torch.max(probs, dim=1).values.float().cpu().numpy()
        mag = torch.sigmoid(outputs["magnitude"]).float().cpu().numpy()
        rep = outputs["sentiment_embedding"].float().cpu().numpy().astype(np.float32)

        for j, row in enumerate(rows):
            pos = row_to_output_pos[int(row)]
            pred_score[pos] = scores[j]
            pred_class_idx[pos] = classes[j]
            pred_conf[pos] = conf[j]
            pred_mag[pos] = mag[j]
            repr_out[pos, :] = rep[j]

    del repr_out


    # ══════════════════════════════════════════════════════════
    # XAI: Gradient-based feature importance (default, fast)
    # or full SHAP (opt-in with --xai-method shap, slow)
    # ══════════════════════════════════════════════════════════
    xai_method = getattr(cfg, 'xai_method', 'gradient')
    xai_rows_sample = getattr(cfg, 'xai_sample_size', 1000)
    
    xai_feature_names = [f"finbert_dim_{i}" for i in range(cfg.input_dim)]
    if preprocessor.feature_dim > 0:
        xai_feature_names += [f"meta_{i}" for i in range(preprocessor.feature_dim)]
    
    xai_importance = np.full((n_rows, len(xai_feature_names)), np.nan, dtype=np.float32)
    xai_explanations: List[Dict] = []
    
    if xai_method == 'shap':
        # ── Option A: Full SHAP (slow but complete) ──────
        try:
            import shap
            print(f"[xai] Computing SHAP on {xai_rows_sample} sample rows (this may take several minutes)...")
            
            # Get background data from a subset of rows
            bg_indices = row_indices[:min(100, len(row_indices))]
            background = []
            for idx in bg_indices:
                emb = np.asarray(embeddings[int(idx)], dtype=np.float32)
                mt = np.asarray(meta[int(idx)], dtype=np.float32)
                background.append(np.concatenate([emb, mt]))
            background = np.stack(background)
            
            # Get sample for explanation
            sample_indices = row_indices[:min(xai_rows_sample, len(row_indices))]
            sample_data = []
            for idx in sample_indices:
                emb = np.asarray(embeddings[int(idx)], dtype=np.float32)
                mt = np.asarray(meta[int(idx)], dtype=np.float32)
                sample_data.append(np.concatenate([emb, mt]))
            sample_data = np.stack(sample_data)
            
            # SHAP DeepExplainer
            explainer = shap.DeepExplainer(
                model, 
                torch.from_numpy(background).to(device)
            )
            shap_values = explainer.shap_values(
                torch.from_numpy(sample_data).to(device)
            )
            
            # shap_values shape: (n_samples, n_features) for regression
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # take first output head
            
            for i, sample_idx in enumerate(sample_indices):
                pos = row_to_output_pos[int(sample_idx)]
                xai_importance[pos, :] = shap_values[i].astype(np.float32)
                
                # Build explanation dict per XAI spec
                top_pos_idx = np.argsort(shap_values[i])[-5:][::-1]
                top_neg_idx = np.argsort(shap_values[i])[:5]
                
                xai_explanations.append({
                    "row_index": int(sample_idx),
                    "predicted_score": float(pred_score[pos]),
                    "top_positive_factors": [
                        {"factor": xai_feature_names[j], "weight": float(shap_values[i][j]), 
                         "direction": "positive"}
                        for j in top_pos_idx if shap_values[i][j] > 0
                    ],
                    "top_negative_factors": [
                        {"factor": xai_feature_names[j], "weight": float(abs(shap_values[i][j])), 
                         "direction": "negative"}
                        for j in top_neg_idx if shap_values[i][j] < 0
                    ],
                })
            
            print(f"[xai] SHAP complete — {len(xai_explanations)} explanations generated")
        except ImportError:
            print("[xai] shap not installed — falling back to gradient-based importance. pip install shap")
            xai_method = 'gradient'
        except Exception as e:
            print(f"[xai] SHAP failed: {e} — falling back to gradient-based importance")
            xai_method = 'gradient'
    
    if xai_method == 'gradient':
        # ── Option C: Gradient-based importance (fast, default) ──
        print(f"[xai] Computing gradient-based feature importance on {xai_rows_sample} sample rows...")
        
        sample_indices = row_indices[:min(xai_rows_sample, len(row_indices))]
        model.eval()
        
        for idx in sample_indices:
            emb = np.asarray(embeddings[int(idx)], dtype=np.float32)
            mt = np.asarray(meta[int(idx)], dtype=np.float32)
            x = np.concatenate([emb, mt])
            x_tensor = torch.from_numpy(x).unsqueeze(0).to(device)
            x_tensor.requires_grad_(True)
            
            with torch.enable_grad():
                outputs = model(x_tensor)
                score = outputs["sentiment_score"]
                score.backward()
            
            grads = x_tensor.grad.cpu().numpy().flatten()
            importance = np.abs(grads).astype(np.float32)
            
            pos = row_to_output_pos[int(idx)]
            xai_importance[pos, :] = importance
            
            top_idx = np.argsort(importance)[-5:][::-1]
            xai_explanations.append({
                "row_index": int(idx),
                "predicted_score": float(pred_score[pos]),
                "top_positive_factors": [
                    {"factor": xai_feature_names[j], "weight": float(importance[j]),
                     "direction": "positive"}
                    for j in top_idx if grads[j] > 0
                ],
                "top_negative_factors": [
                    {"factor": xai_feature_names[j], "weight": float(importance[j]),
                     "direction": "negative"}
                    for j in top_idx if grads[j] < 0
                ],
            })
            
            x_tensor.grad = None
        
        print(f"[xai] Gradient importance complete — {len(xai_explanations)} explanations generated")
    
    # Save XAI outputs
    xai_dir = cfg.results_dir / "xai"
    ensure_dir(xai_dir)
    
    xai_importance_path = xai_dir / f"chunk{cfg.chunk_id}_{split}_feature_importance.npy"
    np.save(str(xai_importance_path), xai_importance)
    
    xai_explanations_path = xai_dir / f"chunk{cfg.chunk_id}_{split}_explanations.json"
    with open(xai_explanations_path, "w") as f:
        json.dump({
            "xai_method": xai_method,
            "n_samples": len(xai_explanations),
            "feature_names": xai_feature_names,
            "explanations": xai_explanations,
        }, f, indent=2, default=str)
    
    print(f"[xai] Saved: {xai_importance_path}")
    print(f"[xai] Saved: {xai_explanations_path}")


    if max_rows is None:
        out_df = labels.copy()
    else:
        out_df = labels.iloc[row_indices].copy().reset_index(drop=True)
    out_df["predicted_sentiment_score"] = pred_score
    out_df["predicted_sentiment_class_index"] = pred_class_idx
    out_df["predicted_sentiment_class"] = [INDEX_TO_CLASS.get(int(x), np.nan) for x in pred_class_idx]
    out_df["predicted_sentiment_confidence"] = pred_conf
    out_df["predicted_sentiment_uncertainty"] = 1.0 - pred_conf
    out_df["predicted_sentiment_magnitude"] = pred_mag
    out_df["sentiment_embedding_file"] = str(analyst_embedding_path(cfg, split))
    out_df.to_csv(predictions_path(cfg, split), index=False)

    metrics: Dict[str, Any] = {
        "created_at": now_stamp(),
        "chunk_id": cfg.chunk_id,
        "split": split,
        "checkpoint": checkpoint,
        "rows_predicted": int(len(out_df)),
        "predictions_file": str(predictions_path(cfg, split)),
        "sentiment_embedding_file": str(analyst_embedding_path(cfg, split)),
    }

    eval_indices = valid_supervised_rows(out_df)
    if len(eval_indices) > 0:
        target_score = pd.to_numeric(out_df.iloc[eval_indices][TARGET_SCORE_COLUMN], errors="coerce").to_numpy(dtype=np.float32)
        target_class_raw = pd.to_numeric(out_df.iloc[eval_indices][TARGET_CLASS_COLUMN], errors="coerce").to_numpy(dtype=np.int64)
        target_class = np.array([CLASS_TO_INDEX[int(x)] for x in target_class_raw], dtype=np.int64)
        pred_score_eval = out_df.iloc[eval_indices]["predicted_sentiment_score"].to_numpy(dtype=np.float32)
        pred_class_eval = out_df.iloc[eval_indices]["predicted_sentiment_class_index"].to_numpy(dtype=np.int64)
        metrics.update({f"polarity_{k}": v for k, v in regression_metrics(pred_score_eval, target_score).items()})
        metrics["class_accuracy"] = float(np.mean(pred_class_eval == target_class))
        metrics["confusion_matrix_target_rows_pred_cols"] = confusion_matrix_3(pred_class_eval, target_class)
        metrics["rows_evaluated"] = int(len(eval_indices))
    else:
        metrics["rows_evaluated"] = 0

    write_json(metrics_path(cfg, split), metrics)
    print(f"[predict] split={split} rows={len(out_df):,} predictions={predictions_path(cfg, split)} embeddings={analyst_embedding_path(cfg, split)}")
    return metrics


def inspect_labels(cfg: SentimentConfig) -> Dict[str, Any]:
    cfg.resolve()
    rows: List[Dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        labels = read_label_file(cfg, split)
        emb = load_embeddings(cfg, split, mmap_mode="r")
        validate_alignment(cfg, split, labels, emb)
        valid = valid_supervised_rows(labels)
        rows.append(
            {
                "chunk_id": cfg.chunk_id,
                "split": split,
                "label_rows": int(len(labels)),
                "embedding_shape": list(emb.shape),
                "valid_supervised_rows": int(len(valid)),
                "label_available_rows": int(labels["label_available"].astype(str).str.lower().isin(["true", "1", "yes"]).sum()),
                "missing_score_rows": int(pd.to_numeric(labels[TARGET_SCORE_COLUMN], errors="coerce").isna().sum()),
                "missing_class_rows": int(pd.to_numeric(labels[TARGET_CLASS_COLUMN], errors="coerce").isna().sum()),
                "ticker_in_market_panel_rows": int(labels["ticker_in_market_panel"].astype(str).str.lower().isin(["true", "1", "yes"]).sum()),
            }
        )
    out = {"created_at": now_stamp(), "rows": rows}
    ensure_dir(cfg.results_dir)
    write_json(cfg.results_dir / f"chunk{cfg.chunk_id}_input_inspection.json", out)
    print(pd.DataFrame(rows).to_string(index=False))
    return out



def hpo_root(cfg: SentimentConfig, study_name: str) -> Path:
    return cfg.code_results_dir / "hpo" / study_name


def hpo_best_params_path(cfg: SentimentConfig, study_name: str) -> Path:
    return hpo_root(cfg, study_name) / f"{study_name}_best_params.json"


def hpo_trials_path(cfg: SentimentConfig, study_name: str) -> Path:
    return hpo_root(cfg, study_name) / f"{study_name}_trials.csv"


def hpo_storage_uri(cfg: SentimentConfig, study_name: str) -> str:
    root = hpo_root(cfg, study_name)
    ensure_dir(root)
    return f"sqlite:///{root / (study_name + '.db')}"


def suggest_hpo_config(base_cfg: SentimentConfig, trial: Any, args: argparse.Namespace, study_name: str) -> SentimentConfig:
    cfg = SentimentConfig(**asdict(base_cfg))
    cfg.resolve()
    cfg.seed = int(base_cfg.seed) + int(trial.number)
    cfg.resume = False
    cfg.learning_rate = float(trial.suggest_float("learning_rate", float(args.hpo_lr_min), float(args.hpo_lr_max), log=True))
    cfg.weight_decay = float(trial.suggest_float("weight_decay", float(args.hpo_weight_decay_min), float(args.hpo_weight_decay_max), log=True))
    cfg.dropout = float(trial.suggest_float("dropout", float(args.hpo_dropout_min), float(args.hpo_dropout_max)))
    cfg.batch_size = int(trial.suggest_categorical("batch_size", parse_int_list(args.hpo_batch_sizes)))
    cfg.eval_batch_size = int(max(cfg.batch_size, cfg.batch_size * int(args.hpo_eval_batch_multiplier)))
    hidden_options = [x.strip() for x in str(args.hpo_hidden_dims).split(";") if x.strip()]
    cfg.hidden_dims = parse_hidden_dims(str(trial.suggest_categorical("hidden_dims", hidden_options)))
    cfg.representation_dim = int(trial.suggest_categorical("representation_dim", parse_int_list(args.hpo_representation_dims)))
    cfg.epochs = int(trial.suggest_int("epochs", int(args.hpo_epochs_min), int(args.hpo_epochs_max)))
    cfg.early_stop_patience = int(trial.suggest_int("early_stop_patience", int(args.hpo_patience_min), int(args.hpo_patience_max)))
    cfg.gradient_clip = float(trial.suggest_float("gradient_clip", float(args.hpo_gradient_clip_min), float(args.hpo_gradient_clip_max)))
    cfg.class_loss_weight = float(trial.suggest_float("class_loss_weight", float(args.hpo_class_loss_weight_min), float(args.hpo_class_loss_weight_max)))
    cfg.magnitude_loss_weight = float(trial.suggest_float("magnitude_loss_weight", float(args.hpo_magnitude_loss_weight_min), float(args.hpo_magnitude_loss_weight_max)))
    cfg.polarity_loss_weight = 1.0
    cfg.use_metadata_features = bool(trial.suggest_categorical("use_metadata_features", [True, False])) if bool(args.hpo_search_metadata_features) else bool(base_cfg.use_metadata_features)
    cfg.max_train_rows = int(args.hpo_max_train_rows) if args.hpo_max_train_rows is not None else base_cfg.max_train_rows
    cfg.max_val_rows = int(args.hpo_max_val_rows) if args.hpo_max_val_rows is not None else base_cfg.max_val_rows
    cfg.models_dir = base_cfg.models_dir / "hpo" / study_name / f"trial_{trial.number:04d}"
    cfg.results_dir = base_cfg.results_dir / "hpo" / study_name / f"trial_{trial.number:04d}"
    cfg.code_results_dir = base_cfg.code_results_dir / "hpo" / study_name / f"trial_{trial.number:04d}"
    cfg.analyst_embeddings_dir = base_cfg.analyst_embeddings_dir / "hpo" / study_name / f"trial_{trial.number:04d}"
    return cfg.resolve()


def metric_from_summary(summary: Dict[str, Any], metric_name: str = "best_val_loss") -> float:
    value = summary.get(metric_name)
    if value is None:
        raise ValueError(f"Metric {metric_name} not found in training summary")
    value = float(value)
    if not np.isfinite(value):
        return float("inf")
    return value


def run_hpo_for_chunk(base_cfg: SentimentConfig, args: argparse.Namespace) -> Dict[str, Any]:
    if optuna is None:
        raise ImportError("Optuna is required for hyperparameter search. Install it with: pip install optuna")

    base_cfg.resolve()
    study_name = str(args.study_name or f"sentiment_analyst_chunk{base_cfg.chunk_id}_hpo")
    ensure_dir(hpo_root(base_cfg, study_name))

    sampler = optuna.samplers.TPESampler(seed=int(base_cfg.seed), multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=int(args.hpo_pruner_startup_trials), n_warmup_steps=int(args.hpo_pruner_warmup_steps))
    study = optuna.create_study(direction="minimize", study_name=study_name, storage=hpo_storage_uri(base_cfg, study_name), sampler=sampler, pruner=pruner, load_if_exists=True)

    print("========== SENTIMENT ANALYST HPO ==========")
    print(json.dumps({"chunk_id": base_cfg.chunk_id, "study_name": study_name, "storage": hpo_storage_uri(base_cfg, study_name), "trials_requested": int(args.trials)}, indent=2))

    def objective(trial: Any) -> float:
        trial_cfg = suggest_hpo_config(base_cfg, trial, args, study_name)
        write_json(trial_cfg.code_results_dir / "trial_config.json", trial_cfg.serialisable())
        try:
            summary = train_one_chunk(trial_cfg)
            objective_value = metric_from_summary(summary, "best_val_loss")
            trial.set_user_attr("best_checkpoint", str(summary.get("best_checkpoint")))
            trial.set_user_attr("history_file", str(summary.get("history_file")))
            trial.set_user_attr("trial_config", trial_cfg.serialisable())
            return objective_value
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise optuna.TrialPruned(f"OOM pruned: {exc}")
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    study.optimize(objective, n_trials=int(args.trials), timeout=args.timeout, gc_after_trial=True)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(hpo_trials_path(base_cfg, study_name), index=False)

    best_params: Dict[str, Any] = {}
    best_value: Optional[float] = None
    best_trial_number: Optional[int] = None
    try:
        best_params = dict(study.best_params)
        best_value = float(study.best_value)
        best_trial_number = int(study.best_trial.number)
    except Exception:
        pass

    best_record = {
        "created_at": now_stamp(),
        "chunk_id": int(base_cfg.chunk_id),
        "study_name": study_name,
        "storage": hpo_storage_uri(base_cfg, study_name),
        "trials_completed": int(len(study.trials)),
        "best_trial_number": best_trial_number,
        "best_value": best_value,
        "objective": "minimise validation composite loss",
        "best_params": best_params,
        "trials_file": str(hpo_trials_path(base_cfg, study_name)),
    }
    write_json(hpo_best_params_path(base_cfg, study_name), best_record)
    print(json.dumps(best_record, indent=2, default=str))
    return best_record


def apply_best_params(cfg: SentimentConfig, best_params: Dict[str, Any]) -> SentimentConfig:
    if not best_params:
        raise ValueError("No best_params were found. Run HPO first or provide a valid HPO JSON file.")
    cfg.learning_rate = float(best_params.get("learning_rate", cfg.learning_rate))
    cfg.weight_decay = float(best_params.get("weight_decay", cfg.weight_decay))
    cfg.dropout = float(best_params.get("dropout", cfg.dropout))
    cfg.batch_size = int(best_params.get("batch_size", cfg.batch_size))
    cfg.eval_batch_size = int(max(cfg.batch_size, cfg.batch_size * 2))
    if "hidden_dims" in best_params:
        cfg.hidden_dims = parse_hidden_dims(str(best_params["hidden_dims"]))
    cfg.representation_dim = int(best_params.get("representation_dim", cfg.representation_dim))
    cfg.epochs = int(best_params.get("epochs", cfg.epochs))
    cfg.early_stop_patience = int(best_params.get("early_stop_patience", cfg.early_stop_patience))
    cfg.gradient_clip = float(best_params.get("gradient_clip", cfg.gradient_clip))
    cfg.class_loss_weight = float(best_params.get("class_loss_weight", cfg.class_loss_weight))
    cfg.magnitude_loss_weight = float(best_params.get("magnitude_loss_weight", cfg.magnitude_loss_weight))
    cfg.polarity_loss_weight = float(best_params.get("polarity_loss_weight", cfg.polarity_loss_weight))
    if "use_metadata_features" in best_params:
        cfg.use_metadata_features = bool(best_params["use_metadata_features"])
    return cfg


def load_hpo_best_record(cfg: SentimentConfig, study_name: Optional[str]) -> Dict[str, Any]:
    cfg.resolve()
    resolved_study_name = str(study_name or f"sentiment_analyst_chunk{cfg.chunk_id}_hpo")
    path = hpo_best_params_path(cfg, resolved_study_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing HPO best-params file: {path}. Run the hpo command first.")
    return json.loads(path.read_text(encoding="utf-8"))


def train_best_for_chunk(cfg: SentimentConfig, study_name: Optional[str], final_epochs: Optional[int] = None) -> Dict[str, Any]:
    cfg.resolve()
    best_record = load_hpo_best_record(cfg, study_name)
    cfg = apply_best_params(cfg, dict(best_record.get("best_params", {})))
    if final_epochs is not None and int(final_epochs) > 0:
        cfg.epochs = int(final_epochs)
    cfg.resume = False
    ensure_dir(cfg.code_results_dir)
    write_json(cfg.code_results_dir / f"chunk{cfg.chunk_id}_best_hpo_params_used.json", best_record)
    print("========== TRAINING WITH HPO BEST PARAMS ==========")
    print(json.dumps({"chunk_id": cfg.chunk_id, "study_name": best_record.get("study_name"), "best_value": best_record.get("best_value"), "best_params": best_record.get("best_params"), "final_epochs": cfg.epochs}, indent=2, default=str))
    return train_one_chunk(cfg)

def build_cfg_from_args(args: argparse.Namespace, chunk_id: Optional[int] = None) -> SentimentConfig:
    cfg = SentimentConfig()
    cfg.repo_root = Path(args.repo_root)
    cfg.env_file = Path(args.env_file)
    cfg.resolve()

    if chunk_id is not None:
        cfg.chunk_id = int(chunk_id)
    elif hasattr(args, "chunk") and args.chunk is not None:
        cfg.chunk_id = int(args.chunk)

    if args.embeddings_dir is not None:
        cfg.embeddings_dir = resolve_repo_path(cfg.repo_root, Path(args.embeddings_dir))
    if args.labels_dir is not None:
        cfg.labels_dir = resolve_repo_path(cfg.repo_root, Path(args.labels_dir))
    if args.models_dir is not None:
        cfg.models_dir = resolve_repo_path(cfg.repo_root, Path(args.models_dir))
    if args.results_dir is not None:
        cfg.results_dir = resolve_repo_path(cfg.repo_root, Path(args.results_dir))
    if args.code_results_dir is not None:
        cfg.code_results_dir = resolve_repo_path(cfg.repo_root, Path(args.code_results_dir))
    if args.analyst_embeddings_dir is not None:
        cfg.analyst_embeddings_dir = resolve_repo_path(cfg.repo_root, Path(args.analyst_embeddings_dir))

    cfg.device = args.device
    cfg.seed = int(args.seed)
    cfg.num_workers = int(args.num_workers)
    cfg.batch_size = int(args.batch_size)
    cfg.eval_batch_size = int(args.eval_batch_size)
    cfg.epochs = int(args.epochs)
    cfg.learning_rate = float(args.learning_rate)
    cfg.weight_decay = float(args.weight_decay)
    cfg.dropout = float(args.dropout)
    cfg.gradient_clip = float(args.gradient_clip)
    cfg.early_stop_patience = int(args.early_stop_patience)
    if hasattr(args, "hidden_dims") and args.hidden_dims is not None:
        cfg.hidden_dims = parse_hidden_dims(args.hidden_dims)
    if hasattr(args, "representation_dim") and args.representation_dim is not None:
        cfg.representation_dim = int(args.representation_dim)
    if hasattr(args, "polarity_loss_weight") and args.polarity_loss_weight is not None:
        cfg.polarity_loss_weight = float(args.polarity_loss_weight)
    if hasattr(args, "class_loss_weight") and args.class_loss_weight is not None:
        cfg.class_loss_weight = float(args.class_loss_weight)
    if hasattr(args, "magnitude_loss_weight") and args.magnitude_loss_weight is not None:
        cfg.magnitude_loss_weight = float(args.magnitude_loss_weight)
    cfg.mixed_precision = not bool(args.no_mixed_precision)
    cfg.use_metadata_features = not bool(args.no_metadata_features)
    cfg.resume = bool(args.resume)
    cfg.max_train_rows = args.max_train_rows
    cfg.max_val_rows = args.max_val_rows
    cfg.max_test_rows = args.max_test_rows
    return cfg.resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and run the supervised fin-glassbox Sentiment Analyst on real FinBERT embeddings and real market labels.")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--repo-root", default=".")
        sp.add_argument("--env-file", default=".env")
        sp.add_argument("--embeddings-dir", default=None)
        sp.add_argument("--labels-dir", default=None)
        sp.add_argument("--models-dir", default=None)
        sp.add_argument("--results-dir", default=None)
        sp.add_argument("--code-results-dir", default=None)
        sp.add_argument("--analyst-embeddings-dir", default=None)
        sp.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--num-workers", type=int, default=0)
        sp.add_argument("--batch-size", type=int, default=512)
        sp.add_argument("--eval-batch-size", type=int, default=1024)
        sp.add_argument("--epochs", type=int, default=25)
        sp.add_argument("--learning-rate", type=float, default=1e-4)
        sp.add_argument("--weight-decay", type=float, default=1e-4)
        sp.add_argument("--dropout", type=float, default=0.15)
        sp.add_argument("--hidden-dims", default="128,64")
        sp.add_argument("--representation-dim", type=int, default=64)
        sp.add_argument("--polarity-loss-weight", type=float, default=1.0)
        sp.add_argument("--class-loss-weight", type=float, default=0.5)
        sp.add_argument("--magnitude-loss-weight", type=float, default=0.25)
        sp.add_argument("--gradient-clip", type=float, default=1.0)
        sp.add_argument("--early-stop-patience", type=int, default=8)
        sp.add_argument("--no-mixed-precision", action="store_true")
        sp.add_argument("--no-metadata-features", action="store_true")
        sp.add_argument("--resume", action="store_true")
        sp.add_argument("--max-train-rows", type=int, default=None)
        sp.add_argument("--max-val-rows", type=int, default=None)
        sp.add_argument("--max-test-rows", type=int, default=None)    
        sp.add_argument("--xai-method", default="gradient", choices=["gradient", "shap"],
                        help="XAI method: gradient (fast) or shap (slow but complete)")
        sp.add_argument("--xai-sample-size", type=int, default=1000,
                        help="Number of rows to explain (more = slower)")

    def add_hpo(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--trials", type=int, default=30)
        sp.add_argument("--timeout", type=int, default=None)
        sp.add_argument("--study-name", default=None)
        sp.add_argument("--hpo-max-train-rows", type=int, default=None)
        sp.add_argument("--hpo-max-val-rows", type=int, default=None)
        sp.add_argument("--hpo-lr-min", type=float, default=1e-5)
        sp.add_argument("--hpo-lr-max", type=float, default=8e-4)
        sp.add_argument("--hpo-weight-decay-min", type=float, default=1e-7)
        sp.add_argument("--hpo-weight-decay-max", type=float, default=5e-3)
        sp.add_argument("--hpo-dropout-min", type=float, default=0.05)
        sp.add_argument("--hpo-dropout-max", type=float, default=0.35)
        sp.add_argument("--hpo-batch-sizes", default="256,512,1024")
        sp.add_argument("--hpo-eval-batch-multiplier", type=int, default=2)
        sp.add_argument("--hpo-hidden-dims", default="128,64;256,128;256,128,64;512,256,128;128,128,64")
        sp.add_argument("--hpo-representation-dims", default="32,64,128")
        sp.add_argument("--hpo-epochs-min", type=int, default=5)
        sp.add_argument("--hpo-epochs-max", type=int, default=25)
        sp.add_argument("--hpo-patience-min", type=int, default=3)
        sp.add_argument("--hpo-patience-max", type=int, default=8)
        sp.add_argument("--hpo-gradient-clip-min", type=float, default=0.5)
        sp.add_argument("--hpo-gradient-clip-max", type=float, default=2.0)
        sp.add_argument("--hpo-class-loss-weight-min", type=float, default=0.2)
        sp.add_argument("--hpo-class-loss-weight-max", type=float, default=1.5)
        sp.add_argument("--hpo-magnitude-loss-weight-min", type=float, default=0.05)
        sp.add_argument("--hpo-magnitude-loss-weight-max", type=float, default=0.75)
        sp.add_argument("--hpo-search-metadata-features", action="store_true")
        sp.add_argument("--hpo-pruner-startup-trials", type=int, default=4)
        sp.add_argument("--hpo-pruner-warmup-steps", type=int, default=2)

    sp_train = sub.add_parser("train", help="Train one chronological chunk with explicitly supplied parameters")
    add_common(sp_train)
    sp_train.add_argument("--chunk", type=int, choices=[1, 2, 3], required=True)

    sp_train_all = sub.add_parser("train-all", help="Train separate Sentiment Analysts for multiple chunks with explicitly supplied parameters")
    add_common(sp_train_all)
    sp_train_all.add_argument("--chunks", default="1,2,3")

    sp_hpo = sub.add_parser("hpo", help="Run Optuna/TPE hyperparameter search for one chronological chunk")
    add_common(sp_hpo)
    add_hpo(sp_hpo)
    sp_hpo.add_argument("--chunk", type=int, choices=[1, 2, 3], required=True)

    sp_hpo_all = sub.add_parser("hpo-all", help="Run Optuna/TPE hyperparameter search independently for multiple chunks")
    add_common(sp_hpo_all)
    add_hpo(sp_hpo_all)
    sp_hpo_all.add_argument("--chunks", default="1,2,3")

    sp_train_best = sub.add_parser("train-best", help="Train one chronological chunk using saved HPO best parameters")
    add_common(sp_train_best)
    sp_train_best.add_argument("--chunk", type=int, choices=[1, 2, 3], required=True)
    sp_train_best.add_argument("--study-name", default=None)
    sp_train_best.add_argument("--final-epochs", type=int, default=None)

    sp_train_best_all = sub.add_parser("train-best-all", help="Train multiple chunks using their saved HPO best parameters")
    add_common(sp_train_best_all)
    sp_train_best_all.add_argument("--chunks", default="1,2,3")
    sp_train_best_all.add_argument("--study-name-prefix", default="sentiment_analyst_chunk")
    sp_train_best_all.add_argument("--final-epochs", type=int, default=None)

    sp_inspect = sub.add_parser("inspect", help="Inspect label/embedding availability and alignment")
    add_common(sp_inspect)
    sp_inspect.add_argument("--chunk", type=int, choices=[1, 2, 3], required=True)

    sp_predict = sub.add_parser("predict", help="Export predictions and sentiment analyst embeddings")
    add_common(sp_predict)
    sp_predict.add_argument("--chunk", type=int, choices=[1, 2, 3], required=True)
    sp_predict.add_argument("--split", choices=["train", "val", "test"], required=True)
    sp_predict.add_argument("--checkpoint", default="best", help="best, latest, or path to a .pt checkpoint")

    sp_predict_all = sub.add_parser("predict-all", help="Export predictions for train/val/test for one or more chunks")
    add_common(sp_predict_all)
    sp_predict_all.add_argument("--chunks", default="1,2,3")
    sp_predict_all.add_argument("--splits", default="train,val,test")
    sp_predict_all.add_argument("--checkpoint", default="best")

    return p.parse_args()

def main() -> None:
    args = parse_args()
    if args.command == "train":
        cfg = build_cfg_from_args(args)
        train_one_chunk(cfg)
    elif args.command == "train-all":
        summaries = []
        for chunk_id in parse_int_list(args.chunks):
            cfg = build_cfg_from_args(args, chunk_id=chunk_id)
            summaries.append(train_one_chunk(cfg))
        print(json.dumps(summaries, indent=2, default=str))
    elif args.command == "hpo":
        cfg = build_cfg_from_args(args)
        result = run_hpo_for_chunk(cfg, args)
        print(json.dumps(result, indent=2, default=str))
    elif args.command == "hpo-all":
        outputs = []
        original_study_name = args.study_name
        for chunk_id in parse_int_list(args.chunks):
            cfg = build_cfg_from_args(args, chunk_id=chunk_id)
            args.study_name = original_study_name or f"sentiment_analyst_chunk{chunk_id}_hpo"
            outputs.append(run_hpo_for_chunk(cfg, args))
        print(json.dumps(outputs, indent=2, default=str))
    elif args.command == "train-best":
        cfg = build_cfg_from_args(args)
        result = train_best_for_chunk(cfg, study_name=args.study_name, final_epochs=args.final_epochs)
        print(json.dumps(result, indent=2, default=str))
    elif args.command == "train-best-all":
        outputs = []
        for chunk_id in parse_int_list(args.chunks):
            cfg = build_cfg_from_args(args, chunk_id=chunk_id)
            study_name = f"{args.study_name_prefix}{chunk_id}_hpo"
            outputs.append(train_best_for_chunk(cfg, study_name=study_name, final_epochs=args.final_epochs))
        print(json.dumps(outputs, indent=2, default=str))
    elif args.command == "inspect":
        cfg = build_cfg_from_args(args)
        inspect_labels(cfg)
    elif args.command == "predict":
        cfg = build_cfg_from_args(args)
        if getattr(cfg, 'xai_method', 'gradient') == 'shap':
            print("=" * 60)
            print("   FULL SHAP SELECTED")
            print("=" * 60)
            print(f"SHAP DeepExplainer will process {cfg.xai_sample_size} rows.")
            print("This may take 10-30+ minutes depending on sample size.")
            print("For faster results, use --xai-method gradient (default).")
            print("=" * 60)
        max_rows = {"train": cfg.max_train_rows, "val": cfg.max_val_rows, "test": cfg.max_test_rows}[args.split]
        metrics = run_prediction_export(cfg, split=args.split, checkpoint=args.checkpoint, max_rows=max_rows)
        print(json.dumps(metrics, indent=2, default=str))
    elif args.command == "predict-all":
        outputs = []
        for chunk_id in parse_int_list(args.chunks):
            cfg = build_cfg_from_args(args, chunk_id=chunk_id)
            
            if getattr(cfg, 'xai_method', 'gradient') == 'shap':
                print("=" * 60)
                print("   FULL SHAP SELECTED")
                print("=" * 60)
                print(f"SHAP DeepExplainer will process {cfg.xai_sample_size} rows.")
                print("This may take 10-30+ minutes depending on sample size.")
                print("For faster results, use --xai-method gradient (default).")
                print("=" * 60)
            for split in parse_split_list(args.splits):
                max_rows = {"train": cfg.max_train_rows, "val": cfg.max_val_rows, "test": cfg.max_test_rows}[split]
                outputs.append(run_prediction_export(cfg, split=split, checkpoint=args.checkpoint, max_rows=max_rows))
        print(json.dumps(outputs, indent=2, default=str))
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

# Optional command for good working:
# python -m py_compile code/analysts/sentiment_analyst.py

# Recommended HPO-first workflow examples:(these are irrelevant but just kind of dry run commands)
# python code/analysts/sentiment_analyst.py inspect --repo-root ~/fin-glassbox --chunk 1 --device cpu
# python code/analysts/sentiment_analyst.py hpo --repo-root ~/fin-glassbox --chunk 1 --device cpu --trials 5 --hpo-max-train-rows 10000 --hpo-max-val-rows 3000
# python code/analysts/sentiment_analyst.py train-best --repo-root ~/fin-glassbox --chunk 1 --device cpu
# python code/analysts/sentiment_analyst.py predict --repo-root ~/fin-glassbox --chunk 1 --split val --device cpu --checkpoint best

# CPU-safe HPO smoke test(a small test to see if code works)
# This uses real embeddings and real labels, but only a subset:
# python code/analysts/sentiment_analyst.py hpo --repo-root ~/fin-glassbox --chunk 1 --device cpu --trials 3 --hpo-max-train-rows 10000 --hpo-max-val-rows 3000 --hpo-epochs-min 2 --hpo-epochs-max 5
# Then train from the best HPO result:
# python code/analysts/sentiment_analyst.py train-best --repo-root ~/fin-glassbox --chunk 1 --device cpu

# (this is the real code runner)
# First Hyper Parameter finding:
# python code/analysts/sentiment_analyst.py hpo-all --repo-root ~/fin-glassbox --chunks 1,2,3 --trials 30 --device # use "cuda" only if you are using gpu otherwise use "cpu" here
# Then train final models using the saved best parameters:
# python code/analysts/sentiment_analyst.py train-best-all --repo-root ~/fin-glassbox --chunks 1,2,3 --device # use "cuda" only if you are using gpu otherwise use "cpu" here
# Then export predictions and 64-dim sentiment analyst embeddings:
# python code/analysts/sentiment_analyst.py predict-all --repo-root ~/fin-glassbox --chunks 1,2,3 --splits train,val,test --checkpoint best --device # use "cuda" only if you are using gpu otherwise use "cpu" here

