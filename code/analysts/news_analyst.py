#!/usr/bin/env python3
"""
Supervised News Analyst for the fin-glassbox project.

This module trains a document-level attention-pooling analyst on real FinBERT text
embeddings and real market-derived labels produced by:

    code/analysts/text_market_label_builder.py

It does not create dummy data, synthetic labels, or fake embeddings.

Core contract:
- Input embeddings: outputs/embeddings/FinBERT/chunk{N}_{split}_embeddings.npy
- Input labels: outputs/results/analysts/labels/text_market_labels_chunk{N}_{split}.csv
- Embedding dimension: 256 float32 values per SEC text chunk
- Label rows must align row-for-row with the corresponding FinBERT embedding matrix
- News Analyst aggregates chunk-level embeddings to document-level representations
- Training uses only the train split of the selected chronological chunk
- Validation/test are never used to fit preprocessing statistics, labels, or thresholds

Outputs:
- PyTorch checkpoints: outputs/models/analysts/news/chunk{N}/latest.pt and best.pt
- Training history: outputs/results/analysts/news/chunk{N}_training_history.csv
- Evaluation metrics: outputs/results/analysts/news/chunk{N}_{split}_metrics.json
- Document predictions: outputs/results/analysts/news/chunk{N}_{split}_news_predictions.csv
- Attention trace: outputs/results/analysts/news/chunk{N}_{split}_attention.csv
- Analyst embeddings: outputs/embeddings/analysts/news/chunk{N}_{split}_news_embeddings.npy

"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Keep this file PyTorch-only.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

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
except Exception:
    optuna = None


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
    "news_event_impact_target",
    "news_importance_target",
    "risk_relevance_target",
]

TARGET_IMPACT_COLUMN = "news_event_impact_target"
TARGET_IMPORTANCE_COLUMN = "news_importance_target"
TARGET_RISK_COLUMN = "risk_relevance_target"


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


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


@dataclass
class NewsConfig:
    repo_root: Path = Path(".")
    env_file: Path = Path(".env")

    embeddings_dir: Path = Path("outputs/embeddings/FinBERT")
    labels_dir: Path = Path("outputs/results/analysts/labels")
    analyst_embeddings_dir: Path = Path("outputs/embeddings/analysts/news")
    models_dir: Path = Path("outputs/models/analysts/news")
    results_dir: Path = Path("outputs/results/analysts/news")
    code_results_dir: Path = Path("outputs/codeResults/analysts/news")

    chunk_id: int = 1
    input_dim: int = 256
    d_model: int = 128
    attention_heads: int = 4
    self_attention_layers: int = 1
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    representation_dim: int = 128
    dropout: float = 0.15
    max_chunks_per_document: int = 64
    use_metadata_features: bool = True

    seed: int = 42
    device: str = "cuda"
    torch_num_threads: Optional[int] = None
    num_workers: int = 0
    batch_size: int = 96
    eval_batch_size: int = 192
    epochs: int = 40
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    early_stop_patience: int = 8
    mixed_precision: bool = True

    impact_loss_weight: float = 1.0
    importance_loss_weight: float = 0.75
    risk_loss_weight: float = 0.75
    volatility_loss_weight: float = 0.25
    drawdown_loss_weight: float = 0.25

    max_train_groups: Optional[int] = None
    max_val_groups: Optional[int] = None
    max_test_groups: Optional[int] = None
    resume: bool = False
    overwrite_predictions: bool = True

    def resolve(self) -> "NewsConfig":
        self.repo_root = Path(self.repo_root).resolve()
        env = read_env_file(self.repo_root / self.env_file)

        if env.get("FinBERTembeddingsPath"):
            self.embeddings_dir = Path(env["FinBERTembeddingsPath"])
        elif env.get("embeddingsPathGlobal"):
            self.embeddings_dir = Path(env["embeddingsPathGlobal"]) / "FinBERT"

        if env.get("embeddingsPathGlobal"):
            self.analyst_embeddings_dir = Path(env["embeddingsPathGlobal"]) / "analysts" / "news"

        if env.get("modelsPathGlobal"):
            self.models_dir = Path(env["modelsPathGlobal"]) / "analysts" / "news"

        if env.get("resultsPathGlobal"):
            self.labels_dir = Path(env["resultsPathGlobal"]) / "analysts" / "labels"
            self.results_dir = Path(env["resultsPathGlobal"]) / "analysts" / "news"

        if env.get("codeOutputsPathGlobal"):
            self.code_results_dir = Path(env["codeOutputsPathGlobal"]) / "analysts" / "news"

        self.embeddings_dir = resolve_repo_path(self.repo_root, self.embeddings_dir)
        self.labels_dir = resolve_repo_path(self.repo_root, self.labels_dir)
        self.analyst_embeddings_dir = resolve_repo_path(self.repo_root, self.analyst_embeddings_dir)
        self.models_dir = resolve_repo_path(self.repo_root, self.models_dir)
        self.results_dir = resolve_repo_path(self.repo_root, self.results_dir)
        self.code_results_dir = resolve_repo_path(self.repo_root, self.code_results_dir)

        if self.d_model % self.attention_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by attention_heads={self.attention_heads}")
        if self.max_chunks_per_document <= 0:
            raise ValueError("max_chunks_per_document must be positive")
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
        return self.chunk_model_dir() / "document_metadata_preprocessor.json"

    def latest_checkpoint_path(self) -> Path:
        return self.chunk_model_dir() / "latest.pt"

    def best_checkpoint_path(self) -> Path:
        return self.chunk_model_dir() / "best.pt"

    def best_params_path(self) -> Path:
        return self.code_results_dir / "hpo" / f"news_analyst_chunk{self.chunk_id}_best_params.json"

    def hpo_storage_path(self) -> Path:
        return self.code_results_dir / "hpo" / "news_analyst_optuna.db"


def embedding_path(cfg: NewsConfig, split: str) -> Path:
    return cfg.embeddings_dir / f"chunk{cfg.chunk_id}_{split}_embeddings.npy"


def label_path(cfg: NewsConfig, split: str) -> Path:
    return cfg.labels_dir / f"text_market_labels_chunk{cfg.chunk_id}_{split}.csv"


def predictions_path(cfg: NewsConfig, split: str) -> Path:
    return cfg.results_dir / f"chunk{cfg.chunk_id}_{split}_news_predictions.csv"


def attention_path(cfg: NewsConfig, split: str) -> Path:
    return cfg.results_dir / f"chunk{cfg.chunk_id}_{split}_attention.csv"


def metrics_path(cfg: NewsConfig, split: str) -> Path:
    return cfg.results_dir / f"chunk{cfg.chunk_id}_{split}_metrics.json"


def analyst_embedding_path(cfg: NewsConfig, split: str) -> Path:
    return cfg.analyst_embeddings_dir / f"chunk{cfg.chunk_id}_{split}_news_embeddings.npy"


def history_path(cfg: NewsConfig) -> Path:
    return cfg.results_dir / f"chunk{cfg.chunk_id}_training_history.csv"


def get_optional_target_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    vol_cols = [c for c in df.columns if c.startswith("volatility_spike_") and c.endswith("_target")]
    dd_cols = [c for c in df.columns if c.startswith("drawdown_risk_") and c.endswith("_target")]
    return (vol_cols[0] if vol_cols else None, dd_cols[0] if dd_cols else None)


def read_label_file(cfg: NewsConfig, split: str) -> pd.DataFrame:
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


def load_embeddings(cfg: NewsConfig, split: str, mmap_mode: str = "r") -> np.ndarray:
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


def validate_alignment(cfg: NewsConfig, split: str, labels: pd.DataFrame, embeddings: np.ndarray) -> None:
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


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def valid_news_rows(df: pd.DataFrame) -> np.ndarray:
    impact = pd.to_numeric(df[TARGET_IMPACT_COLUMN], errors="coerce")
    importance = pd.to_numeric(df[TARGET_IMPORTANCE_COLUMN], errors="coerce")
    risk = pd.to_numeric(df[TARGET_RISK_COLUMN], errors="coerce")
    available = bool_series(df["label_available"])
    valid = impact.notna() & importance.notna() & risk.notna() & available
    return np.flatnonzero(valid.to_numpy())


def deterministic_subset(indices: np.ndarray, max_items: Optional[int], seed: int) -> np.ndarray:
    if max_items is None or max_items <= 0 or len(indices) <= max_items:
        return indices
    rng = np.random.default_rng(seed)
    selected = rng.choice(indices, size=max_items, replace=False)
    return np.sort(selected.astype(np.int64))


def select_evenly_spaced(indices: np.ndarray, max_len: int) -> np.ndarray:
    if len(indices) <= max_len:
        return indices.astype(np.int64)
    positions = np.linspace(0, len(indices) - 1, num=max_len)
    positions = np.round(positions).astype(np.int64)
    return indices[positions].astype(np.int64)


@dataclass
class DocumentGroup:
    group_id: str
    doc_id: str
    accession: str
    ticker: str
    cik: str
    filing_date: str
    year: int
    form_type: str
    source_name: str
    row_indices: np.ndarray
    chunk_indices: np.ndarray
    n_chunks_original: int
    total_word_count: float
    mean_word_count: float
    max_chunk_index: int
    target_impact: float
    target_importance: float
    target_risk: float
    target_volatility: float
    target_drawdown: float


def build_document_groups(df: pd.DataFrame, valid_indices: np.ndarray) -> List[DocumentGroup]:
    if len(valid_indices) == 0:
        return []

    work = df.iloc[valid_indices].copy()
    work["_row_index"] = valid_indices.astype(np.int64)
    work["chunk_index"] = pd.to_numeric(work["chunk_index"], errors="coerce").fillna(0).astype(int)
    work["word_count"] = pd.to_numeric(work["word_count"], errors="coerce").fillna(0).astype(float)
    work["year"] = pd.to_numeric(work["year"], errors="coerce").fillna(-1).astype(int)

    vol_col, dd_col = get_optional_target_columns(df)
    groups: List[DocumentGroup] = []
    group_cols = ["doc_id", "accession"]

    for (doc_id, accession), sub in work.groupby(group_cols, sort=False):
        sub = sub.sort_values(["chunk_index", "_row_index"], kind="mergesort")
        row_indices = sub["_row_index"].to_numpy(dtype=np.int64)
        chunk_indices = sub["chunk_index"].to_numpy(dtype=np.int64)
        first = sub.iloc[0]

        impact = pd.to_numeric(sub[TARGET_IMPACT_COLUMN], errors="coerce").mean()
        importance = pd.to_numeric(sub[TARGET_IMPORTANCE_COLUMN], errors="coerce").mean()
        risk = pd.to_numeric(sub[TARGET_RISK_COLUMN], errors="coerce").mean()
        vol_target = pd.to_numeric(sub[vol_col], errors="coerce").mean() if vol_col else np.nan
        dd_target = pd.to_numeric(sub[dd_col], errors="coerce").mean() if dd_col else np.nan

        groups.append(
            DocumentGroup(
                group_id=f"{str(doc_id)}::{str(accession)}",
                doc_id=str(doc_id),
                accession=str(accession),
                ticker=str(first.get("ticker", "")),
                cik=str(first.get("cik", "")),
                filing_date=str(first.get("filing_date", "")),
                year=int(first.get("year", -1)),
                form_type=str(first.get("form_type", "")),
                source_name=str(first.get("source_name", "")),
                row_indices=row_indices,
                chunk_indices=chunk_indices,
                n_chunks_original=int(len(row_indices)),
                total_word_count=float(sub["word_count"].sum()),
                mean_word_count=float(sub["word_count"].mean()),
                max_chunk_index=int(sub["chunk_index"].max()),
                target_impact=safe_float(impact),
                target_importance=safe_float(importance),
                target_risk=safe_float(risk),
                target_volatility=safe_float(vol_target),
                target_drawdown=safe_float(dd_target),
            )
        )
    return groups


def groups_to_frame(groups: Sequence[DocumentGroup]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for g in groups:
        rows.append(
            {
                "group_id": g.group_id,
                "doc_id": g.doc_id,
                "accession": g.accession,
                "ticker": g.ticker,
                "cik": g.cik,
                "filing_date": g.filing_date,
                "year": g.year,
                "form_type": g.form_type,
                "source_name": g.source_name,
                "n_chunks_original": g.n_chunks_original,
                "total_word_count": g.total_word_count,
                "mean_word_count": g.mean_word_count,
                "max_chunk_index": g.max_chunk_index,
                "target_impact": g.target_impact,
                "target_importance": g.target_importance,
                "target_risk": g.target_risk,
                "target_volatility": g.target_volatility,
                "target_drawdown": g.target_drawdown,
            }
        )
    return pd.DataFrame(rows)


class DocumentMetadataPreprocessor:
    """Train-only document metadata encoder for filing-time metadata features."""

    numeric_columns = ["year", "n_chunks_original", "total_word_count", "mean_word_count", "max_chunk_index"]

    def __init__(self, use_metadata_features: bool = True):
        self.use_metadata_features = bool(use_metadata_features)
        self.numeric_mean: Dict[str, float] = {}
        self.numeric_std: Dict[str, float] = {}
        self.form_type_vocab: Dict[str, int] = {"<UNK>": 0}
        self.source_name_vocab: Dict[str, int] = {"<UNK>": 0}
        self.feature_dim: int = 0
        self.fitted: bool = False

    @staticmethod
    def _norm(value: Any) -> str:
        if pd.isna(value):
            return "<UNK>"
        text = str(value).strip().upper()
        return text if text else "<UNK>"

    def fit(self, groups: Sequence[DocumentGroup], group_indices: np.ndarray) -> "DocumentMetadataPreprocessor":
        if not self.use_metadata_features:
            self.feature_dim = 0
            self.fitted = True
            return self

        frame = groups_to_frame([groups[int(i)] for i in group_indices])
        for col in self.numeric_columns:
            values = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            mean = float(values.mean()) if values.notna().any() else 0.0
            std = float(values.std(ddof=0)) if values.notna().any() else 1.0
            if not np.isfinite(std) or std < 1e-8:
                std = 1.0
            self.numeric_mean[col] = mean
            self.numeric_std[col] = std

        forms = sorted({self._norm(x) for x in frame["form_type"].tolist()})
        sources = sorted({self._norm(x) for x in frame["source_name"].tolist()})
        self.form_type_vocab = {"<UNK>": 0}
        self.source_name_vocab = {"<UNK>": 0}
        for value in forms:
            if value not in self.form_type_vocab:
                self.form_type_vocab[value] = len(self.form_type_vocab)
        for value in sources:
            if value not in self.source_name_vocab:
                self.source_name_vocab[value] = len(self.source_name_vocab)

        self.feature_dim = len(self.numeric_columns) + len(self.form_type_vocab) + len(self.source_name_vocab)
        self.fitted = True
        return self

    def transform_frame(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("DocumentMetadataPreprocessor must be fitted before transform")
        if not self.use_metadata_features:
            return np.zeros((len(frame), 0), dtype=np.float32)

        numeric_parts: List[np.ndarray] = []
        for col in self.numeric_columns:
            values = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(self.numeric_mean[col])
            scaled = (values.to_numpy(dtype=np.float32) - np.float32(self.numeric_mean[col])) / np.float32(self.numeric_std[col])
            numeric_parts.append(scaled.reshape(-1, 1))

        form_mat = np.zeros((len(frame), len(self.form_type_vocab)), dtype=np.float32)
        source_mat = np.zeros((len(frame), len(self.source_name_vocab)), dtype=np.float32)
        for i, value in enumerate(frame["form_type"].tolist()):
            form_mat[i, self.form_type_vocab.get(self._norm(value), 0)] = 1.0
        for i, value in enumerate(frame["source_name"].tolist()):
            source_mat[i, self.source_name_vocab.get(self._norm(value), 0)] = 1.0

        return np.concatenate(numeric_parts + [form_mat, source_mat], axis=1).astype(np.float32)

    def transform_groups(self, groups: Sequence[DocumentGroup]) -> np.ndarray:
        return self.transform_frame(groups_to_frame(groups))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "use_metadata_features": self.use_metadata_features,
            "numeric_columns": self.numeric_columns,
            "numeric_mean": self.numeric_mean,
            "numeric_std": self.numeric_std,
            "form_type_vocab": self.form_type_vocab,
            "source_name_vocab": self.source_name_vocab,
            "feature_dim": self.feature_dim,
            "fitted": self.fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadataPreprocessor":
        obj = cls(use_metadata_features=bool(data.get("use_metadata_features", True)))
        obj.numeric_mean = {str(k): float(v) for k, v in data.get("numeric_mean", {}).items()}
        obj.numeric_std = {str(k): float(v) for k, v in data.get("numeric_std", {}).items()}
        obj.form_type_vocab = {str(k): int(v) for k, v in data.get("form_type_vocab", {"<UNK>": 0}).items()}
        obj.source_name_vocab = {str(k): int(v) for k, v in data.get("source_name_vocab", {"<UNK>": 0}).items()}
        obj.feature_dim = int(data.get("feature_dim", 0))
        obj.fitted = bool(data.get("fitted", True))
        return obj

    def save(self, path: Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> "DocumentMetadataPreprocessor":
        if not path.exists():
            raise FileNotFoundError(f"Missing document metadata preprocessor: {path}")
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


class NewsDocumentDataset(Dataset):
    def __init__(
        self,
        embeddings: np.ndarray,
        groups: Sequence[DocumentGroup],
        group_indices: np.ndarray,
        metadata_features: np.ndarray,
        max_chunks_per_document: int,
        require_targets: bool = True,
    ):
        self.embeddings = embeddings
        self.groups = list(groups)
        self.group_indices = group_indices.astype(np.int64)
        self.metadata_features = metadata_features.astype(np.float32)
        self.max_chunks_per_document = int(max_chunks_per_document)
        self.require_targets = bool(require_targets)

        if self.metadata_features.shape[0] != len(self.groups):
            raise ValueError("metadata_features must have one row per group")

    def __len__(self) -> int:
        return len(self.group_indices)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        group_pos = int(self.group_indices[i])
        group = self.groups[group_pos]
        selected_rows = select_evenly_spaced(group.row_indices, self.max_chunks_per_document)
        selected_chunk_indices = select_evenly_spaced(group.chunk_indices, self.max_chunks_per_document)
        x = np.asarray(self.embeddings[selected_rows], dtype=np.float32)
        meta = np.asarray(self.metadata_features[group_pos], dtype=np.float32)

        item: Dict[str, Any] = {
            "x": torch.from_numpy(x),
            "metadata": torch.from_numpy(meta),
            "group_pos": torch.tensor(group_pos, dtype=torch.long),
            "row_indices": torch.from_numpy(selected_rows.astype(np.int64)),
            "chunk_indices": torch.from_numpy(selected_chunk_indices.astype(np.int64)),
        }
        if self.require_targets:
            item["target_impact"] = torch.tensor(group.target_impact, dtype=torch.float32)
            item["target_importance"] = torch.tensor(group.target_importance, dtype=torch.float32)
            item["target_risk"] = torch.tensor(group.target_risk, dtype=torch.float32)
            item["target_volatility"] = torch.tensor(0.0 if np.isnan(group.target_volatility) else group.target_volatility, dtype=torch.float32)
            item["target_drawdown"] = torch.tensor(0.0 if np.isnan(group.target_drawdown) else group.target_drawdown, dtype=torch.float32)
            item["has_volatility_target"] = torch.tensor(0.0 if np.isnan(group.target_volatility) else 1.0, dtype=torch.float32)
            item["has_drawdown_target"] = torch.tensor(0.0 if np.isnan(group.target_drawdown) else 1.0, dtype=torch.float32)
        return item


def news_collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_len = max(int(item["x"].shape[0]) for item in batch)
    input_dim = int(batch[0]["x"].shape[1])
    meta_dim = int(batch[0]["metadata"].shape[0])

    x = torch.zeros((batch_size, max_len, input_dim), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    row_indices = torch.full((batch_size, max_len), -1, dtype=torch.long)
    chunk_indices = torch.full((batch_size, max_len), -1, dtype=torch.long)
    metadata = torch.zeros((batch_size, meta_dim), dtype=torch.float32)
    group_pos = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        n = int(item["x"].shape[0])
        x[i, :n, :] = item["x"]
        mask[i, :n] = True
        row_indices[i, :n] = item["row_indices"]
        chunk_indices[i, :n] = item["chunk_indices"]
        metadata[i] = item["metadata"]
        group_pos[i] = item["group_pos"]

    out: Dict[str, torch.Tensor] = {
        "x": x,
        "mask": mask,
        "metadata": metadata,
        "group_pos": group_pos,
        "row_indices": row_indices,
        "chunk_indices": chunk_indices,
    }

    if "target_impact" in batch[0]:
        for key in [
            "target_impact",
            "target_importance",
            "target_risk",
            "target_volatility",
            "target_drawdown",
            "has_volatility_target",
            "has_drawdown_target",
        ]:
            out[key] = torch.stack([item[key] for item in batch])
    return out


class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, d_model: int, attention_heads: int, dropout: float):
        super().__init__()
        self.d_model = int(d_model)
        self.attention_heads = int(attention_heads)
        self.score = nn.Linear(self.d_model, self.attention_heads)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(self.d_model * self.attention_heads, self.d_model)
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(x)  # (B, S, H)
        scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        weights = torch.softmax(scores, dim=1)
        weights = self.dropout(weights)
        pooled_heads = torch.einsum("bsh,bsd->bhd", weights, x)
        pooled = pooled_heads.reshape(x.shape[0], self.attention_heads * self.d_model)
        pooled = self.output(pooled)
        pooled = self.norm(pooled)
        return pooled, weights.permute(0, 2, 1).contiguous()  # (B, H, S)


class NewsAnalystAttentionModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        metadata_dim: int = 0,
        d_model: int = 128,
        attention_heads: int = 4,
        self_attention_layers: int = 1,
        hidden_dims: Sequence[int] = (128, 64),
        representation_dim: int = 128,
        dropout: float = 0.15,
    ):
        super().__init__()
        if d_model % attention_heads != 0:
            raise ValueError("d_model must be divisible by attention_heads")
        self.input_dim = int(input_dim)
        self.metadata_dim = int(metadata_dim)
        self.d_model = int(d_model)
        self.attention_heads = int(attention_heads)
        self.self_attention_layers = int(self_attention_layers)
        self.hidden_dims = [int(x) for x in hidden_dims]
        self.representation_dim = int(representation_dim)
        self.dropout = float(dropout)

        self.token_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.Tanh(),
            nn.Dropout(self.dropout),
        )

        if self.metadata_dim > 0:
            self.metadata_projection = nn.Sequential(
                nn.Linear(self.metadata_dim, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.Tanh(),
            )
        else:
            self.metadata_projection = None

        self.context_layers = nn.ModuleList()
        for _ in range(self.self_attention_layers):
            attn = nn.MultiheadAttention(
                embed_dim=self.d_model,
                num_heads=self.attention_heads,
                dropout=self.dropout,
                batch_first=True,
            )
            ff = nn.Sequential(
                nn.Linear(self.d_model, self.d_model * 2),
                nn.Tanh(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model * 2, self.d_model),
            )
            self.context_layers.append(nn.ModuleDict({"attn": attn, "norm1": nn.LayerNorm(self.d_model), "ff": ff, "norm2": nn.LayerNorm(self.d_model)}))

        self.pool = MultiHeadAttentionPooling(self.d_model, self.attention_heads, self.dropout)

        layers: List[nn.Module] = []
        prev = self.d_model
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

        self.impact_head = nn.Linear(self.representation_dim, 1)
        self.importance_head = nn.Linear(self.representation_dim, 1)
        self.risk_head = nn.Linear(self.representation_dim, 1)
        self.volatility_head = nn.Linear(self.representation_dim, 1)
        self.drawdown_head = nn.Linear(self.representation_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, metadata: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        h = self.token_projection(x)
        if self.metadata_projection is not None and metadata is not None and metadata.shape[1] > 0:
            meta = self.metadata_projection(metadata).unsqueeze(1)
            h = h + meta

        key_padding_mask = ~mask
        for layer in self.context_layers:
            attn_out, _ = layer["attn"](h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
            h = layer["norm1"](h + attn_out)
            ff_out = layer["ff"](h)
            h = layer["norm2"](h + ff_out)
            h = h.masked_fill(~mask.unsqueeze(-1), 0.0)

        pooled, attention_weights = self.pool(h, mask)
        rep = self.trunk(pooled)
        return {
            "news_embedding": rep,
            "event_impact_score": torch.tanh(self.impact_head(rep)).squeeze(-1),
            "news_importance_score": torch.sigmoid(self.importance_head(rep)).squeeze(-1),
            "risk_relevance_score": torch.sigmoid(self.risk_head(rep)).squeeze(-1),
            "volatility_spike_score": torch.sigmoid(self.volatility_head(rep)).squeeze(-1),
            "drawdown_risk_score": torch.sigmoid(self.drawdown_head(rep)).squeeze(-1),
            "attention_weights": attention_weights,
        }

    def model_config(self) -> Dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "metadata_dim": self.metadata_dim,
            "d_model": self.d_model,
            "attention_heads": self.attention_heads,
            "self_attention_layers": self.self_attention_layers,
            "hidden_dims": self.hidden_dims,
            "representation_dim": self.representation_dim,
            "dropout": self.dropout,
            "activation": "tanh",
            "impact_output_activation": "tanh",
            "importance_output_activation": "sigmoid",
            "risk_output_activation": "sigmoid",
            "pooling": "multi_head_attention_pooling",
        }


@dataclass
class SplitBundle:
    split: str
    labels: pd.DataFrame
    embeddings: np.ndarray
    groups: List[DocumentGroup]
    metadata_features: np.ndarray
    group_indices: np.ndarray


def prepare_base_split(cfg: NewsConfig, split: str) -> Tuple[pd.DataFrame, np.ndarray, List[DocumentGroup]]:
    labels = read_label_file(cfg, split)
    embeddings = load_embeddings(cfg, split, mmap_mode="r")
    validate_alignment(cfg, split, labels, embeddings)
    valid_indices = valid_news_rows(labels)
    groups = build_document_groups(labels, valid_indices)
    if len(groups) == 0:
        raise ValueError(f"No valid news supervised document groups found for chunk{cfg.chunk_id}_{split}")
    return labels, embeddings, groups


def prepare_split_bundle(
    cfg: NewsConfig,
    split: str,
    preprocessor: DocumentMetadataPreprocessor,
    max_groups: Optional[int] = None,
) -> SplitBundle:
    labels, embeddings, groups = prepare_base_split(cfg, split)
    metadata_features = preprocessor.transform_groups(groups)
    indices = np.arange(len(groups), dtype=np.int64)
    indices = deterministic_subset(indices, max_groups, cfg.seed + {"train": 11, "val": 17, "test": 23}[split])
    return SplitBundle(split=split, labels=labels, embeddings=embeddings, groups=groups, metadata_features=metadata_features, group_indices=indices)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, cfg: NewsConfig, device: torch.device) -> DataLoader:
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
        collate_fn=news_collate,
        persistent_workers=(cfg.num_workers > 0),
    )


def masked_bce(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask > 0.5
    if not torch.any(valid):
        return pred.new_tensor(0.0)
    return F.binary_cross_entropy(pred[valid], target[valid])


def compute_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], cfg: NewsConfig) -> Tuple[torch.Tensor, Dict[str, float]]:
    target_impact = batch["target_impact"].clamp(-1.0, 1.0)
    target_importance = batch["target_importance"].clamp(0.0, 1.0)
    target_risk = batch["target_risk"].clamp(0.0, 1.0)
    target_vol = batch["target_volatility"].clamp(0.0, 1.0)
    target_dd = batch["target_drawdown"].clamp(0.0, 1.0)
    has_vol = batch["has_volatility_target"]
    has_dd = batch["has_drawdown_target"]

    impact_loss = F.mse_loss(outputs["event_impact_score"], target_impact)
    importance_loss = F.binary_cross_entropy(outputs["news_importance_score"], target_importance)
    risk_loss = F.binary_cross_entropy(outputs["risk_relevance_score"], target_risk)
    vol_loss = masked_bce(outputs["volatility_spike_score"], target_vol, has_vol)
    dd_loss = masked_bce(outputs["drawdown_risk_score"], target_dd, has_dd)

    loss = (
        cfg.impact_loss_weight * impact_loss
        + cfg.importance_loss_weight * importance_loss
        + cfg.risk_loss_weight * risk_loss
        + cfg.volatility_loss_weight * vol_loss
        + cfg.drawdown_loss_weight * dd_loss
    )
    parts = {
        "loss": float(loss.detach().cpu().item()),
        "impact_loss": float(impact_loss.detach().cpu().item()),
        "importance_loss": float(importance_loss.detach().cpu().item()),
        "risk_loss": float(risk_loss.detach().cpu().item()),
        "volatility_loss": float(vol_loss.detach().cpu().item()),
        "drawdown_loss": float(dd_loss.detach().cpu().item()),
    }
    return loss, parts


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    if np.nanstd(a) < 1e-12 or np.nanstd(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(pred) & np.isfinite(target)
    if valid.sum() == 0:
        return {"mse": float("nan"), "mae": float("nan"), "rmse": float("nan"), "corr": float("nan")}
    p = pred[valid]
    t = target[valid]
    err = p - t
    return {
        "mse": float(np.mean(err * err)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "corr": safe_corr(p, t),
    }


def binary_metrics(pred_score: np.ndarray, target_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    valid = np.isfinite(pred_score) & np.isfinite(target_score)
    if valid.sum() == 0:
        return {"accuracy": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    pred = pred_score[valid] >= threshold
    target = target_score[valid] >= threshold
    tp = float(np.sum(pred & target))
    tn = float(np.sum((~pred) & (~target)))
    fp = float(np.sum(pred & (~target)))
    fn = float(np.sum((~pred) & target))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def run_epoch(
    model: NewsAnalystAttentionModel,
    loader: DataLoader,
    optimiser: Optional[torch.optim.Optimizer],
    scaler: Optional[GradScaler],
    cfg: NewsConfig,
    device: torch.device,
    train: bool,
) -> Dict[str, float]:
    model.train(mode=train)
    totals = {"loss": 0.0, "impact_loss": 0.0, "importance_loss": 0.0, "risk_loss": 0.0, "volatility_loss": 0.0, "drawdown_loss": 0.0}
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        if train:
            assert optimiser is not None
            optimiser.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with autocast(enabled=(cfg.mixed_precision and device.type == "cuda")):
                outputs = model(batch["x"], batch["mask"], batch["metadata"])
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
def evaluate_supervised(model: NewsAnalystAttentionModel, loader: DataLoader, cfg: NewsConfig, device: torch.device) -> Dict[str, Any]:
    model.eval()
    losses = {"loss": 0.0, "impact_loss": 0.0, "importance_loss": 0.0, "risk_loss": 0.0, "volatility_loss": 0.0, "drawdown_loss": 0.0}
    n_batches = 0
    preds: Dict[str, List[np.ndarray]] = {"impact": [], "importance": [], "risk": [], "volatility": [], "drawdown": []}
    targets: Dict[str, List[np.ndarray]] = {"impact": [], "importance": [], "risk": [], "volatility": [], "drawdown": []}
    masks: Dict[str, List[np.ndarray]] = {"volatility": [], "drawdown": []}

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with autocast(enabled=(cfg.mixed_precision and device.type == "cuda")):
            outputs = model(batch["x"], batch["mask"], batch["metadata"])
            _, parts = compute_loss(outputs, batch, cfg)
        for key in losses:
            losses[key] += parts[key]
        n_batches += 1

        preds["impact"].append(outputs["event_impact_score"].float().cpu().numpy())
        preds["importance"].append(outputs["news_importance_score"].float().cpu().numpy())
        preds["risk"].append(outputs["risk_relevance_score"].float().cpu().numpy())
        preds["volatility"].append(outputs["volatility_spike_score"].float().cpu().numpy())
        preds["drawdown"].append(outputs["drawdown_risk_score"].float().cpu().numpy())

        targets["impact"].append(batch["target_impact"].float().cpu().numpy())
        targets["importance"].append(batch["target_importance"].float().cpu().numpy())
        targets["risk"].append(batch["target_risk"].float().cpu().numpy())
        targets["volatility"].append(batch["target_volatility"].float().cpu().numpy())
        targets["drawdown"].append(batch["target_drawdown"].float().cpu().numpy())
        masks["volatility"].append(batch["has_volatility_target"].float().cpu().numpy())
        masks["drawdown"].append(batch["has_drawdown_target"].float().cpu().numpy())

    out: Dict[str, Any] = {key: value / max(1, n_batches) for key, value in losses.items()}
    for name in ["impact", "importance", "risk"]:
        p = np.concatenate(preds[name]) if preds[name] else np.array([], dtype=np.float32)
        t = np.concatenate(targets[name]) if targets[name] else np.array([], dtype=np.float32)
        out.update({f"{name}_{k}": v for k, v in regression_metrics(p, t).items()})
        if name in ["importance", "risk"]:
            out.update({f"{name}_binary_{k}": v for k, v in binary_metrics(p, t).items()})

    for name in ["volatility", "drawdown"]:
        p = np.concatenate(preds[name]) if preds[name] else np.array([], dtype=np.float32)
        t = np.concatenate(targets[name]) if targets[name] else np.array([], dtype=np.float32)
        m = np.concatenate(masks[name]) if masks[name] else np.array([], dtype=np.float32)
        valid = m > 0.5
        out.update({f"{name}_{k}": v for k, v in regression_metrics(p[valid], t[valid]).items()})
        out.update({f"{name}_binary_{k}": v for k, v in binary_metrics(p[valid], t[valid]).items()})
        out[f"{name}_rows_evaluated"] = int(valid.sum())

    out["rows_evaluated"] = int(len(np.concatenate(targets["impact"])) if targets["impact"] else 0)
    return out


def save_checkpoint(
    cfg: NewsConfig,
    model: NewsAnalystAttentionModel,
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
            "impact": TARGET_IMPACT_COLUMN,
            "importance": TARGET_IMPORTANCE_COLUMN,
            "risk": TARGET_RISK_COLUMN,
        },
    }
    torch.save(state, cfg.chunk_model_dir() / f"epoch_{epoch:03d}.pt")
    torch.save(state, cfg.latest_checkpoint_path())
    if is_best:
        torch.save(state, cfg.best_checkpoint_path())


def load_checkpoint_for_training(
    cfg: NewsConfig,
    model: NewsAnalystAttentionModel,
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


def load_model_for_inference(cfg: NewsConfig, checkpoint: str, device: torch.device) -> NewsAnalystAttentionModel:
    if checkpoint == "best":
        path = cfg.best_checkpoint_path()
    elif checkpoint == "latest":
        path = cfg.latest_checkpoint_path()
    else:
        path = Path(checkpoint)
        if not path.is_absolute():
            path = cfg.repo_root / path
    if not path.exists():
        raise FileNotFoundError(f"Missing News Analyst checkpoint: {path}")
    state = torch.load(path, map_location=device)
    model_cfg = state["model_config"]
    model = NewsAnalystAttentionModel(
        input_dim=int(model_cfg["input_dim"]),
        metadata_dim=int(model_cfg["metadata_dim"]),
        d_model=int(model_cfg["d_model"]),
        attention_heads=int(model_cfg["attention_heads"]),
        self_attention_layers=int(model_cfg["self_attention_layers"]),
        hidden_dims=list(model_cfg["hidden_dims"]),
        representation_dim=int(model_cfg["representation_dim"]),
        dropout=float(model_cfg["dropout"]),
    )
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()
    return model


def train_one_chunk(cfg: NewsConfig) -> Dict[str, Any]:
    cfg.resolve()
    if cfg.chunk_id not in APPROVED_SPLITS:
        raise ValueError(f"Unsupported chunk_id={cfg.chunk_id}")
    if cfg.torch_num_threads is not None and cfg.torch_num_threads > 0:
        torch.set_num_threads(int(cfg.torch_num_threads))

    ensure_dir(cfg.chunk_model_dir())
    ensure_dir(cfg.results_dir)
    ensure_dir(cfg.code_results_dir)
    ensure_dir(cfg.analyst_embeddings_dir)
    write_json(cfg.code_results_dir / f"chunk{cfg.chunk_id}_run_config.json", cfg.serialisable())

    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print("========== NEWS ANALYST TRAINING ==========")
    print(json.dumps(cfg.serialisable(), indent=2))
    print(f"[device] requested={cfg.device} resolved={device} cuda_available={torch.cuda.is_available()} torch_threads={torch.get_num_threads()}")

    train_labels, train_embeddings, train_groups = prepare_base_split(cfg, "train")
    train_indices = np.arange(len(train_groups), dtype=np.int64)
    train_indices = deterministic_subset(train_indices, cfg.max_train_groups, cfg.seed + 101)
    if len(train_indices) == 0:
        raise ValueError(f"No valid supervised training document groups for chunk{cfg.chunk_id}")

    if cfg.resume and cfg.preprocessor_path().exists():
        preprocessor = DocumentMetadataPreprocessor.load(cfg.preprocessor_path())
    else:
        preprocessor = DocumentMetadataPreprocessor(use_metadata_features=cfg.use_metadata_features).fit(train_groups, train_indices)
        preprocessor.save(cfg.preprocessor_path())

    train_meta = preprocessor.transform_groups(train_groups)
    train_bundle = SplitBundle("train", train_labels, train_embeddings, train_groups, train_meta, train_indices)
    val_bundle = prepare_split_bundle(cfg, "val", preprocessor, max_groups=cfg.max_val_groups)

    train_dataset = NewsDocumentDataset(train_bundle.embeddings, train_bundle.groups, train_bundle.group_indices, train_bundle.metadata_features, cfg.max_chunks_per_document, require_targets=True)
    val_dataset = NewsDocumentDataset(val_bundle.embeddings, val_bundle.groups, val_bundle.group_indices, val_bundle.metadata_features, cfg.max_chunks_per_document, require_targets=True)

    train_loader = make_loader(train_dataset, cfg.batch_size, shuffle=True, cfg=cfg, device=device)
    val_loader = make_loader(val_dataset, cfg.eval_batch_size, shuffle=False, cfg=cfg, device=device)

    model = NewsAnalystAttentionModel(
        input_dim=cfg.input_dim,
        metadata_dim=preprocessor.feature_dim,
        d_model=cfg.d_model,
        attention_heads=cfg.attention_heads,
        self_attention_layers=cfg.self_attention_layers,
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
            "train_groups": int(len(train_dataset)),
            "val_groups": int(len(val_dataset)),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items() if not isinstance(v, list)},
            "best_val_loss": best_val_loss,
            "improved": bool(improved),
            "no_improve": int(no_improve),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path(cfg), index=False)
        save_checkpoint(cfg, model, optimiser, epoch, best_val_loss, history, is_best=improved)
        print(
            f"[epoch {epoch}] train_loss={train_metrics['loss']:.6f} val_loss={val_loss:.6f} "
            f"impact_mae={val_metrics['impact_mae']:.6f} importance_mae={val_metrics['importance_mae']:.6f} "
            f"risk_mae={val_metrics['risk_mae']:.6f} improved={improved}"
        )

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
def run_prediction_export(cfg: NewsConfig, split: str, checkpoint: str = "best", max_groups: Optional[int] = None) -> Dict[str, Any]:
    cfg.resolve()
    if cfg.torch_num_threads is not None and cfg.torch_num_threads > 0:
        torch.set_num_threads(int(cfg.torch_num_threads))
    device = get_device(cfg.device)
    preprocessor = DocumentMetadataPreprocessor.load(cfg.preprocessor_path())
    labels, embeddings, groups = prepare_base_split(cfg, split)
    meta = preprocessor.transform_groups(groups)
    group_indices = np.arange(len(groups), dtype=np.int64)
    group_indices = deterministic_subset(group_indices, max_groups, cfg.seed + 301) if max_groups is not None else group_indices

    dataset = NewsDocumentDataset(embeddings, groups, group_indices, meta, cfg.max_chunks_per_document, require_targets=False)
    loader = make_loader(dataset, cfg.eval_batch_size, shuffle=False, cfg=cfg, device=device)
    model = load_model_for_inference(cfg, checkpoint=checkpoint, device=device)

    ensure_dir(cfg.results_dir)
    ensure_dir(cfg.analyst_embeddings_dir)

    n_rows = len(dataset)
    group_pos_to_output_pos = {int(g): i for i, g in enumerate(group_indices.tolist())}
    repr_out = np.lib.format.open_memmap(
        analyst_embedding_path(cfg, split),
        mode="w+",
        dtype=np.float32,
        shape=(n_rows, cfg.representation_dim),
    )

    pred_impact = np.full(n_rows, np.nan, dtype=np.float32)
    pred_importance = np.full(n_rows, np.nan, dtype=np.float32)
    pred_risk = np.full(n_rows, np.nan, dtype=np.float32)
    pred_vol = np.full(n_rows, np.nan, dtype=np.float32)
    pred_dd = np.full(n_rows, np.nan, dtype=np.float32)
    attention_rows: List[Dict[str, Any]] = []

    model.eval()
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        metadata = batch["metadata"].to(device, non_blocking=True)
        group_pos_batch = batch["group_pos"].cpu().numpy().astype(np.int64)
        row_idx_batch = batch["row_indices"].cpu().numpy().astype(np.int64)
        chunk_idx_batch = batch["chunk_indices"].cpu().numpy().astype(np.int64)
        mask_np = batch["mask"].cpu().numpy().astype(bool)

        with autocast(enabled=(cfg.mixed_precision and device.type == "cuda")):
            outputs = model(x, mask, metadata)
        impact = outputs["event_impact_score"].float().cpu().numpy()
        importance = outputs["news_importance_score"].float().cpu().numpy()
        risk = outputs["risk_relevance_score"].float().cpu().numpy()
        vol = outputs["volatility_spike_score"].float().cpu().numpy()
        dd = outputs["drawdown_risk_score"].float().cpu().numpy()
        rep = outputs["news_embedding"].float().cpu().numpy().astype(np.float32)
        attn = outputs["attention_weights"].float().cpu().numpy()  # (B, H, S)

        for j, group_pos in enumerate(group_pos_batch):
            out_pos = group_pos_to_output_pos[int(group_pos)]
            pred_impact[out_pos] = impact[j]
            pred_importance[out_pos] = importance[j]
            pred_risk[out_pos] = risk[j]
            pred_vol[out_pos] = vol[j]
            pred_dd[out_pos] = dd[j]
            repr_out[out_pos, :] = rep[j]

            group = groups[int(group_pos)]
            valid_positions = np.flatnonzero(mask_np[j])
            for pos in valid_positions:
                weights = attn[j, :, pos]
                row = {
                    "group_output_index": out_pos,
                    "group_id": group.group_id,
                    "doc_id": group.doc_id,
                    "accession": group.accession,
                    "ticker": group.ticker,
                    "filing_date": group.filing_date,
                    "metadata_row_index": int(row_idx_batch[j, pos]),
                    "chunk_index": int(chunk_idx_batch[j, pos]),
                    "attention_mean": float(np.mean(weights)),
                    "attention_max_head": float(np.max(weights)),
                }
                for h in range(weights.shape[0]):
                    row[f"attention_head_{h}"] = float(weights[h])
                attention_rows.append(row)

    del repr_out

    pred_rows: List[Dict[str, Any]] = []
    selected_groups = [groups[int(i)] for i in group_indices]
    for i, group in enumerate(selected_groups):
        pred_rows.append(
            {
                "group_output_index": i,
                "group_id": group.group_id,
                "doc_id": group.doc_id,
                "accession": group.accession,
                "ticker": group.ticker,
                "cik": group.cik,
                "filing_date": group.filing_date,
                "year": group.year,
                "form_type": group.form_type,
                "source_name": group.source_name,
                "n_chunks_original": group.n_chunks_original,
                "total_word_count": group.total_word_count,
                "target_news_event_impact": group.target_impact,
                "target_news_importance": group.target_importance,
                "target_risk_relevance": group.target_risk,
                "target_volatility_spike": group.target_volatility,
                "target_drawdown_risk": group.target_drawdown,
                "predicted_news_event_impact": pred_impact[i],
                "predicted_news_importance": pred_importance[i],
                "predicted_risk_relevance": pred_risk[i],
                "predicted_volatility_spike": pred_vol[i],
                "predicted_drawdown_risk": pred_dd[i],
                "predicted_news_uncertainty": float(1.0 - max(pred_importance[i], pred_risk[i])),
                "news_embedding_file": str(analyst_embedding_path(cfg, split)),
                "attention_file": str(attention_path(cfg, split)),
            }
        )

    out_df = pd.DataFrame(pred_rows)
    out_df.to_csv(predictions_path(cfg, split), index=False)
    pd.DataFrame(attention_rows).to_csv(attention_path(cfg, split), index=False)

    metrics: Dict[str, Any] = {
        "created_at": now_stamp(),
        "chunk_id": cfg.chunk_id,
        "split": split,
        "checkpoint": checkpoint,
        "documents_predicted": int(len(out_df)),
        "attention_rows": int(len(attention_rows)),
        "predictions_file": str(predictions_path(cfg, split)),
        "attention_file": str(attention_path(cfg, split)),
        "news_embedding_file": str(analyst_embedding_path(cfg, split)),
    }
    metrics.update({f"impact_{k}": v for k, v in regression_metrics(out_df["predicted_news_event_impact"].to_numpy(dtype=np.float32), out_df["target_news_event_impact"].to_numpy(dtype=np.float32)).items()})
    metrics.update({f"importance_{k}": v for k, v in regression_metrics(out_df["predicted_news_importance"].to_numpy(dtype=np.float32), out_df["target_news_importance"].to_numpy(dtype=np.float32)).items()})
    metrics.update({f"risk_{k}": v for k, v in regression_metrics(out_df["predicted_risk_relevance"].to_numpy(dtype=np.float32), out_df["target_risk_relevance"].to_numpy(dtype=np.float32)).items()})
    metrics.update({f"importance_binary_{k}": v for k, v in binary_metrics(out_df["predicted_news_importance"].to_numpy(dtype=np.float32), out_df["target_news_importance"].to_numpy(dtype=np.float32)).items()})
    metrics.update({f"risk_binary_{k}": v for k, v in binary_metrics(out_df["predicted_risk_relevance"].to_numpy(dtype=np.float32), out_df["target_risk_relevance"].to_numpy(dtype=np.float32)).items()})
    write_json(metrics_path(cfg, split), metrics)
    print(f"[predict] split={split} documents={len(out_df):,} predictions={predictions_path(cfg, split)} attention={attention_path(cfg, split)} embeddings={analyst_embedding_path(cfg, split)}")
    return metrics


def inspect_labels(cfg: NewsConfig) -> Dict[str, Any]:
    cfg.resolve()
    rows: List[Dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        labels = read_label_file(cfg, split)
        emb = load_embeddings(cfg, split, mmap_mode="r")
        validate_alignment(cfg, split, labels, emb)
        valid = valid_news_rows(labels)
        groups = build_document_groups(labels, valid)
        lens = np.array([g.n_chunks_original for g in groups], dtype=np.int64) if groups else np.array([], dtype=np.int64)
        rows.append(
            {
                "chunk_id": cfg.chunk_id,
                "split": split,
                "label_rows": int(len(labels)),
                "embedding_shape": list(emb.shape),
                "valid_supervised_rows": int(len(valid)),
                "document_groups": int(len(groups)),
                "mean_chunks_per_doc": float(np.mean(lens)) if len(lens) else float("nan"),
                "max_chunks_per_doc": int(np.max(lens)) if len(lens) else 0,
                "label_available_rows": int(bool_series(labels["label_available"]).sum()),
                "missing_impact_rows": int(pd.to_numeric(labels[TARGET_IMPACT_COLUMN], errors="coerce").isna().sum()),
                "missing_importance_rows": int(pd.to_numeric(labels[TARGET_IMPORTANCE_COLUMN], errors="coerce").isna().sum()),
                "missing_risk_rows": int(pd.to_numeric(labels[TARGET_RISK_COLUMN], errors="coerce").isna().sum()),
                "ticker_in_market_panel_rows": int(bool_series(labels["ticker_in_market_panel"]).sum()),
            }
        )
    out = {"created_at": now_stamp(), "rows": rows}
    ensure_dir(cfg.results_dir)
    write_json(cfg.results_dir / f"chunk{cfg.chunk_id}_input_inspection.json", out)
    print(pd.DataFrame(rows).to_string(index=False))
    return out


def apply_best_params(cfg: NewsConfig, best_params: Dict[str, Any]) -> NewsConfig:
    cfg.learning_rate = float(best_params.get("learning_rate", cfg.learning_rate))
    cfg.weight_decay = float(best_params.get("weight_decay", cfg.weight_decay))
    cfg.dropout = float(best_params.get("dropout", cfg.dropout))
    cfg.batch_size = int(best_params.get("batch_size", cfg.batch_size))
    cfg.eval_batch_size = int(best_params.get("eval_batch_size", max(cfg.batch_size * 2, cfg.eval_batch_size)))
    cfg.epochs = int(best_params.get("epochs", cfg.epochs))
    cfg.early_stop_patience = int(best_params.get("early_stop_patience", cfg.early_stop_patience))
    cfg.gradient_clip = float(best_params.get("gradient_clip", cfg.gradient_clip))
    cfg.d_model = int(best_params.get("d_model", cfg.d_model))
    cfg.attention_heads = int(best_params.get("attention_heads", cfg.attention_heads))
    cfg.self_attention_layers = int(best_params.get("self_attention_layers", cfg.self_attention_layers))
    if "hidden_dims" in best_params:
        cfg.hidden_dims = list(best_params["hidden_dims"])
    cfg.representation_dim = int(best_params.get("representation_dim", cfg.representation_dim))
    cfg.max_chunks_per_document = int(best_params.get("max_chunks_per_document", cfg.max_chunks_per_document))
    cfg.use_metadata_features = bool(best_params.get("use_metadata_features", cfg.use_metadata_features))
    cfg.impact_loss_weight = float(best_params.get("impact_loss_weight", cfg.impact_loss_weight))
    cfg.importance_loss_weight = float(best_params.get("importance_loss_weight", cfg.importance_loss_weight))
    cfg.risk_loss_weight = float(best_params.get("risk_loss_weight", cfg.risk_loss_weight))
    cfg.volatility_loss_weight = float(best_params.get("volatility_loss_weight", cfg.volatility_loss_weight))
    cfg.drawdown_loss_weight = float(best_params.get("drawdown_loss_weight", cfg.drawdown_loss_weight))
    return cfg.resolve()


class NewsAnalystHPO:
    def __init__(self, base_cfg: NewsConfig, trials: int = 30, study_name: Optional[str] = None):
        if optuna is None:
            raise ImportError("optuna is required for HPO. Install it in the environment before running hpo commands.")
        self.base_cfg = base_cfg.resolve()
        self.trials = int(trials)
        self.study_name = study_name or f"news_analyst_chunk{self.base_cfg.chunk_id}"
        ensure_dir(self.base_cfg.code_results_dir / "hpo")
        self.storage = f"sqlite:///{self.base_cfg.hpo_storage_path()}"

    def suggest_hidden_dims(self, trial: Any) -> List[int]:
        choice = trial.suggest_categorical("hidden_arch", ["128,64", "256,128", "256,128,64", "384,128", "512,256,128"])
        return parse_hidden_dims(choice)

    def objective(self, trial: Any) -> float:
        cfg = NewsConfig(**asdict(self.base_cfg))
        cfg.resolve()
        cfg.seed = int(self.base_cfg.seed + trial.number)
        cfg.learning_rate = trial.suggest_float("learning_rate", 2e-5, 8e-4, log=True)
        cfg.weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True)
        cfg.dropout = trial.suggest_float("dropout", 0.05, 0.35)
        cfg.batch_size = trial.suggest_categorical("batch_size", [32, 64, 96, 128, 192])
        cfg.eval_batch_size = max(cfg.batch_size * 2, 64)
        cfg.epochs = trial.suggest_int("epochs", int(getattr(self.base_cfg, "hpo_epochs_min", 8)), int(getattr(self.base_cfg, "hpo_epochs_max", 25)))
        cfg.early_stop_patience = trial.suggest_int("early_stop_patience", 4, 10)
        cfg.gradient_clip = trial.suggest_float("gradient_clip", 0.5, 2.0)
        attention_config = trial.suggest_categorical(
            "attention_config",
            [
                "96x2", "96x3", "96x4", "96x6",
                "128x2", "128x4", "128x8",
                "192x2", "192x3", "192x4", "192x6", "192x8",
                "256x2", "256x4", "256x8",
            ],
        )
        d_text, h_text = str(attention_config).split("x", 1)
        cfg.d_model = int(d_text)
        cfg.attention_heads = int(h_text)
        cfg.self_attention_layers = trial.suggest_int("self_attention_layers", 0, 2)
        cfg.hidden_dims = self.suggest_hidden_dims(trial)
        cfg.representation_dim = trial.suggest_categorical("representation_dim", [64, 128])
        cfg.max_chunks_per_document = trial.suggest_categorical("max_chunks_per_document", [16, 32, 64, 96])
        cfg.use_metadata_features = trial.suggest_categorical("use_metadata_features", [True, True, False])
        cfg.impact_loss_weight = trial.suggest_float("impact_loss_weight", 0.75, 1.5)
        cfg.importance_loss_weight = trial.suggest_float("importance_loss_weight", 0.4, 1.25)
        cfg.risk_loss_weight = trial.suggest_float("risk_loss_weight", 0.4, 1.25)
        cfg.volatility_loss_weight = trial.suggest_float("volatility_loss_weight", 0.0, 0.6)
        cfg.drawdown_loss_weight = trial.suggest_float("drawdown_loss_weight", 0.0, 0.6)

        cfg.max_train_groups = self.base_cfg.max_train_groups
        cfg.max_val_groups = self.base_cfg.max_val_groups
        cfg.models_dir = self.base_cfg.models_dir / "hpo" / self.study_name / f"trial_{trial.number:04d}"
        cfg.results_dir = self.base_cfg.results_dir / "hpo" / self.study_name / f"trial_{trial.number:04d}"
        cfg.code_results_dir = self.base_cfg.code_results_dir / "hpo" / self.study_name / f"trial_{trial.number:04d}"
        cfg.env_file = Path("__ignore_env_for_hpo__.env")
        cfg.resume = False

        try:
            summary = train_one_chunk(cfg)
            value = float(summary["best_val_loss"])
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return value

    def run(self) -> Dict[str, Any]:
        sampler = optuna.samplers.TPESampler(seed=self.base_cfg.seed, multivariate=True)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True,
        )
        study.optimize(self.objective, n_trials=self.trials, gc_after_trial=True, catch=(RuntimeError, ValueError))

        best_params: Dict[str, Any] = {}
        best_value: Optional[float] = None
        try:
            best_value = float(study.best_value)
            best_params = dict(study.best_params)
            if "hidden_arch" in best_params:
                best_params["hidden_dims"] = parse_hidden_dims(str(best_params["hidden_arch"]))
            if "attention_config" in best_params:
                d_text, h_text = str(best_params["attention_config"]).split("x", 1)
                best_params["d_model"] = int(d_text)
                best_params["attention_heads"] = int(h_text)
        except Exception:
            best_value = None
            best_params = {}

        best = {
            "study_name": self.study_name,
            "chunk_id": self.base_cfg.chunk_id,
            "best_value": best_value,
            "best_params": best_params,
            "trials": len(study.trials),
            "storage": self.storage,
            "created_at": now_stamp(),
        }
        out = self.base_cfg.best_params_path()
        write_json(out, best)
        study.trials_dataframe().to_csv(self.base_cfg.code_results_dir / "hpo" / f"{self.study_name}_trials.csv", index=False)
        print(json.dumps(best, indent=2, default=str))
        return best


def load_best_params(cfg: NewsConfig) -> Dict[str, Any]:
    cfg.resolve()
    path = cfg.best_params_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing HPO best params file: {path}. Run hpo or hpo-all first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    params = data.get("best_params", {})
    if not params:
        raise ValueError(f"Best params file has no best_params: {path}")
    return params


def build_cfg_from_args(args: argparse.Namespace, chunk_id: Optional[int] = None) -> NewsConfig:
    cfg = NewsConfig()
    cfg.repo_root = Path(args.repo_root)
    cfg.env_file = Path(args.env_file)
    cfg.resolve()

    if chunk_id is not None:
        cfg.chunk_id = int(chunk_id)
    elif hasattr(args, "chunk") and args.chunk is not None:
        cfg.chunk_id = int(args.chunk)

    for attr, arg_name in [
        ("embeddings_dir", "embeddings_dir"),
        ("labels_dir", "labels_dir"),
        ("models_dir", "models_dir"),
        ("results_dir", "results_dir"),
        ("code_results_dir", "code_results_dir"),
        ("analyst_embeddings_dir", "analyst_embeddings_dir"),
    ]:
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(cfg, attr, resolve_repo_path(cfg.repo_root, Path(value)))

    cfg.device = args.device
    cfg.seed = int(args.seed)
    cfg.torch_num_threads = args.torch_num_threads
    cfg.num_workers = int(args.num_workers)
    cfg.batch_size = int(args.batch_size)
    cfg.eval_batch_size = int(args.eval_batch_size)
    cfg.epochs = int(args.epochs)
    cfg.learning_rate = float(args.learning_rate)
    cfg.weight_decay = float(args.weight_decay)
    cfg.dropout = float(args.dropout)
    cfg.gradient_clip = float(args.gradient_clip)
    cfg.early_stop_patience = int(args.early_stop_patience)
    cfg.mixed_precision = not bool(args.no_mixed_precision)
    cfg.use_metadata_features = not bool(args.no_metadata_features)
    cfg.resume = bool(args.resume)
    cfg.d_model = int(args.d_model)
    cfg.attention_heads = int(args.attention_heads)
    cfg.self_attention_layers = int(args.self_attention_layers)
    cfg.hidden_dims = parse_hidden_dims(args.hidden_dims)
    cfg.representation_dim = int(args.representation_dim)
    cfg.max_chunks_per_document = int(args.max_chunks_per_document)
    cfg.max_train_groups = args.max_train_groups
    cfg.max_val_groups = args.max_val_groups
    cfg.max_test_groups = args.max_test_groups

    if hasattr(args, "hpo_max_train_groups") and args.hpo_max_train_groups is not None:
        cfg.max_train_groups = args.hpo_max_train_groups
    if hasattr(args, "hpo_max_val_groups") and args.hpo_max_val_groups is not None:
        cfg.max_val_groups = args.hpo_max_val_groups
    if hasattr(args, "hpo_epochs_min"):
        setattr(cfg, "hpo_epochs_min", int(args.hpo_epochs_min))
    if hasattr(args, "hpo_epochs_max"):
        setattr(cfg, "hpo_epochs_max", int(args.hpo_epochs_max))

    return cfg.resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and run the supervised fin-glassbox News Analyst on real FinBERT embeddings and real market labels.")
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
        sp.add_argument("--torch-num-threads", type=int, default=None)
        sp.add_argument("--num-workers", type=int, default=0)
        sp.add_argument("--batch-size", type=int, default=96)
        sp.add_argument("--eval-batch-size", type=int, default=192)
        sp.add_argument("--epochs", type=int, default=40)
        sp.add_argument("--learning-rate", type=float, default=1e-4)
        sp.add_argument("--weight-decay", type=float, default=1e-4)
        sp.add_argument("--dropout", type=float, default=0.15)
        sp.add_argument("--gradient-clip", type=float, default=1.0)
        sp.add_argument("--early-stop-patience", type=int, default=8)
        sp.add_argument("--no-mixed-precision", action="store_true")
        sp.add_argument("--no-metadata-features", action="store_true")
        sp.add_argument("--resume", action="store_true")
        sp.add_argument("--d-model", type=int, default=128)
        sp.add_argument("--attention-heads", type=int, default=4)
        sp.add_argument("--self-attention-layers", type=int, default=1)
        sp.add_argument("--hidden-dims", default="128,64")
        sp.add_argument("--representation-dim", type=int, default=128)
        sp.add_argument("--max-chunks-per-document", type=int, default=64)
        sp.add_argument("--max-train-groups", type=int, default=None)
        sp.add_argument("--max-val-groups", type=int, default=None)
        sp.add_argument("--max-test-groups", type=int, default=None)

    sp_train = sub.add_parser("train", help="Train one chronological chunk with explicitly supplied hyperparameters")
    add_common(sp_train)
    sp_train.add_argument("--chunk", type=int, choices=[1, 2, 3], required=True)

    sp_train_all = sub.add_parser("train-all", help="Train News Analysts for multiple chunks with explicitly supplied hyperparameters")
    add_common(sp_train_all)
    sp_train_all.add_argument("--chunks", default="1,2,3")

    sp_train_best = sub.add_parser("train-best", help="Train one chunk using saved HPO best parameters")
    add_common(sp_train_best)
    sp_train_best.add_argument("--chunk", type=int, choices=[1, 2, 3], required=True)

    sp_train_best_all = sub.add_parser("train-best-all", help="Train multiple chunks using saved HPO best parameters")
    add_common(sp_train_best_all)
    sp_train_best_all.add_argument("--chunks", default="1,2,3")

    sp_hpo = sub.add_parser("hpo", help="Run Optuna/TPE HPO for one chronological chunk")
    add_common(sp_hpo)
    sp_hpo.add_argument("--chunk", type=int, choices=[1, 2, 3], required=True)
    sp_hpo.add_argument("--trials", type=int, default=30)
    sp_hpo.add_argument("--study-name", default=None)
    sp_hpo.add_argument("--hpo-max-train-groups", type=int, default=None)
    sp_hpo.add_argument("--hpo-max-val-groups", type=int, default=None)
    sp_hpo.add_argument("--hpo-epochs-min", type=int, default=8)
    sp_hpo.add_argument("--hpo-epochs-max", type=int, default=25)

    sp_hpo_all = sub.add_parser("hpo-all", help="Run Optuna/TPE HPO for multiple chunks")
    add_common(sp_hpo_all)
    sp_hpo_all.add_argument("--chunks", default="1,2,3")
    sp_hpo_all.add_argument("--trials", type=int, default=30)
    sp_hpo_all.add_argument("--hpo-max-train-groups", type=int, default=None)
    sp_hpo_all.add_argument("--hpo-max-val-groups", type=int, default=None)
    sp_hpo_all.add_argument("--hpo-epochs-min", type=int, default=8)
    sp_hpo_all.add_argument("--hpo-epochs-max", type=int, default=25)

    sp_inspect = sub.add_parser("inspect", help="Inspect label/embedding availability, alignment, and document grouping")
    add_common(sp_inspect)
    sp_inspect.add_argument("--chunk", type=int, choices=[1, 2, 3], required=True)

    sp_predict = sub.add_parser("predict", help="Export document predictions, attention weights, and news analyst embeddings")
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
    elif args.command == "train-best":
        cfg = build_cfg_from_args(args)
        params = load_best_params(cfg)
        cfg = apply_best_params(cfg, params)
        train_one_chunk(cfg)
    elif args.command == "train-best-all":
        summaries = []
        for chunk_id in parse_int_list(args.chunks):
            cfg = build_cfg_from_args(args, chunk_id=chunk_id)
            params = load_best_params(cfg)
            cfg = apply_best_params(cfg, params)
            summaries.append(train_one_chunk(cfg))
        print(json.dumps(summaries, indent=2, default=str))
    elif args.command == "hpo":
        cfg = build_cfg_from_args(args)
        NewsAnalystHPO(cfg, trials=args.trials, study_name=args.study_name).run()
    elif args.command == "hpo-all":
        outputs = []
        for chunk_id in parse_int_list(args.chunks):
            cfg = build_cfg_from_args(args, chunk_id=chunk_id)
            outputs.append(NewsAnalystHPO(cfg, trials=args.trials, study_name=f"news_analyst_chunk{chunk_id}").run())
        print(json.dumps(outputs, indent=2, default=str))
    elif args.command == "inspect":
        cfg = build_cfg_from_args(args)
        inspect_labels(cfg)
    elif args.command == "predict":
        cfg = build_cfg_from_args(args)
        max_groups = {"train": cfg.max_train_groups, "val": cfg.max_val_groups, "test": cfg.max_test_groups}[args.split]
        metrics = run_prediction_export(cfg, split=args.split, checkpoint=args.checkpoint, max_groups=max_groups)
        print(json.dumps(metrics, indent=2, default=str))
    elif args.command == "predict-all":
        outputs = []
        for chunk_id in parse_int_list(args.chunks):
            cfg = build_cfg_from_args(args, chunk_id=chunk_id)
            for split in parse_split_list(args.splits):
                max_groups = {"train": cfg.max_train_groups, "val": cfg.max_val_groups, "test": cfg.max_test_groups}[split]
                outputs.append(run_prediction_export(cfg, split=split, checkpoint=args.checkpoint, max_groups=max_groups))
        print(json.dumps(outputs, indent=2, default=str))
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# Optional command for checking syntax:
# python -m py_compile code/analysts/news_analyst.py

# Recommended HPO-first workflow examples:
# python code/analysts/news_analyst.py inspect --repo-root ~/fin-glassbox --chunk 1 --device cpu
# python code/analysts/news_analyst.py hpo --repo-root ~/fin-glassbox --chunk 1 --device cpu --trials 5 --hpo-max-train-groups 5000 --hpo-max-val-groups 1500
# python code/analysts/news_analyst.py train-best --repo-root ~/fin-glassbox --chunk 1 --device cpu
# python code/analysts/news_analyst.py predict --repo-root ~/fin-glassbox --chunk 1 --split val --device cpu --checkpoint best

# CPU-safe HPO smoke test using real embeddings and real labels, but only a subset:
# python code/analysts/news_analyst.py hpo --repo-root ~/fin-glassbox --chunk 1 --device cpu --trials 3 --hpo-max-train-groups 5000 --hpo-max-val-groups 1500 --hpo-epochs-min 2 --hpo-epochs-max 5

# Then train from the best HPO result:
# python code/analysts/news_analyst.py train-best --repo-root ~/fin-glassbox --chunk 1 --device cpu

# Real HPO-first runner when gpu compute is available:
# python code/analysts/news_analyst.py hpo-all --repo-root ~/fin-glassbox --chunks 1,2,3 --trials 30 --device cuda

# Then train final models using the saved best parameters using gpu if you wish:
# python code/analysts/news_analyst.py train-best-all --repo-root ~/fin-glassbox --chunks 1,2,3 --device cuda

# Then export document predictions, attention traces, and 128-dim news analyst embeddings(gpu based but can be done on cpu):
# python code/analysts/news_analyst.py predict-all --repo-root ~/fin-glassbox --chunks 1,2,3 --splits train,val,test --checkpoint best --device cuda

# CPU tuning examples when no GPU is available:
# python code/analysts/news_analyst.py hpo --repo-root ~/fin-glassbox --chunk 1 --device cpu --torch-num-threads 8 --num-workers 4 --trials 5 --hpo-max-train-groups 5000 --hpo-max-val-groups 1500
# python code/analysts/news_analyst.py train-best --repo-root ~/fin-glassbox --chunk 1 --device cpu --torch-num-threads 8 --num-workers 4
# -----------------------------------------------------------------------------
