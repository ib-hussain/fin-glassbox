"""
FinBERT encoder for the fin-glassbox project.

Use as both:
1. importable module/header
2. standalone CLI script

Core capabilities:
- Inspect SEC FinBERT chunk CSV.
- Chronological chunk split management.
- Domain-adaptive MLM fine-tuning of FinBERT on SEC filing chunks.
- Resumeable checkpoints after every epoch.
- Optuna/TPE hyperparameter search with SQLite persistence.
- Export frozen and unfrozen HuggingFace model folders.
- Extract honest 768-dim FinBERT embeddings from the fine-tuned base model.
- Fit PCA/IncrementalPCA on TRAIN embeddings only.
- Project 768 -> 256 and save final .npy embeddings + aligned metadata .csv.
- No Parquet.

Important modelling note:
- During MLM fine-tuning, only the FinBERT MLM model is trained.
- A random neural 768->256 projection is NOT used for final MLM embeddings.
- For MLM-stage embeddings, 256-dim outputs are produced by PCA fitted on train-only 768-dim embeddings.
- Later supervised fine-tuning can add a trainable 768->256 projection and task heads after labels are built.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import pickle
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Keep Transformers from importing TensorFlow in this PyTorch-only training file.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

try:
    import optuna
except Exception:
    optuna = None

try:
    from sklearn.decomposition import IncrementalPCA
except Exception:
    IncrementalPCA = None


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def now_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


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
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value

    return values


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def get_device(preferred: str = "cuda") -> torch.device:
    if preferred.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


def cuda_memory_summary(prefix: str = "[cuda]") -> None:
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"{prefix} allocated={allocated:.2f}GB reserved={reserved:.2f}GB")


def path_to_str_dict(obj: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in obj.items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


# -----------------------------------------------------------------------------
# Chronological splits
# -----------------------------------------------------------------------------


@dataclass
class ChronoSplit:
    chunk_id: int
    train_years: List[int]
    val_years: List[int]
    test_years: List[int]


APPROVED_SPLITS: Dict[int, ChronoSplit] = {
    1: ChronoSplit(
        chunk_id=1,
        train_years=[2000, 2001, 2002, 2003, 2004],
        val_years=[2005],
        test_years=[2006],
    ),
    2: ChronoSplit(
        chunk_id=2,
        train_years=list(range(2007, 2015)),
        val_years=[2015],
        test_years=[2016],
    ),
    3: ChronoSplit(
        chunk_id=3,
        train_years=list(range(2017, 2023)),
        val_years=[2023],
        test_years=[2024],
    ),
}


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class FinBERTConfig:
    repo_root: Path = Path(".")
    env_file: Path = Path(".env")

    data_path: Path = Path("data")
    outputs_path: Path = Path("outputs")
    embeddings_path: Path = Path("outputs/embeddings/FinBERT")
    models_path: Path = Path("outputs/models/FinBERT")
    results_path: Path = Path("outputs/results/FinBERT")
    code_results_path: Path = Path("outputs/codeResults/FinBERT")

    dataset_csv: Path = Path("final/filings_finbert_chunks_balanced_25y_cap40000.csv")
    label_csv: Optional[Path] = None

    base_model_name: str = "ProsusAI/finbert"
    max_length: int = 512
    base_embedding_dim: int = 768
    projection_dim: int = 256
    dropout: float = 0.1
    freeze_base: bool = False

    seed: int = 42
    processor: str = "cuda"

    # RTX 3090 Ti / 24 GB safer defaults.
    num_workers: int = 6
    batch_size: int = 24
    eval_batch_size: int = 64

    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_clip: float = 1.0
    mlm_probability: float = 0.15
    early_stop_patience: int = 5
    gradient_accumulation_steps: int = 1
    fp16: bool = True

    save_every_epoch: bool = True
    max_rows: Optional[int] = None
    sample_frac: Optional[float] = None
    sample_mode: str = "balanced-year"  # head, random, balanced-year

    debug_mode: int = 0
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pca_batch_size: int = 4096

    def resolve(self) -> "FinBERTConfig":
        env = read_env_file(self.repo_root / self.env_file)

        if env.get("dataPathGlobal"):
            self.data_path = Path(env["dataPathGlobal"])

        if env.get("outputsPathGlobal"):
            self.outputs_path = Path(env["outputsPathGlobal"])

        if env.get("FinBERTembeddingsPath"):
            self.embeddings_path = Path(env["FinBERTembeddingsPath"])
        elif env.get("embeddingsPathGlobal"):
            self.embeddings_path = Path(env["embeddingsPathGlobal"]) / "FinBERT"

        if env.get("modelsPathGlobal"):
            self.models_path = Path(env["modelsPathGlobal"]) / "FinBERT"

        if env.get("resultsPathGlobal"):
            self.results_path = Path(env["resultsPathGlobal"]) / "FinBERT"

        if env.get("codeOutputsPathGlobal"):
            self.code_results_path = Path(env["codeOutputsPathGlobal"]) / "FinBERT"

        if env.get("PROCESSOR"):
            self.processor = env["PROCESSOR"].strip().lower()

        if env.get("DEBUG_MODE"):
            try:
                self.debug_mode = int(env["DEBUG_MODE"])
            except ValueError:
                self.debug_mode = 0

        self.dataset_csv = Path(self.dataset_csv)
        if not self.dataset_csv.is_absolute():
            self.dataset_csv = self.repo_root / self.dataset_csv

        if self.label_csv is not None:
            self.label_csv = Path(self.label_csv)
            if not self.label_csv.is_absolute():
                self.label_csv = self.repo_root / self.label_csv

        for attr in [
            "outputs_path",
            "embeddings_path",
            "models_path",
            "results_path",
            "code_results_path",
        ]:
            p = getattr(self, attr)
            if not p.is_absolute():
                setattr(self, attr, self.repo_root / p)

        return self

    def serializable(self) -> Dict[str, Any]:
        return path_to_str_dict(asdict(self))

    def save(self, path: Path) -> None:
        ensure_dir(path.parent)
        path.write_text(json.dumps(self.serializable(), indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------


class SECChunkTextDataset(Dataset):
    REQUIRED_COLUMNS = [
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
        "text",
    ]

    def __init__(
        self,
        csv_path: Path,
        years: Optional[Sequence[int]] = None,
        max_rows: Optional[int] = None,
        sample_frac: Optional[float] = None,
        seed: int = 42,
        sample_mode: str = "balanced-year",
    ):
        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path, usecols=self.REQUIRED_COLUMNS)

        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df["year"] = df["year"].astype(int)

        if years is not None:
            years_set = set(int(y) for y in years)
            df = df[df["year"].isin(years_set)]

        df = df.dropna(subset=["text", "chunk_id", "doc_id", "year"])

        df = self._sample(
            df=df,
            years=years,
            max_rows=max_rows,
            sample_frac=sample_frac,
            seed=seed,
            sample_mode=sample_mode,
        )

        df = df.reset_index(drop=True)
        self.df = df

    @staticmethod
    def _sample(
        df: pd.DataFrame,
        years: Optional[Sequence[int]],
        max_rows: Optional[int],
        sample_frac: Optional[float],
        seed: int,
        sample_mode: str,
    ) -> pd.DataFrame:
        if sample_frac is not None and 0 < sample_frac < 1:
            return df.sample(frac=sample_frac, random_state=seed)

        if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
            return df

        if sample_mode == "head":
            return df.head(max_rows)

        if sample_mode == "random":
            return df.sample(n=max_rows, random_state=seed)

        if sample_mode == "balanced-year" and years is not None:
            year_list = sorted(set(int(y) for y in years))
            per_year = int(math.ceil(max_rows / max(1, len(year_list))))
            parts: List[pd.DataFrame] = []

            for y in year_list:
                sub = df[df["year"] == y]
                if len(sub) <= per_year:
                    parts.append(sub)
                else:
                    parts.append(sub.sample(n=per_year, random_state=seed + y))

            out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed)
            if len(out) > max_rows:
                out = out.head(max_rows)
            return out

        return df.sample(n=max_rows, random_state=seed)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        return {
            "text": str(row["text"]),
            "chunk_id": str(row["chunk_id"]),
            "doc_id": str(row["doc_id"]),
            "year": int(row["year"]),
            "form_type": str(row["form_type"]),
            "cik": str(row["cik"]),
            "filing_date": str(row["filing_date"]),
            "accession": str(row["accession"]),
            "source_name": str(row["source_name"]),
            "chunk_index": int(row["chunk_index"]),
            "word_count": int(row["word_count"]),
        }

    def metadata_rows(self) -> pd.DataFrame:
        return self.df.drop(columns=["text"]).copy()


class TokenizedMLMDataset(Dataset):
    def __init__(self, base_dataset: SECChunkTextDataset, tokenizer: AutoTokenizer, max_length: int):
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.base_dataset[idx]
        encoded = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_special_tokens_mask=True,
        )
        return {k: torch.tensor(v, dtype=torch.long) for k, v in encoded.items()}


class TokenizedEmbeddingDataset(Dataset):
    def __init__(self, base_dataset: SECChunkTextDataset, tokenizer: AutoTokenizer, max_length: int):
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.base_dataset[idx]
        encoded = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None,
        )

        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "metadata": {k: v for k, v in item.items() if k != "text"},
        }


def embedding_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "metadata": [x["metadata"] for x in batch],
    }


def make_loader_kwargs(cfg: FinBERTConfig, device: torch.device) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "num_workers": cfg.num_workers,
        "pin_memory": device.type == "cuda",
    }

    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = cfg.persistent_workers
        kwargs["prefetch_factor"] = cfg.prefetch_factor

    return kwargs


# -----------------------------------------------------------------------------
# Optional supervised-stage projection wrapper
# -----------------------------------------------------------------------------


class FinBERTProjectedEncoder(nn.Module):
    """
    Future supervised-stage encoder: FinBERT base + trainable 768->256 projection.

    Do NOT use this random projection for MLM-only embedding generation.
    MLM-stage 256-dim embeddings are produced via train-only PCA.
    """

    def __init__(
        self,
        base_model_name: str,
        projection_dim: int = 256,
        dropout: float = 0.1,
        freeze_base: bool = False,
    ):
        super().__init__()

        self.base_model_name = base_model_name
        self.config = AutoConfig.from_pretrained(base_model_name)
        self.bert = AutoModel.from_pretrained(base_model_name, add_pooling_layer=False)

        hidden = int(self.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden, projection_dim)
        self.layer_norm = nn.LayerNorm(projection_dim)

        if freeze_base:
            freeze_module(self.bert)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = mean_pool(out.last_hidden_state, attention_mask)
        emb = self.projection(self.dropout(pooled))
        return self.layer_norm(emb)


# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------


class CheckpointManager:
    def __init__(self, root: Path, chunk_id: Optional[int] = None):
        self.root = Path(root)
        self.chunk_id = chunk_id
        self.chunk_dir = self.root / (f"chunk{chunk_id}" if chunk_id is not None else "global")

        ensure_dir(self.root)
        ensure_dir(self.chunk_dir)

        self.latest_path = self.root / "latest_checkpoint.pt"
        self.chunk_latest_path = self.chunk_dir / "latest_checkpoint.pt"
        self.best_path = self.chunk_dir / "best_checkpoint.pt"

    def save_latest(self, state: Dict[str, Any], epoch: int) -> None:
        state = dict(state)
        state["saved_at"] = now_stamp()
        state["epoch"] = epoch

        epoch_path = self.chunk_dir / f"epoch_{epoch:03d}.pt"

        torch.save(state, epoch_path)
        torch.save(state, self.chunk_latest_path)
        torch.save(state, self.latest_path)

    def save_best(self, state: Dict[str, Any]) -> None:
        state = dict(state)
        state["saved_at"] = now_stamp()
        torch.save(state, self.best_path)

    def latest_available(self) -> Optional[Path]:
        # For chunk-specific training, never fall back to the global latest checkpoint.
        # The global latest may belong to another chunk and corrupt resume logic.
        if self.chunk_id is not None:
            return self.chunk_latest_path if self.chunk_latest_path.exists() else None
        return self.latest_path if self.latest_path.exists() else None


# -----------------------------------------------------------------------------
# MLM trainer
# -----------------------------------------------------------------------------


class FinBERTMLMTrainer:
    def __init__(self, cfg: FinBERTConfig, chunk_id: int):
        self.cfg = cfg
        self.chunk_id = chunk_id
        self.split = APPROVED_SPLITS[chunk_id]
        self.device = get_device(cfg.processor)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(cfg.base_model_name)

        if cfg.freeze_base:
            freeze_module(self.model.base_model)

        self.model.to(self.device)

        self.ckpt = CheckpointManager(cfg.models_path, chunk_id=chunk_id)

        ensure_dir(cfg.results_path)
        ensure_dir(cfg.code_results_path)

    def make_loader(self, years: Sequence[int], batch_size: int, shuffle: bool) -> DataLoader:
        base = SECChunkTextDataset(
            self.cfg.dataset_csv,
            years=years,
            max_rows=self.cfg.max_rows,
            sample_frac=self.cfg.sample_frac,
            seed=self.cfg.seed,
            sample_mode=self.cfg.sample_mode,
        )

        tokenized = TokenizedMLMDataset(base, self.tokenizer, self.cfg.max_length)

        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.cfg.mlm_probability,
        )

        generator = torch.Generator()
        generator.manual_seed(self.cfg.seed)

        return DataLoader(
            tokenized,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collator,
            generator=generator if shuffle else None,
            **make_loader_kwargs(self.cfg, self.device),
        )

    def train(self, resume: bool = True, trial: Optional[Any] = None) -> Dict[str, Any]:
        set_seed(self.cfg.seed)

        train_loader = self.make_loader(self.split.train_years, self.cfg.batch_size, shuffle=True)
        val_loader = self.make_loader(self.split.val_years, self.cfg.eval_batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        updates_per_epoch = math.ceil(len(train_loader) / max(1, self.cfg.gradient_accumulation_steps))
        total_update_steps = max(1, updates_per_epoch * self.cfg.epochs)
        warmup_steps = int(total_update_steps * self.cfg.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_update_steps,
        )

        scaler = GradScaler(enabled=(self.cfg.fp16 and self.device.type == "cuda"))

        start_epoch = 1
        best_val_loss = float("inf")
        no_improve = 0
        history: List[Dict[str, Any]] = []

        if resume:
            latest = self.ckpt.latest_available()
            if latest is not None:
                print(f"[resume] loading {latest}")
                state = torch.load(latest, map_location=self.device)

                self.model.load_state_dict(state["model_state"])
                optimizer.load_state_dict(state["optimizer_state"])
                scheduler.load_state_dict(state["scheduler_state"])

                if state.get("scaler_state") is not None:
                    scaler.load_state_dict(state["scaler_state"])

                start_epoch = int(state.get("epoch", 0)) + 1
                best_val_loss = float(state.get("best_val_loss", best_val_loss))
                no_improve = int(state.get("no_improve", 0))
                history = list(state.get("history", []))

        print(
            f"[train-mlm] chunk={self.chunk_id} device={self.device} "
            f"train_years={self.split.train_years} val_years={self.split.val_years}"
        )
        print(
            f"[train-mlm] train_batches={len(train_loader)} val_batches={len(val_loader)} "
            f"epochs={self.cfg.epochs} batch={self.cfg.batch_size} "
            f"eval_batch={self.cfg.eval_batch_size} workers={self.cfg.num_workers}"
        )

        cuda_memory_summary("[before-train]")

        for epoch in range(start_epoch, self.cfg.epochs + 1):
            train_loss = self._train_epoch(train_loader, optimizer, scheduler, scaler, epoch)
            val_loss = self.evaluate(val_loader, split_name="val")

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "time": now_stamp(),
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.append(row)

            improved = val_loss < best_val_loss

            if improved:
                best_val_loss = val_loss
                no_improve = 0
            else:
                no_improve += 1

            state = {
                "chunk_id": self.chunk_id,
                "model_state": self.model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                "best_val_loss": best_val_loss,
                "no_improve": no_improve,
                "history": history,
                "config": self.cfg.serializable(),
            }

            self.ckpt.save_latest(state, epoch)

            if improved:
                self.ckpt.save_best(state)

            self._write_history(history)

            print(
                f"[epoch {epoch}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"best={best_val_loss:.6f} improved={improved} no_improve={no_improve}"
            )

            cuda_memory_summary(f"[after-epoch-{epoch}]")

            if trial is not None:
                trial.report(val_loss, step=epoch)
                if trial.should_prune():
                    print(f"[optuna] pruning trial at epoch={epoch} val_loss={val_loss:.6f}")
                    raise optuna.TrialPruned()

            if no_improve >= self.cfg.early_stop_patience:
                print(f"[early-stop] no improvement for {no_improve} epochs")
                break

        self.export_models()

        return {
            "chunk_id": self.chunk_id,
            "best_val_loss": best_val_loss,
            "history": history,
        }

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: GradScaler,
        epoch: int,
    ) -> float:
        self.model.train()

        total_loss = 0.0
        steps = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(loader, desc=f"chunk{self.chunk_id} train epoch {epoch}", leave=True)
        accum = max(1, self.cfg.gradient_accumulation_steps)

        for batch_idx, batch in enumerate(pbar, start=1):
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            with autocast(enabled=(self.cfg.fp16 and self.device.type == "cuda")):
                out = self.model(**batch)
                loss = out.loss / accum

            scaler.scale(loss).backward()

            do_step = (batch_idx % accum == 0) or (batch_idx == len(loader))

            if do_step:
                scaler.unscale_(optimizer)
                clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip)

                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                new_scale = scaler.get_scale()

                # Scheduler must only step when optimizer step was not skipped by GradScaler.
                # If overflow occurred, GradScaler reduces the scale and skips optimizer.step().
                if new_scale >= old_scale:
                    scheduler.step()

                optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.detach().item()) * accum
            steps += 1

            pbar.set_postfix(
                loss=total_loss / max(1, steps),
                lr=optimizer.param_groups[0]["lr"],
            )

        return total_loss / max(1, steps)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split_name: str = "val") -> float:
        self.model.eval()

        total_loss = 0.0
        steps = 0

        for batch in tqdm(loader, desc=f"chunk{self.chunk_id} eval {split_name}", leave=False):
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            with autocast(enabled=(self.cfg.fp16 and self.device.type == "cuda")):
                out = self.model(**batch)

            total_loss += float(out.loss.detach().item())
            steps += 1

        return total_loss / max(1, steps)

    def export_models(self) -> None:
        best = self.ckpt.best_path if self.ckpt.best_path.exists() else self.ckpt.chunk_latest_path

        if not best.exists():
            raise FileNotFoundError(f"No checkpoint found to export for chunk {self.chunk_id}")

        state = torch.load(best, map_location=self.device)
        self.model.load_state_dict(state["model_state"])

        unfreezed_dir = self.cfg.models_path / f"chunk{self.chunk_id}" / "model_unfreezed"
        freezed_dir = self.cfg.models_path / f"chunk{self.chunk_id}" / "model_freezed"

        ensure_dir(unfreezed_dir)
        ensure_dir(freezed_dir)

        unfreeze_module(self.model)
        self.model.save_pretrained(unfreezed_dir)
        self.tokenizer.save_pretrained(unfreezed_dir)

        freeze_module(self.model)
        self.model.save_pretrained(freezed_dir)
        self.tokenizer.save_pretrained(freezed_dir)

        freeze_note = {
            "frozen_for_inference": True,
            "note": (
                "requires_grad flags are applied when loading; HuggingFace weights are "
                "numerically identical to best MLM checkpoint."
            ),
            "saved_at": now_stamp(),
            "best_checkpoint": str(best),
        }
        (freezed_dir / "FREEZE_NOTE.json").write_text(
            json.dumps(freeze_note, indent=2),
            encoding="utf-8",
        )

        print(f"[export] saved unfreezed={unfreezed_dir}")
        print(f"[export] saved freezed={freezed_dir}")

    def _write_history(self, history: List[Dict[str, Any]]) -> None:
        path = self.cfg.results_path / f"chunk{self.chunk_id}_mlm_history.csv"
        ensure_dir(path.parent)
        pd.DataFrame(history).to_csv(path, index=False)


# -----------------------------------------------------------------------------
# 768-dim extraction and PCA 768->256 projection
# -----------------------------------------------------------------------------


class FinBERTBaseEmbeddingExtractor:
    def __init__(self, cfg: FinBERTConfig, chunk_id: int, checkpoint_dir: Optional[Path] = None):
        self.cfg = cfg
        self.chunk_id = chunk_id
        self.split = APPROVED_SPLITS[chunk_id]
        self.device = get_device(cfg.processor)

        self.checkpoint_dir = checkpoint_dir or (cfg.models_path / f"chunk{chunk_id}" / "model_freezed")

        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"Frozen model directory not found: {self.checkpoint_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_dir)
        self.model = AutoModel.from_pretrained(self.checkpoint_dir, add_pooling_layer=False)

        freeze_module(self.model)
        self.model.to(self.device)
        self.model.eval()

    def years_for_split(self, split_name: str) -> List[int]:
        if split_name == "train":
            return self.split.train_years
        if split_name == "val":
            return self.split.val_years
        if split_name == "test":
            return self.split.test_years
        raise ValueError("split_name must be train, val, or test")

    def paths_768(self, split_name: str) -> Tuple[Path, Path, Path]:
        emb_path = self.cfg.embeddings_path / f"chunk{self.chunk_id}_{split_name}_embeddings768.npy"
        meta_path = self.cfg.embeddings_path / f"chunk{self.chunk_id}_{split_name}_metadata.csv"
        manifest_path = self.cfg.embeddings_path / f"chunk{self.chunk_id}_{split_name}_manifest768.json"
        return emb_path, meta_path, manifest_path

    @torch.no_grad()
    def extract_split_768(
        self,
        split_name: str,
        batch_size: Optional[int] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        years = self.years_for_split(split_name)
        emb_path, meta_path, manifest_path = self.paths_768(split_name)

        ensure_dir(self.cfg.embeddings_path)

        if emb_path.exists() and meta_path.exists() and manifest_path.exists() and not overwrite:
            print(f"[embed768] exists, skipping: {emb_path}")
            return json.loads(manifest_path.read_text(encoding="utf-8"))

        base = SECChunkTextDataset(
            self.cfg.dataset_csv,
            years=years,
            max_rows=self.cfg.max_rows,
            sample_frac=self.cfg.sample_frac,
            seed=self.cfg.seed,
            sample_mode=self.cfg.sample_mode,
        )

        tokenized = TokenizedEmbeddingDataset(base, self.tokenizer, self.cfg.max_length)

        loader = DataLoader(
            tokenized,
            batch_size=batch_size or self.cfg.eval_batch_size,
            shuffle=False,
            collate_fn=embedding_collate,
            **make_loader_kwargs(self.cfg, self.device),
        )

        arr = np.lib.format.open_memmap(
            emb_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(base), self.cfg.base_embedding_dim),
        )

        metadata_rows: List[Dict[str, Any]] = []
        offset = 0

        for batch in tqdm(loader, desc=f"embed768 chunk{self.chunk_id} {split_name}"):
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)

            with autocast(enabled=(self.cfg.fp16 and self.device.type == "cuda")):
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                emb = mean_pool(out.last_hidden_state, attention_mask)

            block = emb.detach().float().cpu().numpy().astype(np.float32)

            arr[offset : offset + block.shape[0], :] = block
            offset += block.shape[0]

            metadata_rows.extend(batch["metadata"])

        del arr

        pd.DataFrame(metadata_rows).to_csv(meta_path, index=False)

        manifest = {
            "chunk_id": self.chunk_id,
            "split": split_name,
            "years": years,
            "rows": int(offset),
            "dim": self.cfg.base_embedding_dim,
            "embedding_file": str(emb_path),
            "metadata_file": str(meta_path),
            "sha256_embeddings": sha256_file(emb_path),
            "created_at": now_stamp(),
            "projection": "none_raw_768",
        }

        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))

        return manifest

    def extract_all_768(self, overwrite: bool = False) -> List[Dict[str, Any]]:
        return [self.extract_split_768(s, overwrite=overwrite) for s in ["train", "val", "test"]]


class FinBERTPCAProjector:
    def __init__(self, cfg: FinBERTConfig, chunk_id: int):
        if IncrementalPCA is None:
            raise ImportError("scikit-learn is required for PCA projection. Install with: pip install scikit-learn")

        self.cfg = cfg
        self.chunk_id = chunk_id
        self.split = APPROVED_SPLITS[chunk_id]

        ensure_dir(cfg.embeddings_path)

        self.pca_path = cfg.embeddings_path / f"chunk{chunk_id}_pca_768_to_{cfg.projection_dim}.pkl"
        self.pca_manifest_path = cfg.embeddings_path / f"chunk{chunk_id}_pca_manifest.json"

    def raw_path(self, split_name: str) -> Path:
        return self.cfg.embeddings_path / f"chunk{self.chunk_id}_{split_name}_embeddings768.npy"

    def final_paths(self, split_name: str) -> Tuple[Path, Path]:
        emb_path = self.cfg.embeddings_path / f"chunk{self.chunk_id}_{split_name}_embeddings.npy"
        manifest_path = self.cfg.embeddings_path / f"chunk{self.chunk_id}_{split_name}_manifest.json"
        return emb_path, manifest_path

    def fit_train_pca(self, overwrite: bool = False) -> Dict[str, Any]:
        train_path = self.raw_path("train")

        if not train_path.exists():
            raise FileNotFoundError(f"Train 768 embeddings missing: {train_path}. Run embed768 first.")

        if self.pca_path.exists() and self.pca_manifest_path.exists() and not overwrite:
            print(f"[pca] exists, skipping fit: {self.pca_path}")
            return json.loads(self.pca_manifest_path.read_text(encoding="utf-8"))

        x = np.load(train_path, mmap_mode="r")

        if x.shape[1] != self.cfg.base_embedding_dim:
            raise ValueError(f"Expected 768-dim embeddings, got shape {x.shape}")

        batch = max(self.cfg.pca_batch_size, self.cfg.projection_dim + 1)

        pca = IncrementalPCA(
            n_components=self.cfg.projection_dim,
            batch_size=batch,
        )

        for start in tqdm(range(0, x.shape[0], batch), desc=f"fit PCA chunk{self.chunk_id}"):
            block = np.asarray(x[start : start + batch], dtype=np.float32)

            # IncrementalPCA requires batch rows >= n_components for partial_fit.
            if block.shape[0] >= self.cfg.projection_dim:
                pca.partial_fit(block)

        with self.pca_path.open("wb") as f:
            pickle.dump(pca, f)

        explained = float(np.sum(getattr(pca, "explained_variance_ratio_", np.array([0.0]))))

        manifest = {
            "chunk_id": self.chunk_id,
            "input_dim": self.cfg.base_embedding_dim,
            "output_dim": self.cfg.projection_dim,
            "train_rows": int(x.shape[0]),
            "pca_file": str(self.pca_path),
            "explained_variance_ratio_sum": explained,
            "fit_on": "train split only",
            "created_at": now_stamp(),
        }

        self.pca_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))

        return manifest

    def transform_split(self, split_name: str, overwrite: bool = False) -> Dict[str, Any]:
        raw = self.raw_path(split_name)
        emb_path, manifest_path = self.final_paths(split_name)

        if emb_path.exists() and manifest_path.exists() and not overwrite:
            print(f"[pca-transform] exists, skipping: {emb_path}")
            return json.loads(manifest_path.read_text(encoding="utf-8"))

        if not raw.exists():
            raise FileNotFoundError(f"Raw 768 embeddings missing: {raw}")

        if not self.pca_path.exists():
            raise FileNotFoundError(f"PCA model missing: {self.pca_path}. Run fit-pca first.")

        with self.pca_path.open("rb") as f:
            pca = pickle.load(f)

        x = np.load(raw, mmap_mode="r")

        out = np.lib.format.open_memmap(
            emb_path,
            mode="w+",
            dtype=np.float32,
            shape=(x.shape[0], self.cfg.projection_dim),
        )

        batch = max(self.cfg.pca_batch_size, self.cfg.projection_dim + 1)
        offset = 0

        for start in tqdm(range(0, x.shape[0], batch), desc=f"PCA transform chunk{self.chunk_id} {split_name}"):
            block = np.asarray(x[start : start + batch], dtype=np.float32)
            z = pca.transform(block).astype(np.float32)

            out[offset : offset + z.shape[0], :] = z
            offset += z.shape[0]

        del out

        manifest = {
            "chunk_id": self.chunk_id,
            "split": split_name,
            "rows": int(x.shape[0]),
            "dim": self.cfg.projection_dim,
            "embedding_file": str(emb_path),
            "metadata_file": str(self.cfg.embeddings_path / f"chunk{self.chunk_id}_{split_name}_metadata.csv"),
            "pca_file": str(self.pca_path),
            "sha256_embeddings": sha256_file(emb_path),
            "created_at": now_stamp(),
            "projection": "IncrementalPCA fitted on train split only",
        }

        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))

        return manifest

    def transform_all(self, overwrite: bool = False) -> List[Dict[str, Any]]:
        return [self.transform_split(s, overwrite=overwrite) for s in ["train", "val", "test"]]


# -----------------------------------------------------------------------------
# HPO
# -----------------------------------------------------------------------------


class FinBERTHyperparameterSearch:
    def __init__(
        self,
        base_cfg: FinBERTConfig,
        chunk_id: int,
        trials: int = 20,
        study_name: Optional[str] = None,
    ):
        if optuna is None:
            raise ImportError("optuna is required for hyperparameter search. Install with: pip install optuna")

        self.base_cfg = base_cfg
        self.chunk_id = chunk_id
        self.trials = trials
        self.study_name = study_name or f"finbert_mlm_chunk{chunk_id}"

        ensure_dir(base_cfg.code_results_path / "hpo")

        self.storage = f"sqlite:///{base_cfg.code_results_path / 'hpo' / 'finbert_optuna.db'}"

    def objective(self, trial: Any) -> float:
        cfg = FinBERTConfig(**asdict(self.base_cfg))

        cfg.learning_rate = trial.suggest_float("learning_rate", 5e-6, 5e-5, log=True)
        cfg.weight_decay = trial.suggest_float("weight_decay", 1e-4, 0.05, log=True)
        cfg.warmup_ratio = trial.suggest_float("warmup_ratio", 0.03, 0.15)
        cfg.batch_size = trial.suggest_categorical("batch_size", [16, 24, 32])
        cfg.eval_batch_size = min(96, max(32, cfg.batch_size * 2))
        cfg.gradient_accumulation_steps = trial.suggest_categorical("gradient_accumulation_steps", [1, 2])
        cfg.mlm_probability = trial.suggest_float("mlm_probability", 0.10, 0.20)
        cfg.epochs = trial.suggest_int("epochs", 1, 2)

        if cfg.max_rows is None:
            cfg.max_rows = 30000

        cfg.models_path = cfg.models_path / "hpo" / f"trial_{trial.number:04d}"
        cfg.results_path = cfg.results_path / "hpo" / f"trial_{trial.number:04d}"

        trainer = FinBERTMLMTrainer(cfg, self.chunk_id)

        result = trainer.train(resume=False, trial=trial)
        val = float(result["best_val_loss"])

        del trainer
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return val

    def run(self) -> Dict[str, Any]:
        sampler = optuna.samplers.TPESampler(seed=self.base_cfg.seed, multivariate=True)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True,
        )

        study.optimize(
            self.objective,
            n_trials=self.trials,
            gc_after_trial=True,
            catch=(RuntimeError,),
        )

        best_params: Dict[str, Any] = {}
        best_value: Optional[float] = None

        try:
            best_value = float(study.best_value)
            best_params = dict(study.best_params)
        except Exception:
            best_value = None
            best_params = {}

        best = {
            "study_name": self.study_name,
            "best_value": best_value,
            "best_params": best_params,
            "trials": len(study.trials),
            "storage": self.storage,
        }

        out = self.base_cfg.code_results_path / "hpo" / f"{self.study_name}_best_params.json"
        ensure_dir(out.parent)
        out.write_text(json.dumps(best, indent=2), encoding="utf-8")

        df = study.trials_dataframe()
        df.to_csv(self.base_cfg.code_results_path / "hpo" / f"{self.study_name}_trials.csv", index=False)

        print(json.dumps(best, indent=2))
        return best


# -----------------------------------------------------------------------------
# Supervised hooks and label notes
# -----------------------------------------------------------------------------


class FinBERTSupervisedTrainer:
    """
    Placeholder for future supervised market/risk-label fine-tuning.

    Current CSV has no supervised labels. Build labels first in a data engineering
    script, then train supervised heads here.
    """

    def __init__(self, cfg: FinBERTConfig, chunk_id: int):
        self.cfg = cfg
        self.chunk_id = chunk_id

    def train(self, resume: bool = True) -> None:
        raise NotImplementedError("Supervised fine-tuning requires label CSV generation first.")


def write_label_spec(cfg: FinBERTConfig) -> Path:
    spec = {
        "status": "label generation not yet executed",
        "recommended_labels": {
            "excess_return_10d_class": (
                "3-class label using stock forward return minus market forward return, "
                "thresholds from training years only"
            ),
            "volatility_spike_30d": (
                "binary label using future realized volatility above training-period percentile"
            ),
            "drawdown_risk_30d": (
                "binary or regression label using future max drawdown"
            ),
        },
        "anti_leakage_rules": [
            "Use only prices after filing_date.",
            "Use next trading day as event start if filing timestamp is unavailable.",
            "Compute quantile thresholds from train years only for each chronological chunk.",
            "Attach document-level labels to all chunks belonging to the same doc_id.",
        ],
    }

    out = cfg.code_results_path / "finbert_label_spec.json"
    ensure_dir(out.parent)
    out.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(json.dumps(spec, indent=2))

    return out


# -----------------------------------------------------------------------------
# Inspection
# -----------------------------------------------------------------------------


def inspect_dataset(cfg: FinBERTConfig) -> None:
    print(f"[inspect] dataset={cfg.dataset_csv}")

    df = pd.read_csv(cfg.dataset_csv)

    print(f"rows={len(df):,} columns={list(df.columns)}")
    print("years:", df["year"].value_counts().sort_index().to_dict())
    print("forms:", df["form_type"].value_counts().to_dict())
    print("sources top20:", df["source_name"].value_counts().head(20).to_dict())
    print("unique docs:", df["doc_id"].nunique())
    print("unique cik:", df["cik"].nunique())
    print("word_count:", df["word_count"].describe().to_dict())

    sample = df.head(3).copy()
    sample["text"] = sample["text"].astype(str).str.slice(0, 220) + "..."
    print(sample.to_string(index=False))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_config_from_args(args: argparse.Namespace) -> FinBERTConfig:
    cfg = FinBERTConfig()
    cfg.repo_root = Path(args.repo_root).resolve()
    cfg.env_file = Path(args.env_file)
    cfg.dataset_csv = Path(args.dataset_csv)
    cfg.resolve()

    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    if args.eval_batch_size is not None:
        cfg.eval_batch_size = args.eval_batch_size

    if args.epochs is not None:
        cfg.epochs = args.epochs

    if args.lr is not None:
        cfg.learning_rate = args.lr

    if args.max_rows is not None:
        cfg.max_rows = args.max_rows

    if args.sample_frac is not None:
        cfg.sample_frac = args.sample_frac

    if args.sample_mode is not None:
        cfg.sample_mode = args.sample_mode

    if args.workers is not None:
        cfg.num_workers = args.workers

    if args.pca_batch_size is not None:
        cfg.pca_batch_size = args.pca_batch_size

    if args.cpu:
        cfg.processor = "cpu"

    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FinBERT encoder, fine-tuning, HPO, and embedding extraction for fin-glassbox."
    )

    p.add_argument(
        "command",
        choices=[
            "inspect",
            "label-spec",
            "train-mlm",
            "train-all-mlm",
            "hpo",
            "embed768",
            "fit-pca",
            "project-pca",
            "embed",
            "embed-all",
            "freeze",
        ],
        help="Command to run",
    )

    p.add_argument("--repo-root", default=".")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--dataset-csv", default="final/filings_finbert_chunks_balanced_25y_cap40000.csv")
    p.add_argument("--chunk", type=int, choices=[1, 2, 3], default=1)
    p.add_argument("--split", choices=["train", "val", "test"], default="train")

    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--eval-batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)

    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--sample-mode", choices=["head", "random", "balanced-year"], default=None)

    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--pca-batch-size", type=int, default=None)

    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--cpu", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_config_from_args(args)

    set_seed(cfg.seed)

    ensure_dir(cfg.models_path)
    ensure_dir(cfg.embeddings_path)
    ensure_dir(cfg.results_path)
    ensure_dir(cfg.code_results_path)

    cfg.save(cfg.code_results_path / "finbert_config_resolved.json")

    print(
        f"[device] preferred={cfg.processor} "
        f"cuda_available={torch.cuda.is_available()} "
        f"device={get_device(cfg.processor)}"
    )

    if torch.cuda.is_available():
        print(f"[gpu] {torch.cuda.get_device_name(0)}")

    if args.command == "inspect":
        inspect_dataset(cfg)

    elif args.command == "label-spec":
        write_label_spec(cfg)

    elif args.command == "train-mlm":
        FinBERTMLMTrainer(cfg, args.chunk).train(resume=not args.no_resume)

    elif args.command == "train-all-mlm":
        for chunk in [1, 2, 3]:
            trainer = FinBERTMLMTrainer(cfg, chunk)
            trainer.train(resume=not args.no_resume)

            del trainer
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    elif args.command == "hpo":
        FinBERTHyperparameterSearch(cfg, args.chunk, trials=args.trials).run()

    elif args.command == "embed768":
        FinBERTBaseEmbeddingExtractor(cfg, args.chunk).extract_split_768(
            args.split,
            overwrite=args.overwrite,
        )

    elif args.command == "fit-pca":
        FinBERTPCAProjector(cfg, args.chunk).fit_train_pca(overwrite=args.overwrite)

    elif args.command == "project-pca":
        FinBERTPCAProjector(cfg, args.chunk).transform_split(
            args.split,
            overwrite=args.overwrite,
        )

    elif args.command == "embed":
        extractor = FinBERTBaseEmbeddingExtractor(cfg, args.chunk)
        projector = FinBERTPCAProjector(cfg, args.chunk)

        # Need train 768 embeddings to fit PCA.
        if args.split == "train" or not projector.pca_path.exists():
            extractor.extract_split_768("train", overwrite=args.overwrite)
            projector.fit_train_pca(overwrite=args.overwrite)

        extractor.extract_split_768(args.split, overwrite=args.overwrite)
        projector.transform_split(args.split, overwrite=args.overwrite)

    elif args.command == "embed-all":
        for chunk in [1, 2, 3]:
            extractor = FinBERTBaseEmbeddingExtractor(cfg, chunk)
            projector = FinBERTPCAProjector(cfg, chunk)

            extractor.extract_all_768(overwrite=args.overwrite)
            projector.fit_train_pca(overwrite=args.overwrite)
            projector.transform_all(overwrite=args.overwrite)

            del extractor, projector
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    elif args.command == "freeze":
        FinBERTMLMTrainer(cfg, args.chunk).export_models()

    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
