# `code/encoders/` Folder Documentation

## 1. Purpose

The `code/encoders/` directory contains the upstream representation-learning modules for **An Explainable Multimodal Neural Framework for Financial Risk Management**. These encoders convert raw or engineered financial data into dense machine-learning representations used by downstream analysts, risk modules, and fusion.

The directory supports two major modalities:

1. **Market time-series data** through the Shared Temporal Attention Encoder.
2. **Financial text data** through the FinBERT Financial Text Encoder.

The encoders do not make final trading decisions. Their purpose is to produce reusable embeddings that downstream modules can consume consistently:

```text
Raw / engineered data
        │
        ▼
Encoders
        │
        ├── Temporal embeddings: ticker-date market-sequence representation
        └── FinBERT embeddings: filing/text-event representation
        │
        ▼
Analysts, Risk Engine, Regime Model, Fusion
```

---

## 2. Directory contents

```text
code/encoders/
├── temporal_encoder.py
├── finbert_encoder.py
├── build_embedding_manifest.py
├── run_finbert_full_pipeline.py
├── run_finbert_resume_after_hpo.py
├── TemporalEncoder.md
├── FinBERT_Encoder.md
└── TextEncoder.md
```

### Core code files

| File | Role |
|---|---|
| `temporal_encoder.py` | Shared attention-based encoder for market time-series windows. |
| `finbert_encoder.py` | Full FinBERT MLM fine-tuning, HPO, embedding extraction, PCA projection, and model export pipeline. |
| `build_embedding_manifest.py` | One-time helper for reconstructing temporal embedding manifests when needed. |
| `run_finbert_full_pipeline.py` | Full orchestration script for FinBERT HPO, MLM training, embedding extraction, and PCA projection. |
| `run_finbert_resume_after_hpo.py` | Resume script for FinBERT training and embedding generation after HPO has already produced best parameters. |

### Documentation files

| File | Role |
|---|---|
| `TemporalEncoder.md` | Full documentation for the Shared Temporal Attention Encoder. |
| `FinBERT_Encoder.md` | Full documentation for the FinBERT Financial Text Encoder. |
| `TextEncoder.md` | Extended text encoder context, data contracts, training decisions, and downstream interface notes. |

---

## 3. Role in the full architecture

The encoders sit directly after data processing and before specialised downstream modules:

```text
INPUT DATA
├── Time-Series Market Data
│   └── Shared Temporal Attention Encoder
│       ├── Technical Analyst
│       ├── Volatility Model
│       ├── Drawdown Risk Model
│       └── Regime Detection
│
└── Financial Text Data
    └── FinBERT Financial Text Encoder
        ├── Sentiment Analyst
        ├── News Analyst
        ├── Qualitative Analyst
        └── Regime Detection
```

The project intentionally avoids using one monolithic model. Instead, encoders produce modality-specific representations that are consumed by specialised modules.

---

## 4. Temporal Encoder overview

### 4.1 Purpose

The Temporal Encoder converts rolling market-feature windows into 256-dimensional ticker-date embeddings. It captures time dependencies in price, volatility, momentum, volume, and market-position indicators.

It is shared across multiple downstream modules so that technical and risk models use a consistent market representation.

### 4.2 Input data

Primary input:

```text
data/yFinance/processed/features_temporal.csv
```

Expected feature columns include:

```text
log_return
vol_5d
vol_21d
rsi_14
macd_hist
bb_pos
volume_ratio
hl_ratio
price_pos
spy_corr_63d
```

The model uses rolling sequence windows, typically:

```text
sequence length = 30 trading days
embedding dimension = 256
```

### 4.3 Model design

The Temporal Encoder uses:

```text
input projection
+ positional encoding
+ Transformer encoder layers
+ pooling
+ 256-dimensional embedding output
```

The main technical encoder is deliberately attention-based rather than plain LSTM/CNN. GNNs are not used here; graph modelling is reserved for contagion and regime risk modules.

### 4.4 Training objective

The Temporal Encoder uses self-supervised masked temporal reconstruction. Parts of the input sequence are masked and the model learns to reconstruct the hidden market features. This allows it to learn market-state representations without requiring supervised labels at the encoder stage.

### 4.5 Leakage control

The normaliser must be fitted only on the relevant train split. Validation and test splits reuse the train-fitted normaliser. This prevents validation/test distribution information from leaking into training.

### 4.6 Final output contract

Expected output directory:

```text
outputs/embeddings/TemporalEncoder/
```

Expected files:

```text
chunk1_train_embeddings.npy
chunk1_train_manifest.csv
chunk1_val_embeddings.npy
chunk1_val_manifest.csv
chunk1_test_embeddings.npy
chunk1_test_manifest.csv
chunk2_train_embeddings.npy
chunk2_train_manifest.csv
chunk2_val_embeddings.npy
chunk2_val_manifest.csv
chunk2_test_embeddings.npy
chunk2_test_manifest.csv
chunk3_train_embeddings.npy
chunk3_train_manifest.csv
chunk3_val_embeddings.npy
chunk3_val_manifest.csv
chunk3_test_embeddings.npy
chunk3_test_manifest.csv
```

Embeddings are stored as `.npy`; manifests are stored as `.csv` with ticker/date row alignment.

---

## 5. FinBERT Encoder overview

### 5.1 Purpose

The FinBERT encoder converts SEC filing text chunks into 256-dimensional text embeddings. These embeddings are used by:

- Sentiment Analyst,
- News Analyst,
- Qualitative Analyst,
- Regime Detection,
- and any later text-aware fusion or explanation layer.

### 5.2 Model strategy

The pipeline starts from FinBERT and performs domain-adaptive masked language modelling on the project’s SEC filings corpus. After training, it extracts 768-dimensional FinBERT hidden representations and projects them to 256 dimensions using Incremental PCA fitted only on the train split.

The final embedding dimension matches the Temporal Encoder output dimension:

```text
Temporal embedding: 256
FinBERT text embedding: 256
```

This makes multimodal integration cleaner and keeps downstream model sizes controlled.

### 5.3 Input data

Primary dataset:

```text
final/filings_finbert_chunks_balanced_25y_cap40000.csv
```

Important metadata fields include:

```text
chunk_id
doc_id
year
form_type
cik
filing_date
accession
source_name
chunk_index
word_count
```

### 5.4 Chronological chunks

The project uses three chronological chunks:

```text
Chunk 1: 2000–2004 train, 2005 val, 2006 test
Chunk 2: 2007–2014 train, 2015 val, 2016 test
Chunk 3: 2017–2022 train, 2023 val, 2024 test
```

This split design is critical for preventing look-ahead bias.

### 5.5 Final output contract

Expected output directory:

```text
outputs/embeddings/FinBERT/
```

Expected final 256-dimensional files:

```text
chunk1_train_embeddings.npy
chunk1_train_metadata.csv
chunk1_train_manifest.json
chunk1_val_embeddings.npy
chunk1_val_metadata.csv
chunk1_val_manifest.json
chunk1_test_embeddings.npy
chunk1_test_metadata.csv
chunk1_test_manifest.json

chunk2_train_embeddings.npy
chunk2_train_metadata.csv
chunk2_train_manifest.json
chunk2_val_embeddings.npy
chunk2_val_metadata.csv
chunk2_val_manifest.json
chunk2_test_embeddings.npy
chunk2_test_metadata.csv
chunk2_test_manifest.json

chunk3_train_embeddings.npy
chunk3_train_metadata.csv
chunk3_train_manifest.json
chunk3_val_embeddings.npy
chunk3_val_metadata.csv
chunk3_val_manifest.json
chunk3_test_embeddings.npy
chunk3_test_metadata.csv
chunk3_test_manifest.json
```

Intermediate 768-dimensional files may also exist, especially during PCA generation:

```text
chunk*_train_embeddings768.npy
chunk*_val_embeddings768.npy
chunk*_test_embeddings768.npy
chunk*_pca_768_to_256.pkl
chunk*_pca_manifest.json
```

PCA must be fitted on train split only.

---

## 6. `build_embedding_manifest.py`

This helper reconstructs temporal embedding manifests from `features_temporal.csv`. It maps each embedding row back to:

```text
ticker
date
seq_start
seq_end
```

The manifest is essential because downstream modules must know which ticker-date each embedding row represents. The helper is useful if manifests are missing, corrupted, or need to be regenerated after embedding production.

Main command:

```bash
cd ~/fin-glassbox && python code/encoders/build_embedding_manifest.py
```

Modern Temporal Encoder runs usually write manifests directly during embedding generation, so this file is mostly a repair/one-time utility.

---

## 7. FinBERT orchestration scripts

### 7.1 `run_finbert_full_pipeline.py`

Runs the full FinBERT lifecycle:

```text
HPO on chunk sample
→ train MLM on chunks 1, 2, 3
→ extract 768-dimensional embeddings
→ fit train-only PCA
→ project train/val/test embeddings to 256 dimensions
```

This script is useful for a clean full rerun when enough time and GPU availability are available.

### 7.2 `run_finbert_resume_after_hpo.py`

Resumes FinBERT training after HPO has already completed. It reads best parameters from:

```text
outputs/codeResults/FinBERT/hpo/finbert_mlm_chunk3_final_best_params.json
```

Then it trains/resumes all chunks and regenerates embeddings/PCA outputs. This is useful when HPO has already succeeded and training or projection needs to continue after interruption.

---

## 8. Important CLI commands

### 8.1 Temporal Encoder

Inspect:

```bash
cd ~/fin-glassbox && python code/encoders/temporal_encoder.py inspect --repo-root .
```

Run HPO:

```bash
cd ~/fin-glassbox && python code/encoders/temporal_encoder.py hpo --repo-root . --chunk 1 --trials 30 --device cuda
```

Train best:

```bash
cd ~/fin-glassbox && python code/encoders/temporal_encoder.py train-best --repo-root . --chunk 1 --device cuda
```

Generate embeddings for one split:

```bash
cd ~/fin-glassbox && python code/encoders/temporal_encoder.py embed --repo-root . --chunk 1 --split train --device cuda --batch-size 4096 --num-workers 8 --prefetch-factor 4
```

Generate all embeddings:

```bash
cd ~/fin-glassbox && python code/encoders/temporal_encoder.py embed-all --repo-root . --device cuda --batch-size 4096 --num-workers 8 --prefetch-factor 4
```

### 8.2 FinBERT Encoder

Inspect:

```bash
cd ~/fin-glassbox && python code/encoders/finbert_encoder.py inspect --repo-root .
```

Run HPO:

```bash
cd ~/fin-glassbox && python code/encoders/finbert_encoder.py hpo --repo-root . --chunk 3 --trials 12 --processor cuda
```

Train MLM:

```bash
cd ~/fin-glassbox && python code/encoders/finbert_encoder.py train-mlm --repo-root . --chunk 1 --processor cuda
```

Train all MLM chunks:

```bash
cd ~/fin-glassbox && python code/encoders/finbert_encoder.py train-all-mlm --repo-root . --processor cuda
```

Export frozen/unfrozen model folders:

```bash
cd ~/fin-glassbox && python code/encoders/finbert_encoder.py freeze --repo-root . --chunk 1
```

Extract 768-dimensional embeddings:

```bash
cd ~/fin-glassbox && python code/encoders/finbert_encoder.py embed768 --repo-root . --chunk 1 --split train --eval-batch-size 64 --workers 6 --overwrite
```

Fit PCA on train split:

```bash
cd ~/fin-glassbox && python code/encoders/finbert_encoder.py fit-pca --repo-root . --chunk 1 --pca-batch-size 4096 --overwrite
```

Project split to 256 dimensions:

```bash
cd ~/fin-glassbox && python code/encoders/finbert_encoder.py project-pca --repo-root . --chunk 1 --split train --pca-batch-size 4096 --overwrite
```

Full embedding helper:

```bash
cd ~/fin-glassbox && python code/encoders/finbert_encoder.py embed --repo-root . --chunk 1 --split train --processor cuda --overwrite
```

Full all-chunk embedding helper:

```bash
cd ~/fin-glassbox && python code/encoders/finbert_encoder.py embed-all --repo-root . --processor cuda --overwrite
```

---

## 9. Validation commands

### 9.1 Temporal embedding audit

```bash
cd ~/fin-glassbox && python - <<'PY'
import numpy as np
import pandas as pd
from pathlib import Path
base = Path('outputs/embeddings/TemporalEncoder')
for c in [1, 2, 3]:
    for s in ['train', 'val', 'test']:
        emb = base / f'chunk{c}_{s}_embeddings.npy'
        man = base / f'chunk{c}_{s}_manifest.csv'
        if not emb.exists() or not man.exists():
            print(f'MISSING chunk{c}_{s}: emb={emb.exists()} manifest={man.exists()}')
            continue
        arr = np.load(emb, mmap_mode='r')
        m = pd.read_csv(man)
        finite = float(np.isfinite(arr[:min(len(arr), 10000)]).mean())
        print(f'chunk{c}_{s}: emb={arr.shape}, manifest={m.shape}, finite_sample={finite:.6f}')
PY
```

### 9.2 FinBERT embedding audit

```bash
cd ~/fin-glassbox && python - <<'PY'
import numpy as np
import pandas as pd
from pathlib import Path
base = Path('outputs/embeddings/FinBERT')
for c in [1, 2, 3]:
    for s in ['train', 'val', 'test']:
        emb = base / f'chunk{c}_{s}_embeddings.npy'
        meta = base / f'chunk{c}_{s}_metadata.csv'
        if not emb.exists() or not meta.exists():
            print(f'MISSING chunk{c}_{s}: emb={emb.exists()} metadata={meta.exists()}')
            continue
        arr = np.load(emb, mmap_mode='r')
        m = pd.read_csv(meta)
        finite = float(np.isfinite(arr[:min(len(arr), 10000)]).mean())
        print(f'chunk{c}_{s}: emb={arr.shape}, metadata={m.shape}, finite_sample={finite:.6f}')
PY
```

---

## 10. XAI responsibilities

The encoders are not decision modules, but they still support explainability.

### 10.1 Temporal Encoder XAI

The Temporal Encoder supports:

- attention timestep summaries,
- top-timestep reporting,
- gradient feature-importance sampling,
- manifest-level traceability from embedding row to ticker-date window.

This allows downstream explanations to refer back to the time window that produced a representation.

### 10.2 FinBERT Encoder XAI

The FinBERT encoder supports:

- metadata traceability from embedding row to document/chunk,
- filing date, form type, CIK, accession, source section, and chunk index,
- downstream token/text explanation via Sentiment and News Analysts,
- PCA manifest documentation proving train-only projection.

The encoder itself does not produce final sentiment explanations; instead, it preserves the alignment required for downstream text analysts to generate event-level XAI.

---

## 11. Leakage and reproducibility rules

### 11.1 Temporal Encoder

- Train-fitted normaliser only.
- No validation/test fitting.
- Chronological chunking must be preserved.
- Manifest row count must match embedding row count.
- Embedding generation should use the correct model checkpoint for the same chunk.

### 11.2 FinBERT Encoder

- MLM training must respect chronological chunks.
- PCA must be fitted on train split only.
- Metadata rows must match embedding rows.
- Projection artefacts must be stored with manifests.
- Frozen and unfrozen model folders must be preserved for downstream reproducibility.

---

## 12. Downstream consumers

### Temporal Encoder outputs are consumed by:

```text
Technical Analyst
Volatility Risk Model
Drawdown Risk Model
Regime Detection
Quantitative Analyst indirectly through upstream risk/technical outputs
```

### FinBERT outputs are consumed by:

```text
Sentiment Analyst
News Analyst
Qualitative Analyst
Regime Detection
Fusion indirectly through qualitative branch outputs
```

---

## 13. Common failure modes

### Missing temporal manifest

Run:

```bash
cd ~/fin-glassbox && python code/encoders/build_embedding_manifest.py
```

### Temporal training is too slow

Use the best saved checkpoint if validation has plateaued, then generate embeddings with large inference batches. The project already demonstrated that embedding extraction can be much faster than prolonged training.

### FinBERT PCA mismatch

Ensure PCA was fitted on train split only and then applied to train/val/test. Do not fit PCA separately on validation or test.

### Metadata row mismatch

Check `.npy` shape against metadata CSV row count. Any mismatch means downstream row alignment is unsafe.

### Downstream modules fail after rerun

Check whether output filenames, dimensions, and metadata schemas remained consistent. Downstream modules expect 256-dimensional embeddings.

---

## 14. Recommended working order

For a clean full rerun:

```text
1. Process raw data and create clean market/text datasets.
2. Train or load Temporal Encoder.
3. Generate Temporal Encoder embeddings and manifests for all chunks.
4. Train or resume FinBERT MLM.
5. Extract FinBERT 768-dimensional embeddings.
6. Fit PCA on train split only.
7. Project FinBERT embeddings to 256 dimensions.
8. Audit all shapes, manifests, metadata, and finite ratios.
9. Run downstream analyst and risk modules.
10. Run quantitative/qualitative synthesis and fusion.
```

---

## 15. Related documentation

Use these documents for deeper module-level details:

- [`TemporalEncoder.md`](TemporalEncoder.md)
- [`FinBERT_Encoder.md`](FinBERT_Encoder.md)
- [`TextEncoder.md`](TextEncoder.md)

This README is the folder-level overview; the linked files provide the detailed implementation and methodology notes.

---


`code/encoders/` is the project’s representation-learning layer. It turns large-scale market and text data into aligned embeddings that downstream modules can train on without repeatedly processing raw data. Its main engineering responsibilities are:

```text
chronological correctness
train-only normalisation/projection
reproducible embedding files
metadata/manifest alignment
GPU-efficient inference
XAI traceability
```

Without this directory, the rest of the multimodal framework would not have stable, reusable inputs. It is therefore one of the foundational layers of the full financial risk-management system.
