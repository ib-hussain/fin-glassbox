# FinBERT Encoder

**Project:** An Explainable Distributed Deep Learning Framework for Financial Risk Management  
**Module:** FinBERT Financial Text Encoder  
**Primary implementation:** `code/encoders/finbert_encoder.py`  
**Pipeline helpers:** `run_finbert_full_pipeline.py`, `run_finbert_resume_after_hpo.py`  
**Previous documentation reference:** [TextEncoder.md](TextEncoder.md)  
**Output root:** `outputs/embeddings/FinBERT/`, `outputs/models/FinBERT/`, `outputs/results/FinBERT/`, `outputs/codeResults/FinBERT/`  

---

## 1. Purpose of this document

This document is the comprehensive updated documentation for the FinBERT encoder in the fin-glassbox project. It replaces the older text-encoder documentation as the broader context document for how SEC filing text is transformed into reusable financial text embeddings.

The FinBERT encoder is the project’s main text encoder. It converts SEC filing chunks into dense vector representations that are consumed by:

```text
Sentiment Analyst
News Analyst
Regime Risk Module
Qualitative Analyst
Fusion Engine
XAI Layer
```

The encoder does not directly output Buy/Hold/Sell decisions. It produces reusable representations for downstream specialist models.

---

## 2. Role in the final architecture

The final architecture separates market, graph, macro, risk, and text processing. FinBERT is responsible for the text stream:

```text
SEC filing text chunks
        ↓
FinBERT domain-adaptive MLM fine-tuning
        ↓
768-dimensional pooled FinBERT embeddings
        ↓
train-only PCA projection
        ↓
256-dimensional text embeddings
        ↓
Sentiment Analyst / News Analyst / Regime Risk / Qualitative Analyst / Fusion
```

The encoder belongs to the **Data Processing / Encoder Layer**, not the final analyst layer. Its output is a learned representation; interpretation happens in downstream modules.

---

## 3. Why FinBERT is used

FinBERT is used because the input language is financial. SEC filings contain specialised terminology, risk disclosures, accounting language, forward-looking statements, management discussion, business sections, governance text, and risk factors. General language models are less aligned with this domain.

The project uses FinBERT for:

- financial text representation;
- sentiment-relevant semantic encoding;
- news/event relevance encoding;
- SEC filing section understanding;
- text context for regime detection;
- text-side evidence for qualitative synthesis.

The base model is:

```text
ProsusAI/finbert
```

The project then performs domain-adaptive fine-tuning using Masked Language Modelling on the project’s own SEC filing corpus.

---

## 4. Important design distinction

The FinBERT stage is primarily **self-supervised domain adaptation**, not supervised market prediction.

The training objective is:

```text
Masked Language Modelling loss
```

This means the model learns SEC disclosure language better, but it does not directly learn returns, drawdowns, risk classes, or sentiment labels. Those tasks are handled later by downstream trained modules.

This distinction is important for thesis defensibility:

```text
FinBERT improves the representation of financial text.
Sentiment Analyst and News Analyst learn task-specific predictions from that representation.
Qualitative Analyst learns how to combine those task-specific outputs.
```

---

## 5. Chronological split design

The project uses three chronological chunks:

| Chunk | Train years | Validation year | Test year |
|---|---:|---:|---:|
| Chunk 1 | 2000–2004 | 2005 | 2006 |
| Chunk 2 | 2007–2014 | 2015 | 2016 |
| Chunk 3 | 2017–2022 | 2023 | 2024 |

This design prevents look-ahead bias. Each chunk has its own model/export and embedding set. The train split is used for fitting the chunk’s PCA projection; validation and test are transformed using the train-fitted PCA only.

---

## 6. Primary source files

### 6.1 `finbert_encoder.py`

This is the main implementation. It includes:

- configuration and path resolution;
- chronological split logic;
- SEC text dataset loading;
- tokenisation for MLM training;
- tokenisation for embedding extraction;
- MLM trainer;
- checkpoint management;
- frozen/unfrozen model export;
- raw 768-dimensional embedding extraction;
- train-only PCA fitting;
- 256-dimensional projection;
- HPO support;
- inspection and label-spec utilities.

### 6.2 `run_finbert_full_pipeline.py`

This is a helper runner for executing the full FinBERT process across chunks. It is useful for scripted execution but the primary reusable implementation is still `finbert_encoder.py`.

### 6.3 `run_finbert_resume_after_hpo.py`

This helper exists to resume or continue the pipeline after HPO. It is useful in long-running GPU workflows where HPO, training, and embedding generation may be executed in separate sessions.

### 6.4 `TextEncoder.md`

This is the older documentation reference. It should now be superseded by this file, but it remains useful as historical context for early encoder design and output expectations.

---

## 7. Main classes and responsibilities

### 7.1 `FinBERTConfig`

Central configuration object. It defines:

- repository root;
- `.env` handling;
- dataset path;
- output paths;
- base model;
- embedding dimensions;
- training settings;
- dataloader settings;
- PCA settings;
- sampling settings;
- debug mode.

Important defaults include:

```text
base_model_name = ProsusAI/finbert
max_length = 512
base_embedding_dim = 768
projection_dim = 256
batch_size = 24
eval_batch_size = 64
mlm_probability = 0.15
pca_batch_size = 4096
```

### 7.2 `SECChunkTextDataset`

Loads the SEC filing text chunks from the final text corpus CSV.

Expected columns include:

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
text
```

It supports:

- year filtering;
- maximum-row sampling;
- fractional sampling;
- balanced-year sampling;
- metadata extraction without text.

### 7.3 `TokenizedMLMDataset`

Converts SEC text into tokenised inputs for masked language modelling. It performs truncation to the configured maximum sequence length and returns the tensors needed by the MLM data collator.

### 7.4 `TokenizedEmbeddingDataset`

Converts SEC text into padded tokenised inputs for deterministic embedding extraction. It also preserves row-level metadata.

### 7.5 `CheckpointManager`

Manages checkpoint saving and recovery. It saves:

```text
latest_checkpoint.pt
chunk{n}/latest_checkpoint.pt
chunk{n}/best_checkpoint.pt
chunk{n}/epoch_XXX.pt
```

This makes long-running MLM training resumable.

### 7.6 `FinBERTMLMTrainer`

Runs domain-adaptive MLM training. It:

- loads the base or resumed model;
- tokenises train and validation data;
- applies dynamic masking;
- trains with mixed precision where enabled;
- records train/validation loss;
- saves best/latest checkpoints;
- exports frozen and unfrozen model directories.

### 7.7 `FinBERTBaseEmbeddingExtractor`

Loads the frozen model and extracts 768-dimensional mean-pooled embeddings. The output is written as memory-mapped `.npy` arrays with matching metadata CSV and manifest JSON.

### 7.8 `FinBERTPCAProjector`

Fits an `IncrementalPCA` model on train-only 768-dimensional embeddings, then transforms train/validation/test embeddings into 256-dimensional final vectors.

This train-only PCA rule is critical:

```text
PCA is fit on train only.
Validation/test are transformed only.
```

### 7.9 `FinBERTHyperparameterSearch`

Runs Optuna-based HPO for MLM settings. HPO is used to estimate useful training hyperparameters, but final training can still be extended if validation loss is clearly improving.

### 7.10 `FinBERTProjectedEncoder`

This class is a future supervised-stage wrapper for a trainable 768-to-256 projection. It is not the main path for MLM-only embedding generation. The current final embeddings use train-only PCA projection.

---

## 8. Dataset input

The default dataset path is:

```text
final/filings_finbert_chunks_balanced_25y_cap40000.csv
```

This file is expected to contain cleaned SEC filing chunks with metadata and text. The dataset is intentionally not the full raw SEC corpus; it is a cleaned, filtered, and balanced text dataset suitable for FinBERT processing.

The design supports the project’s storage and compute constraints: text extraction was large and difficult, but final FinBERT training uses a manageable chunked corpus.

---

## 9. Output artefacts

### 9.1 Model outputs

Models are written under:

```text
outputs/models/FinBERT/chunk1/
outputs/models/FinBERT/chunk2/
outputs/models/FinBERT/chunk3/
```

Expected model/export files include:

```text
latest_checkpoint.pt
best_checkpoint.pt
model_freezed/
model_unfreezed/
```

The frozen export is used for deterministic embedding extraction. The unfrozen export is useful if additional fine-tuning is needed later.

### 9.2 Raw 768-dimensional embeddings

For each chunk and split:

```text
outputs/embeddings/FinBERT/chunk{chunk}_{split}_embeddings768.npy
outputs/embeddings/FinBERT/chunk{chunk}_{split}_metadata.csv
outputs/embeddings/FinBERT/chunk{chunk}_{split}_manifest768.json
```

These are the direct FinBERT pooled embeddings before dimensionality reduction.

### 9.3 PCA projection artefacts

For each chunk:

```text
outputs/embeddings/FinBERT/chunk{chunk}_pca_768_to_256.pkl
outputs/embeddings/FinBERT/chunk{chunk}_pca_manifest.json
```

The PCA is fit on the train split only.

### 9.4 Final 256-dimensional embeddings

For each chunk and split:

```text
outputs/embeddings/FinBERT/chunk{chunk}_{split}_embeddings.npy
outputs/embeddings/FinBERT/chunk{chunk}_{split}_manifest.json
```

The corresponding metadata CSV is shared with the 768 extraction:

```text
outputs/embeddings/FinBERT/chunk{chunk}_{split}_metadata.csv
```

The final 256-dimensional embeddings are the primary downstream inputs.

---

## 10. Embedding row alignment

Each embedding row must align exactly with one metadata row.

For each split:

```text
chunk{chunk}_{split}_embeddings.npy row i
        ↕
chunk{chunk}_{split}_metadata.csv row i
```

The metadata contains the document identity and filing context:

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

This alignment is essential for:

- analyst training;
- XAI traceability;
- qualitative evidence extraction;
- debugging document-level predictions;
- avoiding silent row mismatches.

---

## 11. Training process

### 11.1 Inspect dataset

```bash
cd ~/fin-glassbox && python code/encoders/finbert_encoder.py inspect --repo-root .
```

### 11.2 Run HPO

```bash
cd ~/fin-glassbox && python code/encoders/finbert_encoder.py hpo --repo-root . --chunk 1 --trials 20 --batch-size 16 --eval-batch-size 64 --workers 6
```

HPO should be treated as guidance, not a blind final instruction. If HPO only tries very short training runs, final MLM training may still need more epochs if validation loss is clearly improving.

### 11.3 Train MLM

```bash
cd ~/fin-glassbox && python code/encoders/finbert_encoder.py train-mlm --repo-root . --chunk 1 --epochs 6 --batch-size 16 --eval-batch-size 64 --workers 6 --lr 2.5e-5 --weight-decay 0.0003 --warmup-ratio 0.03 --mlm-probability 0.14 --gradient-accumulation-steps 1 --early-stop-patience 2
```

### 11.4 Export frozen/unfrozen models

```bash
cd ~/fin-glassbox && python code/encoders/finbert_encoder.py freeze --repo-root . --chunk 1
```

### 11.5 Extract raw 768-dimensional embeddings

```bash
cd ~/fin-glassbox && python code/encoders/finbert_encoder.py embed768 --repo-root . --chunk 1 --split train --eval-batch-size 64 --workers 6 --overwrite && python code/encoders/finbert_encoder.py embed768 --repo-root . --chunk 1 --split val --eval-batch-size 64 --workers 6 --overwrite && python code/encoders/finbert_encoder.py embed768 --repo-root . --chunk 1 --split test --eval-batch-size 64 --workers 6 --overwrite
```

### 11.6 Fit PCA and project to 256 dimensions

```bash
cd ~/fin-glassbox && python code/encoders/finbert_encoder.py fit-pca --repo-root . --chunk 1 --pca-batch-size 4096 --overwrite && python code/encoders/finbert_encoder.py project-pca --repo-root . --chunk 1 --split train --pca-batch-size 4096 --overwrite && python code/encoders/finbert_encoder.py project-pca --repo-root . --chunk 1 --split val --pca-batch-size 4096 --overwrite && python code/encoders/finbert_encoder.py project-pca --repo-root . --chunk 1 --split test --pca-batch-size 4096 --overwrite
```

---

## 12. Full per-chunk improved pipeline commands

### 12.1 Chunk 1

```bash
cd ~/fin-glassbox && python code/encoders/finbert_encoder.py train-mlm --repo-root . --chunk 1 --base-model-name outputs/models/FinBERT/chunk1/model_unfreezed --epochs 15 --batch-size 16 --eval-batch-size 64 --workers 6 --lr 2.5e-5 --weight-decay 0.0003 --warmup-ratio 0.03 --mlm-probability 0.14 --gradient-accumulation-steps 1 --early-stop-patience 2 --no-resume && python code/encoders/finbert_encoder.py freeze --repo-root . --chunk 1 && python code/encoders/finbert_encoder.py embed768 --repo-root . --chunk 1 --split train --eval-batch-size 64 --workers 6 --overwrite && python code/encoders/finbert_encoder.py embed768 --repo-root . --chunk 1 --split val --eval-batch-size 64 --workers 6 --overwrite && python code/encoders/finbert_encoder.py embed768 --repo-root . --chunk 1 --split test --eval-batch-size 64 --workers 6 --overwrite && python code/encoders/finbert_encoder.py fit-pca --repo-root . --chunk 1 --pca-batch-size 4096 --overwrite && python code/encoders/finbert_encoder.py project-pca --repo-root . --chunk 1 --split train --pca-batch-size 4096 --overwrite && python code/encoders/finbert_encoder.py project-pca --repo-root . --chunk 1 --split val --pca-batch-size 4096 --overwrite && python code/encoders/finbert_encoder.py project-pca --repo-root . --chunk 1 --split test --pca-batch-size 4096 --overwrite
```

### 12.2 Chunk 2

```bash
cd ~/fin-glassbox && python code/encoders/finbert_encoder.py train-mlm --repo-root . --chunk 2 --base-model-name outputs/models/FinBERT/chunk2/model_unfreezed --epochs 15 --batch-size 16 --eval-batch-size 64 --workers 6 --lr 2.5e-5 --weight-decay 0.0003 --warmup-ratio 0.03 --mlm-probability 0.14 --gradient-accumulation-steps 1 --early-stop-patience 2 --no-resume && python code/encoders/finbert_encoder.py freeze --repo-root . --chunk 2 && python code/encoders/finbert_encoder.py embed768 --repo-root . --chunk 2 --split train --eval-batch-size 64 --workers 6 --overwrite && python code/encoders/finbert_encoder.py embed768 --repo-root . --chunk 2 --split val --eval-batch-size 64 --workers 6 --overwrite && python code/encoders/finbert_encoder.py embed768 --repo-root . --chunk 2 --split test --eval-batch-size 64 --workers 6 --overwrite && python code/encoders/finbert_encoder.py fit-pca --repo-root . --chunk 2 --pca-batch-size 4096 --overwrite && python code/encoders/finbert_encoder.py project-pca --repo-root . --chunk 2 --split train --pca-batch-size 4096 --overwrite && python code/encoders/finbert_encoder.py project-pca --repo-root . --chunk 2 --split val --pca-batch-size 4096 --overwrite && python code/encoders/finbert_encoder.py project-pca --repo-root . --chunk 2 --split test --pca-batch-size 4096 --overwrite
```

### 12.3 Chunk 3

```bash
cd ~/fin-glassbox && python code/encoders/finbert_encoder.py train-mlm --repo-root . --chunk 3 --base-model-name outputs/models/FinBERT/chunk3/model_unfreezed --epochs 10 --batch-size 16 --eval-batch-size 64 --workers 6 --lr 2.959475667731825e-05 --weight-decay 0.0003438172512115178 --warmup-ratio 0.030662654054832286 --mlm-probability 0.13584657285442728 --gradient-accumulation-steps 1 --early-stop-patience 2 --no-resume && python code/encoders/finbert_encoder.py freeze --repo-root . --chunk 3 && python code/encoders/finbert_encoder.py embed768 --repo-root . --chunk 3 --split train --eval-batch-size 64 --workers 6 --overwrite && python code/encoders/finbert_encoder.py embed768 --repo-root . --chunk 3 --split val --eval-batch-size 64 --workers 6 --overwrite && python code/encoders/finbert_encoder.py embed768 --repo-root . --chunk 3 --split test --eval-batch-size 64 --workers 6 --overwrite && python code/encoders/finbert_encoder.py fit-pca --repo-root . --chunk 3 --pca-batch-size 4096 --overwrite && python code/encoders/finbert_encoder.py project-pca --repo-root . --chunk 3 --split train --pca-batch-size 4096 --overwrite && python code/encoders/finbert_encoder.py project-pca --repo-root . --chunk 3 --split val --pca-batch-size 4096 --overwrite && python code/encoders/finbert_encoder.py project-pca --repo-root . --chunk 3 --split test --pca-batch-size 4096 --overwrite
```

---

## 13. Validation commands

### 13.1 Model and embedding audit

```bash
cd ~/fin-glassbox && python - <<'PY'
import numpy as np
import pandas as pd
from pathlib import Path
base = Path('outputs/embeddings/FinBERT')
for chunk in [1,2,3]:
    print(f'===== chunk{chunk} =====')
    for split in ['train','val','test']:
        emb = base / f'chunk{chunk}_{split}_embeddings.npy'
        meta = base / f'chunk{chunk}_{split}_metadata.csv'
        man = base / f'chunk{chunk}_{split}_manifest.json'
        if emb.exists():
            arr = np.load(emb, mmap_mode='r')
            print(emb, arr.shape, 'finite=', float(np.isfinite(arr[:min(1000, len(arr))]).mean()))
        else:
            print('MISSING', emb)
        if meta.exists():
            print(meta, pd.read_csv(meta, nrows=1).shape, 'rows=', sum(1 for _ in open(meta))-1)
        else:
            print('MISSING', meta)
        print(('OK ' if man.exists() else 'MISSING ') + str(man))
PY
```

### 13.2 Training history audit

```bash
cd ~/fin-glassbox && python - <<'PY'
import pandas as pd
from pathlib import Path
for c in [1,2,3]:
    p = Path(f'outputs/results/FinBERT/chunk{c}_mlm_history.csv')
    print('\n===== chunk', c, '=====')
    if not p.exists():
        print('missing', p)
        continue
    h = pd.read_csv(p)
    print(h.tail(10).to_string(index=False))
    if 'val_loss' in h.columns:
        print('best val_loss:', float(h['val_loss'].min()))
        print(h.loc[h['val_loss'].idxmin()].to_string())
PY
```

### 13.3 File timestamp audit

```bash
cd ~/fin-glassbox && ls -lh --time-style=long-iso outputs/embeddings/FinBERT/chunk*_embeddings.npy outputs/embeddings/FinBERT/chunk*_metadata.csv | sort
```

---

## 14. Decision rule for more FinBERT training

FinBERT should not be trained indefinitely. Use validation loss trends.

Continue training if:

```text
validation loss is still clearly decreasing
and GPU time is available
and downstream sentiment/news performance remains poor
```

Stop training if:

```text
validation loss plateaus
validation loss worsens
learning rate has decayed close to zero
or improvements become too small to justify more GPU time
```

A practical rule:

```text
If new best validation loss improves by less than about 1–2% over several epochs, stop and move downstream.
```

If Sentiment Analyst accuracy remains poor after improved FinBERT embeddings, the bottleneck is likely:

- sentiment label construction;
- class imbalance;
- downstream loss design;
- output thresholding;
- insufficient task-specific supervision;
- not the FinBERT MLM encoder alone.

---

## 15. Downstream modules using FinBERT outputs

### 15.1 Sentiment Analyst

Consumes:

```text
outputs/embeddings/FinBERT/chunk{chunk}_{split}_embeddings.npy
outputs/embeddings/FinBERT/chunk{chunk}_{split}_metadata.csv
```

Produces sentiment polarity, confidence, uncertainty, and related task outputs.

### 15.2 News Analyst

Consumes the same or derived FinBERT-based representations and predicts event impact, importance, risk relevance, volatility-spike risk, and drawdown-risk relevance.

### 15.3 MTGNN/Regime Risk

May use FinBERT embeddings aggregated to a stock/date context and combined with temporal embeddings and macro variables.

### 15.4 Qualitative Analyst

Does not directly re-encode text. It combines the outputs of Sentiment Analyst and News Analyst into a trained qualitative branch signal.

### 15.5 Fusion Engine

Will later consume qualitative and quantitative branch outputs. FinBERT’s contribution enters Fusion indirectly through the trained text-side modules.

---

## 16. XAI and auditability

FinBERT itself is not the final explanation layer. However, it supports XAI through:

- row-aligned metadata;
- manifest JSON files;
- SHA256 hashes for embedding files;
- document IDs and accession numbers;
- source section names;
- filing dates;
- chunk indices;
- word counts.

This allows downstream explanations to point back to the original filing context.

Example XAI chain:

```text
Final decision explanation
→ Qualitative Analyst explanation
→ News/Sentiment Analyst explanation
→ FinBERT embedding row
→ SEC filing metadata
→ document ID / accession / section / filing date
```

This traceability is central to the project’s “glass-box” philosophy.

---

## 17. Storage and file-format discipline

Final FinBERT artefacts use:

```text
.npy   for embeddings
.csv   for metadata and training histories
.json  for manifests and configuration summaries
.pkl   for PCA models only
.pt    for PyTorch checkpoints
```

The project intentionally avoids unnecessary intermediate outputs in final documentation. Raw 768 embeddings may be retained during development, but final downstream modules should use the 256-dimensional embeddings unless a specific experiment requires raw 768 features.

---

## 18. Common failure modes and fixes

### 18.1 Metadata row mismatch

Symptom:

```text
embedding rows != metadata rows
```

Fix: rerun the affected `embed768` command and then rerun PCA projection.

### 18.2 PCA fitted on wrong split

Symptom:

```text
PCA leakage risk
```

Fix: delete the PCA file for that chunk and rerun `fit-pca` using only the train embeddings.

### 18.3 Frozen model missing

Symptom:

```text
Frozen model directory not found
```

Fix:

```bash
cd ~/fin-glassbox && python code/encoders/finbert_encoder.py freeze --repo-root . --chunk 1
```

### 18.4 HPO selects too few epochs

HPO may choose short runs because it is optimising a small trial budget. This should not automatically limit final domain-adaptive MLM training. Use validation-loss trends to decide final epochs.

### 18.5 Downstream sentiment remains weak

Improving FinBERT may help, but it will not guarantee high sentiment accuracy. If downstream accuracy remains low, inspect sentiment labels, class balance, target definitions, and analyst architecture.

---

## 19. Current completion checklist

A chunk is considered complete when all of the following exist:

```text
outputs/models/FinBERT/chunk{chunk}/model_freezed/
outputs/models/FinBERT/chunk{chunk}/model_unfreezed/
outputs/embeddings/FinBERT/chunk{chunk}_train_embeddings.npy
outputs/embeddings/FinBERT/chunk{chunk}_val_embeddings.npy
outputs/embeddings/FinBERT/chunk{chunk}_test_embeddings.npy
outputs/embeddings/FinBERT/chunk{chunk}_train_metadata.csv
outputs/embeddings/FinBERT/chunk{chunk}_val_metadata.csv
outputs/embeddings/FinBERT/chunk{chunk}_test_metadata.csv
outputs/embeddings/FinBERT/chunk{chunk}_train_manifest.json
outputs/embeddings/FinBERT/chunk{chunk}_val_manifest.json
outputs/embeddings/FinBERT/chunk{chunk}_test_manifest.json
outputs/embeddings/FinBERT/chunk{chunk}_pca_768_to_256.pkl
outputs/embeddings/FinBERT/chunk{chunk}_pca_manifest.json
```

---

## 20. Final summary

The FinBERT Encoder is the reusable financial text representation layer for the project. It adapts FinBERT to SEC filing language using masked language modelling, exports frozen models, extracts 768-dimensional document embeddings, projects them to 256 dimensions using train-only PCA, and preserves strict row-level metadata alignment.

Its role is not to make decisions. Its role is to provide the text-side representation that allows downstream analyst modules to reason about sentiment, event risk, news impact, regime context, and qualitative evidence.

The final architecture depends on FinBERT because it is the bridge between raw financial disclosure text and the system’s explainable multimodal decision process.
