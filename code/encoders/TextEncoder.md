# FinBERT Text Encoder Specification

**Project:** `fin-glassbox` — Explainable Distributed Deep Learning Framework for Financial Risk Management  
**Module:** Text Encoder / SEC Filing Encoder  
**Encoder family:** FinBERT, domain-adaptive fine-tuned with Masked Language Modelling  
**Final embedding dimensionality:** 256  
**Status:** Completed, verified, committed, and pushed  
**Final commit:** `88c8817`  

---

## 1. Executive Summary

The FinBERT Text Encoder converts SEC filing text chunks into dense, chronologically safe, row-aligned numerical embeddings for downstream financial-risk modelling. It is the text stream of the wider multi-modal architecture, where final asset representations are expected to combine:

```text
Temporal embedding:     128 dimensions
Text embedding:         256 dimensions
Fundamental embedding:  128 dimensions
Total asset vector:     512 dimensions
```

The encoder was fine-tuned using **domain-adaptive Masked Language Modelling (MLM)** on SEC filing text chunks. This is a legitimate self-supervised fine-tuning stage: FinBERT weights were updated to adapt the financial language model to SEC disclosure language before producing final embeddings.

The current encoder is **not yet supervised on market labels** such as returns, volatility, drawdown, or risk classes. That will be a later phase after labelled outcomes are generated from market data.

The completed run produced:

```text
9 final 256-dimensional embedding matrices
9 row-aligned metadata CSV files
9 final manifest JSON files
3 frozen FinBERT model exports
3 unfrozen FinBERT model exports
```

The final outputs are organised by three chronological chunks:

```text
Chunk 1: train 2000–2004, validation 2005, test 2006
Chunk 2: train 2007–2014, validation 2015, test 2016
Chunk 3: train 2017–2022, validation 2023, test 2024
```

---

## 2. Purpose

The FinBERT Text Encoder is responsible for transforming raw SEC filing text chunks into compact learned representations that can be consumed by:

```text
Sentiment Analyst
News Analyst
Regime Model
Fusion Model
Risk Engine
Downstream explainability modules
```

The encoder addresses the text stream of the architecture. It does not directly make risk predictions. Instead, it produces stable, reusable embeddings that downstream modules can interpret, aggregate, classify, score, or fuse with market, fundamental, macroeconomic, and graph-based features.

The encoder is designed around five constraints:

1. **Chronological safety** — train, validation, and test periods are separated by time.
2. **Reusable embeddings** — final `.npy` arrays can be loaded directly without rerunning FinBERT.
3. **Downstream consistency** — all final text vectors are 256-dimensional.
4. **Traceability** — each embedding row has matching metadata and a manifest.
5. **Storage discipline** — final deliverables exclude intermediate checkpoint/HPO/raw-768/PCA artefacts.

---

## 3. Code Locations

Main encoder implementation:

```text
code/encoders/finbert_encoder.py
```

Full end-to-end pipeline runner:

```text
code/encoders/run_finbert_full_pipeline.py
```

Resume-after-HPO runner:

```text
code/encoders/run_finbert_resume_after_hpo.py
```

Specification document:

```text
code/config/TextEncoder.md
```

Recommended future downstream modules:

```text
code/analysts/text_embedding_loader.py
code/analysts/sentiment_analyst.py
code/analysts/news_analyst.py
code/analysts/AnalystModels.md
```

---

## 4. Input Dataset

Primary dataset(only present on remote gpu):

```text
final/filings_finbert_chunks_balanced_25y_cap40000.csv
```

This file contains SEC filing text chunks from 2000 to 2024.

Expected columns:

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

The `text` column is the direct textual input to FinBERT. All other columns are treated as metadata and are preserved into the output metadata CSVs.

---

## 5. Dataset Size and Year Distribution

Final input dataset size:

```text
Rows excluding header: 989,244
Rows including header: 989,245
Coverage: 25 years, 2000–2024
Cap: approximately 40,000 chunks per year
```

Final year distribution after balancing/capping:

```text
2000:  34,620
2001:  34,624
2002:  40,000
2003:  40,000
2004:  40,000
2005:  40,000
2006:  40,000
2007:  40,000
2008:  40,000
2009:  40,000
2010:  40,000
2011:  40,000
2012:  40,000
2013:  40,000
2014:  40,000
2015:  40,000
2016:  40,000
2017:  40,000
2018:  40,000
2019:  40,000
2020:  40,000
2021:  40,000
2022:  40,000
2023:  40,000
2024:  40,000
```

---

## 6. Chronological Splits

The encoder uses three non-overlapping chronological chunks. Each chunk has its own model export and PCA projection, fitted only on that chunk's training split.

### Chunk 1

```text
Train:       2000–2004
Validation:  2005
Test:        2006
```

Final row counts:

```text
chunk1_train: 189,244 rows
chunk1_val:    40,000 rows
chunk1_test:   40,000 rows
```

### Chunk 2

```text
Train:       2007–2014
Validation:  2015
Test:        2016
```

Final row counts:

```text
chunk2_train: 320,000 rows
chunk2_val:    40,000 rows
chunk2_test:   40,000 rows
```

### Chunk 3

```text
Train:       2017–2022
Validation:  2023
Test:        2024
```

Final row counts:

```text
chunk3_train: 240,000 rows
chunk3_val:    40,000 rows
chunk3_test:   40,000 rows
```

These splits are designed to prevent temporal leakage. Validation and test years are always later than their corresponding training years.

---

## 7. Base Model

Base model:

```text
ProsusAI/finbert
```

Model family:

```text
BERT-based financial language model
```

Tokenizer:

```python
AutoTokenizer
```

MLM training model:

```python
AutoModelForMaskedLM
```

Embedding extraction model:

```python
AutoModel
```

Maximum token length:

```text
512 tokens
```

Texts longer than 512 tokens are truncated during tokenisation.

---

## 8. Fine-Tuning Method

The completed training stage used:

```text
Domain-adaptive Masked Language Modelling
```

This means FinBERT was adapted to SEC filing language by masking tokens inside filing chunks and training the model to predict the masked tokens.

Current objective:

```text
Predict masked tokens from SEC filing text chunks
```

The model was **not** trained directly on:

```text
future returns
future excess returns
volatility spikes
drawdown outcomes
risk labels
```

Those supervised targets will be generated later and may be used for a second-stage supervised fine-tuning run.

---

## 9. Why MLM Was Used First

The current text dataset contains SEC text and metadata but does not yet contain robust supervised financial labels. Therefore, MLM was the correct first-stage fine-tuning objective.

MLM adapts the model to:

```text
SEC disclosure language
risk factor wording
MD&A wording
governance disclosures
legal reporting structure
financial statement phrasing
material event language
formal issuer-reporting style
```

Recommended lifecycle:

```text
Stage 1: Domain-adaptive MLM fine-tuning on SEC filings
Stage 2: Generate supervised market/risk labels
Stage 3: Optional supervised FinBERT fine-tuning
Stage 4: Freeze encoder
Stage 5: Generate stable embeddings
Stage 6: Feed embeddings into analysts/fusion/risk models
```

---

## 10. Hardware and Environment

Remote GPU run hardware:

```text
CPU:        AMD Ryzen 5 7600
CPU cores:  6
RAM:        64 GB
GPU:        NVIDIA GeForce RTX 3090 Ti
GPU memory: 24 GB
CUDA:       working after NVIDIA driver reboot/fix
```

Python environment:

```text
Virtual environment: venv3.12.7
Python: 3.12.7
PyTorch: CUDA-enabled
```

Important CUDA validation command:

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

Expected CUDA validation result:

```text
cuda available True
gpu NVIDIA GeForce RTX 3090 Ti
```

---

## 11. Training Configuration

Final full-run configuration used these operational assumptions:

```text
processor: cuda
fp16: true
num_workers: 6
batch_size: selected by HPO / final run used batch size 16 in observed logs
eval_batch_size: selected by HPO / final run used eval batch size 32 in observed logs
epochs: 3 per chunk
random_seed: 42
sample_mode: balanced-year
```

Training was performed chunk-wise:

```text
Chunk 1: 3 MLM epochs
Chunk 2: 3 MLM epochs
Chunk 3: resumed and completed to 3 MLM epochs
```

Total planned full training schedule:

```text
3 chunks × 3 epochs = 9 chunk-level MLM epochs
```

---

## 12. Hyperparameter Search

Hyperparameter search was implemented with:

```text
Optuna
TPE sampler
Median pruner
SQLite persistent study
```

Search storage during training:

```text
outputs/codeResults/FinBERT/hpo/finbert_optuna.db
```

Best parameter file during training:

```text
outputs/codeResults/FinBERT/hpo/finbert_mlm_chunk3_final_best_params.json
```

Trial history during training:

```text
outputs/codeResults/FinBERT/hpo/finbert_mlm_chunk3_final_trials.csv
```

The HPO stage was run on chunk 3 using a training sample. The selected parameters were then reused for full chronological training across chunks 1, 2, and 3.

The `run_finbert_resume_after_hpo.py` runner does **not** rerun HPO. It loads the best HPO parameters and performs/resumes full training and embedding extraction.

HPO artefacts are not part of the final deliverable commit.

---

## 13. Training Results

### Chunk 2 observed training history

```text
epoch 1: train_loss = 1.5737829372435808, val_loss = 1.0838342635154725
epoch 2: train_loss = 1.1011696563720703, val_loss = 0.9906270583152771
epoch 3: train_loss = 1.031179,             val_loss = 0.952888
```

Chunk 2 improved every epoch and exported successfully.

### Chunk 3 observed training history

```text
epoch 2: train_loss = 1.676405, val_loss = 1.175561
epoch 3: train_loss = 1.201251, val_loss = 1.071729
```

Chunk 3 resumed from checkpoint, completed training, improved validation loss, and exported successfully.

### Chunk 1 training history

Chunk 1 completed successfully and produced model exports and embeddings. Exact epoch-level values should be read from:

```text
outputs/results/FinBERT/chunk1_mlm_history.csv
```

This file was treated as a training artefact and is not part of the minimal final deliverable unless intentionally retained.

---

## 14. Checkpointing During Training

During the training run, checkpoints were saved after every epoch for fault tolerance.

Training-time checkpoint types:

```text
latest_checkpoint.pt
best_checkpoint.pt
epoch_001.pt
epoch_002.pt
epoch_003.pt
```

Checkpoint contents included:

```text
chunk_id
model_state
optimizer_state
scheduler_state
scaler_state
best_val_loss
no_improve
history
config
epoch
saved_at
```

Important resume rule:

```text
Chunk-specific training must only resume from that chunk's latest checkpoint.
A global latest checkpoint must not be used across chunks.
```

A checkpoint-resume bug was corrected so chunk-specific training does not accidentally reuse a checkpoint from a different chunk.

### Final deliverable policy

Checkpoints were useful during training but are **not final deliverables**. They were deleted before final storage cleanup and excluded from the final commit.

Excluded from final deliverable:

```text
outputs/models/FinBERT/latest_checkpoint.pt
outputs/models/FinBERT/**/latest_checkpoint.pt
outputs/models/FinBERT/**/best_checkpoint.pt
outputs/models/FinBERT/**/epoch_*.pt
outputs/models/FinBERT/hpo/
```

---

## 15. Model Export

After each chunk finished training, the best available model was exported in two forms.

Frozen model exports:

```text
outputs/models/FinBERT/chunk1/model_freezed/
outputs/models/FinBERT/chunk2/model_freezed/
outputs/models/FinBERT/chunk3/model_freezed/
```

Unfrozen model exports:

```text
outputs/models/FinBERT/chunk1/model_unfreezed/
outputs/models/FinBERT/chunk2/model_unfreezed/
outputs/models/FinBERT/chunk3/model_unfreezed/
```

The frozen model is intended for stable embedding extraction and downstream reproducibility.

The unfrozen model is preserved so that later supervised fine-tuning can continue from the domain-adapted model rather than from the original `ProsusAI/finbert` base.

Typical exported Hugging Face directory contents:

```text
config.json
generation_config.json
model.safetensors
special_tokens_map.json
tokenizer.json
tokenizer_config.json
vocab.txt
```

Frozen model directories additionally include:

```text
FREEZE_NOTE.json
```

---

## 16. Embedding Extraction

Embedding extraction was performed after model export.

Pooling method:

```text
Mean pooling over last hidden states using attention mask
```

Raw embedding dimensionality:

```text
768
```

Final downstream dimensionality:

```text
256
```

Data type:

```text
float32
```

The raw 768-dimensional embeddings were generated temporarily during the pipeline, but they were not retained in the final minimal deliverable commit.

---

## 17. PCA Projection

The architecture requires a 256-dimensional text vector. The encoder therefore projects FinBERT's 768-dimensional output to 256 dimensions using IncrementalPCA.

Projection procedure per chunk:

```text
1. Extract 768-dimensional train embeddings.
2. Fit IncrementalPCA on train split only.
3. Transform train, validation, and test splits with the train-fitted PCA.
4. Save final 256-dimensional embeddings.
```

This avoids validation/test leakage because PCA is fitted only on the training split.

Observed explained variance ratios:

```text
chunk1 PCA 768→256 explained variance sum: 0.9876773542022481
chunk2 PCA 768→256 explained variance sum: 0.9776208752921536
chunk3 PCA 768→256 explained variance sum: 0.9805763307622046
```

PCA files were generated during the run:

```text
outputs/embeddings/FinBERT/chunk1_pca_768_to_256.pkl
outputs/embeddings/FinBERT/chunk2_pca_768_to_256.pkl
outputs/embeddings/FinBERT/chunk3_pca_768_to_256.pkl
```

PCA files are training/projection artefacts and were excluded from the final minimal deliverable commit after the final 256-dimensional embeddings were generated.

---

## 18. Final Embedding Output Contract

All downstream modules should treat the final text embeddings as the stable public interface.

Input to downstream model per row:

```text
shape: (256,)
dtype: float32
```

Batch input:

```text
shape: (batch_size, 256)
dtype: float32
```

Final files:

```text
outputs/embeddings/FinBERT/chunk1_train_embeddings.npy
outputs/embeddings/FinBERT/chunk1_val_embeddings.npy
outputs/embeddings/FinBERT/chunk1_test_embeddings.npy
outputs/embeddings/FinBERT/chunk2_train_embeddings.npy
outputs/embeddings/FinBERT/chunk2_val_embeddings.npy
outputs/embeddings/FinBERT/chunk2_test_embeddings.npy
outputs/embeddings/FinBERT/chunk3_train_embeddings.npy
outputs/embeddings/FinBERT/chunk3_val_embeddings.npy
outputs/embeddings/FinBERT/chunk3_test_embeddings.npy
```

Final verified shapes:

```text
chunk1_train: (189244, 256)
chunk1_val:   (40000, 256)
chunk1_test:  (40000, 256)

chunk2_train: (320000, 256)
chunk2_val:   (40000, 256)
chunk2_test:  (40000, 256)

chunk3_train: (240000, 256)
chunk3_val:   (40000, 256)
chunk3_test:  (40000, 256)
```

Important correction:

```text
An earlier smoke-test file produced chunk3_val with shape (1000, 256).
This was detected, overwritten, regenerated, verified, amended, and pushed.
The final correct chunk3_val shape is (40000, 256).
```

Verification command:

```bash
python -c 'import numpy as np; from pathlib import Path; base=Path("outputs/embeddings/FinBERT"); [print(f"chunk{c}_{s}", np.load(base/f"chunk{c}_{s}_embeddings.npy", mmap_mode="r").shape) for c in (1,2,3) for s in ("train","val","test")]'
```

Expected output:

```text
chunk1_train (189244, 256)
chunk1_val (40000, 256)
chunk1_test (40000, 256)
chunk2_train (320000, 256)
chunk2_val (40000, 256)
chunk2_test (40000, 256)
chunk3_train (240000, 256)
chunk3_val (40000, 256)
chunk3_test (40000, 256)
```

---

## 19. Metadata Output Contract

Each embedding matrix has a matching metadata file.

Metadata files:

```text
outputs/embeddings/FinBERT/chunk1_train_metadata.csv
outputs/embeddings/FinBERT/chunk1_val_metadata.csv
outputs/embeddings/FinBERT/chunk1_test_metadata.csv
outputs/embeddings/FinBERT/chunk2_train_metadata.csv
outputs/embeddings/FinBERT/chunk2_val_metadata.csv
outputs/embeddings/FinBERT/chunk2_test_metadata.csv
outputs/embeddings/FinBERT/chunk3_train_metadata.csv
outputs/embeddings/FinBERT/chunk3_val_metadata.csv
outputs/embeddings/FinBERT/chunk3_test_metadata.csv
```

The row order of each metadata CSV must match the row order of the corresponding `.npy` file exactly.

Expected metadata fields:

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

Downstream loaders must enforce:

```text
number of embedding rows == number of metadata rows
metadata year values match expected split years
metadata order is never shuffled unless embeddings are shuffled identically
```

---

## 20. Manifest Output Contract

Each final 256-dimensional embedding file has a manifest.

Manifest files:

```text
outputs/embeddings/FinBERT/chunk1_train_manifest.json
outputs/embeddings/FinBERT/chunk1_val_manifest.json
outputs/embeddings/FinBERT/chunk1_test_manifest.json
outputs/embeddings/FinBERT/chunk2_train_manifest.json
outputs/embeddings/FinBERT/chunk2_val_manifest.json
outputs/embeddings/FinBERT/chunk2_test_manifest.json
outputs/embeddings/FinBERT/chunk3_train_manifest.json
outputs/embeddings/FinBERT/chunk3_val_manifest.json
outputs/embeddings/FinBERT/chunk3_test_manifest.json
```

Expected manifest fields:

```text
chunk_id
split
rows
dim
embedding_file
metadata_file
pca_file
sha256_embeddings
created_at
projection
```

The `pca_file` field may reference a PCA file that existed during generation but was not retained in the minimal final commit. The final `.npy` embeddings are the stable deliverable.

---

## 21. Storage Format Policy

Allowed formats:

```text
.npy   final embeddings
.csv   metadata
.json  manifests and configuration
.pt    temporary PyTorch checkpoints during training only
.pkl   temporary PCA files during projection only
Hugging Face model directories for exported models
```

Disallowed for this module:

```text
Parquet
```

The final committed deliverable intentionally uses `.npy`, `.csv`, `.json`, and Hugging Face `model.safetensors` directories.

---

## 22. Final Deliverable Contents

The final pushed deliverable includes the necessary files for downstream use:

```text
code/encoders/finbert_encoder.py
code/encoders/run_finbert_full_pipeline.py
code/encoders/run_finbert_resume_after_hpo.py
outputs/embeddings/FinBERT/chunk*_train_embeddings.npy
outputs/embeddings/FinBERT/chunk*_val_embeddings.npy
outputs/embeddings/FinBERT/chunk*_test_embeddings.npy
outputs/embeddings/FinBERT/chunk*_train_metadata.csv
outputs/embeddings/FinBERT/chunk*_val_metadata.csv
outputs/embeddings/FinBERT/chunk*_test_metadata.csv
outputs/embeddings/FinBERT/chunk*_train_manifest.json
outputs/embeddings/FinBERT/chunk*_val_manifest.json
outputs/embeddings/FinBERT/chunk*_test_manifest.json
outputs/models/FinBERT/chunk*/model_freezed/
outputs/models/FinBERT/chunk*/model_unfreezed/
```

The final pushed commit:

```text
88c8817 add final FinBERT encoder models and embeddings
```

Final push confirmation:

```text
0d9a597..88c8817  main -> main
```

---

## 23. Excluded Artefacts

The following were excluded from the final deliverable because they are training/intermediate artefacts:

```text
outputs/models/FinBERT/hpo/
outputs/models/FinBERT/latest_checkpoint.pt
outputs/models/FinBERT/**/latest_checkpoint.pt
outputs/models/FinBERT/**/best_checkpoint.pt
outputs/models/FinBERT/**/epoch_*.pt
outputs/embeddings/FinBERT/*embeddings768.npy
outputs/embeddings/FinBERT/*manifest768.json
outputs/embeddings/FinBERT/*pca_768_to_256.pkl
outputs/embeddings/FinBERT/*pca_manifest.json
outputs/codeResults/FinBERT/hpo/
outputs/codeResults/FinBERT/*.log
outputs/results/FinBERT/hpo/
```

Recommended `.gitignore` rules:

```text
# FinBERT non-deliverable heavy artefacts
outputs/models/FinBERT/hpo/
outputs/models/FinBERT/latest_checkpoint.pt
outputs/models/FinBERT/**/latest_checkpoint.pt
outputs/models/FinBERT/**/best_checkpoint.pt
outputs/models/FinBERT/**/epoch_*.pt
outputs/embeddings/FinBERT/*embeddings768.npy
outputs/embeddings/FinBERT/*manifest768.json
outputs/embeddings/FinBERT/*pca_768_to_256.pkl
outputs/embeddings/FinBERT/*pca_manifest.json
outputs/codeResults/FinBERT/hpo/
outputs/codeResults/FinBERT/*.log
outputs/results/FinBERT/
```

Recommended safety check before committing:

```bash
git diff --cached --name-only | grep -E "checkpoint|epoch_.*\.pt|/hpo/|embeddings768|manifest768|pca_768|pca_manifest|\.log" && echo "BAD FILES STAGED - STOP" || echo "STAGED FILES ARE CLEAN"
```

Recommended tracked-file check:

```bash
git ls-files | grep -E "checkpoint|epoch_.*\.pt|/hpo/|embeddings768|manifest768|pca_768|pca_manifest|\.log" && echo "BAD TRACKED FILES - STOP" || echo "TRACKED FILES ARE CLEAN"
```

---

## 24. Git LFS Policy

Large binary files should be tracked with Git LFS.

Recommended LFS tracking:

```bash
git lfs track "outputs/embeddings/FinBERT/*.npy" "outputs/models/FinBERT/**/*.safetensors"
```

Useful verification:

```bash
git lfs ls-files | grep -E "outputs/embeddings/FinBERT|outputs/models/FinBERT"
```

Final push uploaded the relevant LFS objects successfully.

---

## 25. Downstream Loader Requirements

Any downstream loader for the text encoder should implement the following checks:

1. Verify `.npy` file exists.
2. Verify matching metadata CSV exists.
3. Verify matching manifest JSON exists.
4. Load `.npy` with `mmap_mode='r'` for memory efficiency where possible.
5. Confirm embedding shape is two-dimensional.
6. Confirm embedding dimension is 256.
7. Confirm number of metadata rows equals number of embedding rows.
8. Confirm split-year metadata matches expected chronological years.
9. Avoid random shuffling unless embeddings and metadata are shuffled together.
10. Preserve row indices for explainability and traceability.

Recommended loader target:

```text
code/analysts/text_embedding_loader.py
```

---

## 26. Downstream Analyst Interface

The Sentiment Analyst and News Analyst should consume the final 256-dimensional embeddings.

Recommended input contract:

```python
embedding: np.ndarray  # shape (256,), dtype float32
metadata: dict         # row-aligned SEC filing metadata
```

Recommended batch contract:

```python
embeddings: np.ndarray  # shape (batch_size, 256), dtype float32
metadata_df: DataFrame  # length == batch_size
```

Recommended MLP preference:

```text
Use tanh for hidden activations.
Use sigmoid only when mathematically required for binary probability output.
```

Suggested downstream modules:

```text
code/analysts/text_embedding_loader.py
code/analysts/sentiment_analyst.py
code/analysts/news_analyst.py
```

---

## 27. Recommended Sentiment Analyst Specification

Input:

```text
256-dimensional FinBERT text embedding
optional metadata features such as year, form_type, word_count, chunk_index
```

Possible outputs:

```text
sentiment_score in [-1, 1]
uncertainty_score
64-dimensional sentiment representation for fusion
```

Recommended architecture:

```text
Input 256
LayerNorm
Linear 256 → 128
Tanh
Dropout
Linear 128 → 64
Tanh
Dropout
Output head(s)
```

Important note:

```text
Do not claim supervised sentiment learning unless labels exist.
Initial implementation may support dummy labels, pseudo-label hooks, and future supervised labels.
```

---

## 28. Recommended News Analyst Specification

Input:

```text
256-dimensional text embedding
metadata describing filing year, form type, document id, ticker/CIK linkage, and filing date
```

Possible outputs:

```text
news/event importance score
risk relevance score
novelty/event intensity score
64-dimensional or 128-dimensional analyst representation for fusion
```

Recommended architecture:

```text
Chunk-level embedding encoder
Optional document-level aggregation
Optional ticker-date-level aggregation
MLP scoring heads with tanh hidden activations
```

Potential aggregation methods:

```text
mean pooling across chunks per document
attention pooling across chunks per document
form-type-aware pooling
recency-weighted pooling across filings per ticker/date
```

---

## 29. Future Supervised Label Generation

Supervised labels should be generated from market and fundamentals data using:

```text
CIK
ticker mapping
filing_date
future stock prices
market benchmark prices
trading calendar
```

Recommended labels:

```text
future_excess_return_10d_class
future_excess_return_30d_class
future_volatility_spike_30d
future_drawdown_risk_30d
future_realised_volatility_30d
future_abnormal_return_5d
future_abnormal_return_20d
```

Anti-leakage rules:

```text
Use only prices after the filing date.
If intraday filing timestamp is unavailable, use the next trading day as event start.
Compute classification thresholds using training years only.
Apply training thresholds unchanged to validation and test years.
Attach document-level labels to all chunks of the same document.
Do not use validation/test distributions to set thresholds.
Do not standardise using validation/test statistics.
```

---

## 30. Chunk-Level to Document-Level Aggregation

Current embeddings are chunk-level. Downstream financial targets will usually be document-level, ticker-date-level, or asset-date-level.

Potential hierarchy:

```text
chunk embedding → document embedding → ticker-date embedding → asset risk input
```

Recommended document-level aggregation options:

```text
simple mean pooling over chunks belonging to the same doc_id
attention pooling where attention weights are learned during supervised training
max/mean hybrid pooling
form-type-aware pooling
risk-section-weighted pooling if section metadata is available
```

Recommended ticker-date aggregation options:

```text
most recent filing embedding
mean of all filings within a lookback window
recency-weighted average
event-type weighted average
attention over filings in lookback window
```

---

## 31. Reproducibility

Random seed:

```text
42
```

Precision:

```text
mixed precision fp16 during CUDA training
float32 for saved embeddings
```

Final embedding format:

```text
.npy
```

Metadata format:

```text
.csv
```

Manifest/configuration format:

```text
.json
```

Model export format:

```text
Hugging Face directory with model.safetensors
```

Parquet usage:

```text
Not used for this module
```

---

## 32. Main Commands

Inspect dataset:

```bash
python code/encoders/finbert_encoder.py inspect
```

Train one chunk:

```bash
python code/encoders/finbert_encoder.py train-mlm --chunk 3 --epochs 3 --workers 6
```

Run HPO:

```bash
python code/encoders/finbert_encoder.py hpo --chunk 3 --trials 12 --max-rows 30000 --workers 6
```

Extract raw 768 embeddings for one split:

```bash
python code/encoders/finbert_encoder.py embed768 --chunk 3 --split val --workers 6 --eval-batch-size 64 --overwrite
```

Fit PCA:

```bash
python code/encoders/finbert_encoder.py fit-pca --chunk 3 --overwrite
```

Project to final 256 dimensions:

```bash
python code/encoders/finbert_encoder.py project-pca --chunk 3 --split val --overwrite
```

Run complete embedding pipeline:

```bash
python code/encoders/finbert_encoder.py embed-all
```

Run resume-after-HPO full pipeline:

```bash
python -u code/encoders/run_finbert_resume_after_hpo.py
```

---

## 33. Production Run Command

The production run was executed through the resume-after-HPO runner:

```bash
python -u code/encoders/run_finbert_resume_after_hpo.py 2>&1 | tee outputs/codeResults/FinBERT/finbert_resume_$(date +%Y%m%d_%H%M%S).log
```

This command:

```text
loads saved HPO best parameters
does not rerun HPO
trains/resumes chunks 1, 2, and 3
exports model_freezed and model_unfreezed per chunk
extracts raw 768-dimensional embeddings
fits IncrementalPCA on training split only
saves final 256-dimensional embeddings
writes metadata and manifest files
```

---

## 34. Validation Checklist

Before using the text embeddings downstream, run:

```bash
python -c 'import numpy as np; from pathlib import Path; base=Path("outputs/embeddings/FinBERT"); [print(f"chunk{c}_{s}", np.load(base/f"chunk{c}_{s}_embeddings.npy", mmap_mode="r").shape) for c in (1,2,3) for s in ("train","val","test")]'
```

Required output:

```text
chunk1_train (189244, 256)
chunk1_val (40000, 256)
chunk1_test (40000, 256)
chunk2_train (320000, 256)
chunk2_val (40000, 256)
chunk2_test (40000, 256)
chunk3_train (240000, 256)
chunk3_val (40000, 256)
chunk3_test (40000, 256)
```

Check no forbidden files are staged:

```bash
git diff --cached --name-only | grep -E "checkpoint|epoch_.*\.pt|/hpo/|embeddings768|manifest768|pca_768|pca_manifest|\.log" && echo "BAD FILES STAGED - STOP" || echo "STAGED FILES ARE CLEAN"
```

Check no forbidden files are tracked:

```bash
git ls-files | grep -E "checkpoint|epoch_.*\.pt|/hpo/|embeddings768|manifest768|pca_768|pca_manifest|\.log" && echo "BAD TRACKED FILES - STOP" || echo "TRACKED FILES ARE CLEAN"
```

---

## 35. Final Status

Final status:

```text
SEC filing chunk dataset is complete.
FinBERT domain-adaptive MLM fine-tuning is complete.
CUDA training on RTX 3090 Ti succeeded.
Chunk 1, chunk 2, and chunk 3 models were exported.
Final 256-dimensional train/validation/test embeddings were generated for all chunks.
The incorrect chunk3_val smoke-test embedding was detected and replaced.
Final embedding shapes were verified.
Unnecessary checkpoints, raw 768 embeddings, PCA files, and HPO trial artefacts were excluded from the final deliverable.
Final deliverables were committed and pushed.
```

Final pushed commit:

```text
88c8817 add final FinBERT encoder models and embeddings
```

The FinBERT Text Encoder is now ready for downstream modules, especially:

```text
text_embedding_loader.py
sentiment_analyst.py
news_analyst.py
fusion model input preparation
risk engine integration
```
