# Model Code Requirements Guide

**Project:** `fin-glassbox` — Explainable Distributed Deep Learning Framework for Financial Risk Management  
**Purpose:** This guide defines the specific things that must be present in every model-code file written for this project.

---

## 1. File Identity

Every model file must clearly state:

```text
module name
project name
purpose of the model
input files
output files
training mode
whether the model is supervised, self-supervised, unsupervised, or rule-based
```

The top docstring must explain exactly what the file does and what it does not do.

---

## 2. No Fake Data Unless Explicitly Requested

Model files must not silently create dummy data, synthetic embeddings, fake labels, or placeholder targets.

Allowed behaviour:

```text
raise a clear missing-file error
run only when real required files exist
use small real-data subsets for smoke tests
```

Not allowed:

```text
auto-generating fake data/embeddings
training on random targets
```

---

## 3. Path Handling

Every model file must support:

```text
--repo-root
--env-file
project .env path resolution
relative and absolute paths
```

The file must respect the project path conventions:

```text
.npy  = embeddings and numerical arrays
.csv  = metadata, labels, predictions, training history
.json = manifests, configs, metrics, HPO results
.pt   = PyTorch checkpoints/models
.pkl  = only when necessary for sklearn/PCA objects
```

No Parquet files should be introduced.

---

## 4. Input Validation

Before training, the code must validate:

```text
all required files exist
all required columns exist
embedding shape is correct
metadata/label rows align with embedding rows
expected chronological split years are correct
train, validation, and test splits are separate
there is no temporal leakage
```

Errors must be clear and actionable.

---

## 5. Chronological Safety

Every model must follow the approved chronological chunks:

```text
chunk 1: train 2000–2004, validation 2005, test 2006
chunk 2: train 2007–2014, validation 2015, test 2016
chunk 3: train 2017–2022, validation 2023, test 2024
```

Training must use only the train split. Validation and test data must never be used to fit thresholds, scalers, preprocessors, PCA, vocabularies, or model weights.

---

## 6. Configuration Object

Every model file should include a dataclass configuration object containing:

```text
repo paths
input paths
output paths
model dimensions
training hyperparameters
HPO settings
batch sizes
seed
device
checkpoint paths
```

The resolved configuration must be saved as `.json` in `outputs/codeResults/...`.

---

## 7. Importable and Executable

Every model file must work both as:

```text
an importable Python module
an independently executable CLI script
```

Required structure:

```python
if __name__ == "__main__":
    main()
```

---

## 8. CLI Commands

Every model file must provide CLI subcommands where relevant:

```text
inspect       check input availability and alignment
hpo           run HPO for one chunk
hpo-all       run HPO for multiple chunks
train         train one model with explicit settings
train-all     train multiple chunks with explicit settings
train-best    train one model using saved HPO best parameters
train-best-all train multiple models using saved HPO best parameters
predict       export predictions for one split
predict-all   export predictions for multiple chunks/splits
```

Not every model needs every command, but supervised trainable models should follow this pattern.

---

## 9. Hyperparameter Search

Trainable neural modules must include HPO unless explicitly excluded.

HPO must:

```text
use Optuna/TPE when available
save SQLite study state
save best-params JSON
save trials CSV
support resume
support CPU-safe subset testing
never tune on test data
```

The normal workflow should be:

```text
inspect → hpo → train-best → predict
```

For all chunks:

```text
inspect → hpo-all → train-best-all → predict-all
```

---

## 10. Checkpointing and Resume

Every trainable model must save:

```text
epoch checkpoint after every epoch
latest.pt
best.pt
training history CSV
training summary JSON
```

Resume support must restore:

```text
model state
optimizer state
scheduler state if used
scaler state if mixed precision is used
epoch number
best validation score
training history
```

---

## 11. CPU and CUDA Support

Every PyTorch model must support:

```text
--device cpu
--device cuda
```

CPU execution must be practical for smoke tests. Useful CPU controls include:

```text
--num-workers
--torch-num-threads
--max-train-rows or --max-train-groups
--max-val-rows or --max-val-groups
```

CUDA execution should use PyTorch efficiently, including mixed precision where appropriate.

---

## 12. Model Architecture Standards

For MLP-style models:

```text
hidden activations must use tanh
use sigmoid only for final binary/probability outputs
use dropout where useful
use layer norm where useful
```

Model classes must expose enough information to reconstruct the model from a checkpoint.

---

## 13. Outputs

Each model must save outputs into the correct folders:

```text
outputs/models/...       checkpoints after every epoch and trained models(the checkpoints must be delted when the execution has finished)
outputs/results/...      metrics, predictions, training history
outputs/codeResults/...  configs, HPO files, manifests, logs
outputs/embeddings/...   downstream learned embeddings
```

Prediction outputs should include:

```text
identifiers
metadata
predicted scores/classes
confidence or uncertainty where available
target columns if available
path to any exported embedding file
```

---

## 14. Explainability Hooks

Every model must preserve enough information for explanation.

Examples:

```text
attention weights
feature/metadata columns used
row indices
source document IDs
chunk IDs
prediction confidence
module-level embeddings
```

The output must be traceable back to the original data row or document.

---

## 15. End-of-File Run Instructions

Every model file should end with commented one-line commands showing:

```text
syntax check
inspection command
CPU-safe HPO smoke test
train-best command
real HPO command
real final training command
prediction/export command
```

Commands must be one line only. Do not use backslash line continuations.

---

## 16. Documentation Style

Documentation and comments must use British English.

All explanations must be precise. Do not describe a file as complete, supervised, final, or production-ready unless the code actually supports that claim.

