# Project Setup

This document describes how to prepare a Linux or WSL2 environment for `fin-glassbox`, the repository for **An Explainable Multimodal Neural Framework for Financial Risk Management**.

The recommended development environment is Linux with Python 3.12.7, Git LFS, a virtual environment, and CUDA-capable PyTorch when GPU acceleration is available.

---

## Table of contents

- [Recommended environment](#recommended-environment)
- [System packages](#system-packages)
- [Clone the repository](#clone-the-repository)
- [Git LFS](#git-lfs)
- [Install Python 3.12.7 with pyenv](#install-python-3127-with-pyenv)
- [Create and activate the virtual environment](#create-and-activate-the-virtual-environment)
- [Install Python dependencies](#install-python-dependencies)
- [CUDA and GPU checks](#cuda-and-gpu-checks)
- [Expected repository paths](#expected-repository-paths)
- [Environment validation](#environment-validation)
- [Smoke-test commands](#smoke-test-commands)
- [Working with large data and outputs](#working-with-large-data-and-outputs)
- [Common troubleshooting](#common-troubleshooting)
- [Contributor notes](#contributor-notes)

---

## Recommended environment

Recommended baseline:

```text
Operating system: Ubuntu 22.04 or Ubuntu 24.04, native Linux or WSL2
Python: 3.12.7
Virtual environment: venv3.12.7
GPU: NVIDIA GPU recommended for model training and embedding generation
CUDA PyTorch: install according to the CUDA version available on the machine
Disk: large local storage recommended for SEC filings, market panels, embeddings, and model outputs
```

The project can run inspection and smaller data-processing tasks on CPU, but encoder training, embedding generation, graph modules, and neural module training are substantially faster with CUDA.

---

## System packages

Check Ubuntu version:

```bash
lsb_release -a
```

Update packages:

```bash
sudo apt update
```

Install base tools:

```bash
sudo apt install -y python3 python3-venv python3-distutils git git-lfs build-essential wget curl
```

Install build dependencies commonly needed by `pyenv` and scientific Python packages:

```bash
sudo apt install -y make zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev llvm xz-utils tk-dev libxml2-dev libxmlsec1-dev liblzma-dev
```

Optional monitoring tools:

```bash
sudo apt install -y htop tmux nvtop sysstat
```

---

## Clone the repository

Clone the repository and enter it:

```bash
git clone https://github.com/ib-hussain/fin-glassbox.git
```

```bash
cd fin-glassbox
```

If the repository is already cloned, enter the repository root:

```bash
cd ~/fin-glassbox
```

Use the actual path on your machine if the repository is stored elsewhere.

---

## Git LFS

The repository may use Git LFS for large files. Install and initialise Git LFS:

```bash
git lfs install
```

Pull LFS-tracked files:

```bash
git lfs pull
```

If local repository-level LFS initialisation is required:

```bash
git lfs install --local
```

---

## Install Python 3.12.7 with pyenv

Install `pyenv`:

```bash
curl https://pyenv.run | bash
```

Add `pyenv` to Bash startup configuration:

```bash
cat >> ~/.bashrc << 'PYENVEOF'
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
PYENVEOF
```

Reload the shell:

```bash
source ~/.bashrc
```

Verify `pyenv`:

```bash
pyenv --version
```

Install Python 3.12.7:

```bash
pyenv install 3.12.7
```

Set Python 3.12.7 locally for the repository:

```bash
cd ~/fin-glassbox && pyenv local 3.12.7
```

Verify Python version:

```bash
cd ~/fin-glassbox && python --version
```

Expected:

```text
Python 3.12.7
```

---

## Create and activate the virtual environment

Create the virtual environment:

```bash
cd ~/fin-glassbox && python -m venv venv3.12.7
```

Activate it:

```bash
cd ~/fin-glassbox && source venv3.12.7/bin/activate
```

Verify activation:

```bash
which python
```

```bash
python --version
```

```bash
pip --version
```

The Python path should point inside `venv3.12.7`.

---

## Install Python dependencies

Upgrade packaging tools:

```bash
cd ~/fin-glassbox && python -m pip install --upgrade pip setuptools wheel
```

Install repository dependencies:

```bash
cd ~/fin-glassbox && pip install -r requirements_linux_venv.txt
```

If the dependency file on a machine is an environment snapshot rather than a strict pip requirements file, regenerate a clean requirements file from a working environment using:

```bash
cd ~/fin-glassbox && pip freeze > requirements.txt
```

For CUDA-specific PyTorch, install the PyTorch wheel recommended for your CUDA and driver version, then rerun the rest of the dependency installation if needed.

Confirm important packages:

```bash
cd ~/fin-glassbox && python -c "import torch, pandas, numpy, sklearn, optuna, transformers; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('pandas=', pandas.__version__); print('numpy=', numpy.__version__); print('optuna=', optuna.__version__); print('transformers=', transformers.__version__)"
```

---

## CUDA and GPU checks

Check the NVIDIA driver and GPU state:

```bash
nvidia-smi
```

Check PyTorch CUDA availability:

```bash
cd ~/fin-glassbox && python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Monitor GPU usage during long runs:

```bash
watch -n 1 nvidia-smi
```

Use `tmux` for long-running training or embedding jobs:

```bash
tmux new -s fin-glassbox
```

Detach from a `tmux` session with `Ctrl+B`, then `D`.

Reattach:

```bash
tmux attach -t fin-glassbox
```

---

## Expected repository paths

The repository expects a consistent folder structure. Common paths include:

```text
code/encoders/
code/analysts/
code/gnn/
code/riskEngine/
code/fusion/
data/yFinance/processed/
data/FRED_data/outputs/
data/graphs/
data/sec_edgar/
outputs/embeddings/
outputs/models/
outputs/results/
outputs/codeResults/
outputs/cache/
```

Most CLI scripts accept:

```text
--repo-root .
--device cuda
--chunk 1
--split train|val|test
```

Run commands from the repository root unless a module-specific document says otherwise.

---

## Environment validation

Compile key Python files:

```bash
cd ~/fin-glassbox && python -m py_compile code/encoders/temporal_encoder.py code/encoders/finbert_encoder.py code/fusion/fusion_layer.py code/fusion/final_fusion.py
```

Inspect Fusion inputs:

```bash
cd ~/fin-glassbox && python code/fusion/final_fusion.py inspect --repo-root .
```

Run Fusion smoke test:

```bash
cd ~/fin-glassbox && python code/fusion/final_fusion.py smoke --repo-root . --device cuda
```

Run encoder inspection commands according to the module documentation:

```bash
cd ~/fin-glassbox && python code/encoders/temporal_encoder.py inspect --repo-root .
```

```bash
cd ~/fin-glassbox && python code/encoders/finbert_encoder.py --help
```

---

## Smoke-test commands

Use smoke tests before long training jobs. A smoke test should check imports, model construction, data shape assumptions, XAI output shape, and basic forward/backward behaviour.

Fusion:

```bash
cd ~/fin-glassbox && python code/fusion/final_fusion.py smoke --repo-root . --device cuda
```

Technical Analyst:

```bash
cd ~/fin-glassbox && python code/analysts/technical_analyst.py smoke --repo-root . --device cuda
```

Drawdown Risk Module:

```bash
cd ~/fin-glassbox && python code/riskEngine/drawdown.py smoke --repo-root . --device cuda
```

Volatility Risk Module:

```bash
cd ~/fin-glassbox && python code/riskEngine/volatility.py smoke --repo-root . --device cuda
```

MTGNN Regime Module:

```bash
cd ~/fin-glassbox && python code/gnn/mtgnn_regime.py smoke --repo-root . --device cuda
```

StemGNN Contagion Module:

```bash
cd ~/fin-glassbox && python code/gnn/stemgnn_contagion.py smoke --repo-root . --device cuda --ticker-limit 32 --batch-size 2 --num-workers 0 --cpu-threads 6 --epochs 1 --max-train-windows 4 --max-eval-windows 2
```

Use module-specific documentation for exact command options:

- [`code/encoders/README.md`](code/encoders/README.md)
- [`code/analysts/README.md`](code/analysts/README.md)
- [`code/gnn/README.md`](code/gnn/README.md)
- [`code/riskEngine/README.md`](code/riskEngine/README.md)
- [`code/fusion/README.md`](code/fusion/README.md)

---

## Working with large data and outputs

This project can generate very large files. Keep the following in mind:

- SEC filings, market panels, and embeddings can require substantial disk space.
- `.npy` embedding files can be several gigabytes.
- `outputs/` is a runtime artefact directory and should generally not be committed directly.
- Use Git LFS only for large files that must be versioned.
- Keep raw data, processed data, checkpoints, and predictions organised by module and chunk.
- Avoid moving or renaming generated artefacts without updating downstream module paths.

Recommended disk checks:

```bash
df -h
```

```bash
du -h --max-depth=2 outputs | sort -h | tail -30
```

```bash
du -h --max-depth=2 data | sort -h | tail -30
```

---

## Common troubleshooting

### `ModuleNotFoundError`

Run commands from the repository root:

```bash
cd ~/fin-glassbox
```

Activate the virtual environment:

```bash
source venv3.12.7/bin/activate
```

Reinstall requirements if needed:

```bash
pip install -r requirements_linux_venv.txt
```

### CUDA is not detected

Check driver:

```bash
nvidia-smi
```

Check PyTorch:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If CUDA is unavailable, reinstall PyTorch with the correct CUDA wheel for your driver/CUDA setup.

### Out-of-memory errors

Reduce batch size or node limit. Examples:

```bash
cd ~/fin-glassbox && python code/fusion/final_fusion.py smoke --repo-root . --device cuda --batch-size 512
```

```bash
cd ~/fin-glassbox && python code/gnn/mtgnn_regime.py predict --repo-root . --chunk 1 --split test --device cuda --node-limit 512
```

Close other GPU jobs or inspect memory:

```bash
nvidia-smi
```

### Old schema detected

Some downstream modules require specific output schemas. If a module reports old schema columns, rerun the upstream module that produced the stale output.

For Fusion, Quantitative Analyst outputs must include trained attention columns such as:

```text
top_attention_risk_driver
attention_pooled_risk_score
risk_attention_volatility
risk_attention_drawdown
risk_attention_var_cvar
risk_attention_contagion
risk_attention_liquidity
risk_attention_regime
```

### Non-finite loss or NaN outputs

Check input finite ratios using the module’s `inspect` command. Common causes include:

- non-finite targets,
- unnormalised extreme values,
- old cached files,
- incompatible checkpoints,
- too-large learning rate,
- stale HPO databases,
- mismatched model config and checkpoint architecture.

Use `--fresh` where appropriate to remove stale checkpoints or HPO databases.

### Too many open files

Raise the open-file limit for the shell session:

```bash
ulimit -n 4096
```

For very multiprocessing-heavy jobs, reduce worker count or avoid excessive parallel DataLoader workers.

---

## Contributor notes

- Keep commands reproducible and run them from the repository root.
- Add or update module-level markdown when changing module behaviour.
- Preserve CLI consistency where possible: `inspect`, `smoke`, `hpo`, `train-best`, `predict`, `predict-all`, `validate`.
- Preserve XAI outputs when adding new predictions.
- Do not introduce silent schema changes; downstream integration depends on stable column names.
- Avoid look-ahead leakage in all financial tasks.
- Fit scalers, PCA, and normalisers on training splits only.
- Use chronological splits for evaluation.
