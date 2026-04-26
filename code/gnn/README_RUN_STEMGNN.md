# Fixed StemGNN GNN Files — Drop-in Replacement

Copy these files into:

```bash
fin-glassbox/code/gnn/
```

Files:

```text
stemgnn_base_model.py
stemgnn_contagion.py
stemgnn_forecast_dataloader.py
stemgnn_handler.py
stemgnn_utils.py
__init__.py
```

## 1. Compile check

```bash
cd ~/fin-glassbox
python -m py_compile code/gnn/stemgnn_base_model.py \
  code/gnn/stemgnn_contagion.py \
  code/gnn/stemgnn_forecast_dataloader.py \
  code/gnn/stemgnn_handler.py \
  code/gnn/stemgnn_utils.py
```

## 2. Inspect data and hardware

```bash
python code/gnn/stemgnn_contagion.py inspect --repo-root . --device cuda --num-workers 6 --cpu-threads 6
```

## 3. Minimum-time smoke test using synthetic data

This checks imports, model construction, target building, forward pass, backward pass, checkpoint save, and one validation pass. It does not touch the real 2,500-stock dataset.

```bash
python code/gnn/stemgnn_contagion.py smoke --repo-root . --device cuda --ticker-limit 32 --batch-size 2 --num-workers 0 --cpu-threads 6 --epochs 1 --max-train-windows 4 --max-eval-windows 2
```

## 4. Minimum-time smoke test using real data

This uses only the first 64 tickers and a few windows, so it should complete quickly.

```bash
python code/gnn/stemgnn_contagion.py smoke --repo-root . --device cuda --real --ticker-limit 64 --batch-size 2 --num-workers 0 --cpu-threads 6 --epochs 1 --max-train-windows 4 --max-eval-windows 2
```

## 5. Full chunk training

```bash
python code/gnn/stemgnn_contagion.py train-best --repo-root . --chunk 1 --device cuda --batch-size 8 --num-workers 6 --cpu-threads 6
```

## 6. Prediction and XAI

```bash
python code/gnn/stemgnn_contagion.py predict --repo-root . --chunk 1 --split test --device cuda --batch-size 8 --num-workers 2 --cpu-threads 6
```

## Notes

- The code assumes `returns_panel_wide.csv` contains daily log returns.
- Forward 5d/20d/60d targets are now computed by summing daily log returns over the horizon.
- BCEWithLogitsLoss is used instead of sigmoid + BCE for numerical stability.
- Class imbalance is handled with per-horizon `pos_weight`, capped by `max_pos_weight`.
- AMP, TF32, pinned memory, DataLoader prefetch, and persistent workers are supported.
- `torch.compile` is optional via `--compile`; do not use it for the minimum smoke test because compile overhead can dominate.
