"""Legacy StemGNN forecasting handler.

This file is retained for compatibility with the Assignment-2 style forecast
workflow. The current fin-glassbox contagion module is implemented in
`stemgnn_contagion.py`. This handler no longer imports non-existent packages
such as `data_loader.forecast_dataloader` or `models.base_model`; it uses the
local files in `code/gnn/`.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as torch_data
from tqdm import tqdm

from stemgnn_base_model import Model
from stemgnn_forecast_dataloader import ForecastDataset, de_normalized
from stemgnn_utils import evaluate


DEFAULT_MODEL_DIR = Path(os.getenv("modelsPathGlobal", "outputs/models")) / "StemGNN"
DEFAULT_RESULT_DIR = Path(os.getenv("result_file_StemGNN_path", "outputs/results/StemGNN/legacy"))


def save_model(model: torch.nn.Module, model_dir: str | Path = DEFAULT_MODEL_DIR, epoch=None) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{epoch}_" if epoch is not None else ""
    file_name = model_dir / f"{prefix}stemgnn.pt"
    torch.save(model.state_dict(), file_name)
    return file_name


def load_model(model: torch.nn.Module, model_dir: str | Path = DEFAULT_MODEL_DIR, epoch=None, device="cpu"):
    model_dir = Path(model_dir)
    prefix = f"{epoch}_" if epoch is not None else ""
    file_name = model_dir / f"{prefix}stemgnn.pt"
    if not file_name.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {file_name}")
    state = torch.load(file_name, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    return model


@torch.no_grad()
def inference(model, dataloader, device, node_cnt, window_size, horizon):
    forecast_set = []
    target_set = []
    model.eval()
    for inputs, target in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        step = 0
        forecast_steps = np.zeros([inputs.size(0), horizon, node_cnt], dtype=np.float32)
        rolling_inputs = inputs.clone()
        while step < horizon:
            forecast_result, _ = model(rolling_inputs)
            len_model_output = forecast_result.size(1)
            if len_model_output == 0:
                raise RuntimeError("Blank inference result")
            usable = min(horizon - step, len_model_output)
            rolling_inputs[:, :window_size - usable, :] = rolling_inputs[:, usable:window_size, :].clone()
            rolling_inputs[:, window_size - usable:, :] = forecast_result[:, :usable, :].clone()
            forecast_steps[:, step:step + usable, :] = forecast_result[:, :usable, :].detach().cpu().numpy()
            step += usable
        forecast_set.append(forecast_steps)
        target_set.append(target.detach().cpu().numpy())
    return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0)


def validate(
    model,
    dataloader,
    device,
    normalize_method,
    statistic,
    node_cnt,
    window_size,
    horizon,
    result_file: str | Path = DEFAULT_RESULT_DIR,
) -> Dict:
    start = datetime.now()
    forecast_norm, target_norm = inference(model, dataloader, device, node_cnt, window_size, horizon)
    if normalize_method and statistic:
        forecast = de_normalized(forecast_norm, normalize_method, statistic)
        target = de_normalized(target_norm, normalize_method, statistic)
    else:
        forecast, target = forecast_norm, target_norm

    score = evaluate(target, forecast)
    score_by_node = evaluate(target, forecast, by_node=True)
    score_norm = evaluate(target_norm, forecast_norm)

    print(f"NORM: MAPE {score_norm[0]:.6%}; MAE {score_norm[1]:.6f}; RMSE {score_norm[2]:.6f}.")
    print(f"RAW : MAPE {score[0]:.6%}; MAE {score[1]:.6f}; RMSE {score[2]:.6f}; A20 {score[3]:.6f}.")
    print(f"Validation time: {datetime.now() - start}")

    result_file = Path(result_file)
    result_file.mkdir(parents=True, exist_ok=True)
    np.savetxt(result_file / "target.csv", target[:, 0, :], delimiter=",")
    np.savetxt(result_file / "predict.csv", forecast[:, 0, :], delimiter=",")
    np.savetxt(result_file / "predict_abs_error.csv", np.abs(forecast[:, 0, :] - target[:, 0, :]), delimiter=",")

    return {
        "mae": score[1],
        "mae_node": score_by_node[1],
        "mape": score[0],
        "mape_node": score_by_node[0],
        "rmse": score[2],
        "rmse_node": score_by_node[2],
        "a20": score[3],
    }


def train(train_data, valid_data, args, result_file: str | Path = DEFAULT_RESULT_DIR):
    result_file = Path(result_file)
    result_file.mkdir(parents=True, exist_ok=True)
    node_count = train_data.shape[1]
    model = Model(node_count, 2, args.window_size, args.multi_layer, horizon=args.horizon, device=args.device)

    if len(train_data) == 0 or len(valid_data) == 0:
        raise ValueError("Cannot organize enough training/validation data")

    if args.norm_method == "z_score":
        train_mean = np.mean(train_data, axis=0)
        train_std = np.where(np.std(train_data, axis=0) < 1e-8, 1.0, np.std(train_data, axis=0))
        normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
    elif args.norm_method == "min_max":
        normalize_statistic = {"min": np.min(train_data, axis=0).tolist(), "max": np.max(train_data, axis=0).tolist()}
    else:
        normalize_statistic = None

    if normalize_statistic is not None:
        with open(result_file / "norm_stat.json", "w") as f:
            json.dump(normalize_statistic, f)

    if args.optimizer == "RMSProp":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, eps=1e-8)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate)

    train_set = ForecastDataset(train_data, args.window_size, args.horizon, args.norm_method, normalize_statistic)
    valid_set = ForecastDataset(valid_data, args.window_size, args.horizon, args.norm_method, normalize_statistic)

    workers = int(getattr(args, "num_workers", 0))
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=workers, pin_memory=(args.device == "cuda" and workers > 0))
    train_loader = torch_data.DataLoader(train_set, shuffle=True, drop_last=False, **loader_kwargs)
    valid_loader = torch_data.DataLoader(valid_set, shuffle=False, drop_last=False, **loader_kwargs)

    loss_fn = nn.MSELoss(reduction="mean").to(args.device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params: {total_params:,}")

    best_validate_mae = np.inf
    no_improve = 0
    performance_metrics = {}
    for epoch in tqdm(range(args.epoch), desc="StemGNN legacy training"):
        epoch_start_time = time.time()
        model.train()
        loss_total = 0.0
        count = 0
        for inputs, target in train_loader:
            inputs = inputs.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            forecast, _ = model(inputs)
            loss = loss_fn(forecast, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), getattr(args, "gradient_clip", 1.0))
            optimizer.step()
            loss_total += float(loss.detach().cpu())
            count += 1
        print(f"| epoch {epoch:3d} | time {time.time() - epoch_start_time:6.2f}s | train loss {loss_total / max(count, 1):.6f}")

        if (epoch + 1) % args.exponential_decay_step == 0:
            scheduler.step()
        if (epoch + 1) % args.validate_freq == 0:
            performance_metrics = validate(model, valid_loader, args.device, args.norm_method, normalize_statistic, node_count, args.window_size, args.horizon, result_file)
            if best_validate_mae > performance_metrics["mae"]:
                best_validate_mae = performance_metrics["mae"]
                no_improve = 0
                save_model(model, result_file)
            else:
                no_improve += 1
        if getattr(args, "early_stop", False) and no_improve >= getattr(args, "early_stop_step", 10):
            break
    return performance_metrics, normalize_statistic
