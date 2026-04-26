"""Forecast data utilities used by the StemGNN baseline/legacy handler."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.utils.data as torch_data


def normalized(data: np.ndarray, normalize_method: str, norm_statistic: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
    data = np.asarray(data, dtype=np.float32)
    if normalize_method is None or normalize_method == "none":
        return data, norm_statistic or {}

    if normalize_method == "min_max":
        if norm_statistic is None:
            norm_statistic = {"max": np.max(data, axis=0), "min": np.min(data, axis=0)}
        max_v = np.asarray(norm_statistic["max"], dtype=np.float32)
        min_v = np.asarray(norm_statistic["min"], dtype=np.float32)
        scale = np.maximum(max_v - min_v, 1e-8)
        return np.clip((data - min_v) / scale, 0.0, 1.0).astype(np.float32), norm_statistic

    if normalize_method == "z_score":
        if norm_statistic is None:
            norm_statistic = {"mean": np.mean(data, axis=0), "std": np.std(data, axis=0)}
        mean = np.asarray(norm_statistic["mean"], dtype=np.float32)
        std = np.asarray(norm_statistic["std"], dtype=np.float32)
        std = np.where(std < 1e-8, 1.0, std)
        norm_statistic["std"] = std
        return ((data - mean) / std).astype(np.float32), norm_statistic

    raise ValueError(f"Unknown normalize_method: {normalize_method}")


def de_normalized(data: np.ndarray, normalize_method: str, norm_statistic: Optional[Dict]) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if normalize_method is None or normalize_method == "none" or not norm_statistic:
        return data

    if normalize_method == "min_max":
        max_v = np.asarray(norm_statistic["max"], dtype=np.float32)
        min_v = np.asarray(norm_statistic["min"], dtype=np.float32)
        scale = np.maximum(max_v - min_v, 1e-8)
        return data * scale + min_v

    if normalize_method == "z_score":
        mean = np.asarray(norm_statistic["mean"], dtype=np.float32)
        std = np.asarray(norm_statistic["std"], dtype=np.float32)
        std = np.where(std < 1e-8, 1.0, std)
        return data * std + mean

    raise ValueError(f"Unknown normalize_method: {normalize_method}")


class ForecastDataset(torch_data.Dataset):
    """Sliding-window dataset for baseline forecasting tasks."""

    def __init__(
        self,
        df,
        window_size: int,
        horizon: int,
        normalize_method: Optional[str] = None,
        norm_statistic: Optional[Dict] = None,
        interval: int = 1,
    ) -> None:
        self.window_size = int(window_size)
        self.interval = int(interval)
        self.horizon = int(horizon)
        self.normalize_method = normalize_method
        self.norm_statistic = norm_statistic

        values = pd.DataFrame(df).ffill().bfill().values.astype(np.float32)
        if normalize_method:
            values, self.norm_statistic = normalized(values, normalize_method, norm_statistic)
        self.data = values
        self.df_length = len(values)
        self.x_end_idx = self.get_x_end_idx()

    def __getitem__(self, index: int):
        hi = self.x_end_idx[index]
        lo = hi - self.window_size
        x = torch.from_numpy(self.data[lo:hi]).float()
        y = torch.from_numpy(self.data[hi:hi + self.horizon]).float()
        return x, y

    def __len__(self) -> int:
        return len(self.x_end_idx)

    def get_x_end_idx(self):
        return list(range(self.window_size, self.df_length - self.horizon + 1, self.interval))
