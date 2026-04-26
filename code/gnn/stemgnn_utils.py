"""Small metric utilities for StemGNN legacy forecasting code."""

from __future__ import annotations

import numpy as np


def masked_MAPE(v, v_, axis=None):
    v = np.asarray(v, dtype=np.float64)
    v_ = np.asarray(v_, dtype=np.float64)
    mask = np.isclose(v, 0.0)
    percentage = np.abs(v_ - v) / np.maximum(np.abs(v), 1e-12)
    if np.any(mask):
        percentage = np.ma.masked_array(percentage, mask=mask)
        result = percentage.mean(axis=axis)
        return result.filled(np.nan) if isinstance(result, np.ma.MaskedArray) else result
    return np.mean(percentage, axis=axis).astype(np.float64)


def MAPE(v, v_, axis=None):
    v = np.asarray(v, dtype=np.float64)
    v_ = np.asarray(v_, dtype=np.float64)
    mape = np.abs(v_ - v) / np.maximum(np.abs(v), 1e-12)
    mape = np.clip(mape, 0.0, 5.0)
    return np.mean(mape, axis=axis).astype(np.float64)


def RMSE(v, v_, axis=None):
    v = np.asarray(v, dtype=np.float64)
    v_ = np.asarray(v_, dtype=np.float64)
    return np.sqrt(np.mean((v_ - v) ** 2, axis=axis)).astype(np.float64)


def MAE(v, v_, axis=None):
    v = np.asarray(v, dtype=np.float64)
    v_ = np.asarray(v_, dtype=np.float64)
    return np.mean(np.abs(v_ - v), axis=axis).astype(np.float64)


def a20_index(v, v_):
    """Percentage of predictions within ±20% relative error.

    This removes the external `permetrics` dependency from the uploaded file.
    """
    v = np.asarray(v, dtype=np.float64).reshape(v.shape[0], -1)
    v_ = np.asarray(v_, dtype=np.float64).reshape(v_.shape[0], -1)
    denom = np.maximum(np.abs(v), 1e-12)
    return np.mean((np.abs(v_ - v) / denom) <= 0.20)


def evaluate(y, y_hat, by_step=False, by_node=False):
    y = np.asarray(y)
    y_hat = np.asarray(y_hat)
    if not by_step and not by_node:
        return MAPE(y, y_hat), MAE(y, y_hat), RMSE(y, y_hat), a20_index(y, y_hat)
    if by_step and by_node:
        return MAPE(y, y_hat, axis=0), MAE(y, y_hat, axis=0), RMSE(y, y_hat, axis=0)
    if by_step:
        return MAPE(y, y_hat, axis=(0, 2)), MAE(y, y_hat, axis=(0, 2)), RMSE(y, y_hat, axis=(0, 2))
    return MAPE(y, y_hat, axis=(0, 1)), MAE(y, y_hat, axis=(0, 1)), RMSE(y, y_hat, axis=(0, 1))
