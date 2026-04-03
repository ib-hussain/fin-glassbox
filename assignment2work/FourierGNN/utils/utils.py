# -*- coding:utf-8 -*-
"""
This module provides standard utility operations for loss calculations, model saving/loading, and slicing logic.

System Arguments:
    This module only provides helper functions and expects no system arguments.
"""
"""

Author:
    Weichen Shen,weichenswc@163.com

"""
import os
import dotenv
dotenv.load_dotenv()  # Load environment variables from .env file
debugOption = bool(int(os.getenv("DEBUG_MODE", "0")))
import numpy as np
import torch
from permetrics import RegressionMetric

def concat_fun(inputs, axis=-1):
    if debugOption: print(f"[Debug_Output]: Function 'concat_fun' called with inputs type={type(inputs)}, axis={axis}")
    """
    Concatenates PyTorch tensors intelligently avoiding unnecessary graph connections if sequences contain singular entries.
    
    Args:
        inputs (list): Tensors arrays ready for mapping. Example: [tensor_1, tensor2]
        axis (int): Concatenation target logic limit. Example: -1
    """
    if len(inputs) == 1:
        return inputs[0]
    else:
        return torch.cat(inputs, dim=axis)
def slice_arrays(arrays, start=None, stop=None):
    if debugOption: print(f"[Debug_Output]: Function 'slice_arrays' called with arrays type={type(arrays)}, start={start}, stop={stop}")
    """Slice an array or list of arrays.

    This takes an array-like, or a list of
    array-likes, and outputs:
        - arrays[start:stop] if `arrays` is an array-like
        - [x[start:stop] for x in arrays] if `arrays` is a list

    Can also work on list/array of indices: `slice_arrays(x, indices)`

    Arguments:
        arrays: Single array or list of arrays.
        start: can be an integer index (start index)
            or a list/array of indices
        stop: integer (stop index); should be None if
            `start` was a list.

    Returns:
        A slice of the array(s).

    Raises:
        ValueError: If the value of start is a list and stop is not None.
    """

    if arrays is None:
        return [None]

    if isinstance(arrays, np.ndarray):
        arrays = [arrays]

    if isinstance(start, list) and stop is not None:
        raise ValueError("The stop argument has to be None if the value of start "
                         "is a list.")
    elif isinstance(arrays, list):
        if hasattr(start, "__len__"):
            # hdf5 datasets only support list objects as indices
            if hasattr(start, "shape"):
                start = start.tolist()
            return [None if x is None else x[start] for x in arrays]
        else:
            if len(arrays) == 1:
                return arrays[0][start:stop]
            return [None if x is None else x[start:stop] for x in arrays]
    else:
        if hasattr(start, "__len__"):
            if hasattr(start, "shape"):
                start = start.tolist()
            return arrays[start]
        elif hasattr(start, "__getitem__"):
            return arrays[start:stop]
        else:
            return [None]
def save_model(model, model_dir, epoch=None):
    if debugOption: print(f"[Debug_Output]: Function 'save_model' called with model_dir={model_dir}, epoch={epoch}")
    """
    Dumps standard model representations sequentially tagging files leveraging specific epoch marks.
    
    Args:
        model (nn.Module): Trainee mapping model.
        model_dir (str): Logging target base directory constraint. Example: 'output/ECG'
        epoch (int): Naming sequence offset index representation. Example: 10
    """
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ""
    file_name = os.path.join(model_dir, epoch + "_dhfm.pt")
    with open(file_name, "wb") as f:
        torch.save(model, f)
def load_model(model_dir, epoch=None):
    if debugOption: print(f"[Debug_Output]: Function 'load_model' called with model_dir={model_dir}, epoch={epoch}")
    """
    Loads saved `.pt` states conditionally logic mapping files matching directory configurations over specific epochs.
    
    Args:
        model_dir (str): Location constraint directory variable. Example: 'output/ECG'
        epoch (int): Trailing naming offset sequences index locator limit.
        
    Returns:
        nn.Module: Instantiated populated system models.
    """
    if not model_dir:
        return
    epoch = str(epoch) if epoch else ""
    file_name = os.path.join(model_dir, epoch + "_dhfm.pt")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(file_name):
        return
    with open(file_name, "rb") as f:
        model = torch.load(f)
    return model
def masked_MAPE(v, v_, axis=None):
    if debugOption: print(f"[Debug_Output]: Function 'masked_MAPE' called with axis={axis}")
    """
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAPE averages on all elements of input.
    """
    mask = v == 0
    percentage = np.abs(v_ - v) / np.abs(v)
    if np.any(mask):
        masked_array = np.ma.masked_array(percentage, mask=mask)  # mask the dividing-zero as invalid
        result = masked_array.mean(axis=axis)
        if isinstance(result, np.ma.MaskedArray):
            return result.filled(np.nan)
        else:
            return result
    return np.mean(percentage, axis).astype(np.float64)
def MAPE(v, v_, axis=None):
    if debugOption: print(f"[Debug_Output]: Function 'MAPE' called with axis={axis}")
    """
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAPE averages on all elements of input.
    """
    mape = (np.abs(v_ - v) / (np.abs(v) + 1e-5)).astype(np.float64)
    mape = np.where(mape > 5, 5, mape)
    return np.mean(mape, axis)
# def MAPE(true, pred):
#    return np.mean(np.abs((pred - true) / (true+1e-5)))
def smape(P, A):
    if debugOption: print(f"[Debug_Output]: Function 'smape' called")
    """
    Calculates symmetric mean absolute percentage error limiting values by masking out 0 constraints.
    
    Args:
        P (np.array): Predicted vectors.
        A (np.array): Absolute Actual true values vectors.
    """
    nz = np.where(A > 0)
    Pz = P[nz]
    Az = A[nz]

    return np.mean(2 * np.abs(Az - Pz) / (np.abs(Az) + np.abs(Pz)))
def RMSE(v, v_, axis=None):
    if debugOption: print(f"[Debug_Output]: Function 'RMSE' called with axis={axis}")
    """
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, RMSE averages on all elements of input.
    """
    return np.sqrt(np.mean((v_ - v)**2, axis)).astype(np.float64)
def MAE(v, v_, axis=None):
    if debugOption: print(f"[Debug_Output]: Function 'MAE' called with axis={axis}")
    """
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAE averages on all elements of input.
    """
    return np.mean(np.abs(v_ - v), axis).astype(np.float64)
def a20_index(v, v_):
    if debugOption: print(f"[Debug_Output]: Function 'a20_index' called")
    """
    Resolves A20 indicator limits for threshold-driven accuracy verification processes relying upon RegressionMetric targets.
    
    Args:
        v: Authentic target boundaries matching vector.
        v_: Scaled evaluated outputs mapping limit vector.
    """
    evaluator = RegressionMetric(v.reshape(v.shape[0], -1), v_.reshape(v_.shape[0], -1))
    a20 = evaluator.a20_index()
    return np.mean(a20)
def evaluate(y, y_hat, by_step=False, by_node=False):
    if debugOption: print(f"[Debug_Output]: Function 'evaluate' called with by_step={by_step}, by_node={by_node}")
    """
    :param y: array in shape of [count, time_step, node].
    :param y_hat: in same shape with y.
    :param by_step: evaluate by time_step dim.
    :param by_node: evaluate by node dim.
    :return: array of mape, mae and rmse.
    """
    if not by_step and not by_node:
        return MAPE(y, y_hat), MAE(y, y_hat), RMSE(y, y_hat), a20_index(y, y_hat)
    if by_step and by_node:
        return MAPE(y, y_hat, axis=0), MAE(y, y_hat, axis=0), RMSE(y, y_hat, axis=0)
    if by_step:
        return (
            MAPE(y, y_hat, axis=(0, 2)),
            MAE(y, y_hat, axis=(0, 2)),
            RMSE(y, y_hat, axis=(0, 2)),
        )
    if by_node:
        return (
            MAPE(y, y_hat, axis=(0, 1)),
            MAE(y, y_hat, axis=(0, 1)),
            RMSE(y, y_hat, axis=(0, 1)),
        )