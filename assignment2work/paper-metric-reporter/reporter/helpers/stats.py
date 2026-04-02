"""
Scalar error and scaling helpers for tables in ``Reporter``.

System arguments:
    None.
"""

import numpy as np


class Stats:
    """MAPE, MBD, CFE, and row-wise standardization used in metric tables."""

    @staticmethod
    def mape(actual, prediction):
        """
        Mean absolute percentage error (as percentage points, i.e. scaled by 100).

        Args:
            actual (numpy.ndarray): Ground truth.
            prediction (numpy.ndarray): Model output (same shape as ``actual``).

        Returns:
            float: MAPE in percent. Example: ``5.2`` means 5.2%.
        """
        print(f"[Debug_Output]: Stats.mape | actual.shape={actual.shape} | prediction.shape={prediction.shape}")
        return np.mean(np.abs((actual - prediction) / actual)) * 100

    @staticmethod
    def std_mean(mat):
        """
        Per-column z-score then take mean across columns for each row (summary trajectory).

        Args:
            mat (numpy.ndarray): 2D series, shape ``(time, assets)``.

        Returns:
            numpy.ndarray: 1D length ``time`` row means of standardized ``mat``.
        """
        print(f"[Debug_Output]: Stats.std_mean | mat.shape={mat.shape}")
        mean = np.mean(mat, axis=0)
        std = np.std(mat, axis=0)

        std_adj = np.where(std < 1e-6, 1, std)

        standardized_mat = (mat - mean) / std_adj
        row_means = np.mean(standardized_mat, axis=1)
        return row_means

    @staticmethod
    def mbd(actual, prediction):
        """
        Mean bias deviation: average signed percentage error.

        Args:
            actual, prediction: Same-shaped arrays.

        Returns:
            float: Average ``100 * (pred - actual) / actual``.
        """
        print(f"[Debug_Output]: Stats.mbd | shapes actual={actual.shape} prediction={prediction.shape}")
        percentage_error = 100 * (prediction - actual) / actual
        return np.mean(percentage_error)

    @staticmethod
    def cfe(actual, prediction):
        """
        Cumulative forecast error: sum over time and assets of ``prediction - actual``.

        Args:
            actual, prediction: Same-shaped arrays.

        Returns:
            float: Scalar total error mass.
        """
        print(f"[Debug_Output]: Stats.cfe | shapes actual={actual.shape} prediction={prediction.shape}")
        forecast_error = prediction - actual
        cfe_per_column = np.sum(forecast_error, axis=0)
        total_cfe = np.sum(cfe_per_column)
        return total_cfe
