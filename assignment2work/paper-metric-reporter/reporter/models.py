"""
Lightweight data containers used across the reporter (actuals, predictions, portfolio series).

System arguments:
    None.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Data:
    """
    Wraps a numeric array with a human-readable model or series label.

    Attributes:
        data (numpy.ndarray): Prices or returns, shape ``(num_weeks, num_assets)`` (or similar).
        label (str): Short name for plots/tables. Example: ``\"FourierGNN\"``.
    """

    data: np.ndarray
    label: str

    def __repr__(self):
        return f"Data(label={self.label}, data={self.data})"

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Data(self.data[key], self.label)
        else:
            return Data(self.data[key:key + 1], self.label)


@dataclass
class DataPath:
    """
    Filesystem location plus display label (and optional return interpretation).

    Attributes:
        path (str): Directory with ``1.npy``…``num_weeks.npy`` or a CSV path for actuals.
            Example: ``\"../experiment-results/crypto/fouriergnn-results/20250318\"``.
        label (str): Model name shown in reports. Example: ``\"FourierGNN\"``.
        data_type (str | None): ``\"log-return\"`` if ``Reader`` should ``np.exp`` predictions;
            else ``None`` / ``\"simple-return\"`` for simple return factors.
    """

    path: str
    label: str
    data_type: Optional[str] = None

    def __repr__(self):
        return f"Path(label={self.label}, path={self.path})"


@dataclass
class PortfolioResult:
    """
    Time series of portfolio capital under a strategy, with optional predicted track and metadata.

    Attributes:
        data (numpy.ndarray): Realized capital path (weekly). Example shape ``(105,)``.
        label (str): Strategy name. Example: ``\"Random\"``.
        long_short_combination (tuple): ``(n_long, n_short)`` assets held each week.
        predicted_result_data (numpy.ndarray | None): Simulated capital if predictions drove picks.
        asset_selection_accuracy (float | None): Overlap metric vs oracle asset picks.
    """

    data: np.ndarray
    label: str
    long_short_combination: tuple
    predicted_result_data: Optional[np.ndarray] = None
    asset_selection_accuracy: Optional[float] = None

    def __repr__(self):
        if self.predicted_result_data is not None:
            return f"PortfolioResult(label={self.label}, change={self.data[0]}->{self.data[-1]}, predicted_change={self.data[0]}->{self.predicted_result_data[-1]}, long_short_combination={self.long_short_combination})"
        else:
            return f"PortfolioResult(label={self.label}, change={self.data[0]}->{self.data[-1]}, long_short_combination={self.long_short_combination})"

    def final_portfolio_value(self):
        """Last element of the realized capital array."""
        return self.data[-1]


@dataclass
class PortfolioValues:
    """
    Pair of actual and optional predicted value paths inside portfolio simulation.

    Attributes:
        actual (numpy.ndarray): Capital under actual prices.
        prediction (numpy.ndarray | None): Capital if forecast prices were used for returns.
    """

    actual: np.ndarray
    prediction: Optional[np.ndarray] = None

    def __repr__(self):
        return f"PortfolioValues(actual={self.actual}, prediction={self.prediction})"
