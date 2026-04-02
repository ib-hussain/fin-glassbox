"""
Load actual marked CSV prices and stacked model predictions from weekly ``.npy`` files.

System arguments:
    None.
"""

import numpy as np
import pandas as pd

from models import Data


class Reader:
    """Static helpers for reading inputs referenced by ``config.*Paths``."""

    @staticmethod
    def get_actual_prices(path, num_weeks):
        """
        Read the marked price CSV and take the last ``num_weeks`` rows where ``split_point`` is true.

        Args:
            path (str): CSV with ``Date``, ``split_point``, and asset price columns.
                Example: ``\"../ticker-collector/out/crypto/daily_20_2190_marked.csv\"``.
            num_weeks (int): How many evaluation weeks to keep (from the tail). Example: ``104``.

        Returns:
            Data: ``label=\"Actual Prices\"``, ``data`` shape ``(num_weeks, num_assets)``.
        """
        print(f"[Debug_Output]: Reader.get_actual_prices | path={path!r} | num_weeks={num_weeks}")
        actual_prices = pd.read_csv(path)
        return Data(
            actual_prices[actual_prices["split_point"]].drop(columns=["Date", "split_point"])[-num_weeks:].to_numpy(),
            "Actual Prices",
        )

    @staticmethod
    def get_predicted_returns(predictions_path, num_weeks, num_assets, data_type="simple-return"):
        """
        Stack per-week prediction files ``1.npy`` … ``num_weeks.npy`` into a return matrix.

        Args:
            predictions_path (str): Folder containing ``i.npy`` for each week ``i``.
                Example: ``\"../experiment-results/fx/lstm-results/202312062100/weeks\"``.
            num_weeks (int): Number of files to load. Example: ``104``.
            num_assets (int): Expected width after load (used for empty init shape).
                Example: ``10`` (FX) or ``20`` (crypto).
            data_type (str): ``\"log-return\"`` applies ``exp`` to match price scaling; else raw factors.

        Returns:
            numpy.ndarray: Shape ``(num_weeks, num_assets)`` predicted returns/factors.
        """
        print(
            f"[Debug_Output]: Reader.get_predicted_returns | path={predictions_path!r} | "
            f"num_weeks={num_weeks} | num_assets={num_assets} | data_type={data_type!r}"
        )
        predicted_prices = np.empty([0, num_assets])
        for i in range(1, num_weeks + 1):
            data = np.load(f"{predictions_path}/{i}.npy")[-1:]
            data = np.exp(data) if data_type == "log-return" else data
            predicted_prices = np.concatenate((predicted_prices, data))
        return predicted_prices
