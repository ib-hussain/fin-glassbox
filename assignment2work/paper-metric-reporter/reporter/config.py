"""
Static configuration for the paper metric reporter: filesystem paths to actual CSVs and to each
model's prediction artifacts, plus portfolio hyperparameters.

System arguments:
    None. This module is imported by ``main.py``; paths are edited here before a run.
"""

from dataclasses import dataclass

import seaborn as sns

from models import DataPath

# Starting cash for simulated portfolios (USD). Example: 100 means $100 initial capital.
initial_capital = 100
# Legacy/unused roots kept for reference when wiring older experiments.
connecting_the_dots_base = "../MTGNN/old_simulations/crypto/20221121"
lstm_base = "../lstm-predictor/prediction"


# Before running analysis, adjust paths
class FXPaths:
    """
    Paths for the FX benchmark: one ``actual`` CSV and many ``raw_prediction_paths`` (folders of .npy).

    Attributes:
        actual (DataPath): Ground-truth marked price CSV.
        raw_prediction_paths (list[DataPath]): Each entry is one model's prediction directory + label.
    """
    actual = DataPath("../ticker-collector/out/fx/daily_10_4506_marked.csv", "Actual")
    #TODO: Check path
    raw_prediction_paths = [
        DataPath(f"../experiment-results/fx/mtgnn-results/202312090808", "MTGNN"),
        #TODO: Check path
        DataPath(
            f"../experiment-results/fx/stemgnn-results/202312080300/__fx_daily_simple_returns",
            "StemGNN",
        ),
        #TODO: Check path
        DataPath(f"../experiment-results/fx/lstm-results/202312062100/weeks", "LSTM"),
        #TODO: Check path
        DataPath(f"../experiment-results/fx/fouriergnn-results/20250319", "FourierGNN"),
        #TODO: Check path
        DataPath(f"../experiment-results/fx/arima-results/20250321", "ARIMA"),
        #TODO: Check path
        DataPath(f"../experiment-results/fx/var-results/202503221825", "VAR"),
        #TODO: Check path
    ]


class CryptoPaths:
    """
    Paths for the cryptocurrency benchmark (same layout as ``FXPaths``).

    Attributes:
        actual (DataPath): Marked crypto daily CSV (prices + split_point column).
        raw_prediction_paths (list[DataPath]): Model outputs; some entries set ``data_type`` for log returns.
    """

    actual = DataPath("../ticker-collector/out/crypto/daily_20_2190_marked.csv", "Actual")
    #TODO: Check path
    raw_prediction_paths = [
        DataPath(f"../experiment-results/crypto/stemgnn-results/202312090602", "StemGNN"),
        DataPath(f"../experiment-results/crypto/mtgnn-results/7", "MTGNN:7"),
        DataPath(f"../experiment-results/crypto/mtgnn-results/21", "MTGNN:21"),
        DataPath(f"../experiment-results/crypto/mtgnn-results/35", "MTGNN:35"),
        DataPath(f"../experiment-results/crypto/mtgnn-results/42", "MTGNN:42"),
        DataPath(f"../experiment-results/crypto/mtgnn-results/63", "MTGNN:63"),
        DataPath(f"../experiment-results/crypto/mtgnn-results/84", "MTGNN:84"),
        DataPath(
            f"../experiment-results/crypto/lstm-results/202302221651",
            "LSTM:202302221651",
        ),
        DataPath(
            f"../experiment-results/crypto/lstm-results/202303221739",
            "LSTM:202303221739",
        ),
        DataPath(
            f"../experiment-results/crypto/lstm-results/202309210830",
            "LSTM:202309210830",
            "log-return",
        ),
        DataPath(f"../experiment-results/crypto/fouriergnn-results/20250318", "FourierGNN"),
        DataPath(f"../experiment-results/crypto/arima-results/20250321", "ARIMA"),
        DataPath(f"../experiment-results/crypto/var-results/202503221852", "VAR"),
    ]


# Rolling weeks of evaluation aligned with saved prediction files (1..num_weeks). Example: 104.
num_weeks = 104
# Monte Carlo samples for the random long/short baseline. Example: 1000 (overridden when ``quick``).
random_trials = 1000
# Relative price gap below which ``get_trading_decisions`` labels a "Hold". Example: 0.01 (1%).
hold_threshold = 0.01
# One-way transaction cost fraction per rebalance. Example: 0.005 (0.5%).
transaction_cost = 0.005
# If True, ``main.py`` shrinks ``random_trials`` and long/short grid for fast smoke tests.
quick = False
sns.set_theme(style="darkgrid")


@dataclass
class OutputFlags:
    """
    Toggle optional matplotlib plots (expensive or clutter-reducing).

    Attributes:
        plot_standardized_prices (bool): If True, plot z-scored price series.
        plot_pairwise_returns (bool): If True, plot every pairwise return series combination.
        plot_predicted_and_actual_portfolio_values (bool): If True, per-model capital curves.
    """

    plot_standardized_prices: bool = True
    plot_pairwise_returns: bool = False
    plot_predicted_and_actual_portfolio_values: bool = False
