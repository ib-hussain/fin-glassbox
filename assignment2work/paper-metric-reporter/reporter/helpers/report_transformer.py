"""
Transform predictions into prices, portfolio DataFrames, trading labels, and best-model selections.

System arguments:
    None.
"""

from itertools import groupby

import numpy as np
import pandas as pd

import config
from models import Data


class ReportTransformer:
    """Numeric and naming utilities between ``Reader`` outputs and ``Reporter`` tables."""

    @staticmethod
    def get_predicted_prices(actual_prices, predicted_returns):
        """
        Integrate predicted simple returns/factors into price level paths per asset.

        Args:
            actual_prices (Data): Observed prices, shape ``(T, n_assets)``.
            predicted_returns (list[Data]): Each element is one model's returns, shape ``(T, n_assets)``.

        Returns:
            list[Data]: Same labels; each ``data`` holds reconstructed prices.
        """
        print(
            f"[Debug_Output]: ReportTransformer.get_predicted_prices | n_models={len(predicted_returns)} | "
            f"actual.shape={actual_prices.data.shape}"
        )
        num_assets = actual_prices.data.shape[1]
        predicted_prices = []
        for j in range(len(predicted_returns)):
            predicted_asset_prices = []
            for ind in range(num_assets):
                a = actual_prices.data[:, ind]
                r = predicted_returns[j].data[:, ind]
                p = [a[0]]
                for i in range(1, len(r)):
                    p.append(a[i - 1] * r[i])
                predicted_asset_prices.append(p)

            predicted_asset_prices = np.asarray(predicted_asset_prices).T
            predicted_prices.append(Data(predicted_asset_prices, predicted_returns[j].label))
        return predicted_prices

    @staticmethod
    def get_best_models(all_model_results):
        """
        For each model family prefix (text before ``:``), keep the portfolio with highest final value.

        Args:
            all_model_results (iterable[PortfolioResult]): Flattened results across long/short and models.

        Returns:
            list[PortfolioResult]: One winner per prefix group.
        """
        all_model_results = list(all_model_results)
        print(f"[Debug_Output]: ReportTransformer.get_best_models | n_input={len(all_model_results)}")
        sorted_models = sorted(all_model_results, key=ReportTransformer.get_model_prefix)
        return [
            max(g, key=lambda model: model.final_portfolio_value())
            for k, g in groupby(sorted_models, ReportTransformer.get_model_prefix)
        ]

    @staticmethod
    def remove_model_prefixes(models):
        """Mutate labels in place to strip ``MTGNN:7`` style suffixes."""
        print(f"[Debug_Output]: ReportTransformer.remove_model_prefixes | n={len(models)}")
        for model_result in models:
            model_result.label = ReportTransformer.get_model_prefix(model_result)
        return models

    @staticmethod
    def remove_model_prefix(model):
        """Strip trailing ``:variant`` from a single result's label."""
        print(f"[Debug_Output]: ReportTransformer.remove_model_prefix | label_before={model.label!r}")
        model.label = ReportTransformer.get_model_prefix(model)
        return model

    @staticmethod
    def get_model_prefix(model):
        """Return substring before first ``:`` in ``model.label`` (fallback: full label)."""
        return model.label.split(":")[0]

    @staticmethod
    def create_portfolio_values_df(random_results, all_results, best_model_results):
        """
        Merge baseline curves (Random, All-asset) with each best model's capital series.

        Returns:
            pandas.DataFrame: Columns = strategy labels; rows = weeks.
        """
        print(
            f"[Debug_Output]: ReportTransformer.create_portfolio_values_df | "
            f"best_model_count={len(best_model_results)}"
        )
        df = pd.DataFrame()
        df[random_results.label] = random_results.data
        df[all_results.label] = all_results.data

        for model_result in best_model_results:
            df[model_result.label] = model_result.data
        return df

    @staticmethod
    def get_returns(prices, normalize=True, append_ones=False):
        """
        Week-over-week return factors (or simple returns if ``normalize``).

        Args:
            prices (pandas.DataFrame | numpy.ndarray): Capital or price level.
            normalize (bool): If True, subtract 1 so result is ``r - 1``.
            append_ones (bool): Append a row of ones (padding) for index alignment tricks.

        Returns:
            pandas.DataFrame or numpy.ndarray: Length along time is one shorter unless ``append_ones``.
        """
        print(
            f"[Debug_Output]: ReportTransformer.get_returns | normalize={normalize} | append_ones={append_ones} | "
            f"type={type(prices).__name__}"
        )
        next_values = prices[1:]
        current_values = prices[:-1]

        if isinstance(prices, pd.DataFrame):
            next_values = next_values.reset_index(drop=True)

        returns = next_values / current_values
        if normalize:
            returns -= 1

        if append_ones:
            if isinstance(prices, pd.DataFrame):
                returns = pd.concat([returns, pd.DataFrame(np.ones(returns.shape[1])).T])
            else:
                returns = np.concatenate((returns, np.ones((1, returns.shape[1]))))
        return returns

    @staticmethod
    def get_trading_decisions(actual, prediction, hold_threshold=config.hold_threshold):
        """
        Label each week/asset as Buy / Hold / Sell using relative error vs ``hold_threshold``.

        Args:
            actual (numpy.ndarray): True prices.
            prediction (numpy.ndarray): Forecast prices (shifted internally).
            hold_threshold (float): Band for Hold. Example: ``0.01``.

        Returns:
            numpy.ndarray: String decisions, shape ``(T-1, n_assets)``.
        """
        print(
            f"[Debug_Output]: ReportTransformer.get_trading_decisions | hold_threshold={hold_threshold} | "
            f"actual.shape={actual.shape} prediction.shape={prediction.shape}"
        )
        actual = actual[:-1]
        prediction = prediction[1:]

        diff = np.abs((prediction - actual) / actual)
        decisions = np.full(actual.shape, "Hold", dtype=object)
        hold = diff <= hold_threshold
        buy = prediction > actual
        sell = prediction < actual

        decisions[~hold & buy] = "Buy"
        decisions[~hold & sell] = "Sell"
        return decisions
