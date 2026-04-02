"""
High-level portfolio experiments: random baseline, all-asset baseline, and best long/short per model.

System arguments:
    None.
"""

from tqdm import tqdm

from helpers import Portfolio
from models import PortfolioResult


class PortfolioGroup:
    """Factories that return ``PortfolioResult`` bundles for ``main.py``."""

    @staticmethod
    def calculate_random_portfolio_for_predictions(random_trials, actual_prices, long_short_combination):
        """
        Average capital paths over ``random_trials`` random long/short selections.

        Args:
            random_trials (int): Monte Carlo count. Example: ``1000``.
            actual_prices (Data): Observed price tensor.
            long_short_combination (tuple): ``(n_long, n_short)``. Example: ``(5, 0)``.

        Returns:
            PortfolioResult: Label ``\"Random\"``, ``data`` is mean path.
        """
        print(
            f"[Debug_Output]: PortfolioGroup.calculate_random_portfolio | random_trials={random_trials} | "
            f"ls={long_short_combination}"
        )
        acc_results = 0
        acc_asset_selection_accuracy = 0
        for _ in tqdm(range(random_trials), desc="Calculating random portfolio"):
            results, asset_selection_accuracy = Portfolio.get_random_results(actual_prices, *long_short_combination)
            acc_results += results
            acc_asset_selection_accuracy += asset_selection_accuracy
        acc_results /= random_trials
        acc_asset_selection_accuracy /= random_trials
        return PortfolioResult(
            acc_results,
            "Random",
            long_short_combination,
            asset_selection_accuracy=acc_asset_selection_accuracy,
        )

    @staticmethod
    def calculate_all_portfolio_for_predictions(actual_prices):
        """
        Equal exposure to every asset (long-only aggregate path).

        Returns:
            PortfolioResult: Label ``\"All-asset\"``.
        """
        print("[Debug_Output]: PortfolioGroup.calculate_all_portfolio_for_predictions")
        results = Portfolio.get_all_results(actual_prices)
        return PortfolioResult(results, "All-asset", (actual_prices.data.shape[1], 0))

    @classmethod
    def calculate_best_portfolio_for_all_predictions(cls, actual_prices, predicted_returns, long_short_combinations):
        """
        Generator: for each model's ``predicted_returns`` series, pick best long/short on first 10 weeks
        then evaluate on the full horizon.

        Yields:
            PortfolioResult: One best portfolio per prediction model.
        """
        print(
            f"[Debug_Output]: PortfolioGroup.calculate_best_portfolio_for_all_predictions | "
            f"n_long_short_pairs_estimated_from_generator=stream"
        )
        return (cls._calculate_best_portfolio_for_predictions(actual_prices, pp, long_short_combinations)
                for pp in tqdm(predicted_returns, desc="Calculating best portfolio for each model"))

    @staticmethod
    def _calculate_best_portfolio_for_predictions(actual_prices, predicted_returns, long_short_combinations):
        """
        Inner helper: grid-search long/short on a short warm-up window, then full backtest.

        Args:
            actual_prices (Data): Full ``Data`` (sliced inside with ``[:10]`` for selection).
            predicted_returns (Data): One model's predictions.
            long_short_combinations (iterable): Candidate ``(long, short)`` pairs.
        """

        def determine_best_long_short_combination():
            print(
                f"[Debug_Output]: PortfolioGroup._calculate_best_portfolio | model_label={predicted_returns.label!r}"
            )
            all_portfolio_results_according_to_first_10_weeks = [
                Portfolio.get_equally_weighted_portfolio_results(actual_prices[:10], predicted_returns[:10],
                                                                 *ls_combination)
                for ls_combination in tqdm(
                    long_short_combinations,
                    desc=f"Calculating best portfolio for {predicted_returns.label}",
                )
            ]
            best_portfolio = max(
                all_portfolio_results_according_to_first_10_weeks,
                key=lambda x: x.final_portfolio_value(),
            )

            return best_portfolio.long_short_combination

        best_portfolio_long_short_combination = determine_best_long_short_combination()
        return Portfolio.get_equally_weighted_portfolio_results(actual_prices, predicted_returns,
                                                                *best_portfolio_long_short_combination)
