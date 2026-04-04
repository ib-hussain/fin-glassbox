'''
ib-hussain: This file needs to be run from the CLI and it has paths. The CLI args need to be figured out.

(Added by AI Agent)
Module: train_weeks_ibVersion.py
This script manages the looping training procedure applied across datasets divided by weeks.
It establishes target hyperparameters specifically configured by ib-hussain.
'''
import sys

import pandas as pd
from tqdm import tqdm

from train_single_step import SingleStep, print_results


class Config:
    """
    Weekly models runtime variables.
    (Added by AI Agent)
    """
    conv_channels = 4
    epoch = 2000
    layers = 3
    lr = 0.00014215115225721316
    num_of_weeks_in_window = 2
    weight_decay = 3.485648994073535e-05
def main(device, data_path, horizon, starting_week, num_weeks):
    """
    Sequence driver that runs the SingleStep trainer natively across the series of weeks.
    (Added by AI Agent)

    Args:
        device (str): System device interface.
        data_path (str): Location of datasets.
        horizon (int): Scope of forecasting predictions.
        starting_week (int): Training start offset.
        num_weeks (int): Range of complete weeks to execute.
    """
    num_assets = len(pd.read_csv(data_path).columns) - 2
    single_step = get_trainer(device=device, data_path=data_path, horizon=horizon, num_assets=num_assets, week=starting_week, num_weeks=num_weeks)
    vacc, vrae, vcorr, vmape, acc, rae, corr, mape = [], [], [], [], [], [], [], []
    all_metrics_arrays = [vacc, vrae, vcorr, vmape, acc, rae, corr, mape]
    runs = 2
    for i in range(runs):
        all_metrics = single_step.run()
        [a.append(m) for a, m in zip(all_metrics_arrays, all_metrics)]
    print_results(runs, acc, corr, mape, rae, vacc, vcorr, vmape, vrae)
    # ib-hussain: The version below is an adapted version from predict_weeks.py but the above version is from train_fast_single_step_for_speed_testing.py as the code over there matches the code here better.
    # num_assets = len(pd.read_csv(data_path).columns) - 2
    # for week in tqdm(range(starting_week, num_weeks), desc="predict_weeks.py > week"):
    #     print(f"Training week: {week}")
    #     get_trainer(data_path, horizon, num_assets, week, num_weeks, device).train()
def get_trainer(data_path, horizon, num_assets, week, num_weeks, device):
    """
    Bootstraps the SingleStep instance targeting the specific week span parameters.
    (Added by AI Agent)
    """
    return SingleStep(
        data_path=data_path,
        week=week,
        num_weeks=num_weeks,
        device=device,
        num_nodes=num_assets,
        subgraph_size=int(num_assets * 0.4),
        seq_in_len=Config.num_of_weeks_in_window * horizon,
        horizon=horizon,
        batch_size=30,
        epochs=Config.epoch,
        lr=Config.lr,
        weight_decay=Config.weight_decay,
        layers=Config.layers,
        conv_channels=Config.conv_channels,
        residual_channels=Config.conv_channels,
        skip_channels=Config.conv_channels * 2,
        end_channels=Config.conv_channels * 4,
        training_split=0.74,
        validation_split=0.24,
    )
if __name__ == "__main__":
    device, data_path, horizon, starting_week, num_weeks = sys.argv[1:]
    main(device, data_path, int(horizon), int(starting_week), int(num_weeks))
