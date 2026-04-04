'''
ib-hussain: This file needs to be run from the CLI and it has paths. The CLI args need to be figured out.

(Added by AI Agent)
Module: predict_weeks.py
This script iteratively predicts weekly outcomes using a trained MTGNN model over a specified
range of weeks. It sets up fixed hyperparameters for configuration.
'''
import sys

import pandas as pd
from tqdm import tqdm

from train_single_step import SingleStep


class Config:
    """
    Configuration settings for weekly prediction.
    (Added by AI Agent)
    """
    conv_channels = 4
    epoch = 200
    # ib-hussain changed num of epochs from 2000 to 200
    layers = 3
    lr = 0.00014215115225721316
    num_of_weeks_in_window = 2
    weight_decay = 3.485648994073535e-05
def main(device, data_path, horizon, starting_week, num_weeks):
    """
    Main loop to predict sequentially across weeks.
    (Added by AI Agent)

    Args:
        device (str): Computing device ('cpu', 'cuda').
        data_path (str): File path for data.
        horizon (int): Forecasting steps.
        starting_week (int): The week index to begin predictions.
        num_weeks (int): Total number of weeks up to which predictions run.
    """
    num_assets = len(pd.read_csv(data_path).columns) - 2
    for week in tqdm(range(starting_week, num_weeks), desc="predict_weeks.py > week"):
        print(f"Predicting week: {week}")
        get_predictor(data_path, horizon, num_assets, week, num_weeks, device).predict_with_the_best_model()
def get_trainer(data_path, horizon, num_assets, week, num_weeks, device):
    """
    Retrieve an initialized SingleStep class instance configured for training.
    (Added by AI Agent)

    Args:
        data_path (str): Data file path.
        horizon (int): Forecast horizon.
        num_assets (int): Total asset nodes based on the dataset.
        week (int): Current week index.
        num_weeks (int): End week target for iteration.
        device (str): Device to run training on.

    Returns:
        SingleStep: Instantiated class object configured with predefined metrics.
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
def get_predictor(data_path, horizon, num_assets, week, num_weeks, device):
    """
    Retrieve an initialized SingleStep class instance configured for prediction.
    (Added by AI Agent)

    Args:
        data_path (str): Data file path.
        horizon (int): Forecast horizon.
        num_assets (int): Total asset nodes based on the dataset.
        week (int): Current week index.
        num_weeks (int): Total weeks bounds.
        device (str): Devices processing the tensor network.

    Returns:
        SingleStep: Instantiated class object with `run_for_prediction=True`.
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
        run_for_prediction=True,
    )
if __name__ == "__main__":
    device, data_path, horizon, starting_week, num_weeks = sys.argv[1:]
    main(device, data_path, int(horizon), int(starting_week), int(num_weeks))
