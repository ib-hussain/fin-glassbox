'''
ib-hussain: This file needs to be run from the CLI and it has paths. The CLI args need to be figured out.

(Added by AI Agent)
Module: train_and_predict_weeks.py
Combined training and prediction execution for the MTGNN over a weekly timescale. 
Iterates over a specified range of weeks to train and immediately evaluate/predict the best model.
'''
import sys

import pandas as pd
from tqdm import tqdm

from train_single_step import SingleStep

class Config:
    """
    Configuration parameters for the train-and-predict cycle.
    (Added by AI Agent)
    """
    conv_channels = 4
    epoch = 2000
    layers = 3
    lr = 0.0006351708427814676
    num_of_weeks_in_window = 5
    weight_decay = 2.0337851025938947e-05
def main(device, data_path, horizon, starting_week, num_weeks):
    """
    Main loop to run training and prediction back-to-back across target weeks.
    (Added by AI Agent)

    Args:
        device (str): Compute device.
        data_path (str): File path logic for evaluation.
        horizon (int): Number of future steps to predict.
        starting_week (int): Weekly index base.
        num_weeks (int): Stop condition for total weeks.
    """
    num_assets = len(pd.read_csv(data_path).columns) - 2
    for week in tqdm(range(starting_week, num_weeks), desc="train_and_predict_weeks.py > week"):
        print(f"Training week: {week}")
        get_trainer(data_path, horizon, num_assets, week, num_weeks, device).run_train_only()
        print(f"Predicting week: {week}")
        get_predictor(data_path, horizon, num_assets, week, num_weeks, device).predict_with_the_best_model()
def get_trainer(data_path, horizon, num_assets, week, num_weeks, device):
    """
    Generates a SingleStep configuration specifically intended for training mode.
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
def get_predictor(data_path, horizon, num_assets, week, num_weeks, device):
    """
    Generates a SingleStep configuration specifically intended for testing/prediction mode.
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
        run_for_prediction=True,
    )
if __name__ == "__main__":
    device, data_path, horizon, starting_week, num_weeks = sys.argv[1:]
    main(device, data_path, int(horizon), int(starting_week), int(num_weeks))
