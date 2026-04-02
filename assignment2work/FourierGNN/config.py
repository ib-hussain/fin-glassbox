"""
This module provides configuration dictionaries and mappings for data path routing.
It defines device selection (CPU or GPU) and dataset configurations.

System Arguments Expected:
    This module only provides configuration values and expects NO system arguments.
"""
import torch
import dotenv
from data.data_loader import (
    Dataset_Dhfm,
    Dataset_ECG,
    Dataset_Solar,
    Dataset_Wiki,
    DatasetFinancial,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_information = {
    "traffic": {
        "root_path": f"data/traffic.npy",
        "type": "0"
    },
    "ECG": {
        "root_path": f"data/ECG_data.csv",
        "type": "1"
    },
    "COVID": {
        "root_path": f"data/covid.csv",
        "type": "1"
    },
    "electricity": {
        "root_path": f"data/electricity.csv",
        "type": "1"
    },
    "solar": {
        "root_path": f"data/solar",
        "type": "1"
    },
    "metr": {
        "root_path": f"data/metr.csv",
        "type": "1"
    },
    "wiki": {
        "root_path": f"data/wiki.csv",
        "type": "1"
    },
    "crypto": {
        "root_path": f"data/crypto/daily_20_2190_marked.csv",
        "type": "1",
    },
    "fx": {
        "root_path": f"data/fx/daily_10_4506_marked.csv",
        "type": "1",
    },
}

data_dict = {
    "ECG": Dataset_ECG,
    "COVID": Dataset_ECG,
    "traffic": Dataset_Dhfm,
    "solar": Dataset_Solar,
    "wiki": Dataset_Wiki,
    "electricity": Dataset_ECG,
    "metr": Dataset_ECG,
    "crypto": DatasetFinancial,
    "fx": DatasetFinancial,
}
