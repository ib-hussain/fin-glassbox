"""
This module provides standard torch Dataset abstractions for slicing historical sliding windows
off various benchmark datasets including ECG, Solar, Crypto, and predefined numerical arrays.

System Arguments:
    This module only provides class definitions; it expects no system arguments.
"""
import datetime
import os
import dotenv
dotenv.load_dotenv()  # Load environment variables from .env file

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset


# traffic data
class Dataset_Dhfm(Dataset):
    """
    Standard loader sequence sliding window for DHFM numerical configurations.
    """

    def __init__(self, root_path, flag, seq_len, pre_len, type, train_ratio, val_ratio):
        print(f"[Debug_Output]: Function 'Dataset_Dhfm.__init__' called with root_path={root_path}, flag={flag}, seq_len={seq_len}, pre_len={pre_len}, type={type}, train_ratio={train_ratio}, val_ratio={val_ratio}")
        """
        Initializes sequence boundaries and fetches targeted data.

        Args:
            root_path (str): Relative URI. Example: "assignment2work/ForuierGNN/data/traffic.npy"
            flag (str): Operation phase limiters ("train", "val", "test"). Example: "train"
            seq_len (int): X input sequence size. Example: 12
            pre_len (int): Y target prediction size. Example: 12
            type (str): Signals initialization triggers (e.g., standardizing logic). Example: "1"
            train_ratio (float): Traing set size multiplier. Example: 0.7
            val_ratio (float): Validation set multiplier. Example: 0.2
        """
        assert flag in ["train", "test", "val"]
        self.path = root_path
        self.flag = flag
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        load_data = np.load(root_path)
        data = load_data.transpose()
        if type == "1":
            mms = MinMaxScaler(feature_range=(0, 1))
            data = mms.fit_transform(data)
        if self.flag == "train":
            begin = 0
            end = int(len(data) * self.train_ratio)
            self.trainData = data[begin:end]
        if self.flag == "val":
            begin = int(len(data) * self.train_ratio)
            end = int(len(data) * (self.val_ratio + self.train_ratio))
            self.valData = data[begin:end]
        if self.flag == "test":
            begin = int(len(data) * (self.val_ratio + self.train_ratio))
            end = len(data)
            self.testData = data[begin:end]

    def __getitem__(self, index):
        print(f"[Debug_Output]: Function 'Dataset_Dhfm.__getitem__' called with index={index}")
        begin = index
        end = index + self.seq_len
        next_end = end + self.pre_len
        if self.flag == "train":
            data = self.trainData[begin:end]
            next_data = self.trainData[end:next_end]
        elif self.flag == "val":
            data = self.valData[begin:end]
            next_data = self.valData[end:next_end]
        else:
            data = self.testData[begin:end]
            next_data = self.testData[end:next_end]
        return data, next_data

    def __len__(self):
        print(f"[Debug_Output]: Function 'Dataset_Dhfm.__len__' called")
        if self.flag == "train":
            return len(self.trainData) - self.seq_len - self.pre_len
        elif self.flag == "val":
            return len(self.valData) - self.seq_len - self.pre_len
        else:
            return len(self.testData) - self.seq_len - self.pre_len
# ECG dataset
class Dataset_ECG(Dataset):
    """
    Data generator for the ECG raw numeric baseline mapped over csv columns.
    """

    def __init__(self, root_path, flag, seq_len, pre_len, type, train_ratio, val_ratio):
        print(f"[Debug_Output]: Function 'Dataset_ECG.__init__' called with root_path={root_path}, flag={flag}, seq_len={seq_len}, pre_len={pre_len}, type={type}, train_ratio={train_ratio}, val_ratio={val_ratio}")
        """
        Initializes boundaries and limits data based on ratios.

        Args:
            root_path (str): CSV location path. Example: "assignment2work/ForuierGNN/data/ECG_data.csv"
            flag (str): Phase setting ("train", "val", "test"). Example: "test"
            seq_len (int): Historical trace size. Example: 12
            pre_len (int): Predictive forecast size. Example: 12
            type (str): Standardization routing parameters. Example: "0"
            train_ratio (float): Relative split proportion. Example: 0.7
            val_ratio (float): Relative split proportion. Example: 0.2
        """
        assert flag in ["train", "test", "val"]
        self.path = root_path
        self.flag = flag
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.standard_scaler = StandardScaler()
        data = pd.read_csv(root_path)

        if type == "1":
            training_end = int(len(data) * self.train_ratio)
            self.standard_scaler.fit(data[:training_end])
            data = self.standard_scaler.transform(data)
        data = np.array(data)
        if self.flag == "train":
            begin = 0
            end = int(len(data) * self.train_ratio)
            self.trainData = data[begin:end]
        if self.flag == "val":
            begin = int(len(data) * self.train_ratio)
            end = int(len(data) * (self.val_ratio + self.train_ratio))
            self.valData = data[begin:end]
        if self.flag == "test":
            begin = int(len(data) * (self.val_ratio + self.train_ratio))
            end = len(data)
            self.testData = data[begin:end]

    def __getitem__(self, index):
        print(f"[Debug_Output]: Function 'Dataset_ECG.__getitem__' called with index={index}")
        begin = index
        end = index + self.seq_len
        next_begin = end
        next_end = next_begin + self.pre_len
        if self.flag == "train":
            data = self.trainData[begin:end]
            next_data = self.trainData[next_begin:next_end]
        elif self.flag == "val":
            data = self.valData[begin:end]
            next_data = self.valData[next_begin:next_end]
        else:
            data = self.testData[begin:end]
            next_data = self.testData[next_begin:next_end]
        return data, next_data

    def __len__(self):
        print(f"[Debug_Output]: Function 'Dataset_ECG.__len__' called")
        # minus the label length
        if self.flag == "train":
            return len(self.trainData) - self.seq_len - self.pre_len
        elif self.flag == "val":
            return len(self.valData) - self.seq_len - self.pre_len
        else:
            return len(self.testData) - self.seq_len - self.pre_len

class Dataset_Solar(Dataset):
    """
    Reads multiple disjoint hourly/daily sequences extracting target periods per iteration for solar prediction tasks.
    """

    def __init__(self, root_path, flag, seq_len, pre_len, type, train_ratio, val_ratio):
        print(f"[Debug_Output]: Function 'Dataset_Solar.__init__' called with root_path={root_path}, flag={flag}, seq_len={seq_len}, pre_len={pre_len}, type={type}, train_ratio={train_ratio}, val_ratio={val_ratio}")
        assert flag in ["train", "test", "val"]
        self.path = root_path
        self.flag = flag
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        files = os.listdir(root_path)
        solar_data = []
        time_data = []
        for file in files:
            if not os.path.isdir(file):
                if file.startswith("DA_"):
                    data = pd.read_csv(root_path + "\\" + file).values
                    raw_time = data[:, 0:1]
                    if time_data == []:
                        time_data = raw_time
                    raw_data = data[:, 1:data.shape[1]]
                    raw_data = raw_data.transpose()
                    solar_data.append(raw_data)
        solar_data = np.array(solar_data).squeeze(1).transpose()
        time_data = np.array(time_data)
        out = np.concatenate((time_data, solar_data), axis=1)
        self.data = []
        for item in out:
            tmp = item[0]
            dt = datetime.datetime.strptime(tmp, "%m/%d/%y %H:%M")
            if dt.hour >= 8 and dt.hour <= 17:
                self.data.append(item[1:out.shape[1] - 1])

        if type == "1":
            mms = MinMaxScaler(feature_range=(0, 1))
        self.data = mms.fit_transform(self.data)
        if self.flag == "train":
            begin = 0
            end = int(len(self.data) * self.train_ratio)
            self.trainData = self.data[begin:end]
            self.train_nextData = self.data[begin:end]
        if self.flag == "val":
            begin = int(len(self.data) * self.train_ratio)
            end = int(len(self.data) * (self.train_ratio + self.val_ratio))
            self.valData = self.data[begin:end]
            self.val_nextData = self.data[begin:end]
        if self.flag == "test":
            begin = int(len(self.data) * (self.train_ratio + self.val_ratio))
            end = len(self.data)
            self.testData = self.data[begin:end]
            self.test_nextData = self.data[begin:end]

    def __getitem__(self, index):
        print(f"[Debug_Output]: Function 'Dataset_Solar.__getitem__' called with index={index}")
        begin = index
        end = index + self.seq_len
        next_end = end + self.pre_len
        if self.flag == "train":
            data = self.trainData[begin:end]
            next_data = self.trainData[end:next_end]
        elif self.flag == "val":
            data = self.valData[begin:end]
            next_data = self.valData[end:next_end]
        else:
            data = self.testData[begin:end]
            next_data = self.testData[end:next_end]
        return data, next_data

    def __len__(self):
        print(f"[Debug_Output]: Function 'Dataset_Solar.__len__' called")
        if self.flag == "train":
            return len(self.trainData) - self.seq_len - self.pre_len
        elif self.flag == "val":
            return len(self.valData) - self.seq_len - self.pre_len
        else:
            return len(self.testData) - self.seq_len - self.pre_len

class Dataset_Wiki(Dataset):
    """
    Parser loading multi-variate Wikipedia numerical arrays skipping nan alignments.
    """

    def __init__(self, root_path, flag, seq_len, pre_len, type, train_ratio, val_ratio):
        print(f"[Debug_Output]: Function 'Dataset_Wiki.__init__' called with root_path={root_path}, flag={flag}, seq_len={seq_len}, pre_len={pre_len}, type={type}, train_ratio={train_ratio}, val_ratio={val_ratio}")
        assert flag in ["train", "test", "val"]
        self.path = root_path
        self.flag = flag
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        data = pd.read_csv(root_path).values
        raw_data = data[:, 1:data.shape[1]]
        df = pd.DataFrame(raw_data)
        # data cleaning
        self.data = df.dropna(axis=0, how="any").values.transpose()
        if type == "1":
            mms = MinMaxScaler(feature_range=(0, 1))
            self.data = mms.fit_transform(self.data)
        if self.flag == "train":
            begin = 0
            end = int(len(self.data) * self.train_ratio)
            self.trainData = self.data[begin:end]
            self.train_nextData = self.data[begin:end]
        if self.flag == "val":
            begin = int(len(self.data) * self.train_ratio)
            end = int(len(self.data) * (self.train_ratio + self.val_ratio))
            self.valData = self.data[begin:end]
            self.val_nextData = self.data[begin:end]
        if self.flag == "test":
            begin = int(len(self.data) * (self.train_ratio + self.val_ratio))
            end = len(self.data)
            self.testData = self.data[begin:end]
            self.test_nextData = self.data[begin:end]

    def __getitem__(self, index):
        print(f"[Debug_Output]: Function 'Dataset_Wiki.__getitem__' called with index={index}")
        # data timestamp
        begin = index
        end = index + self.seq_len
        next_end = end + self.pre_len
        if self.flag == "train":
            data = self.trainData[begin:end]
            next_data = self.trainData[end:next_end]
        elif self.flag == "val":
            data = self.valData[begin:end]
            next_data = self.valData[end:next_end]
        else:
            data = self.testData[begin:end]
            next_data = self.testData[end:next_end]
        # return the time data , next time data and time
        return data, next_data

    def __len__(self):
        print(f"[Debug_Output]: Function 'Dataset_Wiki.__len__' called")
        # minus the label length
        if self.flag == "train":
            return len(self.trainData) - self.seq_len - self.pre_len
        elif self.flag == "val":
            return len(self.valData) - self.seq_len - self.pre_len
        else:
            return len(self.testData) - self.seq_len - self.pre_len

# class DatasetFinancial(Dataset):
#     """
#     Specific implementation mapping trailing temporal financial records backward matching target evaluation thresholds.
#     """

#     def __init__(self, root_path, week, flag, seq_len, pre_len, type, train_ratio, val_ratio):
#         print(f"[Debug_Output]: Function 'DatasetFinancial.__init__' called with root_path={root_path}, week={week}, flag={flag}, seq_len={seq_len}, pre_len={pre_len}, type={type}, train_ratio={train_ratio}, val_ratio={val_ratio}")
#         """
#         Splits pre-marked CSV outputs extracting only rows mapped to limits.
        
#         Args:
#             root_path (str): The merged CSV. Example: "assignment2work/datasetsOut/crypto/daily_20_2190_marked.csv"
#             week (int): Evaluation temporal index logic parameter. Example: 50
#             flag (str): Phase logic parameter. Example: "train"
#             seq_len, pre_len, type, train_ratio, val_ratio: Dimension sizing.
#         """
#         assert flag in ["train", "test", "val"]
#         self.flag = flag
#         self.seq_len = seq_len
#         self.pre_len = pre_len
#         self.train_ratio = train_ratio
#         self.val_ratio = val_ratio
#         self.standard_scaler = StandardScaler()
#         data = self.read_price_returns(root_path, int(week), 104, self.pre_len)

#         if type == "1":
#             training_end = int(len(data) * self.train_ratio)
#             print(f"[Debug_Output]: DatasetFinancial.__init__ processing step => len(data)={len(data)}, training_end={training_end}")
#             self.standard_scaler.fit(data[:training_end])
#             data = self.standard_scaler.transform(data)
#         data = np.array(data)
#         if self.flag == "train":
#             begin = 0
#             end = int(len(data) * self.train_ratio)
#             self.trainData = data[begin:end]
#         if self.flag == "val":
#             begin = int(len(data) * self.train_ratio)
#             end = int(len(data) * (self.val_ratio + self.train_ratio))
#             self.valData = data[begin:end]
#         if self.flag == "test":
#             begin = int(len(data) * (self.val_ratio + self.train_ratio))
#             end = len(data)
#             self.testData = data[begin:end]

#     def __getitem__(self, index):
#         print(f"[Debug_Output]: Function 'DatasetFinancial.__getitem__' called with index={index}")
#         begin = index
#         end = index + self.seq_len
#         next_begin = end
#         next_end = next_begin + self.pre_len
#         if self.flag == "train":
#             data = self.trainData[begin:end]
#             next_data = self.trainData[next_begin:next_end]
#         elif self.flag == "val":
#             data = self.valData[begin:end]
#             next_data = self.valData[next_begin:next_end]
#         else:
#             data = self.testData[begin:end]
#             next_data = self.testData[next_begin:next_end]
#         return data, next_data

#     def __len__(self):
#         print(f"[Debug_Output]: Function 'DatasetFinancial.__len__' called")
#         # minus the label length
#         if self.flag == "train":
#             return len(self.trainData) - self.seq_len - self.pre_len
#         elif self.flag == "val":
#             return len(self.valData) - self.seq_len - self.pre_len
#         else:
#             return len(self.testData) - self.seq_len - self.pre_len

#     @classmethod
#     def read_price_returns(cls, path, week, num_weeks, horizon):
#         print(f"[Debug_Output]: Function 'read_price_returns' called with path={path}, week={week}, num_weeks={num_weeks}, horizon={horizon}")
#         """
#         Coordinates price sequence gathering executing conversion logic explicitly over specified horizon variables.
        
#         Args:
#             path (str): File boundary path limit.
#             week (int): Stopping mark for index tracking. Example: 104
#             num_weeks (int): Length logic maximum. Example: 156
#             horizon (int): Division scale mapping offset limit. Example: 14
#         """
#         prices_np = cls._read_raw_data(path=path, week=week, num_weeks=num_weeks)
#         simple_returns_np = cls._convert_to_simple_returns(prices_np, horizon)
#         return simple_returns_np

#     @staticmethod
#     def _read_raw_data(path, week, num_weeks):
#         print(f"[Debug_Output]: Function '_read_raw_data' called with path={path}, week={week}, num_weeks={num_weeks}")
#         fin = open(path)
#         raw_data = pd.read_csv(fin)

#         truncate_index = len(raw_data)
#         stopping_point = -1
#         truncation_mark = num_weeks - week

#         # ── DIAGNOSTIC: inspect the split_point column before the loop ──────────
#         sp_col = raw_data["split_point"]
#         print(
#             f"[Debug_Output]: _read_raw_data DIAGNOSTIC | split_point dtype={sp_col.dtype} | "
#             f"unique_values={sp_col.unique().tolist()} | "
#             f"value_counts={sp_col.value_counts().to_dict()} | "
#             f"sum_truthy={sp_col.astype(bool).sum()} | "
#             f"first_5={sp_col.head().tolist()} | last_5={sp_col.tail().tolist()}"
#         )
#         # ────────────────────────────────────────────────────────────────────────

#         for point in reversed(raw_data["split_point"]):
#             if stopping_point == truncation_mark:
#                 break

#             if point:
#                 stopping_point += 1

#             truncate_index -= 1

#         out = raw_data.loc[:truncate_index].drop(["split_point"], axis=1).drop(columns=["Date"]).to_numpy()
#         print(
#             f"[Debug_Output]: _read_raw_data done | n_rows_full={len(raw_data)} | truncate_index={truncate_index} | "
#             f"out_shape={out.shape} | week={week} truncation_mark={truncation_mark}"
#         )
#         return out

#     @staticmethod
#     def _convert_to_simple_returns(prices_np, horizon):
#         print(f"[Debug_Output]: Function '_convert_to_simple_returns' called with horizon={horizon}")
#         simple_returns = prices_np[horizon:] / prices_np[:-horizon]
#         return simple_returns
class DatasetFinancial(Dataset):
    """
    Specific implementation mapping trailing temporal financial records backward matching target evaluation thresholds.
    """

    def __init__(self, root_path, week, flag, seq_len, pre_len, type, train_ratio, val_ratio):
        print(f"[Debug_Output]: Function 'DatasetFinancial.__init__' called with root_path={root_path}, week={week}, flag={flag}, seq_len={seq_len}, pre_len={pre_len}, type={type}, train_ratio={train_ratio}, val_ratio={val_ratio}")
        """
        Splits pre-marked CSV outputs extracting only rows mapped to limits.
        
        Args:
            root_path (str): The merged CSV. Example: "assignment2work/datasetsOut/crypto/daily_20_2190_marked.csv"
            week (int): Evaluation temporal index logic parameter. Example: 50
            flag (str): Phase logic parameter. Example: "train"
            seq_len, pre_len, type, train_ratio, val_ratio: Dimension sizing.
        """
        assert flag in ["train", "test", "val"]
        self.flag = flag
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.standard_scaler = StandardScaler()
        data = self.read_price_returns(root_path, int(week), 104, self.pre_len)

        # FIX: Handle empty data case
        if len(data) == 0:
            print(f"[Warning]: No data loaded from {root_path} for week={week}")
            # Create empty arrays to avoid errors
            self.trainData = np.array([])
            self.valData = np.array([])
            self.testData = np.array([])
            return

        if type == "1" and len(data) > 0:
            training_end = max(1, int(len(data) * self.train_ratio))  # Ensure at least 1 sample
            print(f"[Debug_Output]: DatasetFinancial.__init__ processing step => len(data)={len(data)}, training_end={training_end}")
            self.standard_scaler.fit(data[:training_end])
            data = self.standard_scaler.transform(data)
        data = np.array(data)
        
        if self.flag == "train":
            begin = 0
            end = int(len(data) * self.train_ratio)
            if end <= begin:
                end = min(begin + 1, len(data))  # Ensure at least 1 sample
            self.trainData = data[begin:end]
        if self.flag == "val":
            begin = int(len(data) * self.train_ratio)
            end = int(len(data) * (self.val_ratio + self.train_ratio))
            if end <= begin:
                end = min(begin + 1, len(data))
            self.valData = data[begin:end]
        if self.flag == "test":
            begin = int(len(data) * (self.val_ratio + self.train_ratio))
            end = len(data)
            if end <= begin:
                begin = max(0, end - 1)
            self.testData = data[begin:end]
        
        # Debug print to verify data sizes
        print(f"[Debug]: {flag} data shape: {getattr(self, f'{flag}Data').shape if len(getattr(self, f'{flag}Data')) > 0 else 'empty'}")

    def __getitem__(self, index):
        print(f"[Debug_Output]: Function 'DatasetFinancial.__getitem__' called with index={index}")
        begin = index
        end = index + self.seq_len
        next_begin = end
        next_end = next_begin + self.pre_len
        
        if self.flag == "train":
            data = self.trainData[begin:end]
            next_data = self.trainData[next_begin:next_end]
        elif self.flag == "val":
            data = self.valData[begin:end]
            next_data = self.valData[next_begin:next_end]
        else:
            data = self.testData[begin:end]
            next_data = self.testData[next_begin:next_end]
        return data, next_data

    def __len__(self):
        print(f"[Debug_Output]: Function 'DatasetFinancial.__len__' called")
        # minus the label length
        if self.flag == "train":
            return max(0, len(self.trainData) - self.seq_len - self.pre_len)
        elif self.flag == "val":
            return max(0, len(self.valData) - self.seq_len - self.pre_len)
        else:
            return max(0, len(self.testData) - self.seq_len - self.pre_len)

    @classmethod
    def read_price_returns(cls, path, week, num_weeks, horizon):
        print(f"[Debug_Output]: Function 'read_price_returns' called with path={path}, week={week}, num_weeks={num_weeks}, horizon={horizon}")
        """
        Coordinates price sequence gathering executing conversion logic explicitly over specified horizon variables.
        
        Args:
            path (str): File boundary path limit.
            week (int): Stopping mark for index tracking. Example: 104
            num_weeks (int): Length logic maximum. Example: 156
            horizon (int): Division scale mapping offset limit. Example: 14
        """
        prices_np = cls._read_raw_data(path=path, week=week, num_weeks=num_weeks)
        if len(prices_np) == 0:
            print(f"[Warning]: No prices data loaded from {path}")
            return np.array([])
        simple_returns_np = cls._convert_to_simple_returns(prices_np, horizon)
        return simple_returns_np

    @staticmethod
    def _read_raw_data(path, week, num_weeks):
        print(f"[Debug_Output]: Function '_read_raw_data' called with path={path}, week={week}, num_weeks={num_weeks}")
        fin = open(path)
        raw_data = pd.read_csv(fin)

        # FIX: Check if split_point column exists and has True values
        if "split_point" not in raw_data.columns:
            print(f"[Warning]: No 'split_point' column found in {path}. Using all data.")
            # If no split_point column, return all data
            out = raw_data.drop(columns=["Date"], errors='ignore').to_numpy()
            print(f"[Debug]: Returning all {len(raw_data)} rows")
            return out

        truncate_index = len(raw_data)
        stopping_point = -1
        truncation_mark = num_weeks - week

        # FIX: If no split points exist, don't truncate
        sp_col = raw_data["split_point"]
        has_split_points = sp_col.any()
        
        print(
            f"[Debug_Output]: _read_raw_data DIAGNOSTIC | split_point dtype={sp_col.dtype} | "
            f"unique_values={sp_col.unique().tolist()} | "
            f"sum_truthy={sp_col.astype(bool).sum()} | "
            f"has_split_points={has_split_points}"
        )

        if not has_split_points:
            print(f"[Info]: No split points found in {path}. Using all data without truncation.")
            # Return all data, no truncation
            out = raw_data.drop(["split_point"], axis=1).drop(columns=["Date"], errors='ignore').to_numpy()
            print(f"[Debug]: Returning all {len(raw_data)} rows, out_shape={out.shape}")
            return out

        # Only truncate if we have split points
        for point in reversed(raw_data["split_point"]):
            if stopping_point == truncation_mark:
                break

            if point:
                stopping_point += 1

            truncate_index -= 1

        out = raw_data.loc[:truncate_index].drop(["split_point"], axis=1).drop(columns=["Date"], errors='ignore').to_numpy()
        print(
            f"[Debug_Output]: _read_raw_data done | n_rows_full={len(raw_data)} | truncate_index={truncate_index} | "
            f"out_shape={out.shape} | week={week} truncation_mark={truncation_mark}"
        )
        return out

    @staticmethod
    def _convert_to_simple_returns(prices_np, horizon):
        print(f"[Debug_Output]: Function '_convert_to_simple_returns' called with prices_np.shape={prices_np.shape}, horizon={horizon}")
        if len(prices_np) <= horizon:
            print(f"[Warning]: Not enough data for conversion. prices_np length={len(prices_np)}, horizon={horizon}")
            return np.array([])
        simple_returns = prices_np[horizon:] / prices_np[:-horizon]
        return simple_returns