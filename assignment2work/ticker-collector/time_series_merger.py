"""
This script merges individual time series CSV files into a single combined CSV file.
It aligns multiple asset streams into a unified dataset using either an inner join or backfilling.

System Arguments Expected:
    1. input_path (str): The base directory holding the subfolder with the time series files.
                         Example: "assignment2work/datasetsIn/crypto"
    2. output_path (str): The directory to save the output merged CSV file.
                          Example: "assignment2work/datasetsOut/crypto"
    3. frequency (str): The frequency string expected as a suffix to the input folder name.
                        Example: "daily"
    4. num_assets (str): The number of top assets to include, ranked by the count of available dates.
                         Example: "20"
    5. b_fill_days (str): The number of days from the end of the series to slice and backward fill.
                          Example: "2190"
"""
import os
import sys
from dataclasses import dataclass
from functools import reduce
import pandas as pd


@dataclass
class StockDF:
    """
    Data class representing a stock's dataframe and its corresponding ticker name.
    """
    name: str
    df: pd.DataFrame

    def copy(self, df):
        """
        Returns a new StockDF instance with the same name but a different dataframe.
        
        Args:
            df (pd.DataFrame): The new dataframe to associate with this stock's name.
                               Example: pd.DataFrame({"Date": [...], "Close": [...]})
        """
        return StockDF(self.name, df)
def main(input_path, output_path, frequency, num_assets, b_fill_days=None):
    """
    Coordinates the loading of multiple single-ticker CSVs, merges them by date, and saves the output.

    Args:
        input_path (str): Base path to the input data directory. Example: "assignment2work/datasetsIn/crypto"
        output_path (str): Path to write the output merged CSV. Example: "assignment2work/datasetsOut/crypto"
        frequency (str): Suffix defining the subfolder resolution. Example: "daily"
        num_assets (int): Keep only this many tickers with the most data rows. Example: 20
        b_fill_days (int, optional): If provided, applies backfilling for this many recent days; 
                                     otherwise, uses an inner join. Example: 2190
    """
    paths = next(os.walk(f"{input_path}-{frequency}"))[2]
    dfs = sorted(
        [StockDF(
            remove_csv_extension(path),
            pd.read_csv(f"{input_path}-{frequency}/{path}"),
        ) for path in paths],
        key=lambda stock_df: len(stock_df.df),
    )[-num_assets:]
    merged_df = (merge_by_bfilling(dfs, b_fill_days) if b_fill_days else merge_by_inner_join(dfs))
    os.makedirs(output_path, exist_ok=True)
    target_path = f"{output_path}/{frequency}_{num_assets}_{len(merged_df)}.csv"
    print(f"Target path: {target_path}")
    merged_df.to_csv(target_path)
def remove_csv_extension(file_name): 
    """
    Utility to remove the .csv extension from a file name.
    
    Args:
        file_name (str): Original file name. Example: "BTC-USD.csv"
    """
    return file_name.replace(".csv", "")
def merge_by_bfilling(dfs, b_fill_days):
    """
    Merges dataframes using an outer join on 'Date' and backward fills missing values.
    Limits each ticker's dataframe to its last `b_fill_days` rows prior to merging.

    Args:
        dfs (list of StockDF): Dataframe wrappers to merge. Example: [StockDF("BTC", df1), ...]
        b_fill_days (int): Limit data to the most recent X days. Example: 2190
    """
    dfs_aligned = [stock_df.copy(df=stock_df.df[-b_fill_days:]) for stock_df in dfs]
    dfs_close = [
        stock_df.df[["Date", "Close"]].rename(columns={
            "Close": stock_df.name
        }).set_index("Date") for stock_df in dfs_aligned
    ]
    merged_df = reduce(lambda df1, df2: df1.merge(df2, on="Date", how="outer"), dfs_close)
    return merged_df.sort_values(by="Date").fillna(method="bfill", axis=0)
def merge_by_inner_join(dfs):
    """
    Merges dataframes iteratively using an inner join on date.
    Before joining, it trims all assets down to the length of the shortest sequence.
    
    Args:
        dfs (list of StockDF): Dataframe wrappers to merge. Example: [StockDF("EUR=X", df1), ...]
    """
    number_of_days = sorted(list(map(lambda stock_df: len(stock_df.df), dfs)))[0]
    dfs_aligned = [stock_df.copy(df=stock_df.df[-number_of_days:]) for stock_df in dfs]
    dfs_close = [
        stock_df.df[["Date", "Close"]].rename(columns={
            "Close": stock_df.name
        }).set_index("Date") for stock_df in dfs_aligned
    ]
    return reduce(lambda df1, df2: df1.merge(df2, on="Date"), dfs_close)
if __name__ == "__main__":
    input_path, output_path, frequency, num_assets, b_fill_days = sys.argv[1:]
    main(input_path, output_path, frequency, int(num_assets), int(b_fill_days))
