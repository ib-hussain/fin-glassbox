"""
This module processes an existing time series CSV to insert rows for any missing weekdays.
It fills missing sequence using forward filling (ffill) and outputs both the explicitly marked missing days file 
and the completely filled days file.

It does not expect system arguments from the command line; inputs are instead passed programmatically.
"""
from datetime import datetime, timedelta

import pandas as pd


def main(input_file):
    """
    Reads a CSV, identifies missing weekdays between the start and end dates, 
    fills the missing row values via forward filling, and saves the new CSV files.
    
    Args:
        input_file (str): Path to the single asset or merged CSV file.
                          Example: "out/fx/daily_10_3000_marked.csv"
    """
    df = pd.read_csv(input_file)

    all_weekdays = generate_all_weekdays(
        datetime.strptime(df.iloc[0]["Date"], "%Y-%m-%d"),
        datetime.strptime(df.iloc[-1]["Date"], "%Y-%m-%d"),
    )
    all_weekdays_df = pd.DataFrame({"Date": all_weekdays})

    merged_df = pd.merge(df, all_weekdays_df, on="Date", how="right")
    filled_df = merged_df.fillna(method="ffill")

    df.to_csv(input_file.replace(".csv", "_with_missing_days.csv"), index=False)
    filled_df.to_csv(input_file, index=False)


def generate_all_weekdays(start_date, end_date):
    """
    Generates a list of all weekday dates string representations between start_date and end_date.
    
    Args:
        start_date (datetime): The beginning date for weekday generation.
                               Example: datetime(2020, 1, 1)
        end_date (datetime): The ending date for weekday generation.
                             Example: datetime(2021, 1, 1)
                             
    Returns:
        list of str: List of date strings in 'YYYY-MM-DD' format.
    """
    all_weekdays = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:
            all_weekdays.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)

    return all_weekdays
