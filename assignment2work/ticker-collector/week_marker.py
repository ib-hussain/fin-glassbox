"""
This script processes a time series CSV to add a boolean 'split_point' column.
A split point indicates the final recorded day of each calendar week (often a Friday or Sunday depending on the asset), 
which is useful for defining boundaries in sequence-to-sequence deep learning models or graph neural networks.

It does not expect system arguments from the command line.
"""
from datetime import datetime

import pandas as pd
import os


def main(input_path):
    """
    Reads a merged CSV file, calculates which rows correspond to the last days of their respective calendar weeks,
    and appends a 'split_point' boolean column. The annotated dataframe is saved with a '_marked' suffix.
    
    Args:
        input_path (str): The filename/path of the aggregated time series CSV.
                          Example: "assignment2work/datasetsOut/crypto/daily_20_2190.csv"
    """
    output_path = f"{input_path.replace('.csv', '')}_marked.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.read_csv(input_path)
    dates_df = df["Date"]
    datetimes = [datetime.fromisoformat(date).isocalendar() for date in dates_df]
    last_days_of_weeks_df = (pd.DataFrame(datetimes,
                                          columns=["year", "week_of_year",
                                                   "day_of_week"]).groupby(["year",
                                                                            "week_of_year"])["day_of_week"].max())
    last_one_years_weeks_df = pd.DataFrame(
        last_days_of_weeks_df.reset_index().apply(
            lambda row: datetime.strptime(f"{row['year']} {row['week_of_year']} {row['day_of_week']}", "%G %V %u"),
            axis=1,
        ),
        columns=["Date"],
    )
    last_one_years_weeks_df["Date"] = last_one_years_weeks_df["Date"].apply(lambda date: date.strftime("%Y-%m-%d"))
    last_one_years_weeks_df["split_point"] = True
    df_with_split_points = df.merge(last_one_years_weeks_df, how="left", on="Date").fillna(False)

    df_with_split_points.set_index(["Date"]).to_csv(output_path)
    print(f"Work completed successfullly. \nMarked CSV saved to: {output_path}")


if __name__ == "__main__":
    main("assignment2work/datasetsOut/crypto/daily_20_2190.csv")
