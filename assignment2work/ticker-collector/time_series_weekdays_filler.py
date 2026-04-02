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
                          Example: "assignment2work/datasetsOut/fx/daily_10_3000_marked.csv"
    """
    df = pd.read_csv(input_file)
    
    # CHANGE: Parse dates with flexible format handling
    # Some files have date only, others have datetime with time component
    def parse_date(date_str):
        """Flexible date parser that handles both YYYY-MM-DD and YYYY-MM-DD HH:MM:SS formats"""
        try:
            # First try parsing with datetime format (includes time)
            return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            try:
                # If that fails, try date-only format
                return datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                # If both fail, raise original error with context
                raise ValueError(f"Unable to parse date string: '{date_str}'. Expected format YYYY-MM-DD or YYYY-MM-DD HH:MM:SS")
    
    # Get first and last dates using flexible parser
    first_date_str = df.iloc[0]["Date"]
    last_date_str = df.iloc[-1]["Date"]
    
    start_date = parse_date(first_date_str)
    end_date = parse_date(last_date_str)
    
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    
    all_weekdays = generate_all_weekdays(start_date, end_date)
    all_weekdays_df = pd.DataFrame({"Date": all_weekdays})
    
    # Convert Date columns to string for consistent merging
    # Use date-only string format for the merge
    all_weekdays_df["Date"] = all_weekdays_df["Date"].dt.strftime("%Y-%m-%d")
    
    # For the original dataframe, extract just the date part for merging
    # but preserve original values
    df["Date_for_merge"] = df["Date"].apply(lambda x: x[:10] if ' ' in x else x)
    
    merged_df = pd.merge(df, all_weekdays_df, left_on="Date_for_merge", right_on="Date", how="right")
    
    # Drop the temporary merge column and clean up
    merged_df = merged_df.drop("Date_for_merge", axis=1)
    
    # Rename columns to handle potential duplicates
    if "Date_x" in merged_df.columns and "Date_y" in merged_df.columns:
        # Keep the original Date column from the original dataframe
        merged_df["Date"] = merged_df["Date_y"]
        merged_df = merged_df.drop(["Date_x", "Date_y"], axis=1)
    
    # Fill missing values using forward fill
    # Note: ffill() is preferred over fillna(method='ffill')
    filled_df = merged_df.fillna(method="ffill")
    
    # Save files
    df.to_csv(input_file.replace(".csv", "_with_missing_days.csv"), index=False)
    filled_df.to_csv(input_file, index=False)
    
    print(f"Missing days filled. Output saved to: {input_file}")


def generate_all_weekdays(start_date, end_date):
    """
    Generates a list of all weekday dates datetime objects between start_date and end_date.
    
    Args:
        start_date (datetime): The beginning date for weekday generation.
                               Example: datetime(2020, 1, 1)
        end_date (datetime): The ending date for weekday generation.
                             Example: datetime(2021, 1, 1)
                             
    Returns:
        list of datetime: List of datetime objects for weekdays only.
    """
    all_weekdays = []
    current_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date_only = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    while current_date <= end_date_only:
        if current_date.weekday() < 5:  # Monday=0, Sunday=6, so <5 is Monday-Friday
            all_weekdays.append(current_date)
        current_date += timedelta(days=1)
    
    return all_weekdays


# Alternative simpler version if you want to just work with dates:
def main_simple(input_file):
    """
    Simpler version that converts all dates to date-only format first.
    Use this if you don't need to preserve the time component.
    """
    df = pd.read_csv(input_file)
    
    # Convert Date column to datetime and then to date-only format
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    
    start_date = datetime.strptime(df.iloc[0]["Date"], "%Y-%m-%d")
    end_date = datetime.strptime(df.iloc[-1]["Date"], "%Y-%m-%d")
    
    all_weekdays = generate_all_weekdays(start_date, end_date)
    all_weekdays_df = pd.DataFrame({"Date": [d.strftime("%Y-%m-%d") for d in all_weekdays]})
    
    merged_df = pd.merge(df, all_weekdays_df, on="Date", how="right")
    filled_df = merged_df.fillna(method="ffill")
    
    df.to_csv(input_file.replace(".csv", "_with_missing_days.csv"), index=False)
    filled_df.to_csv(input_file, index=False)
    
    print(f"Missing days filled. Output saved to: {input_file}")


# If you want to use the simpler version, uncomment below and comment out the main function above
# main = main_simple