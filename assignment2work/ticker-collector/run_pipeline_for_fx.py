"""
This script executes the complete data collection, merging, week-marking, and missing-day filling
pipeline for foreign exchange (FX) currency tickers.

System Arguments Expected:
    This script is designed to be run without system arguments. It uses predefined directories and logic parameters.
"""
import os
import time

import time_series_collector #understood this
import time_series_merger    
import time_series_weekdays_filler
import week_marker
from tickers import fx_tickers

start_time = time.time()
print("Collecting time series...")
time_series_collector.main(fx_tickers, "assignment2work/datasetsIn/fx-daily")
print(f"Time series are collected in {time.time() - start_time} seconds.")

start_time = time.time()
print("Merging time series...")
time_series_merger.main(f"assignment2work/datasetsIn/fx", "assignment2work/datasetsOut/fx", "daily", len(fx_tickers))
print(f"Time series are merged in {time.time() - start_time} seconds.")

start_time = time.time()
print("Marking weeks...")
merged_file_name = next(os.walk("assignment2work/datasetsOut/fx"))[2][0]
week_marker.main(f"assignment2work/datasetsOut/fx/{merged_file_name}")
print(f"Weeks are marked in {time.time() - start_time} seconds.")

start_time = time.time()
print("Filling missing days...")
marked_file_name = merged_file_name.replace(".csv", "_marked.csv")
time_series_weekdays_filler.main(f"assignment2work/datasetsOut/fx/{marked_file_name}")
print(f"Filled missing days in {time.time() - start_time} seconds.")
