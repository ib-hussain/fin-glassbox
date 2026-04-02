"""
This script executes the complete data collection, merging, and week-marking pipeline for cryptocurrency tickers.
It runs the jobs sequentially and prints the time taken for each respective step.

System Arguments Expected:
    This script encapsulates the pipeline and runs with predefined data paths and parameters; 
    it expects no system arguments.
"""
import time

import time_series_collector
import time_series_merger
import week_marker
from tickers import crypto_tickers

start_time = time.time()
print("Collecting time series...")
time_series_collector.main(crypto_tickers, "assignment2work/datasetsIn/crypto-daily")
print(f"Time series are collected in {time.time() - start_time} seconds.")

start_time = time.time()
print("Merging time series...")
time_series_merger.main("assignment2work/datasetsIn/crypto", "assignment2work/datasetsOut/crypto", "daily", 20, 2190)
print(f"Time series are merged in {time.time() - start_time} seconds.")

start_time = time.time()
print("Marking weeks...")
week_marker.main("assignment2work/datasetsOut/crypto/daily_20_2190.csv")
print(f"Weeks are marked in {time.time() - start_time} seconds.")
