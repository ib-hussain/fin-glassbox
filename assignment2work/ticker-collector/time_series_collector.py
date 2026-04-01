"""
This script collects historical time series data for a given list of tickers from Yahoo Finance 
and stores them as CSV files in the specified output directory.

It does not expect system arguments from the command line; inputs are instead passed programmatically.
"""
import os
import sys
import time
import yfinance as yf
from tqdm import tqdm
from tickers import crypto_tickers


def main(tickers, output_path):
    """
    Main function to download historical stock/crypto data and save it as CSV files.
    
    Args:
        tickers (list of str): List of ticker symbols to download data for. 
                               Example: ["BTC-USD", "ETH-USD"]
        output_path (str): The directory path where the downloaded CSV files will be saved.
                           Example: "datasetsIn/crypto-daily"
    """
    os.makedirs(output_path, exist_ok=True)
    print("Starting download of stock data...")
    # Downloader part is taken from @derekbanas, see: https://github.com/derekbanas/Python4Finance/blob/3064e244048930631d0a6c174709f4b6f561c4d0/Download%20Every%20Stock.ipynb
    stocks_not_downloaded = []
    def save_to_csv_from_yahoo(folder, ticker):
        """
        Helper function to download historical data for a specific ticker from Yahoo Finance and save it.
        
        Args:
            folder (str): The directory path to save the CSV file. Example: "data/crypto-daily"
            ticker (str): The ticker symbol to download data for. Example: "BTC-USD"
        """
        stock = yf.Ticker(ticker)  # Use default proxy settings from yfinance configuration
        # noinspection PyBroadException
        try:
            df = stock.history(period="max")
            if df.empty:   stocks_not_downloaded.append(ticker)
            else:
                the_file = folder + "/" + ticker.replace(".", "_") + ".csv"
                df.to_csv(the_file)
        except:
            stocks_not_downloaded.append(ticker)
            print("Couldn't Get Data for:", ticker)
    [save_to_csv_from_yahoo(output_path, ticker) for ticker in tqdm(tickers)]
    print("Save completed.")

    if stocks_not_downloaded:
        print("Stocks not downloaded:")
        print(stocks_not_downloaded)
if __name__ == "__main__":
    main(crypto_tickers, sys.argv[1])
