import datetime as _datetime
import dateutil as _dateutil
import logging
import numpy as np
import pandas as pd
import time as _time
import requests
import io
import json

from yfinance_ib import shared, utils
from yfinance_ib.const import _BASE_URL_, _PRICE_COLNAMES_

class PriceHistory:
    def __init__(self, data, ticker, tz, session=None, proxy=None):
        self._data = data
        self.ticker = ticker.upper()
        self.tz = tz
        self.proxy = proxy
        self.session = session

        self._history = None
        self._history_metadata = None
        self._history_metadata_formatted = False

        # Limit recursion depth when repairing prices
        self._reconstruct_start_interval = None
    @utils.log_indent_decorator
    def history(self, period="1mo", interval="1d",
                start=None, end=None, prepost=False, actions=True,
                auto_adjust=True, back_adjust=False, repair=False, keepna=False,
                proxy=None, rounding=False, timeout=10,
                raise_errors=False) -> pd.DataFrame:
        """
        CHANGE: Modified to use JSON API endpoint that works reliably.
        """
        logger = utils.get_yf_logger()
        logger.info(f"=== USING CUSTOM HISTORY METHOD for {self.ticker} ===")
        
        try:
            # Use JSON API method that we know works
            df = self._json_api_download(period, start, end, interval, timeout)
            
            if df is None or df.empty:
                logger.warning(f"No data received for {self.ticker}")
                return utils.empty_df()
            
            # Set index name
            df.index.name = "Date"
            
            logger.info(f"Successfully downloaded {len(df)} rows for {self.ticker}")
            return df
            
        except Exception as e:
            error_msg = f"JSON API download failed for {self.ticker}: {e}"
            logger.error(error_msg)
            if raise_errors:
                raise Exception(error_msg)
            return utils.empty_df()
    def _json_api_download(self, period="max", start=None, end=None, interval="1d", timeout=10):
        """
        CHANGE: New method - uses the JSON API endpoint that worked in our tests.
        This is the same endpoint: https://query2.finance.yahoo.com/v8/finance/chart/BTC-USD
        """
        try:
            # Calculate date range
            if end is None:
                end_date = _datetime.datetime.now()
            else:
                end_date = pd.to_datetime(end)
                
            if start is None:
                if period == "max":
                    start_date = end_date - _datetime.timedelta(days=20*365)  # 20 years
                elif period == "1y":
                    start_date = end_date - _datetime.timedelta(days=365)
                elif period == "6mo":
                    start_date = end_date - _datetime.timedelta(days=180)
                elif period == "3mo":
                    start_date = end_date - _datetime.timedelta(days=90)
                elif period == "1mo":
                    start_date = end_date - _datetime.timedelta(days=30)
                elif period == "5d":
                    start_date = end_date - _datetime.timedelta(days=5)
                elif period == "1d":
                    start_date = end_date - _datetime.timedelta(days=1)
                else:
                    start_date = end_date - _datetime.timedelta(days=365)
            else:
                start_date = pd.to_datetime(start)
            
            # Convert to timestamps (seconds since epoch)
            start_timestamp = int(start_date.timestamp())
            end_timestamp = int(end_date.timestamp())
            
            # Use the JSON API endpoint that worked in testing
            url = f"{_BASE_URL_}/v8/finance/chart/{self.ticker}"
            
            params = {
                'period1': start_timestamp,
                'period2': end_timestamp,
                'interval': interval,
                'events': 'history',
                'includePrePost': 'false'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://finance.yahoo.com/',
            }
            
            session = requests.Session()
            response = session.get(url, headers=headers, params=params, timeout=timeout)
            
            if response.status_code == 200:
                data = response.json()
                
                # Parse the JSON response
                if 'chart' in data and data['chart']['result']:
                    result = data['chart']['result'][0]
                    
                    # Extract timestamps and price data
                    timestamps = result.get('timestamp', [])
                    quote_data = result.get('indicators', {}).get('quote', [{}])[0]
                    
                    if timestamps and quote_data:
                        # Create DataFrame
                        df = pd.DataFrame({
                            'Open': quote_data.get('open', []),
                            'High': quote_data.get('high', []),
                            'Low': quote_data.get('low', []),
                            'Close': quote_data.get('close', []),
                            'Volume': quote_data.get('volume', [])
                        })
                        
                        # Add adjusted close if available
                        adjclose_data = result.get('indicators', {}).get('adjclose', [{}])
                        if adjclose_data and 'adjclose' in adjclose_data[0]:
                            df['Adj Close'] = adjclose_data[0]['adjclose']
                        else:
                            # If no adjclose, use close
                            df['Adj Close'] = df['Close']
                        
                        # Set index to timestamps
                        df.index = pd.to_datetime(timestamps, unit='s')
                        
                        # Remove any rows with all NaN values
                        df = df.dropna(how='all')
                        
                        # Add empty dividends and splits columns
                        df['Dividends'] = 0.0
                        df['Stock Splits'] = 0.0
                        
                        return df
                    else:
                        utils.get_yf_logger().error(f"No quote data in response for {self.ticker}")
                        return pd.DataFrame()
                else:
                    utils.get_yf_logger().error(f"No chart data in response for {self.ticker}")
                    return pd.DataFrame()
            else:
                utils.get_yf_logger().error(f"JSON API request failed with status {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            utils.get_yf_logger().error(f"JSON API error: {e}")
            return pd.DataFrame()

    # Keep simplified versions of other methods for compatibility
    def get_history_metadata(self, proxy=None) -> dict:
        """CHANGE: Simplified metadata"""
        if self._history_metadata is None:
            self._history_metadata = {
                'exchangeTimezoneName': self.tz if self.tz else 'UTC',
                'instrumentType': 'CRYPTOCURRENCY',
                'currency': 'USD',
                'exchangeName': 'CCC'
            }
        return self._history_metadata

    def get_dividends(self, proxy=None) -> pd.Series:
        return pd.Series()

    def get_capital_gains(self, proxy=None) -> pd.Series:
        return pd.Series()

    def get_splits(self, proxy=None) -> pd.Series:
        return pd.Series()

    def get_actions(self, proxy=None) -> pd.Series:
        return pd.Series()

    # Simplified repair methods
    def _reconstruct_intervals_batch(self, df, interval, prepost, tag=-1):
        return df

    def _fix_unit_mixups(self, df, interval, tz_exchange, prepost):
        return df

    def _fix_unit_random_mixups(self, df, interval, tz_exchange, prepost):
        return df

    def _fix_unit_switch(self, df, interval, tz_exchange):
        return df

    def _fix_zeroes(self, df, interval, tz_exchange, prepost):
        return df

    def _fix_missing_div_adjust(self, df, interval, tz_exchange):
        return df

    def _fix_bad_stock_split(self, df, interval, tz_exchange):
        return df

    def _fix_prices_sudden_change(self, df, interval, tz_exchange, change, correct_volume=False):
        return df