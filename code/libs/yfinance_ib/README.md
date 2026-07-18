# yfinance-ib - Custom Yahoo Finance Library

## Overview

This is a custom-modified version of the `yfinance` library, optimized to work around authentication issues that were causing "Expecting value: line 1 column 1 (char 0)" errors when fetching historical data. The library has been patched to use a direct JSON API approach that bypasses the problematic cookie/crumb authentication system.

## Why This Version?

The original `yfinance` library encountered issues when downloading cryptocurrency and stock data, specifically failing with JSON parsing errors. This modified version:

- Bypasses the problematic timezone fetch that was causing errors
- Uses a reliable JSON API endpoint (`query2.finance.yahoo.com/v8/finance/chart/`) that works consistently
- Eliminates the need for complex cookie/crumb authentication
- Maintains full API compatibility with the original `yfinance` library

## Key Changes

### 1. `base.py` Modifications
- **`_get_ticker_tz()`**: Bypasses timezone fetch and returns 'UTC' directly
- **`_fetch_ticker_tz()`**: Returns 'UTC' without making network requests

### 2. `scrapers/history.py` Rewrite
- Replaced the entire history fetching mechanism with a direct JSON API approach
- Uses the same endpoint that was tested and confirmed working with BTC-USD
- Simplified data parsing and error handling

### 3. Maintained API Compatibility
- All public methods (`history()`, `get_dividends()`, etc.) remain unchanged
- Your existing code that uses `yfinance` will work without modification

## Installation

### Method 1: Replace System yfinance
```bash
# Replace with custom version
cp -r /path_to/yfinance_ib  /path_to_venv/lib/python3.12.7/site-packages/yfinance_ib
```

## Usage

The library works exactly like the original `yfinance`:

```python
import yfinance as yf

# Download data for a single ticker
btc = yf.Ticker("BTC-USD")
data = btc.history(period="max")
print(data.head())

# Download multiple tickers
tickers = ["BTC-USD", "ETH-USD", "SOL-USD"]
data = yf.download(tickers, period="1y")
print(data.head())
```

## Troubleshooting

### Issues I Encountered

**Error: "Expecting value: line 1 column 1 (char 0)"**
- This error should not occur with this patched version
- If it does, verify that the modifications were applied correctly

**404 Errors for Certain Tickers**
- Some tickers may be delisted or have different symbols
- Try checking Yahoo Finance directly for the correct symbol
- Example: MIOTA-USD might be IOTA-USD

**FutureWarnings about fillna**
- These warnings come from the original pipeline code, not this library
- To fix, update calls from `fillna(method='bfill')` to `bfill()`

### Verifying the Patch

To verify the patch is active, check for debug messages:
```python
import yfinance as yf
ticker = yf.Ticker("BTC-USD")
data = ticker.history(period="1mo")
# Should see: "=== USING CUSTOM HISTORY METHOD for BTC-USD ==="
```

## Files Structure

```
yfinance_ib/
├── __init__.py          # Main module entry point
├── base.py             # Modified TickerBase class
├── cache.py            # Caching system (unchanged)
├── const.py            # Constants (unchanged)
├── data.py             # Data fetching (unchanged)
├── exceptions.py       # Exception classes (unchanged)
├── multi.py            # Multi-ticker download (unchanged)
├── shared.py           # Shared state (unchanged)
├── ticker.py           # Ticker class (unchanged)
├── tickers.py          # Tickers class (unchanged)
├── utils.py            # Utilities (unchanged)
├── version.py          # Version info (unchanged)
└── scrapers/
    ├── __init__.py     # Scrapers init (unchanged)
    ├── analysis.py     # Analysis scrapers (unchanged)
    ├── fundamentals.py # Financial data (unchanged)
    ├── history.py      # **REWRITTEN** - Core history fetching
    ├── history_old.py  # Backup of original history.py
    ├── holders.py      # Holders data (unchanged)
    └── quote.py        # Quote data (unchanged)
```

## License

This library is a modified version of the original `yfinance` library, which is licensed under the Apache License, Version 2.0.

## Credits

- Original yfinance library: [ranaroussi/yfinance](https://github.com/ranaroussi/yfinance)
- This modification: Customized for stable cryptocurrency data fetching

## Support

If you encounter issues specific to this modified version:
1. Verify the modifications are correctly applied
2. Check the debug output for error messages
3. Test with a known working ticker (BTC-USD) first
4. For general yfinance questions, refer to the original project documentation

---

**Note**: This is a custom version created to work around specific authentication issues. 
