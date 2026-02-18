"""
Data fetching utilities for stock market data
"""

import logging
from typing import Any

import pandas as pd
import requests
import yfinance as yf

from streamlit.config import USER_AGENT

logger = logging.getLogger(__name__)


def create_yfinance_session():
    """Create a requests session with proper headers for yfinance"""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def fetch_stock_data(symbol: str, period: str, interval: str, start: Any = None, end: Any = None) -> pd.DataFrame:
    """
    Fetch stock data with better error handling

    Args:
        symbol: Stock ticker symbol
        period: Time period (1mo, 3mo, 6mo, 1y, 2y, 5y)
        interval: Data interval (1h, 1d, 1wk)

    Returns:
        DataFrame with OHLCV data
    """
    try:
        # Method 1: Use Ticker with session
        create_yfinance_session()
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)

        if data.empty:
            # Method 2: Try download as fallback
            logger.warning(f"Ticker method failed for {symbol}, trying download method")
            data = yf.download(
                symbol,
                start,
                end,
                period=period,
                interval=interval,
                progress=False,
            )

        if not data.empty:
            logger.info(f"Successfully fetched {len(data)} data points for {symbol}")

        return data

    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()


def validate_interval_period(interval: str, period: str) -> tuple:
    """
    Validate and adjust interval/period combinations

    Args:
        interval: Data interval
        period: Time period

    Returns:
        Tuple of (valid_interval, valid_period)
    """
    # Hourly data only available for shorter periods
    if interval == "1h" and period in ["1y", "2y", "5y"]:
        logger.warning(f"Adjusting interval from {interval} to 1d for period {period}")
        return "1d", period

    return interval, period
