import logging
from typing import Any, Optional
import pandas as pd
import requests
from backend.app.core.data.base_provider import DataProvider
from backend.app.config import settings

logger = logging.getLogger(__name__)

class YahooProvider(DataProvider):
    """
    Data provider for Yahoo Finance using yfinance.
    """

    def __init__(self):
        self.user_agent = getattr(settings, "USER_AGENT", "Mozilla/5.0")

    def fetch_data(
        self, 
        symbol: str, 
        period: str, 
        interval: str, 
        start: Optional[Any] = None, 
        end: Optional[Any] = None
    ) -> pd.DataFrame:
        try:
            import yfinance as yf

            # Sanitize crypto symbols
            if len(symbol) <= 5 and symbol.isalpha() and symbol in ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE"]:
                logger.debug(f"YahooProvider: Auto-correcting {symbol} to {symbol}-USD")
                symbol = f"{symbol}-USD"

            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval, start=start, end=end)

            if data.empty:
                logger.warning(f"YahooProvider: Ticker method failed for {symbol}, trying download method")
                data = yf.download(
                    symbol,
                    start=start,
                    end=end,
                    period=period,
                    interval=interval,
                    progress=False
                )

            if not data.empty:
                logger.info(f"YahooProvider: Fetched {len(data)} bars for {symbol}")
            
            return data

        except Exception as e:
            logger.error(f"YahooProvider: Error fetching {symbol}: {e}")
            return pd.DataFrame()
