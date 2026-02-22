import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd
import requests

from backend.app.config import settings
from backend.app.core.data.providers.base_provider import DataProvider

logger = logging.getLogger(__name__)


class AlpacaProvider(DataProvider):
    """
    Data provider for Alpaca using Market Data v2 REST API.
    """

    BASE_URL = "https://data.alpaca.markets/v2"

    def __init__(self):
        self.api_key = getattr(settings, "ALPACA_API_KEY", "")
        self.api_secret = getattr(settings, "ALPACA_SECRET", "")
        if not self.api_key or not self.api_secret:
            logger.warning("AlpacaProvider: ALPACA_API_KEY or ALPACA_SECRET is not set.")

    def _interval_to_alpaca(self, interval: str) -> str:
        """Convert yfinance interval to Alpaca timeframe."""
        mapping = {
            "1m": "1Min",
            "1h": "1Hour",
            "1d": "1Day",
            "1wk": "1Week",
        }
        return mapping.get(interval, "1Day")

    def _period_to_dates(self, period: str) -> tuple[str, str]:
        """Convert yfinance period to RFC3339 start/end dates for Alpaca."""
        end_date = datetime.now()

        mapping = {
            "1mo": 30,
            "1y": 365,
            "2y": 730,
            "5y": 1825,
            "max": 3650,
        }

        days = mapping.get(period, 365)
        start_date = end_date - timedelta(days=days)

        return start_date.strftime("%Y-%m-%dT%H:%M:%SZ"), end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

    def fetch_data(self, symbol: str, period: str, interval: str, start: Optional[Any] = None, end: Optional[Any] = None) -> pd.DataFrame:
        if not self.api_key or not self.api_secret:
            return pd.DataFrame()

        try:
            timeframe = self._interval_to_alpaca(interval)

            if not start or not end:
                start_date, end_date = self._period_to_dates(period)
            else:
                start_date = start if isinstance(start, str) else start.strftime("%Y-%m-%dT%H:%M:%SZ")
                end_date = end if isinstance(end, str) else end.strftime("%Y-%m-%dT%H:%M:%SZ")

            # Alpaca expects upper case
            ticker = symbol.upper().replace("-", "")

            # Use stocks bar endpoint
            url = f"{self.BASE_URL}/stocks/bars"

            params = {
                "symbols": ticker,
                "timeframe": timeframe,
                "start": start_date,
                "end": end_date,
                "adjustment": "all",  # Split and dividend adjusted
                "feed": "sip",  # Use SIP feed if available, otherwise 'iex' if preferred
            }

            headers = {"APCA-API-KEY-ID": self.api_key, "APCA-API-SECRET-KEY": self.api_secret}

            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            result = response.json()

            if not result or "bars" not in result or not result["bars"] or ticker not in result["bars"]:
                logger.warning(f"AlpacaProvider: No data found for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(result["bars"][ticker])

            # Map Alpaca columns to standard OHLCV
            # o: open, h: high, l: low, c: close, v: volume, t: timestamp, vw: volume weighted, n: transactions
            df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume", "t": "Timestamp"})

            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            df = df.set_index("Timestamp")

            # Ensure standard columns are present
            cols = ["Open", "High", "Low", "Close", "Volume"]
            df = df[cols]

            logger.info(f"AlpacaProvider: Fetched {len(df)} bars for {symbol}")
            return df

        except Exception as e:
            logger.error(f"AlpacaProvider: Error fetching {symbol}: {e}")
            return pd.DataFrame()
