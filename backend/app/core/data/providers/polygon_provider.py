import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd
import requests

from backend.app.config import settings
from backend.app.core.data.providers.base_provider import DataProvider

logger = logging.getLogger(__name__)


class PolygonProvider(DataProvider):
    """
    Data provider for Polygon.io using REST API.
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(self):
        self.api_key = getattr(settings, "POLYGON_API_KEY", "")
        if not self.api_key:
            logger.warning("PolygonProvider: POLYGON_API_KEY is not set.")

    def _interval_to_polygon(self, interval: str) -> tuple[int, str]:
        """Convert yfinance interval to Polygon multiplier/timespan."""
        mapping = {
            "1m": (1, "minute"),
            "1h": (1, "hour"),
            "1d": (1, "day"),
            "1wk": (1, "week"),
            "1mo": (1, "month"),
        }
        return mapping.get(interval, (1, "day"))

    def _period_to_dates(self, period: str) -> tuple[str, str]:
        """Convert yfinance period to start/end dates for Polygon."""
        end_date = datetime.now()

        mapping = {
            "1mo": 30,
            "1y": 365,
            "2y": 730,
            "5y": 1825,
            "max": 3650,  # 10 years fallback
        }

        days = mapping.get(period, 365)
        start_date = end_date - timedelta(days=days)

        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    def fetch_data(self, symbol: str, period: str, interval: str, start: Optional[Any] = None, end: Optional[Any] = None) -> pd.DataFrame:
        if not self.api_key:
            return pd.DataFrame()

        try:
            multiplier, timespan = self._interval_to_polygon(interval)

            if not start or not end:
                start_date, end_date = self._period_to_dates(period)
            else:
                start_date = start if isinstance(start, str) else start.strftime("%Y-%m-%d")
                end_date = end if isinstance(end, str) else end.strftime("%Y-%m-%d")

            # Polygon standard ticker for stocks
            ticker = symbol.upper().replace("-", "")

            url = f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}"

            params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": self.api_key}

            response = requests.get(url, params=params)
            response.raise_for_status()
            result = response.json()

            if "results" not in result:
                logger.warning(f"PolygonProvider: No data found for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(result["results"])

            # Map Polygon columns to standard OHLCV
            # v: volume, vw: volume weighted, o: open, c: close, h: high, l: low, t: timestamp, n: transactions
            df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume", "t": "Timestamp"})

            df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
            df = df.set_index("Timestamp")

            # Ensure standard columns are present
            cols = ["Open", "High", "Low", "Close", "Volume"]
            df = df[cols]

            logger.info(f"PolygonProvider: Fetched {len(df)} bars for {symbol}")
            return df

        except Exception as e:
            logger.error(f"PolygonProvider: Error fetching {symbol}: {e}")
            return pd.DataFrame()
