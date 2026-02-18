import logging
import requests
from datetime import datetime
from typing import Any, Optional
import pandas as pd
from backend.app.core.data.base_provider import DataProvider
from backend.app.config import settings

logger = logging.getLogger(__name__)

class IEXProvider(DataProvider):
    """
    Data provider for IEX Cloud using REST API.
    """

    BASE_URL = "https://cloud.iexapis.com/stable"

    def __init__(self):
        self.api_key = getattr(settings, "IEX_API_KEY", "")
        if not self.api_key:
            logger.warning("IEXProvider: IEX_API_KEY is not set.")

    def _interval_to_iex(self, interval: str) -> str:
        """IEX generally handles chart periods rather than specific multipliers in stable endpoint."""
        # This is a simplification; IEX has specific endpoints for different granularities
        return interval

    def fetch_data(
        self, 
        symbol: str, 
        period: str, 
        interval: str, 
        start: Optional[Any] = None, 
        end: Optional[Any] = None
    ) -> pd.DataFrame:
        if not self.api_key:
            return pd.DataFrame()

        try:
            # IEX chart range mapping
            # 1m, 5m, 1y, 2y, 5y, max
            range_map = {
                "1mo": "1m",
                "1y": "1y",
                "2y": "2y",
                "5y": "5y",
                "max": "max"
            }
            iex_range = range_map.get(period, "1y")
            
            ticker = symbol.upper()
            url = f"{self.BASE_URL}/stock/{ticker}/chart/{iex_range}"
            
            params = {
                "token": self.api_key,
                "chartCloseOnly": "false"
            }

            response = requests.get(url, params=params)
            response.raise_for_status()
            result = response.json()

            if not result:
                logger.warning(f"IEXProvider: No data found for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(result)
            
            # Map IEX columns to standard OHLCV
            # date, open, high, low, close, volume
            df = df.rename(columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
                "date": "Timestamp"
            })

            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            df = df.set_index("Timestamp")
            
            # Filter by interval if needed (IEX returns daily by default for long ranges)
            # For simplicity, we assume daily for now as per chart endpoint standard
            
            # Ensure standard columns are present
            cols = ["Open", "High", "Low", "Close", "Volume"]
            df = df[cols]

            logger.info(f"IEXProvider: Fetched {len(df)} bars for {symbol}")
            return df

        except Exception as e:
            logger.error(f"IEXProvider: Error fetching {symbol}: {e}")
            return pd.DataFrame()
