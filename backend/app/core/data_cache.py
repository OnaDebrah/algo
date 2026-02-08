"""
Local Parquet-based caching service for market data
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.getcwd(), "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


class DataCache:
    """Handles local storage and retrieval of market data using Parquet format"""

    def __init__(self, cache_dir: str = CACHE_DIR, ttl_hours: int = 24):
        self.cache_dir = cache_dir
        self.ttl_hours = ttl_hours
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, symbol: str, period: str, interval: str) -> str:
        """Generate a unique filename for the cache entry"""
        # Sanitize symbol for filename (replace special characters)
        safe_symbol = symbol.replace("-", "_").replace("=", "_").replace(".", "_")
        return os.path.join(self.cache_dir, f"{safe_symbol}_{period}_{interval}.parquet")

    def get(self, symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Retrieve data from cache if it exists and is not expired"""
        cache_path = self._get_cache_path(symbol, period, interval)

        if not os.path.exists(cache_path):
            return None

        # Check TTL
        file_mtime = datetime.fromtimestamp(os.path.getmtime(cache_path), tz=timezone.utc)
        if datetime.now(timezone.utc) - file_mtime > timedelta(hours=self.ttl_hours):
            logger.info(f"Cache expired for {symbol} ({period}, {interval})")
            return None

        try:
            data = pd.read_parquet(cache_path)
            if not data.empty:
                logger.info(f"Cache hit for {symbol} ({period}, {interval})")
                return data
        except Exception as e:
            logger.error(f"Error reading cache for {symbol}: {e}")

        return None

    def set(self, symbol: str, period: str, interval: str, data: pd.DataFrame):
        """Save data to local Parquet cache"""
        if data.empty:
            return

        cache_path = self._get_cache_path(symbol, period, interval)
        try:
            data.to_parquet(cache_path)
            logger.info(f"Cached data for {symbol} ({period}, {interval}) at {cache_path}")
        except Exception as e:
            logger.error(f"Error saving cache for {symbol}: {e}")

    def clear(self):
        """Clear all cached files"""
        for f in os.listdir(self.cache_dir):
            if f.endswith(".parquet"):
                os.remove(os.path.join(self.cache_dir, f))
        logger.info("Cache cleared")


# Global cache instance
data_cache = DataCache()
