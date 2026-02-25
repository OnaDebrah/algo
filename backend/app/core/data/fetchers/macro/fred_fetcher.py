import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

try:
    from fredapi import Fred
except ImportError:
    Fred = None  # fredapi not installed — FREDFetcher will log a warning

from .....config import settings
from .....utils.helpers import SERIES_MAP
from .base_macro_fetcher import BaseMacroFetcher

logger = logging.getLogger(__name__)


class FREDFetcher(BaseMacroFetcher):
    """
    Federal Reserve Economic Data (FRED) Provider
    Sources: GDP, unemployment, CPI, interest rates, M2, etc.
    """

    def __init__(self):
        self.api_key = getattr(settings, "FRED_API_KEY", None)
        if not self.api_key:
            logger.warning("FRED_API_KEY not set. FRED provider will not work.")

        # Initialize fredapi client [citation:10]
        if Fred is None:
            logger.warning("fredapi package not installed. FRED provider will not work.")
            self.client = None
        else:
            self.client = Fred(api_key=self.api_key) if self.api_key else None

        # Cache for rate limiting
        self._cache = {}

        self.SERIES_MAP = SERIES_MAP

    async def get_indicators(
        self,
        indicators: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = "monthly",
        country: str = None,
    ) -> pd.DataFrame:
        """
        Fetch multiple FRED indicators

        Example:
            provider = FREDProvider()
            df = await provider.get_indicators(
                ["gdp_growth", "cpi_yoy", "unemployment_rate"],
                start_date=datetime(2015, 1, 1)
            )
        """
        if not self.client:
            logger.error("FRED client not initialized")
            return pd.DataFrame()

        data = {}

        for indicator in indicators:
            if indicator not in self.SERIES_MAP:
                logger.warning(f"Indicator {indicator} not mapped to FRED series")
                continue

            series_id = self.SERIES_MAP[indicator]
            try:
                # Fetch series data [citation:7][citation:10]
                series = await self._fetch_series(series_id, start_date, end_date)

                if series is not None and not series.empty:
                    # Apply frequency conversion
                    if frequency == "quarterly":
                        series = series.resample("Q").last()
                    elif frequency == "monthly":
                        series = series.resample("M").last()
                    elif frequency == "weekly":
                        series = series.resample("W").last()

                    data[indicator] = series

            except Exception as e:
                logger.error(f"Error fetching {indicator} ({series_id}): {e}")
                continue

        if not data:
            return pd.DataFrame()

        # Combine all series
        df = pd.DataFrame(data)

        # Calculate derived indicators
        df = self._calculate_derived_indicators(df)

        return df

    async def get_indicator(
        self,
        indicator_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = "monthly",
        country: str = None,
    ) -> pd.Series:
        """Fetch a single FRED indicator"""
        df = await self.get_indicators([indicator_id], start_date, end_date, frequency)
        return df[indicator_id] if not df.empty else pd.Series()

    async def _fetch_series(self, series_id: str, start_date: Optional[datetime], end_date: Optional[datetime]) -> pd.Series:
        """Fetch series from FRED with caching [citation:7]"""
        import asyncio

        # Check cache
        cache_key = f"{series_id}_{start_date}_{end_date}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # FRED API has rate limits, so add delay
            await asyncio.sleep(0.1)  # 10 requests per second max

            # Fetch using fredapi [citation:10]
            series = self.client.get_series(series_id, observation_start=start_date, observation_end=end_date)

            # Cache result
            self._cache[cache_key] = series

            logger.debug(f"Fetched {series_id}: {len(series)} observations")
            return series

        except Exception as e:
            logger.error(f"FRED API error for {series_id}: {e}")
            return pd.Series()

    def _calculate_derived_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived indicators like YoY changes, spreads"""

        # Calculate YoY changes for price indices
        for col in ["cpi_yoy", "gdp_growth", "retail_sales"]:
            if col in df.columns:
                df[f"{col}_yoy"] = df[col].pct_change(12) * 100

        # Calculate yield curve spread
        if "10y_treasury_yield" in df.columns and "2y_treasury_yield" in df.columns:
            df["2y10y_spread"] = df["10y_treasury_yield"] - df["2y_treasury_yield"]

        # Calculate real rate
        if "fed_funds_rate" in df.columns and "cpi_yoy" in df.columns:
            df["real_rate"] = df["fed_funds_rate"] - df["cpi_yoy"].pct_change(12) * 100

        return df

    def get_available_indicators(self) -> Dict[str, str]:
        """Return all available FRED indicators"""
        return {key: f"FRED series: {value}" for key, value in self.SERIES_MAP.items()}
