import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import requests

from .....config import settings
from .....utils.helpers import BLS_SERIES_MAP
from .base_macro_fetcher import BaseMacroFetcher

BLS_BASE_URL = settings.BLS_BASE_URL


logger = logging.getLogger(__name__)


class BLSFetcher(BaseMacroFetcher):
    """
    Bureau of Labor Statistics Provider
    Sources: Employment, CPI components, wages, productivity [citation:2][citation:5]
    """

    def __init__(self):
        self.api_key = getattr(settings, "BLS_API_KEY", None)
        self.base_url = BLS_BASE_URL

        # BLS allows 25 series per request [citation:2]
        self.batch_size = 25

        self._cache = {}

        self.BLS_SERIES_MAP = BLS_SERIES_MAP

    async def get_indicators(
        self,
        indicators: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = "monthly",
        country: str = None,
    ) -> pd.DataFrame:
        """
        Fetch multiple BLS indicators

        Example:
            provider = BLSProvider()
            df = await provider.get_indicators(
                ["unemployment_rate", "cpi_all", "avg_hourly_earnings"],
                start_date=datetime(2015, 1, 1)
            )
        """
        if not indicators:
            return pd.DataFrame()

        # Map to series IDs
        series_ids = []
        indicator_map = {}

        for indicator in indicators:
            if indicator in self.BLS_SERIES_MAP:
                series_id = self.BLS_SERIES_MAP[indicator]
                series_ids.append(series_id)
                indicator_map[series_id] = indicator
            else:
                logger.warning(f"Indicator {indicator} not mapped to BLS series")

        if not series_ids:
            return pd.DataFrame()

        # Fetch in batches
        all_data = []

        for i in range(0, len(series_ids), self.batch_size):
            batch = series_ids[i : i + self.batch_size]
            batch_data = await self._fetch_batch(batch, start_date, end_date)
            all_data.append(batch_data)

        # Combine all data
        combined = {}
        for batch_df in all_data:
            for col in batch_df.columns:
                if col in indicator_map:
                    combined[indicator_map[col]] = batch_df[col]
                else:
                    combined[col] = batch_df[col]

        df = pd.DataFrame(combined)

        # Ensure datetime index
        df.index = pd.to_datetime(df.index)

        # Resample to desired frequency
        if frequency == "quarterly":
            df = df.resample("Q").last()
        elif frequency == "monthly":
            df = df.resample("M").last()

        return df

    async def _fetch_batch(self, series_ids: List[str], start_date: Optional[datetime], end_date: Optional[datetime]) -> pd.DataFrame:
        """Fetch a batch of series from BLS API [citation:2]"""

        payload = {
            "seriesid": series_ids,
            "startyear": start_date.year if start_date else "2010",
            "endyear": end_date.year if end_date else datetime.now().year,
            "registrationkey": self.api_key,
            "catalog": False,
            "calculations": False,
            "annualaverage": False,
            "aspects": False,
        }

        try:
            # BLS API requires POST requests [citation:2]
            response = requests.post(self.base_url, json=payload, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            result = response.json()

            if result.get("status") != "REQUEST_SUCCEEDED":
                logger.error(f"BLS API error: {result.get('message', 'Unknown error')}")
                return pd.DataFrame()

            # Parse response
            data = {}
            for series in result.get("Results", {}).get("series", []):
                series_id = series["seriesID"]
                series_data = []

                for item in series.get("data", []):
                    try:
                        date_str = f"{item['year']}-{item['period'][1:]}-01"
                        value = float(item["value"])
                        series_data.append({"date": pd.to_datetime(date_str), "value": value, "period": item["period"]})
                    except (ValueError, KeyError):
                        continue

                if series_data:
                    df = pd.DataFrame(series_data)
                    df = df.set_index("date")
                    df = df.sort_index()
                    data[series_id] = df["value"]

            return pd.DataFrame(data)

        except Exception as e:
            logger.error(f"Error fetching BLS batch: {e}")
            return pd.DataFrame()

    async def get_indicator(
        self,
        indicator_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = "monthly",
        country: str = None,
    ) -> pd.Series:
        """Fetch a single BLS indicator"""
        df = await self.get_indicators([indicator_id], start_date, end_date, frequency)
        return df[indicator_id] if not df.empty else pd.Series()

    def get_available_indicators(self) -> Dict[str, str]:
        """Return all available BLS indicators"""
        descriptions = {
            "cpi_all": "CPI-U All Items, Urban Consumers",
            "unemployment_rate": "Unemployment Rate (U-3) [citation:5]",
            "employment_total": "Total Nonfarm Employment [citation:2]",
            "avg_hourly_earnings": "Average Hourly Earnings of All Employees [citation:8]",
            "job_openings": "Job Openings: Total Nonfarm [citation:2]",
            "quits_rate": "Quits Rate: Total Nonfarm [citation:2]",
            "labor_force": "Civilian Labor Force Level [citation:5]",
            "employment_level": "Civilian Employment Level [citation:5]",
        }

        return {key: descriptions.get(key, "BLS economic indicator") for key in self.BLS_SERIES_MAP.keys()}
