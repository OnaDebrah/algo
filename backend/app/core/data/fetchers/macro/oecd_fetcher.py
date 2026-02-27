import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import requests

from .....utils.helpers import COUNTRIES, DATASET_MAP
from .base_macro_fetcher import BaseMacroFetcher

logger = logging.getLogger(__name__)


class OECDFetcher(BaseMacroFetcher):
    """
    OECD Data Provider
    Sources: PMI, consumer confidence, leading indicators, international comparisons
    """

    def __init__(self):
        self.base_url = "https://sdmx.oecd.org/public/rest/data/v1"
        self._cache = {}

        self.DATASET_MAP = DATASET_MAP
        self.COUNTRIES = COUNTRIES

    async def get_indicators(
        self,
        indicators: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = "monthly",
        country: str = "USA",
    ) -> pd.DataFrame:
        """
        Fetch multiple OECD indicators

        Example:
            provider = OECDProvider()
            df = await provider.get_indicators(
                ["manufacturing_pmi", "consumer_confidence", "leading_indicator"],
                start_date=datetime(2015, 1, 1),
                country="USA"
            )
        """
        data = {}

        for indicator in indicators:
            try:
                series = await self._fetch_indicator(indicator, country, start_date, end_date)

                if series is not None and not series.empty:
                    data[indicator] = series

            except Exception as e:
                logger.error(f"Error fetching OECD {indicator} for {country}: {e}")
                continue

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Resample to desired frequency
        if frequency == "quarterly":
            df = df.resample("Q").last()
        elif frequency == "monthly":
            df = df.resample("M").last()

        return df

    async def _fetch_indicator(self, indicator: str, country: str, start_date: Optional[datetime], end_date: Optional[datetime]) -> pd.Series:
        """Fetch a single OECD indicator using SDMX query [citation:3]"""

        # SDMX query structure
        # Example: /ALL/MEI_CLI.M+...?startPeriod=2010&endPeriod=2024

        if indicator not in self.DATASET_MAP:
            logger.warning(f"Indicator {indicator} not mapped to OECD dataset")
            return pd.Series()

        dataset = self.DATASET_MAP[indicator]

        # Build SDMX query
        query_parts = []

        if indicator == "leading_indicator":
            query_parts = [dataset, "M", "LI...", "AA", country]  # Amplitude adjusted CLI
        elif indicator == "consumer_confidence":
            query_parts = [dataset, "M", "CCICP...", country]  # Consumer confidence
        elif indicator == "manufacturing_pmi":
            query_parts = [dataset, "M", "PMI...", "M", country]  # Manufacturing PMI
        else:
            # Generic fallback
            query_parts = [dataset, "M", country]

        query = "/".join(query_parts)

        # Build URL
        params = {
            "format": "jsondata",  # JSON format
            "startPeriod": start_date.year if start_date else "2010",
            "endPeriod": end_date.year if end_date else str(datetime.now().year),
        }

        url = f"{self.base_url}/{query}"

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Parse OECD JSON format
            series_data = self._parse_oecd_response(data)

            return series_data

        except Exception as e:
            logger.error(f"OECD API error for {indicator}: {e}")
            return pd.Series()

    def _parse_oecd_response(self, data: Dict) -> pd.Series:
        """Parse OECD SDMX JSON response into time series"""

        observations = []

        try:
            # Navigate to observation values
            for series in data.get("dataSets", [{}])[0].get("series", {}).values():
                for obs_key, obs_value in series.get("observations", {}).items():
                    # Parse period from structure
                    # This is simplified; actual parsing depends on OECD structure
                    if len(obs_value) > 0:
                        observations.append(float(obs_value[0]))

            # Get time periods from structure
            time_periods = []
            for index in data.get("structure", {}).get("dimensions", {}).get("observation", []):
                if index.get("id") == "TIME_PERIOD":
                    time_periods = index.get("values", [])
                    break

            # Create series
            if observations and time_periods:
                dates = [pd.to_datetime(t["name"]) for t in time_periods[: len(observations)]]
                return pd.Series(observations, index=dates)

        except Exception as e:
            logger.error(f"Error parsing OECD response: {e}")

        return pd.Series()

    async def get_indicator(
        self,
        indicator_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = "monthly",
        country: str = None,
    ) -> pd.Series:
        """Fetch a single OECD indicator"""
        df = await self.get_indicators([indicator_id], start_date, end_date, frequency)
        return df[indicator_id] if not df.empty else pd.Series()

    def get_available_indicators(self) -> Dict[str, str]:
        """Return all available OECD indicators"""
        return {
            "leading_indicator": "Composite Leading Indicator (CLI) - Amplitude Adjusted",
            "business_confidence": "Business Confidence Index (BCI)",
            "consumer_confidence": "Consumer Confidence Index (CCI) [citation:3]",
            "manufacturing_pmi": "Manufacturing Purchasing Managers Index",
            "gdp_growth": "GDP - Quarterly National Accounts",
            "cpi_yoy": "Consumer Price Index - All Items",
            "unemployment_rate": "Harmonised Unemployment Rate",
            "long_term_rates": "Long-term Interest Rates",
        }

    def get_countries(self) -> Dict[str, str]:
        """Return available countries"""
        return self.COUNTRIES
