import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from .....utils.helpers import SOURCE_PRIORITY
from .base_macro_fetcher import BaseMacroFetcher
from .bls_fetcher import BLSFetcher
from .cache import MacroCache
from .fred_fetcher import FREDFetcher
from .oecd_fetcher import OECDFetcher

logger = logging.getLogger(__name__)


class MacroManager:
    """
    Unified manager for all macro data providers
    Combines FRED, BLS, and OECD with intelligent fallback [citation:3][citation:6]
    """

    def __init__(self, db: AsyncSession = None):
        self.providers = {
            "fred": FREDFetcher(),
            "bls": BLSFetcher(),
            "oecd": OECDFetcher(),
        }

        self.cache = MacroCache(db)

        self.provider_status = {name: True for name in self.providers.keys()}
        self.SOURCE_PRIORITY = SOURCE_PRIORITY

        logger.info("MacroManager initialized with FRED, BLS, and OECD providers")

    async def get_indicators(
        self,
        indicators: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = "monthly",
        country: str = "USA",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Get macro indicators with automatic source selection and fallback

        Args:
            indicators: List of indicator names (e.g., ["gdp_growth", "cpi_yoy"])
            start_date: Start date for data
            end_date: End date for data
            frequency: 'daily', 'weekly', 'monthly', 'quarterly'
            country: Country code (for OECD data)
            use_cache: Whether to use cached data

        Returns:
            DataFrame with indicators as columns
        """

        # Check cache first
        if use_cache:
            cached = await self.cache.get(indicators, start_date, end_date, frequency)
            if cached is not None:
                logger.debug(f"Returning cached macro data for {len(indicators)} indicators")
                return cached

        # Group indicators by source based on priority
        source_groups = self._group_by_source(indicators)

        # Fetch from each source concurrently
        tasks = []
        for source, source_indicators in source_groups.items():
            if source in self.providers and self.provider_status[source]:
                provider = self.providers[source]
                tasks.append(self._fetch_from_provider(provider, source_indicators, start_date, end_date, frequency, country))

        if not tasks:
            logger.error("No providers available")
            return pd.DataFrame()

        # Wait for all fetches to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        combined_data = {}
        failed_sources = []

        for i, (source, _) in enumerate(source_groups.items()):
            result = results[i]
            if isinstance(result, Exception):
                logger.error(f"Provider {source} failed: {result}")
                self.provider_status[source] = False
                failed_sources.append(source)
            elif isinstance(result, pd.DataFrame) and not result.empty:
                for col in result.columns:
                    combined_data[col] = result[col]

        if not combined_data:
            logger.error("No data retrieved from any provider")
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(combined_data)

        # Try fallbacks for failed indicators
        if failed_sources:
            df = await self._try_fallbacks(df, failed_sources, indicators, start_date, end_date, frequency, country)

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Resample to consistent frequency
        if frequency == "monthly":
            df = df.resample("M").last()
        elif frequency == "quarterly":
            df = df.resample("Q").last()
        elif frequency == "weekly":
            df = df.resample("W").last()

        # Fill missing values
        df = df.fillna(method="ffill").fillna(method="bfill")

        # Cache result
        if use_cache:
            await self.cache.set(indicators, start_date, end_date, frequency, df)

        return df

    def _group_by_source(self, indicators: List[str]) -> Dict[str, List[str]]:
        """Group indicators by their primary data source"""
        groups = {}

        for indicator in indicators:
            sources = self.SOURCE_PRIORITY.get(indicator, ["fred"])

            # Use first available source
            for source in sources:
                if source in self.providers and self.provider_status[source]:
                    if source not in groups:
                        groups[source] = []
                    groups[source].append(indicator)
                    break
            else:
                # No source found, try FRED as last resort
                if "fred" not in groups:
                    groups["fred"] = []
                groups["fred"].append(indicator)

        return groups

    async def _fetch_from_provider(
        self,
        provider: BaseMacroFetcher,
        indicators: List[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        frequency: str,
        country: str,
    ) -> pd.DataFrame:
        """Fetch indicators from a specific provider"""

        if "OECD" in provider.__class__.__name__:
            return await provider.get_indicators(indicators, start_date, end_date, frequency, country=country)
        else:
            return await provider.get_indicators(indicators, start_date, end_date, frequency)

    async def _try_fallbacks(
        self,
        df: pd.DataFrame,
        failed_sources: List[str],
        indicators: List[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        frequency: str,
        country: str,
    ) -> pd.DataFrame:
        """Try fallback sources for indicators that failed"""

        # Identify missing indicators
        missing = [ind for ind in indicators if ind not in df.columns]

        if not missing:
            return df

        logger.info(f"Attempting fallbacks for missing indicators: {missing}")

        # Try each missing indicator with its next priority source
        for indicator in missing:
            sources = self.SOURCE_PRIORITY.get(indicator, ["fred"])

            for source in sources:
                if source in failed_sources or source not in self.providers:
                    continue

                try:
                    provider = self.providers[source]

                    if "OECD" in provider.__class__.__name__:
                        series = await provider.get_indicator(indicator, start_date, end_date, frequency, country=country)
                    else:
                        series = await provider.get_indicator(indicator, start_date, end_date, frequency)

                    if not series.empty:
                        df[indicator] = series
                        logger.info(f"Retrieved {indicator} from fallback source {source}")
                        break

                except Exception as e:
                    logger.debug(f"Fallback {source} failed for {indicator}: {e}")
                    continue

        return df

    async def get_indicator(
        self,
        indicator: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = "monthly",
        country: str = "USA",
    ) -> pd.Series:
        """Get a single indicator"""
        df = await self.get_indicators([indicator], start_date, end_date, frequency, country)
        return df[indicator] if not df.empty else pd.Series()

    def get_all_available_indicators(self) -> Dict[str, Dict[str, str]]:
        """Get all available indicators from all providers"""
        all_indicators = {}

        for provider_name, provider in self.providers.items():
            all_indicators[provider_name] = provider.get_available_indicators()

        return all_indicators

    def get_indicator_info(self, indicator: str) -> Optional[Dict]:
        """Get information about a specific indicator"""
        for provider_name, provider in self.providers.items():
            available = provider.get_available_indicators()
            if indicator in available:
                return {
                    "indicator": indicator,
                    "description": available[indicator],
                    "primary_source": provider_name,
                    "alternate_sources": self.SOURCE_PRIORITY.get(indicator, [])[1:],
                }

        return None

    def reset_provider_status(self):
        """Reset provider status (e.g., after rate limit cooldown)"""
        for provider in self.providers.keys():
            self.provider_status[provider] = True
        logger.info("Provider status reset")
