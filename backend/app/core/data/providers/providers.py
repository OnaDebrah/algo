import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ....config import settings
from ....models import UserSettings
from ....models.user import User
from ..providers.alpaca_provider import AlpacaProvider
from ..providers.base_provider import (
    FundamentalsProvider,
    NewsProvider,
    OptionsDataProvider,
    QuoteProvider,
    RecommendationsProvider,
)
from ..providers.iex_provider import IEXProvider
from ..providers.polygon_provider import PolygonProvider
from ..providers.yahoo_provider import YahooProvider

logger = logging.getLogger(__name__)


class ProviderFactory:
    """
    Factory / facade for data providers.

    For each data type, resolves the user's configured provider if it supports
    the requested capability (via isinstance check), otherwise falls back to
    the default YahooProvider which implements every interface.
    """

    _instance = None
    _default_provider = YahooProvider()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def _get_provider(
        self, user: Optional[User] = None, db: Optional[AsyncSession] = None
    ) -> Union[PolygonProvider, AlpacaProvider, IEXProvider, YahooProvider]:
        """Get provider for user or default"""
        if not user or not db:
            return self._default_provider

        try:
            stmt = select(UserSettings).where(UserSettings.user_id == user.id)
            result = await db.execute(stmt)
            user_settings = result.scalars().first()

            provider_type = user_settings.data_source

            if provider_type == "polygon" and settings.POLYGON_API_KEY:
                logger.debug("Using Polygon provider")
                return PolygonProvider()

            if provider_type == "alpaca" and settings.ALPACA_API_KEY:
                logger.debug("Using Alpaca provider")
                return AlpacaProvider()

            if provider_type == "iex" and settings.IEX_API_KEY:
                logger.debug("Using IEX provider")
                return IEXProvider()

        except Exception as e:
            logger.error(f"Error getting provider: {e}")

        return self._default_provider

    # ── OHLCV (existing) ────────────────────────────────────────────────

    async def fetch_data(
        self,
        symbol: str,
        period: str,
        interval: str,
        start: Any = None,
        end: Any = None,
        user: Optional[User] = None,
        db: Optional[AsyncSession] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data using appropriate provider."""
        provider = await self._get_provider(user, db)
        logger.info(f"Fetching {symbol} data from {provider.__class__.__name__}")
        return await asyncio.to_thread(provider.fetch_data, symbol, period, interval, start, end)

    # ── Quotes ──────────────────────────────────────────────────────────

    async def get_quote(self, symbol: str, user: Optional[User] = None, db: Optional[AsyncSession] = None) -> Dict:
        """Get real-time quote data. Falls back to Yahoo if provider lacks QuoteProvider."""
        provider = await self._get_provider(user, db)
        target = provider if isinstance(provider, QuoteProvider) else self._default_provider
        return await asyncio.to_thread(target.get_quote, symbol)

    # ── Options ─────────────────────────────────────────────────────────

    async def get_option_expirations(self, symbol: str, user: Optional[User] = None, db: Optional[AsyncSession] = None) -> List[str]:
        """Get available option expiration dates."""
        provider = await self._get_provider(user, db)
        target = provider if isinstance(provider, OptionsDataProvider) else self._default_provider
        return await asyncio.to_thread(target.get_option_expirations, symbol)

    async def get_option_chain(self, symbol: str, expiration: str, user: Optional[User] = None, db: Optional[AsyncSession] = None) -> Dict:
        """Fetch option chain for a specific expiration."""
        provider = await self._get_provider(user, db)
        target = provider if isinstance(provider, OptionsDataProvider) else self._default_provider
        return await asyncio.to_thread(target.get_option_chain, symbol, expiration)

    # ── Fundamentals ────────────────────────────────────────────────────

    async def get_ticker_info(self, symbol: str, user: Optional[User] = None, db: Optional[AsyncSession] = None) -> Dict:
        """Get ticker metadata / info dict."""
        provider = await self._get_provider(user, db)
        target = provider if isinstance(provider, FundamentalsProvider) else self._default_provider
        return await asyncio.to_thread(target.get_ticker_info, symbol)

    async def get_financials(self, symbol: str, user: Optional[User] = None, db: Optional[AsyncSession] = None) -> Dict:
        """Get comprehensive financial data (statements + info)."""
        provider = await self._get_provider(user, db)
        target = provider if isinstance(provider, FundamentalsProvider) else self._default_provider
        return await asyncio.to_thread(target.get_financials, symbol)

    # ── News ────────────────────────────────────────────────────────────

    async def get_news(self, symbol: str, limit: int = 10, user: Optional[User] = None, db: Optional[AsyncSession] = None) -> List[Dict]:
        """Get recent news items."""
        provider = await self._get_provider(user, db)
        target = provider if isinstance(provider, NewsProvider) else self._default_provider
        return await asyncio.to_thread(target.get_news, symbol, limit)

    # ── Recommendations ─────────────────────────────────────────────────

    async def get_recommendations(self, symbol: str, user: Optional[User] = None, db: Optional[AsyncSession] = None):
        """Get analyst recommendations."""
        provider = await self._get_provider(user, db)
        target = provider if isinstance(provider, RecommendationsProvider) else self._default_provider
        return await asyncio.to_thread(target.get_recommendations, symbol)
