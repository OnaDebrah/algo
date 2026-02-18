import asyncio
import logging
from typing import Any, Optional, Union

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import settings
from backend.app.core.data.alpaca_provider import AlpacaProvider
from backend.app.core.data.iex_provider import IEXProvider
from backend.app.core.data.polygon_provider import PolygonProvider
from backend.app.core.data.yahoo_provider import YahooProvider
from backend.app.models import UserSettings
from backend.app.models.user import User

logger = logging.getLogger(__name__)


class ProviderFactory:
    """Factory class for managing data providers"""

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
        """Fetch data using appropriate provider"""
        provider = await self._get_provider(user, db)

        logger.info(f"Fetching {symbol} data from {provider.__class__.__name__}")

        return await asyncio.to_thread(provider.fetch_data, symbol, period, interval, start, end)
