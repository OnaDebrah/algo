import asyncio
import logging
from typing import Any, Optional

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.data.providers import ProviderFactory
from backend.app.core.data.yahoo_provider import YahooProvider
from backend.app.core.data_cache import data_cache
from backend.app.models import User

logger = logging.getLogger(__name__)


async def fetch_stock_data(
    symbol: str, period: str, interval: str, start: Any = None, end: Any = None, user: Optional[User] = None, db: Optional[AsyncSession] = None
) -> pd.DataFrame:
    """
    Fetch market data with caching using configured multi-provider layer.
    """
    cached_data = data_cache.get(symbol, period, interval, start, end)
    if cached_data is not None:
        return cached_data

    try:
        provider = ProviderFactory()
        data = await provider.fetch_data(symbol, period, interval, start, end, user, db)
    except Exception as e:
        logger.error(f"Primary provider failed: {e}")
        logger.info(f"Falling back to Yahoo for {symbol}")
        yahoo = YahooProvider()
        data = await asyncio.to_thread(yahoo.fetch_data, symbol, period, interval, start, end)

    if not data.empty:
        data_cache.set(symbol, period, interval, data, start, end)

    return data


def validate_interval_period(interval: str, period: str) -> tuple:
    """
    Validate and adjust interval/period combinations (Provider specific constraints)
    """
    if interval == "1h" and period in ["1y", "2y", "5y"]:
        logger.warning(f"Adjusting interval from {interval} to 1d for period {period}")
        return "1d", period

    return interval, period
