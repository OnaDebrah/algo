import asyncio
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.data.providers.providers import ProviderFactory
from ..core.data.providers.yahoo_provider import YahooProvider
from ..core.data_cache import data_cache
from ..models import User

logger = logging.getLogger(__name__)

_factory = ProviderFactory()


# ── OHLCV (existing) ───────────────────────────────────────────────────


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
        data = await _factory.fetch_data(symbol, period, interval, start, end, user, db)
    except Exception as e:
        logger.error(f"Primary provider failed: {e}")
        logger.info(f"Falling back to Yahoo for {symbol}")
        yahoo = YahooProvider()
        data = await asyncio.to_thread(yahoo.fetch_data, symbol, period, interval, start, end)

    if not data.empty:
        data_cache.set(symbol, period, interval, data, start, end)

    return data


# ── Convenience wrappers ────────────────────────────────────────────────
# Thin async functions so consumers can import from data_fetcher
# instead of instantiating ProviderFactory themselves.


async def fetch_quote(symbol: str, user: Optional[User] = None, db: Optional[AsyncSession] = None) -> Dict:
    """Get real-time quote data for a symbol."""
    return await _factory.get_quote(symbol, user, db)


async def fetch_option_expirations(symbol: str, user: Optional[User] = None, db: Optional[AsyncSession] = None) -> List[str]:
    """Get available option expiration dates."""
    return await _factory.get_option_expirations(symbol, user, db)


async def fetch_option_chain(symbol: str, expiration: str, user: Optional[User] = None, db: Optional[AsyncSession] = None) -> Dict:
    """Fetch option chain for a specific expiration."""
    return await _factory.get_option_chain(symbol, expiration, user, db)


async def fetch_ticker_info(symbol: str, user: Optional[User] = None, db: Optional[AsyncSession] = None) -> Dict:
    """Get ticker metadata / info dict."""
    return await _factory.get_ticker_info(symbol, user, db)


async def fetch_financials(symbol: str, user: Optional[User] = None, db: Optional[AsyncSession] = None) -> Dict:
    """Get comprehensive financial data (statements + info)."""
    return await _factory.get_financials(symbol, user, db)


async def fetch_news(symbol: str, limit: int = 10, user: Optional[User] = None, db: Optional[AsyncSession] = None) -> List[Dict]:
    """Get recent news items."""
    return await _factory.get_news(symbol, limit, user, db)


async def fetch_recommendations(symbol: str, user: Optional[User] = None, db: Optional[AsyncSession] = None):
    """Get analyst recommendations."""
    return await _factory.get_recommendations(symbol, user, db)


# ── Utility ─────────────────────────────────────────────────────────────


def validate_interval_period(interval: str, period: str) -> tuple:
    """
    Validate and adjust interval/period combinations (Provider specific constraints)
    """
    if interval == "1h" and period in ["1y", "2y", "5y"]:
        logger.warning(f"Adjusting interval from {interval} to 1d for period {period}")
        return "1d", period

    return interval, period
