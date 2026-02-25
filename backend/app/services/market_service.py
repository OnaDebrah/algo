import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from ..core.data.market_data_cache import MarketDataCache
from ..core.data.providers.providers import ProviderFactory
from ..utils.helpers import SECTOR_MAP

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Supported data sources"""

    YAHOO = "yahoo"
    INTERACTIVE_BROKERS = "ib"
    ALPACA = "alpaca"
    POLYGON = "polygon"


class MarketService:
    """
    Market data service

    Features:
    - Multiple data source support via ProviderFactory
    - Intelligent caching
    - Retry logic with exponential backoff
    - Comprehensive error handling
    - Rich data types (quotes, historical, options, fundamentals, news)
    """

    def __init__(self, data_source: DataSource = DataSource.YAHOO, cache_ttl: int = 60, max_retries: int = 3):
        self.data_source = data_source
        self.cache = MarketDataCache(ttl_seconds=cache_ttl)
        self.max_retries = max_retries
        self._provider = ProviderFactory()

        logger.info(f"MarketService initialized with {data_source.value} data source")

    async def _retry_with_backoff(self, coro_func, *args, **kwargs):
        """Execute an async function with exponential backoff retry logic"""
        for attempt in range(self.max_retries):
            try:
                return await coro_func(*args, **kwargs)
            except Exception as e:
                # Special handling for 429 Too Many Requests
                if "429" in str(e) or "Too Many Requests" in str(e):
                    wait_time = (2**attempt) + 1
                else:
                    wait_time = 2**attempt

                if attempt == self.max_retries - 1:
                    logger.error(f"Failed after {self.max_retries} attempts: {str(e)}")
                    raise

                logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)}")
                await asyncio.sleep(wait_time)

    # ============================================================
    # QUOTE DATA
    # ============================================================

    async def get_quote(self, db: AsyncSession, symbol: str, use_cache: bool = True) -> Dict:
        """Get real-time quote for a symbol"""
        cache_key = f"quote_{symbol}"

        if use_cache:
            cached = await self.cache.get(db, cache_key)
            if cached:
                return cached

        try:
            quote = await self._retry_with_backoff(self._provider.get_quote, symbol)
            await self.cache.set(db, cache_key, quote, "quote")
            return quote
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {str(e)}")
            raise

    async def get_quotes(self, db: AsyncSession, symbols: List[str], use_cache: bool = True) -> List[Dict]:
        """Get quotes for multiple symbols concurrently"""
        tasks = [self.get_quote(db, symbol, use_cache) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        quotes = []
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch quote for {symbol}: {str(result)}")
            else:
                quotes.append(result)

        return quotes

    # ============================================================
    # HISTORICAL DATA
    # ============================================================

    async def get_historical_data(
        self,
        db: AsyncSession,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_cache: bool = False,
    ) -> Dict:
        """Get historical OHLCV data"""
        cache_key = f"hist_{symbol}_{period}_{interval}_{start}_{end}"

        if use_cache:
            cached = await self.cache.get(db, cache_key)
            if cached:
                return cached

        try:
            hist = await self._retry_with_backoff(self._provider.fetch_data, symbol, period, interval, start, end)

            if hist.empty:
                raise ValueError(f"No historical data available for {symbol}")

            data = {
                "symbol": symbol,
                "period": period,
                "interval": interval,
                "data": hist.reset_index().to_dict(orient="records"),
                "dataframe": hist,
            }

            if use_cache:
                await self.cache.set(db, cache_key, data, "historical")
            return data
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            raise

    # ============================================================
    # OPTIONS DATA
    # ============================================================

    async def get_option_chain(self, symbol: str, expiration: Optional[str] = None) -> Dict:
        """Get option chain data"""
        try:
            if expiration is None:
                # Get available expirations first
                expirations = await self._retry_with_backoff(self._provider.get_option_expirations, symbol)
                if not expirations:
                    raise ValueError(f"No options available for {symbol}")
                expiration = expirations[0]

            chain = await self._retry_with_backoff(self._provider.get_option_chain, symbol, expiration)

            return {
                "symbol": symbol,
                "expiration": expiration,
                "available_expirations": chain.get("expirations", [expiration]),
                "calls": chain["calls"].to_dict(orient="records") if hasattr(chain["calls"], "to_dict") else chain["calls"],
                "puts": chain["puts"].to_dict(orient="records") if hasattr(chain["puts"], "to_dict") else chain["puts"],
                "underlying_price": chain.get("underlying_price", 0),
            }

        except Exception as e:
            logger.error(f"Error fetching option chain for {symbol}: {str(e)}")
            raise

    # ============================================================
    # FUNDAMENTAL DATA
    # ============================================================

    async def get_fundamentals(self, db: AsyncSession, symbol: str) -> Dict:
        """Get fundamental data for a symbol"""
        cache_key = f"fundamentals_{symbol}"
        cached = await self.cache.get(db, cache_key)
        if cached:
            return cached

        try:
            info = await self._retry_with_backoff(self._provider.get_ticker_info, symbol)

            fundamentals = {
                "symbol": symbol,
                "company_name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0),
                "enterprise_value": info.get("enterpriseValue", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "forward_pe": info.get("forwardPE", 0),
                "peg_ratio": info.get("pegRatio", 0),
                "price_to_book": info.get("priceToBook", 0),
                "price_to_sales": info.get("priceToSalesTrailing12Months", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "profit_margin": info.get("profitMargins", 0),
                "operating_margin": info.get("operatingMargins", 0),
                "roe": info.get("returnOnEquity", 0),
                "roa": info.get("returnOnAssets", 0),
                "revenue": info.get("totalRevenue", 0),
                "revenue_growth": info.get("revenueGrowth", 0),
                "earnings_growth": info.get("earningsGrowth", 0),
                "debt_to_equity": info.get("debtToEquity", 0),
                "current_ratio": info.get("currentRatio", 0),
                "beta": info.get("beta", 0),
                "shares_outstanding": info.get("sharesOutstanding", 0),
                "float_shares": info.get("floatShares", 0),
                "timestamp": datetime.now().isoformat(),
            }

            await self.cache.set(db, cache_key, fundamentals, "fundamentals")
            return fundamentals
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {str(e)}")
            raise

    # ============================================================
    # NEWS DATA
    # ============================================================

    async def get_news(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Get news for a symbol"""
        try:
            news = await self._retry_with_backoff(self._provider.get_news, symbol, limit)

            return [
                {
                    "title": item.get("title", ""),
                    "publisher": item.get("publisher", ""),
                    "link": item.get("link", ""),
                    "published": datetime.fromtimestamp(item.get("providerPublishTime", 0)).isoformat() if item.get("providerPublishTime") else "",
                    "type": item.get("type", ""),
                    "thumbnail": item.get("thumbnail", {}).get("resolutions", [{}])[0].get("url", "") if item.get("thumbnail") else "",
                }
                for item in news
            ]
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {str(e)}")
            return []

    # ============================================================
    # SEARCH & DISCOVERY
    # ============================================================

    async def search_symbols(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for symbols"""
        try:
            info = await self._provider.get_ticker_info(query.upper())

            if info.get("symbol"):
                return [
                    {
                        "symbol": info.get("symbol", query.upper()),
                        "name": info.get("longName", query),
                        "type": info.get("quoteType", "EQUITY"),
                        "exchange": info.get("exchange", ""),
                        "currency": info.get("currency", "USD"),
                        "market_cap": info.get("marketCap", 0),
                    }
                ]
        except Exception as e:
            logger.error(f"Error searching symbols for '{query}': {str(e)}")

        return []

    # ============================================================
    # UTILITY METHODS
    # ============================================================

    async def clear_cache(self, db: AsyncSession, data_type: Optional[str] = None):
        """Clear cached data"""
        await self.cache.clear(db, data_type)
        logger.info(f"Market data cache cleared: {data_type or 'all'}")

    async def cleanup_cache(self, db: AsyncSession):
        """Cleanup expired cache entries"""
        await self.cache.cleanup_expired(db=db)

    async def get_cache_stats(self, db: AsyncSession) -> Dict:
        """Get cache statistics"""
        return await self.cache.get_stats(db)

    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol exists"""
        try:
            info = await self._provider.get_ticker_info(symbol)
            return bool(info.get("symbol") or info.get("longName"))
        except Exception:
            return False

    async def get_market_status(self, db: AsyncSession) -> Dict:
        """Get market status (open/closed)"""
        try:
            quote = await self.get_quote(db, "SPY")
            return {
                "is_open": True,
                "last_update": quote["timestamp"],
                "market": "US",
            }
        except Exception as e:
            logger.error(f"Failed to get market status {e}")
            return {"is_open": False, "last_update": datetime.now().isoformat(), "market": "US"}

    async def get_sector(self, symbol: str) -> str:
        """Get sector for a symbol"""
        if symbol in SECTOR_MAP:
            return SECTOR_MAP[symbol]

        try:
            info = await self._provider.get_ticker_info(symbol)
            return info.get("sector", "Unknown")
        except Exception:
            return "Unknown"


# Singleton instance for easy import
_market_service_instance = None


def get_market_service(data_source: DataSource = DataSource.YAHOO, cache_ttl: int = 60) -> MarketService:
    """Get or create singleton MarketService instance"""
    global _market_service_instance

    if _market_service_instance is None:
        _market_service_instance = MarketService(data_source=data_source, cache_ttl=cache_ttl)

    return _market_service_instance
