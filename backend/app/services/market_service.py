"""
Market Data Service

Comprehensive market data service with:
- Yahoo Finance as default data source
- PostgreSQL caching for performance
- Retry logic for reliability
- Error handling and fallbacks
- Support for multiple data sources (extensible for IB, Alpaca, etc.)
- Comprehensive data types (quotes, historical, options, fundamentals)
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import yfinance as yf
from psycopg2.extras import RealDictCursor

from backend.app.core import DatabaseManager

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Supported data sources"""

    YAHOO = "yahoo"
    INTERACTIVE_BROKERS = "ib"  # Future
    ALPACA = "alpaca"  # Future
    POLYGON = "polygon"  # Future


class MarketDataCache:
    """
    Hybrid caching system with in-memory (hot) and database (cold) tiers

    Strategy:
    - In-memory: Fast access for quotes and recent data (60s TTL)
    - Database: Persistent storage for historical and fundamental data (longer TTL)
    """

    def __init__(self, ttl_seconds: int = 60, use_db: bool = True, db_manager: DatabaseManager = None):
        self.memory_cache: Dict[str, tuple[Any, float]] = {}
        self.memory_ttl = ttl_seconds
        self.db = db_manager or DatabaseManager()
        self.use_db = use_db

        if self.use_db:
            self._init_db_cache()

    def _init_db_cache(self):
        """Initialize database cache table"""
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute("""
                           CREATE TABLE IF NOT EXISTS market_data_cache (
                                                                            cache_key TEXT PRIMARY KEY,
                                                                            data_type TEXT NOT NULL,
                                                                            data_json TEXT NOT NULL,
                                                                            created_at DOUBLE PRECISION NOT NULL,
                                                                            expires_at DOUBLE PRECISION NOT NULL,
                                                                            access_count INTEGER DEFAULT 0,
                                                                            last_accessed DOUBLE PRECISION
                           )
                           """)

            # Create index for faster expiration queries
            cursor.execute("""
                           CREATE INDEX IF NOT EXISTS idx_market_cache_expires_at
                               ON market_data_cache(expires_at)
                           """)

            conn.commit()

            logger.info("Database cache initialized")
        except Exception as e:
            conn.rollback()
            logger.warning(f"Database cache initialization failed: {e}. Using memory-only cache.")
            self.use_db = False
            raise
        finally:
            cursor.close()
            self.db.return_connection(conn)

    def get(self, key: str, data_type: str = "quote") -> Optional[Any]:
        """
        Get cached value from memory or database

        Args:
            key: Cache key
            data_type: Type of data (quote, historical, fundamentals, etc.)
        """
        # Try memory cache first (fastest)
        if key in self.memory_cache:
            value, timestamp = self.memory_cache[key]
            if time.time() - timestamp < self.memory_ttl:
                logger.debug(f"Memory cache hit: {key}")
                return value
            else:
                del self.memory_cache[key]

        # Try database cache (persistent)
        if self.use_db:
            conn = self.db.get_connection()
            try:
                cursor = conn.cursor(cursor_factory=RealDictCursor)

                cursor.execute(
                    """
                    SELECT data_json, expires_at, access_count
                    FROM market_data_cache
                    WHERE cache_key = %s AND expires_at > %s
                    """,
                    (key, time.time()),
                )

                row = cursor.fetchone()

                if row:
                    data_json = row["data_json"]
                    access_count = row["access_count"]
                    value = json.loads(data_json)

                    # Update access statistics
                    cursor.execute(
                        """
                        UPDATE market_data_cache
                        SET access_count = %s, last_accessed = %s
                        WHERE cache_key = %s
                        """,
                        (access_count + 1, time.time(), key),
                    )

                    conn.commit()
                    conn.close()

                    # Promote to memory cache for faster subsequent access
                    self.memory_cache[key] = (value, time.time())

                    logger.debug(f"Database cache hit: {key}")
                    return value

            except Exception as e:
                conn.rollback()
                logger.error(f"Database cache read error: {e}. Using memory-only cache.")
                raise
            finally:
                cursor.close()
                self.db.return_connection(conn)

        return None

    def set(self, key: str, value: Any, data_type: str = "quote", ttl_override: Optional[int] = None):
        """
        Set cache value in both memory and database

        Args:
            key: Cache key
            value: Data to cache
            data_type: Type of data (determines TTL strategy)
            ttl_override: Override default TTL in seconds
        """
        current_time = time.time()

        # Always set in memory cache for fast access
        self.memory_cache[key] = (value, current_time)

        # Set in database for persistence (with longer TTL for certain data types)
        if self.use_db:
            conn = self.db.get_connection()
            try:
                # Determine TTL based on data type
                if ttl_override:
                    ttl = ttl_override
                elif data_type == "quote":
                    ttl = 60  # 1 minute for quotes
                elif data_type == "historical":
                    ttl = 3600  # 1 hour for historical data
                elif data_type == "fundamentals":
                    ttl = 86400  # 24 hours for fundamentals
                elif data_type == "options":
                    ttl = 300  # 5 minutes for options
                else:
                    ttl = self.memory_ttl

                expires_at = current_time + ttl

                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO market_data_cache
                    (cache_key, data_type, data_json, created_at, expires_at, access_count, last_accessed)
                    VALUES (%s, %s, %s, %s, %s, 0, %s)
                        ON CONFLICT (cache_key)
                    DO UPDATE SET
                        data_json = EXCLUDED.data_json,
                                                   created_at = EXCLUDED.created_at,
                                                   expires_at = EXCLUDED.expires_at,
                                                   access_count = 0,
                                                   last_accessed = EXCLUDED.last_accessed
                    """,
                    (key, data_type, json.dumps(value), current_time, expires_at, current_time),
                )

                conn.commit()

                logger.debug(f"Cached to database: {key} (TTL: {ttl}s)")

            except Exception as e:
                conn.rollback()
                logger.error(f"Database cache write failed: {e}. Using memory-only cache.")
                raise
            finally:
                cursor.close()
                self.db.return_connection(conn)

    def clear(self, data_type: Optional[str] = None):
        """
        Clear cached data

        Args:
            data_type: If specified, only clear this type. Otherwise clear all.
        """
        # Clear memory cache
        if data_type:
            keys_to_remove = [k for k in self.memory_cache.keys() if k.startswith(data_type)]
            for key in keys_to_remove:
                del self.memory_cache[key]
        else:
            self.memory_cache.clear()

        # Clear database cache
        if self.use_db:
            conn = self.db.get_connection()
            try:
                cursor = conn.cursor()

                if data_type:
                    cursor.execute("DELETE FROM market_data_cache WHERE data_type = %s", (data_type,))
                else:
                    cursor.execute("DELETE FROM market_data_cache")

                conn.commit()

                logger.info(f"Database cache cleared: {data_type or 'all'}")

            except Exception as e:
                conn.rollback()
                logger.error(f"Database cache clear error: {e}. ")
                raise
            finally:
                cursor.close()
                self.db.return_connection(conn)

    def cleanup_expired(self):
        """Remove expired entries from database"""
        if self.use_db:
            conn = self.db.get_connection()
            try:
                cursor = conn.cursor()

                cursor.execute("DELETE FROM market_data_cache WHERE expires_at < %s", (time.time(),))
                deleted = cursor.rowcount

                conn.commit()

                if deleted > 0:
                    logger.info(f"Cleaned up {deleted} expired cache entries")

            except Exception as e:
                conn.rollback()
                logger.error(f"Cache cleanup error: {e}.")
                raise
            finally:
                cursor.close()
                self.db.return_connection(conn)

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        stats = {"memory_entries": len(self.memory_cache), "database_enabled": self.use_db}

        if self.use_db:
            conn = self.db.get_connection()
            try:
                cursor = conn.cursor(cursor_factory=RealDictCursor)

                cursor.execute(
                    """
                    SELECT COUNT(*) as count, COALESCE(SUM(access_count), 0) as total_accesses
                    FROM market_data_cache
                    WHERE expires_at > %s
                    """,
                    (time.time(),),
                )
                row = cursor.fetchone()

                stats["database_entries"] = row["count"] if row else 0
                stats["total_accesses"] = row["total_accesses"] if row else 0

                # Get breakdown by data type
                cursor.execute(
                    """
                    SELECT data_type, COUNT(*) as count, AVG(access_count) as avg_accesses
                    FROM market_data_cache
                    WHERE expires_at > %s
                    GROUP BY data_type
                    """,
                    (time.time(),),
                )

                stats["by_type"] = {
                    row["data_type"]: {"count": row["count"], "avg_accesses": float(row["avg_accesses"]) if row["avg_accesses"] else 0}
                    for row in cursor.fetchall()
                }

                conn.close()
            except Exception as e:
                logger.error(f"Error getting cache stats: {e}.")
                raise
            finally:
                cursor.close()
                self.db.return_connection(conn)

        return stats


class MarketService:
    """
    Market data service

    Features:
    - Multiple data source support (Yahoo Finance default)
    - Intelligent caching
    - Retry logic with exponential backoff
    - Comprehensive error handling
    - Rich data types (quotes, historical, options, fundamentals, news)
    """

    def __init__(self, data_source: DataSource = DataSource.YAHOO, cache_ttl: int = 60, max_retries: int = 3):
        self.data_source = data_source
        self.cache = MarketDataCache(ttl_seconds=cache_ttl)
        self.max_retries = max_retries

        logger.info(f"MarketService initialized with {data_source.value} data source")

    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry logic"""
        for attempt in range(self.max_retries):
            try:
                return await asyncio.to_thread(func, *args, **kwargs)
            except Exception as e:
                # Special handling for 429 Too Many Requests
                if "429" in str(e) or "Too Many Requests" in str(e):
                    wait_time = (2**attempt) + 1  # Standard backoff + 1s jitter
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

    async def get_quote(self, symbol: str, use_cache: bool = True) -> Dict:
        """
        Get real-time quote for a symbol

        Args:
            symbol: Stock symbol
            use_cache: Whether to use cached data

        Returns:
            Dictionary with quote data
        """
        cache_key = f"quote_{symbol}"

        if use_cache:
            cached = self.cache.get(cache_key, "quote")
            if cached:
                return cached

        def _fetch():
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Get fast info for real-time data
            try:
                fast_info = ticker.fast_info
                current_price = fast_info.get("last_price", info.get("currentPrice", 0))
                previous_close = fast_info.get("previous_close", info.get("previousClose", 0))
            except Exception:
                current_price = info.get("currentPrice", 0)
                previous_close = info.get("previousClose", 0)

            change = current_price - previous_close if current_price and previous_close else 0
            change_pct = (change / previous_close * 100) if previous_close else 0

            quote = {
                "symbol": symbol,
                "price": current_price,
                "change": change,
                "changePct": change_pct,
                "volume": info.get("volume", 0),
                "marketCap": info.get("marketCap", 0),
                "high": info.get("dayHigh", 0),
                "low": info.get("dayLow", 0),
                "open": info.get("open", 0),
                "previousClose": previous_close,
                "bid": info.get("bid", 0),
                "ask": info.get("ask", 0),
                "bidSize": info.get("bidSize", 0),
                "askSize": info.get("askSize", 0),
                "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh", 0),
                "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow", 0),
                "avgVolume": info.get("averageVolume", 0),
                "timestamp": datetime.now().isoformat(),
            }

            return quote

        try:
            quote = await self._retry_with_backoff(_fetch)
            self.cache.set(cache_key, quote, "quote")
            return quote
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {str(e)}")
            raise

    async def get_quotes(self, symbols: List[str], use_cache: bool = True) -> List[Dict]:
        """
        Get quotes for multiple symbols concurrently

        Args:
            symbols: List of stock symbols
            use_cache: Whether to use cached data

        Returns:
            List of quote dictionaries
        """
        tasks = [self.get_quote(symbol, use_cache) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and log them
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
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_cache: bool = False,  # Historical data changes less frequently
    ) -> Dict:
        """
        Get historical OHLCV data

        Args:
            symbol: Stock symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data

        Returns:
            Dictionary with historical data
        """
        cache_key = f"hist_{symbol}_{period}_{interval}_{start}_{end}"

        if use_cache:
            cached = self.cache.get(cache_key, "historical")
            if cached:
                return cached

        def _fetch():
            ticker = yf.Ticker(symbol)

            if start and end:
                hist = ticker.history(start=start, end=end, interval=interval)
            else:
                hist = ticker.history(period=period, interval=interval)

            if hist.empty:
                raise ValueError(f"No historical data available for {symbol}")

            return {
                "symbol": symbol,
                "period": period,
                "interval": interval,
                "data": hist.reset_index().to_dict(orient="records"),
                "dataframe": hist,  # Include raw dataframe for internal use
            }

        try:
            data = await self._retry_with_backoff(_fetch)
            if use_cache:
                self.cache.set(cache_key, data, "historical")
            return data
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            raise

    # ============================================================
    # OPTIONS DATA
    # ============================================================

    async def get_option_chain(self, symbol: str, expiration: Optional[str] = None) -> Dict:
        """
        Get option chain data

        Args:
            symbol: Stock symbol
            expiration: Specific expiration date (YYYY-MM-DD), or None for nearest

        Returns:
            Dictionary with option chain data
        """

        def _fetch():
            ticker = yf.Ticker(symbol)

            # Get available expirations
            expirations = ticker.options
            if not expirations:
                raise ValueError(f"No options available for {symbol}")

            # Use specified expiration or nearest
            exp_date = expiration if expiration and expiration in expirations else expirations[0]

            # Get option chain
            opt_chain = ticker.option_chain(exp_date)

            return {
                "symbol": symbol,
                "expiration": exp_date,
                "available_expirations": list(expirations),
                "calls": opt_chain.calls.to_dict(orient="records"),
                "puts": opt_chain.puts.to_dict(orient="records"),
                "underlying_price": ticker.info.get("currentPrice", 0),
            }

        try:
            return await self._retry_with_backoff(_fetch)
        except Exception as e:
            logger.error(f"Error fetching option chain for {symbol}: {str(e)}")
            raise

    # ============================================================
    # FUNDAMENTAL DATA
    # ============================================================

    async def get_fundamentals(self, symbol: str) -> Dict:
        """
        Get fundamental data for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with fundamental data
        """
        cache_key = f"fundamentals_{symbol}"
        cached = self.cache.get(cache_key, "fundamentals")
        if cached:
            return cached

        def _fetch():
            ticker = yf.Ticker(symbol)
            info = ticker.info

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

            return fundamentals

        try:
            data = await self._retry_with_backoff(_fetch)
            self.cache.set(cache_key, data, "fundamentals")
            return data
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {str(e)}")
            raise

    # ============================================================
    # NEWS DATA
    # ============================================================

    async def get_news(self, symbol: str, limit: int = 10) -> List[Dict]:
        """
        Get news for a symbol

        Args:
            symbol: Stock symbol
            limit: Maximum number of news items

        Returns:
            List of news dictionaries
        """

        def _fetch():
            ticker = yf.Ticker(symbol)
            news = ticker.news

            return [
                {
                    "title": item.get("title", ""),
                    "publisher": item.get("publisher", ""),
                    "link": item.get("link", ""),
                    "published": datetime.fromtimestamp(item.get("providerPublishTime", 0)).isoformat(),
                    "type": item.get("type", ""),
                    "thumbnail": item.get("thumbnail", {}).get("resolutions", [{}])[0].get("url", "") if item.get("thumbnail") else "",
                }
                for item in news[:limit]
            ]

        try:
            return await self._retry_with_backoff(_fetch)
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {str(e)}")
            return []

    # ============================================================
    # SEARCH & DISCOVERY
    # ============================================================

    async def search_symbols(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for symbols

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching symbols
        """

        def _search():
            # Try exact match first
            try:
                ticker = yf.Ticker(query.upper())
                info = ticker.info

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
            except Exception:
                pass

            return []

        try:
            return await asyncio.to_thread(_search)
        except Exception as e:
            logger.error(f"Error searching symbols for '{query}': {str(e)}")
            return []

    # ============================================================
    # UTILITY METHODS
    # ============================================================

    def clear_cache(self, data_type: Optional[str] = None):
        """Clear cached data"""
        self.cache.clear(data_type)
        logger.info(f"Market data cache cleared: {data_type or 'all'}")

    def cleanup_cache(self):
        """Cleanup expired cache entries"""
        self.cache.cleanup_expired()

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return self.cache.get_stats()

    async def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol exists

        Args:
            symbol: Stock symbol

        Returns:
            True if symbol is valid
        """
        try:
            ticker = yf.Ticker(symbol)
            info = await asyncio.to_thread(lambda: ticker.info)
            return bool(info.get("symbol") or info.get("longName"))
        except Exception:
            return False

    async def get_market_status(self) -> Dict:
        """
        Get market status (open/closed)

        Returns:
            Dictionary with market status
        """
        # Use SPY as proxy for US market
        try:
            quote = await self.get_quote("SPY")
            return {
                "is_open": True,  # Simplified - would need more logic for actual market hours
                "last_update": quote["timestamp"],
                "market": "US",
            }
        except Exception:
            return {"is_open": False, "last_update": datetime.now().isoformat(), "market": "US"}


# Singleton instance for easy import
_market_service_instance = None


def get_market_service(data_source: DataSource = DataSource.YAHOO, cache_ttl: int = 60) -> MarketService:
    """Get or create singleton MarketService instance"""
    global _market_service_instance

    if _market_service_instance is None:
        _market_service_instance = MarketService(data_source=data_source, cache_ttl=cache_ttl)

    return _market_service_instance
