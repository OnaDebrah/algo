import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from pandas import DataFrame, Series
from requests.adapters import HTTPAdapter
from tenacity import retry, stop_after_attempt, wait_exponential
from urllib3.util.retry import Retry

from ....config import ALPHA_VANTAGE_BASE_URL, CALL_DELAY, settings
from ..providers.base_provider import (
    DataProvider,
    FundamentalsProvider,
    NewsProvider,
    QuoteProvider,
)

logger = logging.getLogger(__name__)


class AlphaVantageProvider(DataProvider, QuoteProvider, FundamentalsProvider, NewsProvider):
    """
    Alpha Vantage data provider implementation.

    Provides:
    - OHLCV data (Time Series)
    - Real-time quotes (Global Quote)
    - Fundamentals (Income Statement, Balance Sheet, Cash Flow)
    - Company overview
    - News sentiment (via News API endpoint)

    API Documentation: https://www.alphavantage.co/documentation/
    """

    def __init__(self):
        self.api_key = getattr(settings, "ALPHA_VANTAGE_API_KEY", "")
        if not self.api_key:
            logger.warning("AlphaVantageProvider: ALPHA_VANTAGE_API_KEY is not set.")

        self.is_premium = getattr(settings, "ALPHA_VANTAGE_PREMIUM", False)
        self.call_delay = CALL_DELAY if not self.is_premium else 2
        self.last_call_time = 0

        self.BASE_URL = ALPHA_VANTAGE_BASE_URL

        # Configure session with retry strategy
        self.session = self._create_session()

        # Cache for API responses to reduce calls
        self._cache = {}
        self._cache_timeout = 300  # 5 minutes

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy."""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _rate_limit(self):
        """Implement rate limiting to respect API tiers."""
        current_time = time.time()
        time_since_last = current_time - self.last_call_time

        if time_since_last < self.call_delay:
            sleep_time = self.call_delay - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)

        self.last_call_time = time.time()

    def _get_cache_key(self, function: str, symbol: str, **kwargs) -> str:
        """Generate cache key for API responses."""
        key_parts = [function, symbol]
        for k, v in sorted(kwargs.items()):
            if v is not None:
                key_parts.append(f"{k}={v}")
        return ":".join(key_parts)

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get data from cache if not expired."""
        if key in self._cache:
            data, timestamp = self._cache[key]
            if time.time() - timestamp < self._cache_timeout:
                return data
            else:
                del self._cache[key]
        return None

    def _add_to_cache(self, key: str, data: Any):
        """Add data to cache."""
        self._cache[key] = (data, time.time())

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make rate-limited request to Alpha Vantage API.

        Args:
            params: API parameters

        Returns:
            JSON response as dictionary
        """
        self._rate_limit()

        # Always include API key
        params["apikey"] = self.api_key

        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Check for API error messages
            if "Error Message" in data:
                logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return {}

            if "Note" in data:  # Rate limit message
                logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                return {}

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            raise

    # ── DataProvider (OHLCV) ────────────────────────────────────────────

    def fetch_data(self, symbol: str, period: str, interval: str, start: Optional[Any] = None, end: Optional[Any] = None) -> pd.DataFrame:
        """
        Fetch OHLCV data from Alpha Vantage.

        Maps to different API functions based on interval:
        - Intraday: TIME_SERIES_INTRADAY
        - Daily: TIME_SERIES_DAILY_ADJUSTED
        - Weekly: TIME_SERIES_WEEKLY_ADJUSTED
        - Monthly: TIME_SERIES_MONTHLY_ADJUSTED
        """
        cache_key = self._get_cache_key("fetch_data", symbol, period=period, interval=interval, start=start, end=end)
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached.copy()

        try:
            # Determine which API function to use
            if interval in ["1min", "5min", "15min", "30min", "60min"]:
                df = self._fetch_intraday(symbol, interval)
            elif interval == "1d":
                df = self._fetch_daily(symbol)
            elif interval == "1wk":
                df = self._fetch_weekly(symbol)
            elif interval == "1mo":
                df = self._fetch_monthly(symbol)
            else:
                logger.warning(f"Unsupported interval {interval}, defaulting to daily")
                df = self._fetch_daily(symbol)

            if df.empty:
                return df

            # Filter by date range if provided
            if start is not None:
                start_dt = pd.to_datetime(start)
                df = df[df.index >= start_dt]

            if end is not None:
                end_dt = pd.to_datetime(end)
                df = df[df.index <= end_dt]

            # Filter by period if no start/end provided
            if start is None and end is None and period != "max":
                period_days = self._period_to_days(period)
                if period_days:
                    cutoff = datetime.now() - timedelta(days=period_days)
                    df = df[df.index >= cutoff]

            self._add_to_cache(cache_key, df)
            logger.info(f"AlphaVantageProvider: Fetched {len(df)} bars for {symbol}")

            return df

        except Exception as e:
            logger.error(f"AlphaVantageProvider: Error fetching {symbol}: {e}")
            return pd.DataFrame()

    def _period_to_days(self, period: str) -> int:
        """Convert yfinance period to days."""
        mapping = {
            "1d": 1,
            "5d": 5,
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825,
            "10y": 3650,
            "max": 0,  # No filtering
        }
        return mapping.get(period, 365)

    def _fetch_intraday(self, symbol: str, interval: str) -> DataFrame | Series:
        """Fetch intraday data."""
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol.upper(),
            "interval": interval,
            "outputsize": "full",  # 'compact' for last 100, 'full' for up to 30 days
            "adjusted": "true",
        }

        data = self._make_request(params)
        if not data:
            return pd.DataFrame()

        # Find the time series key
        time_series_key = f"Time Series ({interval})"
        if time_series_key not in data:
            logger.warning(f"No intraday data found for {symbol}")
            return pd.DataFrame()

        time_series = data[time_series_key]

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient="index")

        # Rename columns
        df = df.rename(columns={"1. open": "Open", "2. high": "High", "3. low": "Low", "4. close": "Close", "5. volume": "Volume"})

        # Convert to numeric
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col])

        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        return df[["Open", "High", "Low", "Close", "Volume"]]

    def _fetch_daily(self, symbol: str) -> DataFrame | Series:
        """Fetch daily adjusted data."""
        params = {"function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": symbol.upper(), "outputsize": "full"}

        data = self._make_request(params)
        if not data:
            return pd.DataFrame()

        if "Time Series (Daily)" not in data:
            logger.warning(f"No daily data found for {symbol}")
            return pd.DataFrame()

        time_series = data["Time Series (Daily)"]

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient="index")

        # Rename columns
        df = df.rename(
            columns={
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. adjusted close": "Adj Close",
                "6. volume": "Volume",
                "7. dividend amount": "Dividend",
                "8. split coefficient": "Split",
            }
        )

        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])

        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Use adjusted close for Close if available
        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]

        return df[["Open", "High", "Low", "Close", "Volume"]]

    def _fetch_weekly(self, symbol: str) -> DataFrame | Series:
        """Fetch weekly adjusted data."""
        params = {"function": "TIME_SERIES_WEEKLY_ADJUSTED", "symbol": symbol.upper()}

        data = self._make_request(params)
        if not data:
            return pd.DataFrame()

        if "Weekly Adjusted Time Series" not in data:
            logger.warning(f"No weekly data found for {symbol}")
            return pd.DataFrame()

        time_series = data["Weekly Adjusted Time Series"]

        df = pd.DataFrame.from_dict(time_series, orient="index")

        # Rename columns
        df = df.rename(
            columns={
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. adjusted close": "Adj Close",
                "6. volume": "Volume",
            }
        )

        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])

        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        return df[["Open", "High", "Low", "Close", "Volume"]]

    def _fetch_monthly(self, symbol: str) -> DataFrame | Series:
        """Fetch monthly adjusted data."""
        params = {"function": "TIME_SERIES_MONTHLY_ADJUSTED", "symbol": symbol.upper()}

        data = self._make_request(params)
        if not data:
            return pd.DataFrame()

        if "Monthly Adjusted Time Series" not in data:
            logger.warning(f"No monthly data found for {symbol}")
            return pd.DataFrame()

        time_series = data["Monthly Adjusted Time Series"]

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient="index")

        # Rename columns
        df = df.rename(
            columns={
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. adjusted close": "Adj Close",
                "6. volume": "Volume",
            }
        )

        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])

        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        return df[["Open", "High", "Low", "Close", "Volume"]]

    # ── QuoteProvider ───────────────────────────────────────────────────

    def get_quote(self, symbol: str) -> Dict:
        """
        Get real-time quote data using Global Quote endpoint.

        Returns:
            Dict with keys: price, change, changePercent, volume, etc.
        """
        cache_key = self._get_cache_key("get_quote", symbol)
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached.copy()

        try:
            params = {"function": "GLOBAL_QUOTE", "symbol": symbol.upper()}

            data = self._make_request(params)

            if not data or "Global Quote" not in data:
                logger.warning(f"No quote data found for {symbol}")
                return self._fallback_quote(symbol)

            quote = data["Global Quote"]

            # Extract and convert values
            price = self._safe_float(quote.get("05. price", 0))
            change = self._safe_float(quote.get("09. change", 0))
            change_pct = self._safe_float(quote.get("10. change percent", "0%").replace("%", ""))
            volume = self._safe_int(quote.get("06. volume", 0))
            previous_close = self._safe_float(quote.get("08. previous close", 0))
            open_price = self._safe_float(quote.get("02. open", 0))
            high = self._safe_float(quote.get("03. high", 0))
            low = self._safe_float(quote.get("04. low", 0))

            # Get additional info from company overview for market cap
            overview = self._get_company_overview(symbol)
            market_cap = overview.get("MarketCapitalization", 0)

            quote_data = {
                "symbol": symbol,
                "price": price,
                "change": change,
                "changePercent": change_pct,
                "volume": volume,
                "marketCap": self._safe_int(market_cap),
                "high": high,
                "low": low,
                "open": open_price,
                "previousClose": previous_close,
                "bid": 0,  # Alpha Vantage doesn't provide bid/ask in free tier
                "ask": 0,
                "bidSize": 0,
                "askSize": 0,
                "fiftyTwoWeekHigh": overview.get("52WeekHigh", 0),
                "fiftyTwoWeekLow": overview.get("52WeekLow", 0),
                "avgVolume": overview.get("AverageDailyVolume", 0),
                "timestamp": datetime.now().isoformat(),
            }

            self._add_to_cache(cache_key, quote_data)
            return quote_data

        except Exception as e:
            logger.error(f"AlphaVantageProvider: Error fetching quote for {symbol}: {e}")
            return self._fallback_quote(symbol)

    def _fallback_quote(self, symbol: str) -> Dict:
        """Return fallback quote data when API fails."""
        return {
            "symbol": symbol,
            "price": 0,
            "change": 0,
            "changePercent": 0,
            "volume": 0,
            "marketCap": 0,
            "high": 0,
            "low": 0,
            "open": 0,
            "previousClose": 0,
            "bid": 0,
            "ask": 0,
            "bidSize": 0,
            "askSize": 0,
            "fiftyTwoWeekHigh": 0,
            "fiftyTwoWeekLow": 0,
            "avgVolume": 0,
            "timestamp": datetime.now().isoformat(),
            "error": "No data available",
        }

    # ── FundamentalsProvider ────────────────────────────────────────────

    def get_ticker_info(self, symbol: str) -> Dict:
        """
        Get company overview and basic info.

        Returns:
            Dict with company name, sector, market cap, etc.
        """
        return self._get_company_overview(symbol)

    def _get_company_overview(self, symbol: str) -> Dict:
        """Fetch company overview from Alpha Vantage."""
        cache_key = self._get_cache_key("overview", symbol)
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached.copy()

        try:
            params = {"function": "OVERVIEW", "symbol": symbol.upper()}

            data = self._make_request(params)

            if not data or "Symbol" not in data:
                logger.warning(f"No overview data found for {symbol}")
                return {}

            self._add_to_cache(cache_key, data)
            return data

        except Exception as e:
            logger.error(f"Error fetching overview for {symbol}: {e}")
            return {}

    def get_financials(self, symbol: str) -> Dict:
        """
        Get comprehensive financial data.

        Returns:
            Dict with keys: info, financials (income), balance_sheet, cash_flow
        """
        cache_key = self._get_cache_key("financials", symbol)
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached.copy()

        try:
            # Fetch all financial statements
            income_stmt = self._get_income_statement(symbol)
            balance_sheet = self._get_balance_sheet(symbol)
            cash_flow = self._get_cash_flow(symbol)
            overview = self._get_company_overview(symbol)

            # Convert to DataFrames
            financials_df = self._financials_to_dataframe(income_stmt, "income")
            balance_sheet_df = self._financials_to_dataframe(balance_sheet, "balance")
            cash_flow_df = self._financials_to_dataframe(cash_flow, "cash_flow")

            result = {
                "info": overview,
                "financials": financials_df,
                "balance_sheet": balance_sheet_df,
                "cash_flow": cash_flow_df,
                "recommendations": pd.DataFrame(),  # Alpha Vantage doesn't provide recommendations
            }

            self._add_to_cache(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"Error fetching financials for {symbol}: {e}")
            return {
                "info": {},
                "financials": pd.DataFrame(),
                "balance_sheet": pd.DataFrame(),
                "cash_flow": pd.DataFrame(),
                "recommendations": pd.DataFrame(),
            }

    def _get_income_statement(self, symbol: str) -> Dict:
        """Fetch income statement."""
        params = {"function": "INCOME_STATEMENT", "symbol": symbol.upper()}
        return self._make_request(params)

    def _get_balance_sheet(self, symbol: str) -> Dict:
        """Fetch balance sheet."""
        params = {"function": "BALANCE_SHEET", "symbol": symbol.upper()}
        return self._make_request(params)

    def _get_cash_flow(self, symbol: str) -> Dict:
        """Fetch cash flow statement."""
        params = {"function": "CASH_FLOW", "symbol": symbol.upper()}
        return self._make_request(params)

    def _financials_to_dataframe(self, data: Dict, stmt_type: str) -> pd.DataFrame:
        """Convert financial statement API response to DataFrame."""
        if not data:
            return pd.DataFrame()

        # Find the annual reports key
        if "annualReports" not in data:
            return pd.DataFrame()

        reports = data["annualReports"]
        if not reports:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(reports)

        # Set fiscal date as index
        if "fiscalDateEnding" in df.columns:
            df["fiscalDateEnding"] = pd.to_datetime(df["fiscalDateEnding"])
            df = df.set_index("fiscalDateEnding")
            df = df.sort_index(ascending=False)

        # Convert all columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    # ── NewsProvider ────────────────────────────────────────────────────

    def get_news(self, symbol: str, limit: int = 10) -> List[Dict]:
        """
        Get news and sentiment for a symbol.

        Alpha Vantage has a News & Sentiment endpoint that returns
        articles with sentiment scores.
        """
        cache_key = self._get_cache_key("news", symbol, limit=limit)
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached.copy()

        try:
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": symbol.upper(),
                "limit": min(limit, 50),  # API limit is 50
            }

            data = self._make_request(params)

            if not data or "feed" not in data:
                logger.warning(f"No news found for {symbol}")
                return []

            feed = data["feed"]
            news_items = []

            for item in feed[:limit]:
                # Extract sentiment if available
                sentiment_score = 0
                sentiment_label = "neutral"

                if "ticker_sentiment" in item:
                    for ticker_sent in item["ticker_sentiment"]:
                        if ticker_sent["ticker"] == symbol.upper():
                            sentiment_score = self._safe_float(ticker_sent.get("ticker_sentiment_score", 0))
                            sentiment_label = ticker_sent.get("ticker_sentiment_label", "neutral")
                            break

                news_item = {
                    "title": item.get("title", ""),
                    "summary": item.get("summary", ""),
                    "url": item.get("url", ""),
                    "source": item.get("source", ""),
                    "published_at": item.get("time_published", ""),
                    "authors": item.get("authors", []),
                    "sentiment_score": sentiment_score,
                    "sentiment_label": sentiment_label,
                    "topics": [topic.get("topic", "") for topic in item.get("topics", [])],
                }
                news_items.append(news_item)

            self._add_to_cache(cache_key, news_items)
            return news_items

        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []

    # ── Helper Methods ──────────────────────────────────────────────────

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Safely convert value to float."""
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        """Safely convert value to int."""
        try:
            return int(float(value))  # Handle string numbers like "123.45"
        except (ValueError, TypeError):
            return default

    def get_available_functions(self) -> List[str]:
        """Return list of available API functions for this symbol."""
        return [
            "TIME_SERIES_INTRADAY",
            "TIME_SERIES_DAILY_ADJUSTED",
            "TIME_SERIES_WEEKLY_ADJUSTED",
            "TIME_SERIES_MONTHLY_ADJUSTED",
            "GLOBAL_QUOTE",
            "OVERVIEW",
            "INCOME_STATEMENT",
            "BALANCE_SHEET",
            "CASH_FLOW",
            "NEWS_SENTIMENT",
        ]

    def get_api_usage(self) -> Dict:
        """Get API usage statistics."""
        return {
            "calls_made": getattr(self, "_call_count", 0),
            "cache_size": len(self._cache),
            "rate_limit": f"{self.call_delay}s between calls",
            "tier": "premium" if self.is_premium else "free",
        }
