import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from ....config import settings
from ..providers.base_provider import (
    DataProvider,
    FundamentalsProvider,
    NewsProvider,
    OptionsDataProvider,
    QuoteProvider,
    RecommendationsProvider,
)

logger = logging.getLogger(__name__)


class YahooProvider(
    DataProvider,
    QuoteProvider,
    OptionsDataProvider,
    FundamentalsProvider,
    NewsProvider,
    RecommendationsProvider,
):
    """
    Yahoo Finance data provider — the ONLY file that imports yfinance.
    Implements all provider interfaces so it can serve as the universal fallback.
    """

    def __init__(self):
        self.user_agent = getattr(settings, "USER_AGENT", "Mozilla/5.0")

    # ── DataProvider (OHLCV) ────────────────────────────────────────────

    def fetch_data(self, symbol: str, period: str, interval: str, start: Optional[Any] = None, end: Optional[Any] = None) -> pd.DataFrame:
        try:
            import yfinance as yf

            # Sanitize crypto symbols
            if len(symbol) <= 5 and symbol.isalpha() and symbol in ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE"]:
                logger.debug(f"YahooProvider: Auto-correcting {symbol} to {symbol}-USD")
                symbol = f"{symbol}-USD"

            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval, start=start, end=end)

            if data.empty:
                logger.warning(f"YahooProvider: Ticker method failed for {symbol}, trying download method")
                data = yf.download(symbol, start=start, end=end, period=period, interval=interval, progress=False)

            if not data.empty:
                logger.info(f"YahooProvider: Fetched {len(data)} bars for {symbol}")

            return data

        except Exception as e:
            logger.error(f"YahooProvider: Error fetching {symbol}: {e}")
            return pd.DataFrame()

    # ── QuoteProvider ───────────────────────────────────────────────────

    def get_quote(self, symbol: str) -> Dict:
        """Get real-time quote data for a symbol."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Get fast info for real-time data
            try:
                fast_info = ticker.fast_info
                current_price = fast_info.last_price or info.get("currentPrice") or 0
                previous_close = fast_info.previous_close or info.get("previousClose") or 0
            except Exception as e:
                logger.error(f"Failed to get quote for {symbol}: {e}")
                current_price = info.get("currentPrice", 0)
                previous_close = info.get("previousClose", 0)

            change = current_price - previous_close if current_price and previous_close else 0
            change_pct = ((change / previous_close) * 100) if previous_close else 0

            return {
                "symbol": symbol,
                "price": current_price,
                "change": change,
                "changePercent": change_pct,
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

        except Exception as e:
            logger.error(f"YahooProvider: Error fetching quote for {symbol}: {e}")
            return {"symbol": symbol, "price": 0, "error": str(e)}

    # ── OptionsDataProvider ─────────────────────────────────────────────

    def get_option_expirations(self, symbol: str) -> List[str]:
        """Return available option expiration dates."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            return list(ticker.options) if ticker.options else []

        except Exception as e:
            logger.error(f"YahooProvider: Error fetching option expirations for {symbol}: {e}")
            return []

    def get_option_chain(self, symbol: str, expiration: str) -> Dict:
        """Fetch option chain for a specific expiration."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            opt = ticker.option_chain(expiration)
            expirations = list(ticker.options) if ticker.options else [expiration]
            underlying_price = ticker.info.get("currentPrice", 0)

            return {
                "calls": opt.calls,
                "puts": opt.puts,
                "expirations": expirations,
                "underlying_price": underlying_price,
            }

        except Exception as e:
            logger.error(f"YahooProvider: Error fetching option chain for {symbol} ({expiration}): {e}")
            return {
                "calls": pd.DataFrame(),
                "puts": pd.DataFrame(),
                "expirations": [],
                "underlying_price": 0,
            }

    # ── FundamentalsProvider ────────────────────────────────────────────

    def get_ticker_info(self, symbol: str) -> Dict:
        """Return ticker metadata / info dict."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            return ticker.info or {}

        except Exception as e:
            logger.error(f"YahooProvider: Error fetching ticker info for {symbol}: {e}")
            return {}

    def get_financials(self, symbol: str) -> Dict:
        """Return comprehensive financial data (statements + info)."""
        try:
            import yfinance as yf

            stock = yf.Ticker(symbol)
            return {
                "info": stock.info or {},
                "financials": stock.financials if stock.financials is not None else pd.DataFrame(),
                "balance_sheet": stock.balance_sheet if stock.balance_sheet is not None else pd.DataFrame(),
                "cash_flow": stock.cashflow if stock.cashflow is not None else pd.DataFrame(),
                "recommendations": stock.recommendations if stock.recommendations is not None else pd.DataFrame(),
            }

        except Exception as e:
            logger.error(f"YahooProvider: Error fetching financials for {symbol}: {e}")
            return {
                "info": {},
                "financials": pd.DataFrame(),
                "balance_sheet": pd.DataFrame(),
                "cash_flow": pd.DataFrame(),
                "recommendations": pd.DataFrame(),
            }

    # ── NewsProvider ────────────────────────────────────────────────────

    def get_news(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Return recent news items."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            news = ticker.news
            return news[:limit] if news else []

        except Exception as e:
            logger.error(f"YahooProvider: Error fetching news for {symbol}: {e}")
            return []

    # ── RecommendationsProvider ─────────────────────────────────────────

    def get_recommendations(self, symbol: str):
        """Return analyst recommendations."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            return ticker.recommendations

        except Exception as e:
            logger.error(f"YahooProvider: Error fetching recommendations for {symbol}: {e}")
            return None
