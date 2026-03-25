import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from ....config import settings
from ....core.metrics import yfinance_request_duration_seconds, yfinance_requests_total
from ..providers.base_provider import (
    DataProvider,
    FundamentalsProvider,
    NewsProvider,
    OptionsDataProvider,
    QuoteProvider,
    RecommendationsProvider,
)

logger = logging.getLogger(__name__)


def _safe_num(val: Any, default: float = 0.0) -> float:
    """Convert a value to a safe float, treating None/NaN/Inf as default."""
    if val is None:
        return default
    try:
        f = float(val)
        if not (f == f):  # NaN check (faster than math.isnan)
            return default
        if f == float("inf") or f == float("-inf"):
            return default
        return f
    except (TypeError, ValueError):
        return default


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

            # yfinance rejects period + start/end together:
            # "Setting period, start and end is nonsense. Set maximum 2 of them."
            # Prefer start/end when provided, fall back to period otherwise.
            if start is not None or end is not None:
                use_period = None
            else:
                use_period = period

            ticker = yf.Ticker(symbol)
            data = ticker.history(period=use_period, interval=interval, start=start, end=end)

            if data.empty:
                logger.warning(f"YahooProvider: Ticker method failed for {symbol}, trying download method")
                data = yf.download(symbol, start=start, end=end, period=use_period, interval=interval, progress=False)

                # yfinance >= 0.2.31 returns MultiIndex columns from download().
                # Flatten to simple column names so downstream code can use data["Close"].
                if not data.empty and isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

            if not data.empty:
                logger.info(f"YahooProvider: Fetched {len(data)} bars for {symbol}")

            return data

        except Exception as e:
            logger.error(f"YahooProvider: Error fetching {symbol}: {e}")
            return pd.DataFrame()

    # ── QuoteProvider ───────────────────────────────────────────────────

    def get_quote(self, symbol: str) -> Dict:
        """Get real-time quote data for a symbol."""
        t0 = time.monotonic()
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Get fast info for real-time data
            try:
                fast_info = ticker.fast_info
                current_price = _safe_num(fast_info.last_price) or _safe_num(info.get("currentPrice"))
                previous_close = _safe_num(fast_info.previous_close) or _safe_num(info.get("previousClose"))
            except Exception as e:
                logger.error(f"Failed to get quote for {symbol}: {e}")
                current_price = _safe_num(info.get("currentPrice"))
                previous_close = _safe_num(info.get("previousClose"))

            change = current_price - previous_close if current_price and previous_close else 0.0
            change_pct = ((change / previous_close) * 100) if previous_close else 0.0

            result = {
                "symbol": symbol,
                "price": current_price,
                "change": change,
                "changePercent": change_pct,
                "volume": _safe_num(info.get("volume")),
                "marketCap": _safe_num(info.get("marketCap")),
                "high": _safe_num(info.get("dayHigh")),
                "low": _safe_num(info.get("dayLow")),
                "open": _safe_num(info.get("open")),
                "previousClose": previous_close,
                "bid": _safe_num(info.get("bid")),
                "ask": _safe_num(info.get("ask")),
                "bidSize": _safe_num(info.get("bidSize")),
                "askSize": _safe_num(info.get("askSize")),
                "fiftyTwoWeekHigh": _safe_num(info.get("fiftyTwoWeekHigh")),
                "fiftyTwoWeekLow": _safe_num(info.get("fiftyTwoWeekLow")),
                "avgVolume": _safe_num(info.get("averageVolume")),
                "timestamp": datetime.now().isoformat(),
            }
            yfinance_requests_total.labels(endpoint="quote", status="success").inc()
            yfinance_request_duration_seconds.labels(endpoint="quote").observe(time.monotonic() - t0)
            return result

        except Exception as e:
            yfinance_requests_total.labels(endpoint="quote", status="error").inc()
            yfinance_request_duration_seconds.labels(endpoint="quote").observe(time.monotonic() - t0)
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
