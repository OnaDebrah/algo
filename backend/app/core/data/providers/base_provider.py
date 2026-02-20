from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd


class DataProvider(ABC):
    """
    Abstract base class for all data providers.
    Ensures a consistent interface for fetching OHLCV market data.
    """

    @abstractmethod
    def fetch_data(self, symbol: str, period: str, interval: str, start: Optional[Any] = None, end: Optional[Any] = None) -> pd.DataFrame:
        """
        Fetch OHLCV market data for a given symbol.

        Args:
            symbol: Ticker symbol.
            period: Time period (e.g., '1y', 'max').
            interval: Data interval (e.g., '1d', '1h').
            start: Optional start date.
            end: Optional end date.

        Returns:
            DataFrame with OHLCV data.
        """
        pass


# ── Optional capability mixins ─────────────────────────────────────────
# Providers implement only the interfaces they support.
# ProviderFactory checks isinstance() and falls back to Yahoo for unsupported types.


class QuoteProvider(ABC):
    """Provider that can supply real-time quote / ticker info snapshots."""

    @abstractmethod
    def get_quote(self, symbol: str) -> Dict:
        """
        Get real-time quote data.

        Returns:
            Dict with keys: price, change, changePct, volume, marketCap,
            high, low, open, previousClose, bid, ask, timestamp, etc.
        """
        pass


class OptionsDataProvider(ABC):
    """Provider that can supply options chain data."""

    @abstractmethod
    def get_option_expirations(self, symbol: str) -> List[str]:
        """Return list of available expiration date strings."""
        pass

    @abstractmethod
    def get_option_chain(self, symbol: str, expiration: str) -> Dict:
        """
        Fetch option chain for a specific expiration.

        Returns:
            Dict with keys: calls (DataFrame), puts (DataFrame),
            expirations (List[str]), underlying_price (float).
        """
        pass


class FundamentalsProvider(ABC):
    """Provider that can supply company info and financial statements."""

    @abstractmethod
    def get_ticker_info(self, symbol: str) -> Dict:
        """Return ticker metadata / info dict (company name, sector, market cap, etc.)."""
        pass

    @abstractmethod
    def get_financials(self, symbol: str) -> Dict:
        """
        Return comprehensive financial data.

        Returns:
            Dict with keys: info (dict), financials (DataFrame),
            balance_sheet (DataFrame), cash_flow (DataFrame),
            recommendations (DataFrame).
        """
        pass


class NewsProvider(ABC):
    """Provider that can supply news headlines."""

    @abstractmethod
    def get_news(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Return list of news item dicts (title, publisher, link, etc.)."""
        pass


class RecommendationsProvider(ABC):
    """Provider that can supply analyst recommendations."""

    @abstractmethod
    def get_recommendations(self, symbol: str) -> Any:
        """Return analyst recommendations (typically a DataFrame or list)."""
        pass
