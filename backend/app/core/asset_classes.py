"""
Multi-Asset Class Support
Support for Stocks, Crypto, Forex, Commodities, ETFs, Bonds, and Indices
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd

from backend.app.core.data.providers.providers import ProviderFactory

logger = logging.getLogger(__name__)


class AssetClass(Enum):
    """Supported asset classes"""

    STOCK = "Stock"
    CRYPTO = "Cryptocurrency"
    FOREX = "Foreign Exchange"
    COMMODITY = "Commodity"
    ETF = "ETF"
    INDEX = "Index"
    BOND = "Bond"
    FUTURES = "Futures"


@dataclass
class AssetInfo:
    """Information about an asset"""

    symbol: str
    asset_class: AssetClass
    name: str
    exchange: Optional[str] = None
    currency: str = "USD"
    trading_hours: Optional[str] = None
    min_tick: float = 0.01
    lot_size: int = 1
    leverage_available: bool = False
    metadata: Dict = None


class AssetClassManager:
    """Manage different asset classes and their specifics"""

    def __init__(self):
        self.asset_patterns = self._build_patterns()

    def _build_patterns(self) -> Dict:
        """Build patterns to detect asset class from symbol"""
        return {
            AssetClass.CRYPTO: {
                "suffixes": ["-USD", "-USDT", "-BTC", "-ETH"],
                "patterns": ["BTC", "ETH", "DOGE", "SHIB", "ADA", "SOL", "AVAX"],
            },
            AssetClass.FOREX: {
                "patterns": ["=X"],  # e.g., EURUSD=X
                "pairs": ["EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"],
            },
            AssetClass.COMMODITY: {
                "suffixes": ["=F"],  # Futures notation
                "symbols": ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F"],
            },
            AssetClass.INDEX: {
                "prefixes": ["^"],  # e.g., ^GSPC
                "symbols": ["^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX"],
            },
            AssetClass.ETF: {
                "symbols": [
                    "SPY",
                    "QQQ",
                    "IWM",
                    "DIA",
                    "VOO",
                    "VTI",
                    "GLD",
                    "SLV",
                    "USO",
                    "TLT",
                    "HYG",
                ]
            },
        }

    def detect_asset_class(self, symbol: str) -> AssetClass:
        """
        Detect asset class from symbol

        Args:
            symbol: Asset symbol

        Returns:
            Detected asset class
        """
        symbol = symbol.upper()

        # Check crypto
        for suffix in self.asset_patterns[AssetClass.CRYPTO]["suffixes"]:
            if symbol.endswith(suffix):
                return AssetClass.CRYPTO

        for pattern in self.asset_patterns[AssetClass.CRYPTO]["patterns"]:
            if pattern in symbol:
                return AssetClass.CRYPTO

        # Check forex
        if "=X" in symbol:
            return AssetClass.FOREX

        # Check commodities
        if symbol.endswith("=F"):
            return AssetClass.COMMODITY

        if symbol in self.asset_patterns[AssetClass.COMMODITY]["symbols"]:
            return AssetClass.COMMODITY

        # Check indices
        if symbol.startswith("^"):
            return AssetClass.INDEX

        # Check ETF
        if symbol in self.asset_patterns[AssetClass.ETF]["symbols"]:
            return AssetClass.ETF

        # Default to stock
        return AssetClass.STOCK

    async def get_asset_info(self, symbol: str) -> AssetInfo:
        """
        Get detailed information about an asset

        Args:
            symbol: Asset symbol

        Returns:
            Asset information
        """
        asset_class = self.detect_asset_class(symbol)

        try:
            factory = ProviderFactory()
            info = await factory.get_ticker_info(symbol)

            return AssetInfo(
                symbol=symbol,
                asset_class=asset_class,
                name=info.get("longName", info.get("shortName", symbol)),
                exchange=info.get("exchange"),
                currency=info.get("currency", "USD"),
                trading_hours=self._get_trading_hours(asset_class),
                min_tick=self._get_min_tick(asset_class),
                lot_size=self._get_lot_size(asset_class),
                leverage_available=self._has_leverage(asset_class),
                metadata=info,
            )
        except Exception as e:
            logger.warning(f"Could not fetch full info for {symbol}: {e}")

            return AssetInfo(
                symbol=symbol,
                asset_class=asset_class,
                name=symbol,
                trading_hours=self._get_trading_hours(asset_class),
                min_tick=self._get_min_tick(asset_class),
                lot_size=self._get_lot_size(asset_class),
                leverage_available=self._has_leverage(asset_class),
            )

    def _get_trading_hours(self, asset_class: AssetClass) -> str:
        """Get typical trading hours for asset class"""
        hours = {
            AssetClass.STOCK: "9:30 AM - 4:00 PM EST (Mon-Fri)",
            AssetClass.CRYPTO: "24/7",
            AssetClass.FOREX: "24/5 (Sun 5pm - Fri 5pm EST)",
            AssetClass.COMMODITY: "Varies by commodity",
            AssetClass.ETF: "9:30 AM - 4:00 PM EST (Mon-Fri)",
            AssetClass.INDEX: "9:30 AM - 4:00 PM EST (Mon-Fri)",
            AssetClass.BOND: "9:00 AM - 5:00 PM EST (Mon-Fri)",
            AssetClass.FUTURES: "23/5 (varies by contract)",
        }
        return hours.get(asset_class, "Check exchange")

    def _get_min_tick(self, asset_class: AssetClass) -> float:
        """Get minimum tick size for asset class"""
        ticks = {
            AssetClass.STOCK: 0.01,
            AssetClass.CRYPTO: 0.01,
            AssetClass.FOREX: 0.0001,
            AssetClass.COMMODITY: 0.01,
            AssetClass.ETF: 0.01,
            AssetClass.INDEX: 0.01,
            AssetClass.BOND: 0.01,
            AssetClass.FUTURES: 0.01,
        }
        return ticks.get(asset_class, 0.01)

    def _get_lot_size(self, asset_class: AssetClass) -> int:
        """Get standard lot size for asset class"""
        lots = {
            AssetClass.STOCK: 1,
            AssetClass.CRYPTO: 1,
            AssetClass.FOREX: 1000,  # Mini lot
            AssetClass.COMMODITY: 1,
            AssetClass.ETF: 1,
            AssetClass.INDEX: 1,
            AssetClass.BOND: 1,
            AssetClass.FUTURES: 1,
        }
        return lots.get(asset_class, 1)

    def _has_leverage(self, asset_class: AssetClass) -> bool:
        """Check if leverage is typically available"""
        return asset_class in [
            AssetClass.FOREX,
            AssetClass.CRYPTO,
            AssetClass.FUTURES,
            AssetClass.COMMODITY,
        ]

    def get_popular_symbols(self, asset_class: AssetClass) -> List[str]:
        """
        Get popular symbols for an asset class

        Args:
            asset_class: Asset class

        Returns:
            List of popular symbols
        """
        popular = {
            AssetClass.STOCK: [
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "TSLA",
                "NVDA",
                "META",
                "JPM",
                "V",
                "WMT",
                "JNJ",
                "PG",
            ],
            AssetClass.CRYPTO: [
                "BTC-USD",
                "ETH-USD",
                "BNB-USD",
                "SOL-USD",
                "ADA-USD",
                "DOGE-USD",
                "MATIC-USD",
                "DOT-USD",
            ],
            AssetClass.FOREX: [
                "EURUSD=X",
                "GBPUSD=X",
                "USDJPY=X",
                "AUDUSD=X",
                "USDCAD=X",
                "USDCHF=X",
                "NZDUSD=X",
                "EURGBP=X",
            ],
            AssetClass.COMMODITY: [
                "GC=F",  # Gold
                "SI=F",  # Silver
                "CL=F",  # Crude Oil
                "NG=F",  # Natural Gas
                "HG=F",  # Copper
                "ZW=F",  # Wheat
            ],
            AssetClass.ETF: [
                "SPY",
                "QQQ",
                "IWM",
                "DIA",
                "VOO",
                "VTI",
                "GLD",
                "SLV",
                "USO",
                "TLT",
                "HYG",
                "LQD",
            ],
            AssetClass.INDEX: [
                "^GSPC",  # S&P 500
                "^DJI",  # Dow Jones
                "^IXIC",  # NASDAQ
                "^RUT",  # Russell 2000
                "^VIX",  # VIX
                "^FTSE",  # FTSE 100
            ],
            AssetClass.BOND: ["TLT", "IEF", "SHY", "AGG", "BND", "LQD", "HYG"],
            AssetClass.FUTURES: [
                "ES=F",  # E-mini S&P 500
                "NQ=F",  # E-mini NASDAQ
                "YM=F",  # E-mini Dow
            ],
        }

        return popular.get(asset_class, [])

    async def validate_symbol(self, symbol: str) -> tuple[bool, str]:
        """
        Validate if symbol exists and is tradeable

        Args:
            symbol: Asset symbol

        Returns:
            Tuple of (is_valid, message)
        """
        try:
            factory = ProviderFactory()
            data = await factory.fetch_data(symbol, "5d", "1d")

            if data.empty:
                return False, f"No data available for {symbol}"

            return True, "Valid symbol"

        except Exception as e:
            return False, f"Invalid symbol: {str(e)}"

    def get_asset_class_info(self, asset_class: AssetClass) -> Dict:
        """
        Get general information about an asset class

        Args:
            asset_class: Asset class

        Returns:
            Information dictionary
        """
        info = {
            AssetClass.STOCK: {
                "description": "Equity shares of publicly traded companies",
                "risk_level": "Medium to High",
                "liquidity": "High",
                "trading_hours": "Market hours only",
                "leverage": "Up to 2x (margin)",
                "typical_spread": "0.01-0.10%",
                "best_for": "Long-term growth, dividends",
                "considerations": "Company fundamentals, earnings, sector rotation",
            },
            AssetClass.CRYPTO: {
                "description": "Digital cryptocurrencies",
                "risk_level": "Very High",
                "liquidity": "High for major coins",
                "trading_hours": "24/7",
                "leverage": "Up to 100x (varies by exchange)",
                "typical_spread": "0.1-1%",
                "best_for": "High risk/reward, diversification",
                "considerations": "Extreme volatility, regulatory risk, technology risk",
            },
            AssetClass.FOREX: {
                "description": "Currency pairs",
                "risk_level": "High",
                "liquidity": "Very High",
                "trading_hours": "24/5",
                "leverage": "Up to 50x (US) or higher",
                "typical_spread": "0.001-0.01%",
                "best_for": "Short-term trading, hedging",
                "considerations": "Interest rates, geopolitics, economic data",
            },
            AssetClass.COMMODITY: {
                "description": "Physical goods (metals, energy, agriculture)",
                "risk_level": "High",
                "liquidity": "Medium to High",
                "trading_hours": "Varies by commodity",
                "leverage": "Up to 20x",
                "typical_spread": "0.1-0.5%",
                "best_for": "Inflation hedge, diversification",
                "considerations": "Supply/demand, weather, geopolitics",
            },
            AssetClass.ETF: {
                "description": "Exchange-traded funds",
                "risk_level": "Low to High (varies)",
                "liquidity": "High",
                "trading_hours": "Market hours",
                "leverage": "Some 2-3x leveraged ETFs",
                "typical_spread": "0.01-0.20%",
                "best_for": "Diversification, passive investing",
                "considerations": "Expense ratios, tracking error, composition",
            },
            AssetClass.INDEX: {
                "description": "Market indices",
                "risk_level": "Medium",
                "liquidity": "Very High (via derivatives)",
                "trading_hours": "Market hours",
                "leverage": "Via futures/options",
                "typical_spread": "0.01-0.05%",
                "best_for": "Market exposure, benchmarking",
                "considerations": "Trade via ETFs or futures",
            },
            AssetClass.BOND: {
                "description": "Fixed income securities",
                "risk_level": "Low to Medium",
                "liquidity": "Medium to High",
                "trading_hours": "Market hours",
                "leverage": "Limited",
                "typical_spread": "0.05-0.50%",
                "best_for": "Income, capital preservation",
                "considerations": "Interest rates, credit risk, duration",
            },
            AssetClass.FUTURES: {
                "description": "Derivative contracts",
                "risk_level": "Very High",
                "liquidity": "Very High",
                "trading_hours": "23/5",
                "leverage": "Built-in (10-50x)",
                "typical_spread": "0.01-0.10%",
                "best_for": "Hedging, speculation",
                "considerations": "Expiration dates, rollover costs, margin requirements",
            },
        }

        return info.get(asset_class, {})

    async def fetch_data(
        self,
        symbol: str,
        period: str,
        interval: str,
        asset_class: Optional[AssetClass] = None,
    ) -> pd.DataFrame:
        """
        Fetch data for any asset class

        Args:
            symbol: Asset symbol
            period: Time period
            interval: Data interval
            asset_class: Optional asset class (will detect if not provided)

        Returns:
            DataFrame with OHLCV data
        """
        if asset_class is None:
            asset_class = self.detect_asset_class(symbol)

        logger.info(f"Fetching {asset_class.value} data for {symbol}")

        try:
            from backend.app.core import fetch_stock_data

            data = await fetch_stock_data(symbol, period, interval)

            if not data.empty:
                # Add asset class metadata
                data.attrs["asset_class"] = asset_class.value
                data.attrs["symbol"] = symbol

            return data

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()


# Global instance
asset_manager = AssetClassManager()


def get_asset_manager() -> AssetClassManager:
    """Get global asset class manager"""
    return asset_manager


def get_asset_classes() -> List[AssetClass]:
    """Get list of supported asset classes"""
    return list(AssetClass)


def get_asset_class_names() -> List[str]:
    """Get list of asset class display names"""
    return [ac.value for ac in AssetClass]
