import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from ....utils.helpers import (
    FUNDAMENTAL_FACTORS as FUNDAMENTAL_FACTORS_,
    MACRO_INDICATORS as MACRO_INDICATORS_,
    SECTOR_ETFS as SECTOR_ETFS_,
    SECTOR_MAPPINGS as SECTOR_MAPPINGS_,
)

logger = logging.getLogger(__name__)


class FundamentalsFetcher:
    """Provider for fundamental and macro data"""

    SECTOR_MAPPINGS = SECTOR_MAPPINGS_
    SECTOR_ETFS = SECTOR_ETFS_

    FUNDAMENTAL_FACTORS = FUNDAMENTAL_FACTORS_
    MACRO_INDICATORS = MACRO_INDICATORS_

    def __init__(self, db_session=None):
        self.db = db_session
        self.cache = {}
        self.SECTOR_MAPPINGS = SECTOR_MAPPINGS_

    async def get_sector_universe(self, sector: str) -> List[str]:
        """Get all stocks in a sector"""
        return self.SECTOR_MAPPINGS.get(sector, [])

    async def get_all_sectors(self) -> List[str]:
        """Get list of all sectors"""
        return list(self.SECTOR_MAPPINGS.keys())

    async def fetch_fundamentals_batch(self, symbols: List[str], lookback_years: int = 5) -> pd.DataFrame:
        """
        Fetch fundamental data for multiple symbols

        Returns:
            DataFrame with multi-index (date, symbol) and fundamental factors
        """
        all_data = []

        for symbol in symbols:
            try:
                financials = await self._get_financials(symbol)
                factors = self._extract_fundamentals_from_info(symbol, financials)

                if factors:
                    row = {"date": datetime.now(), "symbol": symbol, **factors}
                    all_data.append(row)

            except Exception as e:
                logger.error(f"Error fetching fundamentals for {symbol}: {e}")
                continue

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df = df.set_index(["date", "symbol"])
        return df

    async def fetch_macro_data(self, lookback_years: int = 10) -> pd.DataFrame:
        """
        Fetch macroeconomic indicators via MacroManager (FRED/BLS)

        Returns:
            DataFrame with dates as index and macro indicators as columns
        """
        try:
            from ....core.data.fetchers.macro.macro_manager import MacroManager

            manager = MacroManager(self.db)
            start_date = datetime.now() - timedelta(days=lookback_years * 365)

            indicators = ["gdp_growth", "cpi_yoy", "unemployment_rate", "fed_funds_rate", "10y_treasury_yield", "vix"]

            data = await manager.get_indicators(indicators=indicators, start_date=start_date, frequency="monthly", country="USA")

            if data is not None and not data.empty:
                return data

        except Exception as e:
            logger.warning(f"MacroManager fetch failed, using VIX from yfinance: {e}")

        # Fallback: fetch VIX from yfinance as minimal macro proxy
        try:
            from backend.app.core.data.providers.providers import ProviderFactory

            provider = ProviderFactory()
            vix_data = await provider.fetch_data("^VIX", f"{lookback_years}y", "1mo")
            if not vix_data.empty:
                macro = pd.DataFrame(index=vix_data.index)
                macro["vix"] = vix_data["Close"]
                return macro
        except Exception as e:
            logger.error(f"VIX fallback also failed: {e}")

        return pd.DataFrame()

    async def _get_financials(self, symbol: str) -> Dict:
        """Get financial data for a symbol via ProviderFactory"""
        from backend.app.core.data.providers.providers import ProviderFactory

        provider = ProviderFactory()
        return await provider.get_ticker_info(symbol)

    def _extract_fundamentals_from_info(self, symbol: str, info: Dict) -> Optional[Dict]:
        """Extract fundamental factors from yfinance info dict"""
        if not info:
            return None

        factors: Dict = {}

        # Valuation
        factors["pe_ratio"] = info.get("trailingPE", 0) or 0
        factors["forward_pe"] = info.get("forwardPE", 0) or 0
        factors["peg_ratio"] = info.get("pegRatio", 0) or 0
        factors["pb_ratio"] = info.get("priceToBook", 0) or 0
        factors["ps_ratio"] = info.get("priceToSalesTrailing12Months", 0) or 0

        # Profitability
        factors["roe"] = info.get("returnOnEquity", 0) or 0
        factors["roa"] = info.get("returnOnAssets", 0) or 0
        factors["operating_margin"] = info.get("operatingMargins", 0) or 0
        factors["net_margin"] = info.get("profitMargins", 0) or 0

        # Growth
        factors["revenue_growth"] = info.get("revenueGrowth", 0) or 0
        factors["eps_growth"] = info.get("earningsGrowth", 0) or 0

        # Balance sheet
        factors["debt_to_equity"] = info.get("debtToEquity", 0) or 0
        factors["current_ratio"] = info.get("currentRatio", 0) or 0

        # Dividends
        factors["dividend_yield"] = info.get("dividendYield", 0) or 0
        factors["payout_ratio"] = info.get("payoutRatio", 0) or 0

        # Market data
        factors["market_cap"] = info.get("marketCap", 0) or 0
        factors["beta"] = info.get("beta", 0) or 0

        return factors
