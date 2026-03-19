"""
Options Strategy Builder and Backtesting Module
Supports complex options strategies with Greeks calculation
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, cast

import numpy as np
import pandas as pd
from config import DEFAULT_ANNUAL_LOOKBACK

from ..core.data.providers.providers import ProviderFactory

logger = logging.getLogger(__name__)


class OptionType(Enum):
    """Option types"""

    CALL = "Call"
    PUT = "Put"
    STOCK = "Stock"


class OptionsStrategy(Enum):
    """Pre-defined option strategies"""

    COVERED_CALL = "Covered Call"
    CASH_SECURED_PUT = "Cash-Secured Put"
    PROTECTIVE_PUT = "Protective Put"
    VERTICAL_CALL_SPREAD = "Vertical Call Spread"
    VERTICAL_PUT_SPREAD = "Vertical Put Spread"
    IRON_CONDOR = "Iron Condor"
    BUTTERFLY_SPREAD = "Butterfly Spread"
    LONG_STRADDLE = "Long Straddle"
    LONG_STRANGLE = "Long Strangle"
    CALENDAR_SPREAD = "Calendar Spread"
    DIAGONAL_SPREAD = "Diagonal Spread"
    COLLAR = "Collar"
    RATIO_SPREAD = "Ratio Spread"


@dataclass
class OptionLeg:
    """Individual option leg in a strategy"""

    option_type: OptionType
    strike: float
    expiry: datetime
    quantity: int  # Positive for long, negative for short
    premium: float
    underlying_price: float

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    def __repr__(self):
        pos = "Long" if self.is_long else "Short"
        return f"{pos} {abs(self.quantity)} {self.option_type.value} ${self.strike}"


@dataclass
class Greeks:
    """Option Greeks"""

    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


class OptionsChain:
    """Fetch and manage options chain data via the provider layer."""

    def __init__(self, symbol: str, provider_factory: ProviderFactory = None):
        self.symbol = symbol
        self._factory = provider_factory or ProviderFactory()

    async def get_expirations(self) -> List[str]:
        """Get available expiration dates"""
        try:
            return await self._factory.get_option_expirations(self.symbol)
        except Exception as e:
            logger.error(f"Error fetching expirations for {self.symbol}: {e}")
            return []

    async def get_chain(self, expiration: str = None):
        """
        Get options chain data

        Args:
            expiration: Specific expiration date (YYYY-MM-DD) or None for metadata only

        Returns:
            dict with 'calls' and 'puts' DataFrames and 'expiration_dates'
        """
        try:
            # If no expiration specified, return metadata only
            if expiration is None:
                expirations = await self.get_expirations()
                if not expirations:
                    raise ValueError(f"No options available for {self.symbol}")
                return {"expiration_dates": expirations, "calls": pd.DataFrame(), "puts": pd.DataFrame()}

            # Get options for specific expiration
            chain = await self._factory.get_option_chain(self.symbol, expiration)

            return {
                "calls": chain.get("calls", pd.DataFrame()),
                "puts": chain.get("puts", pd.DataFrame()),
                "expiration_dates": chain.get("expirations", [expiration]),
            }

        except Exception as e:
            logger.error(f"Error fetching options chain: {str(e)}")
            raise

    async def get_current_price(self) -> float:
        """Get current stock price"""
        try:
            quote = await self._factory.get_quote(self.symbol)
            return quote.get("price", 0.0)
        except Exception as e:
            logger.error(f"Error fetching price for {self.symbol}: {e}")
            return 0.0

    async def get_implied_volatility(self, expiration: str) -> float:
        """Estimate implied volatility from options chain"""
        try:
            chain_data = await self.get_chain(expiration)
            calls = chain_data.get("calls", pd.DataFrame())

            # Use ATM options for IV estimation
            current_price = await self.get_current_price()

            # Find closest to ATM
            if not calls.empty and "strike" in calls.columns:
                atm_call = cast(pd.DataFrame, cast(object, calls.iloc[(calls["strike"] - current_price).abs().argsort()[:1]]))
                if not atm_call.empty and "impliedVolatility" in atm_call.columns:
                    return float(atm_call["impliedVolatility"].iloc[0])

            # Fallback to historical volatility
            hist = await self._factory.fetch_data(self.symbol, "1y", "1d")
            if not hist.empty:
                returns = hist["Close"].pct_change().dropna()
                return returns.std() * np.sqrt(DEFAULT_ANNUAL_LOOKBACK)

            return 0.3  # Default 30%

        except Exception as e:
            logger.error(f"Error calculating IV for {self.symbol}: {e}")
            return 0.3  # Default 30%


def get_strategy_description(strategy: OptionsStrategy) -> Dict[str, str]:
    """Get description and characteristics of a strategy"""

    descriptions = {
        OptionsStrategy.COVERED_CALL: {
            "description": "Sell a call against stock you own to generate income",
            "outlook": "Neutral to slightly bullish",
            "max_profit": "Limited to strike - stock cost + premium",
            "max_loss": "Stock cost - premium (if stock goes to zero)",
            "best_for": "Income generation on stocks you own",
            "breakeven": "Stock cost - premium received",
        },
        OptionsStrategy.CASH_SECURED_PUT: {
            "description": "Sell a put with cash reserved to buy stock if assigned",
            "outlook": "Neutral to bullish",
            "max_profit": "Premium received",
            "max_loss": "Strike price - premium (if stock goes to zero)",
            "best_for": "Acquiring stock at a discount or income generation",
            "breakeven": "Strike - premium received",
        },
        OptionsStrategy.VERTICAL_CALL_SPREAD: {
            "description": "Buy a call and sell a higher strike call",
            "outlook": "Moderately bullish",
            "max_profit": "Spread width - net debit",
            "max_loss": "Net debit paid",
            "best_for": "Directional bullish trades with defined risk",
            "breakeven": "Long strike + net debit",
        },
        OptionsStrategy.IRON_CONDOR: {
            "description": "Sell OTM put spread and call spread simultaneously",
            "outlook": "Neutral (expect low volatility)",
            "max_profit": "Net premium received",
            "max_loss": "Width of widest spread - net premium",
            "best_for": "Range-bound markets, high IV",
            "breakeven": "Two breakevens at short strikes +/- net premium",
        },
        OptionsStrategy.LONG_STRADDLE: {
            "description": "Buy a call and put at same strike (usually ATM)",
            "outlook": "Expecting large move in either direction",
            "max_profit": "Unlimited",
            "max_loss": "Total premium paid",
            "best_for": "High volatility events (earnings, news)",
            "breakeven": "Strike +/- total premium paid",
        },
        OptionsStrategy.VERTICAL_PUT_SPREAD: {
            "description": "Buy a put and sell a lower strike put",
            "outlook": "Moderately bearish",
            "max_profit": "Spread width - net debit",
            "max_loss": "Net debit paid",
            "best_for": "Directional bearish trades with defined risk",
            "breakeven": "Long strike - net debit",
        },
        OptionsStrategy.BUTTERFLY_SPREAD: {
            "description": "Buy 1 ITM call, sell 2 ATM calls, buy 1 OTM call",
            "outlook": "Neutral (target specific price at expiration)",
            "max_profit": "Distance between strikes - net debit",
            "max_loss": "Net debit paid",
            "best_for": "Low volatility; targeting a specific price pinpoint",
            "breakeven": "Lower strike + debit / Higher strike - debit",
        },
        OptionsStrategy.LONG_STRANGLE: {
            "description": "Buy OTM call and OTM put at different strikes",
            "outlook": "Expecting very large move in either direction",
            "max_profit": "Unlimited",
            "max_loss": "Total premium paid",
            "best_for": "Lower cost than straddle; betting on high volatility",
            "breakeven": "Call strike + premium / Put strike - premium",
        },
        OptionsStrategy.CALENDAR_SPREAD: {
            "description": "Sell a short-term option and buy a long-term option",
            "outlook": "Neutral to slightly directional (targeting time decay)",
            "max_profit": "Limited (depends on implied volatility shift)",
            "max_loss": "Net debit paid",
            "best_for": "Profiting from faster decay of near-term options",
            "breakeven": "Dynamic (varies with volatility)",
        },
        OptionsStrategy.PROTECTIVE_PUT: {
            "description": "Buy a put for a stock you already own",
            "outlook": "Bullish but concerned about short-term downside",
            "max_profit": "Unlimited (offset by put cost)",
            "max_loss": "Stock cost + put premium - put strike",
            "best_for": "Hedging stock positions against a market crash",
            "breakeven": "Stock cost + put premium",
        },
        OptionsStrategy.COLLAR: {
            "description": "Long stock + Short OTM Call + Long OTM Put",
            "outlook": "Neutral to slightly bullish",
            "max_profit": "Call strike - stock price + net credit/debit",
            "max_loss": "Stock price - put strike - net credit/debit",
            "best_for": "Protecting large gains while funding the hedge with income",
            "breakeven": "Stock cost - net credit (or + net debit)",
        },
        OptionsStrategy.RATIO_SPREAD: {
            "description": "Buy 1 option and sell 2+ options at a further strike",
            "outlook": "Directional but with a specific target price",
            "max_profit": "Width of spread + net credit (if any)",
            "max_loss": "Unlimited (on the short side if price exceeds strikes)",
            "best_for": "Aggressive directional traders; potential zero-cost entry",
            "breakeven": "Short strike + width of spread (for call ratio)",
        },
    }

    return descriptions.get(strategy, {})
