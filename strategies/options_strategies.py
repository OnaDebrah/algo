"""
Options Strategy Builder and Backtesting Module
Supports complex options strategies with Greeks calculation
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

from strategies import BaseStrategy

logger = logging.getLogger(__name__)


class OptionType(Enum):
    """Option types"""

    CALL = "Call"
    PUT = "Put"


class OptionsStrategy(Enum, BaseStrategy):
    """Pre-defined option strategies"""

    COVERED_CALL = "Covered Call"
    CASH_SECURED_PUT = "Cash-Secured Put"
    PROTECTIVE_PUT = "Protective Put"
    VERTICAL_CALL_SPREAD = "Vertical Call Spread"
    VERTICAL_PUT_SPREAD = "Vertical Put Spread"
    IRON_CONDOR = "Iron Condor"
    BUTTERFLY_SPREAD = "Butterfly Spread"
    STRADDLE = "Long Straddle"
    STRANGLE = "Long Strangle"
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


class BlackScholesCalculator:
    """Black-Scholes option pricing and Greeks calculation"""

    @staticmethod
    def calculate_option_price(
        S: float,  # Current stock price
        K: float,  # Strike price
        T: float,  # Time to expiration (years)
        r: float,  # Risk-free rate
        sigma: float,  # Volatility
        option_type: OptionType,
        q: float = 0.0,  # Dividend yield
    ) -> float:
        """Calculate option price using Black-Scholes"""

        if T <= 0:
            # At expiration
            if option_type == OptionType.CALL:
                return max(S - K, 0)
            else:
                return max(K - S, 0)

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == OptionType.CALL:
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

        return price

    @staticmethod
    def calculate_greeks(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        q: float = 0.0,
    ) -> Greeks:
        """Calculate option Greeks"""

        if T <= 0:
            return Greeks(delta=0, gamma=0, theta=0, vega=0, rho=0)

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Delta
        if option_type == OptionType.CALL:
            delta = np.exp(-q * T) * norm.cdf(d1)
        else:
            delta = np.exp(-q * T) * (norm.cdf(d1) - 1)

        # Gamma (same for calls and puts)
        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

        # Theta
        term1 = -(S * norm.pdf(d1) * sigma * np.exp(-q * T)) / (2 * np.sqrt(T))
        if option_type == OptionType.CALL:
            term2 = q * S * norm.cdf(d1) * np.exp(-q * T)
            term3 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            term2 = -q * S * norm.cdf(-d1) * np.exp(-q * T)
            term3 = r * K * np.exp(-r * T) * norm.cdf(-d2)

        theta = (term1 + term2 + term3) / 365  # Per day

        # Vega (same for calls and puts)
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% change

        # Rho
        if option_type == OptionType.CALL:
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

        return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)


class OptionsChain:
    """Fetch and manage options chain data"""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)

    def get_expirations(self) -> List[str]:
        """Get available expiration dates"""
        try:
            return list(self.ticker.options)
        except Exception as e:
            logger.error(f"Error fetching expirations for {self.symbol}: {e}")
            return []

    def get_chain(self, expiration: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get options chain for specific expiration

        Returns:
            Tuple of (calls_df, puts_df)
        """
        try:
            opt = self.ticker.option_chain(expiration)
            return opt.calls, opt.puts
        except Exception as e:
            logger.error(f"Error fetching chain for {self.symbol} {expiration}: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def get_current_price(self) -> float:
        """Get current stock price"""
        try:
            return self.ticker.history(period="1d")["Close"].iloc[-1]
        except Exception as e:
            logger.error(f"Error fetching price for {self.symbol}: {e}")
            return 0.0

    def get_implied_volatility(self, expiration: str) -> float:
        """Estimate implied volatility from options chain"""
        try:
            calls, puts = self.get_chain(expiration)

            # Use ATM options for IV estimation
            current_price = self.get_current_price()

            # Find closest to ATM
            if not calls.empty:
                atm_call = calls.iloc[(calls["strike"] - current_price).abs().argsort()[:1]]
                if not atm_call.empty and "impliedVolatility" in atm_call.columns:
                    return float(atm_call["impliedVolatility"].iloc[0])

            # Fallback to historical volatility
            hist = self.ticker.history(period="1y")
            returns = hist["Close"].pct_change().dropna()
            return returns.std() * np.sqrt(252)

        except Exception as e:
            logger.warning(f"Error calculating IV for {self.symbol}: {e}")
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
        OptionsStrategy.STRADDLE: {
            "description": "Buy a call and put at same strike (usually ATM)",
            "outlook": "Expecting large move in either direction",
            "max_profit": "Unlimited",
            "max_loss": "Total premium paid",
            "best_for": "High volatility events (earnings, news)",
            "breakeven": "Strike +/- total premium paid",
        },
    }

    return descriptions.get(strategy, {})
