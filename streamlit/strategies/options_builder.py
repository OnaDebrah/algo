"""
Options Strategy Builder and Backtesting Module
Supports complex options strategies with Greeks calculation
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from streamlit.strategies.options_strategies import (
    BlackScholesCalculator,
    OptionLeg,
    OptionsChain,
    OptionsStrategy,
    OptionType,
)

logger = logging.getLogger(__name__)


class OptionsStrategyBuilder:
    """Build and analyze complex options strategies"""

    def __init__(self, symbol: str, risk_free_rate: float = 0.05, dividend_yield: float = 0.0):
        self.symbol = symbol
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.legs: List[OptionLeg] = []
        self.chain = OptionsChain(symbol)
        self.current_price = self.chain.get_current_price()

    def add_leg(
        self,
        option_type: OptionType,
        strike: float,
        expiry: datetime,
        quantity: int,
        premium: Optional[float] = None,
        volatility: Optional[float] = None,
    ):
        """Add an option leg to the strategy"""

        if premium is None:
            # Calculate theoretical premium
            expiry_naive = expiry.replace(tzinfo=None) if expiry.tzinfo else expiry
            T = (expiry_naive - datetime.now()).days / 365.0

            if volatility is None:
                # Estimate IV from chain
                exp_str = expiry.strftime("%Y-%m-%d")
                volatility = self.chain.get_implied_volatility(exp_str)

            premium = BlackScholesCalculator.calculate_option_price(
                S=self.current_price,
                K=strike,
                T=T,
                r=self.risk_free_rate,
                sigma=volatility,
                option_type=option_type,
                q=self.dividend_yield,
            )

        leg = OptionLeg(
            option_type=option_type,
            strike=strike,
            expiry=expiry,
            quantity=quantity,
            premium=premium,
            underlying_price=self.current_price,
        )

        self.legs.append(leg)
        logger.info(f"Added leg: {leg}")

    def clear_legs(self):
        """Clear all legs"""
        self.legs = []

    def get_initial_cost(self) -> float:
        """Calculate initial cost/credit of the strategy"""
        cost = sum(leg.premium * leg.quantity * 100 for leg in self.legs)
        return cost

    def calculate_payoff(self, underlying_prices: np.ndarray) -> np.ndarray:
        """
        Calculate strategy payoff at expiration

        Args:
            underlying_prices: Array of underlying prices to evaluate

        Returns:
            Array of payoffs
        """
        payoff = np.zeros_like(underlying_prices)

        for leg in self.legs:
            if leg.option_type == OptionType.CALL:
                intrinsic = np.maximum(underlying_prices - leg.strike, 0)
            else:
                intrinsic = np.maximum(leg.strike - underlying_prices, 0)

            # Account for quantity and initial premium
            payoff += leg.quantity * (intrinsic * 100 - leg.premium * 100)

        return payoff

    def calculate_greeks(self, volatility: Optional[float] = None) -> Dict[str, float]:
        """Calculate aggregate Greeks for the strategy"""

        total_greeks = {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}

        for leg in self.legs:
            T = (leg.expiry - datetime.now()).days / 365.0

            if volatility is None:
                exp_str = leg.expiry.strftime("%Y-%m-%d")
                vol = self.chain.get_implied_volatility(exp_str)
            else:
                vol = volatility

            greeks = BlackScholesCalculator.calculate_greeks(
                S=self.current_price,
                K=leg.strike,
                T=T,
                r=self.risk_free_rate,
                sigma=vol,
                option_type=leg.option_type,
                q=self.dividend_yield,
            )

            # Aggregate Greeks (multiply by quantity)
            total_greeks["delta"] += greeks.delta * leg.quantity
            total_greeks["gamma"] += greeks.gamma * leg.quantity
            total_greeks["theta"] += greeks.theta * leg.quantity
            total_greeks["vega"] += greeks.vega * leg.quantity
            total_greeks["rho"] += greeks.rho * leg.quantity

        return total_greeks

    def get_breakeven_points(self) -> List[float]:
        """Calculate breakeven points for the strategy"""

        # Sample prices around current price
        price_range = np.linspace(self.current_price * 0.5, self.current_price * 1.5, 1000)

        payoffs = self.calculate_payoff(price_range)

        # Find zero crossings
        breakevens = []
        for i in range(len(payoffs) - 1):
            if payoffs[i] * payoffs[i + 1] < 0:  # Sign change
                # Linear interpolation
                x1, x2 = price_range[i], price_range[i + 1]
                y1, y2 = payoffs[i], payoffs[i + 1]
                breakeven = x1 - y1 * (x2 - x1) / (y2 - y1)
                breakevens.append(breakeven)

        return breakevens

    def get_max_profit(self) -> Tuple[float, str]:
        """Calculate maximum profit and condition"""

        price_range = np.linspace(self.current_price * 0.5, self.current_price * 1.5, 1000)

        payoffs = self.calculate_payoff(price_range)
        max_profit = np.max(payoffs)

        if max_profit == payoffs[-1] and payoffs[-1] > payoffs[-2]:
            condition = "Unlimited (upside)"
        elif max_profit == payoffs[0] and payoffs[0] > payoffs[1]:
            condition = "Unlimited (downside)"
        else:
            max_idx = np.argmax(payoffs)
            condition = f"At ${price_range[max_idx]:.2f}"

        return max_profit, condition

    def get_max_loss(self) -> Tuple[float, str]:
        """Calculate maximum loss and condition"""

        price_range = np.linspace(self.current_price * 0.5, self.current_price * 1.5, 1000)

        payoffs = self.calculate_payoff(price_range)
        max_loss = np.min(payoffs)

        if max_loss == payoffs[-1] and payoffs[-1] < payoffs[-2]:
            condition = "Unlimited (upside)"
        elif max_loss == payoffs[0] and payoffs[0] < payoffs[1]:
            condition = "Unlimited (downside)"
        else:
            min_idx = np.argmin(payoffs)
            condition = f"At ${price_range[min_idx]:.2f}"

        return max_loss, condition

    def calculate_probability_of_profit(
        self,
        volatility: Optional[float] = None,
        days_to_expiration: Optional[int] = None,
    ) -> float:
        """
        Calculate probability of profit at expiration

        Uses log-normal distribution assumption
        """

        if not self.legs:
            return 0.0

        if days_to_expiration is None:
            days_to_expiration = (self.legs[0].expiry - datetime.now()).days

        if volatility is None:
            exp_str = self.legs[0].expiry.strftime("%Y-%m-%d")
            volatility = self.chain.get_implied_volatility(exp_str)

        T = days_to_expiration / 365.0

        # Get breakeven points
        breakevens = self.get_breakeven_points()

        if not breakevens:
            return 0.0

        # Calculate probability for each breakeven region
        # This is simplified - for complex strategies, use Monte Carlo

        total_prob = 0.0

        # Sample many prices
        num_samples = 10000
        prices = self.current_price * np.exp((self.risk_free_rate - 0.5 * volatility**2) * T + volatility * np.sqrt(T) * np.random.randn(num_samples))

        payoffs = self.calculate_payoff(prices)
        total_prob = np.sum(payoffs > 0) / num_samples

        return total_prob


def create_preset_strategy(
    strategy_type: OptionsStrategy,
    symbol: str,
    current_price: float,
    expiration: datetime,
    **kwargs,
) -> OptionsStrategyBuilder:
    """
    Create a preset options strategy matching the frontend templates.
    """
    builder = OptionsStrategyBuilder(symbol)
    builder.current_price = current_price

    # Helper for secondary expiration (Calendar/Diagonal)
    expiration_long = kwargs.get("expiration_long", expiration + timedelta(days=30))

    if strategy_type == OptionsStrategy.COVERED_CALL:
        # Long 100 shares + Short 1 OTM call
        strike = kwargs.get("strike", round(current_price * 1.05, 2))
        builder.add_leg(OptionType.STOCK, 0, None, 100) # Underlying leg
        builder.add_leg(OptionType.CALL, strike, expiration, -1)

    elif strategy_type == OptionsStrategy.CASH_SECURED_PUT:
        # Short 1 OTM put
        strike = kwargs.get("strike", current_price * 0.95)
        builder.add_leg(OptionType.PUT, strike, expiration, -1)

    elif strategy_type == OptionsStrategy.PROTECTIVE_PUT:
        # Long stock + Long 1 OTM put
        strike = kwargs.get("strike", round(current_price * 0.95, 2))
        builder.add_leg(OptionType.STOCK, 0, None, 100)
        builder.add_leg(OptionType.PUT, strike, expiration, 1)

    elif strategy_type == OptionsStrategy.VERTICAL_CALL_SPREAD:
        # Long lower strike call + Short higher strike call (Bull Call)
        long_strike = kwargs.get("long_strike", round(current_price, 2))
        short_strike = kwargs.get("short_strike", round(current_price * 1.05, 2))
        builder.add_leg(OptionType.CALL, long_strike, expiration, 1)
        builder.add_leg(OptionType.CALL, short_strike, expiration, -1)

    elif strategy_type == OptionsStrategy.VERTICAL_PUT_SPREAD:
        # Long higher strike put + Short lower strike put (Bear Put)
        long_strike = kwargs.get("long_strike", round(current_price, 2))
        short_strike = kwargs.get("short_strike", round(current_price * 0.95, 2))
        builder.add_leg(OptionType.PUT, long_strike, expiration, 1)
        builder.add_leg(OptionType.PUT, short_strike, expiration, -1)

    elif strategy_type == OptionsStrategy.IRON_CONDOR:
        # OTM put spread + OTM call spread
        put_long = kwargs.get("put_long_strike", round(current_price * 0.90, 2))
        put_short = kwargs.get("put_short_strike", round(current_price * 0.95, 2))
        call_short = kwargs.get("call_short_strike", round(current_price * 1.05, 2))
        call_long = kwargs.get("call_long_strike", round(current_price * 1.10, 2))

        builder.add_leg(OptionType.PUT, put_long, expiration, 1)
        builder.add_leg(OptionType.PUT, put_short, expiration, -1)
        builder.add_leg(OptionType.CALL, call_short, expiration, -1)
        builder.add_leg(OptionType.CALL, call_long, expiration, 1)

    elif strategy_type == OptionsStrategy.BUTTERFLY_SPREAD:
        opt_type = kwargs.get("option_type", OptionType.CALL)
        lower = kwargs.get("lower_strike", round(current_price * 0.95, 2))
        middle = kwargs.get("middle_strike", round(current_price, 2))
        upper = kwargs.get("upper_strike", round(current_price * 1.05, 2))

        builder.add_leg(opt_type, lower, expiration, 1)
        builder.add_leg(opt_type, middle, expiration, -2)
        builder.add_leg(opt_type, upper, expiration, 1)

    elif strategy_type == OptionsStrategy.LONG_STRADDLE:
        strike = kwargs.get("strike", round(current_price, 2))
        builder.add_leg(OptionType.CALL, strike, expiration, 1)
        builder.add_leg(OptionType.PUT, strike, expiration, 1)

    elif strategy_type == OptionsStrategy.LONG_STRANGLE:
        call_strike = kwargs.get("call_strike", round(current_price * 1.05, 2))
        put_strike = kwargs.get("put_strike", round(current_price * 0.95, 2))
        builder.add_leg(OptionType.CALL, call_strike, expiration, 1)
        builder.add_leg(OptionType.PUT, put_strike, expiration, 1)

    elif strategy_type == OptionsStrategy.CALENDAR_SPREAD:
        # Sell Near-term, Buy Long-term (Same Strike)
        strike = kwargs.get("strike", round(current_price, 2))
        opt_type = kwargs.get("option_type", OptionType.CALL)
        builder.add_leg(opt_type, strike, expiration, -1) # Short near
        builder.add_leg(opt_type, strike, expiration_long, 1) # Long far

    elif strategy_type == OptionsStrategy.DIAGONAL_SPREAD:
        # Sell Near OTM, Buy Far ITM/ATM
        short_strike = kwargs.get("short_strike", round(current_price * 1.05, 2))
        long_strike = kwargs.get("long_strike", round(current_price, 2))
        opt_type = kwargs.get("option_type", OptionType.CALL)
        builder.add_leg(opt_type, short_strike, expiration, -1)
        builder.add_leg(opt_type, long_strike, expiration_long, 1)

    elif strategy_type == OptionsStrategy.COLLAR:
        # Long stock + Long OTM put + Short OTM call
        put_strike = kwargs.get("put_strike", round(current_price * 0.95, 2))
        call_strike = kwargs.get("call_strike", round(current_price * 1.05, 2))
        builder.add_leg(OptionType.STOCK, 0, None, 100)
        builder.add_leg(OptionType.PUT, put_strike, expiration, 1)
        builder.add_leg(OptionType.CALL, call_strike, expiration, -1)

    elif strategy_type == OptionsStrategy.RATIO_SPREAD:
        # Buy 1 ITM/ATM, Sell 2 OTM
        long_strike = kwargs.get("long_strike", round(current_price, 2))
        short_strike = kwargs.get("short_strike", round(current_price * 1.05, 2))
        opt_type = kwargs.get("option_type", OptionType.CALL)
        builder.add_leg(opt_type, long_strike, expiration, 1)
        builder.add_leg(opt_type, short_strike, expiration, -2)

    return builder