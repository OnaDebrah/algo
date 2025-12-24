"""
Options Strategy Builder and Backtesting Module
Supports complex options strategies with Greeks calculation
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from strategies.options_strategies import (
    BlackScholesCalculator,
    OptionLeg,
    OptionsChain,
    OptionStrategy,
    OptionType,
)

logger = logging.getLogger(__name__)


class OptionsStrategyBuilder:
    """Build and analyze complex options strategies"""

    def __init__(
        self, symbol: str, risk_free_rate: float = 0.05, dividend_yield: float = 0.0
    ):
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
            T = (expiry - datetime.now()).days / 365.0

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
        price_range = np.linspace(
            self.current_price * 0.5, self.current_price * 1.5, 1000
        )

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

        price_range = np.linspace(
            self.current_price * 0.5, self.current_price * 1.5, 1000
        )

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

        price_range = np.linspace(
            self.current_price * 0.5, self.current_price * 1.5, 1000
        )

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
        prices = self.current_price * np.exp(
            (self.risk_free_rate - 0.5 * volatility**2) * T
            + volatility * np.sqrt(T) * np.random.randn(num_samples)
        )

        payoffs = self.calculate_payoff(prices)
        total_prob = np.sum(payoffs > 0) / num_samples

        return total_prob


def create_preset_strategy(
    strategy_type: OptionStrategy,
    symbol: str,
    current_price: float,
    expiration: datetime,
    **kwargs,
) -> OptionsStrategyBuilder:
    """
    Create a preset options strategy

    Args:
        strategy_type: Type of strategy to create
        symbol: Underlying symbol
        current_price: Current stock price
        expiration: Expiration date
        **kwargs: Strategy-specific parameters

    Returns:
        OptionsStrategyBuilder with legs added
    """

    builder = OptionsStrategyBuilder(symbol)
    builder.current_price = current_price

    if strategy_type == OptionStrategy.COVERED_CALL:
        # Long 100 shares + Short 1 OTM call
        strike = kwargs.get("strike", current_price * 1.05)
        builder.add_leg(OptionType.CALL, strike, expiration, -1)

    elif strategy_type == OptionStrategy.CASH_SECURED_PUT:
        # Short 1 OTM put
        strike = kwargs.get("strike", current_price * 0.95)
        builder.add_leg(OptionType.PUT, strike, expiration, -1)

    elif strategy_type == OptionStrategy.PROTECTIVE_PUT:
        # Long stock + Long 1 OTM put
        strike = kwargs.get("strike", current_price * 0.95)
        builder.add_leg(OptionType.PUT, strike, expiration, 1)

    elif strategy_type == OptionStrategy.VERTICAL_CALL_SPREAD:
        # Long lower strike call + Short higher strike call
        long_strike = kwargs.get("long_strike", current_price)
        short_strike = kwargs.get("short_strike", current_price * 1.05)
        builder.add_leg(OptionType.CALL, long_strike, expiration, 1)
        builder.add_leg(OptionType.CALL, short_strike, expiration, -1)

    elif strategy_type == OptionStrategy.VERTICAL_PUT_SPREAD:
        # Long higher strike put + Short lower strike put
        long_strike = kwargs.get("long_strike", current_price)
        short_strike = kwargs.get("short_strike", current_price * 0.95)
        builder.add_leg(OptionType.PUT, long_strike, expiration, 1)
        builder.add_leg(OptionType.PUT, short_strike, expiration, -1)

    elif strategy_type == OptionStrategy.IRON_CONDOR:
        # OTM put spread + OTM call spread
        put_short = kwargs.get("put_short_strike", current_price * 0.95)
        put_long = kwargs.get("put_long_strike", current_price * 0.90)
        call_short = kwargs.get("call_short_strike", current_price * 1.05)
        call_long = kwargs.get("call_long_strike", current_price * 1.10)

        builder.add_leg(OptionType.PUT, put_long, expiration, 1)
        builder.add_leg(OptionType.PUT, put_short, expiration, -1)
        builder.add_leg(OptionType.CALL, call_short, expiration, -1)
        builder.add_leg(OptionType.CALL, call_long, expiration, 1)

    elif strategy_type == OptionStrategy.BUTTERFLY_SPREAD:
        # 1 lower strike + 2 middle strike + 1 higher strike (all same type)
        opt_type = kwargs.get("option_type", OptionType.CALL)
        lower = kwargs.get("lower_strike", current_price * 0.95)
        middle = kwargs.get("middle_strike", current_price)
        upper = kwargs.get("upper_strike", current_price * 1.05)

        builder.add_leg(opt_type, lower, expiration, 1)
        builder.add_leg(opt_type, middle, expiration, -2)
        builder.add_leg(opt_type, upper, expiration, 1)

    elif strategy_type == OptionStrategy.STRADDLE:
        # Long 1 call + Long 1 put (same strike, ATM)
        strike = kwargs.get("strike", current_price)
        builder.add_leg(OptionType.CALL, strike, expiration, 1)
        builder.add_leg(OptionType.PUT, strike, expiration, 1)

    elif strategy_type == OptionStrategy.STRANGLE:
        # Long 1 OTM call + Long 1 OTM put
        call_strike = kwargs.get("call_strike", current_price * 1.05)
        put_strike = kwargs.get("put_strike", current_price * 0.95)
        builder.add_leg(OptionType.CALL, call_strike, expiration, 1)
        builder.add_leg(OptionType.PUT, put_strike, expiration, 1)

    elif strategy_type == OptionStrategy.COLLAR:
        # Long stock + Long OTM put + Short OTM call
        put_strike = kwargs.get("put_strike", current_price * 0.95)
        call_strike = kwargs.get("call_strike", current_price * 1.05)
        builder.add_leg(OptionType.PUT, put_strike, expiration, 1)
        builder.add_leg(OptionType.CALL, call_strike, expiration, -1)

    return builder
