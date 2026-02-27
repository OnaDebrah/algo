"""
QuantLib-based options hedging calculator
Provides delta hedging calculations and Greeks analysis
Based on QuantLib implementation examples [citation:1][citation:6]
"""

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Optional

import QuantLib as ql

logger = logging.getLogger(__name__)


@dataclass
class OptionContract:
    """Option contract parameters"""

    symbol: str
    option_type: str  # 'call' or 'put'
    strike: float
    expiration: date
    style: str = "european"  # or 'american'
    position: int = 1  # +1 for long, -1 for short


@dataclass
class HedgePosition:
    """Hedge position details"""

    instrument: str  # 'option', 'stock', 'future', etc.
    quantity: float
    delta: float
    value: float
    metadata: Dict


class QuantLibHedgeEngine:
    """
    QuantLib-based hedging engine for options positions

    Provides:
    - Option pricing using Black-Scholes
    - Greeks calculation (delta, gamma, vega, theta, rho)
    - Delta hedging recommendations
    - Position sizing for portfolio hedging
    """

    def __init__(self):
        self.calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        self.day_count = ql.Actual365Fixed()

    def setup_market(
        self, spot_price: float, risk_free_rate: float, dividend_yield: float = 0.0, volatility: float = 0.2, valuation_date: Optional[date] = None
    ):
        """
        Setup market conditions for pricing
        Based on QuantLib market setup examples [citation:6]
        """
        if valuation_date is None:
            valuation_date = date.today()

        # Convert to QuantLib date
        ql_date = ql.Date(valuation_date.day, valuation_date.month, valuation_date.year)
        ql.Settings.instance().evaluationDate = ql_date

        # Create market handles
        self.spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))

        self.rate_handle = ql.YieldTermStructureHandle(ql.FlatForward(ql_date, risk_free_rate, self.day_count))

        self.dividend_handle = ql.YieldTermStructureHandle(ql.FlatForward(ql_date, dividend_yield, self.day_count))

        self.vol_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(ql_date, self.calendar, volatility, self.day_count))

        # Create Black-Scholes-Merton process
        self.bsm_process = ql.BlackScholesMertonProcess(self.spot_handle, self.dividend_handle, self.rate_handle, self.vol_handle)

        logger.info(f"Market setup complete: Spot={spot_price}, Rate={risk_free_rate}, Vol={volatility}")

    def price_option(self, option: OptionContract) -> Dict:
        """
        Price a single option and calculate all Greeks
        Based on QuantLib FX option pricing example [citation:4]
        """
        # Convert dates to QuantLib
        today = ql.Date.todaysDate()
        expiry = ql.Date(option.expiration.day, option.expiration.month, option.expiration.year)

        # Create option
        payoff = ql.PlainVanillaPayoff(ql.Option.Call if option.option_type.lower() == "call" else ql.Option.Put, option.strike)

        if option.style.lower() == "european":
            exercise = ql.EuropeanExercise(expiry)
        else:
            exercise = ql.AmericanExercise(today, expiry)

        ql_option = ql.VanillaOption(payoff, exercise)

        # Set pricing engine
        if option.style.lower() == "european":
            engine = ql.AnalyticEuropeanEngine(self.bsm_process)
        else:
            # Use binomial tree for American options
            engine = ql.BinomialVanillaEngine(self.bsm_process, "crr", 100)

        ql_option.setPricingEngine(engine)

        # Calculate price and Greeks
        price = ql_option.NPV()
        delta = ql_option.delta()
        gamma = ql_option.gamma()
        vega = ql_option.vega()
        theta = ql_option.theta()
        rho = ql_option.rho()

        # Get implied volatility
        implied_vol = ql_option.impliedVolatility(price, self.bsm_process)

        # Adjust theta for daily decay
        theta_daily = theta / 365.0  # Daily time decay [citation:4]

        return {
            "price": price,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "theta_daily": theta_daily,
            "rho": rho,
            "implied_vol": implied_vol,
            "days_to_expiry": (option.expiration - date.today()).days,
        }

    def calculate_delta_hedge(self, option: OptionContract, portfolio_value: float, hedge_ratio: float = 1.0) -> Dict:
        """
        Calculate delta hedge position for an option

        Based on delta hedging methodology [citation:3][citation:6]

        Args:
            option: Option contract to hedge
            portfolio_value: Total portfolio value
            hedge_ratio: Fraction of delta to hedge (1.0 = full hedge)
        """
        # Price option and get Greeks
        option_data = self.price_option(option)

        # Calculate delta exposure
        delta = option_data["delta"] * option.position * hedge_ratio

        # Calculate shares needed to hedge
        spot = self.spot_handle.value()
        shares_to_hedge = -delta  # Opposite position

        # Hedge value
        hedge_value = shares_to_hedge * spot

        # Calculate hedge ratio relative to portfolio
        hedge_percentage = abs(hedge_value / portfolio_value)

        # Calculate cost of hedge
        option_premium = option_data["price"] * abs(option.position)

        # Greeks for the hedged position
        hedged_delta = 0  # Delta neutral after hedge
        hedged_gamma = option_data["gamma"] * option.position
        hedged_vega = option_data["vega"] * option.position

        # For put spreads or collars, we'd need multiple options
        # This is simplified single-option hedge

        return {
            "option": {"type": option.option_type, "strike": option.strike, "expiration": option.expiration.isoformat(), "position": option.position},
            "option_greeks": {
                "delta": option_data["delta"],
                "gamma": option_data["gamma"],
                "vega": option_data["vega"],
                "theta_daily": option_data["theta_daily"],
            },
            "hedge_recommendation": {
                "instrument": "stock",
                "shares": shares_to_hedge,
                "value": hedge_value,
                "hedge_percentage": hedge_percentage,
                "cost": option_premium,
            },
            "hedged_position_greeks": {"delta": hedged_delta, "gamma": hedged_gamma, "vega": hedged_vega},
            "analysis": {
                "break_even": self._calculate_break_even(option, option_data["price"]),
                "max_loss": self._calculate_max_loss(option, option_data["price"]),
                "time_decay_cost": option_data["theta_daily"] * abs(option.position),
            },
        }

    def calculate_portfolio_hedge(
        self,
        portfolio_value: float,
        portfolio_beta: float,
        index_symbol: str,
        index_price: float,
        hedge_percentage: float = 0.5,
        days_to_expiry: int = 90,
    ) -> Dict:
        """
        Calculate index put hedge for entire portfolio

        Based on beta-weighted hedging approach

        Args:
            portfolio_value: Total portfolio value
            portfolio_beta: Portfolio beta relative to index
            index_symbol: Underlying index (e.g., SPY)
            index_price: Current index price
            hedge_percentage: Percentage of portfolio to hedge
            days_to_expiry: Days until option expiry
        """
        # Calculate beta-adjusted exposure
        beta_adjusted_exposure = portfolio_value * portfolio_beta * hedge_percentage

        # Number of index units to hedge
        index_units = beta_adjusted_exposure / index_price

        # For index options, typically 100 shares per contract
        contracts_needed = index_units / 100

        # Suggest strike prices for put hedge
        strikes = {
            "conservative": round(index_price * 0.95, 2),  # 5% OTM
            "moderate": round(index_price * 0.90, 2),  # 10% OTM
            "aggressive": round(index_price * 0.85, 2),  # 15% OTM
        }

        # Estimate put prices (simplified - would need full pricing)
        put_prices = {}
        for level, strike in strikes.items():
            # Create put option
            expiry = date.today() + timedelta(days=days_to_expiry)
            put = OptionContract(symbol=index_symbol, option_type="put", strike=strike, expiration=expiry, style="european")

            # Price it
            option_data = self.price_option(put)
            put_prices[level] = option_data["price"] * 100  # Per contract

        return {
            "portfolio": {"value": portfolio_value, "beta": portfolio_beta, "hedge_percentage": hedge_percentage},
            "hedge_calculation": {"beta_adjusted_exposure": beta_adjusted_exposure, "index_units": index_units, "contracts_needed": contracts_needed},
            "option_suggestions": {
                "expiry": (date.today() + timedelta(days=days_to_expiry)).isoformat(),
                "strikes": strikes,
                "estimated_put_prices": put_prices,
                "total_cost": {level: price * contracts_needed for level, price in put_prices.items()},
            },
            "hedge_effectiveness": {
                "protection_level": hedge_percentage,
                "max_protection": f"{(1 - 0.85) * 100:.0f}% downside" if hedge_percentage == 1.0 else "Partial",
                "cost_bps": min(put_prices.values()) * contracts_needed / portfolio_value * 10000,  # Basis points
            },
        }

    def _calculate_break_even(self, option: OptionContract, premium: float) -> float:
        """Calculate break-even price for option position"""
        if option.option_type == "call":
            return option.strike + premium if option.position > 0 else option.strike - premium
        else:  # put
            return option.strike - premium if option.position > 0 else option.strike + premium

    def _calculate_max_loss(self, option: OptionContract, premium: float) -> float:
        """Calculate maximum loss for option position"""
        if option.position > 0:  # Long option
            return premium * option.position  # Limited to premium paid
        else:  # Short option
            # For short options, theoretically unlimited
            return float("inf") if option.option_type == "call" else option.strike * option.position

    def calculate_greeks_sensitivity(
        self, option: OptionContract, spot_shock: float = 0.01, vol_shock: float = 0.01, rate_shock: float = 0.001
    ) -> Dict:
        """
        Calculate Greeks sensitivities to market changes
        Using bump-and-revalue method [citation:4]
        """
        base = self.price_option(option)

        sensitivities = {}

        # Delta is already first derivative, but let's verify
        original_spot = self.spot_handle.value()

        # Bump spot up
        self.spot_handle.setValue(original_spot * (1 + spot_shock))
        price_up = self.price_option(option)["price"]

        # Bump spot down
        self.spot_handle.setValue(original_spot * (1 - spot_shock))
        price_down = self.price_option(option)["price"]

        # Reset spot
        self.spot_handle.setValue(original_spot)

        # Calculate gamma numerically
        numerical_gamma = (price_up - 2 * base["price"] + price_down) / (original_spot * spot_shock) ** 2

        sensitivities["gamma_verification"] = {
            "model_gamma": base["gamma"],
            "numerical_gamma": numerical_gamma,
            "difference": abs(base["gamma"] - numerical_gamma),
        }

        return sensitivities

    def suggest_hedge_structure(
        self, portfolio_value: float, portfolio_beta: float, index_price: float, crash_probability: float, crash_intensity: str
    ) -> Dict:
        """
        Suggest optimal hedge structure based on crash probability and intensity

        Args:
            portfolio_value: Total portfolio value
            portfolio_beta: Portfolio beta
            index_price: Current index price
            crash_probability: From ML models
            crash_intensity: 'mild', 'moderate', 'severe'
        """

        if crash_intensity == "mild" or crash_probability < 0.3:
            # Light hedge: covered calls, slight reduction
            return {
                "strategy": "covered_calls",
                "description": "Sell OTM covered calls for income",
                "implementation": "Sell calls at 5-10% above current price",
                "cost": 0,
                "protection": "Limited",
            }

        elif crash_intensity == "moderate" or crash_probability < 0.6:
            # Cost-effective hedge: put spreads
            hedge = self.calculate_portfolio_hedge(portfolio_value, portfolio_beta, "SPY", index_price, hedge_percentage=0.5, days_to_expiry=90)

            return {
                "strategy": "put_spread",
                "description": "Buy put spreads for cost-effective protection",
                "implementation": "Buy ATM puts, sell OTM puts to reduce cost",
                "cost": hedge["option_suggestions"]["estimated_put_prices"]["moderate"] * hedge["hedge_calculation"]["contracts_needed"],
                "protection": "Up to 10% downside protection",
                "details": hedge,
            }

        else:  # Severe crash expected
            # Full tail-risk hedge
            hedge = self.calculate_portfolio_hedge(
                portfolio_value,
                portfolio_beta,
                "SPY",
                index_price,
                hedge_percentage=1.0,
                days_to_expiry=60,  # Shorter expiry for tail risk
            )

            return {
                "strategy": "tail_risk",
                "description": "Buy OTM puts for catastrophic protection",
                "implementation": "Buy 15-20% OTM puts, consider VIX calls",
                "cost": hedge["option_suggestions"]["estimated_put_prices"]["aggressive"] * hedge["hedge_calculation"]["contracts_needed"],
                "protection": "Unlimited downside protection",
                "details": hedge,
            }
