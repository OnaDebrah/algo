"""
Options Pricing Engine
Provides multiple pricing methodologies for options valuation including
Black-Scholes, Binomial Tree, Monte Carlo, and Finite Difference methods.
"""

import logging
from typing import List

import pandas as pd

from ...options.pricers.binomial_tree import BinomialTreeModel
from ...options.pricers.black_scholes import BlackScholesModel
from ...options.pricers.finite_difference import FiniteDifferenceModel
from ...options.pricers.models import ExerciseType, Greeks, OptionType, PricingModel, PricingResult
from ...options.pricers.monte_carlo import MonteCarloModel

logger = logging.getLogger(__name__)


class OptionsPricingEngine:
    """Main options pricing engine with multiple methodologies"""

    def __init__(self):
        self.models = {
            PricingModel.BLACK_SCHOLES: BlackScholesModel(),
            PricingModel.BINOMIAL_TREE: BinomialTreeModel(),
            PricingModel.MONTE_CARLO: MonteCarloModel(),
            PricingModel.FINITE_DIFFERENCE: FiniteDifferenceModel(),
        }

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        exercise_type: ExerciseType = ExerciseType.EUROPEAN,
        q: float = 0.0,
        model: PricingModel = PricingModel.BLACK_SCHOLES,
        **kwargs,
    ) -> PricingResult:
        """
        Price an option using specified model

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: Option type (Call/Put)
            exercise_type: Exercise style (European/American)
            q: Dividend yield
            model: Pricing model to use
            **kwargs: Additional model-specific parameters

        Returns:
            PricingResult containing price, greeks, and metadata
        """

        if model == PricingModel.BLACK_SCHOLES:
            if exercise_type == ExerciseType.AMERICAN:
                logger.warning(
                    "Black-Scholes model is for European options only. Consider using Binomial Tree or Finite Difference for American options."
                )

            price = self.models[model].price(S, K, T, r, sigma, option_type, q)
            greeks = self.models[model].greeks(S, K, T, r, sigma, option_type, q)

            return PricingResult(price=price, model=model, greeks=greeks)

        elif model == PricingModel.BINOMIAL_TREE:
            steps = kwargs.get("steps", 100)
            price = self.models[model].price(S, K, T, r, sigma, option_type, exercise_type, q, steps)

            if kwargs.get("calculate_greeks", False):
                greeks = self.models[model].greeks(S, K, T, r, sigma, option_type, exercise_type, q, steps)
            else:
                greeks = None

            return PricingResult(price=price, model=model, greeks=greeks, convergence={"steps": steps})

        elif model == PricingModel.MONTE_CARLO:
            num_simulations = kwargs.get("num_simulations", 10000)
            num_steps = kwargs.get("num_steps", 252)
            seed = kwargs.get("seed", None)
            antithetic = kwargs.get("antithetic", True)
            control_variate = kwargs.get("control_variate", False)

            result = self.models[model].price(
                S, K, T, r, sigma, option_type, exercise_type, q, num_simulations, num_steps, seed, antithetic, control_variate
            )

            if kwargs.get("calculate_greeks", False):
                result.greeks = self.models[model].greeks(S, K, T, r, sigma, option_type, exercise_type, q, num_simulations, num_steps)

            return result

        elif model == PricingModel.FINITE_DIFFERENCE:
            S_max = kwargs.get("S_max", None)
            num_S = kwargs.get("num_S", 100)
            num_t = kwargs.get("num_t", 100)

            price = self.models[model].price(S, K, T, r, sigma, option_type, exercise_type, q, S_max, num_S, num_t)

            # Greeks can be calculated numerically if needed
            greeks = None
            if kwargs.get("calculate_greeks", False):
                epsilon = S * 0.01

                # Delta and Gamma using finite differences
                price_up = self.models[model].price(S + epsilon, K, T, r, sigma, option_type, exercise_type, q, S_max, num_S, num_t)
                price_down = self.models[model].price(S - epsilon, K, T, r, sigma, option_type, exercise_type, q, S_max, num_S, num_t)

                delta = (price_up - price_down) / (2 * epsilon)
                gamma = (price_up - 2 * price + price_down) / (epsilon**2)

                # Vega
                vega_shift = sigma * 0.01
                price_vol_up = self.models[model].price(S, K, T, r, sigma + vega_shift, option_type, exercise_type, q, S_max, num_S, num_t)
                price_vol_down = self.models[model].price(S, K, T, r, sigma - vega_shift, option_type, exercise_type, q, S_max, num_S, num_t)
                vega = (price_vol_up - price_vol_down) / (2 * vega_shift) / 100

                # Theta
                if T > 1 / 365:
                    price_time = self.models[model].price(S, K, T - 1 / 365, r, sigma, option_type, exercise_type, q, S_max, num_S, num_t)
                    theta = (price_time - price) * 365
                else:
                    theta = 0

                # Rho
                rho_shift = 0.0001
                price_rate_up = self.models[model].price(S, K, T, r + rho_shift, sigma, option_type, exercise_type, q, S_max, num_S, num_t)
                price_rate_down = self.models[model].price(S, K, T, r - rho_shift, sigma, option_type, exercise_type, q, S_max, num_S, num_t)
                rho = (price_rate_up - price_rate_down) / (2 * rho_shift) / 100

                greeks = Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)

            return PricingResult(price=price, model=model, greeks=greeks, convergence={"S_grid_points": num_S, "time_steps": num_t})

        else:
            raise ValueError(f"Unsupported pricing model: {model}")

    def compare_models(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        exercise_type: ExerciseType = ExerciseType.EUROPEAN,
        q: float = 0.0,
        models: List[PricingModel] = None,
    ) -> pd.DataFrame:
        """Compare prices across different models"""

        if models is None:
            models = list(PricingModel)

        results = []
        for model in models:
            try:
                result = self.price(S, K, T, r, sigma, option_type, exercise_type, q, model)
                results.append(
                    {
                        "Model": model.value,
                        "Price": result.price,
                        "Delta": result.greeks.delta if result.greeks else None,
                        "Gamma": result.greeks.gamma if result.greeks else None,
                        "Theta": result.greeks.theta if result.greeks else None,
                        "Vega": result.greeks.vega if result.greeks else None,
                        "Rho": result.greeks.rho if result.greeks else None,
                    }
                )
            except Exception as e:
                logger.error(f"Error with model {model.value}: {e}")
                results.append({"Model": model.value, "Price": None, "Error": str(e)})

        return pd.DataFrame(results)

    def implied_volatility(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: OptionType,
        q: float = 0.0,
        model: PricingModel = PricingModel.BLACK_SCHOLES,
        **kwargs,
    ) -> float:
        """
        Calculate implied volatility using specified model

        Only Black-Scholes model is currently supported for IV calculation
        """

        if model == PricingModel.BLACK_SCHOLES:
            return self.models[model].implied_volatility(market_price, S, K, T, r, option_type, q, **kwargs)
        else:
            raise NotImplementedError(f"Implied volatility calculation not implemented for {model.value}")


# Convenience functions
def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType, q: float = 0.0) -> float:
    """Quick Black-Scholes price calculation"""
    return BlackScholesModel.price(S, K, T, r, sigma, option_type, q)


def black_scholes_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType, q: float = 0.0) -> Greeks:
    """Quick Black-Scholes Greeks calculation"""
    return BlackScholesModel.greeks(S, K, T, r, sigma, option_type, q)


def implied_volatility(market_price: float, S: float, K: float, T: float, r: float, option_type: OptionType, q: float = 0.0) -> float:
    """Quick implied volatility calculation"""
    return BlackScholesModel.implied_volatility(market_price, S, K, T, r, option_type, q)
