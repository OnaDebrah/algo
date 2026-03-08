import numpy as np

from ...options.pricers.models import ExerciseType, Greeks, OptionType


class BinomialTreeModel:
    """Binomial tree model for American and European options"""

    @staticmethod
    def price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        exercise_type: ExerciseType = ExerciseType.EUROPEAN,
        q: float = 0.0,
        steps: int = 100,
    ) -> float:
        """Price option using binomial tree"""

        dt = T / steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)
        discount = np.exp(-r * dt)

        # Initialize asset prices at maturity
        asset_prices = np.zeros(steps + 1)
        for i in range(steps + 1):
            asset_prices[i] = S * (u ** (steps - i)) * (d**i)

        # Initialize option values at maturity
        if option_type == OptionType.CALL:
            option_values = np.maximum(asset_prices - K, 0)
        else:
            option_values = np.maximum(K - asset_prices, 0)

        # Backward induction
        for step in range(steps - 1, -1, -1):
            for i in range(step + 1):
                # Calculate continuation value
                continuation = discount * (p * option_values[i] + (1 - p) * option_values[i + 1])

                if exercise_type == ExerciseType.AMERICAN:
                    # Check for early exercise
                    current_asset = S * (u ** (step - i)) * (d**i)
                    if option_type == OptionType.CALL:
                        exercise_value = max(current_asset - K, 0)
                    else:
                        exercise_value = max(K - current_asset, 0)
                    option_values[i] = max(continuation, exercise_value)
                else:
                    option_values[i] = continuation

        return option_values[0]

    @staticmethod
    def greeks(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        exercise_type: ExerciseType = ExerciseType.EUROPEAN,
        q: float = 0.0,
        steps: int = 100,
    ) -> Greeks:
        """Calculate Greeks using finite differences on binomial tree"""

        epsilon = S * 0.01  # 1% price change

        # Delta
        price_up = BinomialTreeModel.price(S + epsilon, K, T, r, sigma, option_type, exercise_type, q, steps)
        price_down = BinomialTreeModel.price(S - epsilon, K, T, r, sigma, option_type, exercise_type, q, steps)
        delta = (price_up - price_down) / (2 * epsilon)

        # Gamma
        gamma = (price_up - 2 * BinomialTreeModel.price(S, K, T, r, sigma, option_type, exercise_type, q, steps) + price_down) / (epsilon**2)

        # Vega
        vega_shift = sigma * 0.01  # 1% volatility change
        price_vol_up = BinomialTreeModel.price(S, K, T, r, sigma + vega_shift, option_type, exercise_type, q, steps)
        price_vol_down = BinomialTreeModel.price(S, K, T, r, sigma - vega_shift, option_type, exercise_type, q, steps)
        vega = (price_vol_up - price_vol_down) / (2 * vega_shift) / 100  # Per 1%

        # Theta
        time_shift = 1 / 365  # One day
        if T > time_shift:
            price_time = BinomialTreeModel.price(S, K, T - time_shift, r, sigma, option_type, exercise_type, q, steps)
            theta = (price_time - BinomialTreeModel.price(S, K, T, r, sigma, option_type, exercise_type, q, steps)) / time_shift
        else:
            theta = 0

        # Rho
        rho_shift = 0.0001  # 1bp rate change
        price_rate_up = BinomialTreeModel.price(S, K, T, r + rho_shift, sigma, option_type, exercise_type, q, steps)
        price_rate_down = BinomialTreeModel.price(S, K, T, r - rho_shift, sigma, option_type, exercise_type, q, steps)
        rho = (price_rate_up - price_rate_down) / (2 * rho_shift) / 100  # Per 1%

        return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)
