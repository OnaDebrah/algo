from typing import Dict, Optional

import numpy as np

from ....strategies.options_strategies import OptionType
from ...options.pricers.black_scholes import BlackScholesModel
from ...options.pricers.models import ExerciseType, Greeks, PricingModel, PricingResult


class MonteCarloModel:
    """Monte Carlo simulation for option pricing"""

    # @staticmethod
    # def price(
    #     S: float,
    #     K: float,
    #     T: float,
    #     r: float,
    #     sigma: float,
    #     option_type: OptionType,
    #     exercise_type: ExerciseType = ExerciseType.EUROPEAN,
    #     q: float = 0.0,
    #     num_simulations: int = 10000,
    #     num_steps: int = 252,
    #     seed: Optional[int] = None,
    #     antithetic: bool = True,
    #     control_variate: bool = False,
    # ) -> PricingResult:
    #     """Price option using Monte Carlo simulation"""
    #
    #     if seed is not None:
    #         np.random.seed(seed)
    #
    #     dt = T / num_steps
    #     drift = (r - q - 0.5 * sigma**2) * dt
    #     vol = sigma * np.sqrt(dt)
    #
    #     # Generate random paths
    #     if antithetic:
    #         # Use antithetic variates for variance reduction
    #         Z = np.random.normal(0, 1, (num_simulations // 2, num_steps))
    #         Z = np.vstack([Z, -Z])
    #         num_simulations = len(Z)
    #     else:
    #         Z = np.random.normal(0, 1, (num_simulations, num_steps))
    #
    #     # Simulate price paths
    #     S_t = np.zeros((num_simulations, num_steps + 1))
    #     S_t[:, 0] = S
    #
    #     for t in range(1, num_steps + 1):
    #         S_t[:, t] = S_t[:, t - 1] * np.exp(drift + vol * Z[:, t - 1])
    #
    #     if exercise_type == ExerciseType.EUROPEAN:
    #         # European option - payoff at maturity
    #         if option_type == OptionType.CALL:
    #             payoffs = np.maximum(S_t[:, -1] - K, 0)
    #         else:
    #             payoffs = np.maximum(K - S_t[:, -1], 0)
    #
    #         # Discount to present value
    #         prices = np.exp(-r * T) * payoffs
    #
    #     else:
    #         # American option - need to check early exercise (simplified approach)
    #         # This is a basic Longstaff-Schwartz approach
    #         cash_flows = np.zeros(num_simulations)
    #         exercise_times = np.full(num_simulations, num_steps)
    #
    #         # Backward induction
    #         for t in range(num_steps, 0, -1):
    #             in_the_money = None
    #             if option_type == OptionType.CALL:
    #                 in_the_money = S_t[:, t] > K
    #             else:
    #                 in_the_money = S_t[:, t] < K
    #
    #             if not np.any(in_the_money):
    #                 continue
    #
    #             # Regression for continuation value
    #             X = S_t[in_the_money, t]
    #             Y = cash_flows[in_the_money] * np.exp(-r * dt * (exercise_times[in_the_money] - t))
    #
    #             if len(X) > 3:  # Need enough points for regression
    #                 # Polynomial regression
    #                 poly = np.polyfit(X, Y, 2)
    #                 continuation = np.polyval(poly, X)
    #
    #                 # Immediate exercise value
    #                 if option_type == OptionType.CALL:
    #                     exercise = np.maximum(X - K, 0)
    #                 else:
    #                     exercise = np.maximum(K - X, 0)
    #
    #                 # Check if exercise is optimal
    #                 exercise_optimal = exercise > continuation
    #                 idx_in_the_money = np.where(in_the_money)[0]
    #
    #                 for i, idx in enumerate(idx_in_the_money[exercise_optimal]):
    #                     cash_flows[idx] = exercise[i]
    #                     exercise_times[idx] = t
    #
    #         # Calculate final payoffs
    #         final_mask = exercise_times == num_steps
    #         if option_type == OptionType.CALL:
    #             cash_flows[final_mask] = np.maximum(S_t[final_mask, -1] - K, 0)
    #         else:
    #             cash_flows[final_mask] = np.maximum(K - S_t[final_mask, -1], 0)
    #
    #         # Discount to present value
    #         discount_factors = np.exp(-r * dt * exercise_times)
    #         prices = cash_flows * discount_factors
    #
    #     # Calculate statistics
    #     mean_price = np.mean(prices)
    #     std_error = np.std(prices) / np.sqrt(num_simulations)
    #
    #     # Control variate (using Black-Scholes as control)
    #     if control_variate and exercise_type == ExerciseType.EUROPEAN:
    #         bs_price = BlackScholesModel.price(S, K, T, r, sigma, option_type, q)
    #
    #         # Generate correlated sample (simplified)
    #         control_payoffs = None
    #         if option_type == OptionType.CALL:
    #             control_payoffs = np.maximum(S_t[:, -1] - K, 0)
    #         else:
    #             control_payoffs = np.maximum(K - S_t[:, -1], 0)
    #
    #         control_prices = np.exp(-r * T) * control_payoffs
    #
    #         # Calculate optimal coefficient
    #         covariance = np.cov(prices, control_prices)[0, 1]
    #         variance = np.var(control_prices)
    #
    #         if variance > 0:
    #             beta = covariance / variance
    #             control_mean = np.mean(control_prices)
    #             prices = prices - beta * (control_prices - control_mean)
    #             mean_price = np.mean(prices)
    #
    #     return PricingResult(
    #         price=mean_price,
    #         model=PricingModel.MONTE_CARLO,
    #         error_estimate=std_error,
    #         convergence={
    #             "num_simulations": num_simulations,
    #             "std_error": std_error,
    #             "confidence_interval_95": (mean_price - 1.96 * std_error, mean_price + 1.96 * std_error),
    #         },
    #     )

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
        num_simulations: int = 10000,
        num_steps: int = 252,
        seed: Optional[int] = None,
        antithetic: bool = True,
        control_variate: bool = False,
    ) -> PricingResult:
        """Price option using Monte Carlo simulation with variance reduction"""

        if seed is not None:
            np.random.seed(seed)

        dt = T / num_steps
        drift = (r - q - 0.5 * sigma**2) * dt
        vol = sigma * np.sqrt(dt)

        # Generate random paths with optional antithetic variates
        if antithetic:
            Z = np.random.normal(0, 1, (num_simulations // 2, num_steps))
            Z = np.vstack([Z, -Z])
            num_simulations = len(Z)
        else:
            Z = np.random.normal(0, 1, (num_simulations, num_steps))

        # Simulate price paths
        S_t = np.zeros((num_simulations, num_steps + 1))
        S_t[:, 0] = S

        for t in range(1, num_steps + 1):
            S_t[:, t] = S_t[:, t - 1] * np.exp(drift + vol * Z[:, t - 1])

        # Calculate payoffs based on exercise type
        if exercise_type == ExerciseType.EUROPEAN:
            # European option - payoff at maturity
            if option_type == OptionType.CALL:
                payoffs = np.maximum(S_t[:, -1] - K, 0)
            else:
                payoffs = np.maximum(K - S_t[:, -1], 0)

            # Discount to present value
            prices = np.exp(-r * T) * payoffs

        else:
            # American option with Longstaff-Schwartz
            prices = MonteCarloModel._price_american(S_t, K, r, dt, option_type, num_simulations, num_steps)

        # Calculate basic statistics
        mean_price = np.mean(prices)
        std_error = np.std(prices) / np.sqrt(num_simulations)

        # Control variate for variance reduction (European only)
        if control_variate and exercise_type == ExerciseType.EUROPEAN:
            # NOW USING bs_price!
            bs_price = BlackScholesModel.price(S, K, T, r, sigma, option_type, q).price

            # Get the discounted payoffs (same as before)
            if option_type == OptionType.CALL:
                control_payoffs = np.maximum(S_t[:, -1] - K, 0)
            else:
                control_payoffs = np.maximum(K - S_t[:, -1], 0)

            control_prices = np.exp(-r * T) * control_payoffs

            # Calculate optimal coefficient
            covariance = np.cov(prices, control_prices)[0, 1]
            variance = np.var(control_prices)

            if variance > 1e-12:  # Avoid division by zero
                beta = covariance / variance

                # Apply control variate correction - NOW USING bs_price as true mean!
                prices_corrected = prices - beta * (control_prices - bs_price)

                # Recalculate statistics
                mean_price = np.mean(prices_corrected)
                std_error = np.std(prices_corrected) / np.sqrt(num_simulations)

                # Store control variate info for diagnostics
                control_info = {
                    "beta": float(beta),
                    "correlation": float(covariance / (np.std(prices) * np.std(control_prices) + 1e-12)),
                    "variance_reduction": float(1 - np.var(prices_corrected) / (np.var(prices) + 1e-12)),
                    "bs_price": float(bs_price),
                }
            else:
                control_info = {"error": "Zero variance in control variate"}
        else:
            control_info = None

        # Calculate confidence intervals
        conf_95_lower = mean_price - 1.96 * std_error
        conf_95_upper = mean_price + 1.96 * std_error

        # Calculate convergence diagnostics
        convergence = {
            "num_simulations": num_simulations,
            "std_error": float(std_error),
            "relative_error": float(std_error / mean_price if mean_price > 1e-12 else 1.0),
            "confidence_interval_95": (float(conf_95_lower), float(conf_95_upper)),
            "antithetic_used": antithetic,
            "control_variate_used": control_variate,
        }

        if control_info:
            convergence["control_variate"] = control_info

        return PricingResult(
            price=float(mean_price),
            model=PricingModel.MONTE_CARLO,
            error_estimate=float(std_error),
            convergence=convergence,
        )

    @staticmethod
    def _price_american(
        S_t: np.ndarray,
        K: float,
        r: float,
        dt: float,
        option_type: OptionType,
        num_simulations: int,
        num_steps: int,
    ) -> np.ndarray:
        """
        Price American options using Longstaff-Schwartz regression

        This is a separate method to keep the main price method clean
        """
        cash_flows = np.zeros(num_simulations)
        exercise_times = np.full(num_simulations, num_steps)

        # Backward induction
        for t in range(num_steps, 0, -1):
            # Find in-the-money paths
            if option_type == OptionType.CALL:
                in_the_money = S_t[:, t] > K
                exercise_value = np.maximum(S_t[:, t] - K, 0)
            else:
                in_the_money = S_t[:, t] < K
                exercise_value = np.maximum(K - S_t[:, t], 0)

            if not np.any(in_the_money):
                continue

            # Get paths that are in the money
            X = S_t[in_the_money, t]  # Stock prices for regression
            Y = cash_flows[in_the_money] * np.exp(-r * dt * (exercise_times[in_the_money] - t))

            if len(X) > 5:  # Need enough points for reliable regression
                # Use polynomial regression (degree 2-3 works well)
                try:
                    # Fit polynomial of degree 3 for better accuracy
                    coeffs = np.polyfit(X, Y, 3)
                    continuation_value = np.polyval(coeffs, X)

                    # Check if exercise is optimal
                    exercise_optimal = exercise_value[in_the_money] > continuation_value

                    # Update cash flows for optimal exercise paths
                    idx_in_the_money = np.where(in_the_money)[0]

                    for i, idx in enumerate(idx_in_the_money[exercise_optimal]):
                        cash_flows[idx] = exercise_value[in_the_money][i]
                        exercise_times[idx] = t

                except (np.linalg.LinAlgError, ValueError):
                    # Fall back to simple comparison if regression fails
                    continuation_value = np.mean(Y)  # Simple average as fallback
                    exercise_optimal = exercise_value[in_the_money] > continuation_value

                    idx_in_the_money = np.where(in_the_money)[0]
                    for i, idx in enumerate(idx_in_the_money[exercise_optimal]):
                        cash_flows[idx] = exercise_value[in_the_money][i]
                        exercise_times[idx] = t

        # Calculate final payoffs for paths never exercised
        final_mask = exercise_times == num_steps
        if option_type == OptionType.CALL:
            cash_flows[final_mask] = np.maximum(S_t[final_mask, -1] - K, 0)
        else:
            cash_flows[final_mask] = np.maximum(K - S_t[final_mask, -1], 0)

        # Discount to present value
        discount_factors = np.exp(-r * dt * exercise_times)
        return cash_flows * discount_factors

    @staticmethod
    def price_with_variance_reduction(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        q: float = 0.0,
        num_simulations: int = 10000,
        method: str = "all",
    ) -> Dict:
        """
        Compare different variance reduction techniques

        Args:
            method: "all", "antithetic", "control", "both"

        Returns:
            Dictionary with prices and variances for each method
        """
        results = {}

        # Standard Monte Carlo
        if method in ["all", "standard"]:
            std_result = MonteCarloModel.price(
                S, K, T, r, sigma, option_type, ExerciseType.EUROPEAN, q, num_simulations, 252, antithetic=False, control_variate=False
            )
            results["standard"] = {
                "price": std_result.price,
                "std_error": std_result.error_estimate,
                "variance": std_result.error_estimate**2 * num_simulations,
            }

        # Antithetic variates only
        if method in ["all", "antithetic"]:
            anti_result = MonteCarloModel.price(
                S, K, T, r, sigma, option_type, ExerciseType.EUROPEAN, q, num_simulations, 252, antithetic=True, control_variate=False
            )
            results["antithetic"] = {
                "price": anti_result.price,
                "std_error": anti_result.error_estimate,
                "variance": anti_result.error_estimate**2 * num_simulations,
            }

        # Control variate only
        if method in ["all", "control"]:
            control_result = MonteCarloModel.price(
                S, K, T, r, sigma, option_type, ExerciseType.EUROPEAN, q, num_simulations, 252, antithetic=False, control_variate=True
            )
            results["control"] = {
                "price": control_result.price,
                "std_error": control_result.error_estimate,
                "variance": control_result.error_estimate**2 * num_simulations,
            }

        # Both techniques
        if method in ["all", "both"]:
            both_result = MonteCarloModel.price(
                S, K, T, r, sigma, option_type, ExerciseType.EUROPEAN, q, num_simulations, 252, antithetic=True, control_variate=True
            )
            results["both"] = {
                "price": both_result.price,
                "std_error": both_result.error_estimate,
                "variance": both_result.error_estimate**2 * num_simulations,
            }

        # Calculate variance reduction ratios
        if "standard" in results:
            base_variance = results["standard"]["variance"]
            for name, result in results.items():
                if name != "standard":
                    result["variance_ratio"] = result["variance"] / base_variance
                    result["variance_reduction"] = 1 - result["variance_ratio"]

        return results

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
        num_simulations: int = 10000,
        num_steps: int = 252,
    ) -> Greeks:
        """Calculate Greeks using pathwise derivatives"""

        epsilon = S * 0.01

        # Price at current level
        base_result = MonteCarloModel.price(S, K, T, r, sigma, option_type, exercise_type, q, num_simulations, num_steps)

        # Price with small changes
        price_up = MonteCarloModel.price(S + epsilon, K, T, r, sigma, option_type, exercise_type, q, num_simulations, num_steps).price
        price_down = MonteCarloModel.price(S - epsilon, K, T, r, sigma, option_type, exercise_type, q, num_simulations, num_steps).price

        delta = (price_up - price_down) / (2 * epsilon)
        gamma = (price_up - 2 * base_result.price + price_down) / (epsilon**2)

        # Vega
        vega_shift = sigma * 0.01
        price_vol_up = MonteCarloModel.price(S, K, T, r, sigma + vega_shift, option_type, exercise_type, q, num_simulations, num_steps).price
        price_vol_down = MonteCarloModel.price(S, K, T, r, sigma - vega_shift, option_type, exercise_type, q, num_simulations, num_steps).price
        vega = (price_vol_up - price_vol_down) / (2 * vega_shift) / 100

        # Theta
        if T > 1 / 365:
            price_time = MonteCarloModel.price(S, K, T - 1 / 365, r, sigma, option_type, exercise_type, q, num_simulations, num_steps).price
            theta = (price_time - base_result.price) * 365
        else:
            theta = 0

        # Rho
        rho_shift = 0.0001
        price_rate_up = MonteCarloModel.price(S, K, T, r + rho_shift, sigma, option_type, exercise_type, q, num_simulations, num_steps).price
        price_rate_down = MonteCarloModel.price(S, K, T, r - rho_shift, sigma, option_type, exercise_type, q, num_simulations, num_steps).price
        rho = (price_rate_up - price_rate_down) / (2 * rho_shift) / 100

        return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)
