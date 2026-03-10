import logging
import warnings
from typing import Any, Dict

import nltk
import numpy as np

warnings.filterwarnings("ignore")

nltk.download("vader_lexicon", quiet=True)

logger = logging.getLogger(__name__)


class MonteCarloEngine:
    """
    Generates Monte Carlo simulations of future price paths
    """

    def __init__(self, num_simulations: int = 10000):
        self.num_simulations = num_simulations

    def simulate_paths(
        self, current_price: float, predicted_return: float, predicted_volatility: float, forecast_horizon: int, dt: float = 1 / 252
    ) -> np.ndarray:
        """
        Generate Monte Carlo price paths using Geometric Brownian Motion

        Args:
            current_price: Starting price
            predicted_return: Expected daily return from ML model
            predicted_volatility: Expected volatility from ML model
            forecast_horizon: Number of days to simulate
            dt: Time step (1/252 for daily)

        Returns:
            Array of shape (num_simulations, forecast_horizon) with price paths
        """
        # Initialize paths
        paths = np.zeros((self.num_simulations, forecast_horizon))
        paths[:, 0] = current_price

        # Generate random shocks
        random_shocks = np.random.normal(0, 1, (self.num_simulations, forecast_horizon - 1))

        # Simulate GBM paths
        for t in range(1, forecast_horizon):
            drift = (predicted_return - 0.5 * predicted_volatility**2) * dt
            diffusion = predicted_volatility * np.sqrt(dt) * random_shocks[:, t - 1]
            paths[:, t] = paths[:, t - 1] * np.exp(drift + diffusion)

        return paths

    def calculate_statistics(self, paths: np.ndarray, confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Calculate statistics from simulated paths
        """
        final_prices = paths[:, -1]

        # Confidence intervals
        lower_percentile = (1 - confidence_level) / 2
        upper_percentile = 1 - lower_percentile

        stats = {
            "mean_price": np.mean(final_prices),
            "median_price": np.median(final_prices),
            "std_price": np.std(final_prices),
            "lower_bound": np.percentile(final_prices, lower_percentile * 100),
            "upper_bound": np.percentile(final_prices, upper_percentile * 100),
            "prob_profit": np.mean(final_prices > paths[0, 0]),
            "expected_return": np.mean((final_prices - paths[0, 0]) / paths[0, 0]),
            "var_95": np.percentile(final_prices - paths[0, 0], 5),
            "cvar_95": np.mean((final_prices - paths[0, 0])[final_prices - paths[0, 0] <= np.percentile(final_prices - paths[0, 0], 5)]),
        }

        return stats
