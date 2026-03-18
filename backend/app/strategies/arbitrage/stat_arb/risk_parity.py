import warnings
from typing import Dict, List, cast

import numpy as np
import pandas as pd

from .base_stat_arb import StatisticalArbitrageStrategy

warnings.filterwarnings("ignore")


class RiskParityStatArb(StatisticalArbitrageStrategy):
    """
    Risk-Parity Statistical Arbitrage
    Allocates based on risk contribution rather than equal weights
    """

    def __init__(
        self,
        risk_targeting_method: str = "inverse_volatility",
        # "inverse_volatility", "equal_risk_contribution", "marcowitz"
        risk_lookback: int = 63,  # 3 months for volatility estimation
        max_risk_contribution: float = 0.3,  # Maximum risk contribution from any single asset
        min_risk_contribution: float = 0.05,  # Minimum risk contribution
        **kwargs,
    ):
        """
        Initialize Risk-Parity Statistical Arbitrage Strategy

        Args:
            risk_targeting_method: Method for risk parity calculation
                "inverse_volatility" - Weight proportional to 1/volatility
                "equal_risk_contribution" - True risk parity (equal contribution to portfolio risk)
                "marcowitz" - Risk budgeting with volatility and correlation
            risk_lookback: Period for volatility and correlation estimation
            max_risk_contribution: Maximum risk contribution constraint
            min_risk_contribution: Minimum risk contribution constraint
        """
        super().__init__(**kwargs)

        self.risk_targeting_method = risk_targeting_method
        self.risk_lookback = risk_lookback
        self.max_risk_contribution = max_risk_contribution
        self.min_risk_contribution = min_risk_contribution

        self.method = "risk_parity_" + self.method

        self.risk_metrics = {}

    def _normalize_weights(self, weights: np.ndarray, prices: pd.DataFrame = None, assets: List[str] = None) -> np.ndarray:
        """
        Override for risk parity weighting
        Allocates based on risk contribution rather than equal weights

        Args:
            weights: Initial weights (from cointegration/PCA/etc.)
            prices: Price data for volatility calculation
            assets: Asset list corresponding to weights

        Returns:
            Risk-parity adjusted weights
        """
        if prices is None or assets is None or len(weights) != len(assets):
            # Fallback to inverse volatility if we don't have price data
            return self._inverse_volatility_weights(weights, prices, assets)

        # Calculate risk parity weights based on selected method
        if self.risk_targeting_method == "inverse_volatility":
            return self._inverse_volatility_weights(weights, prices, assets)
        elif self.risk_targeting_method == "equal_risk_contribution":
            return self._equal_risk_contribution_weights(weights, prices, assets)
        elif self.risk_targeting_method == "marcowitz":
            return self._marcowitz_risk_weights(weights, prices, assets)
        else:
            # Default to inverse volatility
            return self._inverse_volatility_weights(weights, prices, assets)

    def _inverse_volatility_weights(self, weights: np.ndarray, prices: pd.DataFrame, assets: List[str]) -> np.ndarray:
        """
        Calculate inverse volatility weights
        More volatile assets get smaller weights
        """
        # Calculate returns and volatility
        asset_prices = prices[assets].dropna()
        if len(asset_prices) < self.risk_lookback:
            # Not enough data, return original weights
            return weights / np.sum(np.abs(weights))

        returns = cast(pd.DataFrame, cast(object, np.log(asset_prices / asset_prices.shift(1)))).dropna()

        recent_returns = returns.iloc[-min(self.risk_lookback, len(returns)) :]

        volatilities = recent_returns.std() * np.sqrt(252)

        # Inverse volatility weighting
        inv_vol = 1 / volatilities.values

        # Scale to match original weight direction (long/short)
        risk_weights = np.sign(weights) * inv_vol

        # Apply min/max constraints
        risk_weights = np.clip(np.abs(risk_weights), self.min_risk_contribution, self.max_risk_contribution) * np.sign(risk_weights)

        # Normalize to sum of absolute values = 1
        risk_weights = risk_weights / np.sum(np.abs(risk_weights))

        # Store risk metrics
        self.risk_metrics[tuple(assets)] = {
            "volatilities": volatilities.to_dict(),
            "risk_weights": risk_weights.tolist(),
            "method": "inverse_volatility",
        }

        return risk_weights

    def _equal_risk_contribution_weights(self, weights: np.ndarray, prices: pd.DataFrame, assets: List[str]) -> np.ndarray:
        """
        Calculate equal risk contribution weights
        Each asset contributes equally to portfolio risk
        """
        asset_prices = prices[assets].dropna()
        if len(asset_prices) < self.risk_lookback:
            return weights / np.sum(np.abs(weights))

        # Calculate returns
        returns = cast(pd.DataFrame, cast(object, np.log(asset_prices / asset_prices.shift(1)))).dropna()
        recent_returns = cast(pd.DataFrame, cast(object, returns.iloc[-min(self.risk_lookback, len(returns)) :]))

        # Calculate covariance matrix
        cov_matrix = recent_returns.cov().values

        # Initial weights (preserve direction from original)
        initial_weights = np.abs(weights) / np.sum(np.abs(weights))

        # Optimize for equal risk contribution
        # Using iterative method (simplified)
        risk_weights = self._optimize_equal_risk_contribution(initial_weights, cov_matrix)

        # Apply sign from original weights
        risk_weights = risk_weights * np.sign(weights)

        # Apply constraints
        risk_weights = np.clip(np.abs(risk_weights), self.min_risk_contribution, self.max_risk_contribution) * np.sign(risk_weights)

        # Renormalize
        risk_weights = risk_weights / np.sum(np.abs(risk_weights))

        # Calculate risk contributions for verification
        portfolio_variance = risk_weights @ cov_matrix @ risk_weights
        marginal_contrib = cov_matrix @ risk_weights
        risk_contributions = risk_weights * marginal_contrib / np.sqrt(portfolio_variance)

        self.risk_metrics[tuple(assets)] = {
            "risk_contributions": risk_contributions.tolist(),
            "risk_weights": risk_weights.tolist(),
            "method": "equal_risk_contribution",
        }

        return risk_weights

    def _optimize_equal_risk_contribution(
        self, initial_weights: np.ndarray, cov_matrix: np.ndarray, max_iter: int = 1000, tolerance: float = 1e-6
    ) -> np.ndarray:
        """
        Optimize weights for equal risk contribution
        Uses iterative risk parity algorithm
        """
        n = len(initial_weights)
        weights = initial_weights.copy()

        for iteration in range(max_iter):
            portfolio_variance = weights @ cov_matrix @ weights
            portfolio_vol = np.sqrt(portfolio_variance)

            marginal_contrib = (cov_matrix @ weights) / portfolio_vol

            risk_contributions = weights * marginal_contrib

            target_risk = portfolio_vol / n

            if np.max(np.abs(risk_contributions - target_risk)) < tolerance:
                break

            for i in range(n):
                weights[i] = weights[i] * (target_risk / risk_contributions[i])

            weights /= np.sum(weights)

        return weights

    def _marcowitz_risk_weights(self, weights: np.ndarray, prices: pd.DataFrame, assets: List[str]) -> np.ndarray:
        """
        Calculate risk weights using Marcowitz risk budgeting
        Accounts for both volatility and correlation structure
        """
        asset_prices = prices[assets].dropna()
        if len(asset_prices) < self.risk_lookback:
            return weights / np.sum(np.abs(weights))

        # Calculate returns
        returns = cast(pd.DataFrame, cast(object, np.log(asset_prices / asset_prices.shift(1)))).dropna()
        recent_returns = cast(pd.DataFrame, cast(object, returns.iloc[-min(self.risk_lookback, len(returns)) :]))

        # Calculate volatility and correlation
        volatilities = recent_returns.std() * np.sqrt(252)
        correlation = recent_returns.corr()

        # Adjust weights based on correlation
        # Assets with high correlation get reduced weights
        n = len(assets)
        correlation_penalty = np.ones(n)

        for i in range(n):
            # Calculate average absolute correlation with other assets
            avg_corr = np.mean([abs(correlation.iloc[i, j]) for j in range(n) if j != i])
            correlation_penalty[i] = 1 / (1 + float(avg_corr))

        # Combine inverse volatility with correlation penalty
        risk_weights = np.sign(weights) * (1 / volatilities.values) * correlation_penalty

        # Apply constraints
        risk_weights = np.clip(np.abs(risk_weights), self.min_risk_contribution, self.max_risk_contribution) * np.sign(risk_weights)

        # Normalize
        risk_weights = risk_weights / np.sum(np.abs(risk_weights))

        self.risk_metrics[tuple(assets)] = {
            "volatilities": volatilities.to_dict(),
            "correlation_penalty": correlation_penalty.tolist(),
            "risk_weights": risk_weights.tolist(),
            "method": "marcowitz",
        }

        return risk_weights

    def update_baskets(self, prices: pd.DataFrame, force_update: bool = False) -> List[Dict]:
        """
        Override to incorporate risk parity in basket construction
        """
        # Get base baskets from parent class
        baskets = super().update_baskets(prices, force_update)

        # Apply risk parity weighting to each basket
        for i, basket in enumerate(baskets):
            assets = basket["assets"]
            weights = basket["weights"]

            # Get recent prices for these assets
            basket_prices = prices[assets].dropna()

            if len(basket_prices) >= self.risk_lookback:
                # Apply risk parity weighting
                risk_weights = self._normalize_weights(weights, basket_prices, assets)

                # Update basket with risk parity weights
                basket["weights"] = risk_weights
                basket["risk_metrics"] = self.risk_metrics.get(tuple(assets), {})
                basket["risk_method"] = self.risk_targeting_method

        return baskets

    def get_risk_contributions(self, basket_id: str = None) -> Dict:
        """
        Get risk contribution metrics for active baskets

        Args:
            basket_id: Optional specific basket ID

        Returns:
            Dictionary with risk contribution information
        """
        if basket_id is not None and basket_id in self.risk_metrics:
            return self.risk_metrics[basket_id]
        return self.risk_metrics

    def _calculate_basket_risk(self, weights: np.ndarray, cov_matrix: np.ndarray) -> Dict:
        """
        Calculate detailed risk metrics for a basket

        Args:
            weights: Portfolio weights
            cov_matrix: Covariance matrix

        Returns:
            Dictionary with risk metrics
        """
        # Portfolio variance
        portfolio_variance = weights @ cov_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_variance)

        # Marginal risk contributions
        marginal_contrib = cov_matrix @ weights

        # Component risk contributions
        risk_contributions = weights * marginal_contrib
        risk_contributions_pct = risk_contributions / portfolio_variance

        return {
            "portfolio_volatility": portfolio_vol,
            "marginal_risk_contributions": marginal_contrib.tolist(),
            "risk_contributions": risk_contributions.tolist(),
            "risk_contributions_pct": risk_contributions_pct.tolist(),
        }
