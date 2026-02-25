from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

from ...strategies.volatility.base_volatility import BaseVolatilityStrategy


class DynamicVolatilityScalingStrategy(BaseVolatilityStrategy):
    """
    Dynamic Volatility Scaling

    Applies volatility scaling to any strategy dynamically
    Integrates with existing risk manager
    """

    def __init__(
        self,
        risk_manager: callable = None,  # External risk manager
        scaling_method: str = "multiplicative",  # "multiplicative", "additive", "optimal"
        update_frequency: str = "daily",
        forecast_horizon: int = 21,  # 21-day volatility forecast
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.risk_manager = risk_manager
        self.scaling_method = scaling_method
        self.update_frequency = update_frequency
        self.forecast_horizon = forecast_horizon

        # Scaling factors
        self.scaling_factors = {}
        self.volatility_forecasts = {}

    def forecast_volatility(self, returns: pd.Series, horizon: int = None) -> float:
        """
        Forecast future volatility

        Args:
            returns: Historical returns
            horizon: Forecast horizon in days

        Returns:
            Volatility forecast
        """
        if horizon is None:
            horizon = self.forecast_horizon

        if len(returns) < 20:
            return self.target_volatility

        # Simple HAR (Heterogeneous Autoregressive) model
        # Use daily, weekly, monthly volatilities

        # Daily volatility (1-day)
        vol_d = returns.std()

        # Weekly volatility (5-day)
        if len(returns) >= 5:
            weekly_returns = returns.rolling(5).sum()
            vol_w = weekly_returns.std()
        else:
            vol_w = vol_d * np.sqrt(5)

        # Monthly volatility (21-day)
        if len(returns) >= 21:
            monthly_returns = returns.rolling(21).sum()
            vol_m = monthly_returns.std()
        else:
            vol_m = vol_d * np.sqrt(21)

        # HAR regression coefficients (simplified)
        # In production, estimate via OLS
        b0, b1, b2, b3 = 0.01, 0.6, 0.3, 0.1

        # Forecast
        vol_forecast = b0 + b1 * vol_d + b2 * vol_w + b3 * vol_m

        # Annualize
        vol_forecast = vol_forecast * np.sqrt(252)

        return vol_forecast

    def calculate_scaling_factor(self, asset: str, returns: pd.Series, current_position: float = 0.0) -> float:
        """
        Calculate dynamic scaling factor for an asset

        Args:
            asset: Asset identifier
            returns: Asset returns
            current_position: Current position

        Returns:
            Scaling factor (0 to 1+)
        """
        # Forecast volatility
        vol_forecast = self.forecast_volatility(returns)
        self.volatility_forecasts[asset] = vol_forecast

        # Base scaling: inverse volatility
        base_scale = self.target_volatility / max(vol_forecast, self.min_vol)

        # Apply risk manager constraints if available
        if self.risk_manager is not None:
            risk_constraints = self.risk_manager.get_constraints(asset, current_position)

            if "max_position" in risk_constraints:
                max_pos = risk_constraints["max_position"]
                if current_position != 0:
                    current_scale = abs(current_position)
                    max_scale = max_pos / current_scale if current_position != 0 else 1.0
                    base_scale = min(base_scale, max_scale)

            if "var_limit" in risk_constraints:
                # Adjust for VaR constraints
                var_limit = risk_constraints["var_limit"]
                portfolio_var = self._estimate_portfolio_var(asset, current_position, vol_forecast)

                if portfolio_var > 0:
                    var_scale = var_limit / portfolio_var
                    base_scale = min(base_scale, var_scale)

        # Apply scaling method
        if self.scaling_method == "multiplicative":
            scale_factor = base_scale

        elif self.scaling_method == "additive":
            # Adjust position additively
            target_exposure = self.target_volatility / vol_forecast
            current_exposure = abs(current_position) if current_position != 0 else 0
            scale_factor = 1.0 + (target_exposure - current_exposure) * 0.1  # 10% adjustment

        elif self.scaling_method == "optimal":
            # Kelly-optimal scaling
            # Assume Sharpe ratio of 0.5 (adjust based on strategy)
            sharpe_ratio = 0.5
            optimal_fraction = sharpe_ratio / vol_forecast
            scale_factor = optimal_fraction / max(abs(current_position), 0.01)

        # Apply bounds
        scale_factor = np.clip(scale_factor, 0.1, 3.0)

        # Store
        self.scaling_factors[asset] = scale_factor

        return scale_factor

    def _estimate_portfolio_var(self, asset: str, position: float, volatility: float, confidence: float = 0.95) -> float:
        """Estimate portfolio VaR for position"""
        # Simplified VaR calculation
        z_score = stats.norm.ppf(confidence)
        var = abs(position) * volatility * z_score / np.sqrt(252)
        return var

    def scale_position(self, asset: str, position: float, returns: pd.Series) -> float:
        """
        Scale position based on volatility and risk constraints

        Args:
            asset: Asset identifier
            position: Current position
            returns: Asset returns

        Returns:
            Scaled position
        """
        scale_factor = self.calculate_scaling_factor(asset, returns, position)
        scaled_position = position * scale_factor

        return scaled_position

    def generate_risk_report(self) -> Dict:
        """
        Generate risk report with volatility metrics
        """
        report = {
            "timestamp": pd.Timestamp.now(),
            "target_volatility": self.target_volatility,
            "scaling_factors": self.scaling_factors.copy(),
            "volatility_forecasts": self.volatility_forecasts.copy(),
            "current_leverage": self.current_leverage,
            "volatility_history": (self.volatility_history[-20:] if len(self.volatility_history) >= 20 else self.volatility_history),
            "metrics": self.metrics.copy(),
        }

        # Add risk concentrations
        if hasattr(self, "portfolio_positions"):
            report["risk_concentrations"] = self._calculate_risk_concentrations()

        return report

    def _calculate_risk_concentrations(self) -> Dict:
        """Calculate risk concentrations in portfolio"""
        concentrations = {
            "volatility_buckets": {"low": 0, "medium": 0, "high": 0},
            "leverage_distribution": {"low": 0, "medium": 0, "high": 0},
        }

        for asset, vol in self.volatility_forecasts.items():
            if vol < 0.15:
                concentrations["volatility_buckets"]["low"] += 1
            elif vol < 0.30:
                concentrations["volatility_buckets"]["medium"] += 1
            else:
                concentrations["volatility_buckets"]["high"] += 1

        return concentrations
