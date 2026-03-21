from typing import Dict, Union, cast

import numpy as np
import pandas as pd
from scipy import stats

from ...config import DEFAULT_ANNUAL_LOOKBACK
from ...strategies.volatility.base_volatility import BaseVolatilityStrategy


class DynamicVolatilityScalingStrategy(BaseVolatilityStrategy):
    """
    Dynamic Volatility Scaling

    Applies volatility scaling to any strategy dynamically.
    Standalone mode: uses momentum direction scaled by HAR vol forecast.
    """

    def __init__(
        self,
        risk_manager: callable = None,  # External risk manager
        scaling_method: str = "multiplicative",  # "multiplicative", "additive", "optimal"
        update_frequency: str = "daily",
        forecast_horizon: int = 21,  # 21-day volatility forecast
        trend_lookback: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.risk_manager = risk_manager
        self.scaling_method = scaling_method
        self.update_frequency = update_frequency
        self.forecast_horizon = forecast_horizon
        self.trend_lookback = trend_lookback

        # Scaling factors
        self.scaling_factors = {}
        self.volatility_forecasts = {}

    def generate_signal(self, data: Union[pd.Series, pd.DataFrame]) -> Dict:
        """Generate signal using momentum direction scaled by vol forecast."""
        if isinstance(data, pd.DataFrame):
            if "Close" in data.columns:
                prices = data["Close"]
            elif "close" in data.columns:
                prices = data["close"]
            else:
                prices = data.iloc[:, 0]
        else:
            prices = data

        min_required = max(self.vol_lookback, self.trend_lookback)
        if len(prices) < min_required:
            return {"signal": 0, "position_size": 0.0, "metadata": {"strategy": "dynamic_scaling"}}

        returns = self.calculate_returns(prices)
        self.update_volatility_state(returns)

        # Momentum direction
        momentum = prices.iloc[-1] / prices.iloc[-self.trend_lookback] - 1
        signal = 1 if momentum > 0 else (-1 if momentum < 0 else 0)

        position_size = self.current_leverage * abs(signal)

        return {
            "signal": signal,
            "position_size": position_size,
            "leverage": self.current_leverage,
            "current_volatility": self.volatility_history[-1] if self.volatility_history else 0.0,
            "metadata": {"strategy": "dynamic_scaling", "scaling_method": self.scaling_method},
        }

    def generate_signals_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """Vectorized signal generation using momentum direction."""
        close = data["Close"] if "Close" in data.columns else data.get("close", data.iloc[:, 0])
        signals = pd.Series(0, index=data.index)

        momentum = close / close.shift(self.trend_lookback) - 1
        direction = pd.Series(0, index=data.index)
        direction[momentum > 0] = 1
        direction[momentum < 0] = -1

        direction_change = direction != direction.shift(1)
        signals[(direction == 1) & direction_change] = 1
        signals[(direction != 1) & direction_change & (direction.shift(1) == 1)] = -1

        return signals

    def forecast_volatility(self, returns: pd.Series, horizon: int = None) -> float:
        """
        Forecast future volatility using a calibrated HAR model.
        Calibrates b0, b1, b2, b3 based on historical realized volatility.
        """
        if horizon is None:
            _ = getattr(self, "forecast_horizon", 1)

        min_train_size = 60  # 3 months of data
        if len(returns) < min_train_size:
            return float(returns.std() * np.sqrt(DEFAULT_ANNUAL_LOOKBACK))

        # Calculate Realized Volatility (RV) series
        # Using daily absolute returns as the proxy for daily volatility
        rv = np.abs(returns)

        # Create the HAR Feature Matrix (X) and Target (y)
        # y = RV_tomorrow
        # X1 = RV_today (Daily)
        # X2 = Avg(RV_last_5_days) (Weekly)
        # X3 = Avg(RV_last_22_days) (Monthly)

        df_har = pd.DataFrame({"y": rv}).shift(-1)  # Target is tomorrow's vol
        df_har["d"] = rv
        df_har["w"] = cast(pd.Series, cast(object, rv)).rolling(5).mean()
        df_har["m"] = cast(pd.Series, cast(object, rv)).rolling(22).mean()

        # Drop rows with NaNs (due to rolling and shift)
        df_har = df_har.dropna()

        # Perform OLS Calibration (y = Xb)
        # b = (X^T * X)^-1 * X^T * y
        y = df_har["y"].values
        X = df_har[["d", "w", "m"]].values
        X = np.column_stack([np.ones(len(X)), X])  # Add intercept (b0)

        try:
            # Solve for coefficients [b0, b1, b2, b3]
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            b0, b1, b2, b3 = coeffs
        except np.linalg.LinAlgError:
            # Fallback to robust defaults if the matrix is singular
            b0, b1, b2, b3 = 0.0001, 0.45, 0.30, 0.20

        # 5. Generate the Forecast for "Tomorrow"
        # We use the most recent available data points
        curr_d = cast(pd.Series, cast(object, rv)).iloc[-1]
        curr_w = cast(pd.Series, cast(object, rv)).tail(5).mean()
        curr_m = cast(pd.Series, cast(object, rv)).tail(22).mean()

        daily_forecast = b0 + b1 * curr_d + b2 * curr_w + b3 * curr_m

        annualized_vol = max(daily_forecast, 0) * np.sqrt(DEFAULT_ANNUAL_LOOKBACK)

        hist_vol = returns.std() * np.sqrt(DEFAULT_ANNUAL_LOOKBACK)
        return float(np.clip(annualized_vol, hist_vol * 0.5, hist_vol * 3.0))

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
        vol_forecast = self.forecast_volatility(returns)
        self.volatility_forecasts[asset] = vol_forecast

        base_scale = self.target_volatility / max(vol_forecast, self.min_vol)

        if self.risk_manager is not None:
            risk_constraints = self.risk_manager.get_constraints(asset, current_position)

            if "max_position" in risk_constraints:
                max_pos = risk_constraints["max_position"]
                if current_position != 0:
                    current_scale = abs(current_position)
                    max_scale = max_pos / current_scale if current_position != 0 else 1.0
                    base_scale = min(base_scale, max_scale)

            if "var_limit" in risk_constraints:
                var_limit = risk_constraints["var_limit"]
                portfolio_var = self._estimate_portfolio_var(asset, current_position, vol_forecast)

                if portfolio_var > 0:
                    var_scale = var_limit / portfolio_var
                    base_scale = min(base_scale, var_scale)

        if self.scaling_method == "multiplicative":
            scale_factor = base_scale

        elif self.scaling_method == "additive":
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

        self.scaling_factors[asset] = scale_factor

        return scale_factor

    def _estimate_portfolio_var(self, asset: str, position: float, volatility: float, confidence: float = 0.95) -> float:
        """
        Estimate Parametric Value at Risk (VaR) for a specific position.

        Formula: VaR = Position_Value * Volatility * Z-Score * sqrt(Time_Horizon)
        """
        if volatility <= 0 or position == 0:
            return 0.0

        # 0.95 -> ~1.645, 0.99 -> ~2.326
        z_score = stats.norm.ppf(confidence)

        # Time Scaling Logic
        # If 'volatility' is annualized (e.g., 0.20 for 20%):
        # To get 1-Day VaR, we scale by sqrt(1/252)
        annual_days = DEFAULT_ANNUAL_LOOKBACK  # Standard trading year
        time_scale = 1 / np.sqrt(annual_days)

        var = abs(position) * volatility * z_score * time_scale

        return float(var)

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
