from typing import Tuple, cast

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ....config import DEFAULT_ANNUAL_LOOKBACK
from ....strategies.arbitrage.derivative.garch.garch_ensemble import GARCHEnsemble


class VolatilityForecaster:
    """Forecasts realized volatility using various models"""

    def __init__(self, model: str = "garch", lookback_period: int = 60):
        self.model = model
        self.lookback_period = lookback_period

    def forecast(self, prices: pd.Series, horizon: int = DEFAULT_ANNUAL_LOOKBACK) -> float:
        """Forecast realized volatility over horizon"""

        forecasters = {
            "historical": self._historical_volatility,
            "garch": self._garch_forecast,
            "ewma": self._ewma_forecast,
            "har": self._har_forecast,
            "jump_diffusion": self._jump_diffusion_forecast,
        }

        forecaster = forecasters.get(self.model, self._historical_volatility)
        return forecaster(prices, horizon)

    def _historical_volatility(self, prices: pd.Series, horizon: int = DEFAULT_ANNUAL_LOOKBACK) -> float:
        """Simple historical volatility"""
        returns = cast(pd.Series, cast(object, np.log(prices / prices.shift(1)))).dropna()
        return returns.iloc[-self.lookback_period :].std() * np.sqrt(horizon)

    def _garch_forecast(self, prices: pd.Series, horizon: int) -> float:
        """
        GARCH(1,1) forecast using ensemble of models

        Args:
            prices: Historical price series
            horizon: Forecast horizon in days

        Returns:
            Forecasted annualized volatility
        """
        returns = cast(pd.Series, cast(object, np.log(prices / prices.shift(1)))).dropna()

        returns_array = returns.values

        if len(returns_array) < 50:
            return returns.std() * np.sqrt(DEFAULT_ANNUAL_LOOKBACK)

        try:
            ensemble = GARCHEnsemble()
            ensemble.fit(returns_array)
            forecast_result = ensemble.forecast(horizon=horizon)

            if isinstance(forecast_result, (float, np.float64)):
                return forecast_result * np.sqrt(DEFAULT_ANNUAL_LOOKBACK)  # Annualize

            elif isinstance(forecast_result, dict):
                forecast = forecast_result.get("forecast", 0)
                if isinstance(forecast, (list, np.ndarray)):
                    return forecast[-1]  # Get last value for horizon
                return float(forecast) * np.sqrt(DEFAULT_ANNUAL_LOOKBACK)

            elif isinstance(forecast_result, (list, np.ndarray)):
                if len(forecast_result) > 0:
                    if horizon <= len(forecast_result):
                        return forecast_result[horizon - 1] * np.sqrt(DEFAULT_ANNUAL_LOOKBACK)
                    else:
                        return forecast_result[-1] * np.sqrt(DEFAULT_ANNUAL_LOOKBACK)

            # Fallback
            return returns.std() * np.sqrt(DEFAULT_ANNUAL_LOOKBACK)

        except Exception:
            return returns.std() * np.sqrt(DEFAULT_ANNUAL_LOOKBACK)

    def _ewma_forecast(self, prices: pd.Series, horizon: int = DEFAULT_ANNUAL_LOOKBACK) -> float:
        """Exponentially weighted moving average"""
        returns = cast(pd.Series, cast(object, np.log(prices / prices.shift(1)))).dropna()
        lambda_param = 0.94

        weights = np.array([(1 - lambda_param) * lambda_param**i for i in range(len(returns))][::-1])
        weights = weights / weights.sum()

        var_ewma = np.sum(weights * returns.values**2)
        return np.sqrt(var_ewma * horizon)

    def _har_forecast(self, prices: pd.Series, horizon: int = DEFAULT_ANNUAL_LOOKBACK) -> float:
        """
        Heterogeneous Autoregressive (HAR-RV) model forecast.
        Predicts future volatility based on daily, weekly, and monthly
        historical realized volatility components.
        """
        intercept, b_d, b_w, b_m = self._calibrate_har_parameters(prices)

        log_returns = cast(pd.Series, cast(object, np.log(prices / prices.shift(1)))).dropna()
        rv_series = log_returns**2

        rv_d = rv_series.iloc[-1]
        rv_w = rv_series.tail(5).mean()
        rv_m = rv_series.tail(22).mean()

        forecast_var = intercept + (b_d * rv_d) + (b_w * rv_w) + (b_m * rv_m)

        return float(np.sqrt(forecast_var * horizon))

    def _calibrate_har_parameters(self, prices: pd.Series) -> Tuple[float, float, float, float]:
        """
        Calibrates HAR-RV parameters (intercept, beta_d, beta_w, beta_m)
        using historical OLS regression.
        """
        if len(prices) < 60:  # Need enough history for a stable regression
            return 0.000001, 0.45, 0.30, 0.20  # Fallback to defaults

        log_returns = cast(pd.Series, cast(object, np.log(prices / prices.shift(1)))).dropna()
        rv = log_returns**2

        df = pd.DataFrame({"rv_d": rv})
        df["rv_w"] = df["rv_d"].rolling(5).mean()
        df["rv_m"] = df["rv_d"].rolling(22).mean()

        df["target"] = df["rv_d"].shift(-1)

        df = df.dropna()

        X = df[["rv_d", "rv_w", "rv_m"]]
        X = sm.add_constant(X)  # Adds the intercept (const)
        y = df["target"]

        try:
            model = sm.OLS(y, X).fit()

            intercept = model.params["const"]
            beta_d = model.params["rv_d"]
            beta_w = model.params["rv_w"]
            beta_m = model.params["rv_m"]

            return max(0, intercept), max(0, beta_d), max(0, beta_w), max(0, beta_m)

        except Exception:
            return 0.000001, 0.45, 0.30, 0.20  # Emergency fallback

    def _jump_diffusion_forecast(self, prices: pd.Series, horizon: int) -> float:
        """
        Estimates Merton Jump Diffusion parameters and forecasts
        total expected volatility over a given horizon.
        """
        if len(prices) < 2:
            return 0.0

        returns = cast(pd.Series, cast(object, np.log(prices / prices.shift(1)))).dropna()

        if returns.empty:
            return 0.0

        std_dev = returns.std()
        jump_threshold = 3 * std_dev

        jumps = returns[np.abs(returns) > jump_threshold]
        continuous = returns[np.abs(returns) <= jump_threshold]

        sigma_c = continuous.std() if not continuous.empty else std_dev

        lambda_daily = len(jumps) / len(returns)

        mu_j = jumps.mean() if not jumps.empty else 0.0
        sigma_j = jumps.std() if len(jumps) > 1 else 0.0

        # Forecast Total Variance over the Horizon
        # Formula: Total Variance = (Diffusion Var) + (Jump Contribution)
        # Jump Contribution = lambda * (mean_jump_size^2 + var_jump_size)

        daily_diffusion_var = sigma_c**2
        daily_jump_var = lambda_daily * (mu_j**2 + sigma_j**2)

        total_daily_var = daily_diffusion_var + daily_jump_var

        forecasted_vol = np.sqrt(total_daily_var * horizon)

        return float(forecasted_vol)
