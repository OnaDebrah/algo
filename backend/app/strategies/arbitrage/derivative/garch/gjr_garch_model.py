import warnings
from typing import cast

import numpy as np
from scipy.optimize import minimize

from .....strategies.arbitrage.derivative.garch.base_garch_model import BaseGARCHModel

warnings.filterwarnings("ignore")


class GJRGARCHModel(BaseGARCHModel):
    """GJR-GARCH model for leverage effects"""

    def __init__(self):
        super().__init__(name="GJR-GARCH(1,1)")
        self.returns = None
        self.omega = 0.00001
        self.alpha = 0.05
        self.gamma = 0.1
        self.beta = 0.85

    def fit(self, returns: np.ndarray) -> "GJRGARCHModel":
        """Fit GJR-GARCH model"""
        self.returns = returns

        def negative_log_likelihood(params):
            omega, alpha, gamma, beta = params

            if omega <= 0 or alpha < 0 or gamma < 0 or beta < 0:
                return 1e10

            T = len(self.returns)
            sigma2 = np.zeros(T)
            sigma2[0] = np.var(self.returns)

            for t in range(1, T):
                leverage = gamma * (self.returns[t - 1] < 0) * self.returns[t - 1] ** 2
                sigma2[t] = omega + alpha * self.returns[t - 1] ** 2 + leverage + beta * sigma2[t - 1]

            likelihood = -0.5 * float(np.sum(np.log(2 * np.pi * sigma2[1:] + 1e-8) + self.returns[1:] ** 2 / (sigma2[1:] + 1e-8)))
            return -likelihood

        init_params = [0.00001, 0.05, 0.1, 0.85]
        bounds = [(1e-8, 1), (0, 1), (0, 1), (0, 1)]

        try:
            result = minimize(negative_log_likelihood, init_params, bounds=bounds, method="L-BFGS-B")
            self.omega, self.alpha, self.gamma, self.beta = result.x
        except Exception:
            pass

        self.params = {"omega": self.omega, "alpha": self.alpha, "gamma": self.gamma, "beta": self.beta}
        self.fitted = True
        self._compute_conditional_volatility()

        return self

    def _compute_conditional_volatility(self):
        """Compute conditional volatility series"""
        T = len(self.returns)
        sigma2 = np.zeros(T)
        sigma2[0] = np.var(self.returns)

        for t in range(1, T):
            leverage = self.gamma * (self.returns[t - 1] < 0) * self.returns[t - 1] ** 2
            sigma2[t] = self.omega + self.alpha * self.returns[t - 1] ** 2 + leverage + self.beta * sigma2[t - 1]

        self.conditional_volatility = np.sqrt(sigma2)
        self.residuals = self.returns
        self.sigma2 = sigma2

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """Generate volatility forecasts"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")

        forecasts = []
        last_var = self.sigma2[-1]
        last_return = self.returns[-1]

        for step in range(horizon):
            if step == 0:
                leverage = self.gamma * (last_return < 0) * last_return**2
                current_var = self.omega + self.alpha * last_return**2 + leverage + self.beta * last_var
            else:
                expected_leverage = 0.5 * self.gamma * last_var
                current_var = self.omega + (self.alpha * last_var) + expected_leverage + (self.beta * last_var)

            forecasts.append(np.sqrt(current_var))
            last_var = current_var

        return np.array(forecasts)

    def forecast_monte_carlo(self, horizon: int = 1, n_sims: int = 5000) -> np.ndarray:
        """
        Generate volatility forecasts using Monte Carlo simulation.
        Returns the expected volatility (mean of all simulations) for each step.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")

        all_sim_paths = np.zeros((n_sims, horizon))

        initial_var = self.sigma2[-1]
        initial_return = self.returns[-1]

        for i in range(n_sims):
            last_var = initial_var
            last_ret = initial_return

            for step in range(horizon):
                leverage_effect = self.gamma * (last_ret**2) if last_ret < 0 else 0

                current_var = self.omega + self.alpha * (last_ret**2) + leverage_effect + self.beta * last_var

                # Return = StdDev * Z (where Z is Standard Normal)
                z = np.random.standard_normal()
                current_ret = np.sqrt(current_var) * z

                # Store the volatility (Std Dev)
                all_sim_paths[i, step] = np.sqrt(current_var)

                last_var = current_var
                last_ret = current_ret

        return cast(np.ndarray, cast(object, np.mean(all_sim_paths, axis=0)))
