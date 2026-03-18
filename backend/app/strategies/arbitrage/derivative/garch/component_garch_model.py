import warnings

import numpy as np
from scipy.optimize import minimize

from .....strategies.arbitrage.derivative.garch.base_garch_model import BaseGARCHModel

warnings.filterwarnings("ignore")


class ComponentGARCHModel(BaseGARCHModel):
    """Component GARCH model for long memory"""

    def __init__(self):
        super().__init__(name="Component GARCH")
        self.returns = None
        self.omega = None
        self.alpha = 0.05
        self.beta = 0.9
        self.rho = 0.99
        self.phi = 0.05

    def fit(self, returns: np.ndarray) -> "ComponentGARCHModel":
        """Fit Component GARCH model"""
        self.returns = returns

        def negative_log_likelihood(params):
            omega, alpha, beta, rho, phi = params

            T = len(self.returns)
            sigma2 = np.zeros(T)
            q_t = np.zeros(T)

            initial_var = np.var(self.returns)
            sigma2[0] = initial_var
            q_t[0] = initial_var

            for t in range(1, T):
                q_t[t] = omega + rho * (q_t[t - 1] - omega) + phi * (self.returns[t - 1] ** 2 - sigma2[t - 1])
                sigma2[t] = q_t[t] + alpha * (self.returns[t - 1] ** 2 - q_t[t - 1]) + beta * (sigma2[t - 1] - q_t[t - 1])

            likelihood = -0.5 * float(np.sum(np.log(2 * np.pi * sigma2[1:] + 1e-8) + self.returns[1:] ** 2 / (sigma2[1:] + 1e-8)))
            return -likelihood

        initial_var = np.var(self.returns)
        init_params = [(initial_var * 0.1), 0.05, 0.9, 0.99, 0.05]

        try:
            result = minimize(negative_log_likelihood, init_params, method="L-BFGS-B")
            self.omega, self.alpha, self.beta, self.rho, self.phi = result.x
        except Exception:
            self.omega = initial_var * 0.1

        self.params = {"omega": self.omega, "alpha": self.alpha, "beta": self.beta, "rho": self.rho, "phi": self.phi}
        self.fitted = True
        self._compute_conditional_volatility()

        return self

    def _compute_conditional_volatility(self):
        """Compute conditional volatility series"""
        T = len(self.returns)
        sigma2 = np.zeros(T)
        q_t = np.zeros(T)

        initial_var = np.var(self.returns)
        sigma2[0] = initial_var
        q_t[0] = initial_var

        for t in range(1, T):
            q_t[t] = self.omega + self.rho * (q_t[t - 1] - self.omega) + self.phi * (self.returns[t - 1] ** 2 - sigma2[t - 1])
            sigma2[t] = q_t[t] + self.alpha * (self.returns[t - 1] ** 2 - q_t[t - 1]) + self.beta * (sigma2[t - 1] - q_t[t - 1])

        self.conditional_volatility = np.sqrt(sigma2)
        self.residuals = self.returns
        self.sigma2 = sigma2
        self.q_t = q_t

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """Generate volatility forecasts"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")

        forecasts = []
        current_q = self.q_t[-1]
        current_sigma2 = self.sigma2[-1]

        for _ in range(horizon):
            new_q = self.omega + self.rho * (current_q - self.omega)
            current_sigma2 = new_q + self.alpha * (current_sigma2 - current_q) + self.beta * (current_sigma2 - current_q)
            current_q = new_q

            forecasts.append(np.sqrt(current_sigma2))

        return np.array(forecasts)
