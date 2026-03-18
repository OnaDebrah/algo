import numpy as np
from scipy.optimize import minimize

from .....strategies.arbitrage.derivative.garch.base_garch_model import BaseGARCHModel


class EGARCHModel(BaseGARCHModel):
    """EGARCH(1,1) model for asymmetric volatility"""

    def __init__(self):
        super().__init__(name="EGARCH(1,1)")
        self.omega = -0.1
        self.alpha = 0.1
        self.gamma = -0.05
        self.beta = 0.95

    def fit(self, returns: np.ndarray) -> "EGARCHModel":
        """Fit EGARCH model"""
        self.returns = returns

        def negative_log_likelihood(params):
            omega, alpha, gamma, beta = params

            T = len(self.returns)
            log_sigma2 = np.zeros(T)
            sigma2 = np.zeros(T)

            log_sigma2[0] = np.log(np.var(self.returns))
            sigma2[0] = np.exp(log_sigma2[0])

            for t in range(1, T):
                z = self.returns[t - 1] / np.sqrt(sigma2[t - 1] + 1e-8)
                log_sigma2[t] = omega + alpha * (abs(z) - np.sqrt(2 / np.pi)) + gamma * z + beta * log_sigma2[t - 1]
                sigma2[t] = np.exp(log_sigma2[t])

            likelihood = -0.5 * np.sum(np.log(2 * np.pi * sigma2[1:] + 1e-8) + self.returns[1:] ** 2 / (sigma2[1:] + 1e-8))
            return -likelihood

        init_params = [-0.1, 0.1, -0.05, 0.95]

        try:
            result = minimize(negative_log_likelihood, init_params, method="L-BFGS-B")
            self.omega, self.alpha, self.gamma, self.beta = result.x
        except Exception:
            pass  # Keep initial parameters

        self.params = {"omega": self.omega, "alpha": self.alpha, "gamma": self.gamma, "beta": self.beta}
        self.fitted = True
        self._compute_conditional_volatility()

        return self

    def _compute_conditional_volatility(self):
        """Compute conditional volatility series"""
        T = len(self.returns)
        log_sigma2 = np.zeros(T)
        sigma2 = np.zeros(T)

        log_sigma2[0] = np.log(np.var(self.returns))
        sigma2[0] = np.exp(log_sigma2[0])

        for t in range(1, T):
            z = self.returns[t - 1] / np.sqrt(sigma2[t - 1] + 1e-8)
            log_sigma2[t] = self.omega + self.alpha * (abs(z) - np.sqrt(2 / np.pi)) + self.gamma * z + self.beta * log_sigma2[t - 1]
            sigma2[t] = np.exp(log_sigma2[t])

        self.conditional_volatility = np.sqrt(sigma2)
        self.residuals = self.returns
        self.log_sigma2 = log_sigma2

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """Generate volatility forecasts"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")

        forecasts = []
        last_log_var = self.log_sigma2[-1]
        last_z = self.returns[-1] / (self.conditional_volatility[-1] + 1e-8)

        for step in range(horizon):
            if step == 0:
                z = last_z
            else:
                z = 0  # Expected value of z is 0

            last_log_var = self.omega + self.alpha * (abs(z) - np.sqrt(2 / np.pi)) + self.gamma * z + self.beta * last_log_var
            forecasts.append(np.sqrt(np.exp(last_log_var)))

        return np.array(forecasts)
