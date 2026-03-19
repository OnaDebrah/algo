import numpy as np
from scipy.optimize import minimize

from .....strategies.arbitrage.derivative.garch.base_garch_model import BaseGARCHModel


class GARCHModel(BaseGARCHModel):
    """Standard GARCH(1,1) model"""

    def __init__(self):
        super().__init__(name="GARCH(1,1)")
        self.omega = 0.00001
        self.alpha = 0.1
        self.beta = 0.85

    def fit(self, returns: np.ndarray, method: str = "mle") -> "GARCHModel":
        """Fit GARCH(1,1) model"""
        self.returns = returns

        if method == "mle":
            self._fit_mle()
        else:
            self._fit_method_of_moments()

        self.fitted = True
        self._compute_conditional_volatility()

        return self

    def _fit_mle(self):
        """MLE parameter estimation"""

        def negative_log_likelihood(params):
            omega, alpha, beta = params

            # Constraints
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10

            T = len(self.returns)
            sigma2 = np.zeros(T)
            sigma2[0] = np.var(self.returns)

            for t in range(1, T):
                sigma2[t] = omega + alpha * self.returns[t - 1] ** 2 + beta * sigma2[t - 1]

            # Gaussian likelihood
            likelihood = -0.5 * np.sum(np.log(2 * np.pi * sigma2[1:] + 1e-8) + self.returns[1:] ** 2 / (sigma2[1:] + 1e-8))
            return -likelihood

        # Initial guesses
        init_params = [0.00001, 0.1, 0.85]
        bounds = [(1e-8, 1), (0, 1), (0, 1)]

        try:
            result = minimize(negative_log_likelihood, init_params, bounds=bounds, method="L-BFGS-B")
            self.omega, self.alpha, self.beta = result.x

            # Ensure stationarity
            if self.alpha + self.beta >= 1:
                scale = 0.99 / (self.alpha + self.beta)
                self.alpha *= scale
                self.beta *= scale

        except Exception:
            self._fit_method_of_moments()

        self.params = {"omega": self.omega, "alpha": self.alpha, "beta": self.beta}

    def _fit_method_of_moments(self):
        """Method of moments estimation (faster)"""
        returns_sq = self.returns**2

        if len(returns_sq) > 10:
            acf1 = np.corrcoef(returns_sq[:-1], returns_sq[1:])[0, 1]
            acf2 = np.corrcoef(returns_sq[:-2], returns_sq[2:])[0, 1]

            if acf1 > 0 and acf2 > 0:
                self.beta = (acf2 / acf1) if acf1 != 0 else 0.85
                self.alpha = acf1 * (1 - self.beta**2) / (1 + acf1 * self.beta)

                # Apply constraints
                self.alpha = max(0.01, min(0.3, self.alpha))
                self.beta = max(0.5, min(0.98, self.beta))

                if self.alpha + self.beta >= 1:
                    scale = 0.99 / (self.alpha + self.beta)
                    self.alpha *= scale
                    self.beta *= scale

        unconditional_var = np.var(self.returns)
        self.omega = unconditional_var * (1 - self.alpha - self.beta)
        self.params = {"omega": self.omega, "alpha": self.alpha, "beta": self.beta}

    def _compute_conditional_volatility(self):
        """Compute conditional volatility series"""
        T = len(self.returns)
        sigma2 = np.zeros(T)
        sigma2[0] = np.var(self.returns)

        for t in range(1, T):
            sigma2[t] = self.omega + self.alpha * self.returns[t - 1] ** 2 + self.beta * sigma2[t - 1]

        self.conditional_volatility = np.sqrt(sigma2)
        self.residuals = self.returns

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """Generate volatility forecasts"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")

        forecasts = []
        last_var = self.conditional_volatility[-1] ** 2
        last_return = self.returns[-1]

        for step in range(horizon):
            if step == 0:
                current_var = self.omega + self.alpha * last_return**2 + self.beta * last_var
            else:
                current_var = self.omega + (self.alpha + self.beta) * last_var

            forecasts.append(np.sqrt(current_var))
            last_var = current_var

        return np.array(forecasts)
