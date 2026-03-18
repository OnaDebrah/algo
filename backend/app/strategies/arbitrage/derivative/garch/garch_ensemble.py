from typing import Dict, List

import numpy as np

from .....strategies.arbitrage.derivative.garch.base_garch_model import BaseGARCHModel
from .component_garch_model import ComponentGARCHModel
from .egarch_model import EGARCHModel
from .garch_model import GARCHModel
from .gjr_garch_model import GJRGARCHModel


class GARCHEnsemble:
    """Ensemble of GARCH models for robust forecasting"""

    def __init__(self, models: List[BaseGARCHModel] = None):
        if models is None:
            self.models = [GARCHModel(), EGARCHModel(), GJRGARCHModel(), ComponentGARCHModel()]
        else:
            self.models = models

        self.weights = None
        self.fitted = False

    def fit(self, returns: np.ndarray, method: str = "equal") -> "GARCHEnsemble":
        """Fit all models in ensemble"""
        for model in self.models:
            model.fit(returns)

        self.fitted = True

        if method == "aic":
            self._compute_aic_weights()
        elif method == "bic":
            self._compute_bic_weights()
        elif method == "cross_validation":
            self._compute_cv_weights(returns)
        else:
            self.weights = np.ones(len(self.models)) / len(self.models)

        return self

    def _compute_aic_weights(self):
        """Compute model weights based on AICc (Corrected AIC)"""
        aics = []
        for model in self.models:
            if hasattr(model, "get_info"):
                k = len(model.get_info()["parameters"])
                n = len(model.residuals) if model.residuals is not None else 0

                log_lik = getattr(model, "log_likelihood", 0)

                # Standard AIC
                aic = 2 * k - 2 * log_lik

                if n > (k + 1):
                    aic_c = aic + (2 * k**2 + 2 * k) / (n - k - 1)
                    aics.append(aic_c)
                else:
                    aics.append(aic)

        aics = np.array(aics)
        delta = aics - np.min(aics)

        weights = np.exp(-0.5 * delta)
        self.weights = weights / np.sum(weights)

    def _compute_bic_weights(self):
        """
        Compute model weights based on BIC (Bayesian Information Criterion).
        BIC = k * ln(n) - 2 * ln(L)
        Lower BIC indicates a better model.
        """
        bics = []

        for model in self.models:
            # Extract parameters (k) and sample size (n)
            if hasattr(model, "get_info"):
                params = model.get_info().get("parameters", {})
                k = len(params)

                # n is the number of observations (residuals)
                n = len(model.residuals) if hasattr(model, "residuals") and model.residuals is not None else 0

                # Get the actual Log-Likelihood
                # If your model doesn't have it, we calculate it from the residuals
                if hasattr(model, "log_likelihood"):
                    log_lik = model.log_likelihood
                else:
                    # Fallback: Calculate Log-Likelihood assuming normal distribution of residuals
                    ssr = np.sum(model.residuals**2)
                    if n > 0 and ssr > 0:
                        log_lik = -0.5 * n * (np.log(2 * np.pi) + np.log(ssr / n) + 1)
                    else:
                        log_lik = -1e10  # Penalty for failed models

                # Calculate BIC
                # Formula: ln(n) * k - 2 * log_likelihood
                if n > 0:
                    bic = k * np.log(n) - 2 * log_lik
                    bics.append(bic)
                else:
                    bics.append(1e10)

        if not bics:
            self.weights = np.array([1.0 / len(self.models)] * len(self.models))
            return

        # Convert BICs to Schwarz Weights (Posterior Probabilities)
        # We subtract the minimum BIC for numerical stability (prevents overflow in exp)
        bics = np.array(bics)
        delta_bic = bics - np.min(bics)

        # Weight = exp(-0.5 * delta)
        # This represents the relative posterior probability of the model being 'correct'
        raw_weights = np.exp(-0.5 * delta_bic)

        self.weights = raw_weights / np.sum(raw_weights)

    def _compute_cv_weights(self, returns: np.ndarray, n_folds: int = 5):
        """Compute weights using cross-validation"""
        fold_size = len(returns) // n_folds
        errors = np.zeros((len(self.models), n_folds))

        for i in range(n_folds):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size

            train = np.concatenate([returns[:test_start], returns[test_end:]])
            test = returns[test_start:test_end]

            for j, model in enumerate(self.models):
                try:
                    model.fit(train)
                    forecasts = model.forecast(len(test))
                    mse = np.mean((forecasts - np.abs(test)) ** 2)
                    errors[j, i] = mse
                except Exception:
                    errors[j, i] = np.inf

        # Inverse error weighting
        mean_errors = np.nanmean(errors, axis=1)
        weights = 1 / (mean_errors + 1e-8)
        self.weights = weights / np.sum(weights)

    def forecast(self, horizon: int = 1, return_all: bool = False) -> Dict:
        """
        Generate ensemble forecast

        Args:
            horizon: Forecast horizon
            return_all: If True, return all model forecasts

        Returns:
            Dictionary with ensemble forecast and individual model forecasts
        """
        if not self.fitted:
            raise ValueError("Ensemble must be fitted first")

        all_forecasts = []
        individual_forecasts = {}

        for model, weight in zip(self.models, self.weights):
            forecast = model.forecast(horizon)
            all_forecasts.append(forecast * weight)
            individual_forecasts[model.name] = forecast * 252  # Annualized

        ensemble_forecast = np.sum(all_forecasts, axis=0) * 252  # Annualized

        # Calculate confidence intervals from ensemble dispersion
        forecast_array = np.array([f * 252 for f in all_forecasts])
        forecast_std = np.std(forecast_array, axis=0)

        result = {
            "forecast": ensemble_forecast[-1] if horizon > 1 else ensemble_forecast[0],
            "term_structure": ensemble_forecast.tolist(),
            "individual_forecasts": individual_forecasts,
            "weights": self.weights.tolist(),
            "confidence_interval": {
                "lower": (ensemble_forecast - 1.96 * float(forecast_std)).tolist(),
                "upper": (ensemble_forecast + 1.96 * float(forecast_std)).tolist(),
            },
        }

        return result
