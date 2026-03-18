import warnings
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")


class BaseGARCHModel(ABC):
    """Abstract base class for all GARCH models"""

    def __init__(self, name: str = "BaseGARCH"):
        self.name = name
        self.params = {}
        self.fitted = False
        self.residuals = None
        self.conditional_volatility = None

    @abstractmethod
    def fit(self, returns: np.ndarray) -> "BaseGARCHModel":
        """Fit the GARCH model to returns data"""
        pass

    @abstractmethod
    def forecast(self, horizon: int = 1) -> np.ndarray:
        """Generate volatility forecasts"""
        pass

    def diagnostic_tests(self) -> Dict:
        """Run diagnostic tests on fitted model"""
        if not self.fitted or self.residuals is None:
            raise ValueError("Model must be fitted first")

        std_residuals = self.residuals / (self.conditional_volatility + 1e-8)

        return {
            "ljung_box": self._ljung_box_test(std_residuals),
            "jarque_bera": self._jarque_bera_test(std_residuals),
            "arch_lm": self._arch_lm_test(std_residuals),
            "skewness": stats.skew(std_residuals),
            "kurtosis": stats.kurtosis(std_residuals),
        }

    def _ljung_box_test(self, residuals: np.ndarray, lags: int = 10) -> Dict:
        """Ljung-Box test for autocorrelation"""
        from statsmodels.stats.diagnostic import acorr_ljungbox

        try:
            result = acorr_ljungbox(residuals, lags=[lags], return_df=True)
            return {"statistic": result.iloc[0, 0], "p_value": result.iloc[0, 1]}
        except Exception:
            return {"statistic": 0, "p_value": 1.0}

    def _jarque_bera_test(self, residuals: np.ndarray) -> Dict:
        """Jarque-Bera test for normality"""
        statistic, p_value = stats.jarque_bera(residuals)
        return {"statistic": statistic, "p_value": p_value}

    def _arch_lm_test(self, residuals: np.ndarray, lags: int = 5) -> Dict:
        """ARCH LM test for remaining ARCH effects"""
        squared_resid = residuals**2

        if len(squared_resid) <= lags + 10:
            return {"statistic": 0, "p_value": 1.0}

        X = np.column_stack([squared_resid[lags - i - 1 : -i - 1] for i in range(lags)])
        y = squared_resid[lags:]

        from sklearn.linear_model import LinearRegression

        model = LinearRegression().fit(X, y)
        r_squared = model.score(X, y)
        lm_stat = len(X) * r_squared
        lm_pval = 1 - stats.chi2.cdf(lm_stat, lags)

        return {"statistic": lm_stat, "p_value": lm_pval}

    def get_info(self) -> Dict:
        """Get model information"""
        return {
            "name": self.name,
            "fitted": self.fitted,
            "parameters": self.params,
            "n_obs": len(self.residuals) if self.residuals is not None else 0,
        }
