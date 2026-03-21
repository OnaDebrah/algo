"""
Volatility-Based Trading Strategies
Alpha source: Risk premia, volatility regimes, term structure
Examples: Volatility breakout, Vol targeting portfolios, Variance risk premium (sell vol proxy)
Signal: Position ∝ target_vol / realized_vol
Why add it: Natural risk control, Very clean integration with risk manager, Useful for portfolio-level control
"""

import warnings
from typing import Dict, Union, cast

import numpy as np
import pandas as pd

from ...config import DEFAULT_ANNUAL_LOOKBACK
from ...strategies import BaseStrategy

warnings.filterwarnings("ignore")


class BaseVolatilityStrategy(BaseStrategy):
    """
    Base Volatility Strategy Class

    Core principle: Position sizing inversely proportional to volatility
    Position ∝ target_vol / realized_vol
    """

    def __init__(
        self,
        target_volatility: float = 0.15,  # 15% annualized
        vol_lookback: int = 63,  # ~3 months
        min_vol: float = 0.05,  # 5% min volatility floor
        max_vol: float = 0.50,  # 50% max volatility cap
        rebalance_freq: str = "daily",
        vol_estimator: str = "garch",  # "ewma", "garch", "parkinson", "yang_zhang"
        **kwargs,
    ):
        """
        Initialize Volatility Strategy

        Args:
            target_volatility: Annualized target volatility
            vol_lookback: Lookback period for volatility estimation
            min_vol: Minimum volatility (floor)
            max_vol: Maximum volatility (cap)
            rebalance_freq: Rebalancing frequency
            vol_estimator: Method for volatility estimation
        """
        self.target_volatility = target_volatility
        self.vol_lookback = vol_lookback
        self.min_vol = min_vol
        self.max_vol = max_vol
        self.rebalance_freq = rebalance_freq
        self.vol_estimator = vol_estimator

        # State tracking
        self.current_position = 0.0
        self.current_leverage = 1.0
        self.volatility_history = []
        self.leverage_history = []

        # Performance metrics
        self.metrics = {
            "annualized_vol": 0.0,
            "realized_vol": 0.0,
            "vol_target_hit_rate": 0.0,
            "max_leverage": 0.0,
            "min_leverage": 0.0,
        }

        params = {
            "target_volatility": target_volatility,
            "vol_lookback": vol_lookback,
            "min_vol": min_vol,
            "max_vol": max_vol,
            "rebalance_freq": rebalance_freq,
            "vol_estimator": vol_estimator,
        }
        super().__init__("Base Volatility Strategy", params)

    def generate_signal(self, data: pd.DataFrame) -> Union[int, Dict]:
        pass

    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate logarithmic returns"""
        return cast(pd.Series, cast(object, np.log(prices / prices.shift(1))))

    def estimate_volatility(self, returns: pd.Series) -> float:
        """
        Estimate volatility using specified method

        Args:
            returns: Return series

        Returns:
            Annualized volatility estimate
        """
        if len(returns) < 2:
            return self.min_vol

        returns_clean = returns.dropna()

        if self.vol_estimator == "simple":
            # Simple historical volatility
            vol = returns_clean.std() * np.sqrt(DEFAULT_ANNUAL_LOOKBACK)

        elif self.vol_estimator == "ewma":
            # Exponentially Weighted Moving Average (RiskMetrics)
            lambda_ = 0.94  # RiskMetrics decay factor
            weights = (1 - lambda_) * lambda_ ** np.arange(len(returns_clean))[::-1]
            weights = weights / weights.sum()

            mean_return = (returns_clean * weights).sum()
            vol = np.sqrt(((weights * (returns_clean - mean_return) ** 2).sum())) * np.sqrt(DEFAULT_ANNUAL_LOOKBACK)

        elif self.vol_estimator == "parkinson":
            # Parkinson estimator using high-low range
            # Requires OHLC data
            if hasattr(self, "high_prices") and hasattr(self, "low_prices"):
                h = np.log(self.high_prices / self.low_prices)
                parkinson_vol = np.sqrt((1 / (4 * np.log(2))) * (h**2).mean())
                vol = parkinson_vol * np.sqrt(DEFAULT_ANNUAL_LOOKBACK)
            else:
                vol = returns_clean.std() * np.sqrt(DEFAULT_ANNUAL_LOOKBACK)

        elif self.vol_estimator == "garch":
            # Simplified GARCH(1,1) estimate
            vol = self._garch_volatility(returns_clean)

        elif self.vol_estimator == "yang_zhang":
            # Yang-Zhang estimator (accounts for opening jumps)
            vol = self._yang_zhang_volatility(returns_clean)

        else:
            vol = returns_clean.std() * np.sqrt(DEFAULT_ANNUAL_LOOKBACK)

        # Apply volatility bounds
        vol = np.clip(vol, self.min_vol, self.max_vol)

        return vol

    def _garch_volatility(self, returns: pd.Series) -> float:
        """Simplified GARCH(1,1) volatility estimation"""
        omega = 0.000001  # Long-run variance
        alpha = 0.1  # ARCH parameter
        beta = 0.85  # GARCH parameter

        # Initialize
        n = len(returns)
        variances = np.zeros(n)

        if n > 0:
            # Initial variance (unconditional)
            variances[0] = returns.var()

            # GARCH recursion
            for t in range(1, n):
                variances[t] = omega + alpha * returns.iloc[t - 1] ** 2 + beta * variances[t - 1]

            # Current volatility (annualized)
            vol = np.sqrt(variances[-1]) * np.sqrt(DEFAULT_ANNUAL_LOOKBACK)
        else:
            vol = self.min_vol

        return vol

    def _yang_zhang_volatility(self, data: pd.DataFrame, window: int = 22) -> float:
        """
        Full Yang-Zhang Volatility Estimator (2000).
        The most efficient estimator for handling overnight gaps and intraday drift.

        Args:
            data: DataFrame with 'open', 'high', 'low', 'close'
            window: The lookback period (N)

        Returns:
            Annualized Yang-Zhang Volatility
        """
        if len(data) < window:
            return 0.3  # Default or fallback

        # Calculate Log Returns
        # u: Open-to-Close (Intraday)
        # d: Close-to-Open (Overnight Gap)
        # c: Close-to-Close
        o = data["open"]
        h = data["high"]
        low = data["low"]
        c = data["close"]
        c_prev = c.shift(1)

        u = cast(pd.DataFrame, cast(object, np.log(c / o)))
        d = cast(pd.DataFrame, cast(object, np.log(o / c_prev)))

        # Calculate Components over the window
        # Overnight Variance (V_overnight)
        v_overnight = d.tail(window).var()

        # Open-to-Close Variance (V_open_to_close)
        v_open_to_close = u.tail(window).var()

        # Rogers-Satchell Variance (V_rs)
        # This captures intraday volatility using High/Low/Open/Close
        rs_elements = cast(pd.DataFrame, cast(object, (np.log(h / c) * np.log(h / o)) + (np.log(low / c) * np.log(low / o))))
        v_rs = rs_elements.tail(window).mean()

        # Calculate Weight (k)
        # This constant minimizes the overall estimation error
        n = window
        k = 0.34 / (1.34 + (n + 1) / (n - 1))

        # Combined Yang-Zhang Variance
        # Sigma^2 = Var_overnight + k * Var_open_to_close + (1-k) * Var_rs
        yz_var = v_overnight + (k * v_open_to_close) + ((1 - k) * v_rs)

        # Ensure variance is positive before sqrt
        annual_days = getattr(self, "annual_lookback", 252)
        vol = np.sqrt(max(yz_var, 0)) * np.sqrt(annual_days)

        return float(vol)

    def calculate_leverage(self, current_vol: float) -> float:
        """
        Calculate position leverage based on volatility

        Position ∝ target_vol / realized_vol
        """
        if current_vol <= 0:
            return 1.0

        leverage = self.target_volatility / current_vol

        # Apply sensible bounds
        leverage = np.clip(leverage, 0.1, 3.0)  # 10% to 300% position

        return leverage

    def update_volatility_state(self, returns: pd.Series):
        """Update volatility estimates and state"""
        current_vol = self.estimate_volatility(returns)
        self.volatility_history.append(current_vol)

        # Update leverage
        self.current_leverage = self.calculate_leverage(current_vol)
        self.leverage_history.append(self.current_leverage)

        # Update metrics
        if len(self.volatility_history) > 20:
            self.metrics["realized_vol"] = float(np.mean(self.volatility_history[-20:]))
            self.metrics["vol_target_hit_rate"] = float(
                np.mean([abs(v - self.target_volatility) / self.target_volatility < 0.2 for v in self.volatility_history[-20:]])
            )
            self.metrics["max_leverage"] = max(self.leverage_history)
            self.metrics["min_leverage"] = min(self.leverage_history)
