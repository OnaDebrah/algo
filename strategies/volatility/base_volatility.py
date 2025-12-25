"""
Volatility-Based Trading Strategies
Alpha source: Risk premia, volatility regimes, term structure
Examples: Volatility breakout, Vol targeting portfolios, Variance risk premium (sell vol proxy)
Signal: Position ∝ target_vol / realized_vol
Why add it: Natural risk control, Very clean integration with risk manager, Useful for portfolio-level control
"""

import warnings

import numpy as np
import pandas as pd

from strategies.volatility.dynamic_scaling import DynamicVolatilityScaling
from strategies.volatility.variance_risk_premium import VarianceRiskPremiumStrategy
from strategies.volatility.volatility_breakout import VolatilityBreakoutStrategy
from strategies.volatility.volatility_risk_manager import VolatilityRiskManager
from strategies.volatility.volatility_targeting import VolatilityTargetingStrategy

warnings.filterwarnings("ignore")


class VolatilityStrategy:
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

    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate logarithmic returns"""
        return np.log(prices / prices.shift(1))

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
            vol = returns_clean.std() * np.sqrt(252)

        elif self.vol_estimator == "ewma":
            # Exponentially Weighted Moving Average (RiskMetrics)
            lambda_ = 0.94  # RiskMetrics decay factor
            weights = (1 - lambda_) * lambda_ ** np.arange(len(returns_clean))[::-1]
            weights = weights / weights.sum()

            mean_return = (returns_clean * weights).sum()
            vol = np.sqrt(((weights * (returns_clean - mean_return) ** 2).sum())) * np.sqrt(252)

        elif self.vol_estimator == "parkinson":
            # Parkinson estimator using high-low range
            # Requires OHLC data
            if hasattr(self, "high_prices") and hasattr(self, "low_prices"):
                h = np.log(self.high_prices / self.low_prices)
                parkinson_vol = np.sqrt((1 / (4 * np.log(2))) * (h**2).mean())
                vol = parkinson_vol * np.sqrt(252)
            else:
                vol = returns_clean.std() * np.sqrt(252)

        elif self.vol_estimator == "garch":
            # Simplified GARCH(1,1) estimate
            vol = self._garch_volatility(returns_clean)

        elif self.vol_estimator == "yang_zhang":
            # Yang-Zhang estimator (accounts for opening jumps)
            vol = self._yang_zhang_volatility(returns_clean)

        else:
            vol = returns_clean.std() * np.sqrt(252)

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
            vol = np.sqrt(variances[-1]) * np.sqrt(252)
        else:
            vol = self.min_vol

        return vol

    def _yang_zhang_volatility(self, returns: pd.Series) -> float:
        """
        Yang-Zhang volatility estimator
        Requires open, high, low, close prices
        """
        # Simplified version - in production use full OHLC
        k = 0.34 / (1.34 + (len(returns) + 1) / (len(returns) - 1))

        # Calculate overnight volatility (close to open)
        if hasattr(self, "overnight_returns"):
            var_overnight = self.overnight_returns.var()
        else:
            var_overnight = 0.1 * returns.var()  # Approximation

        # Calculate open-to-close volatility
        var_intraday = returns.var()

        # Yang-Zhang estimator
        vol = np.sqrt(var_overnight + k * var_intraday + (1 - k) * var_intraday) * np.sqrt(252)

        return vol

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
            self.metrics["realized_vol"] = np.mean(self.volatility_history[-20:])
            self.metrics["vol_target_hit_rate"] = np.mean(
                [abs(v - self.target_volatility) / self.target_volatility < 0.2 for v in self.volatility_history[-20:]]
            )
            self.metrics["max_leverage"] = max(self.leverage_history)
            self.metrics["min_leverage"] = min(self.leverage_history)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=== Volatility-Based Strategies Demo ===")

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="D")

    # Simulate volatile asset
    base_returns = np.random.randn(500) * 0.01
    # Add volatility clusters
    for i in range(100, 200):
        base_returns[i] *= 3.0  # High volatility period
    for i in range(300, 400):
        base_returns[i] *= 0.3  # Low volatility period

    prices = 100 * np.exp(np.cumsum(base_returns))
    price_series = pd.Series(prices, index=dates)

    # Example 1: Volatility Breakout
    print("\n1. Volatility Breakout Strategy")
    breakout_strat = VolatilityBreakoutStrategy(target_volatility=0.20, vol_multiplier=2.0, vol_estimator="ewma")

    breakout_signal = breakout_strat.generate_signal(price_series)
    print(f"Breakout Signal: {breakout_signal}")

    # Example 2: Volatility Targeting
    print("\n2. Volatility Targeting Strategy")

    # Simulate a base strategy
    base_returns_sim = pd.Series(np.random.randn(500) * 0.01, index=dates)

    vol_target_strat = VolatilityTargetingStrategy(target_volatility=0.15, volatility_buffer=0.1)

    # Calculate required leverage
    required_leverage = vol_target_strat.calculate_required_leverage(base_returns_sim)
    print(f"Required Leverage: {required_leverage:.2f}")

    # Example 3: Variance Risk Premium
    print("\n3. Variance Risk Premium Strategy")

    # Simulate implied and realized vol
    implied_vol = 0.25  # 25% implied
    realized_vol = 0.18  # 18% realized

    vrp_strat = VarianceRiskPremiumStrategy(entry_threshold=0.02, position_method="delta_hedged")

    vrp_signal = vrp_strat.generate_vrp_signal(implied_vol, realized_vol)
    print(f"VRP Signal: {vrp_signal}")

    # Example 4: Dynamic Volatility Scaling
    print("\n4. Dynamic Volatility Scaling")

    scaling = DynamicVolatilityScaling(target_volatility=0.15, scaling_method="multiplicative")

    returns = breakout_strat.calculate_returns(price_series)
    scaled_pos = scaling.scale_position("TEST_ASSET", 1.0, returns)
    print(f"Scaled Position: {scaled_pos:.2f}")

    # Example 5: Integration with Risk Manager
    print("\n5. Volatility Risk Manager Integration")

    risk_config = {
        "target_volatility": 0.15,
        "position_limits": {"TEST_ASSET": {"max": 2.0}},
        "var_limits": {"TEST_ASSET": 10000},
    }

    risk_manager = VolatilityRiskManager(risk_config)

    # Check position risk
    risk_check = risk_manager.check_position_risk(
        "TEST_ASSET",
        position=2.5,
        returns=returns,
        metadata={"portfolio_returns": returns},
    )

    print(f"Risk Check: {risk_check}")
