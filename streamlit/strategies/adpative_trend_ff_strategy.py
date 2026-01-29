"""
Time-Series Momentum (Trend Following) Strategy
Alpha source: Persistent trends
Used by: AQR, Winton, Man Group
Examples: 3-12 month momentum, Dual moving average crossover, Volatility-scaled trend
Signal: sign(P_t − MA_t)
Why add it: Extremely robust, Works across asset classes, Excellent for drawdown diversification
"""

import warnings
from typing import Dict, Union

import numpy as np
import pandas as pd

from streamlit.strategies.ts_momentum_strategy import TimeSeriesMomentumStrategy

warnings.filterwarnings("ignore")


class AdaptiveTrendFollowingStrategy(TimeSeriesMomentumStrategy):
    """
    Adaptive Trend Following Strategy
    Dynamically adjusts parameters based on market regime
    """

    def __init__(self, asset_symbol: str, **kwargs):
        super().__init__(asset_symbol, **kwargs)

        # Regime detection parameters
        self.volatility_regimes = ["low", "medium", "high"]
        self.trend_regimes = ["strong_trend", "weak_trend", "choppy"]

        # Adaptive parameters
        self.adaptive_lookbacks = {
            "low_vol": {"fast": 10, "slow": 30},
            "medium_vol": {"fast": 20, "slow": 50},
            "high_vol": {"fast": 40, "slow": 100},
        }

    def detect_market_regime(self, prices: pd.Series) -> Dict:
        """
        Detect current market regime using statistical tests
        """
        if len(prices) < 100:
            return {"volatility": "medium", "trend": "unknown"}

        # Calculate volatility regime (existing code remains)
        returns = self._calculate_returns(prices)
        recent_vol = self._calculate_volatility(returns.iloc[-63:] if len(returns) >= 63 else returns)

        if recent_vol < 0.1:
            vol_regime = "low_vol"
        elif recent_vol < 0.25:
            vol_regime = "medium_vol"
        else:
            vol_regime = "high_vol"

        # IMPROVED TREND DETECTION
        recent_prices = prices.iloc[-252:] if len(prices) >= 252 else prices

        if len(recent_prices) >= 60:  # Need enough data for meaningful tests
            # Option 1: Simple price-based trend detection (current approach)
            trend_strength = abs(recent_prices.iloc[-1] / recent_prices.iloc[-60] - 1)

            # Option 2: Linear regression slope for trend direction
            from scipy import stats

            x = np.arange(len(recent_prices))
            slope, _, r_value, _, _ = stats.linregress(x, recent_prices.values)
            r_squared = r_value**2

            # Classify based on R² (explained variance by trend)
            if r_squared > 0.7:
                trend_type = "strong_trend"
            elif r_squared > 0.3:
                trend_type = "weak_trend"
            else:
                trend_type = "choppy"

            # Add direction
            direction = "up" if slope > 0 else "down"
            trend_regime = f"{trend_type}_{direction}"

        else:
            trend_regime = "unknown"

        return {
            "volatility": vol_regime,
            "trend": trend_regime,
            "recent_volatility": recent_vol,
            "trend_strength": trend_strength if "trend_strength" in locals() else None,
        }

    def generate_signal(self, data: Union[pd.DataFrame, pd.Series], **kwargs) -> Dict:
        """
        Generate adaptive trend following signal
        """
        # Extract price series
        if isinstance(data, pd.DataFrame):
            prices = data["close"] if "close" in data.columns else data.iloc[:, 0]
        else:
            prices = data

        # Detect market regime
        regime = self.detect_market_regime(prices)

        # Adjust parameters based on regime
        if regime["volatility"] in self.adaptive_lookbacks:
            params = self.adaptive_lookbacks[regime["volatility"]]
            self.fast_period = params["fast"]
            self.slow_period = params["slow"]

        # Adjust position sizing based on regime
        position_multiplier = 1.0
        if regime["trend"] == "strong_trend":
            position_multiplier = 1.2
        elif regime["trend"] == "choppy":
            position_multiplier = 0.5

        # Generate base signal
        base_result = super().generate_signal(data, **kwargs)

        # Adjust position size based on regime
        if "position_size" in base_result:
            base_result["position_size"] *= position_multiplier
            base_result["position_size"] = min(base_result["position_size"], self.max_position)

        # Add regime info to metadata
        base_result["metadata"]["regime"] = regime
        base_result["metadata"]["adaptive_params"] = {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "position_multiplier": position_multiplier,
        }

        return base_result
