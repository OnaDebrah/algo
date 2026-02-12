"""
Kaufman's Adaptive Moving Average (KAMA) Strategy
File: strategies/kama_strategy.py

An intelligent trend-following strategy that adapts its speed based on market conditions.
Fast during trends, slow during choppy markets.
"""

from typing import Dict

import pandas as pd

from backend.app.strategies import BaseStrategy


class KAMAStrategy(BaseStrategy):
    """Kaufman's Adaptive Moving Average Strategy"""

    def __init__(
        self,
        period: int = 10,
        fast_ema: int = 2,
        slow_ema: int = 30,
        signal_threshold: float = 0.0,
    ):
        """
        Initialize KAMA strategy

        Args:
            period: Period for efficiency ratio calculation (default: 10)
            fast_ema: Fast EMA constant for trending markets (default: 2)
            slow_ema: Slow EMA constant for choppy markets (default: 30)
            signal_threshold: Minimum price % above/below KAMA for signal (default: 0.0)
        """
        params = {
            "period": period,
            "fast_ema": fast_ema,
            "slow_ema": slow_ema,
            "signal_threshold": signal_threshold,
        }
        super().__init__("KAMA Strategy", params)

    def calculate_kama(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Kaufman's Adaptive Moving Average

        KAMA adapts based on market efficiency:
        - Efficiency Ratio (ER) = |net change| / sum of absolute changes
        - ER near 1 = trending (KAMA moves fast)
        - ER near 0 = choppy (KAMA moves slow)
        """
        close = data["Close"].copy()
        period = self.params["period"]

        # Calculate Efficiency Ratio (ER)
        # ER = |Price Change| / Sum of Absolute Price Changes
        change = abs(close - close.shift(period))
        volatility = close.diff().abs().rolling(window=period).sum()

        # Avoid division by zero
        efficiency_ratio = change / volatility
        efficiency_ratio = efficiency_ratio.fillna(0)

        # Calculate Smoothing Constant (SC)
        # SC = [ER * (fastest - slowest) + slowest]^2
        fastest = 2 / (self.params["fast_ema"] + 1)
        slowest = 2 / (self.params["slow_ema"] + 1)

        smoothing_constant = (efficiency_ratio * (fastest - slowest) + slowest) ** 2

        # Calculate KAMA
        kama = pd.Series(index=close.index, dtype=float)
        kama.iloc[period] = close.iloc[period]  # Initialize with first valid price

        for i in range(period + 1, len(close)):
            if pd.notna(kama.iloc[i - 1]):
                sc = smoothing_constant.iloc[i]
                kama.iloc[i] = kama.iloc[i - 1] + sc * (close.iloc[i] - kama.iloc[i - 1])
            else:
                kama.iloc[i] = close.iloc[i]

        return kama

    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        Generate signal based on price position relative to KAMA

        Buy signal: Price crosses above KAMA (+ optional threshold)
        Sell signal: Price crosses below KAMA (+ optional threshold)

        Returns:
            1: Buy signal
            -1: Sell signal
            0: Hold/No signal
        """
        if len(data) < self.params["period"] + 2:
            return 0

        kama = self.calculate_kama(data)

        if kama.isna().all():
            return 0

        # Get current and previous values
        current_price = data["Close"].iloc[-1]
        prev_price = data["Close"].iloc[-2]
        current_kama = kama.iloc[-1]
        prev_kama = kama.iloc[-2]

        if pd.isna(current_kama) or pd.isna(prev_kama):
            return 0

        # Apply signal threshold (optional filter to reduce whipsaws)
        threshold = self.params["signal_threshold"]

        # Buy signal: Price crosses above KAMA (with threshold)
        if prev_price <= prev_kama and current_price > current_kama * (1 + threshold):
            return 1

        # Sell signal: Price crosses below KAMA (with threshold)
        if prev_price >= prev_kama and current_price < current_kama * (1 - threshold):
            return -1

        return 0

    def generate_signals_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """Vectorized KAMA crossover signal generation"""
        close = data["Close"]
        kama = self.calculate_kama(data)
        threshold = self.params["signal_threshold"]

        signals = pd.Series(0, index=data.index)
        prev_close = close.shift(1)
        prev_kama = kama.shift(1)

        # Buy: Price crosses above KAMA (with threshold)
        signals[(prev_close <= prev_kama) & (close > kama * (1 + threshold))] = 1

        # Sell: Price crosses below KAMA (with threshold)
        signals[(prev_close >= prev_kama) & (close < kama * (1 - threshold))] = -1

        return signals

    def get_indicator_values(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Get KAMA and additional indicators for analysis/visualization

        Returns:
            Dictionary with:
                - kama: KAMA values
                - efficiency_ratio: Market efficiency (0-1)
                - price: Close prices
        """
        close = data["Close"].copy()
        period = self.params["period"]

        # Calculate efficiency ratio for analysis
        change = abs(close - close.shift(period))
        volatility = close.diff().abs().rolling(window=period).sum()
        efficiency_ratio = change / volatility
        efficiency_ratio = efficiency_ratio.fillna(0)

        kama = self.calculate_kama(data)

        return {
            "kama": kama,
            "efficiency_ratio": efficiency_ratio,
            "price": close,
        }


# ============================================================
# USAGE EXAMPLES
# ============================================================

"""
# Basic KAMA Strategy
kama = KAMAStrategy(
    period=10,          # Efficiency ratio period
    fast_ema=2,        # Fast during trends
    slow_ema=30,       # Slow during chop
    signal_threshold=0.005  # 0.5% threshold to reduce whipsaws
)

# Multi-Timeframe KAMA (more conservative)
mtf_kama = MultiTimeframeKAMAStrategy(
    short_period=5,    # Fast KAMA
    long_period=20,    # Slow KAMA
    fast_ema=2,
    slow_ema=30
)

# Generate signals
signal = kama.generate_signal(data)

# Get indicator values for plotting
indicators = kama.get_indicator_values(data)
# Plot: indicators['price'], indicators['kama'], indicators['efficiency_ratio']
"""
