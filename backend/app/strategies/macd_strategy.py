"""
Moving Average Convergence Divergence (MACD) Strategy
"""

import pandas as pd

from backend.app.strategies import BaseStrategy
from config import DEFAULT_MACD_FAST, DEFAULT_MACD_SIGNAL, DEFAULT_MACD_SLOW


class MACDStrategy(BaseStrategy):
    """Moving Average Convergence Divergence Strategy"""

    def __init__(
        self,
        fast: int = DEFAULT_MACD_FAST,
        slow: int = DEFAULT_MACD_SLOW,
        signal: int = DEFAULT_MACD_SIGNAL,
    ):
        """
        Initialize MACD strategy

        Args:
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
        """
        params = {"fast": fast, "slow": slow, "signal": signal}
        super().__init__("MACD Strategy", params)

    def calculate_macd(self, data: pd.DataFrame) -> tuple:
        """
        Calculate MACD indicator

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        ema_fast = data["Close"].ewm(span=self.params["fast"], adjust=False).mean()
        ema_slow = data["Close"].ewm(span=self.params["slow"], adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.params["signal"], adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        Generate signal based on MACD crossover

        Buy when MACD crosses above signal line
        Sell when MACD crosses below signal line
        """
        if len(data) < self.params["slow"] + self.params["signal"]:
            return 0

        macd_line, signal_line, _ = self.calculate_macd(data)

        if len(macd_line) < 2:
            return 0

        # Bullish crossover
        if macd_line.iloc[-2] <= signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
            return 1

        # Bearish crossover
        if macd_line.iloc[-2] >= signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
            return -1

        return 0

    def generate_signals_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """Vectorized MACD crossover signal generation"""
        macd_line, signal_line, _ = self.calculate_macd(data)

        signals = pd.Series(0, index=data.index)

        # Bullish crossover: MACD crosses above signal line
        signals[(macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))] = 1

        # Bearish crossover: MACD crosses below signal line
        signals[(macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))] = -1

        return signals
