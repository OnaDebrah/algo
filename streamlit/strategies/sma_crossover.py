"""
Simple Moving Average Crossover Strategy
"""

import pandas as pd

from streamlit.config import DEFAULT_SMA_LONG, DEFAULT_SMA_SHORT
from streamlit.strategies.base_strategy import BaseStrategy


class SMACrossoverStrategy(BaseStrategy):
    """Simple Moving Average Crossover Strategy"""

    def __init__(self, short_window: int = DEFAULT_SMA_SHORT, long_window: int = DEFAULT_SMA_LONG):
        """
        Initialize SMA Crossover strategy

        Args:
            short_window: Short-term moving average window
            long_window: Long-term moving average window
        """
        params = {"short_window": short_window, "long_window": long_window}
        super().__init__("SMA Crossover", params)

    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        Generate signal based on SMA crossover

        Buy when short MA crosses above long MA
        Sell when short MA crosses below long MA
        """
        if len(data) < self.params["long_window"]:
            return 0

        sma_short = data["Close"].rolling(window=self.params["short_window"]).mean()
        sma_long = data["Close"].rolling(window=self.params["long_window"]).mean()

        if len(sma_short) < 2 or len(sma_long) < 2:
            return 0

        # Bullish crossover
        if sma_short.iloc[-2] <= sma_long.iloc[-2] and sma_short.iloc[-1] > sma_long.iloc[-1]:
            return 1

        # Bearish crossover
        elif sma_short.iloc[-2] >= sma_long.iloc[-2] and sma_short.iloc[-1] < sma_long.iloc[-1]:
            return -1

        return 0
