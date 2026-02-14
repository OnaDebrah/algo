"""
Relative Strength Index (RSI) Strategy
"""

import pandas as pd

from backend.app.strategies import BaseStrategy
from config import DEFAULT_RSI_OVERBOUGHT, DEFAULT_RSI_OVERSOLD, DEFAULT_RSI_PERIOD


class RSIStrategy(BaseStrategy):
    """Relative Strength Index Strategy"""

    def __init__(
        self,
        period: int = DEFAULT_RSI_PERIOD,
        oversold: int = DEFAULT_RSI_OVERSOLD,
        overbought: int = DEFAULT_RSI_OVERBOUGHT,
    ):
        """
        Initialize RSI strategy

        Args:
            period: RSI calculation period
            oversold: Oversold threshold (buy signal)
            overbought: Overbought threshold (sell signal)
        """
        params = {"period": period, "oversold": oversold, "overbought": overbought}
        super().__init__("RSI Strategy", params)

    def calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """Calculate RSI indicator"""
        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.params["period"]).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.params["period"]).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        Generate signal based on RSI levels

        Buy when RSI crosses above oversold level
        Sell when RSI crosses below overbought level
        """
        if len(data) < self.params["period"] + 1:
            return 0

        rsi = self.calculate_rsi(data)

        if len(rsi) < 2:
            return 0

        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]

        # Buy signal: RSI crosses above oversold
        if prev_rsi < self.params["oversold"] and current_rsi >= self.params["oversold"]:
            return 1

        # Sell signal: RSI crosses below overbough
        if prev_rsi > self.params["overbought"] and current_rsi <= self.params["overbought"]:
            return -1

        return 0

    def generate_signals_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """Vectorized version of RSI signal generation"""
        rsi = self.calculate_rsi(data)

        signals = pd.Series(0, index=data.index)

        # Buy: crossing above oversold
        signals[(rsi >= self.params["oversold"]) & (rsi.shift(1) < self.params["oversold"])] = 1

        # Sell: crossing below overbought
        signals[(rsi <= self.params["overbought"]) & (rsi.shift(1) > self.params["overbought"])] = -1

        return signals
