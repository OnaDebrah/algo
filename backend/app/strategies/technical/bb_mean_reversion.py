"""
Bollinger Bands Mean Reversion Strategy
"""

import pandas as pd

from ...strategies import BaseStrategy


class BollingerMeanReversionStrategy(BaseStrategy):
    """Bollinger Bands Mean Reversion Strategy"""

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        Initialize Bollinger Bands Mean Reversion strategy

        Args:
            period: Moving average period
            std_dev: Standard deviations for bands
        """
        # Ensure period is integer
        period = int(float(period))

        params = {"period": period, "std_dev": std_dev}
        super().__init__("Bollinger Mean Reversion", params)

    def calculate_bollinger_bands(self, data: pd.DataFrame):
        """Calculate Bollinger Bands"""
        sma = data["Close"].rolling(window=self.params["period"]).mean()
        std = data["Close"].rolling(window=self.params["period"]).std()

        upper_band = sma + (std * self.params["std_dev"])
        lower_band = sma - (std * self.params["std_dev"])

        return sma, upper_band, lower_band

    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        Generate signal based on Bollinger Bands Mean Reversion

        Buy when price dips below lower band and then closes back above it (or simply below).
        Sell when price rallies above upper band and then closes back below it.

        For simplicity and standard mean reversion:
        - Buy: Close < Lower Band (Betting on reversion up)
        - Sell: Close > Upper Band (Betting on reversion down)

        Alternative (Crossing back):
        - Buy: Previous Close < Lower Band AND Current Close > Lower Band
        - Sell: Previous Close > Upper Band AND Current Close < Upper Band
        """
        if len(data) < self.params["period"]:
            return 0

        sma, upper_band, lower_band = self.calculate_bollinger_bands(data)

        if len(upper_band) < 2:
            return 0

        current_close = data["Close"].iloc[-1]
        prev_close = data["Close"].iloc[-2]

        current_upper = upper_band.iloc[-1]
        prev_upper = upper_band.iloc[-2]

        current_lower = lower_band.iloc[-1]
        prev_lower = lower_band.iloc[-2]

        # Mean Reversion Buy: Price was below lower band, now crossing back above
        if prev_close < prev_lower and current_close > current_lower:
            return 1

        # Mean Reversion Sell: Price was above upper band, now crossing back below
        if prev_close > prev_upper and current_close < current_upper:
            return -1

        return 0

    def generate_signals_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """Vectorized Bollinger Bands mean reversion signal generation"""
        sma, upper_band, lower_band = self.calculate_bollinger_bands(data)
        close = data["Close"]
        prev_close = close.shift(1)
        prev_upper = upper_band.shift(1)
        prev_lower = lower_band.shift(1)

        signals = pd.Series(0, index=data.index)

        # Mean Reversion Buy: Price was below lower band, now crossing back above
        signals[(prev_close < prev_lower) & (close > lower_band)] = 1

        # Mean Reversion Sell: Price was above upper band, now crossing back below
        signals[(prev_close > prev_upper) & (close < upper_band)] = -1

        return signals
