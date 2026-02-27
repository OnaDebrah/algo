"""
Parabolic SAR Strategy
"""

import pandas as pd

from ..strategies import BaseStrategy


class ParabolicSARStrategy(BaseStrategy):
    """Parabolic SAR Strategy"""

    def __init__(self, start: float = 0.02, increment: float = 0.02, maximum: float = 0.2):
        """
        Initialize Parabolic SAR strategy

        Args:
            start: Starting acceleration factor
            increment: Acceleration factor increment
            maximum: Maximum acceleration factor
        """
        params = {"start": start, "increment": increment, "maximum": maximum}
        super().__init__("Parabolic SAR", params)

    def calculate_sar(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Parabolic SAR
        Note: This is a simplified iterative implementation.
        """
        high = data["High"]
        low = data["Low"]
        close = data["Close"]

        sar = pd.Series(index=data.index, dtype=float)
        trend = pd.Series(index=data.index, dtype=int)  # 1 for up, -1 for down
        ep = pd.Series(index=data.index, dtype=float)  # Extreme Point
        af = pd.Series(index=data.index, dtype=float)  # Acceleration Factor

        # Initialize
        trend.iloc[0] = 1 if close.iloc[0] > close.iloc[0] else 1  # Default to up for start or check
        sar.iloc[0] = low.iloc[0] if trend.iloc[0] == 1 else high.iloc[0]
        ep.iloc[0] = high.iloc[0] if trend.iloc[0] == 1 else low.iloc[0]
        af.iloc[0] = self.params["start"]

        # Simple calculation loop - not efficient for massive datasets but works for standard OHLCV backtests
        # Optimization would require numba or vectorized logic which is complex for SAR

        for i in range(1, len(data)):
            prev_sar = sar.iloc[i - 1]
            prev_af = af.iloc[i - 1]
            prev_ep = ep.iloc[i - 1]
            prev_trend = trend.iloc[i - 1]

            # Calculate tentative SAR
            new_sar = prev_sar + prev_af * (prev_ep - prev_sar)

            curr_trend = prev_trend

            if prev_trend == 1:  # Uptrend
                # Check for reversal
                if low.iloc[i] < new_sar:
                    curr_trend = -1
                    new_sar = prev_ep  # SAR becomes previous EP
                    new_ep = low.iloc[i]
                    new_af = self.params["start"]
                else:
                    new_ep = max(prev_ep, high.iloc[i])
                    if new_ep > prev_ep and prev_af < self.params["maximum"]:
                        new_af = min(prev_af + self.params["increment"], self.params["maximum"])
                    else:
                        new_af = prev_af

                    # SAR cannot be higher than previous two lows in potential uptrend ?? Standard rule check
                    # Standard SAR rule: SAR(i) <= Low(i-1) AND SAR(i) <= Low(i-2) for uptrend
                    pass  # Simplified for now

            else:  # Downtrend
                # Check for reversal
                if high.iloc[i] > new_sar:
                    curr_trend = 1
                    new_sar = prev_ep
                    new_ep = high.iloc[i]
                    new_af = self.params["start"]
                else:
                    new_ep = min(prev_ep, low.iloc[i])
                    if new_ep < prev_ep and prev_af < self.params["maximum"]:
                        new_af = min(prev_af + self.params["increment"], self.params["maximum"])
                    else:
                        new_af = prev_af

            sar.iloc[i] = new_sar
            trend.iloc[i] = curr_trend
            ep.iloc[i] = new_ep
            af.iloc[i] = new_af

        return sar

    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        Generate signal based on Parabolic SAR

        Buy when Price crosses above SAR (Trend flips to Up)
        Sell when Price crosses below SAR (Trend flips to Down)
        """
        if len(data) < 5:  # Need a few bars
            return 0

        sar = self.calculate_sar(data)

        if len(sar) < 2:
            return 0

        current_close = data["Close"].iloc[-1]
        prev_close = data["Close"].iloc[-2]
        current_sar = sar.iloc[-1]
        prev_sar = sar.iloc[-2]

        # Note: In a live "generate_signal" context, we might re-run SAR or just check the last values.
        # Since SAR depends on history, we ideally run it on the whole window.

        # Check Trend Flip
        # Uptrend: Close > SAR
        # Downtrend: Close < SAR

        # Bullish Flip: Previous Close < Previous SAR (Downtrend) AND Current Close > Current SAR (Uptrend)
        # OR more robustly: SAR flip logic in calculation

        # Let's inspect the SAR relative position

        is_uptrend = current_close > current_sar
        was_uptrend = prev_close > prev_sar

        if is_uptrend and not was_uptrend:
            return 1
        elif not is_uptrend and was_uptrend:
            return -1

        return 0

    def generate_signals_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """Vectorized Parabolic SAR signal generation"""
        sar = self.calculate_sar(data)
        close = data["Close"]

        is_uptrend = close > sar
        was_uptrend = close.shift(1) > sar.shift(1)

        signals = pd.Series(0, index=data.index)

        # Bullish flip: was downtrend, now uptrend
        signals[is_uptrend & ~was_uptrend] = 1

        # Bearish flip: was uptrend, now downtrend
        signals[~is_uptrend & was_uptrend] = -1

        return signals
