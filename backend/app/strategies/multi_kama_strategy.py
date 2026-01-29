import pandas as pd

from backend.app.strategies import BaseStrategy


class MultiTimeframeKAMAStrategy(BaseStrategy):
    """
    KAMA strategy using multiple timeframes for confirmation

    Only generates signals when short and long-term KAMAs align
    """

    def __init__(
        self,
        short_period: int = 5,
        long_period: int = 20,
        fast_ema: int = 2,
        slow_ema: int = 30,
    ):
        """
        Initialize Multi-Timeframe KAMA strategy

        Args:
            short_period: Period for short-term KAMA (default: 5)
            long_period: Period for long-term KAMA (default: 20)
            fast_ema: Fast EMA constant (default: 2)
            slow_ema: Slow EMA constant (default: 30)
        """
        params = {
            "short_period": short_period,
            "long_period": long_period,
            "fast_ema": fast_ema,
            "slow_ema": slow_ema,
        }
        super().__init__("Multi-Timeframe KAMA Strategy", params)

    def calculate_kama(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate KAMA with specified period"""
        close = data["Close"].copy()

        change = abs(close - close.shift(period))
        volatility = close.diff().abs().rolling(window=period).sum()
        efficiency_ratio = change / volatility
        efficiency_ratio = efficiency_ratio.fillna(0)

        fastest = 2 / (self.params["fast_ema"] + 1)
        slowest = 2 / (self.params["slow_ema"] + 1)
        smoothing_constant = (efficiency_ratio * (fastest - slowest) + slowest) ** 2

        kama = pd.Series(index=close.index, dtype=float)
        kama.iloc[period] = close.iloc[period]

        for i in range(period + 1, len(close)):
            if pd.notna(kama.iloc[i - 1]):
                sc = smoothing_constant.iloc[i]
                kama.iloc[i] = kama.iloc[i - 1] + sc * (close.iloc[i] - kama.iloc[i - 1])
            else:
                kama.iloc[i] = close.iloc[i]

        return kama

    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        Generate signal when both short and long-term KAMAs agree

        Buy: Price above both KAMAs and short KAMA > long KAMA
        Sell: Price below both KAMAs and short KAMA < long KAMA
        """
        if len(data) < self.params["long_period"] + 2:
            return 0

        short_kama = self.calculate_kama(data, self.params["short_period"])
        long_kama = self.calculate_kama(data, self.params["long_period"])

        current_price = data["Close"].iloc[-1]
        current_short_kama = short_kama.iloc[-1]
        current_long_kama = long_kama.iloc[-1]

        prev_price = data["Close"].iloc[-2]
        prev_short_kama = short_kama.iloc[-2]
        prev_long_kama = long_kama.iloc[-2]

        if any(pd.isna([current_short_kama, current_long_kama, prev_short_kama, prev_long_kama])):
            return 0

        # Buy: Price crosses above both KAMAs and short > long (uptrend)
        if (
            prev_price <= prev_short_kama
            and current_price > current_short_kama
            and current_price > current_long_kama
            and current_short_kama > current_long_kama
        ):
            return 1

        # Sell: Price crosses below both KAMAs and short < long (downtrend)
        if (
            prev_price >= prev_short_kama
            and current_price < current_short_kama
            and current_price < current_long_kama
            and current_short_kama < current_long_kama
        ):
            return -1

        return 0
