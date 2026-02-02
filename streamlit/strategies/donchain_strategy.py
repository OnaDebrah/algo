"""
Donchian Channel Breakout Strategy
File: strategies/donchian_strategy.py

The classic Turtle Trading strategy. Goes long on N-day highs, exits on M-day lows.
Simple, robust, and effective across timeframes and asset classes.
"""

from typing import Dict

import pandas as pd

from streamlit.strategies.base_strategy import BaseStrategy


class DonchianChannelStrategy(BaseStrategy):
    """
    Donchian Channel Breakout Strategy

    Made famous by Richard Dennis and the Turtle Traders in the 1980s.
    Enters on breakouts from the highest high, exits on lowest low.
    """

    def __init__(
        self,
        entry_period: int = 20,
        exit_period: int = 10,
        use_both_sides: bool = True,
    ):
        """
        Initialize Donchian Channel strategy

        Args:
            entry_period: Period for entry breakout (default: 20)
            exit_period: Period for exit breakout (default: 10)
            use_both_sides: Trade both long and short (default: True)
        """
        params = {
            "entry_period": entry_period,
            "exit_period": exit_period,
            "use_both_sides": use_both_sides,
        }
        super().__init__("Donchian Channel Strategy", params)
        self.position = 0  # Track current position: 1 (long), -1 (short), 0 (flat)

    def calculate_channels(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate Donchian Channel bands

        Returns:
            Dictionary with:
                - upper_entry: N-period highest high
                - lower_entry: N-period lowest low
                - upper_exit: M-period highest high (for short exit)
                - lower_exit: M-period lowest low (for long exit)
        """
        high = data["High"].copy()
        low = data["Low"].copy()

        entry_period = self.params["entry_period"]
        exit_period = self.params["exit_period"]

        # Entry channels (breakout levels)
        upper_entry = high.rolling(window=entry_period).max()
        lower_entry = low.rolling(window=entry_period).min()

        # Exit channels (stop loss levels)
        upper_exit = high.rolling(window=exit_period).max()
        lower_exit = low.rolling(window=exit_period).min()

        return {
            "upper_entry": upper_entry,
            "lower_entry": lower_entry,
            "upper_exit": upper_exit,
            "lower_exit": lower_exit,
        }

    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        Generate signal based on Donchian Channel breakouts

        Entry Rules:
        - Long: Price breaks above N-period high
        - Short: Price breaks below N-period low

        Exit Rules:
        - Exit long: Price breaks below M-period low
        - Exit short: Price breaks above M-period high

        Returns:
            1: Buy signal (go long or exit short)
            -1: Sell signal (go short or exit long)
            0: Hold current position
        """
        entry_period = self.params["entry_period"]
        exit_period = self.params["exit_period"]

        if len(data) < max(entry_period, exit_period) + 1:
            return 0

        channels = self.calculate_channels(data)

        current_high = data["High"].iloc[-1]
        current_low = data["Low"].iloc[-1]

        # Get previous period's channel values (to avoid look-ahead bias)
        prev_upper_entry = channels["upper_entry"].iloc[-2]
        prev_lower_entry = channels["lower_entry"].iloc[-2]
        prev_upper_exit = channels["upper_exit"].iloc[-2]
        prev_lower_exit = channels["lower_exit"].iloc[-2]

        if pd.isna(prev_upper_entry) or pd.isna(prev_lower_entry):
            return 0

        # Check for entry signals
        # Long entry: Break above N-period high
        if current_high > prev_upper_entry:
            if self.position != 1:  # Not already long
                self.position = 1
                return 1

        # Short entry: Break below N-period low
        if self.params["use_both_sides"] and current_low < prev_lower_entry:
            if self.position != -1:  # Not already short
                self.position = -1
                return -1

        # Check for exit signals
        # Exit long: Break below M-period low
        if self.position == 1 and not pd.isna(prev_lower_exit):
            if current_low < prev_lower_exit:
                self.position = 0
                return -1  # Exit long

        # Exit short: Break above M-period high
        if self.position == -1 and not pd.isna(prev_upper_exit):
            if current_high > prev_upper_exit:
                self.position = 0
                return 1  # Exit short

        return 0  # Hold current position

    def get_indicator_values(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Get Donchian Channel values for visualization

        Returns:
            Dictionary with channel bands and middle line
        """
        channels = self.calculate_channels(data)

        # Calculate middle line (average of upper and lower)
        middle = (channels["upper_entry"] + channels["lower_entry"]) / 2

        return {
            "upper_entry": channels["upper_entry"],
            "lower_entry": channels["lower_entry"],
            "upper_exit": channels["upper_exit"],
            "lower_exit": channels["lower_exit"],
            "middle": middle,
            "price": data["Close"],
        }


# ============================================================
# ALTERNATIVE: Donchian with ATR Position Sizing
# ============================================================


class DonchianATRStrategy(BaseStrategy):
    """
    Donchian Channel Strategy with ATR-based position sizing

    Uses Average True Range (ATR) for:
    - Risk-adjusted position sizing
    - Volatility-based stop losses
    """

    def __init__(
        self,
        entry_period: int = 20,
        exit_period: int = 10,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        use_both_sides: bool = True,
    ):
        """
        Initialize Donchian ATR strategy

        Args:
            entry_period: Period for entry breakout (default: 20)
            exit_period: Period for exit breakout (default: 10)
            atr_period: Period for ATR calculation (default: 14)
            atr_multiplier: ATR multiplier for stops (default: 2.0)
            use_both_sides: Trade both long and short (default: True)
        """
        params = {
            "entry_period": entry_period,
            "exit_period": exit_period,
            "atr_period": atr_period,
            "atr_multiplier": atr_multiplier,
            "use_both_sides": use_both_sides,
        }
        super().__init__("Donchian ATR Strategy", params)
        self.position = 0

    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        high = data["High"]
        low = data["Low"]
        close = data["Close"]

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.params["atr_period"]).mean()

        return atr

    def calculate_channels(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Donchian Channel bands"""
        high = data["High"].copy()
        low = data["Low"].copy()

        entry_period = self.params["entry_period"]
        exit_period = self.params["exit_period"]

        upper_entry = high.rolling(window=entry_period).max()
        lower_entry = low.rolling(window=entry_period).min()
        upper_exit = high.rolling(window=exit_period).max()
        lower_exit = low.rolling(window=exit_period).min()

        return {
            "upper_entry": upper_entry,
            "lower_entry": lower_entry,
            "upper_exit": upper_exit,
            "lower_exit": lower_exit,
        }

    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        Generate signal with ATR-based risk management

        Uses ATR to:
        - Confirm breakout strength
        - Set dynamic stop losses
        """
        entry_period = self.params["entry_period"]

        if len(data) < entry_period + self.params["atr_period"]:
            return 0

        channels = self.calculate_channels(data)
        atr = self.calculate_atr(data)

        current_price = data["Close"].iloc[-1]
        current_high = data["High"].iloc[-1]
        current_low = data["Low"].iloc[-1]
        current_atr = atr.iloc[-1]

        prev_upper_entry = channels["upper_entry"].iloc[-2]
        prev_lower_entry = channels["lower_entry"].iloc[-2]
        prev_lower_exit = channels["lower_exit"].iloc[-2]
        prev_upper_exit = channels["upper_exit"].iloc[-2]

        if pd.isna(prev_upper_entry) or pd.isna(current_atr):
            return 0

        # Long entry with ATR filter
        if current_high > prev_upper_entry:
            breakout_strength = (current_high - prev_upper_entry) / current_atr
            if breakout_strength > 0.1 and self.position != 1:  # Minimum 10% ATR breakout
                self.position = 1
                return 1

        # Short entry with ATR filter
        if self.params["use_both_sides"] and current_low < prev_lower_entry:
            breakout_strength = (prev_lower_entry - current_low) / current_atr
            if breakout_strength > 0.1 and self.position != -1:
                self.position = -1
                return -1

        # ATR-based stop loss for long
        if self.position == 1:
            atr_stop = current_price - (current_atr * self.params["atr_multiplier"])
            if current_low < prev_lower_exit or current_low < atr_stop:
                self.position = 0
                return -1

        # ATR-based stop loss for short
        if self.position == -1:
            atr_stop = current_price + (current_atr * self.params["atr_multiplier"])
            if current_high > prev_upper_exit or current_high > atr_stop:
                self.position = 0
                return 1

        return 0

    def get_position_size(self, data: pd.DataFrame, account_value: float, risk_per_trade: float = 0.02) -> float:
        """
        Calculate position size based on ATR

        Args:
            data: Price data
            account_value: Current account value
            risk_per_trade: Risk per trade as fraction (default: 2%)

        Returns:
            Position size in dollars
        """
        atr = self.calculate_atr(data)
        current_atr = atr.iloc[-1]

        if pd.isna(current_atr):
            return 0

        # Risk amount in dollars
        risk_amount = account_value * risk_per_trade

        # Stop distance in dollars (ATR-based)
        stop_distance = current_atr * self.params["atr_multiplier"]

        # Position size = Risk Amount / Stop Distance
        position_size = risk_amount / stop_distance

        return position_size


# ============================================================
# ALTERNATIVE: Filtered Donchian (with trend filter)
# ============================================================


class FilteredDonchianStrategy(BaseStrategy):
    """
    Donchian Channel with trend filter

    Only takes breakouts in the direction of the longer-term trend
    """

    def __init__(
        self,
        entry_period: int = 20,
        exit_period: int = 10,
        trend_period: int = 50,
    ):
        """
        Initialize Filtered Donchian strategy

        Args:
            entry_period: Period for entry breakout (default: 20)
            exit_period: Period for exit breakout (default: 10)
            trend_period: Period for trend filter MA (default: 50)
        """
        params = {
            "entry_period": entry_period,
            "exit_period": exit_period,
            "trend_period": trend_period,
        }
        super().__init__("Filtered Donchian Strategy", params)
        self.position = 0

    def calculate_trend(self, data: pd.DataFrame) -> pd.Series:
        """Calculate trend filter (Simple Moving Average)"""
        return data["Close"].rolling(window=self.params["trend_period"]).mean()

    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        Generate signal only when breakout aligns with trend

        - Only long breakouts when price > MA
        - Only short breakouts when price < MA
        """
        entry_period = self.params["entry_period"]
        trend_period = self.params["trend_period"]

        if len(data) < max(entry_period, trend_period) + 1:
            return 0

        channels = DonchianChannelStrategy(entry_period=entry_period, exit_period=self.params["exit_period"]).calculate_channels(data)

        trend_ma = self.calculate_trend(data)

        current_price = data["Close"].iloc[-1]
        current_high = data["High"].iloc[-1]
        current_low = data["Low"].iloc[-1]
        current_ma = trend_ma.iloc[-1]

        prev_upper_entry = channels["upper_entry"].iloc[-2]
        prev_lower_entry = channels["lower_entry"].iloc[-2]
        prev_lower_exit = channels["lower_exit"].iloc[-2]
        prev_upper_exit = channels["upper_exit"].iloc[-2]

        if pd.isna(current_ma):
            return 0

        # Long entry: Breakout + price above MA (uptrend)
        if current_high > prev_upper_entry and current_price > current_ma:
            if self.position != 1:
                self.position = 1
                return 1

        # Short entry: Breakout + price below MA (downtrend)
        if current_low < prev_lower_entry and current_price < current_ma:
            if self.position != -1:
                self.position = -1
                return -1

        # Exit signals
        if self.position == 1 and current_low < prev_lower_exit:
            self.position = 0
            return -1

        if self.position == -1 and current_high > prev_upper_exit:
            self.position = 0
            return 1

        return 0


# ============================================================
# USAGE EXAMPLES
# ============================================================

"""
# Classic Turtle Trader Setup
donchian = DonchianChannelStrategy(
    entry_period=20,    # Enter on 20-day breakout
    exit_period=10,     # Exit on 10-day breakout
    use_both_sides=True # Trade long and short
)

# With ATR Risk Management
donchian_atr = DonchianATRStrategy(
    entry_period=20,
    exit_period=10,
    atr_period=14,
    atr_multiplier=2.0  # 2x ATR stop loss
)

# With Trend Filter (more conservative)
filtered_donchian = FilteredDonchianStrategy(
    entry_period=20,
    exit_period=10,
    trend_period=50     # Only trade with 50-day trend
)

# Generate signals
signal = donchian.generate_signal(data)

# Get indicator values for plotting
indicators = donchian.get_indicator_values(data)
# Plot: indicators['upper_entry'], indicators['lower_entry'], indicators['middle']

# Calculate position size (for ATR version)
position_size = donchian_atr.get_position_size(data, account_value=100000, risk_per_trade=0.02)
"""
