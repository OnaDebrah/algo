from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd

from streamlit.strategies.volatility.base_volatility import BaseVolatilityStrategy


class VolatilityBreakoutStrategy(BaseVolatilityStrategy):
    """
    Volatility Breakout Strategy

    Trades breakouts when volatility exceeds certain thresholds
    """

    def __init__(
        self,
        vol_multiplier: float = 2.0,  # Volatility threshold multiplier
        trend_lookback: int = 20,  # Trend detection period
        atr_period: int = 14,  # ATR period for stop losses
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vol_multiplier = vol_multiplier
        self.trend_lookback = trend_lookback
        self.atr_period = atr_period

        # Breakout specific state
        self.volatility_bands = {"upper": None, "lower": None}
        self.in_breakout = False
        self.breakout_direction = 0

    def detect_breakout(self, prices: pd.Series, returns: pd.Series) -> Tuple[bool, int]:
        """
        Detect volatility breakout

        Returns:
            Tuple of (breakout_detected, direction)
        """
        if len(returns) < self.vol_lookback:
            return False, 0

        # Calculate current volatility
        current_vol = self.estimate_volatility(returns)

        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=self.vol_lookback).std().dropna() * np.sqrt(252)

        if len(rolling_vol) < 1:
            return False, 0

        # Calculate volatility bands
        vol_mean = rolling_vol.mean()
        vol_std = rolling_vol.std()

        upper_band = vol_mean + self.vol_multiplier * vol_std
        lower_band = vol_mean - self.vol_multiplier * vol_std

        self.volatility_bands = {"upper": upper_band, "lower": lower_band}

        # Check for breakout
        if current_vol > upper_band:
            # High volatility breakout
            # Check trend direction
            if len(prices) >= self.trend_lookback:
                recent_trend = prices.iloc[-1] / prices.iloc[-self.trend_lookback] - 1

                if recent_trend > 0:
                    return True, 1  # Bullish breakout
                else:
                    return True, -1  # Bearish breakout (volatility spike in downtrend)

        elif current_vol < lower_band:
            # Low volatility regime - potential for impending breakout
            # Could be used for mean reversion or as warning signal
            return False, 0

        return False, 0

    def generate_signal(self, data: Union[pd.Series, pd.DataFrame]) -> Dict:
        """
        Generate volatility breakout signal
        """
        if isinstance(data, pd.DataFrame):
            if "close" in data.columns:
                prices = data["close"]
            else:
                prices = data.iloc[:, 0]
        else:
            prices = data

        # Calculate returns
        returns = self.calculate_returns(prices)

        # Update volatility state
        self.update_volatility_state(returns)

        # Detect breakout
        breakout_detected, direction = self.detect_breakout(prices, returns)

        signal = 0
        position_size = 0

        if breakout_detected:
            # Enter breakout position
            signal = direction
            position_size = self.current_leverage * abs(signal)

            # Update state
            self.in_breakout = True
            self.breakout_direction = direction
            self.current_position = position_size * direction

        elif self.in_breakout:
            # Check for breakout end
            current_vol = self.volatility_history[-1] if self.volatility_history else 0
            vol_mean = np.mean(self.volatility_history[-self.vol_lookback :]) if len(self.volatility_history) >= self.vol_lookback else current_vol

            # Exit if volatility returns to normal
            if abs(current_vol - vol_mean) / vol_mean < 0.5:  # Within 50% of mean
                signal = -self.breakout_direction
                position_size = 0
                self.in_breakout = False
                self.breakout_direction = 0
                self.current_position = 0
            else:
                # Hold position
                signal = self.breakout_direction
                position_size = abs(self.current_position)

        return {
            "signal": signal,
            "position_size": position_size,
            "in_breakout": self.in_breakout,
            "breakout_direction": self.breakout_direction,
            "current_volatility": (self.volatility_history[-1] if self.volatility_history else 0),
            "leverage": self.current_leverage,
            "volatility_bands": self.volatility_bands,
            "metadata": {
                "strategy": "volatility_breakout",
                "vol_estimator": self.vol_estimator,
            },
        }
