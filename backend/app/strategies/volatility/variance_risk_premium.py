from typing import Dict, List, Union

import numpy as np
import pandas as pd
from ...config import DEFAULT_ANNUAL_LOOKBACK
from numpy.lib.stride_tricks import sliding_window_view

from ...strategies.volatility.base_volatility import BaseVolatilityStrategy


class VarianceRiskPremiumStrategy(BaseVolatilityStrategy):
    """
    Variance Risk Premium Strategy

    Sell volatility when implied vol > realized vol (VRP positive).
    Standalone mode: estimates implied vol from high-low range (Parkinson proxy).
    """

    def __init__(
        self,
        lookback_vrp: int = 21,  # Lookback for VRP calculation
        entry_threshold: float = 0.02,  # 2% VRP minimum
        exit_threshold: float = 0.0,  # Exit when VRP turns negative
        position_method: str = "delta_hedged",  # "delta_hedged", "straddle", "strangle"
        max_vega_exposure: float = 10000,  # Max vega exposure
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lookback_vrp = lookback_vrp
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.position_method = position_method
        self.max_vega_exposure = max_vega_exposure

        # VRP tracking
        self.vrp_history = []
        self.implied_vol_history = []
        self.realized_vol_history = []
        self.in_vrp_trade = False

    def generate_signal(self, data: Union[pd.Series, pd.DataFrame]) -> Dict:
        """
        Generate VRP signal for standalone backtesting.

        Uses Parkinson high-low range as implied vol proxy, compares to realized vol.
        Positive VRP (implied > realized) -> long equity (short vol proxy).
        """
        if isinstance(data, pd.DataFrame):
            if "Close" in data.columns:
                prices = data["Close"]
            elif "close" in data.columns:
                prices = data["close"]
            else:
                prices = data.iloc[:, 0]
        else:
            prices = data

        if len(prices) < self.lookback_vrp + 5:
            return {"signal": 0, "position_size": 0.0, "metadata": {"strategy": "variance_risk_premium"}}

        returns = self.calculate_returns(prices)
        realized_vol = self.estimate_volatility(returns.iloc[-self.lookback_vrp :])

        # Estimate implied vol proxy: Parkinson (high-low) or scaled short-term vol
        if isinstance(data, pd.DataFrame) and "High" in data.columns and "Low" in data.columns:
            high = data["High"].iloc[-self.lookback_vrp :]
            low = data["Low"].iloc[-self.lookback_vrp :]
            hl_ratio = np.log(high / low)
            implied_vol = np.sqrt((1 / (4 * np.log(2))) * (hl_ratio**2).mean()) * np.sqrt(DEFAULT_ANNUAL_LOOKBACK)
        else:
            # Use short-term vol as proxy (5-day) scaled up
            short_vol = returns.iloc[-5:].std() * np.sqrt(DEFAULT_ANNUAL_LOOKBACK) if len(returns) >= 5 else realized_vol
            implied_vol = short_vol * 1.15  # Implied typically trades at premium

        signal_result = self.generate_vrp_signal(implied_vol, realized_vol, self.vrp_history)

        return signal_result

    def generate_signals_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """Vectorized signal generation based on VRP."""
        close = data["Close"] if "Close" in data.columns else data.get("close", data.iloc[:, 0])
        signals = pd.Series(0, index=data.index)

        returns = np.log(close / close.shift(1))
        realized_vol = returns.rolling(self.lookback_vrp).std() * np.sqrt(DEFAULT_ANNUAL_LOOKBACK)

        # Implied vol proxy: use Parkinson if OHLC available, else scaled short-term vol
        if "High" in data.columns and "Low" in data.columns:
            hl_ratio = np.log(data["High"] / data["Low"])
            windows = sliding_window_view(hl_ratio, window_shape=self.lookback_vrp)

            ms_rolling = (windows**2).mean(axis=-1)

            padding = np.full(self.lookback_vrp - 1, np.nan)
            ms_padded = np.concatenate([padding, ms_rolling])

            implied_vol = np.sqrt((1 / (4 * np.log(2))) * ms_padded) * np.sqrt(DEFAULT_ANNUAL_LOOKBACK)
        else:
            short_vol = returns.rolling(5).std() * np.sqrt(DEFAULT_ANNUAL_LOOKBACK)
            implied_vol = short_vol * 1.15

        vrp = implied_vol - realized_vol

        # VRP positive (implied > realized by threshold) -> long (sell vol proxy)
        direction = pd.Series(0, index=data.index)
        direction[vrp > self.entry_threshold] = 1
        direction[vrp < self.exit_threshold] = -1

        direction_change = direction != direction.shift(1)
        signals[(direction == 1) & direction_change] = 1
        signals[(direction != 1) & direction_change & (direction.shift(1) == 1)] = -1

        return signals

    def calculate_vrp(
        self,
        implied_vol: Union[float, pd.Series],
        realized_vol: Union[float, pd.Series],
    ) -> float:
        """
        Calculate Variance Risk Premium

        VRP = Implied Volatility - Realized Volatility
        Positive VRP: Implied > Realized (sell vol)
        Negative VRP: Implied < Realized (buy vol)
        """
        vrp = implied_vol - realized_vol
        return vrp

    def generate_vrp_signal(self, implied_vol: float, realized_vol: float, vrp_history: List[float] = None) -> Dict:
        """
        Generate VRP trading signal

        Args:
            implied_vol: Current implied volatility
            realized_vol: Current realized volatility
            vrp_history: Historical VRP values

        Returns:
            Trading signal dictionary
        """
        current_vrp = self.calculate_vrp(implied_vol, realized_vol)

        self.vrp_history.append(current_vrp)
        self.implied_vol_history.append(implied_vol)
        self.realized_vol_history.append(realized_vol)

        signal = 0
        position_size = 0
        vrp_zscore = 0

        if vrp_history and len(vrp_history) >= self.lookback_vrp:
            # Calculate VRP z-score
            vrp_mean = np.mean(vrp_history[-self.lookback_vrp :])
            vrp_std = np.std(vrp_history[-self.lookback_vrp :])

            if vrp_std > 0:
                vrp_zscore = (current_vrp - float(vrp_mean)) / float(vrp_std)

        # Trading logic
        if not self.in_vrp_trade:
            if current_vrp > self.entry_threshold:
                # VRP positive and significant - sell volatility
                signal = -1  # Negative = sell vol
                position_size = min(1.0, current_vrp / 0.05)  # Scale with VRP magnitude
                self.in_vrp_trade = True

        else:
            # Exit conditions
            if current_vrp < self.exit_threshold:
                # VRP turned negative - exit position
                signal = 1  # Positive = buy back vol
                position_size = 1.0
                self.in_vrp_trade = False
            else:
                # Hold position
                signal = -1
                position_size = 1.0

        position_details = self._calculate_vrp_position(signal, position_size, implied_vol)

        return {
            "signal": signal,
            "position_size": position_size,
            "in_vrp_trade": self.in_vrp_trade,
            "current_vrp": current_vrp,
            "vrp_zscore": vrp_zscore,
            "implied_vol": implied_vol,
            "realized_vol": realized_vol,
            "position_details": position_details,
            "metadata": {
                "strategy": "variance_risk_premium",
                "position_method": self.position_method,
            },
        }

    def _calculate_vrp_position(
        self,
        signal: int,
        position_size: float,
        implied_vol: float,
    ) -> Dict:
        """
        Calculate detailed VRP position based on selected method
        """
        if self.position_method == "delta_hedged":
            # Short straddle/strangle with delta hedging
            position = {
                "type": "delta_hedged_short_vol",
                "vega_exposure": position_size * self.max_vega_exposure * signal,
                "target_delta": 0.0,
                "hedge_frequency": "daily",
            }

        elif self.position_method == "straddle":
            # Short ATM straddle
            position = {
                "type": "short_straddle",
                "strike": "atm",
                "vega_exposure": position_size * self.max_vega_exposure * signal,
                "delta": 0.0,  # ATM straddle is delta neutral
            }

        elif self.position_method == "strangle":
            # Short OTM strangle
            position = {
                "type": "short_strangle",
                "call_strike": implied_vol * 1.1,  # 10% OTM
                "put_strike": implied_vol * 0.9,  # 10% OTM
                "vega_exposure": position_size * self.max_vega_exposure * signal * 0.8,
                "delta": 0.0,
            }

        else:
            position = {"type": "generic_vol_short", "exposure": position_size * signal}

        return position
