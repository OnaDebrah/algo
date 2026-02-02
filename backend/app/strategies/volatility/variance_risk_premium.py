from typing import Dict, List, Union

import numpy as np
import pandas as pd

from backend.app.strategies.volatility.base_volatility import BaseVolatilityStrategy


class VarianceRiskPremiumStrategy(BaseVolatilityStrategy):
    """
    Variance Risk Premium Strategy

    Sell volatility when implied vol > realized vol (VRP positive)
    Collect risk premium from option selling
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
        # Calculate current VRP
        current_vrp = self.calculate_vrp(implied_vol, realized_vol)

        # Update history
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
                vrp_zscore = (current_vrp - vrp_mean) / vrp_std

        # Trading logic
        if not self.in_vrp_trade:
            # Entry conditions
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

        # Calculate position details based on method
        position_details = self._calculate_vrp_position(signal, position_size, implied_vol, realized_vol)

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

    def _calculate_vrp_position(self, signal: int, position_size: float, implied_vol: float, realized_vol: float) -> Dict:
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
