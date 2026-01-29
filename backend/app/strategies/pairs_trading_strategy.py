from typing import Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from backend.app.strategies import BaseStrategy


class PairsTradingStrategy(BaseStrategy):
    def __init__(
        self,
        asset_1: str,
        asset_2: str,
        lookback: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_loss_z: float = 3.0,
        min_coint_pvalue: float = 0.05,
        hedge_ratio_lookback: int = 20,
        max_position: float = 1.0,
    ):
        params = {
            "asset_1": asset_1,
            "asset_2": asset_2,
            "lookback": lookback,
            "entry_z": entry_z,
            "exit_z": exit_z,
            "stop_loss_z": stop_loss_z,
            "min_coint_pvalue": min_coint_pvalue,
            "hedge_ratio_lookback": hedge_ratio_lookback,
            "max_position": max_position,
        }
        super().__init__("Enhanced Pairs Trading", params)

        # State tracking
        self.current_position = 0
        self.entry_z = 0
        self.hedge_ratio = None

    def _is_cointegrated(self, series_1: pd.Series, series_2: pd.Series) -> bool:
        """Test for cointegration"""
        try:
            _, pvalue, _ = coint(series_1, series_2)
            return pvalue < self.params["min_coint_pvalue"]
        except Exception:
            return False

    def _calculate_rolling_hedge_ratio(self, prices_1: pd.Series, prices_2: pd.Series) -> pd.Series:
        """Calculate time-varying hedge ratio"""
        log1, log2 = np.log(prices_1), np.log(prices_2)
        betas = []

        for i in range(len(log1) - self.params["hedge_ratio_lookback"]):
            x = log2.iloc[i : i + self.params["hedge_ratio_lookback"]].values
            y = log1.iloc[i : i + self.params["hedge_ratio_lookback"]].values
            beta = np.polyfit(x, y, 1)[0]
            betas.append(beta)

        # Pad and return
        betas = [np.nan] * self.params["hedge_ratio_lookback"] + betas
        return pd.Series(betas, index=prices_1.index)

    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """
        Returns dict with signal and metadata
        """
        asset_1 = self.params["asset_1"]
        asset_2 = self.params["asset_2"]
        lookback = self.params["lookback"]

        if len(data) < max(lookback, 252):  # Need data for cointegration test
            return {"signal": 0, "position_size": 0, "metadata": {}}

        prices_1 = data[asset_1]
        prices_2 = data[asset_2]

        # 1. Check cointegration (quarterly)
        if len(data) % 63 == 0:  # Roughly quarterly
            if not self._is_cointegrated(prices_1[-252:], prices_2[-252:]):
                return {
                    "signal": 0,
                    "position_size": 0,
                    "metadata": {"cointegrated": False},
                }

        # 2. Calculate rolling hedge ratio
        hedge_ratios = self._calculate_rolling_hedge_ratio(prices_1, prices_2)
        current_beta = hedge_ratios.iloc[-1]

        if pd.isna(current_beta):
            return {"signal": 0, "position_size": 0, "metadata": {}}

        # 3. Calculate spread and z-score
        spread = np.log(prices_1) - current_beta * np.log(prices_2)
        zscore = (spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()
        current_z = zscore.iloc[-1]

        # 4. Position sizing logic
        position_size = 0

        # Entry logic
        if self.current_position == 0:
            if current_z > self.params["entry_z"]:
                position_size = -self._calculate_position_size(current_z)  # Short spread
                self.entry_z = current_z
            elif current_z < -self.params["entry_z"]:
                position_size = self._calculate_position_size(-current_z)  # Long spread
                self.entry_z = current_z

        # Exit logic
        elif self.current_position != 0:
            # Stop loss
            if abs(current_z) > self.params["stop_loss_z"]:
                position_size = 0

            # Profit target (mean reversion)
            elif abs(current_z) < self.params["exit_z"]:
                position_size = 0

            # Hedge ratio change too large
            elif abs(hedge_ratios.iloc[-1] / hedge_ratios.iloc[-2] - 1) > 0.1:
                position_size = 0

        # Update current position
        self.current_position = position_size

        return {
            "signal": np.sign(position_size),
            "position_size": abs(position_size),
            "metadata": {
                "z_score": float(current_z),
                "hedge_ratio": float(current_beta),
                "spread": float(spread.iloc[-1]),
                "cointegrated": True,
            },
        }

    def _calculate_position_size(self, z_score: float) -> float:
        """Calculate position size based on z-score extremity"""
        excess_z = z_score - self.params["entry_z"]
        max_excess = self.params["stop_loss_z"] - self.params["entry_z"]

        if excess_z <= 0:
            return 0

        # Linear scaling from entry to stop loss
        size = (excess_z / max_excess) * self.params["max_position"]
        return min(size, self.params["max_position"])
