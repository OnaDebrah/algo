from typing import List

import numpy as np

from core.trade.sizing.position_sizer import PositionSizer


class AdaptivePositionSizer(PositionSizer):
    """
    Adaptive position sizing

    Combines multiple methods and adjusts based on:
    - Recent performance
    - Market conditions
    - Win rate
    """

    def __init__(self, base_percent: float = 20.0, max_percent: float = 40.0, min_percent: float = 5.0):
        self.base_percent = base_percent
        self.max_percent = max_percent
        self.min_percent = min_percent

        # Track recent performance
        self.recent_trades: List[float] = []
        self.max_recent_trades = 20

    def add_trade_result(self, pnl_percent: float):
        """Add recent trade result for adaptation"""
        self.recent_trades.append(pnl_percent)
        if len(self.recent_trades) > self.max_recent_trades:
            self.recent_trades.pop(0)

    def calculate_size(self, equity: float, price: float, **kwargs) -> int:
        """
        Adaptive sizing based on recent performance
        """
        position_percent = self.base_percent

        if len(self.recent_trades) >= 5:
            wins = sum(1 for pnl in self.recent_trades if pnl > 0)
            win_rate = wins / len(self.recent_trades)

            avg_win = np.mean([pnl for pnl in self.recent_trades if pnl > 0]) if wins > 0 else 0
            avg_loss = np.mean([pnl for pnl in self.recent_trades if pnl < 0]) if len(self.recent_trades) > wins else 0

            # Increase size if performing well
            if win_rate > 0.6 and avg_win > abs(avg_loss):
                position_percent = min(self.base_percent * 1.5, self.max_percent)

            # Decrease size if performing poorly
            elif win_rate < 0.4:
                position_percent = max(self.base_percent * 0.5, self.min_percent)

            # Recent drawdown adjustment
            recent_pnl = sum(self.recent_trades[-5:])
            if recent_pnl < -10:  # -10% drawdown in recent trades
                position_percent = max(position_percent * 0.7, self.min_percent)

        position_value = equity * (position_percent / 100.0)
        return int(position_value / price)
