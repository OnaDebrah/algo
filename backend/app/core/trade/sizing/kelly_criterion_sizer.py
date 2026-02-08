from core.trade.sizing.position_sizer import PositionSizer


class KellyCriterionSizer(PositionSizer):
    """
    Kelly Criterion position sizing

    Formula: f* = (bp - q) / b
    Where:
    - b = odds (average_win / average_loss)
    - p = win probability
    - q = loss probability (1 - p)

    Often use fractional Kelly (e.g., 0.5 * Kelly) for safety
    """

    def __init__(self, win_rate: float, avg_win: float, avg_loss: float, fraction: float = 0.5):
        """
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade (%)
            avg_loss: Average losing trade (%)
            fraction: Fraction of Kelly to use (0.5 = half Kelly)
        """
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        self.fraction = fraction

        # Calculate Kelly percentage
        if avg_loss != 0:
            b = avg_win / abs(avg_loss)  # Odds
            p = win_rate
            q = 1 - win_rate

            kelly = (b * p - q) / b
            self.kelly_percent = max(0, kelly * fraction)  # Never go negative
        else:
            self.kelly_percent = 0.20  # Default 20%

    def calculate_size(self, equity: float, price: float, **kwargs) -> int:
        position_value = equity * self.kelly_percent
        return int(position_value / price)
