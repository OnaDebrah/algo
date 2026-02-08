from core.trade.sizing.position_sizer import PositionSizer


class FixedPercentSizer(PositionSizer):
    """
    Fixed percentage of equity

    Example: Always invest 20% of account
    Scales with account size
    """

    def __init__(self, percent: float):
        """
        Args:
            percent: Percentage of equity (0-100)
        """
        self.percent = percent / 100.0

    def calculate_size(self, equity: float, price: float, **kwargs) -> int:
        position_value = equity * self.percent
        return int(position_value / price)
