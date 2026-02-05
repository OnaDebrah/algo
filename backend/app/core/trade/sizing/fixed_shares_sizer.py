from core.trade.sizing.position_sizer import PositionSizer


class FixedSharesSizer(PositionSizer):
    """
    Fixed number of shares

    Simple but doesn't scale with account size
    """

    def __init__(self, shares: int):
        self.shares = shares

    def calculate_size(self, equity: float, price: float, **kwargs) -> int:
        return self.shares
