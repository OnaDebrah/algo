from core.trade.sizing.position_sizer import PositionSizer


class FixedDollarSizer(PositionSizer):
    """
    Fixed dollar amount per position

    Example: Always invest $10,000 per position
    """

    def __init__(self, dollars: float):
        self.dollars = dollars

    def calculate_size(self, equity: float, price: float, **kwargs) -> int:
        return int(self.dollars / price)
