import logging

from core.trade.sizing.position_sizer import PositionSizer

logger = logging.getLogger(__name__)


class VolatilityBasedSizer(PositionSizer):
    """
    Volatility-based position sizing

    Inverse relationship: Higher volatility = Smaller position
    Goal: Equalize risk across all positions
    """

    def __init__(self, base_percent: float = 20.0, base_volatility: float = 0.02, max_percent: float = 50.0):
        """
        Args:
            base_percent: Position size at base volatility
            base_volatility: Reference volatility level (e.g., 2% = 0.02)
            max_percent: Maximum position size
        """
        self.base_percent = base_percent
        self.base_volatility = base_volatility
        self.max_percent = max_percent

    def calculate_size(self, equity: float, price: float, **kwargs) -> int:
        """
        Args:
            volatility: Current volatility of asset (required in kwargs)
        """
        volatility = kwargs.get("volatility")

        if volatility is None or volatility <= 0:
            logger.warning("Volatility not provided, using base percent")
            position_percent = self.base_percent
        else:
            # Inverse relationship
            position_percent = (self.base_volatility / volatility) * self.base_percent

            # Cap at maximum
            position_percent = min(position_percent, self.max_percent)

        position_value = equity * (position_percent / 100.0)
        return int(position_value / price)
