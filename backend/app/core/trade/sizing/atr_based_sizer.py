import logging

from core.trade.sizing.position_sizer import PositionSizer

logger = logging.getLogger(__name__)


class ATRBasedSizer(PositionSizer):
    """
    ATR (Average True Range) based position sizing

    Position size based on desired risk and ATR
    Formula: Position Size = Risk Amount / (ATR * Multiplier)
    """

    def __init__(self, risk_percent: float = 2.0, atr_multiplier: float = 2.0):
        """
        Args:
            risk_percent: Percent of equity to risk per trade
            atr_multiplier: ATR multiplier for stop distance
        """
        self.risk_percent = risk_percent / 100.0
        self.atr_multiplier = atr_multiplier

    def calculate_size(self, equity: float, price: float, **kwargs) -> int:
        """
        Args:
            atr: Average True Range (required in kwargs)
        """
        atr = kwargs.get("atr")

        if atr is None or atr <= 0:
            logger.warning("ATR not provided, using fixed percent")
            return int((equity * self.risk_percent) / price)

        # Risk amount in dollars
        risk_amount = equity * self.risk_percent

        # Stop distance (in dollars)
        stop_distance = atr * self.atr_multiplier

        # Position size
        shares = int(risk_amount / stop_distance)

        return shares
