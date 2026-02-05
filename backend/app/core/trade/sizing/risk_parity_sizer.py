import logging

from core.trade.sizing.position_sizer import PositionSizer

logger = logging.getLogger(__name__)


class RiskParitySizer(PositionSizer):
    """
    Risk Parity position sizing

    Allocates equal risk (not equal dollars) to each position
    Considers volatility and correlations
    """

    def __init__(self, target_volatility: float = 0.15):
        """
        Args:
            target_volatility: Target portfolio volatility (e.g., 15% = 0.15)
        """
        self.target_volatility = target_volatility

    def calculate_size(self, equity: float, price: float, **kwargs) -> int:
        """
        Args:
            volatility: Asset volatility (required)
            correlation: Correlation with portfolio (optional)
        """
        volatility = kwargs.get("volatility")
        correlation = kwargs.get("correlation", 1.0)

        if volatility is None or volatility <= 0:
            logger.warning("Volatility not provided")
            return 0

        # Target risk contribution
        target_risk = self.target_volatility

        # Position volatility (considering correlation)
        position_vol = volatility * correlation

        # Position weight to achieve target risk
        weight = target_risk / position_vol

        # Convert to shares
        position_value = equity * weight
        shares = int(position_value / price)

        return shares
