from core.trade.sizing.atr_based_sizer import ATRBasedSizer
from core.trade.sizing.fixed_dollar_sizer import FixedDollarSizer
from core.trade.sizing.fixed_percent_sizer import FixedPercentSizer
from core.trade.sizing.fixed_shares_sizer import FixedSharesSizer
from core.trade.sizing.kelly_criterion_sizer import KellyCriterionSizer
from core.trade.sizing.position_sizer import PositionSizer
from core.trade.sizing.position_sizing_method import PositionSizingMethod
from core.trade.sizing.risk_parity_sizer import RiskParitySizer
from core.trade.sizing.volatility_based_sizer import VolatilityBasedSizer


class PositionSizeManager:
    """
    Manages position sizing for strategies

    Provides interface to different sizing methods
    """

    def __init__(self, method: PositionSizingMethod, **params):
        """
        Args:
            method: Sizing method to use
            **params: Parameters for the method
        """
        self.method = method
        self.params = params

        # Create sizer
        self.sizer = self._create_sizer()

    def _create_sizer(self) -> PositionSizer:
        """Create position sizer based on method"""

        if self.method == PositionSizingMethod.FIXED_SHARES:
            return FixedSharesSizer(self.params.get("shares", 100))

        elif self.method == PositionSizingMethod.FIXED_DOLLAR:
            return FixedDollarSizer(self.params.get("dollars", 10000))

        elif self.method == PositionSizingMethod.FIXED_PERCENT:
            return FixedPercentSizer(self.params.get("percent", 20))

        elif self.method == PositionSizingMethod.KELLY_CRITERION:
            return KellyCriterionSizer(
                win_rate=self.params.get("win_rate", 0.55),
                avg_win=self.params.get("avg_win", 0.02),
                avg_loss=self.params.get("avg_loss", 0.01),
                fraction=self.params.get("fraction", 0.5),
            )

        elif self.method == PositionSizingMethod.VOLATILITY_BASED:
            return VolatilityBasedSizer(
                base_percent=self.params.get("base_percent", 20),
                base_volatility=self.params.get("base_volatility", 0.02),
                max_percent=self.params.get("max_percent", 50),
            )

        elif self.method == PositionSizingMethod.ATR_BASED:
            return ATRBasedSizer(risk_percent=self.params.get("risk_percent", 2), atr_multiplier=self.params.get("atr_multiplier", 2))

        elif self.method == PositionSizingMethod.RISK_PARITY:
            return RiskParitySizer(target_volatility=self.params.get("target_volatility", 0.15))

        else:
            # Default to fixed percent
            return FixedPercentSizer(20)

    def calculate(self, equity: float, price: float, **market_data) -> int:
        """
        Calculate position size

        Args:
            equity: Current account equity
            price: Current asset price
            **market_data: Additional market data (volatility, atr, etc.)

        Returns:
            Number of shares
        """
        shares = self.sizer.calculate_size(equity, price, **market_data)

        # Sanity checks
        shares = max(0, shares)  # Can't be negative

        # Don't exceed equity
        max_shares = int(equity / price)
        shares = min(shares, max_shares)

        return shares
