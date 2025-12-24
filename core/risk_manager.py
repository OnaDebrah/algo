"""
Risk management system
"""

import logging

from config import (
    DEFAULT_MAX_DRAWDOWN,
    DEFAULT_MAX_POSITION_SIZE,
    DEFAULT_STOP_LOSS_PCT,
)

logger = logging.getLogger(__name__)


class RiskManager:
    """Risk management system for position sizing and drawdown control"""

    def __init__(
        self,
        max_position_size: float = DEFAULT_MAX_POSITION_SIZE,
        stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
        max_drawdown: float = DEFAULT_MAX_DRAWDOWN,
    ):
        """
        Initialize risk manager

        Args:
            max_position_size: Maximum position size as fraction of portfolio (0.1 = 10%)
            stop_loss_pct: Stop loss percentage (0.05 = 5%)
            max_drawdown: Maximum drawdown allowed (0.15 = 15%)
        """
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.max_drawdown = max_drawdown
        self.peak_value = 0

    def calculate_position_size(
        self, portfolio_value: float, entry_price: float
    ) -> int:
        """
        Calculate position size based on risk parameters

        Args:
            portfolio_value: Current portfolio value
            entry_price: Entry price for the trade

        Returns:
            Number of shares to trade
        """
        max_investment = portfolio_value * self.max_position_size
        quantity = int(max_investment / entry_price)
        calculated_quantity = max(1, quantity)

        logger.debug(
            f"Position size: {calculated_quantity} shares "
            f"(${max_investment:.2f} / ${entry_price:.2f})"
        )

        return calculated_quantity

    def check_drawdown(self, current_value: float) -> bool:
        """
        Check if maximum drawdown is exceeded

        Args:
            current_value: Current portfolio value

        Returns:
            True if drawdown limit exceeded, False otherwise
        """
        if current_value > self.peak_value:
            self.peak_value = current_value

        if self.peak_value > 0:
            drawdown = (self.peak_value - current_value) / self.peak_value

            if drawdown >= self.max_drawdown:
                logger.warning(
                    f"Drawdown limit exceeded: {drawdown:.2%} "
                    f"(limit: {self.max_drawdown:.2%})"
                )
                return True

        return False

    def calculate_stop_loss(self, entry_price: float) -> float:
        """
        Calculate stop loss price

        Args:
            entry_price: Entry price for the position

        Returns:
            Stop loss price
        """
        stop_loss = entry_price * (1 - self.stop_loss_pct)
        return stop_loss

    def should_exit_position(
        self, entry_price: float, current_price: float
    ) -> tuple[bool, str]:
        """
        Check if position should be exited based on risk parameters

        Args:
            entry_price: Entry price for the position
            current_price: Current market price

        Returns:
            Tuple of (should_exit: bool, reason: str)
        """
        stop_loss = self.calculate_stop_loss(entry_price)

        if current_price <= stop_loss:
            return True, f"Stop loss triggered (${stop_loss:.2f})"

        return False, ""

    def reset_peak(self):
        """Reset peak value (useful for new trading periods)"""
        self.peak_value = 0
        logger.info("Peak value reset")

    def validate_options_position(
        self, strategy_type: str, max_loss: float, portfolio_value: float
    ) -> tuple[bool, str]:
        """Validate options position against risk limits"""

        # Check max loss per position
        max_loss_pct = abs(max_loss) / portfolio_value * 100

        if max_loss_pct > self.max_position_size:
            return False, f"Max loss ({max_loss_pct:.1f}%) exceeds limit"

        # Strategy-specific checks
        if strategy_type == "Iron Condor":
            # Limited risk defined strategies
            return True, "Approved"
        elif strategy_type == "Naked Call":
            # Unlimited risk
            return False, "Unlimited risk strategies not allowed"

        return True, "Approved"
