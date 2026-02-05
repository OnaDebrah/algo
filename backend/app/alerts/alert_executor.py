import logging
from typing import Any, Dict

from schemas.alert import AlertChannel, AlertLevel

from alerts.alert_manager_ import AlertManager

logger = logging.getLogger(__name__)


class AlertingExecutor:
    """
    Strategy executor with integrated alerting

    Sends alerts for:
    - Trade executions
    - Risk limit breaches
    - Errors
    - Daily summaries
    """

    def __init__(self, alert_manager: AlertManager, user_id: int):
        self.alert_manager = alert_manager
        self.user_id = user_id

    async def on_trade_executed(self, trade: Dict[str, Any]):
        """Send alert when trade executes"""
        await self.alert_manager.send_alert(
            user_id=self.user_id,
            level=AlertLevel.INFO,
            title=f"Trade Executed: {trade['symbol']}",
            message=f"{trade['side']} {trade['quantity']} shares @ ${trade['price']:.2f}",
            strategy_id=trade.get("strategy_id"),
            metadata=trade,
        )

    async def on_risk_limit_breach(self, limit_type: str, value: float):
        """Send alert when risk limit breached"""
        await self.alert_manager.send_alert(
            user_id=self.user_id,
            level=AlertLevel.ERROR,
            title=f"Risk Limit Breach: {limit_type}",
            message=f"Strategy exceeded {limit_type}: {value:.2f}",
            channels=[AlertChannel.EMAIL, AlertChannel.SMS],  # Force both channels
        )

    async def on_error(self, error: str):
        """Send alert on error"""
        await self.alert_manager.send_alert(
            user_id=self.user_id, level=AlertLevel.CRITICAL, title="Strategy Error", message=error, channels=[AlertChannel.EMAIL, AlertChannel.SMS]
        )
