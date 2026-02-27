import logging
from datetime import datetime
from typing import Any, Dict

from services.alerts.crash_alert_service import CrashAlertService

from ..alerts.alert_manager import AlertManager
from ..schemas.alert import AlertChannel, AlertLevel

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
        self.crash_alert_service = CrashAlertService(alert_manager)

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

    async def on_crash_detected(
        self,
        probability: float,
        lead_time_days: int,
        risk_factors: list,
        recommended_action: str,
        model_confidence: float = 0.5,
        intensity: str = "moderate",
    ):
        """Send alert when crash is detected by ML models"""
        await self.crash_alert_service.send_crash_alert(
            user_id=self.user_id,
            probability=probability,
            lead_time_days=lead_time_days,
            risk_factors=risk_factors,
            recommended_action=recommended_action,
            model_confidence=model_confidence,
            intensity=intensity,
            metadata={"source": "ml_ensemble"},
        )

    async def on_bubble_detected(self, symbol: str, confidence: float, crash_probability: float, critical_date: datetime, parameters: dict):
        """Send alert when bubble is detected"""
        await self.crash_alert_service.send_bubble_alert(
            user_id=self.user_id,
            symbol=symbol,
            confidence=confidence,
            crash_probability=crash_probability,
            critical_date=critical_date,
            parameters=parameters,
            action="Consider hedging or reducing exposure",
        )

    async def on_stress_increase(self, stress_index: float, change_24h: float, confidence: float, factors: list, tap_deviation: float):
        """Send alert when market stress increases"""
        await self.crash_alert_service.send_stress_alert(
            user_id=self.user_id,
            stress_index=stress_index,
            change_24h=change_24h,
            confidence=confidence,
            factors=factors,
            tap_deviation=tap_deviation,
        )

    async def on_hedge_executed(self, strategy: str, cost: float, coverage: float, expiry: datetime, positions: list, crash_probability: float):
        """Send hedge execution confirmation"""
        await self.crash_alert_service.send_hedge_execution_alert(
            user_id=self.user_id,
            strategy=strategy,
            cost=cost,
            coverage=coverage,
            expiry=expiry,
            positions=positions,
            crash_probability=crash_probability,
        )

    async def send_daily_summary(self, summary_data: dict):
        """Send daily risk summary"""
        await self.crash_alert_service.send_daily_summary(user_id=self.user_id, **summary_data)
