"""
Crash-specific alert service that integrates with your existing AlertManager
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from alerts import AlertManager
from core.data.providers.providers import ProviderFactory

from ...schemas.alert import AlertCategory, AlertLevel

logger = logging.getLogger(__name__)


class CrashAlertService:
    """
    Specialized service for crash-related alerts
    Integrates with existing AlertManager
    """

    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.provider_factory = ProviderFactory()

    async def send_crash_alert(
        self,
        user_id: int,
        probability: float,
        lead_time_days: int,
        risk_factors: List[str],
        recommended_action: str,
        model_confidence: float,
        intensity: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Send crash prediction alert with rich formatting
        """
        # Determine alert level based on probability
        if probability >= 0.7:
            level = AlertLevel.CRASH_HIGH
            emoji = "🚨"
            urgency = "IMMEDIATE ACTION RECOMMENDED"
        elif probability >= 0.4:
            level = AlertLevel.CRASH_MODERATE
            emoji = "⚠️"
            urgency = "Prepare hedging strategy"
        else:
            level = AlertLevel.WARNING
            emoji = "📊"
            urgency = "Monitor closely"

        # Create title with emoji
        title = f"{emoji} Market Crash Alert: {probability:.1%} Probability"

        # Create rich message with details
        message = self._format_crash_message(
            probability=probability,
            lead_time_days=lead_time_days,
            risk_factors=risk_factors,
            recommended_action=recommended_action,
            model_confidence=model_confidence,
            intensity=intensity,
            urgency=urgency,
        )

        alert_metadata = {
            "probability": probability,
            "lead_time_days": lead_time_days,
            "model_confidence": model_confidence,
            "intensity": intensity,
            "risk_factors": risk_factors,
            "recommended_action": recommended_action,
            "timestamp": datetime.now().isoformat(),
            "type": "crash_prediction",
            **(metadata or {}),
        }

        # Send through alert manager
        await self.alert_manager.send_alert(
            user_id=user_id,
            level=level,
            title=title,
            message=message,
            category=AlertCategory.CRASH_PREDICTION,
            metadata=alert_metadata,
            action_required=True,
            action_url=f"/dashboard/crash-analysis?probability={probability:.0%}",
        )

        logger.info(f"Sent crash alert to user {user_id} with probability {probability:.1%}")

    async def send_bubble_alert(
        self,
        user_id: int,
        symbol: str,
        confidence: float,
        crash_probability: float,
        critical_date: datetime,
        parameters: Dict[str, float],
        action: str,
    ):
        """Send bubble detection alert"""

        level = AlertLevel.CRASH_HIGH if confidence > 0.8 else AlertLevel.CRASH_MODERATE if confidence > 0.6 else AlertLevel.WARNING

        days_to_critical = (critical_date - datetime.now()).days

        title = f"🫧 Bubble Detected in {symbol}"

        message = (
            f"**Bubble Detection Alert**\n\n"
            f"Symbol: {symbol}\n"
            f"Confidence: {confidence:.1%}\n"
            f"Crash Probability: {crash_probability:.1%}\n"
            f"Critical Date: {critical_date.strftime('%Y-%m-%d')} ({days_to_critical} days)\n\n"
            f"**LPPLS Parameters:**\n"
            f"• m (power law): {parameters.get('m', 0):.3f}\n"
            f"• ω (frequency): {parameters.get('omega', 0):.1f}\n"
            f"• Amplitude: {parameters.get('amplitude', 0):.3f}\n\n"
            f"**Recommended Action:** {action}"
        )

        metadata = {
            "symbol": symbol,
            "confidence": confidence,
            "crash_probability": crash_probability,
            "critical_date": critical_date.isoformat(),
            "parameters": parameters,
            "type": "bubble_detection",
        }

        await self.alert_manager.send_alert(
            user_id=user_id,
            level=level,
            title=title,
            message=message,
            category=AlertCategory.BUBBLE_DETECTION,
            metadata=metadata,
            action_required=True,
            action_url=f"/dashboard/bubbles/{symbol}",
        )

    async def send_stress_alert(
        self, user_id: int, stress_index: float, change_24h: float, confidence: float, factors: List[str], tap_deviation: float
    ):
        """Send market stress alert"""

        level = AlertLevel.STRESS_HIGH if stress_index > 0.8 else AlertLevel.ERROR if stress_index > 0.6 else AlertLevel.WARNING

        trend = "📈 increasing" if change_24h > 0 else "📉 decreasing"

        title = f"🌊 Market Stress Alert: {stress_index:.1%}"

        message = (
            f"**Market Stress Index Update**\n\n"
            f"Current Stress: {stress_index:.1%} ({trend} by {abs(change_24h):.1%} in 24h)\n"
            f"Model Confidence: {confidence:.1%}\n"
            f"TAP Deviation: {tap_deviation:.4f}\n\n"
            f"**Contributing Factors:**\n" + "\n".join([f"• {factor}" for factor in factors])
        )

        metadata = {
            "stress_index": stress_index,
            "change_24h": change_24h,
            "confidence": confidence,
            "tap_deviation": tap_deviation,
            "factors": factors,
            "type": "market_stress",
        }

        await self.alert_manager.send_alert(
            user_id=user_id,
            level=level,
            title=title,
            message=message,
            category=AlertCategory.MARKET_STRESS,
            metadata=metadata,
            action_required=False,
            action_url="/dashboard/stress",
        )

    async def send_hedge_execution_alert(
        self, user_id: int, strategy: str, cost: float, coverage: float, expiry: datetime, positions: List[Dict], crash_probability: float
    ):
        """Send hedge execution confirmation"""

        title = f"🛡️ Hedge Executed: {strategy.replace('_', ' ').title()}"

        # Format position details
        position_lines = []
        for pos in positions[:3]:  # Show top 3
            position_lines.append(f"  • {pos['quantity']}x {pos['symbol']} {pos['type']} @ ${pos['strike']}")
        if len(positions) > 3:
            position_lines.append(f"  • ... and {len(positions)-3} more positions")

        message = (
            f"**Hedge Position Executed**\n\n"
            f"Strategy: {strategy.replace('_', ' ').title()}\n"
            f"Cost: ${cost:,.2f}\n"
            f"Portfolio Coverage: {coverage:.1%}\n"
            f"Expiry: {expiry.strftime('%Y-%m-%d')}\n"
            f"Current Crash Risk: {crash_probability:.1%}\n\n"
            f"**Position Details:**\n" + "\n".join(position_lines)
        )

        metadata = {
            "strategy": strategy,
            "cost": cost,
            "coverage": coverage,
            "expiry": expiry.isoformat(),
            "positions": positions,
            "crash_probability": crash_probability,
            "type": "hedge_execution",
        }

        await self.alert_manager.send_alert(
            user_id=user_id,
            level=AlertLevel.INFO,
            title=title,
            message=message,
            category=AlertCategory.HEDGE_EXECUTION,
            metadata=metadata,
            action_required=False,
            action_url="/portfolio/hedges",
        )

    async def send_hedge_expiry_alert(
        self, user_id: int, hedge_id: int, strategy: str, days_remaining: int, market_conditions: str, recommendations: List[str]
    ):
        """Send hedge expiry warning"""

        level = AlertLevel.ERROR if days_remaining <= 3 else AlertLevel.WARNING

        title = f"⏰ Hedge Expiring: {days_remaining} days remaining"

        message = (
            f"**Hedge Expiry Warning**\n\n"
            f"Strategy: {strategy}\n"
            f"Days to Expiry: {days_remaining}\n"
            f"Market Conditions: {market_conditions}\n\n"
            f"**Recommendations:**\n" + "\n".join([f"• {rec}" for rec in recommendations])
        )

        metadata = {
            "hedge_id": hedge_id,
            "strategy": strategy,
            "days_remaining": days_remaining,
            "recommendations": recommendations,
            "type": "hedge_expiry",
        }

        await self.alert_manager.send_alert(
            user_id=user_id,
            level=level,
            title=title,
            message=message,
            category=AlertCategory.HEDGE_EXPIRY,
            metadata=metadata,
            action_required=True,
            action_url=f"/portfolio/hedges/{hedge_id}/roll",
        )

    async def send_daily_summary(
        self,
        user_id: int,
        crash_prob: float,
        stress_index: float,
        bubbles_detected: List[str],
        active_hedges: int,
        portfolio_value: float,
        daily_pnl: float,
        top_alerts: List[str],
    ):
        """Send daily risk summary"""

        # Determine overall risk level
        if crash_prob > 0.7 or stress_index > 0.8:
            risk_level = "🔴 HIGH"
            level = AlertLevel.WARNING
        elif crash_prob > 0.4 or stress_index > 0.6:
            risk_level = "🟡 MODERATE"
            level = AlertLevel.INFO
        else:
            risk_level = "🟢 LOW"
            level = AlertLevel.INFO

        title = f"📊 Daily Risk Summary - {datetime.now().strftime('%Y-%m-%d')}"

        message = (
            f"**Market Risk Summary**\n\n"
            f"Overall Risk Level: {risk_level}\n"
            f"Crash Probability: {crash_prob:.1%}\n"
            f"Market Stress: {stress_index:.1%}\n"
            f"Bubbles Detected: {len(bubbles_detected)}\n\n"
            f"**Portfolio Status**\n"
            f"Value: ${portfolio_value:,.2f}\n"
            f"Daily P&L: {daily_pnl:+.2%}\n"
            f"Active Hedges: {active_hedges}\n\n"
            f"**Today's Top Alerts:**\n" + "\n".join([f"• {alert}" for alert in top_alerts[:3]])
        )

        metadata = {
            "crash_probability": crash_prob,
            "stress_index": stress_index,
            "bubbles": bubbles_detected,
            "active_hedges": active_hedges,
            "portfolio_value": portfolio_value,
            "daily_pnl": daily_pnl,
            "risk_level": risk_level,
            "type": "daily_summary",
        }

        await self.alert_manager.send_alert(
            user_id=user_id, level=level, title=title, message=message, category=AlertCategory.SYSTEM, metadata=metadata, action_required=False
        )

    def _format_crash_message(
        self,
        probability: float,
        lead_time_days: int,
        risk_factors: List[str],
        recommended_action: str,
        model_confidence: float,
        intensity: str,
        urgency: str,
    ) -> str:
        """Format crash alert message with rich formatting"""

        # Create risk factors section
        risk_factors_text = "\n".join([f"• {factor}" for factor in risk_factors[:5]])
        if len(risk_factors) > 5:
            risk_factors_text += f"\n• ... and {len(risk_factors)-5} more factors"

        # Map intensity to emoji
        intensity_emoji = {"severe": "🔴", "moderate": "🟡", "mild": "🟢"}.get(intensity, "⚪")

        return (
            f"**{urgency}**\n\n"
            f"**Crash Probability:** {probability:.1%}\n"
            f"**Timeframe:** Within {lead_time_days} days\n"
            f"**Model Confidence:** {model_confidence:.1%}\n"
            f"**Expected Intensity:** {intensity_emoji} {intensity.upper()}\n\n"
            f"**Key Risk Factors:**\n{risk_factors_text}\n\n"
            f"**Recommended Action:** {recommended_action}\n\n"
            f"View detailed analysis in your dashboard."
        )
