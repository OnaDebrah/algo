import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..alerts.alert_preferences import AlertPreferences
from ..alerts.email_provider import EmailProvider
from ..alerts.sms_provider import SMSProvider
from ..schemas.alert import Alert, AlertCategory, AlertChannel, AlertLevel

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Manages alerts across all channels

    Features:
    - Email notifications
    - SMS notifications
    - Rate limiting
    - Alert history
    - User preferences

    USAGE:
        email = EmailProvider(
        smtp_host="smtp.gmail.com",
        smtp_port=587,
        username="your_email@gmail.com",
        password="your_app_password",
        from_email="your_email@gmail.com"
    )

    # Configure SMS (Twilio)
    sms = SMSProvider(
        provider="twilio",
        account_sid="your_twilio_sid",
        auth_token="your_twilio_token",
        from_number="+1234567890"
    )

    # Create alert manager
    alert_mgr = AlertManager(email, sms)

    # Set user preferences
    alert_mgr.user_preferences[1] = AlertPreferences(
        email="user@example.com",
        phone="+1234567890",
        email_enabled=True,
        sms_enabled=True,
        min_level_sms=AlertLevel.ERROR
    )

    # Start manager
    asyncio.run(alert_mgr.start())

    # Send test alert
    asyncio.run(alert_mgr.send_alert(
        user_id=1,
        level=AlertLevel.WARNING,
        title="Test Alert",
        message="This is a test alert from the trading platform"
    ))
    """

    def __init__(self, email_provider: Optional[EmailProvider] = None, sms_provider: Optional[SMSProvider] = None):
        self.email_provider = email_provider
        self.sms_provider = sms_provider

        self.alert_queue: asyncio.Queue = asyncio.Queue()

        self.alert_history: List[Alert] = []

        self.rate_limits: Dict[int, Dict[AlertLevel, datetime]] = {}

        self.user_preferences: Dict[int, "AlertPreferences"] = {}

        self.worker_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start alert worker"""
        logger.info("Starting alert manager")
        self.worker_task = asyncio.create_task(self._process_alerts())

    async def stop(self):
        """Stop alert worker"""
        logger.info("Stopping alert manager")
        if self.worker_task:
            self.worker_task.cancel()

    async def send_alert(
        self,
        user_id: int,
        level: AlertLevel,
        title: str,
        message: str,
        category: Optional[AlertCategory],
        strategy_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        channels: Optional[List[AlertChannel]] = None,
        action_required: Optional[bool] = False,
        action_url: Optional[str] = None,
    ) -> bool:
        """
        Queue an alert for sending

        Args:
            user_id: User to notify
            level: Alert severity
            title: Alert title
            message: Alert message
            strategy_id: Optional strategy ID
            metadata: Optional metadata
            category: Optional alert catefory
            channels: Channels to use (default: user preferences)
            action_required: Required action,
            action_url: Action url

        Returns:
            bool: Whether alert was queued
        """
        if self._is_rate_limited(user_id, level):
            logger.warning(f"Alert rate limited for user {user_id}, level {level}")
            return False

        alert = Alert(level, title, message, strategy_id, metadata, category, action_required, action_url)

        prefs = self.user_preferences.get(user_id)
        if not prefs:
            prefs = AlertPreferences.default()

        if not channels:
            channels = prefs.get_channels_for_alert(level)

        await self.alert_queue.put((user_id, alert, channels, prefs))

        self._update_rate_limit(user_id, level)

        return True

    async def _process_alerts(self):
        """Background worker to process alert queue"""
        while True:
            try:
                # Get next alert
                user_id, alert, channels, prefs = await self.alert_queue.get()

                for channel in channels:
                    await self._send_via_channel(user_id, alert, channel, prefs)

                alert.sent_at = datetime.now(timezone.utc)
                alert.channels_sent = channels

                self.alert_history.append(alert)

                if len(self.alert_history) > 1000:
                    self.alert_history = self.alert_history[-1000:]

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing alert: {e}")

    async def _send_via_channel(self, user_id: int, alert: Alert, channel: AlertChannel, prefs: "AlertPreferences"):
        """Send alert via specific channel"""
        try:
            if channel == AlertChannel.EMAIL:
                await self._send_email(user_id, alert, prefs)

            elif channel == AlertChannel.SMS:
                await self._send_sms(user_id, alert, prefs)

            elif channel == AlertChannel.PUSH:
                await self._send_push(user_id, alert, prefs)

            elif channel == AlertChannel.WEBHOOK:
                await self._send_webhook(user_id, alert, prefs)

        except Exception as e:
            logger.error(f"Failed to send alert via {channel}: {e}")

    async def _send_email(self, user_id: int, alert: Alert, prefs: "AlertPreferences"):
        """Send email alert"""
        if not self.email_provider or not prefs.email:
            return

        # Format email
        subject = f"[{alert.level.value.upper()}] {alert.title}"

        body = f"""
        <html>
        <body>
            <h2>{alert.title}</h2>
            <p><strong>Level:</strong> {alert.level.value.upper()}</p>
            <p><strong>Time:</strong> {alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            {f'<p><strong>Strategy ID:</strong> {alert.strategy_id}</p>' if alert.strategy_id else ''}
            <hr>
            <p>{alert.message}</p>

            {self._format_metadata_html(alert.metadata) if alert.metadata else ''}
        </body>
        </html>
        """

        await self.email_provider.send_email(to_email=prefs.email, subject=subject, body=body, html=True)

    async def _send_sms(self, user_id: int, alert: Alert, prefs: "AlertPreferences"):
        """Send SMS alert"""
        if not self.sms_provider or not prefs.phone:
            return

        # Format SMS (keep it short)
        message = f"[{alert.level.value.upper()}] {alert.title}: {alert.message[:100]}"

        await self.sms_provider.send_sms(to_number=prefs.phone, message=message)

    async def _send_push(self, user_id: int, alert: Alert, prefs: "AlertPreferences"):
        """Send push notification (placeholder)"""
        logger.info(f"Push notification: {alert.title}")
        # Implement with Firebase, OneSignal, etc.

    async def _send_webhook(self, user_id: int, alert: Alert, prefs: "AlertPreferences"):
        """Send webhook notification (placeholder)"""
        logger.info(f"Webhook notification: {alert.title}")
        # Implement HTTP POST to user's webhook URL

    def _format_metadata_html(self, metadata: Dict[str, Any]) -> str:
        """Format metadata as HTML table"""
        if not metadata:
            return ""

        rows = "".join([f"<tr><td><strong>{k}</strong></td><td>{v}</td></tr>" for k, v in metadata.items()])

        return f"<table border='1'>{rows}</table>"

    def _is_rate_limited(self, user_id: int, level: AlertLevel) -> bool:
        """Check if user is rate limited for this alert level"""
        if user_id not in self.rate_limits:
            return False

        if level not in self.rate_limits[user_id]:
            return False

        last_sent = self.rate_limits[user_id][level]

        # Rate limits by level
        limits = {
            AlertLevel.INFO: 60,  # 1 per minute
            AlertLevel.WARNING: 300,  # 1 per 5 minutes
            AlertLevel.ERROR: 600,  # 1 per 10 minutes
            AlertLevel.CRITICAL: 0,  # No limit
        }

        seconds_since = (datetime.now(timezone.utc) - last_sent).total_seconds()
        return seconds_since < limits.get(level, 60)

    def _update_rate_limit(self, user_id: int, level: AlertLevel):
        """Update rate limit tracking"""
        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = {}

        self.rate_limits[user_id][level] = datetime.now(timezone.utc)

    def get_alert_history(
        self, user_id: Optional[int] = None, strategy_id: Optional[int] = None, level: Optional[AlertLevel] = None, limit: int = 100
    ) -> List[Alert]:
        """Get alert history with filters"""
        filtered = self.alert_history

        if user_id:
            # Would filter by user_id if we stored it
            pass

        if strategy_id:
            filtered = [a for a in filtered if a.strategy_id == strategy_id]

        if level:
            filtered = [a for a in filtered if a.level == level]

        return filtered[-limit:]
