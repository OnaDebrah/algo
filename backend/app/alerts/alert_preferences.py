from datetime import datetime
from typing import Dict, List, Optional

from ..schemas.alert import AlertCategory, AlertChannel, AlertLevel


class AlertPreferences:
    def __init__(
        self,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        email_enabled: bool = True,
        sms_enabled: bool = False,
        push_enabled: bool = False,
        webhook_enabled: bool = False,
        slack_enabled: bool = False,
        discord_enabled: bool = False,
        slack_webhook: Optional[str] = None,
        discord_webhook: Optional[str] = None,
        # Min levels
        min_level_email: AlertLevel = AlertLevel.WARNING,
        min_level_sms: AlertLevel = AlertLevel.ERROR,
        min_level_push: AlertLevel = AlertLevel.WARNING,
        min_level_webhook: AlertLevel = AlertLevel.WARNING,
        crash_notifications: bool = True,
        bubble_notifications: bool = True,
        stress_notifications: bool = True,
        hedge_notifications: bool = True,
        category_channels: Dict[AlertCategory, List[AlertChannel]] = None,
        # Quiet hours
        quiet_hours_start: Optional[int] = None,  # Hour (0-23)
        quiet_hours_end: Optional[int] = None,
        quiet_hours_priority: AlertLevel = AlertLevel.CRITICAL,  # Only these levels during quiet hours
        webhook_url: Optional[str] = None,
    ):
        self.email = email
        self.phone = phone
        self.email_enabled = email_enabled
        self.sms_enabled = sms_enabled
        self.push_enabled = push_enabled
        self.webhook_enabled = webhook_enabled
        self.slack_enabled = slack_enabled
        self.discord_enabled = discord_enabled
        self.slack_webhook = slack_webhook
        self.discord_webhook = discord_webhook

        self.min_level_email = min_level_email
        self.min_level_sms = min_level_sms
        self.min_level_push = min_level_push
        self.min_level_webhook = min_level_webhook

        self.crash_notifications = crash_notifications
        self.bubble_notifications = bubble_notifications
        self.stress_notifications = stress_notifications
        self.hedge_notifications = hedge_notifications

        self.category_channels = category_channels or {}
        self.quiet_hours_start = quiet_hours_start
        self.quiet_hours_end = quiet_hours_end
        self.quiet_hours_priority = quiet_hours_priority

        self.webhook_url = webhook_url

    def get_channels_for_alert(self, level: AlertLevel, category: Optional[AlertCategory] = None) -> List[AlertChannel]:
        """Get enabled channels with category-specific overrides"""

        if self._in_quiet_hours() and level.value < self.quiet_hours_priority.value:
            return []

        if category == AlertCategory.CRASH_PREDICTION and not self.crash_notifications:
            return []
        if category == AlertCategory.BUBBLE_DETECTION and not self.bubble_notifications:
            return []
        if category == AlertCategory.MARKET_STRESS and not self.stress_notifications:
            return []
        if category == AlertCategory.HEDGE_EXECUTION and not self.hedge_notifications:
            return []

        if category and category in self.category_channels:
            return self._filter_channels_by_level(self.category_channels[category], level)

        return self._get_default_channels(level)

    def _in_quiet_hours(self) -> bool:
        """Check if current time is within quiet hours"""
        if self.quiet_hours_start is None or self.quiet_hours_end is None:
            return False

        current_hour = datetime.now().hour

        if self.quiet_hours_start <= self.quiet_hours_end:
            return self.quiet_hours_start <= current_hour < self.quiet_hours_end
        else:  # Overnight (e.g., 22:00 to 06:00)
            return current_hour >= self.quiet_hours_start or current_hour < self.quiet_hours_end

    def _filter_channels_by_level(self, channels: List[AlertChannel], level: AlertLevel) -> List[AlertChannel]:
        """Filter channels based on level thresholds"""
        level_priority = {
            AlertLevel.INFO: 0,
            AlertLevel.WARNING: 1,
            AlertLevel.ERROR: 2,
            AlertLevel.CRITICAL: 3,
            AlertLevel.CRASH_HIGH: 4,
            AlertLevel.CRASH_MODERATE: 3,
            AlertLevel.BUBBLE_DETECTED: 2,
            AlertLevel.STRESS_HIGH: 3,
        }

        result = []
        level_val = level_priority.get(level, 0)

        for channel in channels:
            min_level = self._get_min_level_for_channel(channel)
            if level_val >= level_priority.get(min_level, 0):
                result.append(channel)

        return result

    def _get_min_level_for_channel(self, channel: AlertChannel) -> AlertLevel:
        """Get minimum level for a channel"""
        channel_map = {
            AlertChannel.EMAIL: self.min_level_email,
            AlertChannel.SMS: self.min_level_sms,
            AlertChannel.PUSH: self.min_level_push,
            AlertChannel.WEBHOOK: self.min_level_webhook,
            AlertChannel.SLACK: AlertLevel.WARNING,  # Default
            AlertChannel.DISCORD: AlertLevel.WARNING,  # Default
        }
        return channel_map.get(channel, AlertLevel.WARNING)

    def _get_default_channels(self, level: AlertLevel) -> List[AlertChannel]:
        """Get default channels based on level"""
        channels = []

        if self.email_enabled and self.email:
            if self._meets_threshold(level, self.min_level_email):
                channels.append(AlertChannel.EMAIL)

        if self.sms_enabled and self.phone:
            if self._meets_threshold(level, self.min_level_sms):
                channels.append(AlertChannel.SMS)

        if self.push_enabled:
            if self._meets_threshold(level, self.min_level_push):
                channels.append(AlertChannel.PUSH)

        if self.webhook_enabled and self.webhook_url:
            if self._meets_threshold(level, self.min_level_webhook):
                channels.append(AlertChannel.WEBHOOK)

        if self.slack_enabled and self.slack_webhook:
            channels.append(AlertChannel.SLACK)

        if self.discord_enabled and self.discord_webhook:
            channels.append(AlertChannel.DISCORD)

        return channels

    def _meets_threshold(self, level: AlertLevel, min_level: AlertLevel) -> bool:
        """Check if level meets minimum threshold"""
        level_priority = {
            AlertLevel.INFO: 0,
            AlertLevel.WARNING: 1,
            AlertLevel.ERROR: 2,
            AlertLevel.CRITICAL: 3,
            AlertLevel.CRASH_HIGH: 4,
            AlertLevel.CRASH_MODERATE: 3,
            AlertLevel.BUBBLE_DETECTED: 2,
            AlertLevel.STRESS_HIGH: 3,
        }
        return level_priority.get(level, 0) >= level_priority.get(min_level, 0)

    @staticmethod
    def default() -> "AlertPreferences":
        """Default preferences"""
        return AlertPreferences(
            email_enabled=True,
            sms_enabled=False,
            push_enabled=False,
            webhook_enabled=False,
            min_level_email=AlertLevel.WARNING,
            min_level_sms=AlertLevel.CRITICAL,
            min_level_push=AlertLevel.WARNING,
            min_level_webhook=AlertLevel.WARNING,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "email": self.email,
            "phone": self.phone,
            "email_enabled": self.email_enabled,
            "sms_enabled": self.sms_enabled,
            "push_enabled": self.push_enabled,
            "webhook_enabled": self.webhook_enabled,
            "min_level_email": self.min_level_email,
            "min_level_sms": self.min_level_sms,
            "min_level_push": self.min_level_push,
            "min_level_webhook": self.min_level_webhook,
            "webhook_url": self.webhook_url,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AlertPreferences":
        """Create from dictionary"""
        return cls(**data)
