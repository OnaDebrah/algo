import logging
from typing import List, Optional

from ..schemas.alert import AlertChannel, AlertLevel

logger = logging.getLogger(__name__)


class AlertPreferences:
    """User alert preferences with support for multiple channels"""

    def __init__(
        self,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        email_enabled: bool = True,
        sms_enabled: bool = False,
        push_enabled: bool = False,
        webhook_enabled: bool = False,
        min_level_email: AlertLevel = AlertLevel.INFO,
        min_level_sms: AlertLevel = AlertLevel.ERROR,
        min_level_push: AlertLevel = AlertLevel.WARNING,
        min_level_webhook: AlertLevel = AlertLevel.WARNING,
        webhook_url: Optional[str] = None,
    ):
        self.email = email
        self.phone = phone
        self.email_enabled = email_enabled
        self.sms_enabled = sms_enabled
        self.push_enabled = push_enabled
        self.webhook_enabled = webhook_enabled
        self.min_level_email = min_level_email
        self.min_level_sms = min_level_sms
        self.min_level_push = min_level_push
        self.min_level_webhook = min_level_webhook
        self.webhook_url = webhook_url

    def get_channels_for_level(self, level: AlertLevel) -> List[AlertChannel]:
        """Get enabled channels for an alert level"""
        channels = []

        level_priority = {AlertLevel.INFO: 0, AlertLevel.WARNING: 1, AlertLevel.ERROR: 2, AlertLevel.CRITICAL: 3}

        if self.email_enabled and self.email:
            if level_priority[level] >= level_priority[self.min_level_email]:
                channels.append(AlertChannel.EMAIL)

        if self.sms_enabled and self.phone:
            if level_priority[level] >= level_priority[self.min_level_sms]:
                channels.append(AlertChannel.SMS)

        if self.push_enabled:
            if level_priority[level] >= level_priority[self.min_level_push]:
                channels.append(AlertChannel.PUSH)

        if self.webhook_enabled and self.webhook_url:
            if level_priority[level] >= level_priority[self.min_level_webhook]:
                channels.append(AlertChannel.WEBHOOK)

        return channels

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
