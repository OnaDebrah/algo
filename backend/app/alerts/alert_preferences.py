import logging
from typing import List, Optional

from schemas.alert import AlertChannel, AlertLevel

logger = logging.getLogger(__name__)


class AlertPreferences:
    """User alert preferences"""

    def __init__(
        self,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        email_enabled: bool = True,
        sms_enabled: bool = False,
        min_level_email: AlertLevel = AlertLevel.INFO,
        min_level_sms: AlertLevel = AlertLevel.ERROR,
    ):
        self.email = email
        self.phone = phone
        self.email_enabled = email_enabled
        self.sms_enabled = sms_enabled
        self.min_level_email = min_level_email
        self.min_level_sms = min_level_sms

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

        return channels

    @staticmethod
    def default() -> "AlertPreferences":
        """Default preferences"""
        return AlertPreferences(email_enabled=True, sms_enabled=False, min_level_email=AlertLevel.WARNING, min_level_sms=AlertLevel.CRITICAL)
