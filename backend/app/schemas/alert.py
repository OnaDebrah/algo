import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(str, Enum):
    """Alert delivery channels"""

    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"


class Alert:
    """Represents an alert"""

    def __init__(self, level: AlertLevel, title: str, message: str, strategy_id: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None):
        self.id = None
        self.level = level
        self.title = title
        self.message = message
        self.strategy_id = strategy_id
        self.metadata = metadata or {}
        self.created_at = datetime.now(timezone.utc)
        self.sent_at: Optional[datetime] = None
        self.channels_sent: List[AlertChannel] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "strategy_id": self.strategy_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "channels_sent": [ch.value for ch in self.channels_sent],
        }
