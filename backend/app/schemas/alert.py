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
    CRASH_HIGH = "crash_high"
    CRASH_MODERATE = "crash_moderate"
    BUBBLE_DETECTED = "bubble"
    STRESS_HIGH = "stress_high"


class AlertChannel(str, Enum):
    """Alert delivery channels"""

    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    SLACK = "slack"
    DISCORD = "discord"


class AlertCategory(str, Enum):
    """New category enum for better organization"""

    CRASH_PREDICTION = "crash_prediction"
    BUBBLE_DETECTION = "bubble_detection"
    MARKET_STRESS = "market_stress"
    HEDGE_EXECUTION = "hedge_execution"
    HEDGE_EXPIRY = "hedge_expiry"
    PORTFOLIO_RISK = "portfolio_risk"
    STOP_LOSS = "stop_loss"
    SYSTEM = "system"
    TRADE = "trade"


class Alert:
    """Represents an alert"""

    def __init__(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        strategy_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        category: Optional[AlertCategory] = None,
        action_required: bool = False,
        action_url: Optional[str] = None,
        expiry: Optional[datetime] = None,
    ):
        self.id = None
        self.level = level
        self.title = title
        self.message = message
        self.strategy_id = strategy_id
        self.metadata = metadata or {}
        self.category = category
        self.action_required = action_required
        self.action_url = action_url
        self.expiry = expiry
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
            "category": self.category,
            "action_required": self.action_required,
            "action_url": self.action_url,
            "expiry": self.expiry,
            "created_at": self.created_at.isoformat(),
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "channels_sent": [ch.value for ch in self.channels_sent],
        }
