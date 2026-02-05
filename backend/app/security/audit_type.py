"""
Audit Logging & Security Hardening
Complete audit trail, security measures, compliance
"""

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Audit event types"""

    # Authentication
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"

    # Strategy Management
    STRATEGY_DEPLOYED = "strategy_deployed"
    STRATEGY_STARTED = "strategy_started"
    STRATEGY_PAUSED = "strategy_paused"
    STRATEGY_STOPPED = "strategy_stopped"
    STRATEGY_DELETED = "strategy_deleted"
    STRATEGY_MODIFIED = "strategy_modified"

    # Trading
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"

    # Risk Events
    RISK_LIMIT_BREACH = "risk_limit_breach"
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker_triggered"
    EMERGENCY_STOP = "emergency_stop"

    # Configuration
    CONFIG_CHANGED = "config_changed"
    CREDENTIALS_UPDATED = "credentials_updated"

    # Security
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
