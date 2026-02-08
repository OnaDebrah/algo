import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Circuit breaker pattern for error handling

    States:
    - CLOSED: Normal operation
    - OPEN: Too many errors, stop operations
    - HALF_OPEN: Testing if errors resolved
    """

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.state = "CLOSED"
        self.failures = 0
        self.last_failure_time: Optional[datetime] = None

    def should_trip(self, recent_errors: List[Dict]) -> bool:
        """Check if circuit should trip"""
        # Count recent high/critical errors (last 5 minutes)
        now = datetime.now(timezone.utc)
        recent = [e for e in recent_errors if (now - e["timestamp"]).total_seconds() < 300 and e["severity"] in ["high", "critical"]]

        if len(recent) >= self.failure_threshold:
            self.state = "OPEN"
            return True

        return False
