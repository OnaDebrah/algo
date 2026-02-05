from datetime import datetime, timedelta, timezone
from typing import List


class RateLimiter:
    """Simple rate limiter"""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: List[datetime] = []

    def allow_request(self) -> bool:
        """Check if request is allowed"""
        now = datetime.now(timezone.utc)

        # Remove old requests outside window
        cutoff = now - timedelta(seconds=self.window_seconds)
        self.requests = [req for req in self.requests if req > cutoff]

        # Check limit
        if len(self.requests) >= self.max_requests:
            return False

        # Allow and record
        self.requests.append(now)
        return True
