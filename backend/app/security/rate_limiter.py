from datetime import datetime, timedelta, timezone
from typing import Dict, List


class RateLimiter:
    """Simple sliding-window rate limiter"""

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


# ── Login-specific rate limiter (stricter: 5 attempts / 60 seconds per IP) ──

_login_limiters: Dict[str, RateLimiter] = {}


def check_login_rate(client_ip: str) -> bool:
    """Check whether a login attempt from *client_ip* is allowed.

    Uses a separate pool of per-IP limiters with a hard cap of
    5 requests per 60-second window — much stricter than the global
    API rate limit (60/min).
    """
    if client_ip not in _login_limiters:
        _login_limiters[client_ip] = RateLimiter(max_requests=5, window_seconds=60)
    return _login_limiters[client_ip].allow_request()
