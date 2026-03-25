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


# ── Per-user, per-endpoint rate limiter ──────────────────────────────────────

_endpoint_limiters: Dict[str, RateLimiter] = {}


def check_endpoint_rate(user_id: int, endpoint: str, max_requests: int, window_seconds: int = 60) -> bool:
    """Check whether a request from *user_id* to *endpoint* is allowed.

    Each (user_id, endpoint) pair gets its own sliding-window limiter.
    """
    key = f"{user_id}:{endpoint}"
    if key not in _endpoint_limiters:
        _endpoint_limiters[key] = RateLimiter(max_requests=max_requests, window_seconds=window_seconds)
    limiter = _endpoint_limiters[key]
    # Update limits in case config changed (hot-reload friendly)
    limiter.max_requests = max_requests
    limiter.window_seconds = window_seconds
    return limiter.allow_request()


def check_login_rate(client_ip: str) -> bool:
    """Check whether a login attempt from *client_ip* is allowed.

    Uses a separate pool of per-IP limiters with a hard cap of
    5 requests per 60-second window — much stricter than the global
    API rate limit (60/min).
    """
    if client_ip not in _login_limiters:
        _login_limiters[client_ip] = RateLimiter(max_requests=5, window_seconds=60)
    return _login_limiters[client_ip].allow_request()
