"""
Rate Limit Middleware (pure ASGI — avoids BaseHTTPMiddleware body-reading deadlock)
"""

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from ...config import settings
from ...security.rate_limiter import RateLimiter

_SKIP_PREFIXES = ("/docs", "/redoc", "/health", "/metrics")


class RateLimitMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app
        self.limiters: dict = {}
        self.max_requests = settings.RATE_LIMIT_PER_MINUTE
        self.window_seconds = 60

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "")
        if path.startswith(_SKIP_PREFIXES) or path == "/openapi.json":
            await self.app(scope, receive, send)
            return

        client = scope.get("client")
        client_ip = client[0] if client else "unknown"

        if client_ip not in self.limiters:
            self.limiters[client_ip] = RateLimiter(max_requests=self.max_requests, window_seconds=self.window_seconds)

        if not self.limiters[client_ip].allow_request():
            response = JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
