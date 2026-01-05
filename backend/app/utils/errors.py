"""
Error response utilities — prevent leaking internal details to API clients.
"""

import logging

from ..config import settings

logger = logging.getLogger(__name__)


def safe_detail(message: str, error: Exception | None = None) -> str:
    """Return an error detail string safe for API responses.

    In development/test: includes the exception string for easier debugging.
    In production: returns only the human-readable message.
    The full exception is always logged server-side regardless of environment.
    """
    if error is not None:
        logger.error(f"{message}: {error}", exc_info=True)

    if settings.ENVIRONMENT in ("development", "test") and error is not None:
        return f"{message}: {error}"
    return message
