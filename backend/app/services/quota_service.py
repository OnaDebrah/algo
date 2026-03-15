"""
Backtest quota enforcement per user tier.

Tier limits (monthly):
  FREE       – 20
  BASIC      – 100
  PRO        – 500
  ENTERPRISE – unlimited
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..models.usage import UsageTracking

logger = logging.getLogger(__name__)

# Tier → monthly backtest cap (None = unlimited)
TIER_LIMITS = {
    "FREE": settings.BACKTEST_LIMIT_FREE,
    "BASIC": settings.BACKTEST_LIMIT_BASIC,
    "PRO": settings.BACKTEST_LIMIT_PRO,
    "ENTERPRISE": None,
}


class QuotaStatus:
    """Plain object returned by check_quota."""

    def __init__(self, allowed: bool, used: int, limit: Optional[int], remaining: Optional[int]):
        self.allowed = allowed
        self.used = used
        self.limit = limit
        self.remaining = remaining

    def to_dict(self):
        return {
            "allowed": self.allowed,
            "used": self.used,
            "limit": self.limit,
            "remaining": self.remaining,
        }


class QuotaService:
    """Stateless helpers for backtest quota management."""

    @staticmethod
    async def get_monthly_usage(db: AsyncSession, user_id: int) -> int:
        """Count backtest actions logged this calendar month."""
        now = datetime.now(timezone.utc)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        result = await db.execute(
            select(func.count(UsageTracking.id)).where(
                UsageTracking.user_id == user_id,
                UsageTracking.action.like("run_backtest_%"),
                UsageTracking.timestamp >= month_start,
            )
        )
        return result.scalar() or 0

    @staticmethod
    async def check_quota(db: AsyncSession, user_id: int, tier: str) -> QuotaStatus:
        """Return current quota status without raising."""
        limit = TIER_LIMITS.get(tier.upper())
        used = await QuotaService.get_monthly_usage(db, user_id)

        if limit is None:
            return QuotaStatus(allowed=True, used=used, limit=None, remaining=None)

        remaining = max(limit - used, 0)
        return QuotaStatus(allowed=used < limit, used=used, limit=limit, remaining=remaining)

    @staticmethod
    async def enforce_quota(db: AsyncSession, user_id: int, tier: str) -> None:
        """Raise 429 if the user has exhausted their monthly backtest quota."""
        qs = await QuotaService.check_quota(db, user_id, tier)

        if not qs.allowed:
            logger.warning(f"Quota exceeded for user {user_id} (tier={tier}, used={qs.used}/{qs.limit})")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "message": f"Monthly backtest limit reached ({qs.limit} backtests). Upgrade your plan for more.",
                    "used": qs.used,
                    "limit": qs.limit,
                    "tier": tier,
                },
            )
