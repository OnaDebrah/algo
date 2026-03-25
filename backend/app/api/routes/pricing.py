"""
Pricing tiers & quota status API.

GET /pricing/tiers   – public, returns tier definitions + promo status
GET /pricing/quota   – authenticated, returns current user's quota
"""

import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ...api.deps import get_current_active_user, get_db
from ...config import BASIC, ENTERPRISE, FREE, PRO, settings
from ...core.permissions import is_promo_active
from ...models.user import User
from ...services.quota_service import QuotaService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pricing", tags=["Pricing"])

# ── Tier definitions (static) ─────────────────────────────────────
TIER_DEFINITIONS = [
    {
        "tier": "FREE",
        "price": FREE,
        "label": "Free",
        "backtest_limit": settings.BACKTEST_LIMIT_FREE,
        "features": [
            "Performance Dashboard",
            "Single-asset backtesting",
            f"{settings.BACKTEST_LIMIT_FREE} backtests / month",
            "Market data access",
        ],
    },
    {
        "tier": "BASIC",
        "price": BASIC,
        "label": "Basic",
        "backtest_limit": settings.BACKTEST_LIMIT_BASIC,
        "features": [
            "Everything in Free",
            "Multi-asset backtesting",
            "Advanced analytics",
            f"{settings.BACKTEST_LIMIT_BASIC} backtests / month",
            "Sector Scanner",
        ],
    },
    {
        "tier": "PRO",
        "price": PRO,
        "label": "Pro",
        "backtest_limit": settings.BACKTEST_LIMIT_PRO,
        "features": [
            "Everything in Basic",
            "ML strategies & ML Studio",
            "Custom strategy builder",
            f"{settings.BACKTEST_LIMIT_PRO} backtests / month",
            "Walk-Forward Analysis",
            "Options desk",
        ],
    },
    {
        "tier": "ENTERPRISE",
        "price": ENTERPRISE,
        "label": "Enterprise",
        "backtest_limit": None,
        "features": [
            "Everything in Pro",
            "Live trading (Alpaca & IB)",
            "API access",
            "Unlimited backtests",
            "Priority support",
            "Portfolio optimization",
        ],
    },
]


@router.get("/tiers")
async def get_tiers():
    """Public endpoint — returns pricing tiers and promo status."""
    promo_active = is_promo_active()
    promo_end = None
    if promo_active:
        try:
            start = datetime.fromisoformat(settings.LAUNCH_PROMO_START).replace(tzinfo=timezone.utc)
            promo_end = (start + timedelta(days=settings.LAUNCH_PROMO_MONTHS * 30)).isoformat()
        except (ValueError, TypeError):
            pass

    return {
        "tiers": TIER_DEFINITIONS,
        "promo": {
            "active": promo_active,
            "message": "Launch Special: All features unlocked free for 6 months!" if promo_active else None,
            "ends_at": promo_end,
        },
    }


@router.get("/quota")
async def get_quota(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the authenticated user's current backtest quota status."""
    qs = await QuotaService.check_quota(db, current_user.id, current_user.tier)
    return {
        "tier": current_user.tier,
        **qs.to_dict(),
    }
