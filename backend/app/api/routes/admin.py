"""
Admin dashboard API — guarded by require_superuser.

Provides user management, system stats, and usage activity endpoints.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ...api.deps import get_db, require_superuser
from ...models import BacktestRun
from ...models.marketplace import MarketplaceStrategy
from ...models.usage import UsageTracking
from ...models.user import User
from ...schemas.admin import (
    AdminStats,
    AdminUserDetail,
    AdminUserListItem,
    StatusUpdateRequest,
    SubmissionListItem,
    SubmissionRejectRequest,
    TierUpdateRequest,
    UsageLogItem,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"], dependencies=[Depends(require_superuser)])


# ── User listing ────────────────────────────────────────────────────
@router.get("/users", response_model=list[AdminUserListItem])
async def list_users(
    search: Optional[str] = Query(None, description="Search by username or email"),
    tier: Optional[str] = Query(None, description="Filter by tier"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_superuser),
):
    """List all users with optional search/filter."""

    # Sub-query for backtest count per user
    bt_count_sq = select(BacktestRun.user_id, func.count(BacktestRun.id).label("bt_count")).group_by(BacktestRun.user_id).subquery()

    query = select(User, func.coalesce(bt_count_sq.c.bt_count, 0).label("backtest_count")).outerjoin(bt_count_sq, User.id == bt_count_sq.c.user_id)

    if search:
        pattern = f"%{search}%"
        query = query.where((User.username.ilike(pattern)) | (User.email.ilike(pattern)))
    if tier:
        query = query.where(User.tier == tier.upper())
    if is_active is not None:
        query = query.where(User.is_active == is_active)

    query = query.order_by(User.created_at.desc()).offset(offset).limit(limit)
    result = await db.execute(query)
    rows = result.all()

    return [
        AdminUserListItem(
            id=user.id,
            username=user.username,
            email=user.email,
            tier=user.tier,
            is_active=user.is_active,
            is_superuser=user.is_superuser,
            created_at=user.created_at.isoformat() if user.created_at else None,
            last_login=user.last_login.isoformat() if user.last_login else None,
            backtest_count=bt_count,
        )
        for user, bt_count in rows
    ]


# ── Single user detail ─────────────────────────────────────────────
@router.get("/users/{user_id}", response_model=AdminUserDetail)
async def get_user_detail(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_superuser),
):
    """Get detailed view of a single user + recent usage logs."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Backtest count
    bt_result = await db.execute(select(func.count(BacktestRun.id)).where(BacktestRun.user_id == user_id))
    bt_count = bt_result.scalar() or 0

    # Recent usage logs (last 50)
    usage_result = await db.execute(select(UsageTracking).where(UsageTracking.user_id == user_id).order_by(UsageTracking.timestamp.desc()).limit(50))
    logs = usage_result.scalars().all()

    return AdminUserDetail(
        id=user.id,
        username=user.username,
        email=user.email,
        tier=user.tier,
        is_active=user.is_active,
        is_superuser=user.is_superuser,
        created_at=user.created_at.isoformat() if user.created_at else None,
        last_login=user.last_login.isoformat() if user.last_login else None,
        backtest_count=bt_count,
        country=user.country,
        investor_type=user.investor_type,
        risk_profile=user.risk_profile,
        usage_logs=[
            UsageLogItem(
                id=log.id,
                action=log.action,
                timestamp=log.timestamp.isoformat() if log.timestamp else None,
                metadata=log.metadata_json,
            )
            for log in logs
        ],
    )


# ── Tier management ────────────────────────────────────────────────
@router.put("/users/{user_id}/tier")
async def update_user_tier(
    user_id: int,
    body: TierUpdateRequest,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_superuser),
):
    """Change a user's subscription tier."""
    valid_tiers = {"FREE", "BASIC", "PRO", "ENTERPRISE"}
    if body.tier.upper() not in valid_tiers:
        raise HTTPException(status_code=400, detail=f"Invalid tier. Must be one of: {', '.join(valid_tiers)}")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    old_tier = user.tier
    user.tier = body.tier.upper()
    await db.commit()

    logger.info(f"Admin changed user {user_id} tier: {old_tier} → {user.tier}")
    return {"message": f"Tier updated to {user.tier}", "user_id": user_id, "old_tier": old_tier, "new_tier": user.tier}


# ── Status management ──────────────────────────────────────────────
@router.put("/users/{user_id}/status")
async def update_user_status(
    user_id: int,
    body: StatusUpdateRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_superuser),
):
    """Enable or disable a user account."""
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot disable your own account")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.is_active = body.is_active
    await db.commit()

    action = "enabled" if body.is_active else "disabled"
    logger.info(f"Admin {action} user {user_id} ({user.email})")
    return {"message": f"User {action}", "user_id": user_id, "is_active": user.is_active}


# ── System-wide stats ──────────────────────────────────────────────
@router.get("/stats", response_model=AdminStats)
async def get_system_stats(
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_superuser),
):
    """Return aggregate system statistics."""
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=today_start.weekday())
    month_start = today_start.replace(day=1)

    # Total users
    total_users_r = await db.execute(select(func.count(User.id)))
    total_users = total_users_r.scalar() or 0

    # Users by tier
    tier_r = await db.execute(select(User.tier, func.count(User.id)).group_by(User.tier))
    users_by_tier = {tier: count for tier, count in tier_r.all()}

    # Active today (logged in today)
    active_today_r = await db.execute(select(func.count(User.id)).where(User.last_login >= today_start))
    active_today = active_today_r.scalar() or 0

    # Total backtests
    total_bt_r = await db.execute(select(func.count(BacktestRun.id)))
    total_backtests = total_bt_r.scalar() or 0

    # Backtests by type
    bt_type_r = await db.execute(select(BacktestRun.backtest_type, func.count(BacktestRun.id)).group_by(BacktestRun.backtest_type))
    backtests_by_type = {t: c for t, c in bt_type_r.all()}

    # Backtests today / week / month
    bt_today_r = await db.execute(select(func.count(BacktestRun.id)).where(BacktestRun.created_at >= today_start))
    bt_week_r = await db.execute(select(func.count(BacktestRun.id)).where(BacktestRun.created_at >= week_start))
    bt_month_r = await db.execute(select(func.count(BacktestRun.id)).where(BacktestRun.created_at >= month_start))

    # Active live strategies (check via usage tracking for deploy actions)
    # Use a simpler approach — count distinct usage entries with 'deploy' action
    live_r = await db.execute(select(func.count(distinct(UsageTracking.id))).where(UsageTracking.action.like("%deploy%")))
    active_live = live_r.scalar() or 0

    # Models trained (ML Studio train actions)
    models_r = await db.execute(select(func.count(UsageTracking.id)).where(UsageTracking.action == "train_model"))
    models_trained = models_r.scalar() or 0

    return AdminStats(
        total_users=total_users,
        users_by_tier=users_by_tier,
        active_today=active_today,
        total_backtests=total_backtests,
        backtests_by_type=backtests_by_type,
        backtests_today=bt_today_r.scalar() or 0,
        backtests_this_week=bt_week_r.scalar() or 0,
        backtests_this_month=bt_month_r.scalar() or 0,
        active_live_strategies=active_live,
        models_trained=models_trained,
    )


# ── Usage activity feed ─────────────────────────────────────────────
@router.get("/usage")
async def get_usage_activity(
    action: Optional[str] = Query(None, description="Filter by action type"),
    days: int = Query(7, ge=1, le=90, description="Look-back days"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_superuser),
):
    """Recent usage activity across all users."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    query = select(UsageTracking, User.username, User.email).join(User, UsageTracking.user_id == User.id).where(UsageTracking.timestamp >= cutoff)

    if action:
        query = query.where(UsageTracking.action.ilike(f"%{action}%"))

    query = query.order_by(UsageTracking.timestamp.desc()).offset(offset).limit(limit)
    result = await db.execute(query)
    rows = result.all()

    return [
        {
            "id": log.id,
            "user_id": log.user_id,
            "username": username,
            "email": email,
            "action": log.action,
            "timestamp": log.timestamp.isoformat() if log.timestamp else None,
            "metadata": log.metadata_json,
        }
        for log, username, email in rows
    ]


# ── Marketplace submission management ──────────────────────────────


@router.get("/submissions", response_model=list[SubmissionListItem])
async def list_submissions(
    status_filter: Optional[str] = Query("pending_review", alias="status", description="Filter by status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_superuser),
):
    """List marketplace strategy submissions for review."""
    query = select(MarketplaceStrategy)
    if status_filter:
        query = query.where(MarketplaceStrategy.status == status_filter)
    query = query.order_by(MarketplaceStrategy.created_at.desc()).offset(offset).limit(limit)

    result = await db.execute(query)
    strategies = result.scalars().all()

    return [
        SubmissionListItem(
            id=s.id,
            name=s.name,
            creator_name=s.creator_name,
            category=s.category,
            complexity=s.complexity,
            sharpe_ratio=s.sharpe_ratio or 0.0,
            total_return=s.total_return or 0.0,
            max_drawdown=s.max_drawdown or 0.0,
            win_rate=s.win_rate or 0.0,
            price=s.price or 0.0,
            status=s.status or "pending_review",
            submitted_at=s.created_at.isoformat() if s.created_at else None,
        )
        for s in strategies
    ]


@router.put("/submissions/{strategy_id}/approve")
async def approve_submission(
    strategy_id: int,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_superuser),
):
    """Approve a pending marketplace strategy submission."""
    result = await db.execute(select(MarketplaceStrategy).where(MarketplaceStrategy.id == strategy_id))
    strategy = result.scalar_one_or_none()
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    if strategy.status != "pending_review":
        raise HTTPException(status_code=400, detail=f"Strategy is already {strategy.status}")

    strategy.status = "approved"
    strategy.rejection_reason = None
    await db.commit()

    logger.info(f"Admin approved marketplace strategy {strategy_id} ({strategy.name})")
    return {"message": "Strategy approved and published to marketplace", "strategy_id": strategy_id}


@router.put("/submissions/{strategy_id}/reject")
async def reject_submission(
    strategy_id: int,
    body: SubmissionRejectRequest,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_superuser),
):
    """Reject a pending marketplace strategy submission."""
    result = await db.execute(select(MarketplaceStrategy).where(MarketplaceStrategy.id == strategy_id))
    strategy = result.scalar_one_or_none()
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    if strategy.status != "pending_review":
        raise HTTPException(status_code=400, detail=f"Strategy is already {strategy.status}")

    strategy.status = "rejected"
    strategy.rejection_reason = body.rejection_reason
    await db.commit()

    logger.info(f"Admin rejected marketplace strategy {strategy_id} ({strategy.name}): {body.rejection_reason}")
    return {"message": "Strategy rejected", "strategy_id": strategy_id}
