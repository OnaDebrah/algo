"""Scheduled/recurring backtest routes."""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from ...api.deps import get_current_active_user
from ...database import get_db
from ...models import User
from ...models.scheduled_backtest import ScheduledBacktest, ScheduledBacktestRun

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scheduled-backtests", tags=["Scheduled Backtests"])


class ScheduledBacktestCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    strategy_key: str
    strategy_params: dict = {}
    symbols: List[str]
    interval: str = "1d"
    period: str = "1y"
    initial_capital: float = 100000.0
    schedule_cron: str = Field(..., min_length=1)  # e.g. "0 9 * * 1"


class ScheduledBacktestUpdate(BaseModel):
    name: Optional[str] = None
    is_active: Optional[bool] = None
    schedule_cron: Optional[str] = None
    strategy_params: Optional[dict] = None


@router.get("/")
async def list_schedules(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """List all scheduled backtests for the current user."""
    result = await db.execute(
        select(ScheduledBacktest).where(ScheduledBacktest.user_id == current_user.id).order_by(desc(ScheduledBacktest.created_at))
    )
    schedules = result.scalars().all()
    return [
        {
            "id": s.id,
            "name": s.name,
            "strategy_key": s.strategy_key,
            "strategy_params": s.strategy_params,
            "symbols": s.symbols,
            "interval": s.interval,
            "period": s.period,
            "initial_capital": s.initial_capital,
            "schedule_cron": s.schedule_cron,
            "is_active": s.is_active,
            "last_run_at": s.last_run_at.isoformat() if s.last_run_at else None,
            "next_run_at": s.next_run_at.isoformat() if s.next_run_at else None,
            "created_at": s.created_at.isoformat() if s.created_at else None,
        }
        for s in schedules
    ]


@router.post("/", status_code=201)
async def create_schedule(
    request: ScheduledBacktestCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new scheduled backtest."""
    # Limit schedules per user
    result = await db.execute(select(ScheduledBacktest).where(ScheduledBacktest.user_id == current_user.id))
    existing = result.scalars().all()
    if len(existing) >= 20:
        raise HTTPException(status_code=400, detail="Maximum 20 scheduled backtests allowed")

    schedule = ScheduledBacktest(
        user_id=current_user.id,
        name=request.name,
        strategy_key=request.strategy_key,
        strategy_params=request.strategy_params,
        symbols=request.symbols,
        interval=request.interval,
        period=request.period,
        initial_capital=request.initial_capital,
        schedule_cron=request.schedule_cron,
    )
    db.add(schedule)
    await db.commit()
    await db.refresh(schedule)
    return {"id": schedule.id, "message": f"Schedule '{request.name}' created"}


@router.patch("/{schedule_id}")
async def update_schedule(
    schedule_id: int,
    request: ScheduledBacktestUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Update a scheduled backtest."""
    result = await db.execute(
        select(ScheduledBacktest).where(
            ScheduledBacktest.id == schedule_id,
            ScheduledBacktest.user_id == current_user.id,
        )
    )
    schedule = result.scalar_one_or_none()
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")

    if request.name is not None:
        schedule.name = request.name
    if request.is_active is not None:
        schedule.is_active = request.is_active
    if request.schedule_cron is not None:
        schedule.schedule_cron = request.schedule_cron
    if request.strategy_params is not None:
        schedule.strategy_params = request.strategy_params

    await db.commit()
    return {"status": "updated"}


@router.delete("/{schedule_id}")
async def delete_schedule(
    schedule_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a scheduled backtest."""
    result = await db.execute(
        select(ScheduledBacktest).where(
            ScheduledBacktest.id == schedule_id,
            ScheduledBacktest.user_id == current_user.id,
        )
    )
    schedule = result.scalar_one_or_none()
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")

    await db.delete(schedule)
    await db.commit()
    return {"status": "deleted"}


@router.get("/{schedule_id}/runs")
async def get_run_history(
    schedule_id: int,
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Get run history for a scheduled backtest."""
    # Verify ownership
    result = await db.execute(
        select(ScheduledBacktest).where(
            ScheduledBacktest.id == schedule_id,
            ScheduledBacktest.user_id == current_user.id,
        )
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Schedule not found")

    result = await db.execute(
        select(ScheduledBacktestRun)
        .where(ScheduledBacktestRun.scheduled_backtest_id == schedule_id)
        .order_by(desc(ScheduledBacktestRun.started_at))
        .limit(limit)
    )
    runs = result.scalars().all()
    return [
        {
            "id": r.id,
            "status": r.status,
            "result_summary": r.result_summary,
            "error_message": r.error_message,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "completed_at": r.completed_at.isoformat() if r.completed_at else None,
        }
        for r in runs
    ]
