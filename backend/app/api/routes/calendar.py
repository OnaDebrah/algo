"""Economic calendar routes."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, func as sa_func
from sqlalchemy.ext.asyncio import AsyncSession

from ...api.deps import get_current_active_user
from ...database import get_db
from ...models import User
from ...models.economic_event import EconomicEvent
from ...services.calendar_sync import sync_calendar

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/calendar", tags=["Economic Calendar"])


class EventCreate(BaseModel):
    event_name: str
    country: str = "US"
    event_date: datetime
    impact: str = "medium"
    previous_value: Optional[str] = None
    forecast_value: Optional[str] = None
    actual_value: Optional[str] = None
    category: Optional[str] = None


async def _auto_seed_if_empty(db: AsyncSession):
    """Seed calendar on first access if table is empty."""
    result = await db.execute(select(sa_func.count()).select_from(EconomicEvent))
    count = result.scalar() or 0
    if count == 0:
        logger.info("Economic calendar empty — auto-seeding events")
        await sync_calendar(db)


@router.post("/sync")
async def sync_events(
    months: int = Query(6, ge=1, le=12),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Refresh generated economic calendar events."""
    count = await sync_calendar(db, months)
    return {"message": f"Synced {count} events", "count": count}


@router.get("/events")
async def get_events(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    impact: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Get economic calendar events."""
    await _auto_seed_if_empty(db)
    query = select(EconomicEvent).order_by(EconomicEvent.event_date)

    if start_date:
        query = query.where(EconomicEvent.event_date >= datetime.fromisoformat(start_date))
    if end_date:
        query = query.where(EconomicEvent.event_date <= datetime.fromisoformat(end_date))
    if impact:
        query = query.where(EconomicEvent.impact == impact)
    if category:
        query = query.where(EconomicEvent.category == category)

    result = await db.execute(query.limit(200))
    events = result.scalars().all()

    return [
        {
            "id": e.id,
            "event_name": e.event_name,
            "country": e.country,
            "event_date": e.event_date.isoformat() if e.event_date else None,
            "impact": e.impact,
            "previous_value": e.previous_value,
            "forecast_value": e.forecast_value,
            "actual_value": e.actual_value,
            "category": e.category,
        }
        for e in events
    ]


@router.get("/upcoming")
async def get_upcoming(
    days: int = Query(7, ge=1, le=30),
    impact: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Get upcoming economic events for the next N days."""
    await _auto_seed_if_empty(db)
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=days)

    query = (
        select(EconomicEvent)
        .where(EconomicEvent.event_date >= now, EconomicEvent.event_date <= end)
        .order_by(EconomicEvent.event_date)
    )
    if impact:
        query = query.where(EconomicEvent.impact == impact)

    result = await db.execute(query)
    events = result.scalars().all()

    return [
        {
            "id": e.id,
            "event_name": e.event_name,
            "country": e.country,
            "event_date": e.event_date.isoformat() if e.event_date else None,
            "impact": e.impact,
            "previous_value": e.previous_value,
            "forecast_value": e.forecast_value,
            "actual_value": e.actual_value,
            "category": e.category,
        }
        for e in events
    ]


@router.post("/events")
async def create_event(
    request: EventCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Create an economic event (admin/manual entry)."""
    event = EconomicEvent(
        event_name=request.event_name,
        country=request.country,
        event_date=request.event_date,
        impact=request.impact,
        previous_value=request.previous_value,
        forecast_value=request.forecast_value,
        actual_value=request.actual_value,
        category=request.category,
    )
    db.add(event)
    await db.commit()
    await db.refresh(event)
    return {"id": event.id, "message": "Event created"}
