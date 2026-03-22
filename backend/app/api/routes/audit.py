"""Audit trail / trade journal routes."""

import csv
import io
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ...api.deps import get_current_active_user
from ...database import get_db
from ...models import User
from ...schemas.audit import AuditEventList, AuditEventOut, AuditEventUpdate
from ...services.audit_service import AuditService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audit", tags=["Audit Trail"])


def _event_to_out(event) -> AuditEventOut:
    return AuditEventOut(
        id=event.id,
        user_id=event.user_id,
        event_type=event.event_type,
        category=event.category,
        title=event.title,
        description=event.description,
        metadata=event.metadata_,
        tags=event.tags,
        notes=event.notes,
        created_at=event.created_at,
    )


@router.get("/events", response_model=AuditEventList)
async def get_events(
    event_type: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Get paginated audit events with optional filters."""
    events, total = await AuditService.get_events(
        db,
        current_user.id,
        event_type=event_type,
        category=category,
        search=search,
        page=page,
        page_size=page_size,
    )
    return AuditEventList(
        events=[_event_to_out(e) for e in events],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/journal", response_model=AuditEventList)
async def get_journal(
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Get trade journal entries (trade events with notes)."""
    events, total = await AuditService.get_events(
        db,
        current_user.id,
        event_type="trade",
        category="trade_journal",
        search=search,
        page=page,
        page_size=page_size,
    )
    return AuditEventList(
        events=[_event_to_out(e) for e in events],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.patch("/events/{event_id}/notes", response_model=AuditEventOut)
async def update_event_notes(
    event_id: int,
    update: AuditEventUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Update notes and tags on an audit event."""
    event = await AuditService.update_notes(
        db,
        current_user.id,
        event_id,
        notes=update.notes,
        tags=update.tags,
    )
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return _event_to_out(event)


@router.get("/events/export/csv")
async def export_events_csv(
    event_type: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Export audit events as CSV."""
    events, _ = await AuditService.get_events(db, current_user.id, event_type=event_type, page=1, page_size=10000)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Date", "Type", "Category", "Title", "Description", "Tags", "Notes"])
    for e in events:
        writer.writerow(
            [
                e.created_at.isoformat() if e.created_at else "",
                e.event_type,
                e.category,
                e.title,
                e.description or "",
                ", ".join(e.tags or []),
                e.notes or "",
            ]
        )
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="audit_trail.csv"'},
    )
