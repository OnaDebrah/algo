"""Audit trail service with hash chain integrity."""

import hashlib
import json
import logging

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.audit import AuditEvent

logger = logging.getLogger(__name__)


class AuditService:

    @staticmethod
    def _compute_hash(event_type: str, title: str, metadata: dict | None, prev_hash: str | None) -> str:
        content = json.dumps({
            "event_type": event_type,
            "title": title,
            "metadata": metadata or {},
            "prev_hash": prev_hash or "",
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    @staticmethod
    async def log_event(
        db: AsyncSession,
        user_id: int,
        event_type: str,
        title: str,
        category: str = "system",
        description: str | None = None,
        metadata: dict | None = None,
        tags: list[str] | None = None,
    ) -> AuditEvent:
        """Create an audit event with hash chain integrity."""
        # Get previous hash
        result = await db.execute(
            select(AuditEvent.event_hash)
            .where(AuditEvent.user_id == user_id)
            .order_by(desc(AuditEvent.id))
            .limit(1)
        )
        prev = result.scalar_one_or_none()

        event_hash = AuditService._compute_hash(event_type, title, metadata, prev)

        event = AuditEvent(
            user_id=user_id,
            event_type=event_type,
            category=category,
            title=title,
            description=description,
            metadata_=metadata,
            tags=tags,
            prev_hash=prev,
            event_hash=event_hash,
        )
        db.add(event)
        await db.commit()
        await db.refresh(event)
        return event

    @staticmethod
    async def get_events(
        db: AsyncSession,
        user_id: int,
        event_type: str | None = None,
        category: str | None = None,
        tags: list[str] | None = None,
        search: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[AuditEvent], int]:
        """Get paginated audit events with optional filters."""
        query = select(AuditEvent).where(AuditEvent.user_id == user_id)
        count_query = select(func.count(AuditEvent.id)).where(AuditEvent.user_id == user_id)

        if event_type:
            query = query.where(AuditEvent.event_type == event_type)
            count_query = count_query.where(AuditEvent.event_type == event_type)
        if category:
            query = query.where(AuditEvent.category == category)
            count_query = count_query.where(AuditEvent.category == category)
        if search:
            search_filter = AuditEvent.title.ilike(f"%{search}%")
            query = query.where(search_filter)
            count_query = count_query.where(search_filter)

        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0

        query = query.order_by(desc(AuditEvent.created_at))
        query = query.offset((page - 1) * page_size).limit(page_size)

        result = await db.execute(query)
        events = result.scalars().all()

        return list(events), total

    @staticmethod
    async def update_notes(
        db: AsyncSession,
        user_id: int,
        event_id: int,
        notes: str | None = None,
        tags: list[str] | None = None,
    ) -> AuditEvent | None:
        """Update notes and tags on an audit event."""
        result = await db.execute(
            select(AuditEvent).where(AuditEvent.id == event_id, AuditEvent.user_id == user_id)
        )
        event = result.scalar_one_or_none()
        if not event:
            return None

        if notes is not None:
            event.notes = notes
        if tags is not None:
            event.tags = tags

        await db.commit()
        await db.refresh(event)
        return event
