"""
Notification service for creating, querying, and managing notifications
"""

import logging
from datetime import datetime, timezone

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.notification import Notification

logger = logging.getLogger(__name__)


class NotificationService:
    """Service for managing user notifications"""

    @staticmethod
    async def create_notification(
        db: AsyncSession,
        user_id: int,
        type: str,
        title: str,
        message: str,
        data: dict = None,
    ) -> Notification:
        """Create a new notification and attempt real-time delivery via WebSocket."""
        notif = Notification(
            user_id=user_id,
            type=type,
            title=title,
            message=message,
            data=data,
        )
        db.add(notif)
        await db.commit()
        await db.refresh(notif)

        # Attempt WebSocket push (non-critical)
        try:
            from ..api.routes.websocket import notification_manager

            await notification_manager.send_to_user(
                {"type": "new_notification", "notification": notif.to_dict()},
                user_id,
            )
        except Exception as e:
            logger.debug(f"WebSocket push failed (non-critical): {e}")

        return notif

    @staticmethod
    async def get_notifications(
        db: AsyncSession,
        user_id: int,
        limit: int = 50,
        offset: int = 0,
        unread_only: bool = False,
    ):
        """Get notifications for a user with pagination.

        Returns:
            Tuple of (notifications_list, unread_count, total_count)
        """
        query = select(Notification).where(Notification.user_id == user_id)
        if unread_only:
            query = query.where(Notification.is_read == False)  # noqa: E712
        query = query.order_by(Notification.created_at.desc()).offset(offset).limit(limit)

        result = await db.execute(query)
        notifications = list(result.scalars().all())

        # Total count
        total_q = select(func.count(Notification.id)).where(Notification.user_id == user_id)
        if unread_only:
            total_q = total_q.where(Notification.is_read == False)  # noqa: E712
        total = (await db.execute(total_q)).scalar() or 0

        # Unread count
        unread_q = select(func.count(Notification.id)).where(
            Notification.user_id == user_id,
            Notification.is_read == False,  # noqa: E712
        )
        unread_count = (await db.execute(unread_q)).scalar() or 0

        return notifications, unread_count, total

    @staticmethod
    async def mark_read(db: AsyncSession, user_id: int, notification_id: int):
        """Mark a single notification as read."""
        stmt = (
            update(Notification)
            .where(Notification.id == notification_id, Notification.user_id == user_id)
            .values(is_read=True, read_at=datetime.now(timezone.utc))
        )
        await db.execute(stmt)
        await db.commit()

    @staticmethod
    async def mark_all_read(db: AsyncSession, user_id: int):
        """Mark all unread notifications for a user as read."""
        stmt = (
            update(Notification)
            .where(Notification.user_id == user_id, Notification.is_read == False)  # noqa: E712
            .values(is_read=True, read_at=datetime.now(timezone.utc))
        )
        await db.execute(stmt)
        await db.commit()

    @staticmethod
    async def delete_notification(db: AsyncSession, user_id: int, notification_id: int):
        """Delete a notification."""
        stmt = delete(Notification).where(
            Notification.id == notification_id,
            Notification.user_id == user_id,
        )
        await db.execute(stmt)
        await db.commit()

    @staticmethod
    async def get_unread_count(db: AsyncSession, user_id: int) -> int:
        """Get the count of unread notifications for a user."""
        stmt = select(func.count(Notification.id)).where(
            Notification.user_id == user_id,
            Notification.is_read == False,  # noqa: E712
        )
        result = await db.execute(stmt)
        return result.scalar() or 0
