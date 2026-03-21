"""
API routes for notifications and price alerts
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import get_db
from ...api.deps import get_current_active_user
from ...models.user import User
from ...schemas.notification import NotificationOut, NotificationList, PriceAlertCreate, PriceAlertOut
from ...services.notification_service import NotificationService
from ...services.price_alert_service import PriceAlertService

router = APIRouter(tags=["Notifications"])


# ── Notification endpoints ──────────────────────────────────────────────────


@router.get("/notifications", response_model=NotificationList)
async def get_notifications(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    unread_only: bool = Query(False),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get paginated notifications for the current user."""
    notifications, unread_count, total = await NotificationService.get_notifications(
        db, current_user.id, limit=limit, offset=offset, unread_only=unread_only
    )
    return NotificationList(
        notifications=notifications,
        unread_count=unread_count,
        total=total,
    )


@router.get("/notifications/unread-count")
async def get_unread_count(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get count of unread notifications."""
    count = await NotificationService.get_unread_count(db, current_user.id)
    return {"count": count}


@router.post("/notifications/{notification_id}/read")
async def mark_notification_read(
    notification_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Mark a single notification as read."""
    await NotificationService.mark_read(db, current_user.id, notification_id)
    return {"message": "Notification marked as read"}


@router.post("/notifications/read-all")
async def mark_all_notifications_read(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Mark all notifications as read."""
    await NotificationService.mark_all_read(db, current_user.id)
    return {"message": "All notifications marked as read"}


@router.delete("/notifications/{notification_id}")
async def delete_notification(
    notification_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Delete a notification."""
    await NotificationService.delete_notification(db, current_user.id, notification_id)
    return {"message": "Notification deleted"}


# ── Price Alert endpoints ───────────────────────────────────────────────────


@router.get("/price-alerts", response_model=list[PriceAlertOut])
async def get_price_alerts(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get all price alerts for the current user."""
    return await PriceAlertService.get_alerts(db, current_user.id)


@router.post("/price-alerts", response_model=PriceAlertOut)
async def create_price_alert(
    alert: PriceAlertCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Create a new price alert."""
    return await PriceAlertService.create_alert(
        db,
        current_user.id,
        symbol=alert.symbol,
        condition=alert.condition,
        target_price=alert.target_price,
    )


@router.delete("/price-alerts/{alert_id}")
async def delete_price_alert(
    alert_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Delete a price alert."""
    await PriceAlertService.delete_alert(db, current_user.id, alert_id)
    return {"message": "Price alert deleted"}
