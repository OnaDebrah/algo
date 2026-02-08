import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import get_current_active_user, get_db
from backend.app.models import User
from backend.app.schemas.social import ActivityResponse
from backend.app.services.social_service import ActivityService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/social", tags=["Social"])


@router.get("/activity", response_model=List[ActivityResponse])
async def get_activity_feed(
    limit: int = Query(50, ge=1, le=100), db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_active_user)
):
    """
    Get the global activity feed for the ecosystem
    """
    service = ActivityService(db)
    activities = await service.get_global_activities(limit=limit)
    return activities


@router.post("/activity/log")
async def log_manual_activity(
    content: str, activity_type: str = "MANUAL", db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_active_user)
):
    """
    Manually log an activity (Admin or specific triggers)
    """
    service = ActivityService(db)
    activity = await service.log_activity(user_id=current_user.id, activity_type=activity_type, content=content)
    if not activity:
        raise HTTPException(status_code=500, detail="Failed to log activity")
    return {"status": "success"}
