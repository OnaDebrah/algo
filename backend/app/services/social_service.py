import logging
from typing import Optional, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.models.social import Activity
from backend.app.schemas.social import ActivityCreate

logger = logging.getLogger(__name__)

class ActivityService:
    """
    Service for logging and retrieving social activities.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def log_activity(self, user_id: int, activity_type: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Logs a new activity in the ecosystem.
        """
        try:
            activity = Activity(
                user_id=user_id,
                activity_type=activity_type,
                content=content,
                metadata_json=metadata
            )
            self.db.add(activity)
            await self.db.commit()
            logger.info(f"Logged activity: {activity_type} for user {user_id}")
            return activity
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to log activity: {e}")
            return None

    async def get_global_activities(self, limit: int = 50):
        """
        Retrieves the latest global activities.
        """
        # Note: In a real implementation, we'd join with User to get usernames.
        from sqlalchemy import select, desc
        from backend.app.models import User
        
        stmt = select(Activity, User.username).join(User, Activity.user_id == User.id).order_by(desc(Activity.created_at)).limit(limit)
        result = await self.db.execute(stmt)
        
        activities = []
        for row in result.all():
            activity, username = row
            activities.append({
                "id": activity.id,
                "user_id": activity.user_id,
                "username": username,
                "activity_type": activity.activity_type,
                "content": activity.content,
                "metadata_json": activity.metadata_json,
                "created_at": activity.created_at
            })
            
        return activities
