import json
import logging
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.permissions import TIER_PERMISSIONS, Permission, UserTier
from backend.app.models.usage import UsageTracking
from backend.app.models.user import User

logger = logging.getLogger(__name__)


class AuthService:
    """Service for authentication and authorization logic"""

    @staticmethod
    def has_permission(user_tier: str, permission: Permission) -> bool:
        """Check if a user tier has a specific permission"""
        try:
            tier = UserTier(user_tier)
            return permission in TIER_PERMISSIONS.get(tier, [])
        except ValueError:
            logger.warning(f"Invalid user tier: {user_tier}")
            return False

    @staticmethod
    def get_tier_permissions(user_tier: str) -> List[Permission]:
        """Get all permissions for a specific user tier"""
        try:
            tier = UserTier(user_tier)
            return TIER_PERMISSIONS.get(tier, [])
        except ValueError:
            return []

    @staticmethod
    async def track_usage(db: AsyncSession, user_id: int, action: str, metadata: Optional[dict] = None) -> None:
        """Track user action in the database"""
        try:
            usage = UsageTracking(user_id=user_id, action=action, metadata_json=json.dumps(metadata) if metadata else None)
            db.add(usage)
            await db.commit()
        except Exception as e:
            logger.error(f"Failed to track usage for user {user_id}: {str(e)}")
            # We don't raise here to avoid breaking the main flow if tracking fails

    @staticmethod
    async def update_user_tier(db: AsyncSession, user_id: int, new_tier: UserTier) -> bool:
        """Update a user's subscription tier"""
        try:
            from sqlalchemy import select

            result = await db.execute(select(User).where(User.id == user_id))
            user = result.scalar_one_or_none()
            if user:
                user.tier = new_tier.value
                await db.commit()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to update tier for user {user_id}: {str(e)}")
            return False
