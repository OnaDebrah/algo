"""
API dependencies
"""

import logging

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import settings
from backend.app.core.permissions import Permission
from backend.app.database import get_db
from backend.app.models.user import User
from backend.app.services.auth_service import AuthService

logger = logging.getLogger(__name__)

security = HTTPBearer()


def check_permission(permission: Permission):
    """Dependency to check if current user has a specific permission"""

    async def permission_checker(current_user: User = Depends(get_current_active_user)) -> User:
        if current_user.is_superuser:
            logger.info(f"✅ Superuser bypass: {current_user.email} accessing {permission.value}")
            return current_user

        # Regular permission check
        if not AuthService.has_permission(current_user.tier, permission):
            logger.warning(f"❌ Permission denied: {current_user.email} (tier={current_user.tier}) " f"lacks permission {permission.value}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Feature '{permission.value}' requires a higher subscription tier. " f"Current tier: {current_user.tier}",
            )

        logger.info(f"✅ Permission granted: {current_user.email} can access {permission.value}")
        return current_user

    return permission_checker


async def get_current_user(auth: HTTPAuthorizationCredentials = Depends(security), db: AsyncSession = Depends(get_db)) -> User:
    """Get current authenticated user using HTTPBearer"""
    token = auth.credentials
    logger.info(f"🔐 Auth attempt with token: {token[:20]}...")

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            logger.warning("❌ Token missing 'sub' claim")
            raise credentials_exception

        logger.info(f"✅ Token decoded successfully for user_id: {user_id}")

    except JWTError as e:
        logger.warning(f"❌ JWT decode error: {str(e)}")
        raise credentials_exception
    except Exception as e:
        logger.error(f"❌ Unexpected auth error: {str(e)}")
        raise credentials_exception

    # Async query
    result = await db.execute(select(User).filter(User.id == int(user_id)))
    user = result.scalar_one_or_none()

    if user is None:
        logger.warning(f"❌ User not found: {user_id}")
        raise credentials_exception

    logger.info(f"✅ User authenticated: {user.email} (tier={user.tier}, superuser={user.is_superuser})")
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Ensure user is active"""
    if not current_user.is_active:
        logger.warning(f"❌ Inactive user attempted access: {current_user.email}")
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
