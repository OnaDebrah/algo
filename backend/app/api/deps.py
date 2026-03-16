"""
API dependencies
"""

import logging

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..core.permissions import Permission, is_promo_active
from ..database import get_db
from ..models.user import User
from ..security.rate_limiter import check_login_rate
from ..services.auth_service import AuthService
from ..services.quota_service import QuotaService

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)  # auto_error=False so cookie-only requests don't 403


def check_permission(permission: Permission):
    """Dependency to check if current user has a specific permission.

    During the launch promo window every tier gets all permissions
    (superusers always bypass).
    """

    async def permission_checker(current_user: User = Depends(get_current_active_user)) -> User:
        if current_user.is_superuser:
            logger.info(f"✅ Superuser bypass: {current_user.email} accessing {permission.value}")
            return current_user

        # During launch promo all features are unlocked for every tier
        if is_promo_active():
            logger.info(f"✅ Promo bypass: {current_user.email} accessing {permission.value}")
            return current_user

        # Regular permission check
        if not AuthService.has_permission(current_user.tier, permission):
            logger.warning(f"❌ Permission denied: {current_user.email} (tier={current_user.tier}) lacks permission {permission.value}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Feature '{permission.value}' requires a higher subscription tier. Current tier: {current_user.tier}",
            )

        logger.info(f"✅ Permission granted: {current_user.email} can access {permission.value}")
        return current_user

    return permission_checker


def enforce_backtest_quota():
    """Dependency factory — raises 429 if the user's monthly quota is exhausted."""

    async def checker(
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db),
    ) -> User:
        await QuotaService.enforce_quota(db, current_user.id, current_user.tier)
        return current_user

    return checker


async def login_rate_limit(request: Request):
    """Dependency that enforces stricter rate limits on the login endpoint.

    Allows max 5 login attempts per 60 seconds per client IP.
    """
    client_ip = request.client.host if request.client else "unknown"
    if not check_login_rate(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Try again in 1 minute.",
        )


async def get_current_user(
    request: Request,
    auth: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Get current authenticated user.

    Token resolution order:
    1. httpOnly cookie ``access_token`` (set by login/register endpoints)
    2. Bearer header (fallback for API clients / Swagger UI)
    """
    # 1. Try cookie first
    token = request.cookies.get("access_token")
    # 2. Fall back to Bearer header
    if not token and auth:
        token = auth.credentials
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
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


async def require_superuser(current_user: User = Depends(get_current_active_user)) -> User:
    """Dependency that restricts access to superusers only."""
    if not current_user.is_superuser:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return current_user
