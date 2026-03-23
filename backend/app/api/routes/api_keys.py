"""API Key management routes."""

import hashlib
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...api.deps import get_current_active_user
from ...database import get_db
from ...models import User
from ...models.api_key import ApiKey
from ...schemas.api_key import ApiKeyCreate, ApiKeyCreatedResponse, ApiKeyResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api-keys", tags=["API Keys"])


def _hash_key(key: str) -> str:
    """SHA-256 hash for fast lookup (not bcrypt — we need to look up by hash)."""
    return hashlib.sha256(key.encode()).hexdigest()


@router.get("/", response_model=List[ApiKeyResponse])
async def list_api_keys(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """List all API keys for the current user."""
    result = await db.execute(select(ApiKey).where(ApiKey.user_id == current_user.id).order_by(ApiKey.created_at.desc()))
    return result.scalars().all()


@router.post("/", response_model=ApiKeyCreatedResponse)
async def create_api_key(
    request: ApiKeyCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new API key. The full key is only returned once."""
    # Limit keys per user
    result = await db.execute(
        select(ApiKey).where(ApiKey.user_id == current_user.id, ApiKey.is_active == True)  # noqa: E712
    )
    active_keys = result.scalars().all()
    if len(active_keys) >= 10:
        raise HTTPException(status_code=400, detail="Maximum 10 active API keys allowed")

    # Generate key
    full_key = f"orc_{secrets.token_urlsafe(32)}"
    key_prefix = full_key[:8]
    key_hash = _hash_key(full_key)

    expires_at = None
    if request.expires_in_days:
        expires_at = datetime.now(timezone.utc) + timedelta(days=request.expires_in_days)

    api_key = ApiKey(
        user_id=current_user.id,
        key_prefix=key_prefix,
        key_hash=key_hash,
        name=request.name,
        permissions=request.permissions,
        expires_at=expires_at,
    )
    db.add(api_key)
    await db.commit()
    await db.refresh(api_key)

    return ApiKeyCreatedResponse(
        id=api_key.id,
        key_prefix=key_prefix,
        name=api_key.name,
        permissions=api_key.permissions,
        is_active=True,
        created_at=api_key.created_at,
        expires_at=api_key.expires_at,
        full_key=full_key,
    )


@router.delete("/{key_id}")
async def revoke_api_key(
    key_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Revoke (deactivate) an API key."""
    result = await db.execute(select(ApiKey).where(ApiKey.id == key_id, ApiKey.user_id == current_user.id))
    api_key = result.scalar_one_or_none()
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")

    api_key.is_active = False
    await db.commit()
    return {"status": "revoked"}


@router.post("/{key_id}/rotate", response_model=ApiKeyCreatedResponse)
async def rotate_api_key(
    key_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Rotate an API key — revoke old key and create a new one with same settings."""
    result = await db.execute(select(ApiKey).where(ApiKey.id == key_id, ApiKey.user_id == current_user.id))
    old_key = result.scalar_one_or_none()
    if not old_key:
        raise HTTPException(status_code=404, detail="API key not found")

    # Revoke old
    old_key.is_active = False

    # Create new with same settings
    full_key = f"orc_{secrets.token_urlsafe(32)}"
    new_key = ApiKey(
        user_id=current_user.id,
        key_prefix=full_key[:8],
        key_hash=_hash_key(full_key),
        name=old_key.name,
        permissions=old_key.permissions,
        expires_at=old_key.expires_at,
    )
    db.add(new_key)
    await db.commit()
    await db.refresh(new_key)

    return ApiKeyCreatedResponse(
        id=new_key.id,
        key_prefix=new_key.key_prefix,
        name=new_key.name,
        permissions=new_key.permissions,
        is_active=True,
        created_at=new_key.created_at,
        expires_at=new_key.expires_at,
        full_key=full_key,
    )
