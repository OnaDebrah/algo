"""API Key management routes."""

import hashlib
import ipaddress
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...api.deps import get_current_active_user
from ...database import get_db
from ...models import User
from ...models.api_key import ApiKey
from ...schemas.api_key import (
    ApiKeyCreate,
    ApiKeyCreatedResponse,
    ApiKeyIPUpdate,
    ApiKeyResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api-keys", tags=["API Keys"])


def _hash_key(key: str) -> str:
    """SHA-256 hash for fast lookup (not bcrypt — we need to look up by hash)."""
    return hashlib.sha256(key.encode()).hexdigest()


def _validate_ip_list(ips: List[str]) -> List[str]:
    """Validate a list of IP addresses or CIDR ranges. Returns the validated list.

    Raises HTTPException if any entry is invalid.
    """
    validated = []
    for ip_str in ips:
        ip_str = ip_str.strip()
        if not ip_str:
            continue
        try:
            if "/" in ip_str:
                # CIDR notation
                ipaddress.ip_network(ip_str, strict=False)
            else:
                # Single IP
                ipaddress.ip_address(ip_str)
            validated.append(ip_str)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid IP address or CIDR range: '{ip_str}'",
            )
    return validated


def _check_ip_allowed(client_ip: str, allowed_ips: Optional[List[str]]) -> bool:
    """Check if a client IP is allowed by the whitelist.

    Returns True if:
    - allowed_ips is None or empty (no restrictions)
    - client_ip matches any entry in allowed_ips (exact or CIDR)
    """
    if not allowed_ips:
        return True

    try:
        addr = ipaddress.ip_address(client_ip)
    except ValueError:
        return False

    for entry in allowed_ips:
        try:
            if "/" in entry:
                if addr in ipaddress.ip_network(entry, strict=False):
                    return True
            else:
                if addr == ipaddress.ip_address(entry):
                    return True
        except ValueError:
            continue

    return False


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

    # Validate allowed IPs if provided
    allowed_ips = None
    if request.allowed_ips:
        allowed_ips = _validate_ip_list(request.allowed_ips)

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
        allowed_ips=allowed_ips,
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
        allowed_ips=api_key.allowed_ips,
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

    # Create new with same settings (including IP whitelist)
    full_key = f"orc_{secrets.token_urlsafe(32)}"
    new_key = ApiKey(
        user_id=current_user.id,
        key_prefix=full_key[:8],
        key_hash=_hash_key(full_key),
        name=old_key.name,
        permissions=old_key.permissions,
        expires_at=old_key.expires_at,
        allowed_ips=old_key.allowed_ips,
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
        allowed_ips=new_key.allowed_ips,
        full_key=full_key,
    )


@router.put("/{key_id}/whitelist", response_model=ApiKeyResponse)
async def update_ip_whitelist(
    key_id: int,
    request: ApiKeyIPUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Update the IP whitelist for an API key.

    Pass `allowed_ips: null` or `allowed_ips: []` to remove restrictions (allow all IPs).
    Pass a list of IPs/CIDRs to restrict usage to those addresses.
    """
    result = await db.execute(select(ApiKey).where(ApiKey.id == key_id, ApiKey.user_id == current_user.id))
    api_key = result.scalar_one_or_none()
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")

    if not api_key.is_active:
        raise HTTPException(status_code=400, detail="Cannot update a revoked API key")

    # Validate and set allowed IPs
    if request.allowed_ips:
        api_key.allowed_ips = _validate_ip_list(request.allowed_ips)
    else:
        api_key.allowed_ips = None

    await db.commit()
    await db.refresh(api_key)
    return api_key
