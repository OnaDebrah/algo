"""API Key schemas."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class ApiKeyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    permissions: List[str] = ["read"]
    expires_in_days: Optional[int] = None  # None = no expiry
    allowed_ips: Optional[List[str]] = None  # IP whitelist: ["1.2.3.4", "10.0.0.0/8"]


class ApiKeyResponse(BaseModel):
    id: int
    key_prefix: str
    name: str
    permissions: List[str]
    is_active: bool
    created_at: datetime
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    allowed_ips: Optional[List[str]] = None

    model_config = {"from_attributes": True}


class ApiKeyCreatedResponse(ApiKeyResponse):
    """Only returned once at creation — includes the full plaintext key."""

    full_key: str


class ApiKeyIPUpdate(BaseModel):
    """Update allowed IPs for an API key."""

    allowed_ips: Optional[List[str]] = None  # None or [] = allow all IPs
