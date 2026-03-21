"""API Key schemas."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class ApiKeyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    permissions: List[str] = ["read"]
    expires_in_days: Optional[int] = None  # None = no expiry


class ApiKeyResponse(BaseModel):
    id: int
    key_prefix: str
    name: str
    permissions: List[str]
    is_active: bool
    created_at: datetime
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class ApiKeyCreatedResponse(ApiKeyResponse):
    """Only returned once at creation — includes the full plaintext key."""
    full_key: str
