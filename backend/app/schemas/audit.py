"""Audit trail / trade journal schemas."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AuditEventCreate(BaseModel):
    event_type: str
    category: str = "system"
    title: str
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


class AuditEventOut(BaseModel):
    id: int
    user_id: int
    event_type: str
    category: str
    title: str
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class AuditEventUpdate(BaseModel):
    notes: Optional[str] = None
    tags: Optional[List[str]] = None


class AuditEventList(BaseModel):
    events: List[AuditEventOut]
    total: int
    page: int
    page_size: int
