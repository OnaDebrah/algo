from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel


class ActivityBase(BaseModel):
    activity_type: str
    content: str
    metadata_json: Optional[Dict[str, Any]] = None


class ActivityCreate(ActivityBase):
    user_id: int


class ActivityResponse(ActivityBase):
    id: int
    user_id: int
    created_at: datetime
    username: Optional[str] = None

    class Config:
        from_attributes = True
