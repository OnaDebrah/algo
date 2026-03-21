"""Notification and PriceAlert schemas"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class NotificationOut(BaseModel):
    id: int
    type: str
    title: str
    message: str
    data: Optional[dict] = None
    is_read: bool
    created_at: datetime

    class Config:
        from_attributes = True


class NotificationList(BaseModel):
    notifications: list[NotificationOut]
    unread_count: int
    total: int


class PriceAlertCreate(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    condition: str = Field(..., pattern="^(above|below)$")
    target_price: float = Field(..., gt=0)


class PriceAlertOut(BaseModel):
    id: int
    symbol: str
    condition: str
    target_price: float
    is_active: bool
    triggered_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True
