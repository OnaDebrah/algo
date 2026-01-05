"""
Extended user schemas
"""

from typing import Optional

from pydantic import BaseModel, EmailStr


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = None


class UserResponse(BaseModel):
    id: int
    email: str
    full_name: Optional[str] = None
    is_active: bool
    is_superuser: bool
    created_at: Optional[str] = None


class UserPreferences(BaseModel):
    theme: str = "dark"
    default_capital: float = 10000
    default_commission: float = 0.001
    risk_tolerance: str = "medium"
    notifications_enabled: bool = True
