"""
Authentication schemas
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class UserBase(BaseModel):
    username: str
    email: EmailStr


class UserCreate(UserBase):
    password: str = Field(..., min_length=8)
    country: Optional[str] = None
    investor_type: Optional[str] = None
    risk_profile: Optional[str] = None


class UserLogin(BaseModel):
    email: str
    password: str


class User(UserBase):
    id: int
    tier: str
    is_active: bool
    is_superuser: bool
    created_at: datetime
    last_login: Optional[datetime]
    country: Optional[str] = None
    investor_type: Optional[str] = None
    risk_profile: Optional[str] = None

    totp_enabled: Optional[bool] = False

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: Optional[int] = None
    username: Optional[str] = None


class LoginResponse(BaseModel):
    user: Optional[User] = None
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    requires_2fa: bool = False
    pending_2fa_token: Optional[str] = None


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(..., min_length=8)


class TwoFactorSetupResponse(BaseModel):
    secret: str
    qr_uri: str
    qr_image_base64: str


class TwoFactorVerifyRequest(BaseModel):
    code: str = Field(..., min_length=6, max_length=8)


class TwoFactorVerifyLoginRequest(BaseModel):
    pending_2fa_token: str
    code: str = Field(..., min_length=6, max_length=8)


class TwoFactorBackupCodesResponse(BaseModel):
    backup_codes: list[str]
    message: str = "Save these backup codes securely. They can only be shown once."
