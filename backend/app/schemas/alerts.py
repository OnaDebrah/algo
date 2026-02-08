from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr

from backend.app.schemas.alert import AlertChannel, AlertLevel


class EmailAlertRequest(BaseModel):
    subject: str
    message: str
    to_email: Optional[EmailStr] = None


class SMSAlertRequest(BaseModel):
    message: str
    to_number: Optional[str] = None


class AlertTestResponse(BaseModel):
    success: bool
    message: str
    email_status: str
    sms_status: str


class AlertPreferencesResponse(BaseModel):
    """Response model for alert preferences"""

    email: Optional[str] = None
    phone: Optional[str] = None
    email_enabled: bool = True
    sms_enabled: bool = False
    push_enabled: bool = False
    webhook_enabled: bool = False
    min_level_email: AlertLevel = AlertLevel.WARNING
    min_level_sms: AlertLevel = AlertLevel.ERROR
    min_level_push: AlertLevel = AlertLevel.WARNING
    min_level_webhook: AlertLevel = AlertLevel.WARNING
    webhook_url: Optional[str] = None

    class Config:
        from_attributes = True
        json_encoders = {AlertLevel: lambda v: v.value}


class AlertPreferencesUpdate(BaseModel):
    """Update model for alert preferences"""

    email: Optional[str] = None
    phone: Optional[str] = None
    email_enabled: Optional[bool] = None
    sms_enabled: Optional[bool] = None
    push_enabled: Optional[bool] = None
    webhook_enabled: Optional[bool] = None
    min_level_email: Optional[AlertLevel] = None
    min_level_sms: Optional[AlertLevel] = None
    min_level_push: Optional[AlertLevel] = None
    min_level_webhook: Optional[AlertLevel] = None
    webhook_url: Optional[str] = None


class TestAlertRequest(BaseModel):
    """Request model for test alert"""

    channel: str  # 'email' or 'sms'


class SuccessResponse(BaseModel):
    """Generic success response"""

    success: bool


class AlertResponse(BaseModel):
    """Response model for alert"""

    id: str
    level: AlertLevel
    title: str
    message: str
    strategy_id: Optional[int] = None
    metadata: Optional[dict] = None
    created_at: datetime
    sent_at: Optional[datetime] = None
    channels_sent: Optional[List[AlertChannel]] = None

    class Config:
        from_attributes = True
