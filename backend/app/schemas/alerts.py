from typing import Optional

from pydantic import BaseModel, EmailStr


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
