from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import get_current_active_user, get_db
from backend.app.config import settings
from backend.app.models.user import User
from backend.app.schemas.alerts import EmailAlertRequest, SMSAlertRequest, AlertTestResponse
from backend.app.services.auth_service import AuthService
from backend.app.alerts import AlertManager

router = APIRouter(prefix="/alerts", tags=["Alerts"])

def get_alert_manager():
    """Dependency to get a configured AlertManager instance"""
    email_config = {
        "enabled": settings.EMAIL_ENABLED,
        "smtp_server": settings.SMTP_SERVER,
        "smtp_port": settings.SMTP_PORT,
        "from_email": settings.FROM_EMAIL,
        "password": settings.SMTP_PASSWORD,
        "to_email": settings.TO_EMAIL
    }
    sms_config = {
        "enabled": settings.SMS_ENABLED,
        "account_sid": settings.TWILIO_ACCOUNT_SID,
        "auth_token": settings.TWILIO_AUTH_TOKEN,
        "from_number": settings.TWILIO_FROM_NUMBER,
        "to_number": settings.TWILIO_TO_NUMBER
    }
    return AlertManager(email_config=email_config, sms_config=sms_config)

@router.post("/email")
async def send_email_alert(
    request: EmailAlertRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    alert_manager: AlertManager = Depends(get_alert_manager)
):
    """Send a manual email alert"""
    # Track usage
    await AuthService.track_usage(db, current_user.id, "send_email_alert", {"subject": request.subject})
    
    # Update manager with request-specific 'to' if provided
    manager_config = alert_manager.email_config.copy()
    if request.to_email:
        manager_config["to_email"] = request.to_email
    
    # Create temp manager for this specific recipient if needed
    temp_manager = AlertManager(email_config=manager_config, sms_config=alert_manager.sms_config)
    temp_manager.send_email_alert(request.subject, request.message)
    
    return {"message": "Email alert request processed"}

@router.post("/sms")
async def send_sms_alert(
    request: SMSAlertRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    alert_manager: AlertManager = Depends(get_alert_manager)
):
    """Send a manual SMS alert"""
    # Track usage
    await AuthService.track_usage(db, current_user.id, "send_sms_alert")
    
    # Update manager with request-specific 'to' if provided
    sms_config = alert_manager.sms_config.copy()
    if request.to_number:
        sms_config["to_number"] = request.to_number
        
    temp_manager = AlertManager(email_config=alert_manager.email_config, sms_config=sms_config)
    temp_manager.send_sms_alert(request.message)
    
    return {"message": "SMS alert request processed"}

@router.get("/test", response_model=AlertTestResponse)
async def test_alerts(
    current_user: User = Depends(get_current_active_user),
    alert_manager: AlertManager = Depends(get_alert_manager)
):
    """Check alert system configuration status"""
    return AlertTestResponse(
        success=True,
        message="Alert system status retrieved",
        email_status="Enabled" if alert_manager.email_config.get("enabled") else "Disabled",
        sms_status="Enabled" if alert_manager.sms_config.get("enabled") else "Disabled"
    )
