import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.alerts.alert_manager import AlertManager
from backend.app.alerts.alert_preferences import AlertPreferences
from backend.app.alerts.email_provider import EmailProvider
from backend.app.alerts.sms_provider import SMSProvider
from backend.app.api.deps import get_current_active_user, get_db
from backend.app.config import settings
from backend.app.models.user import User
from backend.app.schemas.alert import AlertChannel, AlertLevel
from backend.app.schemas.alerts import (
    AlertPreferencesResponse,
    AlertPreferencesUpdate,
    AlertResponse,
    AlertTestResponse,
    EmailAlertRequest,
    SMSAlertRequest,
    SuccessResponse,
    TestAlertRequest,
)
from backend.app.services.auth_service import AuthService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/alerts", tags=["Alerts"])

_alert_manager_instance: Optional[AlertManager] = None


async def init_alert_manager() -> AlertManager:
    """Initialize and start the AlertManager singleton"""
    global _alert_manager_instance

    if _alert_manager_instance is None:
        email_provider = None
        if settings.EMAIL_ENABLED:
            email_provider = EmailProvider(
                smtp_host=settings.SMTP_SERVER,
                smtp_port=settings.SMTP_PORT,
                username=settings.FROM_EMAIL,
                password=settings.SMTP_PASSWORD,
                from_email=settings.FROM_EMAIL,
                from_name=settings.EMAIL_FROM_NAME or "Trading Platform",
            )

        sms_provider = None
        if settings.SMS_ENABLED and settings.TWILIO_ACCOUNT_SID:
            sms_provider = SMSProvider(
                provider="twilio",
                account_sid=settings.TWILIO_ACCOUNT_SID,
                auth_token=settings.TWILIO_AUTH_TOKEN,
                from_number=settings.TWILIO_FROM_NUMBER,
            )

        _alert_manager_instance = AlertManager(email_provider=email_provider, sms_provider=sms_provider)

        await _alert_manager_instance.start()

        default_prefs = AlertPreferences(
            email=settings.TO_EMAIL if settings.EMAIL_ENABLED else None,
            phone=settings.TWILIO_TO_NUMBER if settings.SMS_ENABLED else None,
            email_enabled=settings.EMAIL_ENABLED,
            sms_enabled=settings.SMS_ENABLED,
            min_level_email=AlertLevel.WARNING,
            min_level_sms=AlertLevel.ERROR,
        )

        # You can set a default user ID or handle per-user preferences in endpoints
        # For now, we'll set a default for user 1
        _alert_manager_instance.user_preferences[1] = default_prefs

        logger.info("AlertManager initialized and started")

    return _alert_manager_instance


async def get_alert_manager() -> AlertManager:
    """Dependency to get the AlertManager instance"""
    return await init_alert_manager()


@router.post("/email")
async def send_email_alert(
    request: EmailAlertRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    alert_manager: AlertManager = Depends(get_alert_manager),
):
    """Send a manual email alert using the new AlertManager"""
    try:
        # Track usage
        await AuthService.track_usage(db, current_user.id, "send_email_alert", {"subject": request.subject, "level": request.level.value})

        # Determine recipient email
        to_email = request.to_email or alert_manager.user_preferences.get(current_user.id, AlertPreferences.default()).email

        if not to_email:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No email address configured for recipient")

        # Update or create user preferences for this user
        user_prefs = alert_manager.user_preferences.get(current_user.id)
        if not user_prefs:
            user_prefs = AlertPreferences(
                email=to_email,
                email_enabled=True,
                min_level_email=AlertLevel.INFO,  # Allow all levels for manual sends
            )
            alert_manager.user_preferences[current_user.id] = user_prefs
        elif to_email != user_prefs.email:
            # Update email if different
            user_prefs.email = to_email

        # Send alert using the new async method
        success = await alert_manager.send_alert(
            user_id=current_user.id,
            level=request.level,
            title=request.subject,
            message=request.message,
            channels=[AlertChannel.EMAIL],  # Explicitly send via email only
        )

        if not success:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Alert rate limited. Please wait before sending another.")

        return {"message": "Email alert queued successfully", "queued": True}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to send email alert: {str(e)}")


@router.post("/sms")
async def send_sms_alert(
    request: SMSAlertRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    alert_manager: AlertManager = Depends(get_alert_manager),
):
    """Send a manual SMS alert using the new AlertManager"""
    try:
        # Track usage
        await AuthService.track_usage(db, current_user.id, "send_sms_alert", {"level": request.level.value})

        # Determine recipient phone number
        user_prefs = alert_manager.user_preferences.get(current_user.id)
        to_number = request.to_number or (user_prefs.phone if user_prefs else None)

        if not to_number:
            # Try to get default from settings
            default_prefs = alert_manager.user_preferences.get(1)
            to_number = request.to_number or (default_prefs.phone if default_prefs else None)

        if not to_number:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No phone number configured for recipient")

        # Update or create user preferences
        if not user_prefs:
            user_prefs = AlertPreferences(
                phone=to_number,
                sms_enabled=True,
                min_level_sms=AlertLevel.INFO,  # Allow all levels for manual sends
            )
            alert_manager.user_preferences[current_user.id] = user_prefs
        elif to_number != user_prefs.phone:
            # Update phone if different
            user_prefs.phone = to_number

        # Send alert using the new async method
        success = await alert_manager.send_alert(
            user_id=current_user.id,
            level=request.level,
            title="SMS Alert",  # SMS doesn't need a separate title
            message=request.message,
            channels=[AlertChannel.SMS],  # Explicitly send via SMS only
        )

        if not success:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Alert rate limited. Please wait before sending another.")

        return {"message": "SMS alert queued successfully", "queued": True}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to send SMS alert: {str(e)}")


@router.get("/test", response_model=AlertTestResponse)
async def test_alerts(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    alert_manager: AlertManager = Depends(get_alert_manager),
):
    """Test alert system configuration"""
    try:
        await AuthService.track_usage(db, current_user.id, "test_alerts")

        # Get user preferences
        user_prefs = alert_manager.user_preferences.get(current_user.id)

        # Determine email status
        email_configured = alert_manager.email_provider is not None and user_prefs and user_prefs.email_enabled and user_prefs.email

        # Determine SMS status
        sms_configured = alert_manager.sms_provider is not None and user_prefs and user_prefs.sms_enabled and user_prefs.phone

        return AlertTestResponse(
            success=True,
            message="Alert system status retrieved",
            email_status="Configured and enabled" if email_configured else "Not configured or disabled",
            sms_status="Configured and enabled" if sms_configured else "Not configured or disabled",
        )

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to test alert system: {str(e)}")


@router.get("/preferences", response_model=AlertPreferencesResponse)
async def get_preferences(current_user=Depends(get_current_active_user), alert_manager: AlertManager = Depends(get_alert_manager)):
    """
    Get user's alert preferences
    """
    user_id = current_user.id

    prefs = alert_manager.user_preferences.get(user_id)

    if not prefs:
        default_prefs = AlertPreferences.default()
        alert_manager.user_preferences[user_id] = default_prefs
        prefs = default_prefs

    return AlertPreferencesResponse(
        email=prefs.email,
        phone=prefs.phone,
        email_enabled=prefs.email_enabled,
        sms_enabled=prefs.sms_enabled,
        push_enabled=prefs.push_enabled,
        webhook_enabled=prefs.webhook_enabled,
        min_level_email=prefs.min_level_email,
        min_level_sms=prefs.min_level_sms,
        min_level_push=prefs.min_level_push,
        min_level_webhook=prefs.min_level_webhook,
        webhook_url=prefs.webhook_url,
    )


@router.put("/preferences", response_model=AlertPreferencesResponse)
async def update_preferences(
    preferences: AlertPreferencesUpdate, current_user=Depends(get_current_active_user), alert_manager: AlertManager = Depends(get_alert_manager)
):
    """
    Update user's alert preferences
    """
    try:
        user_id = current_user.id

        existing_prefs = alert_manager.user_preferences.get(user_id)

        if existing_prefs:
            update_data = preferences.dict(exclude_unset=True)
            for key, value in update_data.items():
                setattr(existing_prefs, key, value)
        else:
            prefs_dict = preferences.dict(exclude_unset=True)
            existing_prefs = AlertPreferences(**prefs_dict)
            alert_manager.user_preferences[user_id] = existing_prefs

        if existing_prefs.webhook_enabled and not existing_prefs.webhook_url:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Webhook URL is required when webhook notifications are enabled")

        if existing_prefs.email_enabled and not existing_prefs.email:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email address is required when email notifications are enabled")

        if existing_prefs.sms_enabled and not existing_prefs.phone:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Phone number is required when SMS notifications are enabled")

        return AlertPreferencesResponse(
            email=existing_prefs.email,
            phone=existing_prefs.phone,
            email_enabled=existing_prefs.email_enabled,
            sms_enabled=existing_prefs.sms_enabled,
            push_enabled=existing_prefs.push_enabled,
            webhook_enabled=existing_prefs.webhook_enabled,
            min_level_email=existing_prefs.min_level_email,
            min_level_sms=existing_prefs.min_level_sms,
            min_level_push=existing_prefs.min_level_push,
            min_level_webhook=existing_prefs.min_level_webhook,
            webhook_url=existing_prefs.webhook_url,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to update preferences: {str(e)}")


@router.get("/history", response_model=List[AlertResponse])
async def get_history(
    limit: int = 10,
    strategy_id: Optional[int] = None,
    level: Optional[AlertLevel] = None,
    current_user: User = Depends(get_current_active_user),
    alert_manager: AlertManager = Depends(get_alert_manager),
):
    """
    Get alert history with optional filters
    """
    try:
        user_id = current_user.id
        history = alert_manager.get_alert_history(user_id=user_id, strategy_id=strategy_id, level=level, limit=limit)

        return [
            AlertResponse(
                id=str(i),
                level=alert.level,
                title=alert.title,
                message=alert.message,
                strategy_id=alert.strategy_id,
                metadata=alert.metadata,
                created_at=alert.created_at,
                sent_at=alert.sent_at,
                channels_sent=alert.channels_sent,
            )
            for i, alert in enumerate(history)
        ]

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to get history: {str(e)}")


@router.post("/test", response_model=SuccessResponse)
async def send_test(
    request: TestAlertRequest, current_user: User = Depends(get_current_active_user), alert_manager: AlertManager = Depends(get_alert_manager)
):
    """
    Send a test alert to the specified channel
    """
    try:
        user_id = current_user.id
        if request.channel.lower() not in ["email", "sms"]:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Channel must be 'email' or 'sms'")

        prefs = alert_manager.user_preferences.get(user_id)
        if not prefs:
            prefs = AlertPreferences.default()
            alert_manager.user_preferences[user_id] = prefs

        if request.channel.lower() == "email" and not prefs.email:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email not configured in preferences")
        elif request.channel.lower() == "sms" and not prefs.phone:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Phone number not configured in preferences")

        test_sent = await alert_manager.send_alert(
            user_id=user_id,
            level=AlertLevel.INFO,
            title="Test Alert",
            message="This is a test alert from the alert system",
            channels=[AlertChannel.EMAIL if request.channel.lower() == "email" else AlertChannel.SMS],
        )

        if not test_sent:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Alert rate limited. Please wait before sending another test.")

        return SuccessResponse(success=True)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to send test alert: {str(e)}")


@router.post("/", response_model=SuccessResponse)
async def send_alert(
    level: AlertLevel,
    title: str,
    message: str,
    strategy_id: Optional[int] = None,
    current_user: User = Depends(get_current_active_user),
    alert_manager: AlertManager = Depends(get_alert_manager),
):
    """
    Send an alert (you might want to secure this endpoint)
    """
    try:
        user_id = current_user.id
        sent = await alert_manager.send_alert(user_id=user_id, level=level, title=title, message=message, strategy_id=strategy_id)

        if not sent:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Alert rate limited. Please wait before sending another alert.")

        return SuccessResponse(success=True)

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to send alert: {str(e)}")
