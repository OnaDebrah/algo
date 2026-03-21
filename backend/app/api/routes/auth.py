"""
Authentication routes
"""

import asyncio
import base64
import io
import json
import logging
import secrets
from datetime import datetime, timedelta, timezone

import pyotp
import qrcode
from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import JSONResponse
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...api.deps import get_current_active_user, login_rate_limit
from ...config import settings
from ...database import get_db
from ...models.user import User
from ...schemas.auth import (
    ForgotPasswordRequest,
    LoginResponse,
    ResetPasswordRequest,
    TwoFactorBackupCodesResponse,
    TwoFactorSetupResponse,
    TwoFactorVerifyLoginRequest,
    TwoFactorVerifyRequest,
    User as UserSchema,
    UserCreate,
    UserLogin,
)
from ...services.auth_service import AuthService
from ...utils.security import create_access_token, get_password_hash, verify_password

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Cookie settings — httpOnly prevents JavaScript access (XSS protection)
_COOKIE_SECURE = settings.ENVIRONMENT not in ("development", "test")
_COOKIE_MAX_AGE = settings.JWT_EXPIRATION_MINUTES * 60  # seconds


def _set_auth_cookies(response: Response, access_token: str, refresh_token: str) -> None:
    """Set httpOnly auth cookies on a response."""
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=_COOKIE_SECURE,
        samesite="lax",
        max_age=_COOKIE_MAX_AGE,
        path="/",
    )
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=_COOKIE_SECURE,
        samesite="lax",
        max_age=_COOKIE_MAX_AGE,
        path="/",
    )


def _clear_auth_cookies(response: Response) -> None:
    """Remove auth cookies from a response."""
    response.delete_cookie("access_token", path="/")
    response.delete_cookie("refresh_token", path="/")


@router.post("/register")
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    """Register new user"""

    result = await db.execute(select(User).where((User.username == user_data.username) | (User.email == user_data.email)))
    existing_user = result.scalar_one_or_none()

    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username or email already registered")

    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
        tier="FREE",
        country=user_data.country,
        investor_type=user_data.investor_type,
        risk_profile=user_data.risk_profile,
    )

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    access_token = create_access_token(data={"sub": str(new_user.id)})
    refresh_token = create_access_token(data={"sub": str(new_user.id), "refresh": True})

    await AuthService.track_usage(db, new_user.id, "register", {"tier": new_user.tier})

    # Fire welcome email in background (non-blocking)
    async def _send_welcome():
        try:
            from ...services.email_service import WelcomeEmailService

            welcome_svc = WelcomeEmailService()
            await welcome_svc.send_welcome_email(new_user, db)
        except Exception as e:
            logger.warning(f"Welcome email failed (non-critical): {e}")

    asyncio.create_task(_send_welcome())

    # Build JSON body (tokens still included for backward compat / API clients)
    body = LoginResponse(
        user=UserSchema.model_validate(new_user),
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
    )
    response = JSONResponse(content=body.model_dump(mode="json"))
    _set_auth_cookies(response, access_token, refresh_token)
    return response


@router.post("/login", dependencies=[Depends(login_rate_limit)])
async def login(credentials: UserLogin, db: AsyncSession = Depends(get_db)):
    """Login user (rate-limited: 5 attempts / minute per IP)"""
    # Allow login with either email or username
    result = await db.execute(select(User).where((User.email == credentials.email) | (User.username == credentials.email)))
    user = result.scalar_one_or_none()

    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")

    user.last_login = datetime.now(timezone.utc)
    await db.commit()

    # Check if 2FA is enabled
    if user.totp_enabled:
        # Issue a short-lived 2FA-pending token (5 minutes)
        pending_token = create_access_token(
            data={"sub": str(user.id), "2fa_pending": True},
            expires_delta=timedelta(minutes=5),
        )
        return JSONResponse(
            content={
                "requires_2fa": True,
                "pending_2fa_token": pending_token,
                "user": None,
                "access_token": "",
                "refresh_token": "",
                "token_type": "bearer",
            }
        )

    access_token = create_access_token(data={"sub": str(user.id)})
    refresh_token = create_access_token(data={"sub": str(user.id), "refresh": True})

    await AuthService.track_usage(db, user.id, "login")

    # Build JSON body (tokens still included for backward compat / API clients)
    body = LoginResponse(
        user=UserSchema.model_validate(user),
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
    )
    response = JSONResponse(content=body.model_dump(mode="json"))
    _set_auth_cookies(response, access_token, refresh_token)
    return response


@router.post("/logout")
async def logout():
    """Logout user — clears httpOnly auth cookies."""
    response = JSONResponse(content={"message": "Successfully logged out"})
    _clear_auth_cookies(response)
    return response


@router.get("/me", response_model=UserSchema)
async def get_me(current_user: User = Depends(get_current_active_user)):
    """Get current user info"""
    return UserSchema.model_validate(current_user)


@router.post("/forgot-password")
async def forgot_password(request: ForgotPasswordRequest, db: AsyncSession = Depends(get_db)):
    """Request password reset — always returns 200 to avoid email enumeration."""
    result = await db.execute(select(User).where(User.email == request.email))
    user = result.scalar_one_or_none()

    if user:
        token = secrets.token_urlsafe(32)
        user.password_reset_token = token
        user.password_reset_expires = datetime.now(timezone.utc) + timedelta(hours=1)
        await db.commit()

        # Send reset email in background
        async def _send_reset():
            try:
                from ...alerts.email_provider import EmailProvider

                if not settings.EMAIL_ENABLED:
                    logger.info(f"📧 [EMAIL_DISABLED] Password reset token for {user.email}: {token}")
                    return

                frontend_url = settings.FRONTEND_URL if hasattr(settings, "FRONTEND_URL") else "http://localhost:3000"
                reset_link = f"{frontend_url}?reset_token={token}"

                email_provider = EmailProvider(
                    smtp_host=settings.SMTP_SERVER,
                    smtp_port=settings.SMTP_PORT,
                    username=settings.SMTP_USERNAME,
                    password=settings.SMTP_PASSWORD,
                    from_email=settings.FROM_EMAIL,
                    from_name="Oraculum Platform",
                )

                body = f"""
                <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 600px; margin: 0 auto; padding: 40px 20px;">
                    <div style="background: linear-gradient(135deg, #7c3aed, #c026d3); padding: 30px; border-radius: 16px 16px 0 0; text-align: center;">
                        <h1 style="color: white; margin: 0; font-size: 28px;">ORACULUM</h1>
                        <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0;">AI-Powered Trading Platform</p>
                    </div>
                    <div style="background: #1e1b4b; padding: 30px; border-radius: 0 0 16px 16px; color: #e2e8f0;">
                        <h2 style="color: #c4b5fd; margin-top: 0;">Password Reset Request</h2>
                        <p>Hi {user.username},</p>
                        <p>We received a request to reset your password. Click the button below to set a new password:</p>
                        <div style="text-align: center; margin: 30px 0;">
                            <a href="{reset_link}" style="background: linear-gradient(135deg, #7c3aed, #c026d3); color: white; padding: 14px 32px; border-radius: 10px; text-decoration: none; font-weight: 600; display: inline-block;">Reset Password</a>
                        </div>
                        <p style="color: #94a3b8; font-size: 14px;">This link expires in 1 hour. If you did not request this, you can safely ignore this email.</p>
                    </div>
                </div>
                """

                await email_provider.send_email(
                    to_email=user.email,
                    subject="Password Reset — Oraculum",
                    body=body,
                    html=True,
                )
            except Exception as e:
                logger.warning(f"Failed to send reset email: {e}")

        asyncio.create_task(_send_reset())

    return {"message": "If an account exists with that email, a password reset link has been sent."}


@router.post("/reset-password")
async def reset_password(request: ResetPasswordRequest, db: AsyncSession = Depends(get_db)):
    """Reset password using a valid reset token."""
    result = await db.execute(select(User).where(User.password_reset_token == request.token))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    if not user.password_reset_expires or user.password_reset_expires < datetime.now(timezone.utc):
        # Clear expired token
        user.password_reset_token = None
        user.password_reset_expires = None
        await db.commit()
        raise HTTPException(status_code=400, detail="Reset token has expired. Please request a new one.")

    # Update password
    user.hashed_password = get_password_hash(request.new_password)
    user.password_reset_token = None
    user.password_reset_expires = None
    await db.commit()

    logger.info(f"Password reset successful for user: {user.email}")
    return {"message": "Password has been reset successfully. You can now sign in with your new password."}


# ---------------------------------------------------------------------------
# Two-Factor Authentication (2FA) endpoints
# ---------------------------------------------------------------------------


@router.post("/2fa/setup", response_model=TwoFactorSetupResponse)
async def setup_2fa(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate a TOTP secret and QR code for 2FA setup."""
    secret = pyotp.random_base32()
    totp = pyotp.totp.TOTP(secret)
    provisioning_uri = totp.provisioning_uri(current_user.email, issuer_name="Oraculum")

    # Generate QR code as base64 PNG
    qr = qrcode.make(provisioning_uri)
    buf = io.BytesIO()
    qr.save(buf, format="PNG")
    qr_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Store secret but don't enable yet (user must verify first)
    current_user.totp_secret = secret
    await db.commit()

    return TwoFactorSetupResponse(
        secret=secret,
        qr_uri=provisioning_uri,
        qr_image_base64=qr_base64,
    )


@router.post("/2fa/verify-setup", response_model=TwoFactorBackupCodesResponse)
async def verify_2fa_setup(
    body: TwoFactorVerifyRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Verify the TOTP code to complete 2FA setup and return backup codes."""
    if not current_user.totp_secret:
        raise HTTPException(status_code=400, detail="2FA setup not initiated. Call /auth/2fa/setup first.")

    totp = pyotp.TOTP(current_user.totp_secret)
    if not totp.verify(body.code, valid_window=1):
        raise HTTPException(status_code=400, detail="Invalid verification code.")

    # Enable 2FA
    current_user.totp_enabled = True

    # Generate 10 backup codes
    plaintext_codes = [secrets.token_hex(4) for _ in range(10)]
    hashed_codes = [get_password_hash(code) for code in plaintext_codes]
    current_user.totp_backup_codes = json.dumps(hashed_codes)

    await db.commit()

    return TwoFactorBackupCodesResponse(backup_codes=plaintext_codes)


@router.post("/2fa/verify", dependencies=[Depends(login_rate_limit)])
async def verify_2fa_login(
    body: TwoFactorVerifyLoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """Verify a TOTP or backup code during login (no auth required)."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid 2FA code or token.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Decode the pending 2FA token
    try:
        payload = jwt.decode(
            body.pending_2fa_token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        user_id: str = payload.get("sub")
        is_pending = payload.get("2fa_pending")
        if user_id is None or not is_pending:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # Fetch user
    result = await db.execute(select(User).where(User.id == int(user_id)))
    user = result.scalar_one_or_none()
    if not user or not user.totp_enabled or not user.totp_secret:
        raise credentials_exception

    # Try TOTP verification first
    totp = pyotp.TOTP(user.totp_secret)
    verified = totp.verify(body.code, valid_window=1)

    # If TOTP fails, try backup codes
    if not verified and user.totp_backup_codes:
        stored_hashes = json.loads(user.totp_backup_codes)
        for i, stored_hash in enumerate(stored_hashes):
            if verify_password(body.code, stored_hash):
                # Remove used backup code
                stored_hashes.pop(i)
                user.totp_backup_codes = json.dumps(stored_hashes)
                verified = True
                break

    if not verified:
        raise credentials_exception

    # Issue full JWT tokens (same as normal login)
    access_token = create_access_token(data={"sub": str(user.id)})
    refresh_token = create_access_token(data={"sub": str(user.id), "refresh": True})

    await AuthService.track_usage(db, user.id, "login")

    login_body = LoginResponse(
        user=UserSchema.model_validate(user),
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
    )
    response = JSONResponse(content=login_body.model_dump(mode="json"))
    _set_auth_cookies(response, access_token, refresh_token)
    return response


@router.post("/2fa/disable")
async def disable_2fa(
    body: TwoFactorVerifyRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Disable 2FA by verifying a current TOTP code."""
    if not current_user.totp_enabled or not current_user.totp_secret:
        raise HTTPException(status_code=400, detail="2FA is not enabled.")

    totp = pyotp.TOTP(current_user.totp_secret)
    if not totp.verify(body.code, valid_window=1):
        raise HTTPException(status_code=400, detail="Invalid verification code.")

    current_user.totp_secret = None
    current_user.totp_enabled = False
    current_user.totp_backup_codes = None
    await db.commit()

    return {"message": "Two-factor authentication has been disabled."}
