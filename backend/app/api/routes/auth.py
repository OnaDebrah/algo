"""
Authentication routes
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...api.deps import get_current_active_user, login_rate_limit
from ...config import settings
from ...database import get_db
from ...models.user import User
from ...schemas.auth import LoginResponse, User as UserSchema, UserCreate, UserLogin
from ...services.auth_service import AuthService
from ...utils.security import create_access_token, get_password_hash, verify_password

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
