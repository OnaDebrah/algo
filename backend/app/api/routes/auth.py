"""
Authentication routes
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import get_current_active_user
from backend.app.database import get_db
from backend.app.models.user import User
from backend.app.schemas.auth import LoginResponse, User as UserSchema, UserCreate, UserLogin
from backend.app.services.auth_service import AuthService
from backend.app.utils.security import create_access_token, get_password_hash, verify_password

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=LoginResponse)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    """Register new user"""
    # Check if user exists
    result = await db.execute(select(User).where((User.username == user_data.username) | (User.email == user_data.email)))
    existing_user = result.scalar_one_or_none()

    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username or email already registered")

    hashed_password = get_password_hash(user_data.password)
    new_user = User(username=user_data.username, email=user_data.email, hashed_password=hashed_password, tier="FREE")

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    access_token = create_access_token(data={"sub": str(new_user.id)})
    refresh_token = create_access_token(data={"sub": str(new_user.id), "refresh": True})

    await AuthService.track_usage(db, new_user.id, "register", {"tier": new_user.tier})

    return LoginResponse(user=UserSchema.from_orm(new_user), access_token=access_token, refresh_token=refresh_token, token_type="bearer")


@router.post("/login", response_model=LoginResponse)
async def login(credentials: UserLogin, db: AsyncSession = Depends(get_db)):
    """Login user"""
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

    # Update last login
    user.last_login = datetime.now(timezone.utc)
    await db.commit()

    access_token = create_access_token(data={"sub": str(user.id)})
    refresh_token = create_access_token(data={"sub": str(user.id), "refresh": True})

    await AuthService.track_usage(db, user.id, "login")

    return LoginResponse(user=UserSchema.model_validate(user), access_token=access_token, refresh_token=refresh_token, token_type="bearer")


@router.post("/logout")
async def logout():
    """Logout user (client should delete token)"""
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=UserSchema)
async def get_me(current_user: User = Depends(get_current_active_user)):
    """Get current user info"""
    return UserSchema.model_validate(current_user)
