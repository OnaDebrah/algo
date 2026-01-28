"""
User settings and preferences routes
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.api.deps import get_current_active_user
from backend.app.models.user import User
from backend.app.schemas.user import UserPreferences, UserResponse, UserUpdate
from backend.app.database import get_db

router = APIRouter(prefix="/user", tags=["User"])


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """Get current user information"""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        is_superuser=current_user.is_superuser,
        created_at=current_user.created_at.isoformat() if current_user.created_at else None,
    )


@router.put("/me", response_model=UserResponse)
async def update_current_user(user_update: UserUpdate, current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    """Update current user information"""
    update_data = user_update.dict(exclude_unset=True)

    for field, value in update_data.items():
        if field == "password" and value:
            from backend.app.utils.security import get_password_hash

            setattr(current_user, "hashed_password", get_password_hash(value))
        elif hasattr(current_user, field):
            setattr(current_user, field, value)

    db.commit()
    db.refresh(current_user)

    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        is_superuser=current_user.is_superuser,
        created_at=current_user.created_at.isoformat() if current_user.created_at else None,
    )


@router.get("/preferences", response_model=UserPreferences)
async def get_user_preferences(current_user: User = Depends(get_current_active_user)):
    """Get user preferences"""
    # This would be stored in a preferences table or JSON field
    # For now, return defaults
    return UserPreferences(theme="dark", default_capital=10000, default_commission=0.001, risk_tolerance="medium", notifications_enabled=True)


@router.put("/preferences", response_model=UserPreferences)
async def update_user_preferences(preferences: UserPreferences, current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    """Update user preferences"""
    # Store preferences (would need a preferences table/field)
    # For now, just return the input
    return preferences
