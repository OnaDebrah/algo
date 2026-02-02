from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.api.deps import get_current_active_user, get_db
from backend.app.models.user import User
from backend.app.models.user_settings import UserSettings as UserSettingsModel
from backend.app.schemas.settings import BacktestSettings, GeneralSettings, SettingsUpdate, UserSettings

router = APIRouter(prefix="/settings", tags=["Settings"])


def get_or_create_settings(db: Session, user_id: int) -> UserSettingsModel:
    """Get existing settings or create default ones"""
    settings = db.query(UserSettingsModel).filter(UserSettingsModel.user_id == user_id).first()

    if not settings:
        # Create default settings
        settings = UserSettingsModel(user_id=user_id)
        db.add(settings)
        db.commit()
        db.refresh(settings)

    return settings


def model_to_schema(model: UserSettingsModel) -> UserSettings:
    """Convert SQLAlchemy model to Pydantic schema"""
    return UserSettings(
        user_id=model.user_id,
        backtest=BacktestSettings(
            data_source=model.data_source, slippage=model.slippage, commission=model.commission, initial_capital=model.initial_capital
        ),
        general=GeneralSettings(
            theme=model.theme, notifications=model.notifications, auto_refresh=model.auto_refresh, refresh_interval=model.refresh_interval
        ),
    )


@router.get("/", response_model=UserSettings)
async def get_user_settings(current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    """Get current user settings"""
    settings = get_or_create_settings(db, current_user.id)
    return model_to_schema(settings)


@router.put("/", response_model=UserSettings)
async def update_user_settings(settings_update: SettingsUpdate, current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    """Update user settings"""
    settings = get_or_create_settings(db, current_user.id)

    # Update backtest settings
    if settings_update.backtest:
        settings.data_source = settings_update.backtest.data_source
        settings.slippage = settings_update.backtest.slippage
        settings.commission = settings_update.backtest.commission
        settings.initial_capital = settings_update.backtest.initial_capital

    # Update general settings
    if settings_update.general:
        settings.theme = settings_update.general.theme
        settings.notifications = settings_update.general.notifications
        settings.auto_refresh = settings_update.general.auto_refresh
        settings.refresh_interval = settings_update.general.refresh_interval

    db.commit()
    db.refresh(settings)

    return model_to_schema(settings)


@router.post("/reset", response_model=UserSettings)
async def reset_user_settings(current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    """Reset settings to default"""
    # Delete existing settings
    db.query(UserSettingsModel).filter(UserSettingsModel.user_id == current_user.id).delete()
    db.commit()

    # Create new default settings
    settings = get_or_create_settings(db, current_user.id)
    return model_to_schema(settings)
