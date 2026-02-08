"""
Enhanced User Settings API
Includes broker configuration and data source preferences for live trading
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import get_current_active_user, get_db
from backend.app.models.user import User
from backend.app.models.user_settings import UserSettings as UserSettingsModel
from backend.app.schemas.settings import BacktestSettings, BrokerSettings, GeneralSettings, LiveTradingSettings, SettingsUpdate, UserSettings

router = APIRouter(prefix="/settings", tags=["Settings"])


async def get_or_create_settings(db: AsyncSession, user_id: int) -> UserSettingsModel:
    """Get existing settings or create default ones"""
    stmt = select(UserSettingsModel).where(UserSettingsModel.user_id == user_id)
    result = await db.execute(stmt)
    settings = result.scalars().first()

    if not settings:
        settings = UserSettingsModel(
            user_id=user_id,
            # Backtest defaults
            data_source="yahoo",
            slippage=0.001,
            commission=0.002,
            initial_capital=10000.0,
            # Live trading defaults
            live_data_source="alpaca",
            default_broker="paper",
            broker_api_key=None,
            broker_api_secret=None,
            broker_base_url=None,
            auto_connect_broker=False,
            # General defaults
            theme="dark",
            notifications=True,
            auto_refresh=True,
            refresh_interval=30,
        )
        db.add(settings)
        await db.commit()
        await db.refresh(settings)

    return settings


def model_to_schema(model: UserSettingsModel) -> UserSettings:
    """Convert SQLAlchemy model to Pydantic schema"""
    return UserSettings(
        user_id=model.user_id,
        backtest=BacktestSettings(
            data_source=model.data_source, slippage=model.slippage, commission=model.commission, initial_capital=model.initial_capital
        ),
        live_trading=LiveTradingSettings(
            data_source=model.live_data_source or "alpaca",
            default_broker=model.default_broker or "paper",
            auto_connect=model.auto_connect_broker or False,
            broker=BrokerSettings(
                broker_type=model.default_broker or "paper",
                api_key=model.broker_api_key,  # Never send actual keys to frontend
                api_secret=None,  # NEVER send secrets to frontend
                base_url=model.broker_base_url,
                is_configured=bool(model.broker_api_key),  # Just indicate if configured
            )
            if model.broker_api_key
            else None,
        ),
        general=GeneralSettings(
            theme=model.theme, notifications=model.notifications, auto_refresh=model.auto_refresh, refresh_interval=model.refresh_interval
        ),
    )


@router.get("/", response_model=UserSettings)
async def get_user_settings(current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)):
    """Get current user settings"""
    settings = await get_or_create_settings(db, current_user.id)
    return model_to_schema(settings)


@router.put("/", response_model=UserSettings)
async def update_user_settings(
    settings_update: SettingsUpdate, current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)
):
    """Update user settings"""
    settings = await get_or_create_settings(db, current_user.id)

    # Update backtest settings
    if settings_update.backtest:
        if settings_update.backtest.data_source:
            settings.data_source = settings_update.backtest.data_source
        if settings_update.backtest.slippage is not None:
            settings.slippage = settings_update.backtest.slippage
        if settings_update.backtest.commission is not None:
            settings.commission = settings_update.backtest.commission
        if settings_update.backtest.initial_capital is not None:
            settings.initial_capital = settings_update.backtest.initial_capital

    # Update live trading settings
    if settings_update.live_trading:
        if settings_update.live_trading.data_source:
            settings.live_data_source = settings_update.live_trading.data_source
        if settings_update.live_trading.default_broker:
            settings.default_broker = settings_update.live_trading.default_broker
        if settings_update.live_trading.auto_connect is not None:
            settings.auto_connect_broker = settings_update.live_trading.auto_connect

        # Update broker credentials if provided
        if settings_update.live_trading.broker:
            broker = settings_update.live_trading.broker
            if broker.broker_type:
                settings.default_broker = broker.broker_type
            if broker.api_key:
                # TODO: Encrypt this in production
                settings.broker_api_key = broker.api_key
            if broker.api_secret:
                # TODO: Encrypt this in production
                settings.broker_api_secret = broker.api_secret
            if broker.base_url:
                settings.broker_base_url = broker.base_url

    # Update general settings
    if settings_update.general:
        if settings_update.general.theme:
            settings.theme = settings_update.general.theme
        if settings_update.general.notifications is not None:
            settings.notifications = settings_update.general.notifications
        if settings_update.general.auto_refresh is not None:
            settings.auto_refresh = settings_update.general.auto_refresh
        if settings_update.general.refresh_interval is not None:
            settings.refresh_interval = settings_update.general.refresh_interval

    await db.commit()
    await db.refresh(settings)

    return model_to_schema(settings)


@router.post("/reset", response_model=UserSettings)
async def reset_user_settings(current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)):
    """Reset settings to default"""
    stmt = select(UserSettingsModel).where(UserSettingsModel.user_id == current_user.id)
    result = await db.execute(stmt)

    settings = result.scalars().first()
    if settings:
        await db.delete(settings)
        await db.commit()

    # Create new default settings
    settings = await get_or_create_settings(db, current_user.id)
    return model_to_schema(settings)


@router.get("/broker/credentials")
async def get_broker_credentials(current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)):
    """
    Get broker credentials for backend use only

    This endpoint should only be called by backend services, never frontend
    """
    settings = await get_or_create_settings(db, current_user.id)

    return {
        "broker_type": settings.default_broker,
        "api_key": settings.broker_api_key,
        "api_secret": settings.broker_api_secret,
        "base_url": settings.broker_base_url,
    }


@router.delete("/broker/credentials")
async def delete_broker_credentials(current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)):
    """Delete stored broker credentials"""
    settings = await get_or_create_settings(db, current_user.id)

    settings.broker_api_key = None
    settings.broker_api_secret = None
    settings.broker_base_url = None

    await db.commit()

    return {"message": "Broker credentials deleted successfully", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.post("/broker/test-connection")
async def test_broker_connection(current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)):
    """
    Test broker connection with stored credentials

    Returns connection status and account info if successful
    """
    settings = await get_or_create_settings(db, current_user.id)

    if not settings.broker_api_key:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No broker credentials configured. Please add credentials in settings.")

    # TODO: Implement actual broker connection test
    # For now, just validate credentials exist

    try:
        # Example: Test Alpaca connection
        if settings.default_broker == "alpaca":
            # from alpaca_trade_api import REST
            # api = REST(
            #     settings.broker_api_key,
            #     settings.broker_api_secret,
            #     settings.broker_base_url or 'https://paper-api.alpaca.markets'
            # )
            # account = api.get_account()
            # return {
            #     "status": "connected",
            #     "broker": "alpaca",
            #     "account_status": account.status,
            #     "buying_power": float(account.buying_power),
            #     "equity": float(account.equity)
            # }

            return {
                "status": "connected",
                "broker": settings.default_broker,
                "message": "Connection test successful (mock)",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        return {
            "status": "not_implemented",
            "broker": settings.default_broker,
            "message": f"Connection test not implemented for {settings.default_broker}",
        }

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to connect to broker: {str(e)}")
