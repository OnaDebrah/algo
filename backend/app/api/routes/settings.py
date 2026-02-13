"""
Enhanced User Settings API
Includes broker configuration and data source preferences for live trading
"""

import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import get_current_active_user, get_db
from backend.app.config import settings
from backend.app.models.user import User
from backend.app.models.user_settings import UserSettings as UserSettingsModel
from backend.app.schemas.settings import (
    BacktestSettings,
    BrokerConnectionResponse,
    BrokerSettings,
    GeneralSettings,
    LiveTradingSettings,
    SettingsUpdate,
    UserSettings,
)
from backend.app.services.brokers.ib_client import IBClient

router = APIRouter(prefix="/settings", tags=["Settings"])


async def get_or_create_settings(db: AsyncSession, user_id: int) -> UserSettingsModel:
    """Get existing settings or create default ones"""
    stmt = select(UserSettingsModel).where(UserSettingsModel.user_id == user_id)
    result = await db.execute(stmt)
    settings_ = result.scalars().first()

    if not settings_:
        settings_ = UserSettingsModel(
            user_id=user_id,
            # Backtest defaults
            data_source="yahoo",
            slippage=settings.DEFAULT_SLIPPAGE_RATE,
            commission=settings.DEFAULT_COMMISSION_RATE,
            initial_capital=settings.DEFAULT_INITIAL_CAPITAL,
            # Live trading defaults
            live_data_source="alpaca",
            default_broker="paper",
            broker_api_key=None,
            broker_api_secret=None,
            broker_base_url=None,
            # IBKR
            broker_host=None,
            broker_port=None,
            broker_client_id=None,
            auto_connect_broker=False,
            # General defaults
            theme="dark",
            notifications=True,
            auto_refresh=True,
            refresh_interval=30,
        )
        db.add(settings_)
        await db.commit()
        await db.refresh(settings_)

    return settings_


def model_to_schema(model: UserSettingsModel) -> UserSettings:
    """Convert SQLAlchemy model to Pydantic schema"""
    has_broker_config = bool(model.broker_api_key or model.broker_host or model.broker_port)
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
                host=model.broker_host,
                port=model.broker_port,
                client_id=model.broker_client_id,
                user_ib_account_id=model.user_ib_account_id,
                is_configured=bool(model.broker_api_key or model.broker_host),  # Just indicate if configured
            )
            if has_broker_config
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

            if broker.host:
                settings.broker_host = broker.host
            if broker.port:
                settings.broker_port = broker.port
                settings.broker_client_id = int(current_user.id % 32700)
            if broker.user_ib_account_id:
                settings.user_ib_account_id = broker.user_ib_account_id

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
        "host": settings.broker_base_url,
        "port": settings.broker_base_url,
        "client_id": settings.broker_client_id,
        "user_ib_account_id": settings.user_ib_account_id,
    }


@router.delete("/broker/credentials")
async def delete_broker_credentials(current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)):
    """Delete stored broker credentials"""
    settings = await get_or_create_settings(db, current_user.id)

    settings.broker_api_key = None
    settings.broker_api_secret = None
    settings.broker_base_url = None
    settings.broker_host = None
    settings.broker_port = None
    settings.broker_client_id = None
    settings.user_ib_account_id = None

    await db.commit()

    return {"message": "Broker credentials deleted successfully", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.post("/broker/test-connection", response_model=BrokerConnectionResponse)
async def test_broker_connection(current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)):
    """
    Test broker connection with stored credentials

    Returns connection status and account info if successful
    """
    settings_: UserSettingsModel = await get_or_create_settings(db, current_user.id)

    if settings_.default_broker == "alpaca" and not settings_.broker_api_key:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Alpaca requires an API Key. Please update settings.")

    if settings_.default_broker == "ibkr" and not settings_.broker_host:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="IBKR requires a Gateway Host. Please update settings.")

    try:
        if settings_.default_broker == "alpaca":
            from alpaca_trade_api import REST

            api = REST(settings_.broker_api_key, settings_.broker_api_secret, settings_.broker_base_url or settings.ALPACA_PAPER_BASE_URL)
            account = api.get_account()
            return {
                "status": "connected",
                "broker": "alpaca",
                "account_status": account.status,
                "buying_power": float(account.buying_power),
                "equity": float(account.equity),
            }

        if settings_.default_broker == "ibkr":
            ib = IBClient()
            try:
                is_connected = await asyncio.wait_for(ib.connect(settings_, lightweight=True), timeout=15.0)

                await ib.disconnect()

                return {
                    "status": "connected" if is_connected else "failed",
                    "broker": "ibkr",
                    "message": "IBKR connection successful" if is_connected else "Connection failed",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            except asyncio.TimeoutError:
                await ib.disconnect()
                return {
                    "status": "timeout",
                    "broker": "ibkr",
                    "message": "Connection timed out during synchronization",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            except Exception as e:
                await ib.disconnect()
                raise HTTPException(status_code=500, detail=str(e))

        return {
            "status": "not_implemented",
            "broker": settings_.default_broker,
            "message": f"Connection test not implemented for {settings_.default_broker}",
        }

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to connect to broker: {str(e)}")
