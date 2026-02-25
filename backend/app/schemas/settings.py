"""
Settings Schemas
Includes broker configuration and live trading preferences
"""

from typing import Optional

from pydantic import BaseModel, Field

from ..config import settings


class BrokerSettings(BaseModel):
    """Broker configuration"""

    broker_type: str = Field(..., description="Broker type: alpaca, interactive_brokers, paper, etc.")
    api_key: Optional[str] = Field(None, description="API key (masked in responses)")
    api_secret: Optional[str] = Field(None, description="API secret (never sent to frontend)")
    base_url: Optional[str] = Field(None, description="Broker API base URL")
    # IBKR Fields
    host: Optional[str] = Field(settings.IB_HOST, description="IB host")
    port: Optional[int] = Field(settings.IB_PAPER_PORT, description="IB port")
    client_id: Optional[int] = Field(settings.IB_CLIENT_ID, description="IB client id")
    user_ib_account_id: Optional[str] = Field(None, description="User IB Account id")

    is_configured: Optional[bool] = Field(False, description="Whether credentials are configured")


class BacktestSettings(BaseModel):
    """Backtest-specific settings"""

    data_source: str = Field(default="yahoo", description="Data provider: yahoo, alpaca, polygon, etc.")
    slippage: float = Field(default=settings.DEFAULT_SLIPPAGE_RATE, ge=0, le=0.1, description="Slippage as decimal (0.001 = 0.1%)")
    commission: float = Field(default=settings.DEFAULT_COMMISSION_RATE, ge=0, le=0.1, description="Commission as decimal (0.002 = 0.2%)")
    initial_capital: float = Field(default=settings.DEFAULT_INITIAL_CAPITAL, gt=0, description="Default initial capital for backtests")


class LiveTradingSettings(BaseModel):
    """Live trading settings"""

    data_source: str = Field(default="alpaca", description="Real-time data provider")
    default_broker: str = Field(default="paper", description="Default broker for live trading")
    auto_connect: bool = Field(default=False, description="Auto-connect to broker on app start")
    broker: Optional[BrokerSettings] = Field(None, description="Broker credentials and configuration")


class GeneralSettings(BaseModel):
    """General application settings"""

    theme: str = Field(default="dark", description="UI theme: dark, light")
    notifications: bool = Field(default=True, description="Enable notifications")
    auto_refresh: bool = Field(default=True, description="Auto-refresh data")
    refresh_interval: int = Field(default=30, ge=5, le=300, description="Refresh interval in seconds")


class UserSettings(BaseModel):
    """Complete user settings"""

    user_id: int
    backtest: BacktestSettings
    live_trading: Optional[LiveTradingSettings] = None
    general: GeneralSettings


class SettingsUpdate(BaseModel):
    """Update request for user settings"""

    backtest: Optional[BacktestSettings] = None
    live_trading: Optional[LiveTradingSettings] = None
    general: Optional[GeneralSettings] = None


class BrokerConnectionTest(BaseModel):
    """Test broker connection"""

    broker_type: Optional[str] = None
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    base_url: Optional[str] = None


class BrokerConnectionResponse(BaseModel):
    """Broker connection test response"""

    status: str = Field(..., description="connected, failed, not_configured")
    broker: str
    message: Optional[str] = None
    account_status: Optional[str] = None
    buying_power: Optional[float] = None
    equity: Optional[float] = None
    timestamp: str
