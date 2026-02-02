from typing import Optional

from pydantic import BaseModel


class BacktestSettings(BaseModel):
    data_source: str = "yahoo"  # yahoo, alpha_vantage, polygon
    slippage: float = 0.001  # 0.1%
    commission: float = 0.002  # 0.2%
    initial_capital: float = 100000.0


class GeneralSettings(BaseModel):
    theme: str = "dark"
    notifications: bool = True
    auto_refresh: bool = True
    refresh_interval: int = 30


class UserSettings(BaseModel):
    user_id: Optional[int] = None
    backtest: BacktestSettings = BacktestSettings()
    general: GeneralSettings = GeneralSettings()


# Request model for updating settings
class SettingsUpdate(BaseModel):
    backtest: Optional[BacktestSettings] = None
    general: Optional[GeneralSettings] = None
