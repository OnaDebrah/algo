"""
Strategy schemas
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from backend.app.schemas.live import EquityPoint, TradeResponse


class StrategyParameter(BaseModel):
    name: str
    type: str  # number, string, boolean, select
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    description: str
    options: Optional[List[str]] = None  # For select type


class StrategyInfo(BaseModel):
    key: str
    name: str
    description: str
    category: str
    complexity: Optional[str] = "Intermediate"
    time_horizon: Optional[str] = "Medium-term"
    best_for: List[str] = []
    parameters: List[StrategyParameter]
    backtest_mode: str = "single"  # "single", "multi", or "both"


class Strategy(BaseModel):
    key: str
    name: str
    description: str
    category: str
    complexity: str
    time_horizon: str
    risk_level: str
    parameters: Dict[str, StrategyParameter]
    pros: List[str]
    cons: List[str]


class DeployStrategyRequest(BaseModel):
    """Request to deploy a strategy"""

    source: str = Field(..., description="'backtest', 'marketplace', or 'custom'")
    backtest_id: Optional[int] = None
    marketplace_id: Optional[int] = None

    name: str
    strategy_key: str
    parameters: Dict[str, Any]
    symbols: List[str]

    deployment_mode: str = Field(..., description="'paper' or 'live'")
    initial_capital: float = Field(..., gt=0)
    max_position_pct: float = Field(20.0, ge=1, le=100)
    stop_loss_pct: float = Field(5.0, ge=0, le=50)
    daily_loss_limit: Optional[float] = None

    broker: Optional[str] = None
    notes: Optional[str] = None


class UpdateStrategyRequest(BaseModel):
    """Request to update strategy parameters"""

    name: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    symbols: Optional[List[str]] = None
    notes: Optional[str] = None


class StrategyResponse(BaseModel):
    """Basic strategy response"""

    id: int
    name: str
    strategy_key: str
    symbols: List[str]
    status: str
    deployment_mode: str
    current_equity: float
    total_return_pct: float
    total_trades: int
    deployed_at: Optional[str]

    class Config:
        from_attributes = True


class StrategyDetailsResponse(BaseModel):
    """Full strategy details"""

    strategy: StrategyResponse
    equity_curve: List[EquityPoint]
    trades: List[TradeResponse]
    backtest_comparison: Optional[Dict[str, Any]] = None
