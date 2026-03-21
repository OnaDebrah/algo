"""
Paper trading Pydantic schemas.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Request schemas ──────────────────────────────────────────────────


class PaperPortfolioCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    initial_cash: float = Field(100000.0, ge=100, le=100_000_000)
    strategy_key: Optional[str] = None
    strategy_params: Optional[dict] = None
    strategy_symbol: Optional[str] = None
    trade_quantity: float = Field(100, gt=0)
    data_interval: Optional[str] = "1d"


class PaperAttachStrategy(BaseModel):
    strategy_key: str = Field(..., min_length=1)
    strategy_symbol: str = Field(..., min_length=1, max_length=20)
    strategy_params: Optional[dict] = None
    trade_quantity: float = Field(100, gt=0)
    data_interval: Optional[str] = "1d"


class PaperTradeRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    side: str = Field(..., pattern=r"^(buy|sell)$")
    quantity: float = Field(..., gt=0)


# ── Response schemas ─────────────────────────────────────────────────


class PaperPositionOut(BaseModel):
    id: int
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    market_value: Optional[float] = None
    opened_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class PaperTradeOut(BaseModel):
    id: int
    symbol: str
    side: str
    quantity: float
    price: float
    slippage: float
    total_cost: float
    realized_pnl: Optional[float] = None
    source: str = "manual"
    executed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class PaperPortfolioOut(BaseModel):
    id: int
    name: str
    initial_cash: float
    current_cash: float
    is_active: bool
    strategy_key: Optional[str] = None
    strategy_symbol: Optional[str] = None
    strategy_params: Optional[dict] = None
    trade_quantity: Optional[float] = None
    data_interval: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    positions: list[PaperPositionOut] = []
    equity: Optional[float] = None
    total_return_pct: Optional[float] = None

    model_config = {"from_attributes": True}


class StrategySignalResult(BaseModel):
    signal: int  # 1=buy, -1=sell, 0=hold
    signal_label: str  # "BUY", "SELL", "HOLD"
    strategy_key: str
    symbol: str
    current_price: float
    trade_executed: bool = False
    trade_detail: Optional[str] = None
    data_interval: Optional[str] = None  # interval used for signal generation
    market_open: bool = True  # whether US market is currently open
    data_as_of: Optional[str] = None  # timestamp of latest price bar used


class PaperEquitySnapshotOut(BaseModel):
    equity: float
    cash: float
    positions_value: float
    timestamp: Optional[datetime] = None

    model_config = {"from_attributes": True}


class PaperPerformanceOut(BaseModel):
    total_return: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    sharpe_ratio: Optional[float] = None
