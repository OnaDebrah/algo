"""
Portfolio schemas
"""

from typing import List, Optional

from pydantic import BaseModel


class PortfolioCreate(BaseModel):
    name: str
    description: Optional[str] = None
    initial_capital: float


class PortfolioUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None


class Trade(BaseModel):
    id: Optional[int] = None
    portfolio_id: Optional[int] = None
    symbol: str
    order_type: str
    quantity: float
    price: float
    commission: float = 0
    total_value: float
    strategy: Optional[str] = None
    notes: Optional[str] = None
    executed_at: Optional[str] = None
    profit: Optional[float] = None
    profit_pct: Optional[float] = None


class Position(BaseModel):
    id: int
    portfolio_id: int
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    market_value: float
    created_at: str
    updated_at: Optional[str] = None

    class Config:
        from_attributes = True


class Portfolio(BaseModel):
    id: int
    user_id: int
    name: str
    description: Optional[str] = None
    initial_capital: float
    current_capital: float
    is_active: bool
    created_at: str
    updated_at: Optional[str] = None
    positions: Optional[List[Position]] = []
    recent_trades: Optional[List[Trade]] = []


class PortfolioMetrics(BaseModel):
    nav: float
    prev_nav: float
    exposure: float
    unrealized_pnl: float
    cash: float
    total_value: float
    daily_return: float
    daily_return_pct: float
