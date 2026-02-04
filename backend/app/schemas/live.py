from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class BrokerType(str, Enum):
    PAPER = "Paper Trading"
    ALPACA = "Alpaca Markets"
    IBKR = "Interactive Brokers"


class EngineStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class ExecutionOrder(BaseModel):
    id: str
    symbol: str
    side: OrderSide
    qty: float
    type: OrderType
    status: OrderStatus
    price: Optional[float] = None
    time: str


class LiveStatus(BaseModel):
    is_connected: bool
    engine_status: EngineStatus
    active_broker: BrokerType


class ConnectRequest(BaseModel):
    broker: BrokerType
    api_key: Optional[str] = None
    api_secret: Optional[str] = None


class EquityPoint(BaseModel):
    """Equity curve point"""

    timestamp: str
    equity: float
    cash: float
    daily_pnl: float

    class Config:
        orm_mode = True


class TradeResponse(BaseModel):
    """Trade information"""

    id: int
    symbol: str
    side: str
    quantity: float
    entry_price: Optional[float]
    exit_price: Optional[float]
    status: str
    profit: Optional[float]
    profit_pct: Optional[float]
    opened_at: str
    closed_at: Optional[str]

    class Config:
        orm_mode = True


class ControlRequest(BaseModel):
    """Control strategy execution"""

    action: str = Field(..., description="'start', 'pause', 'stop', 'restart'")
