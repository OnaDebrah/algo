from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class BrokerType(str, Enum):
    """Supported broker types"""

    PAPER = "paper"
    ALPACA_PAPER = "alpaca_paper"
    ALPACA_LIVE = "alpaca_live"
    IB_PAPER = "ib_paper"
    IB_LIVE = "ib_live"


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
    """Request model for connecting to a broker"""

    broker: BrokerType

    # Alpaca credentials
    api_key: Optional[str] = Field(default=None, description="API key for Alpaca (only for ALPACA_PAPER/ALPACA_LIVE)")
    api_secret: Optional[str] = Field(default=None, description="API secret for Alpaca (only for ALPACA_PAPER/ALPACA_LIVE)")

    # Interactive Brokers credentials
    account_id: Optional[str] = Field(default=None, description="Account ID for Interactive Brokers (only for IB_PAPER/IB_LIVE)")
    host: Optional[str] = Field(default="127.0.0.1", description="Host for Interactive Brokers (default: 127.0.0.1)")
    port: Optional[int] = Field(default=None, description="Port for Interactive Brokers (default: 7497 for paper, 7496 for live)")
    client_id: Optional[int] = Field(default=1, description="Client ID for Interactive Brokers (default: 1)")

    # Paper trading configuration
    initial_capital: Optional[float] = Field(default=100000.0, description="Initial capital for paper trading (only for PAPER)")

    # Generic credentials field for flexibility
    credentials: Optional[Dict[str, Any]] = Field(default=None, description="Additional credentials or overrides")

    class Config:
        schema_extra = {
            "example": {"broker": "alpaca_paper", "api_key": "PK123456789", "api_secret": "secret123"},
            "examples": {
                "alpaca_paper": {
                    "summary": "Connect to Alpaca Paper Trading",
                    "value": {"broker": "alpaca_paper", "api_key": "PK123456789", "api_secret": "secret123"},
                },
                "alpaca_live": {
                    "summary": "Connect to Alpaca Live Trading",
                    "value": {"broker": "alpaca_live", "api_key": "AK123456789", "api_secret": "secret456"},
                },
                "ib_paper": {
                    "summary": "Connect to Interactive Brokers Paper",
                    "value": {"broker": "ib_paper", "account_id": "U1234567", "host": "127.0.0.1", "port": 7497, "client_id": 1},
                },
                "ib_live": {
                    "summary": "Connect to Interactive Brokers Live",
                    "value": {"broker": "ib_live", "account_id": "U1234567", "host": "127.0.0.1", "port": 7496, "client_id": 1},
                },
                "paper": {"summary": "Connect to Paper Trading", "value": {"broker": "paper", "initial_capital": 50000.0}},
            },
        }


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
