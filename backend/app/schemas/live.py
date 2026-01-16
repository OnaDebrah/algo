from enum import Enum
from typing import List, Optional
from pydantic import BaseModel

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
