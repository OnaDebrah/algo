"""
Live Trading Broker Interface
Supports Alpaca, Interactive Brokers, and Paper Trading
"""

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types"""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides"""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status"""

    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
