"""
Advanced Order Types
Limit orders, stop-loss orders, trailing stops, OCO, etc.
"""

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class OrderType(str, Enum):
    """Order types"""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    OCO = "oco"  # One-Cancels-Other


class OrderSide(str, Enum):
    """Order side"""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """Order status"""

    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(str, Enum):
    """Time in force"""

    DAY = "day"  # Good for day
    GTC = "gtc"  # Good til cancelled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill


class Order:
    """
    Base order class
    # Example 1: Stop-Loss with Take-Profit (OCO)

    # Buy 100 shares at market
    market_order = MarketOrder('AAPL', OrderSide.BUY, 100)
    await order_mgr.submit_order(market_order)

    # Set stop-loss at -5% and take-profit at +10%
    entry_price = 150.00

    stop_loss = StopOrder(
        symbol='AAPL',
        side=OrderSide.SELL,
        quantity=100,
        stop_price=entry_price * 0.95  # -5%
    )

    take_profit = LimitOrder(
        symbol='AAPL',
        side=OrderSide.SELL,
        quantity=100,
        limit_price=entry_price * 1.10  # +10%
    )

    # Submit as OCO (one executes, other cancels)
    oco_id = await order_mgr.submit_oco_order(take_profit, stop_loss)


    # Example 2: Trailing Stop for Trend Following

    trailing_stop = TrailingStopOrder(
        symbol='AAPL',
        side=OrderSide.SELL,
        quantity=100,
        trail_percent=5.0  # Trail 5% below high
    )

    await order_mgr.submit_order(trailing_stop)


    # Example 3: Buy on Dip with Limit Order

    limit_buy = LimitOrder(
        symbol='AAPL',
        side=OrderSide.BUY,
        quantity=50,
        limit_price=145.00,  # Only buy if price drops to $145
        time_in_force=TimeInForce.GTC
    )

    await order_mgr.submit_order(limit_buy)
    """

    def __init__(self, symbol: str, side: OrderSide, quantity: float, order_type: OrderType, time_in_force: TimeInForce = TimeInForce.DAY):
        self.id: Optional[str] = None
        self.broker_order_id: Optional[str] = None

        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.order_type = order_type
        self.time_in_force = time_in_force

        self.status = OrderStatus.PENDING
        self.filled_quantity = 0.0
        self.average_fill_price: Optional[float] = None

        self.created_at = datetime.now(timezone.utc)
        self.submitted_at: Optional[datetime] = None
        self.filled_at: Optional[datetime] = None
        self.cancelled_at: Optional[datetime] = None

        self.error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "broker_order_id": self.broker_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "time_in_force": self.time_in_force.value,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "average_fill_price": self.average_fill_price,
            "created_at": self.created_at.isoformat(),
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "error_message": self.error_message,
        }
