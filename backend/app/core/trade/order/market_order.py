import logging

from core.trade.order.order import Order, OrderSide, OrderType, TimeInForce

logger = logging.getLogger(__name__)


class MarketOrder(Order):
    """Market order - execute at current market price"""

    def __init__(self, symbol: str, side: OrderSide, quantity: float, time_in_force: TimeInForce = TimeInForce.DAY):
        super().__init__(symbol, side, quantity, OrderType.MARKET, time_in_force)
