from typing import Any, Dict

from core.trade.order.order import Order, OrderSide, OrderType, TimeInForce


class LimitOrder(Order):
    """Limit order - execute only at specified price or better"""

    def __init__(self, symbol: str, side: OrderSide, quantity: float, limit_price: float, time_in_force: TimeInForce = TimeInForce.DAY):
        super().__init__(symbol, side, quantity, OrderType.LIMIT, time_in_force)
        self.limit_price = limit_price

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["limit_price"] = self.limit_price
        return data
