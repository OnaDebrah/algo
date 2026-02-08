from typing import Any, Dict

from core.trade.order.order import Order, OrderSide, OrderType, TimeInForce


class StopLimitOrder(Order):
    """Stop-limit order - becomes limit order when stop price hit"""

    def __init__(
        self, symbol: str, side: OrderSide, quantity: float, stop_price: float, limit_price: float, time_in_force: TimeInForce = TimeInForce.DAY
    ):
        super().__init__(symbol, side, quantity, OrderType.STOP_LIMIT, time_in_force)
        self.stop_price = stop_price
        self.limit_price = limit_price
        self.triggered = False

    def check_trigger(self, current_price: float) -> bool:
        """Check if stop price triggered"""
        if self.triggered:
            return True

        if self.side == OrderSide.BUY:
            self.triggered = current_price >= self.stop_price
        else:
            self.triggered = current_price <= self.stop_price

        return self.triggered

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["stop_price"] = self.stop_price
        data["limit_price"] = self.limit_price
        data["triggered"] = self.triggered
        return data
