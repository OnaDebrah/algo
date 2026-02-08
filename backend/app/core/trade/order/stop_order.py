from typing import Any, Dict

from core.trade.order.order import Order, OrderSide, OrderType, TimeInForce


class StopOrder(Order):
    """Stop order - becomes market order when stop price hit"""

    def __init__(self, symbol: str, side: OrderSide, quantity: float, stop_price: float, time_in_force: TimeInForce = TimeInForce.DAY):
        super().__init__(symbol, side, quantity, OrderType.STOP, time_in_force)
        self.stop_price = stop_price
        self.triggered = False

    def check_trigger(self, current_price: float) -> bool:
        """Check if stop price triggered"""
        if self.triggered:
            return True

        if self.side == OrderSide.BUY:
            # Buy stop triggers when price goes above stop
            self.triggered = current_price >= self.stop_price
        else:
            # Sell stop triggers when price goes below stop
            self.triggered = current_price <= self.stop_price

        return self.triggered

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["stop_price"] = self.stop_price
        data["triggered"] = self.triggered
        return data
