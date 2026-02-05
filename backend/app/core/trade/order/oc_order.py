from typing import Optional

from core.trade.order.order import Order, OrderStatus


class OCOOrder:
    """
    One-Cancels-Other order

    Two orders where if one executes, the other is cancelled
    Example: Take-profit limit + stop-loss stop
    """

    def __init__(self, primary_order: Order, secondary_order: Order):
        self.id: Optional[str] = None
        self.primary_order = primary_order
        self.secondary_order = secondary_order
        self.status = OrderStatus.PENDING
        self.executed_order: Optional[Order] = None

    def check_status(self):
        """Check if either order has filled"""
        if self.primary_order.status == OrderStatus.FILLED:
            self.executed_order = self.primary_order
            self.secondary_order.status = OrderStatus.CANCELLED
            self.status = OrderStatus.FILLED

        elif self.secondary_order.status == OrderStatus.FILLED:
            self.executed_order = self.secondary_order
            self.primary_order.status = OrderStatus.CANCELLED
            self.status = OrderStatus.FILLED
