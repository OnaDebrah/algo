import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.trade.order.limit_order import LimitOrder
from core.trade.order.oc_order import OCOOrder
from core.trade.order.order import Order, OrderSide, OrderStatus, OrderType
from core.trade.order.stop_limit_order import StopLimitOrder
from core.trade.order.stop_order import StopOrder
from core.trade.order.trailing_stop_order import TrailingStopOrder

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Manages advanced order types

    Features:
    - Order validation
    - Stop order monitoring
    - Trailing stop updates
    - OCO order management
    """

    def __init__(self, broker_client: Any):
        self.broker = broker_client

        # Active orders
        self.pending_orders: Dict[str, Order] = {}
        self.oco_orders: Dict[str, OCOOrder] = {}

        # Order counter
        self.order_counter = 0

    async def submit_order(self, order: Order) -> str:
        """
        Submit an order

        Returns:
            order_id: Unique order ID
        """
        # Assign ID
        self.order_counter += 1
        order.id = f"ORD_{self.order_counter:08d}"

        # Validate order
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            raise ValueError(f"Invalid order: {order.error_message}")

        # Handle different order types
        if order.order_type == OrderType.MARKET:
            # Execute immediately
            await self._execute_market_order(order)

        elif order.order_type == OrderType.LIMIT:
            # Add to pending orders
            self.pending_orders[order.id] = order
            order.status = OrderStatus.SUBMITTED

        elif order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            # Monitor for trigger
            self.pending_orders[order.id] = order
            order.status = OrderStatus.SUBMITTED

        elif order.order_type == OrderType.TRAILING_STOP:
            # Start trailing
            self.pending_orders[order.id] = order
            order.status = OrderStatus.SUBMITTED

        order.submitted_at = datetime.now(timezone.utc)

        logger.info(f"Order submitted: {order.id} - {order.order_type.value} {order.side.value} {order.quantity} {order.symbol}")

        return order.id

    async def submit_oco_order(self, primary_order: Order, secondary_order: Order) -> str:
        """Submit OCO (One-Cancels-Other) order"""

        primary_id = await self.submit_order(primary_order)
        secondary_id = await self.submit_order(secondary_order)

        oco = OCOOrder(primary_order, secondary_order)
        self.order_counter += 1
        oco.id = f"OCO_{self.order_counter:08d}"

        self.oco_orders[oco.id] = oco

        logger.info(f"OCO order with primary id: {primary_id} and secondary id: {secondary_id}submitted: {oco.id}")

        return oco.id

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if order_id not in self.pending_orders:
            logger.warning(f"Order {order_id} not found")
            return False

        order = self.pending_orders[order_id]

        # Cancel with broker if needed
        if order.broker_order_id:
            await self.broker.cancel_order(order.broker_order_id)

        # Update status
        order.status = OrderStatus.CANCELLED
        order.cancelled_at = datetime.now(timezone.utc)

        # Remove from pending
        del self.pending_orders[order_id]

        logger.info(f"Order cancelled: {order_id}")

        return True

    async def update_orders(self, current_price: float, symbol: str):
        """
        Update all orders based on current price

        Call this every time you get a price update
        """
        for order_id, order in list(self.pending_orders.items()):
            if order.symbol != symbol:
                continue

            try:
                if order.order_type == OrderType.LIMIT:
                    await self._check_limit_order(order, current_price)

                elif order.order_type == OrderType.STOP:
                    await self._check_stop_order(order, current_price)

                elif order.order_type == OrderType.STOP_LIMIT:
                    await self._check_stop_limit_order(order, current_price)

                elif order.order_type == OrderType.TRAILING_STOP:
                    await self._check_trailing_stop(order, current_price)

            except Exception as e:
                logger.error(f"Error updating order {order_id}: {e}")

        # Check OCO orders
        for oco_id, oco in list(self.oco_orders.items()):
            oco.check_status()
            if oco.status == OrderStatus.FILLED:
                logger.info(f"OCO order completed: {oco_id}")
                del self.oco_orders[oco_id]

    async def _check_limit_order(self, order: LimitOrder, current_price: float):
        """Check if limit order can execute"""
        can_execute = False

        if order.side == OrderSide.BUY:
            # Buy limit executes when price at or below limit
            can_execute = current_price <= order.limit_price
        else:
            # Sell limit executes when price at or above limit
            can_execute = current_price >= order.limit_price

        if can_execute:
            await self._execute_limit_order(order, current_price)

    async def _check_stop_order(self, order: StopOrder, current_price: float):
        """Check if stop order triggered"""
        if order.check_trigger(current_price):
            # Convert to market order
            await self._execute_market_order(order)

    async def _check_stop_limit_order(self, order: StopLimitOrder, current_price: float):
        """Check if stop-limit order triggered"""
        if order.check_trigger(current_price):
            # Convert to limit order
            order.order_type = OrderType.LIMIT
            logger.info(f"Stop-limit triggered: {order.id} at ${current_price:.2f}")

    async def _check_trailing_stop(self, order: TrailingStopOrder, current_price: float):
        """Update trailing stop"""
        order.update(current_price)

        if order.triggered:
            await self._execute_market_order(order)

    async def _execute_market_order(self, order: Order):
        """Execute market order via broker"""
        try:
            result = await self.broker.place_order(symbol=order.symbol, side=order.side.value, quantity=order.quantity, order_type="market")

            if result and result.get("status") != "rejected":
                order.broker_order_id = result.get("order_id")
                order.status = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                order.average_fill_price = result.get("filled_price")
                order.filled_at = datetime.now(timezone.utc)

                # Remove from pending
                if order.id in self.pending_orders:
                    del self.pending_orders[order.id]

                logger.info(f"Market order filled: {order.id} at ${order.average_fill_price:.2f}")
            else:
                order.status = OrderStatus.REJECTED
                order.error_message = result.get("error", "Unknown error")

        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            logger.error(f"Failed to execute market order: {e}")

    async def _execute_limit_order(self, order: LimitOrder, fill_price: float):
        """Execute limit order"""
        # Simulate fill at limit price
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_fill_price = min(fill_price, order.limit_price) if order.side == OrderSide.BUY else max(fill_price, order.limit_price)
        order.filled_at = datetime.now(timezone.utc)

        # Remove from pending
        if order.id in self.pending_orders:
            del self.pending_orders[order.id]

        logger.info(f"Limit order filled: {order.id} at ${order.average_fill_price:.2f}")

    def _validate_order(self, order: Order) -> bool:
        """Validate order parameters"""
        if order.quantity <= 0:
            order.error_message = "Quantity must be positive"
            return False

        if isinstance(order, LimitOrder) and order.limit_price <= 0:
            order.error_message = "Limit price must be positive"
            return False

        if isinstance(order, StopOrder) and order.stop_price <= 0:
            order.error_message = "Stop price must be positive"
            return False

        return True

    def get_pending_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all pending orders"""
        orders = list(self.pending_orders.values())

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        return orders
