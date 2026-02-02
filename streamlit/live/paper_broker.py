import logging
from datetime import datetime
from typing import Dict, List, Optional

from .base_broker import BaseBroker
from .orders import OrderSide, OrderStatus, OrderType

logger = logging.getLogger(__name__)


class PaperBroker(BaseBroker):
    """Paper trading broker (simulated)"""

    def __init__(self, initial_cash: float = 100000):
        super().__init__("paper", "paper", paper=True)
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions: Dict[str, Dict] = {}
        self.orders: Dict[str, Dict] = {}
        self.order_counter = 0

    def connect(self) -> bool:
        """Connect to paper broker"""
        self.connected = True
        logger.info("Connected to Paper Trading (Simulation)")
        return True

    def disconnect(self):
        """Disconnect"""
        self.connected = False
        logger.info("Disconnected from Paper Trading")

    def get_account(self) -> Dict:
        """Get account information"""
        equity = self.cash
        for pos in self.positions.values():
            equity += pos["quantity"] * pos["current_price"]

        return {
            "account_id": "PAPER",
            "equity": equity,
            "cash": self.cash,
            "buying_power": self.cash,
            "portfolio_value": equity,
            "pattern_day_trader": False,
            "trading_blocked": False,
            "account_blocked": False,
            "currency": "USD",
        }

    def get_positions(self) -> List[Dict]:
        """Get all positions"""
        return list(self.positions.values())

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for symbol"""
        return self.positions.get(symbol)

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Dict:
        """Place an order (simulated)"""
        from streamlit.core.data_fetcher import fetch_stock_data

        # Get current price
        data = fetch_stock_data(symbol, "1d", "1d")
        if data.empty:
            raise ValueError(f"Cannot get price for {symbol}")

        current_price = float(data["Close"].iloc[-1])
        execution_price = limit_price if order_type == OrderType.LIMIT else current_price

        # Create order
        self.order_counter += 1
        order_id = f"PAPER{self.order_counter:06d}"

        order = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side.value,
            "quantity": quantity,
            "order_type": order_type.value,
            "status": OrderStatus.FILLED.value,
            "filled_qty": quantity,
            "filled_avg_price": execution_price,
            "submitted_at": datetime.now(),
            "filled_at": datetime.now(),
            "limit_price": limit_price,
            "stop_price": stop_price,
        }

        # Execute order
        if side == OrderSide.BUY:
            cost = quantity * execution_price
            if cost > self.cash:
                order["status"] = OrderStatus.REJECTED.value
                order["reject_reason"] = "Insufficient funds"
            else:
                self.cash -= cost
                if symbol in self.positions:
                    pos = self.positions[symbol]
                    total_qty = pos["quantity"] + quantity
                    total_cost = pos["entry_price"] * pos["quantity"] + execution_price * quantity
                    pos["entry_price"] = total_cost / total_qty
                    pos["quantity"] = total_qty
                else:
                    self.positions[symbol] = {
                        "symbol": symbol,
                        "quantity": quantity,
                        "side": "long",
                        "entry_price": execution_price,
                        "current_price": current_price,
                        "market_value": quantity * current_price,
                        "unrealized_pl": 0,
                        "unrealized_plpc": 0,
                        "cost_basis": quantity * execution_price,
                    }

        elif side == OrderSide.SELL:
            if symbol not in self.positions or self.positions[symbol]["quantity"] < quantity:
                order["status"] = OrderStatus.REJECTED.value
                order["reject_reason"] = "Insufficient shares"
            else:
                self.cash += quantity * execution_price
                pos = self.positions[symbol]
                pos["quantity"] -= quantity
                if pos["quantity"] == 0:
                    del self.positions[symbol]

        self.orders[order_id] = order
        logger.info(f"Paper order: {side.value} {quantity} {symbol} @ ${execution_price:.2f}")

        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel order (not applicable for instant execution)"""
        return False

    def get_order(self, order_id: str) -> Dict:
        """Get order"""
        return self.orders.get(order_id, {})

    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Dict]:
        """Get all orders"""
        orders = list(self.orders.values())
        if status:
            orders = [o for o in orders if o["status"] == status.value]
        return orders

    def get_quote(self, symbol: str) -> Dict:
        """Get quote (simulated)"""
        from streamlit.core.data_fetcher import fetch_stock_data

        data = fetch_stock_data(symbol, "1d", "1d")
        if data.empty:
            raise ValueError(f"Cannot get quote for {symbol}")

        price = float(data["Close"].iloc[-1])

        return {
            "symbol": symbol,
            "bid": price * 0.9999,
            "ask": price * 1.0001,
            "bid_size": 100,
            "ask_size": 100,
            "timestamp": datetime.now(),
        }
