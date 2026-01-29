import logging
from typing import Dict, List, Optional

from alpaca.common import APIError

from .base_broker import BaseBroker
from .orders import OrderSide, OrderStatus, OrderType

logger = logging.getLogger(__name__)


class AlpacaBroker(BaseBroker):
    """Alpaca broker implementation"""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        super().__init__(api_key, secret_key, paper)
        self.api = None
        self.base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"

    def connect(self) -> bool:
        """Connect to Alpaca API"""
        try:
            from alpaca.trading.client import TradingClient

            self.api = TradingClient(api_key=self.api_key, secret_key=self.secret_key, paper=self.paper)

            # Test connection
            account = self.api.get_account()
            self.connected = True
            logger.info(f"Connected to Alpaca acc: {account} ({'Paper' if self.paper else 'Live'} mode)")
            return True

        except ImportError:
            logger.error("Alpaca package not installed. Run: pip install alpaca-py")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            return False

    def disconnect(self):
        """Disconnect from Alpaca"""
        self.api = None
        self.connected = False
        logger.info("Disconnected from Alpaca")

    def get_account(self) -> Dict:
        """Get account information"""
        if not self.connected:
            raise ConnectionError("Not connected to broker")

        account = self.api.get_account()

        return {
            "account_id": account.id,
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "pattern_day_trader": account.pattern_day_trader,
            "trading_blocked": account.trading_blocked,
            "account_blocked": account.account_blocked,
            "currency": account.currency,
        }

    def get_positions(self) -> List[Dict]:
        """Get all positions"""
        if not self.connected:
            raise ConnectionError("Not connected to broker")

        positions = self.api.get_all_positions()

        return [
            {
                "symbol": pos.symbol,
                "quantity": int(pos.qty),
                "side": "long" if float(pos.qty) > 0 else "short",
                "entry_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "market_value": float(pos.market_value),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc),
                "cost_basis": float(pos.cost_basis),
            }
            for pos in positions
        ]

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for specific symbol"""
        if not self.connected:
            raise ConnectionError("Not connected to broker")

        try:
            pos = self.api.get_open_position(symbol)

            return {
                "symbol": pos.symbol,
                "quantity": int(pos.qty),
                "side": "long" if float(pos.qty) > 0 else "short",
                "entry_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "market_value": float(pos.market_value),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc),
                "cost_basis": float(pos.cost_basis),
            }
        except APIError:
            return None

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Dict:
        """Place an order"""
        if not self.connected:
            raise ConnectionError("Not connected to broker")

        from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

        # Convert side
        alpaca_side = AlpacaOrderSide.BUY if side == OrderSide.BUY else AlpacaOrderSide.SELL

        # Create order request
        if order_type == OrderType.MARKET:
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=alpaca_side,
                time_in_force=TimeInForce.DAY,
            )
        elif order_type == OrderType.LIMIT:
            if limit_price is None:
                raise ValueError("Limit price required for limit orders")

            order_data = LimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=alpaca_side,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
            )
        else:
            raise NotImplementedError(f"Order type {order_type} not implemented")

        # Submit order
        order = self.api.submit_order(order_data)

        logger.info(f"Order placed: {side.value} {quantity} {symbol} " f"@ {order_type.value}")

        return {
            "order_id": order.id,
            "symbol": order.symbol,
            "side": side.value,
            "quantity": quantity,
            "order_type": order_type.value,
            "status": order.status,
            "filled_qty": int(order.filled_qty or 0),
            "filled_avg_price": float(order.filled_avg_price or 0),
            "submitted_at": order.submitted_at,
            "limit_price": limit_price,
            "stop_price": stop_price,
        }

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if not self.connected:
            raise ConnectionError("Not connected to broker")

        try:
            self.api.cancel_order_by_id(order_id)
            logger.info(f"Order {order_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_order(self, order_id: str) -> Dict:
        """Get order status"""
        if not self.connected:
            raise ConnectionError("Not connected to broker")

        order = self.api.get_order_by_id(order_id)

        return {
            "order_id": order.id,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": int(order.qty),
            "order_type": order.order_type,
            "status": order.status,
            "filled_qty": int(order.filled_qty or 0),
            "filled_avg_price": float(order.filled_avg_price or 0),
            "submitted_at": order.submitted_at,
            "filled_at": order.filled_at,
        }

    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Dict]:
        """Get all orders"""
        if not self.connected:
            raise ConnectionError("Not connected to broker")

        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        # Convert status
        alpaca_status = None
        if status == OrderStatus.FILLED:
            alpaca_status = QueryOrderStatus.CLOSED
        elif status == OrderStatus.PENDING:
            alpaca_status = QueryOrderStatus.OPEN

        request = GetOrdersRequest(status=alpaca_status)
        orders = self.api.get_orders(filter=request)

        return [
            {
                "order_id": order.id,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": int(order.qty),
                "order_type": order.order_type,
                "status": order.status,
                "filled_qty": int(order.filled_qty or 0),
                "filled_avg_price": float(order.filled_avg_price or 0),
                "submitted_at": order.submitted_at,
            }
            for order in orders
        ]

    def get_quote(self, symbol: str) -> Dict:
        """Get current quote"""
        if not self.connected:
            raise ConnectionError("Not connected to broker")

        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest

        data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quote = data_client.get_stock_latest_quote(request)[symbol]

        return {
            "symbol": symbol,
            "bid": float(quote.bid_price),
            "ask": float(quote.ask_price),
            "bid_size": int(quote.bid_size),
            "ask_size": int(quote.ask_size),
            "timestamp": quote.timestamp,
        }
