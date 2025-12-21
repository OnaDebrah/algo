"""
Interactive Brokers Integration
Complete implementation using ib_insync library
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from .base_broker import BaseBroker
from .orders import OrderSide, OrderStatus, OrderType

logger = logging.getLogger(__name__)


class IBBroker(BaseBroker):
    """Interactive Brokers broker implementation using ib_insync"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,  # 7497 for TWS paper, 7496 for TWS live, 4002 for Gateway paper, 4001 for Gateway live
        client_id: int = 1,
        paper: bool = True,
    ):
        """
        Initialize IB broker connection

        Args:
            host: IB Gateway/TWS host address
            port: Connection port
                - TWS Paper Trading: 7497
                - TWS Live Trading: 7496
                - IB Gateway Paper: 4002
                - IB Gateway Live: 4001
            client_id: Unique client ID (0-999)
            paper: Paper trading mode flag
        """
        super().__init__("ib_api", "ib_secret", paper)
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = None
        self._event_loop = None
        self._thread = None

    def connect(self) -> bool:
        """Connect to Interactive Brokers"""
        try:
            from ib_insync import IB

            # Create IB instance
            self.ib = IB()

            # Connect to TWS/Gateway
            self.ib.connect(
                host=self.host, port=self.port, clientId=self.client_id, timeout=20
            )

            # Verify connection
            if not self.ib.isConnected():
                logger.error("Failed to connect to IB")
                return False

            self.connected = True

            # Get account info to verify
            accounts = self.ib.managedAccounts()
            account_summary = self.ib.accountSummary()

            mode = "Paper" if self.paper else "Live"
            logger.info(f"Connected to Interactive Brokers ({mode} mode)")
            logger.info(f"Accounts: {accounts}")
            logger.info(f"Port: {self.port}")
            logger.info(f"Account Summary: {account_summary}")

            return True

        except ImportError:
            logger.error("ib_insync not installed. Install with: pip install ib_insync")
            return False
        except ConnectionRefusedError:
            logger.error(
                f"Connection refused on port {self.port}. "
                "Make sure TWS/IB Gateway is running and API is enabled."
            )
            return False
        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            return False

    def disconnect(self):
        """Disconnect from Interactive Brokers"""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from Interactive Brokers")

    def get_account(self) -> Dict:
        """Get account information"""
        if not self.connected:
            raise ConnectionError("Not connected to IB")

        try:
            # Get account summary
            account_values = self.ib.accountSummary()

            # Extract key values
            values_dict = {item.tag: float(item.value) for item in account_values}

            # Get primary account
            account = self.ib.managedAccounts()[0]

            return {
                "account_id": account,
                "equity": values_dict.get("NetLiquidation", 0),
                "cash": values_dict.get("TotalCashValue", 0),
                "buying_power": values_dict.get("BuyingPower", 0),
                "portfolio_value": values_dict.get("NetLiquidation", 0),
                "pattern_day_trader": False,  # IB doesn't flag this way
                "trading_blocked": False,
                "account_blocked": False,
                "currency": "USD",
                "unrealized_pnl": values_dict.get("UnrealizedPnL", 0),
                "realized_pnl": values_dict.get("RealizedPnL", 0),
                "maintenance_margin": values_dict.get("MaintMarginReq", 0),
            }

        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            raise

    def get_positions(self) -> List[Dict]:
        """Get all positions"""
        if not self.connected:
            raise ConnectionError("Not connected to IB")

        try:
            positions = self.ib.positions()

            result = []
            for pos in positions:
                # Get current market price
                contract = pos.contract
                ticker = self.ib.reqTickers(contract)[0]

                current_price = ticker.marketPrice()
                if current_price != current_price:  # NaN check
                    current_price = (
                        (ticker.bid + ticker.ask) / 2
                        if ticker.bid and ticker.ask
                        else 0
                    )

                position_dict = {
                    "symbol": contract.symbol,
                    "quantity": int(pos.position),
                    "side": "long" if pos.position > 0 else "short",
                    "entry_price": pos.avgCost,
                    "current_price": current_price,
                    "market_value": pos.position * current_price,
                    "unrealized_pl": (current_price - pos.avgCost) * pos.position,
                    "unrealized_plpc": (
                        ((current_price - pos.avgCost) / pos.avgCost)
                        if pos.avgCost != 0
                        else 0
                    ),
                    "cost_basis": pos.avgCost * abs(pos.position),
                    "contract": contract,
                    "account": pos.account,
                }

                result.append(position_dict)

            return result

        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for specific symbol"""
        positions = self.get_positions()

        for pos in positions:
            if pos["symbol"] == symbol:
                return pos

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
            raise ConnectionError("Not connected to IB")

        try:
            from ib_insync import LimitOrder, MarketOrder, Stock, StopOrder

            # Create contract (US stocks)
            contract = Stock(symbol, "SMART", "USD")

            # Qualify contract
            self.ib.qualifyContracts(contract)

            # Create order based on type
            if order_type == OrderType.MARKET:
                ib_order = MarketOrder(
                    action="BUY" if side == OrderSide.BUY else "SELL",
                    totalQuantity=quantity,
                )

            elif order_type == OrderType.LIMIT:
                if limit_price is None:
                    raise ValueError("Limit price required for limit orders")

                ib_order = LimitOrder(
                    action="BUY" if side == OrderSide.BUY else "SELL",
                    totalQuantity=quantity,
                    lmtPrice=limit_price,
                )

            elif order_type == OrderType.STOP:
                if stop_price is None:
                    raise ValueError("Stop price required for stop orders")

                ib_order = StopOrder(
                    action="BUY" if side == OrderSide.BUY else "SELL",
                    totalQuantity=quantity,
                    stopPrice=stop_price,
                )

            else:
                raise NotImplementedError(f"Order type {order_type} not implemented")

            # Place order
            trade = self.ib.placeOrder(contract, ib_order)

            # Wait for order to be submitted
            self.ib.sleep(1)

            logger.info(
                f"Order placed: {side.value} {quantity} {symbol} "
                f"@ {order_type.value}"
            )

            return {
                "order_id": str(trade.order.orderId),
                "symbol": symbol,
                "side": side.value,
                "quantity": quantity,
                "order_type": order_type.value,
                "status": trade.orderStatus.status,
                "filled_qty": int(trade.orderStatus.filled),
                "filled_avg_price": (
                    float(trade.orderStatus.avgFillPrice)
                    if trade.orderStatus.avgFillPrice
                    else 0
                ),
                "submitted_at": datetime.now(),
                "limit_price": limit_price,
                "stop_price": stop_price,
                "ib_trade": trade,  # Keep reference for updates
            }

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if not self.connected:
            raise ConnectionError("Not connected to IB")

        try:
            # Find the trade
            trades = self.ib.trades()

            for trade in trades:
                if str(trade.order.orderId) == order_id:
                    self.ib.cancelOrder(trade.order)
                    logger.info(f"Order {order_id} cancelled")
                    return True

            logger.warning(f"Order {order_id} not found")
            return False

        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    def get_order(self, order_id: str) -> Dict:
        """Get order status"""
        if not self.connected:
            raise ConnectionError("Not connected to IB")

        try:
            trades = self.ib.trades()

            for trade in trades:
                if str(trade.order.orderId) == order_id:
                    return {
                        "order_id": order_id,
                        "symbol": trade.contract.symbol,
                        "side": trade.order.action.lower(),
                        "quantity": int(trade.order.totalQuantity),
                        "order_type": trade.order.orderType.lower(),
                        "status": trade.orderStatus.status,
                        "filled_qty": int(trade.orderStatus.filled),
                        "filled_avg_price": (
                            float(trade.orderStatus.avgFillPrice)
                            if trade.orderStatus.avgFillPrice
                            else 0
                        ),
                        "submitted_at": trade.log[0].time if trade.log else None,
                        "filled_at": None,  # Would need to parse from log
                    }

            raise ValueError(f"Order {order_id} not found")

        except Exception as e:
            logger.error(f"Error getting order {order_id}: {e}")
            raise

    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Dict]:
        """Get all orders"""
        if not self.connected:
            raise ConnectionError("Not connected to IB")

        try:
            trades = self.ib.trades()

            result = []
            for trade in trades:
                order_dict = {
                    "order_id": str(trade.order.orderId),
                    "symbol": trade.contract.symbol,
                    "side": trade.order.action.lower(),
                    "quantity": int(trade.order.totalQuantity),
                    "order_type": trade.order.orderType.lower(),
                    "status": trade.orderStatus.status,
                    "filled_qty": int(trade.orderStatus.filled),
                    "filled_avg_price": (
                        float(trade.orderStatus.avgFillPrice)
                        if trade.orderStatus.avgFillPrice
                        else 0
                    ),
                    "submitted_at": trade.log[0].time if trade.log else None,
                }

                # Filter by status if specified
                if status:
                    if (
                        status == OrderStatus.FILLED
                        and trade.orderStatus.status != "Filled"
                    ):
                        continue
                    elif (
                        status == OrderStatus.PENDING
                        and trade.orderStatus.status
                        not in ["Submitted", "PreSubmitted"]
                    ):
                        continue
                    elif (
                        status == OrderStatus.CANCELLED
                        and trade.orderStatus.status != "Cancelled"
                    ):
                        continue

                result.append(order_dict)

            return result

        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []

    def get_quote(self, symbol: str) -> Dict:
        """Get current quote"""
        if not self.connected:
            raise ConnectionError("Not connected to IB")

        try:
            from ib_insync import Stock

            # Create contract
            contract = Stock(symbol, "SMART", "USD")
            self.ib.qualifyContracts(contract)

            # Request market data
            ticker = self.ib.reqMktData(contract, snapshot=True)

            # Wait for data
            self.ib.sleep(2)

            # Get quote
            return {
                "symbol": symbol,
                "bid": float(ticker.bid) if ticker.bid == ticker.bid else 0,
                "ask": float(ticker.ask) if ticker.ask == ticker.ask else 0,
                "last": float(ticker.last) if ticker.last == ticker.last else 0,
                "bid_size": int(ticker.bidSize) if ticker.bidSize else 0,
                "ask_size": int(ticker.askSize) if ticker.askSize else 0,
                "volume": int(ticker.volume) if ticker.volume else 0,
                "timestamp": datetime.now(),
            }

        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            raise

    def get_historical_data(
        self,
        symbol: str,
        duration: str = "1 D",
        bar_size: str = "1 min",
        what_to_show: str = "TRADES",
    ) -> List[Dict]:
        """
        Get historical data from IB

        Args:
            symbol: Stock symbol
            duration: Duration string (e.g., '1 D', '1 W', '1 M')
            bar_size: Bar size (e.g., '1 min', '5 mins', '1 hour', '1 day')
            what_to_show: Data type ('TRADES', 'MIDPOINT', 'BID', 'ASK')

        Returns:
            List of bar dictionaries
        """
        if not self.connected:
            raise ConnectionError("Not connected to IB")

        try:
            from ib_insync import Stock

            contract = Stock(symbol, "SMART", "USD")
            self.ib.qualifyContracts(contract)

            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=True,  # Regular trading hours only
                formatDate=1,
            )

            return [
                {
                    "date": bar.date,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
                for bar in bars
            ]

        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return []

    def place_bracket_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
    ) -> Dict:
        """
        Place a bracket order (entry + stop loss + take profit)

        Args:
            symbol: Stock symbol
            side: Buy or Sell
            quantity: Number of shares
            entry_price: Limit entry price
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price

        Returns:
            Dictionary with order IDs
        """
        if not self.connected:
            raise ConnectionError("Not connected to IB")

        try:
            from ib_insync import Stock

            contract = Stock(symbol, "SMART", "USD")
            self.ib.qualifyContracts(contract)

            # Create bracket order
            bracket = self.ib.bracketOrder(
                action="BUY" if side == OrderSide.BUY else "SELL",
                quantity=quantity,
                limitPrice=entry_price,
                takeProfitPrice=take_profit_price,
                stopLossPrice=stop_loss_price,
            )

            # Place all orders
            trades = []
            for order in bracket:
                trade = self.ib.placeOrder(contract, order)
                trades.append(trade)

            self.ib.sleep(1)

            logger.info(
                f"Bracket order placed: {side.value} {quantity} {symbol} "
                f"@ ${entry_price} (SL: ${stop_loss_price}, TP: ${take_profit_price})"
            )

            return {
                "entry_order_id": str(trades[0].order.orderId),
                "take_profit_order_id": str(trades[1].order.orderId),
                "stop_loss_order_id": str(trades[2].order.orderId),
                "symbol": symbol,
                "side": side.value,
                "quantity": quantity,
                "entry_price": entry_price,
                "stop_loss_price": stop_loss_price,
                "take_profit_price": take_profit_price,
            }

        except Exception as e:
            logger.error(f"Error placing bracket order: {e}")
            raise

    def get_market_data_subscription(self, symbol: str):
        """
        Subscribe to real-time market data

        Args:
            symbol: Stock symbol

        Returns:
            Ticker object for real-time updates
        """
        if not self.connected:
            raise ConnectionError("Not connected to IB")

        try:
            from ib_insync import Stock

            contract = Stock(symbol, "SMART", "USD")
            self.ib.qualifyContracts(contract)

            # Request streaming data
            ticker = self.ib.reqMktData(contract, "", False, False)

            logger.info(f"Subscribed to market data for {symbol}")

            return ticker

        except Exception as e:
            logger.error(f"Error subscribing to market data: {e}")
            raise

    def cancel_market_data_subscription(self, ticker):
        """Cancel market data subscription"""
        try:
            self.ib.cancelMktData(ticker.contract)
            logger.info(f"Cancelled market data for {ticker.contract.symbol}")
        except Exception as e:
            logger.error(f"Error cancelling market data: {e}")


def check_ib_connection(host: str = "127.0.0.1", port: int = 7497) -> Dict:
    """
    Test IB connection and return status

    Args:
        host: IB host
        port: IB port

    Returns:
        Dictionary with connection status
    """
    try:
        from ib_insync import IB

        ib = IB()
        ib.connect(host, port, clientId=999, timeout=5)

        if ib.isConnected():
            accounts = ib.managedAccounts()
            ib.disconnect()

            return {
                "connected": True,
                "host": host,
                "port": port,
                "accounts": accounts,
                "message": "Successfully connected to IB",
            }
        else:
            return {"connected": False, "message": "Failed to connect"}

    except ImportError:
        return {
            "connected": False,
            "message": "ib_insync not installed. Run: pip install ib_insync",
        }
    except ConnectionRefusedError:
        return {
            "connected": False,
            "message": f"Connection refused on port {port}. Is TWS/Gateway running?",
        }
    except Exception as e:
        return {"connected": False, "message": f"Error: {str(e)}"}
