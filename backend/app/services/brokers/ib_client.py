"""
Interactive Brokers Client Implementation
Requires: pip install ib_insync (recommended for async support)
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ib_insync import IB, util

from backend.app.models import UserSettings
from backend.app.services.brokers.base_client import BrokerClient

logger = logging.getLogger(__name__)

util.patchAsyncio()


class IBClient(BrokerClient):
    """
    Interactive Brokers client implementation using ib_insync

    Note: Requires IB Gateway or TWS to be running
    """

    def __init__(self):
        self.ib = IB()
        self.connected = False
        self.account_id = None
        self._bars_cache = {}  # Cache for historical data

    async def connect(self, settings: UserSettings, lightweight: bool = False) -> bool:
        """Connect to IB Gateway/TWS using native async methods"""
        try:
            if self.ib.isConnected():
                self.ib.disconnect()

            host = settings.broker_host
            port = settings.broker_port
            client_id = settings.broker_client_id
            self.account_id: str = settings.user_ib_account_id

            await self.ib.connectAsync(host=host, port=port, clientId=client_id, timeout=20)

            if self.ib.isConnected():
                self.connected = True
                self.ib.errorEvent += self._on_error

                if not lightweight and self.account_id:
                    self.ib.reqAccountUpdates(self.account_id)

                logger.info(f"Connected to Interactive Brokers (Account: {self.account_id})")
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"Error connecting to IB: {e}")
            self.connected = False
            return False

    def _on_error(self, reqId, errorCode, errorString, contract):
        """Handle IB error messages"""
        if errorCode in [2104, 2106, 2158]:  # Market data farm connection messages
            return  # These are just informational
        logger.error(f"IB Error {errorCode}: {errorString} (ReqId: {reqId})")

    async def disconnect(self):
        """Disconnect from IB"""
        if self.ib and self.connected:
            try:
                self.ib.disconnect()
                self.connected = False
                logger.info("Disconnected from Interactive Brokers")
            except Exception as e:
                logger.error(f"Error disconnecting from IB: {e}")
        self.connected = False

    async def is_market_open(self) -> bool:
        """
        Check if market is open for trading

        Note: IB doesn't have a direct market open/close API.
        This implementation checks if current time is within trading hours.
        """
        try:
            # Check for US stock market hours (NYSE/NASDAQ)
            now = datetime.now()

            # Simple check: Monday-Friday, 9:30 AM - 4:00 PM ET
            if now.weekday() >= 5:  # Saturday or Sunday
                return False

            # Convert to ET (simplified - in production, use pytz)
            hour = now.hour - 5  # EST (no DST handling here)
            if hour < 0:
                hour += 24

            return 9.5 <= hour < 16  # 9:30 AM to 4:00 PM

        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False

    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information from IB"""
        if not self.connected:
            return {}

        try:
            # Wait for account values to be populated
            await asyncio.sleep(0.5)

            # Get account summary
            account_values = self.ib.accountValues(self.account_id)

            # Convert to dictionary
            account_dict = {}
            for value in account_values:
                account_dict[value.tag] = value.value

            # Extract key values (tag names are standardized in IB)
            return {
                "cash": float(account_dict.get("AvailableFunds", 0)),
                "equity": float(account_dict.get("NetLiquidation", 0)),
                "buying_power": float(account_dict.get("BuyingPower", 0)),
                "portfolio_value": float(account_dict.get("NetLiquidation", 0)),
                "total_cash": float(account_dict.get("TotalCashValue", 0)),
                "unrealized_pnl": float(account_dict.get("UnrealizedPnL", 0)),
                "realized_pnl": float(account_dict.get("RealizedPnL", 0)),
            }

        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}

    async def get_latest_bars(self, symbol: str, limit: int = 100) -> Optional[Dict[str, List[float]]]:
        """Get latest price bars from IB"""
        if not self.connected:
            return None

        try:
            # Create contract for the symbol
            from ib_insync import Stock

            contract = Stock(symbol, "SMART", "USD")

            # Request historical data
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=f"{limit} D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
                keepUpToDate=False,
            )

            # Convert to required format
            if not bars:
                return None

            return {
                "open": [bar.open for bar in bars[-limit:]],
                "high": [bar.high for bar in bars[-limit:]],
                "low": [bar.low for bar in bars[-limit:]],
                "close": [bar.close for bar in bars[-limit:]],
                "volume": [bar.volume for bar in bars[-limit:]],
                "timestamp": [bar.date for bar in bars[-limit:]],
            }

        except Exception as e:
            logger.error(f"Error fetching bars for {symbol}: {e}")
            return None

    async def place_order(
        self, symbol: str, side: str, quantity: float, order_type: str = "market", limit_price: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Place order with IB"""
        if not self.connected:
            return None

        try:
            from ib_insync import LimitOrder, MarketOrder, Stock

            # Create contract
            contract = Stock(symbol, "SMART", "USD")

            # Create order based on type
            if order_type.lower() == "market":
                order = MarketOrder(action=side.upper(), totalQuantity=quantity, tif="DAY")
            elif order_type.lower() == "limit" and limit_price:
                order = LimitOrder(action=side.upper(), totalQuantity=quantity, lmtPrice=limit_price, tif="DAY")
            else:
                logger.error(f"Unsupported order type: {order_type}")
                return None

            # Place the order
            trade = self.ib.placeOrder(contract, order)

            # Wait for order to be processed (simplified)
            await asyncio.sleep(1)

            # Get order status
            order_status = trade.orderStatus.status

            return {
                "order_id": str(trade.order.orderId),
                "status": order_status,
                "filled_price": trade.orderStatus.avgFillPrice if trade.orderStatus.avgFillPrice else None,
                "filled_quantity": trade.orderStatus.filled,
                "remaining_quantity": trade.orderStatus.remaining,
            }

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {"order_id": None, "status": "rejected", "error": str(e)}

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order in IB"""
        if not self.connected:
            return False

        try:
            # Find the trade by order ID
            for trade in self.ib.openTrades():
                if str(trade.order.orderId) == order_id:
                    self.ib.cancelOrder(trade.order)
                    await asyncio.sleep(0.5)  # Wait for cancellation
                    return True

            logger.warning(f"Order {order_id} not found")
            return False

        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get all positions from IB"""
        if not self.connected:
            return []

        try:
            positions = []

            # Get positions from IB
            ib_positions = self.ib.positions()

            for pos in ib_positions:
                try:
                    # Get current market data for the position
                    contract = pos.contract
                    ticker = self.ib.reqMktData(contract, "", False, False)

                    # Wait for market data
                    await asyncio.sleep(0.5)

                    current_price = ticker.marketPrice() or ticker.close

                    positions.append(
                        {
                            "symbol": contract.symbol,
                            "quantity": float(pos.position),
                            "entry_price": float(pos.avgCost),
                            "current_price": float(current_price) if current_price else float(pos.avgCost),
                            "market_value": float(pos.position * (current_price if current_price else pos.avgCost)),
                            "unrealized_pnl": float(pos.unrealizedPnL),
                        }
                    )

                except Exception as e:
                    logger.error(f"Error processing position {pos.contract.symbol}: {e}")
                    continue

            return positions

        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
