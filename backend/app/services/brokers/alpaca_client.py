"""
Broker Client Interface and Implementations
Supports: Alpaca, Paper Trading, and extensible for other brokers
"""

import logging
from typing import Any, Dict, List, Optional

from backend.app.config import settings
from backend.app.models import UserSettings
from backend.app.services.brokers.base_client import BrokerClient

logger = logging.getLogger(__name__)


class AlpacaClient(BrokerClient):
    """
    Alpaca broker integration

    Requires: pip install alpaca-trade-api
    """

    def __init__(self):
        self.api = None
        self.connected = False

    async def connect(self, user_settings: UserSettings) -> bool:
        """Connect to Alpaca API"""
        try:
            import alpaca_trade_api as tradeapi

            api_key = user_settings.broker_api_key
            api_secret = user_settings.broker_api_secret
            base_url = user_settings.broker_base_url or settings.ALPACA_PAPER_BASE_URL

            if not api_key or not api_secret:
                logger.error("Missing Alpaca credentials")
                return False

            self.api = tradeapi.REST(api_key, api_secret, base_url, api_version="v2")

            # Test connection
            account = self.api.get_account()
            logger.info(f"Connected to Alpaca: {account.status}")

            self.connected = True
            return True

        except ImportError:
            logger.error("alpaca-trade-api not installed. Run: pip install alpaca-trade-api")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            return False

    async def disconnect(self):
        """Disconnect from Alpaca"""
        self.api = None
        self.connected = False

    async def is_market_open(self) -> bool:
        """Check if market is open"""
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False

    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            account = self.api.get_account()
            return {
                "cash": float(account.cash),
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}

    async def get_latest_bars(self, symbol: str, limit: int = 100) -> Optional[Dict[str, List[float]]]:
        """Get latest price bars from Alpaca"""
        try:
            # Get bars
            bars = self.api.get_bars(symbol, "1Min", limit=limit).df

            if bars.empty:
                return None

            return {
                "open": bars["open"].tolist(),
                "high": bars["high"].tolist(),
                "low": bars["low"].tolist(),
                "close": bars["close"].tolist(),
                "volume": bars["volume"].tolist(),
                "timestamp": bars.index.tolist(),
            }

        except Exception as e:
            logger.error(f"Error fetching bars for {symbol}: {e}")
            return None

    async def place_order(
        self, symbol: str, side: str, quantity: float, order_type: str = "market", limit_price: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Place order with Alpaca"""
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=int(quantity),
                side=side,
                type=order_type,
                time_in_force="day",
                limit_price=limit_price if order_type == "limit" else None,
            )

            return {
                "order_id": order.id,
                "status": order.status,
                "filled_price": float(order.filled_avg_price) if order.filled_avg_price else None,
                "filled_quantity": float(order.filled_qty) if order.filled_qty else 0,
            }

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {"order_id": None, "status": "rejected", "error": str(e)}

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        try:
            self.api.cancel_order(order_id)
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get all positions"""
        try:
            positions = self.api.list_positions()

            return [
                {
                    "symbol": pos.symbol,
                    "quantity": float(pos.qty),
                    "entry_price": float(pos.avg_entry_price),
                    "current_price": float(pos.current_price),
                    "market_value": float(pos.market_value),
                    "unrealized_pnl": float(pos.unrealized_pl),
                }
                for pos in positions
            ]

        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
