import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ...models import UserSettings

logger = logging.getLogger(__name__)


class BrokerClient(ABC):
    """
    Abstract base class for broker integrations

    All broker implementations must implement these methods
    """

    @abstractmethod
    async def connect(self, settings: UserSettings) -> bool:
        """Connect to broker API"""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from broker"""
        pass

    @abstractmethod
    async def is_market_open(self) -> bool:
        """Check if market is currently open"""
        pass

    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information

        Returns: {
            'cash': float,
            'equity': float,
            'buying_power': float,
            'portfolio_value': float,
            'initial_margin': float,
            'maintenance_margin': float,
            'day_trade_count': int,
            'last_equity': float
        }
        """
        pass

    @abstractmethod
    async def get_latest_bars(self, symbol: str, limit: int = 100) -> Optional[Dict[str, List[float]]]:
        """
        Get latest price bars

        Returns: {
            'open': [prices],
            'high': [prices],
            'low': [prices],
            'close': [prices],
            'volume': [volumes],
            'timestamp': [timestamps]
        }
        """
        pass

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day",
        extended_hours: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Place a general order (can be used for all order types)

        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            quantity: Number of shares
            order_type: 'market', 'limit', 'stop', 'stop_limit', 'trailing_stop'
            limit_price: Limit price (for limit and stop_limit orders)
            stop_price: Stop price (for stop and stop_limit orders)
            time_in_force: 'day', 'gtc', 'ioc', 'fok'
            extended_hours: Whether to allow trading during extended hours

        Returns: {
            'order_id': str,
            'status': str,
            'filled_price': float,
            'filled_quantity': float,
            'created_at': str,
            'updated_at': str
        }
        """
        pass

    # ── Specialized Order Methods ───────────────────────────────────────

    @abstractmethod
    async def place_market_order(
        self, symbol: str, qty: float, side: str, time_in_force: str = "day", extended_hours: bool = False
    ) -> Dict[str, Any]:
        """
        Place a market order

        Args:
            symbol: Stock symbol
            qty: Number of shares
            side: 'buy' or 'sell'
            time_in_force: 'day', 'gtc', 'ioc', 'fok'
            extended_hours: Whether to allow trading during extended hours

        Returns: {
            'order_id': str,
            'status': str,
            'filled_price': float,
            'filled_quantity': float,
            'created_at': str
        }
        """
        pass

    @abstractmethod
    async def place_limit_order(
        self, symbol: str, qty: float, side: str, limit_price: float, time_in_force: str = "day", extended_hours: bool = False
    ) -> Dict[str, Any]:
        """
        Place a limit order

        Args:
            symbol: Stock symbol
            qty: Number of shares
            side: 'buy' or 'sell'
            limit_price: Limit price
            time_in_force: 'day', 'gtc', 'ioc', 'fok'
            extended_hours: Whether to allow trading during extended hours

        Returns: {
            'order_id': str,
            'status': str,
            'limit_price': float,
            'filled_price': float,
            'filled_quantity': float,
            'created_at': str
        }
        """
        pass

    @abstractmethod
    async def place_stop_order(
        self, symbol: str, qty: float, side: str, stop_price: float, time_in_force: str = "day", extended_hours: bool = False
    ) -> Dict[str, Any]:
        """
        Place a stop order

        Args:
            symbol: Stock symbol
            qty: Number of shares
            side: 'buy' or 'sell'
            stop_price: Stop price
            time_in_force: 'day', 'gtc', 'ioc', 'fok'
            extended_hours: Whether to allow trading during extended hours

        Returns: {
            'order_id': str,
            'status': str,
            'stop_price': float,
            'filled_price': float,
            'filled_quantity': float,
            'created_at': str
        }
        """
        pass

    @abstractmethod
    async def place_stop_limit_order(
        self, symbol: str, qty: float, side: str, stop_price: float, limit_price: float, time_in_force: str = "day", extended_hours: bool = False
    ) -> Dict[str, Any]:
        """
        Place a stop-limit order

        Args:
            symbol: Stock symbol
            qty: Number of shares
            side: 'buy' or 'sell'
            stop_price: Stop price that triggers the limit order
            limit_price: Limit price
            time_in_force: 'day', 'gtc', 'ioc', 'fok'
            extended_hours: Whether to allow trading during extended hours

        Returns: {
            'order_id': str,
            'status': str,
            'stop_price': float,
            'limit_price': float,
            'filled_price': float,
            'filled_quantity': float,
            'created_at': str
        }
        """
        pass

    # ── Option Order Methods ───────────────────────────────────────────

    @abstractmethod
    async def place_option_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        option_type: str,
        strike: float,
        expiration: str,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        time_in_force: str = "day",
    ) -> Dict[str, Any]:
        """
        Place an option order

        Args:
            symbol: Underlying stock symbol
            qty: Number of contracts
            side: 'buy' or 'sell'
            option_type: 'call' or 'put'
            strike: Strike price
            expiration: Expiration date (YYYY-MM-DD)
            order_type: 'market' or 'limit'
            limit_price: Limit price (for limit orders)
            time_in_force: 'day', 'gtc', 'ioc', 'fok'

        Returns: {
            'order_id': str,
            'status': str,
            'filled_price': float,
            'filled_quantity': int,
            'created_at': str
        }
        """
        pass

    @abstractmethod
    async def get_option_positions(self) -> List[Dict[str, Any]]:
        """
        Get all open option positions

        Returns: [{
            'symbol': str,
            'quantity': int,
            'option_type': str,
            'strike': float,
            'expiration': str,
            'entry_price': float,
            'current_price': float,
            'market_value': float,
            'unrealized_pnl': float
        }]
        """
        pass

    # ── Order Management Methods ────────────────────────────────────────

    @abstractmethod
    async def get_orders(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get orders, optionally filtered by status

        Args:
            status: 'open', 'closed', 'all', or None for all orders

        Returns: [{
            'order_id': str,
            'symbol': str,
            'side': str,
            'quantity': float,
            'filled_quantity': float,
            'order_type': str,
            'limit_price': Optional[float],
            'stop_price': Optional[float],
            'status': str,
            'created_at': str,
            'updated_at': str
        }]
        """
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get order status by ID

        Args:
            order_id: Order ID

        Returns: {
            'order_id': str,
            'symbol': str,
            'side': str,
            'quantity': float,
            'filled_quantity': float,
            'order_type': str,
            'limit_price': Optional[float],
            'stop_price': Optional[float],
            'status': str,
            'created_at': str,
            'updated_at': str,
            'filled_price': Optional[float]
        }
        """
        pass

    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get all open stock/equity positions

        Returns: [{
            'symbol': str,
            'quantity': float,
            'entry_price': float,
            'current_price': float,
            'market_value': float,
            'unrealized_pnl': float,
            'unrealized_pnl_percent': float,
            'day_change': float,
            'day_change_percent': float
        }]
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order by ID

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancellation was successful, False otherwise
        """
        pass

    @abstractmethod
    async def replace_order(
        self,
        order_id: str,
        qty: Optional[float] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Replace/modify an existing order

        Args:
            order_id: Order ID to replace
            qty: New quantity (if None, keep original)
            limit_price: New limit price (if applicable)
            stop_price: New stop price (if applicable)
            time_in_force: New time in force

        Returns: Updated order information (same format as get_order_status)
        """
        pass

    # ── Market Data Methods ─────────────────────────────────────────────

    @abstractmethod
    async def get_bars(self, symbol: str, timeframe: str, start: str, end: str, adjustment: str = "raw") -> List[Dict[str, Any]]:
        """
        Get historical bar data

        Args:
            symbol: Stock symbol
            timeframe: '1m', '5m', '15m', '1h', '1d', etc.
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            adjustment: 'raw', 'split', 'dividend'

        Returns: [{
            'timestamp': str,
            'open': float,
            'high': float,
            'low': float,
            'close': float,
            'volume': int,
            'vwap': Optional[float]
        }]
        """
        pass

    @abstractmethod
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time quote for a symbol

        Args:
            symbol: Stock symbol

        Returns: {
            'bid': float,
            'ask': float,
            'bid_size': int,
            'ask_size': int,
            'last': float,
            'volume': int,
            'timestamp': str
        }
        """
        pass
