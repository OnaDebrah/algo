import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from backend.app.schemas.settings import UserSettings

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
            'buying_power': float
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
        self, symbol: str, side: str, quantity: float, order_type: str = "market", limit_price: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Place an order

        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            quantity: Number of shares
            order_type: 'market' or 'limit'
            limit_price: Limit price (for limit orders)

        Returns: {
            'order_id': str,
            'status': str,
            'filled_price': float,
            'filled_quantity': float
        }
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass

    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get all open positions

        Returns: [{
            'symbol': str,
            'quantity': float,
            'entry_price': float,
            'current_price': float,
            'market_value': float,
            'unrealized_pnl': float
        }]
        """
        pass
