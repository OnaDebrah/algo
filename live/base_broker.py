import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .orders import OrderSide, OrderStatus, OrderType

logger = logging.getLogger(__name__)


class BaseBroker(ABC):
    """Abstract base class for all broker implementations"""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        """
        Initialize broker

        Args:
            api_key: API key
            secret_key: Secret key
            paper: Use paper trading mode
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.connected = False

    @abstractmethod
    def connect(self) -> bool:
        """Connect to broker API"""
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect from broker API"""
        pass

    @abstractmethod
    def get_account(self) -> Dict:
        """Get account information"""
        pass

    @abstractmethod
    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for specific symbol"""
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass

    @abstractmethod
    def get_order(self, order_id: str) -> Dict:
        """Get order status"""
        pass

    @abstractmethod
    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Dict]:
        """Get all orders"""
        pass

    @abstractmethod
    def get_quote(self, symbol: str) -> Dict:
        """Get current quote for symbol"""
        pass
