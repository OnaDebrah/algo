import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...models import User, UserSettings
from ...services.brokers.alpaca_client import AlpacaClient
from ...services.brokers.base_client import BrokerClient
from ...services.brokers.ib_client import IBClient
from ...services.brokers.paper_client import PaperTradingClient

logger = logging.getLogger(__name__)


class BrokerService:
    """
    Factory for creating broker clients and executing broker operations.

    For each broker operation, resolves the appropriate broker client based on
    the configured broker type and delegates the operation to it.
    """

    _instance = None
    _default_broker = PaperTradingClient()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def create_broker(broker_type: str) -> BrokerClient:
        """
        Create broker client based on type (static factory method)

        Args:
            broker_type: 'paper', 'alpaca', 'ib', etc.

        Returns:
            BrokerClient instance
        """
        broker_type = broker_type.lower()

        if broker_type in ["paper", "paper_trading"]:
            return PaperTradingClient()

        elif broker_type in ["alpaca", "alpaca_paper"]:
            return AlpacaClient()

        elif broker_type == "alpaca_live":
            # Same client, different credentials
            return AlpacaClient()

        elif broker_type in ["ib_live", "interactive_brokers"]:
            return IBClient()

        elif broker_type in ["ib_paper", "ib_simulated"]:
            return IBClient()

        else:
            logger.warning(f"Unknown broker type: {broker_type}, defaulting to paper trading")
            return PaperTradingClient()

    async def _get_broker(self, user: Optional[User] = None, db: Optional[AsyncSession] = None, broker_type: Optional[str] = None) -> BrokerClient:
        """
        Get broker client based on user settings or explicit broker_type

        Args:
            user: User object to get broker preferences from
            db: Database session for fetching user settings
            broker_type: Explicit broker type override. If provided, this takes precedence.

        Returns:
            BrokerClient instance
        """

        if broker_type:
            return self.create_broker(broker_type)

        if not user or not db:
            return self._default_broker

        try:
            stmt = select(UserSettings).where(UserSettings.user_id == user.id)
            result = await db.execute(stmt)
            user_settings = result.scalars().first()

            if user_settings and user_settings.default_broker:
                broker_type = user_settings.default_broker
                logger.debug(f"Using broker from user settings: {broker_type}")
                return self.create_broker(broker_type)

        except Exception as e:
            logger.error(f"Failed to get broker from user settings: {e}")

        logger.debug("No broker configured, using default paper trading")
        return self._default_broker

    # ── Account Operations ──────────────────────────────────────────────

    async def get_account_info(self, user: Optional[User] = None, db: Optional[AsyncSession] = None, broker_type: Optional[str] = None) -> dict:
        """Get account information from broker."""
        broker = await self._get_broker(user, db, broker_type)
        return await broker.get_account_info()

    async def get_positions(self, user: Optional[User] = None, db: Optional[AsyncSession] = None, broker_type: Optional[str] = None) -> list:
        """Get current positions from broker."""
        broker = await self._get_broker(user, db, broker_type)
        return await broker.get_positions()

    async def get_orders(
        self, user: Optional[User] = None, db: Optional[AsyncSession] = None, broker_type: Optional[str] = None, status: Optional[str] = None
    ) -> list:
        """Get orders from broker, optionally filtered by status."""
        broker = await self._get_broker(user, db, broker_type)
        return await broker.get_orders(status)

    # ── Order Operations ────────────────────────────────────────────────

    async def place_market_order(
        self, symbol: str, qty: float, side: str, user: Optional[User] = None, db: Optional[AsyncSession] = None, broker_type: Optional[str] = None
    ) -> dict:
        """Place a market order."""
        broker = await self._get_broker(user, db, broker_type)
        return await broker.place_market_order(symbol, qty, side)

    async def place_limit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float,
        user: Optional[User] = None,
        db: Optional[AsyncSession] = None,
        broker_type: Optional[str] = None,
    ) -> dict:
        """Place a limit order."""
        broker = await self._get_broker(user, db, broker_type)
        return await broker.place_limit_order(symbol, qty, side, limit_price)

    async def place_stop_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        stop_price: float,
        user: Optional[User] = None,
        db: Optional[AsyncSession] = None,
        broker_type: Optional[str] = None,
    ) -> dict:
        """Place a stop order."""
        broker = await self._get_broker(user, db, broker_type)
        return await broker.place_stop_order(symbol, qty, side, stop_price)

    async def place_stop_limit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        stop_price: float,
        limit_price: float,
        user: Optional[User] = None,
        db: Optional[AsyncSession] = None,
        broker_type: Optional[str] = None,
    ) -> dict:
        """Place a stop-limit order."""
        broker = await self._get_broker(user, db, broker_type)
        return await broker.place_stop_limit_order(symbol, qty, side, stop_price, limit_price)

    # ── Option Operations ───────────────────────────────────────────────

    async def place_option_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        option_type: str,
        strike: float,
        expiration: str,
        order_type: str = "market",
        user: Optional[User] = None,
        db: Optional[AsyncSession] = None,
        limit_price: Optional[float] = None,
        broker_type: Optional[str] = None,
    ) -> dict:
        """Place an option order."""
        broker = await self._get_broker(user, db, broker_type)
        return await broker.place_option_order(symbol, qty, side, option_type, strike, expiration, order_type, limit_price)

    async def get_option_positions(self, user: Optional[User] = None, db: Optional[AsyncSession] = None, broker_type: Optional[str] = None) -> list:
        """Get option positions from broker."""
        broker = await self._get_broker(user, db, broker_type)
        return await broker.get_option_positions()

    # ── Order Management ────────────────────────────────────────────────

    async def cancel_order(
        self, order_id: str, user: Optional[User] = None, db: Optional[AsyncSession] = None, broker_type: Optional[str] = None
    ) -> bool:
        """Cancel an order by ID."""
        broker = await self._get_broker(user, db, broker_type)
        return await broker.cancel_order(order_id)

    async def get_order_status(
        self, order_id: str, user: Optional[User] = None, db: Optional[AsyncSession] = None, broker_type: Optional[str] = None
    ) -> dict:
        """Get order status by ID."""
        broker = await self._get_broker(user, db, broker_type)
        return await broker.get_order_status(order_id)

    async def replace_order(
        self,
        order_id: str,
        qty: Optional[float] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        user: Optional[User] = None,
        db: Optional[AsyncSession] = None,
        broker_type: Optional[str] = None,
    ) -> dict:
        """Replace/modify an existing order."""
        broker = await self._get_broker(user, db, broker_type)
        return await broker.replace_order(order_id, qty, limit_price, stop_price)

    # ── Market Data ─────────────────────────────────────────────────────

    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        user: Optional[User] = None,
        db: Optional[AsyncSession] = None,
        broker_type: Optional[str] = None,
    ) -> list:
        """Get historical bar data."""
        broker = await self._get_broker(user, db, broker_type)
        return await broker.get_bars(symbol, timeframe, start, end)

    async def get_quote(self, symbol: str, broker_type: Optional[str] = None) -> dict:
        """Get real-time quote for a symbol."""
        broker = await self._get_broker(broker_type)
        return await broker.get_quote(symbol)
