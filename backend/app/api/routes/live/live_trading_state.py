import logging
from datetime import datetime, timezone
from typing import List, Optional

from ....models import UserSettings
from ....schemas.live import BrokerType, EngineStatus
from ....services.brokers.base_client import BrokerClient
from ....services.brokers.broker_service import BrokerFactory

logger = logging.getLogger(__name__)


class LiveTradingState:
    """Centralized state management for live trading"""

    def __init__(self):
        self.is_connected: bool = False
        self.engine_status: EngineStatus = EngineStatus.IDLE
        self.active_broker: BrokerType = BrokerType.PAPER
        self.connected_at: Optional[datetime] = None
        self.running_strategy_ids: List[int] = []

        # Broker client instance
        self.broker_client: Optional[BrokerClient] = None
        self.broker_type: Optional[str] = None

    async def connect(self, broker_type: str, user_settings: UserSettings) -> bool:
        """Connect to broker using factory"""
        try:
            self.broker_client = BrokerFactory.create_broker(broker_type)
            self.broker_type = broker_type

            success = await self.broker_client.connect(user_settings)

            if success:
                self.is_connected = True
                self.active_broker = BrokerType(broker_type.lower())
                self.connected_at = datetime.now(timezone.utc)
                logger.info(f"Connected to {broker_type}")
                return True
            else:
                logger.error(f"Failed to connect to {broker_type}")
                return False

        except Exception as e:
            logger.error(f"Error connecting to broker: {e}")
            return False

    async def disconnect(self):
        """Disconnect from broker"""
        if self.broker_client:
            await self.broker_client.disconnect()

        self.is_connected = False
        self.engine_status = EngineStatus.IDLE
        self.running_strategy_ids = []
        self.broker_client = None
        self.broker_type = None

    def start_engine(self, strategy_ids: List[int] = None):
        """Start execution engine"""
        if not self.is_connected:
            raise ValueError("Broker not connected")
        self.engine_status = EngineStatus.RUNNING
        if strategy_ids:
            self.running_strategy_ids = strategy_ids

    def stop_engine(self):
        """Stop execution engine"""
        self.engine_status = EngineStatus.IDLE
        self.running_strategy_ids = []
