import logging

from backend.app.services.brokers.alpaca_client import AlpacaClient
from backend.app.services.brokers.base_client import BrokerClient
from backend.app.services.brokers.ib_client import IBClient
from backend.app.services.brokers.paper_client import PaperTradingClient

logger = logging.getLogger(__name__)


class BrokerFactory:
    """Factory for creating broker clients"""

    @staticmethod
    def create_broker(broker_type: str) -> BrokerClient:
        """
        Create broker client based on type

        Args:
            broker_type: 'paper', 'alpaca', 'ib', etc.

        Returns:
            BrokerClient instance
        """
        broker_type = broker_type.lower()

        if broker_type == "paper" or broker_type == "paper_trading":
            return PaperTradingClient()

        elif broker_type == "alpaca" or broker_type == "alpaca_paper":
            return AlpacaClient()

        elif broker_type == "alpaca_live":
            # Same client, different credentials
            return AlpacaClient()

        elif broker_type == "ib" or broker_type == "interactive_brokers":
            return IBClient()

        elif broker_type == "ib_paper" or broker_type == "ib_simulated":
            return IBClient()

        else:
            logger.warning(f"Unknown broker type: {broker_type}, defaulting to paper trading")
            return PaperTradingClient()
