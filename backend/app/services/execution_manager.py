"""
Live Strategy Execution Manager
Coordinates execution of multiple live strategies
"""

import asyncio
import logging
from typing import Dict, Optional

from sqlalchemy import select

from ..api.routes.settings import get_or_create_settings
from ..models import UserSettings
from ..models.live import LiveStrategy, StrategyStatus
from ..services.brokers.broker_service import BrokerClient, BrokerService
from ..services.strategy_executor_service import StrategyExecutor

logger = logging.getLogger(__name__)


class ExecutionManager:
    """
    Manages execution of all live strategies

    Responsibilities:
    - Start/stop strategy executors
    - Monitor strategy health
    - Coordinate broker connections
    - Handle errors and recovery
    """

    def __init__(self, db_session_factory):
        self.db_session_factory = db_session_factory

        self.executors: Dict[int, StrategyExecutor] = {}

        self.brokers: Dict[str, BrokerClient] = {}

        self.running = False
        self.monitor_task: Optional[asyncio.Task] = None

    async def start(self):
        """
        Start the execution manager

        - Loads all running strategies from DB
        - Creates executors for each
        - Starts monitoring task
        """
        logger.info("Starting Execution Manager")
        self.running = True

        # Load running strategies from database
        await self._load_running_strategies()

        # Start monitoring task
        self.monitor_task = asyncio.create_task(self._monitor_strategies())

        logger.info(f"Execution Manager started with {len(self.executors)} strategies")

    async def stop(self):
        """Stop all strategies and executors"""
        logger.info("Stopping Execution Manager")
        self.running = False

        # Stop all executors
        for strategy_id, executor in list(self.executors.items()):
            await self._stop_executor(strategy_id)

        # Cancel monitor task
        if self.monitor_task:
            self.monitor_task.cancel()

        # Disconnect all brokers
        for broker_name, broker in self.brokers.items():
            await broker.disconnect()

        logger.info("Execution Manager stopped")

    async def deploy_strategy(self, strategy_id: int) -> bool:
        """
        Deploy a new strategy for execution

        Args:
            strategy_id: ID of strategy to deploy

        Returns:
            bool: True if deployed successfully
        """
        logger.info(f"Deploying strategy {strategy_id}")

        # Check if already running
        if strategy_id in self.executors:
            logger.warning(f"Strategy {strategy_id} already running")
            return False

        # Create executor
        try:
            executor = await self._create_executor(strategy_id)
            if not executor:
                return False

            # Start executor in background
            asyncio.create_task(executor.start())

            # Store executor
            self.executors[strategy_id] = executor

            logger.info(f"Strategy {strategy_id} deployed successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to deploy strategy {strategy_id}: {e}")
            return False

    async def pause_strategy(self, strategy_id: int) -> bool:
        """Pause a running strategy"""
        if strategy_id not in self.executors:
            logger.warning(f"Strategy {strategy_id} not running")
            return False

        executor = self.executors[strategy_id]
        await executor.pause()

        logger.info(f"Strategy {strategy_id} paused")
        return True

    async def resume_strategy(self, strategy_id: int) -> bool:
        """Resume a paused strategy"""
        if strategy_id not in self.executors:
            # Strategy not in memory, redeploy
            return await self.deploy_strategy(strategy_id)

        executor = self.executors[strategy_id]
        asyncio.create_task(executor.start())

        logger.info(f"Strategy {strategy_id} resumed")
        return True

    async def stop_strategy(self, strategy_id: int) -> bool:
        """Stop a strategy"""
        return await self._stop_executor(strategy_id)

    async def _create_executor(self, strategy_id: int) -> Optional[StrategyExecutor]:
        """Create a strategy executor"""
        db = self.db_session_factory()

        try:
            stmt = select(LiveStrategy).where(LiveStrategy.id == strategy_id)

            result = await db.execute(stmt)
            strategy = result.scalars().first()

            if not strategy:
                logger.error(f"Strategy {strategy_id} not found")
                return None

            user_settings = await get_or_create_settings(db, strategy.user_id)

            broker_name = strategy.broker or user_settings.default_broker or "paper"

            broker = await self._get_or_create_broker(broker_name, user_settings)

            if not broker:
                logger.error(f"Failed to create broker for strategy {strategy_id}")
                return None

            # Create executor
            executor = StrategyExecutor(strategy_id=strategy_id, db_session=db, broker_client=broker)

            return executor

        except Exception as e:
            logger.error(f"Error creating executor for strategy {strategy_id}: {e}")
            await db.close()
            return None

    async def _get_or_create_broker(self, broker_name: str, user_settings: UserSettings) -> Optional[BrokerClient]:
        """Get existing broker client or create new one"""

        if broker_name in self.brokers:
            return self.brokers[broker_name]

        try:
            broker = BrokerService.create_broker(broker_name)

            connected = await broker.connect(user_settings)

            if not connected:
                logger.error(f"Failed to connect to broker {broker_name}")
                return None

            self.brokers[broker_name] = broker

            logger.info(f"Created and connected to broker: {broker_name}")
            return broker

        except Exception as e:
            logger.error(f"Error creating broker {broker_name}: {e}")
            return None

    async def _stop_executor(self, strategy_id: int) -> bool:
        """Stop and remove an executor"""
        if strategy_id not in self.executors:
            logger.warning(f"Strategy {strategy_id} not in executors")
            return False

        executor = self.executors[strategy_id]

        try:
            await executor.stop()
            del self.executors[strategy_id]

            logger.info(f"Strategy {strategy_id} stopped and removed")
            return True

        except Exception as e:
            logger.error(f"Error stopping strategy {strategy_id}: {e}")
            return False

    async def _load_running_strategies(self):
        """Load all running strategies from database on startup"""
        db = self.db_session_factory()

        try:
            stmt = select(LiveStrategy).where(LiveStrategy.status == StrategyStatus.RUNNING)

            result = await db.execute(stmt)
            strategies = result.scalars().all()

            logger.info(f"Found {len(strategies)} running strategies in database")

            for strategy in strategies:
                await self.deploy_strategy(strategy.id)

        except Exception as e:
            logger.error(f"Error loading running strategies: {e}")

        finally:
            await db.close()

    async def _monitor_strategies(self):
        """
        Background task to monitor strategy health

        Runs every 5 minutes:
        - Checks if executors are still running
        - Restarts crashed executors
        - Cleans up stopped strategies
        """
        logger.info("Starting strategy monitor")

        while self.running:
            try:
                await asyncio.sleep(300)  # 5 minutes

                logger.debug(f"Monitoring {len(self.executors)} strategies")

                db = self.db_session_factory()

                try:
                    for strategy_id, executor in list(self.executors.items()):
                        stmt = select(LiveStrategy).where(LiveStrategy.id == strategy_id)

                        result = await db.execute(stmt)
                        strategy = result.scalars().first()

                        if not strategy:
                            logger.warning(f"Strategy {strategy_id} deleted, stopping executor")
                            await self._stop_executor(strategy_id)
                            continue

                        if strategy.status == StrategyStatus.STOPPED:
                            logger.info(f"Strategy {strategy_id} stopped, removing executor")
                            await self._stop_executor(strategy_id)

                        elif strategy.status == StrategyStatus.PAUSED:
                            if executor.is_running:
                                logger.info(f"Strategy {strategy_id} paused, pausing executor")
                                await executor.pause()

                        elif strategy.status == StrategyStatus.RUNNING:
                            if not executor.is_running:
                                logger.info(f"Strategy {strategy_id} should be running, restarting")
                                asyncio.create_task(executor.start())

                        elif strategy.status == StrategyStatus.ERROR:
                            logger.error(f"Strategy {strategy_id} in error state: {strategy.error_message}")
                            await self._stop_executor(strategy_id)

                finally:
                    await db.close()

            except Exception as e:
                logger.error(f"Error in strategy monitor: {e}")

        logger.info("Strategy monitor stopped")

    def get_executor_count(self) -> int:
        """Get number of active executors"""
        return len(self.executors)

    def get_executor_status(self) -> Dict[int, str]:
        """Get status of all executors"""
        return {strategy_id: "running" if executor.is_running else "stopped" for strategy_id, executor in self.executors.items()}


_execution_manager: Optional[ExecutionManager] = None


def get_execution_manager(db_session_factory) -> ExecutionManager:
    """Get or create global execution manager"""
    global _execution_manager

    if _execution_manager is None:
        _execution_manager = ExecutionManager(db_session_factory)

    return _execution_manager


async def start_execution_manager(db_session_factory):
    """Start the global execution manager (call on app startup)"""
    manager = get_execution_manager(db_session_factory)
    await manager.start()
    logger.info("Global execution manager started")


async def stop_execution_manager():
    """Stop the global execution manager (call on app shutdown)"""
    global _execution_manager

    if _execution_manager:
        await _execution_manager.stop()
        _execution_manager = None
        logger.info("Global execution manager stopped")
