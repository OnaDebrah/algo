"""
Automated execution system for hedge recommendations
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from alerts import AlertManager
from models.options_position import HedgeExecution
from schemas.alert import AlertCategory, AlertLevel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.data.providers.providers import ProviderFactory
from ...models.portfolio import Portfolio
from ...models.position import Position
from ...models.user import User
from ...services.analysis.hedge_service import HedgeRecommendationService
from ..brokers.broker_service import BrokerService

logger = logging.getLogger(__name__)


class AutoHedgeExecutor:
    """
    Automated hedge execution based on ML predictions
    """

    def __init__(self, provider_factory: ProviderFactory, broker_service: BrokerService, db_session: AsyncSession):
        self.provider_factory = provider_factory
        self.broker = broker_service
        self.db = db_session
        self.hedge_service = HedgeRecommendationService()

        # Execution settings
        self.min_confidence = 0.6
        self.max_position_size = 0.3  # Max 30% of portfolio in hedges
        self.rebalance_frequency = timedelta(days=1)
        self.last_execution = {}

    async def monitor_and_execute(self, user_id: int):
        """
        Main monitoring loop - call this periodically
        """
        logger.info(f"Starting hedge monitor for user {user_id}")

        # Get user and portfolio
        user = await self._get_user(user_id)
        portfolio = await self._get_portfolio(user_id)

        if not portfolio:
            logger.warning(f"No portfolio found for user {user_id}")
            return

        # Get current ML predictions
        recommendation = await self.hedge_service.get_hedge_recommendation(
            user=user, db=self.db, portfolio_value=float(portfolio.total_value), portfolio_beta=portfolio.beta or 1.0
        )

        # Check if execution is needed
        if await self._should_execute(user_id, recommendation):
            await self._execute_hedge(user_id, portfolio, recommendation)

    async def _should_execute(self, user_id: int, recommendation: Dict) -> bool:
        """Determine if hedge should be executed"""

        # Check confidence
        ml_signals = recommendation.get("ml_signals", {})
        confidence = ml_signals.get("lppls_confidence", 0)

        if confidence < self.min_confidence:
            logger.info(f"Confidence {confidence} below threshold {self.min_confidence}")
            return False

        # Check last execution time
        last_exec = self.last_execution.get(user_id)
        if last_exec and datetime.now() - last_exec < self.rebalance_frequency:
            logger.info(f"Last execution too recent: {last_exec}")
            return False

        # Check if we already have hedges
        existing_hedges = await self._get_existing_hedges(user_id)

        # If we have hedges and risk is decreasing, maybe exit
        if existing_hedges:
            stress_trend = ml_signals.get("lstm_trend", "stable")
            if stress_trend == "decreasing" and confidence < 0.4:
                logger.info("Risk decreasing, consider exiting hedges")
                return True

        # If no hedges and risk is high, execute
        crash_prob = ml_signals.get("combined_probability", 0)
        if not existing_hedges and crash_prob > 0.5:
            logger.info(f"High crash probability {crash_prob:.1%}, executing hedge")
            return True

        return False

    async def _execute_hedge(self, user_id: int, portfolio: Portfolio, recommendation: Dict):
        """Execute hedge trades"""

        strategy = recommendation.get("strategy")
        details = recommendation.get("details", {})

        logger.info(f"Executing {strategy} hedge for user {user_id}")

        try:
            if strategy == "put_spread":
                await self._execute_put_spread(user_id, portfolio, details)
            elif strategy == "tail_risk":
                await self._execute_tail_risk_hedge(user_id, portfolio, details)
            elif strategy == "covered_calls":
                await self._execute_covered_calls(user_id, portfolio, details)
            elif strategy == "collar":
                await self._execute_collar(user_id, portfolio, details)

            self.last_execution[user_id] = datetime.now()

            await self._log_execution(user_id, recommendation)

        except Exception as e:
            logger.error(f"Hedge execution failed: {e}")
            await self._send_alert(user_id, f"Hedge execution failed: {e}")

    async def _execute_put_spread(self, user_id: int, portfolio: Portfolio, details: Dict):
        """Execute put spread hedge"""

        option_suggestions = details.get("option_suggestions", {})
        contracts = option_suggestions.get("contracts_needed", 0)
        strikes = option_suggestions.get("strikes", {})
        expiry = option_suggestions.get("expiry")

        if contracts <= 0:
            logger.warning("No contracts to execute")
            return

        # Buy ATM puts
        buy_order = await self.broker.place_option_order(
            user_id=user_id,
            symbol="SPY",
            option_type="put",
            strike=strikes.get("moderate", 0),
            expiration=expiry,
            action="buy",
            quantity=int(contracts),
        )
        logger.info(f"BUY order placed: {buy_order}")

        # Sell OTM puts to reduce cost
        sell_order = await self.broker.place_option_order(
            user_id=user_id,
            symbol="SPY",
            option_type="put",
            strike=strikes.get("conservative", 0),
            expiration=expiry,
            action="sell",
            quantity=int(contracts),
        )
        logger.info(f"Sell order placed: {sell_order}")

        logger.info(f"Put spread executed: Bought {contracts} ATM puts, sold {contracts} OTM puts")

    async def _execute_tail_risk_hedge(self, user_id: int, portfolio: Portfolio, details: Dict):
        """Execute tail risk hedge with VIX calls"""

        option_suggestions = details.get("option_suggestions", {})
        contracts = option_suggestions.get("contracts_needed", 0)

        # Buy OTM puts
        put_order = await self.broker.place_option_order(
            user_id=user_id,
            symbol="SPY",
            option_type="put",
            strike=option_suggestions.get("strikes", {}).get("aggressive", 0),
            expiration=option_suggestions.get("expiry"),
            action="buy",
            quantity=int(contracts),
        )
        logger.info(f"Put order placed: {put_order}")

        # Buy VIX calls for additional tail protection
        vix_order = await self.broker.place_option_order(
            user_id=user_id,
            symbol="VIX",
            option_type="call",
            strike=60,  # Far OTM VIX calls
            expiration=(datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
            action="buy",
            quantity=int(contracts * 0.5),  # Half position in VIX
        )
        logger.info(f"VIX order placed: {vix_order}")

        logger.info(f"Tail risk hedge executed: {contracts} OTM puts, {int(contracts*0.5)} VIX calls")

    async def _execute_covered_calls(self, user_id: int, portfolio: Portfolio, details: Dict):
        """Execute covered calls on existing positions"""

        # Get current stock positions
        positions = await self._get_stock_positions(user_id)

        for position in positions:
            if position.quantity > 0:
                # Sell OTM calls
                await self.broker.place_option_order(
                    user_id=user_id,
                    symbol=position.symbol,
                    option_type="call",
                    strike=position.current_price * 1.05,  # 5% OTM
                    expiration=(datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
                    action="sell",
                    quantity=int(position.quantity / 100),  # 1 contract per 100 shares
                )

        logger.info(f"Covered calls executed on {len(positions)} positions")

    async def _execute_collar(self, user_id: int, portfolio: Portfolio, details: Dict):
        """Execute collar strategy (buy put, sell call)"""

        # Get largest positions
        positions = await self._get_stock_positions(user_id)
        positions = sorted(positions, key=lambda x: x.market_value, reverse=True)[:5]

        for position in positions:
            contracts = int(position.quantity / 100)
            if contracts > 0:
                # Buy protective put
                await self.broker.place_option_order(
                    user_id=user_id,
                    symbol=position.symbol,
                    option_type="put",
                    strike=position.current_price * 0.95,  # 5% OTM put
                    expiration=(datetime.now() + timedelta(days=60)).strftime("%Y-%m-%d"),
                    action="buy",
                    quantity=contracts,
                )

                # Sell covered call to offset cost
                await self.broker.place_option_order(
                    user_id=user_id,
                    symbol=position.symbol,
                    option_type="call",
                    strike=position.current_price * 1.10,  # 10% OTM call
                    expiration=(datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
                    action="sell",
                    quantity=contracts,
                )

        logger.info(f"Collar executed on {len(positions)} positions")

    async def _get_user(self, user_id: int) -> User:
        """Get user from database"""
        from ...models.user import User

        result = await self.db.execute(select(User).where(User.id == user_id))
        return result.scalars().first()

    async def _get_portfolio(self, user_id: int) -> Optional[Portfolio]:
        """Get user's portfolio"""
        from ...models.portfolio import Portfolio

        result = await self.db.execute(select(Portfolio).where(Portfolio.user_id == user_id))
        return result.scalars().first()

    async def _get_existing_hedges(self, user_id: int) -> List[Position]:
        """Get existing hedge positions"""
        from ...models.position import Position

        result = await self.db.execute(select(Position).where(Position.user_id == user_id, Position.asset_type.in_(["option", "future"])))
        return result.scalars().all()

    async def _get_stock_positions(self, user_id: int) -> List[Position]:
        """Get stock positions"""
        from ...models.position import Position

        result = await self.db.execute(select(Position).where(Position.user_id == user_id, Position.asset_type == "stock"))
        return result.scalars().all()

    async def _log_execution(self, user_id: int, recommendation: Dict):
        """Log hedge execution to database"""

        execution = HedgeExecution(
            user_id=user_id,
            timestamp=datetime.now(),
            strategy=recommendation.get("strategy"),
            crash_probability=recommendation.get("ml_signals", {}).get("combined_probability"),
            confidence=recommendation.get("ml_signals", {}).get("lppls_confidence"),
            cost=recommendation.get("cost", 0),
            details=recommendation,
        )

        self.db.add(execution)
        await self.db.commit()

    async def _send_alert(self, user_id: int, message: str):
        """Send alert to user"""

        manager = AlertManager()
        await manager.send_alert(
            user_id=user_id, level=AlertLevel.INFO, title="Hedge Execution Alert", message=message, category=AlertCategory.HEDGE_EXECUTION
        )
