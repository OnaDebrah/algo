"""
Live Strategy Execution Engine
Executes trading strategies in real-time with broker integration
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from backend.app.models.live import LiveEquitySnapshot, LiveStrategy, LiveTrade, StrategyStatus, TradeSide, TradeStatus
from backend.app.strategies.base_strategy import BaseStrategy
from backend.app.strategies.strategy_catalog import get_catalog
from backend.app.websockets.manager import ws_manager

logger = logging.getLogger(__name__)


class Position:
    """Represents an open position"""

    def __init__(self, symbol: str, quantity: float, entry_price: float, side: str):
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.side = side  # 'LONG' or 'SHORT'
        self.current_price = entry_price
        self.unrealized_pnl = 0.0

    def update_price(self, current_price: float):
        """Update current price and calculate unrealized P&L"""
        self.current_price = current_price

        if self.side == "LONG":
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity

    @property
    def market_value(self) -> float:
        """Current market value of position"""
        return self.current_price * self.quantity


class StrategyExecutor:
    """
    Executes a single live strategy

    Responsibilities:
    - Fetch market data
    - Generate trading signals
    - Execute trades via broker
    - Track positions and P&L
    - Monitor risk limits
    """

    def __init__(
        self,
        strategy_id: int,
        db_session: Session,
        broker_client: Any,  # BrokerClient interface
    ):
        self.strategy_id = strategy_id
        self.db = db_session
        self.broker = broker_client

        # State
        self.strategy: Optional[LiveStrategy] = None
        self.strategy_instance: Optional[BaseStrategy] = None
        self.positions: Dict[str, Position] = {}
        self.cash: float = 0.0
        self.is_running = False

        # Performance tracking
        self.peak_equity = 0.0
        self.daily_start_equity = 0.0
        self.trades_today = 0

        # Load strategy from DB
        self._load_strategy()

    def _load_strategy(self):
        """Load strategy from database"""
        self.strategy = self.db.query(LiveStrategy).filter(LiveStrategy.id == self.strategy_id).first()

        if not self.strategy:
            raise ValueError(f"Strategy {self.strategy_id} not found")

        # Initialize cash
        self.cash = float(self.strategy.current_equity or self.strategy.initial_capital)
        self.peak_equity = self.cash
        self.daily_start_equity = self.cash

        # Create strategy instance
        catalog = get_catalog()
        self.strategy_instance = catalog.create_strategy(self.strategy.strategy_key, **self.strategy.parameters)

        logger.info(f"Loaded strategy {self.strategy_id}: {self.strategy.name}")

    async def start(self):
        """Start strategy execution"""
        logger.info(f"Starting strategy executor {self.strategy_id}")
        self.is_running = True

        # Update status in DB
        self.strategy.status = StrategyStatus.RUNNING
        self.strategy.started_at = datetime.now(timezone.utc)
        self.db.commit()

        # Main execution loop
        await self._execution_loop()

    async def stop(self):
        """Stop strategy execution"""
        logger.info(f"Stopping strategy executor {self.strategy_id}")
        self.is_running = False

        # Close all open positions
        await self._close_all_positions()

        # Update status in DB
        self.strategy.status = StrategyStatus.STOPPED
        self.strategy.stopped_at = datetime.now(timezone.utc)
        self.db.commit()

    async def pause(self):
        """Pause strategy execution"""
        logger.info(f"Pausing strategy executor {self.strategy_id}")
        self.is_running = False

        self.strategy.status = StrategyStatus.PAUSED
        self.db.commit()

    async def _execution_loop(self):
        """
        Main execution loop

        Runs every minute during market hours:
        1. Fetch latest market data
        2. Generate trading signal
        3. Execute trades if signal
        4. Update positions
        5. Check risk limits
        6. Save equity snapshot
        """
        while self.is_running:
            try:
                # Check if market is open
                if not await self._is_market_open():
                    logger.debug(f"Market closed, strategy {self.strategy_id} waiting...")
                    await asyncio.sleep(60)
                    continue

                # Refresh strategy from DB (check if paused/stopped)
                self.db.refresh(self.strategy)
                if self.strategy.status != StrategyStatus.RUNNING:
                    logger.info(f"Strategy {self.strategy_id} status changed to {self.strategy.status}")
                    break

                # Execute one iteration
                await self._execute_iteration()

                # Wait before next iteration (1 minute)
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Error in execution loop for strategy {self.strategy_id}: {e}")

                # Save error to DB
                self.strategy.error_message = str(e)
                self.strategy.status = StrategyStatus.ERROR
                self.db.commit()

                # Notify via WebSocket
                await ws_manager.broadcast_error(self.strategy_id, str(e))

                break

    async def _execute_iteration(self):
        """Execute one iteration of the strategy"""

        # 1. Fetch market data
        market_data = await self._fetch_market_data()

        if not market_data:
            logger.warning(f"No market data for strategy {self.strategy_id}")
            return

        # 2. Update position prices
        await self._update_positions(market_data)

        # 3. Generate trading signal
        signal = self.strategy_instance.generate_signal(market_data)

        # Normalize signal (handle both int and dict returns)
        if isinstance(signal, dict):
            signal_value = signal.get("signal", 0)
            position_size = signal.get("position_size", 1.0)
            metadata = signal.get("metadata", {})
        else:
            signal_value = signal
            position_size = 1.0
            metadata = {}

        # 4. Execute trades based on signal
        if signal_value != 0:
            await self._execute_signal(signal_value, position_size, metadata, market_data)

        # 5. Check risk limits (circuit breakers)
        risk_check = await self._check_risk_limits()
        if not risk_check["passed"]:
            logger.warning(f"Risk limit breach: {risk_check['reason']}")
            await self.pause()
            await ws_manager.broadcast_status_change(self.strategy_id, "running", "paused", f"Risk limit: {risk_check['reason']}")
            return

        # 6. Calculate and save equity snapshot
        await self._save_equity_snapshot()

        # 7. Update strategy metrics
        await self._update_metrics()

    async def _fetch_market_data(self) -> Optional[Dict[str, Any]]:
        """
        Fetch latest market data for strategy symbols

        Returns DataFrame-compatible dict with OHLCV data
        """
        try:
            # Fetch data for all symbols
            all_data = {}

            for symbol in self.strategy.symbols:
                # Get latest bars from broker
                bars = await self.broker.get_latest_bars(symbol, limit=100)

                if bars:
                    all_data[symbol] = bars

            return all_data

        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None

    async def _update_positions(self, market_data: Dict[str, Any]):
        """Update all positions with current market prices"""
        for symbol, position in self.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol]["close"][-1]
                position.update_price(current_price)

    async def _execute_signal(self, signal: int, position_size: float, metadata: Dict[str, Any], market_data: Dict[str, Any]):
        """
        Execute trading signal

        Signal values:
        - 1: Buy/Long
        - -1: Sell/Short
        - 0: Hold (no action)
        """
        symbol = self.strategy.symbols[0]  # Primary symbol
        current_price = market_data[symbol]["close"][-1]

        # Check if we already have a position
        has_position = symbol in self.positions

        if signal == 1:  # BUY signal
            if not has_position:
                await self._open_long_position(symbol, current_price, position_size, metadata)

        elif signal == -1:  # SELL signal
            if has_position:
                await self._close_position(symbol, current_price, metadata)
            else:
                # Could open short position if enabled
                if self._is_shorting_enabled():
                    await self._open_short_position(symbol, current_price, position_size, metadata)

    async def _open_long_position(self, symbol: str, price: float, position_size: float, metadata: Dict[str, Any]):
        """Open a long position"""

        # Calculate position size
        quantity = self._calculate_position_size(symbol, price, position_size)

        if quantity <= 0:
            logger.warning(f"Invalid quantity {quantity} for {symbol}")
            return

        # Check if we have enough cash
        total_cost = quantity * price
        if total_cost > self.cash:
            logger.warning(f"Insufficient cash: need ${total_cost:.2f}, have ${self.cash:.2f}")
            return

        # Place order with broker
        order = await self.broker.place_order(symbol=symbol, side="buy", quantity=quantity, order_type="market")

        if not order or order.get("status") == "rejected":
            logger.error(f"Order rejected for {symbol}")
            return

        # Update cash
        commission = quantity * price * 0.001  # 0.1% commission
        self.cash -= total_cost + commission

        # Create position
        position = Position(symbol=symbol, quantity=quantity, entry_price=price, side="LONG")
        self.positions[symbol] = position

        # Save trade to database
        trade = LiveTrade(
            strategy_id=self.strategy_id,
            symbol=symbol,
            side=TradeSide.BUY,
            quantity=quantity,
            entry_price=price,
            order_id=order.get("order_id"),
            status=TradeStatus.OPEN,
            commission=commission,
            opened_at=datetime.now(timezone.utc),
            strategy_signal=metadata,
        )

        self.db.add(trade)
        self.db.commit()

        # Update strategy metrics
        self.strategy.total_trades += 1
        self.strategy.daily_trades += 1
        self.trades_today += 1
        self.strategy.last_trade_at = datetime.now(timezone.utc)
        self.db.commit()

        # Broadcast trade execution
        await ws_manager.broadcast_trade_executed(
            self.strategy_id,
            {"symbol": symbol, "side": "BUY", "quantity": quantity, "price": price, "timestamp": datetime.now(timezone.utc).isoformat()},
        )

        logger.info(f"Opened LONG position: {symbol} {quantity} @ ${price:.2f}")

    async def _close_position(self, symbol: str, price: float, metadata: Dict[str, Any]):
        """Close an open position"""

        if symbol not in self.positions:
            logger.warning(f"No position to close for {symbol}")
            return

        position = self.positions[symbol]

        # Place sell order
        order = await self.broker.place_order(symbol=symbol, side="sell", quantity=position.quantity, order_type="market")

        if not order or order.get("status") == "rejected":
            logger.error(f"Close order rejected for {symbol}")
            return

        # Calculate P&L
        if position.side == "LONG":
            profit = (price - position.entry_price) * position.quantity
        else:  # SHORT
            profit = (position.entry_price - price) * position.quantity

        commission = position.quantity * price * 0.001
        net_profit = profit - commission

        # Update cash
        self.cash += position.quantity * price - commission

        # Find and update trade in database
        trade = (
            self.db.query(LiveTrade)
            .filter(LiveTrade.strategy_id == self.strategy_id, LiveTrade.symbol == symbol, LiveTrade.status == TradeStatus.OPEN)
            .first()
        )

        if trade:
            trade.exit_price = price
            trade.closed_at = datetime.now(timezone.utc)
            trade.status = TradeStatus.CLOSED
            trade.profit = net_profit
            trade.profit_pct = (profit / (position.entry_price * position.quantity)) * 100
            trade.commission += commission
            self.db.commit()

            # Update win/loss counters
            if net_profit > 0:
                self.strategy.winning_trades += 1
            else:
                self.strategy.losing_trades += 1

        # Remove position
        del self.positions[symbol]

        # Update strategy total return
        self.strategy.total_return += net_profit
        self.db.commit()

        # Broadcast trade execution
        await ws_manager.broadcast_trade_executed(
            self.strategy_id,
            {
                "symbol": symbol,
                "side": "SELL",
                "quantity": position.quantity,
                "price": price,
                "profit": net_profit,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        logger.info(f"Closed position: {symbol} P&L: ${net_profit:.2f}")

    async def _open_short_position(self, symbol: str, price: float, position_size: float, metadata: Dict[str, Any]):
        """Open a short position"""
        # Similar to _open_long_position but for shorting
        # Implementation depends on broker support
        logger.info(f"Short selling not yet implemented for {symbol}")

    async def _close_all_positions(self):
        """Close all open positions (emergency stop)"""
        logger.info(f"Closing all positions for strategy {self.strategy_id}")

        # Get current market prices
        market_data = await self._fetch_market_data()

        if not market_data:
            logger.error("Cannot close positions: no market data")
            return

        # Close each position
        for symbol in list(self.positions.keys()):
            if symbol in market_data:
                current_price = market_data[symbol]["close"][-1]
                await self._close_position(symbol, current_price, {})

    def _calculate_position_size(self, symbol: str, price: float, position_size_pct: float) -> float:
        """
        Calculate position size based on:
        - Available cash
        - Max position % limit
        - Position size from signal
        - Risk per trade (stop loss)
        """
        # Get current equity
        equity = self._calculate_equity()

        # Apply max position constraint
        max_position_value = equity * (float(self.strategy.max_position_pct) / 100.0)

        # Apply signal position size
        target_value = max_position_value * position_size_pct

        # Cannot exceed available cash
        target_value = min(target_value, self.cash)

        # Calculate shares
        quantity = int(target_value / price)

        return quantity

    def _calculate_equity(self) -> float:
        """Calculate current total equity (cash + positions)"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value

    async def _check_risk_limits(self) -> Dict[str, Any]:
        """
        Check risk limits / circuit breakers

        Returns: {
            'passed': bool,
            'reason': str
        }
        """
        equity = self._calculate_equity()

        # 1. Daily loss limit
        if self.strategy.daily_loss_limit:
            daily_pnl = equity - self.daily_start_equity
            if daily_pnl < -float(self.strategy.daily_loss_limit):
                return {"passed": False, "reason": f"Daily loss limit exceeded: ${abs(daily_pnl):.2f}"}

        # 2. Max drawdown limit
        if equity < self.peak_equity:
            drawdown_pct = ((self.peak_equity - equity) / self.peak_equity) * 100

            if drawdown_pct > float(self.strategy.max_drawdown_limit or 20.0):
                return {"passed": False, "reason": f"Max drawdown exceeded: {drawdown_pct:.2f}%"}

        # 3. Too many trades in one day
        if self.trades_today > 50:
            return {"passed": False, "reason": f"Excessive trading: {self.trades_today} trades today"}

        # All checks passed
        return {"passed": True, "reason": ""}

    async def _save_equity_snapshot(self):
        """Save current equity snapshot to database and broadcast"""
        equity = self._calculate_equity()
        positions_value = sum(pos.market_value for pos in self.positions.values())

        # Calculate P&L
        daily_pnl = equity - self.daily_start_equity
        total_pnl = equity - float(self.strategy.initial_capital)

        # Calculate drawdown
        drawdown_pct = 0.0
        if equity < self.peak_equity:
            drawdown_pct = ((self.peak_equity - equity) / self.peak_equity) * 100
        else:
            self.peak_equity = equity

        # Save to database
        snapshot = LiveEquitySnapshot(
            strategy_id=self.strategy_id,
            timestamp=datetime.now(timezone.utc),
            equity=equity,
            cash=self.cash,
            positions_value=positions_value,
            daily_pnl=daily_pnl,
            total_pnl=total_pnl,
            drawdown_pct=drawdown_pct,
        )

        self.db.add(snapshot)
        self.db.commit()

        # Update strategy
        self.strategy.current_equity = equity
        self.strategy.daily_pnl = daily_pnl
        self.strategy.last_equity_update = datetime.now(timezone.utc)
        self.db.commit()

        # Broadcast to WebSocket clients
        await ws_manager.broadcast_equity_update(
            strategy_id=self.strategy_id,
            equity=equity,
            cash=self.cash,
            positions_value=positions_value,
            daily_pnl=daily_pnl,
            total_pnl=total_pnl,
            drawdown_pct=drawdown_pct,
        )

    async def _update_metrics(self):
        """Update strategy performance metrics"""
        equity = self._calculate_equity()

        # Update return
        self.strategy.total_return = equity - float(self.strategy.initial_capital)
        self.strategy.total_return_pct = (equity - float(self.strategy.initial_capital)) / float(self.strategy.initial_capital) * 100

        # Update max drawdown
        if equity < self.peak_equity:
            drawdown = ((self.peak_equity - equity) / self.peak_equity) * 100
            if drawdown > (self.strategy.max_drawdown or 0):
                self.strategy.max_drawdown = drawdown

        # Calculate Sharpe ratio (simplified - needs more data)
        # TODO: Implement proper Sharpe calculation with returns history

        self.db.commit()

    async def _is_market_open(self) -> bool:
        """Check if market is currently open"""
        # Check with broker
        return await self.broker.is_market_open()

    def _is_shorting_enabled(self) -> bool:
        """Check if short selling is enabled for this strategy"""
        # Can be a strategy parameter
        return self.strategy.parameters.get("allow_shorting", False)
