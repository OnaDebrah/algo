"""
Live Trading Engine
Executes strategies in real-time with risk management
"""

import logging
import threading
import time
from datetime import datetime
from typing import Dict, Optional

from streamlit.alerts.alert_manager import AlertManager
from streamlit.core.data_fetcher import fetch_stock_data
from streamlit.core.database import DatabaseManager
from streamlit.core.risk_manager import RiskManager
from streamlit.live.orders import OrderSide, OrderType
from streamlit.strategies import BaseStrategy

from .base_broker import BaseBroker

logger = logging.getLogger(__name__)


class LiveTradingEngine:
    """Engine for executing live trading strategies"""

    def __init__(
        self,
        broker: BaseBroker,
        strategy: BaseStrategy,
        symbols: list,
        risk_manager: RiskManager,
        db: DatabaseManager,
        alert_manager: Optional[AlertManager] = None,
        check_interval: int = 60,  # seconds
    ):
        """
        Initialize live trading engine

        Args:
            broker: Broker interface
            strategy: Trading strategy
            symbols: List of symbols to trade
            risk_manager: Risk management
            db: Database manager
            alert_manager: Alert manager
            check_interval: How often to check signals (seconds)
        """
        self.broker = broker
        self.strategy = strategy
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.risk_manager = risk_manager
        self.db = db
        self.alert_manager = alert_manager
        self.check_interval = check_interval

        self.running = False
        self.thread = None
        self.last_check = {}
        self.active_positions = {}

    def start(self):
        """Start live trading"""
        if self.running:
            logger.warning("Live trading already running")
            return

        if not self.broker.connected:
            raise ConnectionError("Broker not connected")

        self.running = True
        self.thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.thread.start()

        logger.info(f"Live trading started - Strategy: {self.strategy.name}, " f"Symbols: {self.symbols}, Interval: {self.check_interval}s")

        if self.alert_manager:
            self.alert_manager.alert_system_error(f"Live trading started: {self.strategy.name}")

    def stop(self):
        """Stop live trading"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=10)

        logger.info("Live trading stopped")

        if self.alert_manager:
            self.alert_manager.alert_system_error(f"Live trading stopped: {self.strategy.name}")

    def _trading_loop(self):
        """Main trading loop"""
        while self.running:
            try:
                self._check_and_trade()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                if self.alert_manager:
                    self.alert_manager.alert_system_error(f"Trading loop error: {str(e)}")
                time.sleep(self.check_interval)

    def _check_and_trade(self):
        """Check signals and execute trades"""
        account = self.broker.get_account()

        # Check drawdown
        if self.risk_manager.check_drawdown(account["equity"]):
            logger.warning("Maximum drawdown exceeded, halting trading")
            self.stop()
            if self.alert_manager:
                self.alert_manager.alert_risk_event("Max Drawdown", "Trading halted due to drawdown limit")
            return

        for symbol in self.symbols:
            try:
                self._process_symbol(symbol, account)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

    def _process_symbol(self, symbol: str, account: Dict):
        """Process trading logic for a symbol"""
        # Get historical data
        data = fetch_stock_data(symbol, "1mo", "1d")
        if data.empty:
            logger.warning(f"No data available for {symbol}")
            return

        # Generate signal
        signal = self.strategy.generate_signal(data)

        if signal == 0:
            return  # No action

        current_price = float(data["Close"].iloc[-1])
        position = self.broker.get_position(symbol)

        # Buy signal
        if signal == 1 and position is None:
            quantity = self.risk_manager.calculate_position_size(account["equity"], current_price)

            # Check if we have enough buying power
            cost = quantity * current_price
            if cost > account["buying_power"]:
                logger.warning(f"Insufficient buying power for {symbol}: " f"Need ${cost:.2f}, Have ${account['buying_power']:.2f}")
                return

            # Place buy order
            try:
                order = self.broker.place_order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                )

                logger.info(f"BUY order placed: {quantity} {symbol} @ ${current_price:.2f}")

                # Save to database
                trade_data = {
                    "symbol": symbol,
                    "order_type": "BUY",
                    "quantity": quantity,
                    "price": current_price,
                    "timestamp": datetime.now().isoformat(),
                    "strategy": self.strategy.name,
                    "order_id": order["order_id"],
                }
                self.db.save_trade(trade_data)

                # Send alert
                if self.alert_manager:
                    self.alert_manager.alert_trade_executed(trade_data)

                # Track position
                self.active_positions[symbol] = {
                    "entry_price": current_price,
                    "quantity": quantity,
                    "entry_time": datetime.now(),
                }

            except Exception as e:
                logger.error(f"Failed to place buy order for {symbol}: {e}")
                if self.alert_manager:
                    self.alert_manager.alert_system_error(f"Buy order failed for {symbol}: {str(e)}")

        # Sell signal
        elif signal == -1 and position is not None:
            quantity = position["quantity"]

            # Place sell order
            try:
                order = self.broker.place_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                )

                # Calculate P&L
                entry_price = self.active_positions.get(symbol, {}).get("entry_price", position["entry_price"])
                profit = (current_price - entry_price) * quantity
                profit_pct = ((current_price - entry_price) / entry_price) * 100

                logger.info(f"SELL order placed: {quantity} {symbol} @ ${current_price:.2f} " f"(P&L: ${profit:.2f}, {profit_pct:.2f}%)")

                # Save to database
                trade_data = {
                    "symbol": symbol,
                    "order_type": "SELL",
                    "quantity": quantity,
                    "price": current_price,
                    "timestamp": datetime.now().isoformat(),
                    "strategy": self.strategy.name,
                    "profit": profit,
                    "profit_pct": profit_pct,
                    "order_id": order["order_id"],
                }
                self.db.save_trade(trade_data)

                # Send alert
                if self.alert_manager:
                    self.alert_manager.alert_position_closed(trade_data)

                # Remove from tracking
                if symbol in self.active_positions:
                    del self.active_positions[symbol]

            except Exception as e:
                logger.error(f"Failed to place sell order for {symbol}: {e}")
                if self.alert_manager:
                    self.alert_manager.alert_system_error(f"Sell order failed for {symbol}: {str(e)}")

    def get_status(self) -> Dict:
        """Get current status"""
        account = self.broker.get_account()
        positions = self.broker.get_positions()

        return {
            "running": self.running,
            "strategy": self.strategy.name,
            "symbols": self.symbols,
            "account": account,
            "positions": positions,
            "active_positions": len(positions),
            "check_interval": self.check_interval,
        }

    def close_all_positions(self):
        """Emergency: Close all positions"""
        logger.warning("Closing all positions...")

        positions = self.broker.get_positions()

        for position in positions:
            try:
                self.broker.place_order(
                    symbol=position["symbol"],
                    side=OrderSide.SELL,
                    quantity=position["quantity"],
                    order_type=OrderType.MARKET,
                )
                logger.info(f"Closed position: {position['symbol']}")
            except Exception as e:
                logger.error(f"Failed to close {position['symbol']}: {e}")

        if self.alert_manager:
            self.alert_manager.alert_system_error("All positions closed (emergency)")


class ScheduledTrading:
    """Schedule trading during market hours"""

    def __init__(
        self,
        engine: LiveTradingEngine,
        market_open: str = "09:30",
        market_close: str = "16:00",
        timezone: str = "America/New_York",
    ):
        """
        Initialize scheduled trading

        Args:
            engine: Live trading engine
            market_open: Market open time (HH:MM)
            market_close: Market close time (HH:MM)
            timezone: Timezone
        """
        self.engine = engine
        self.market_open = market_open
        self.market_close = market_close
        self.timezone = timezone
        self.scheduler_running = False
        self.scheduler_thread = None

    def start_scheduler(self):
        """Start the scheduler"""
        if self.scheduler_running:
            logger.warning("Scheduler already running")
            return

        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()

        logger.info(f"Trading scheduler started - Hours: {self.market_open} to {self.market_close}")

    def stop_scheduler(self):
        """Stop the scheduler"""
        self.scheduler_running = False
        if self.engine.running:
            self.engine.stop()

        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=10)

        logger.info("Trading scheduler stopped")

    def _scheduler_loop(self):
        """Scheduler loop"""
        while self.scheduler_running:
            if self._is_market_hours():
                if not self.engine.running:
                    logger.info("Market open - Starting trading")
                    self.engine.start()
            else:
                if self.engine.running:
                    logger.info("Market closed - Stopping trading")
                    self.engine.stop()

            time.sleep(60)  # Check every minute

    def _is_market_hours(self) -> bool:
        """Check if currently in market hours"""
        from datetime import datetime

        now = datetime.now()

        # Skip weekends
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Parse market hours
        open_time = datetime.strptime(self.market_open, "%H:%M").time()
        close_time = datetime.strptime(self.market_close, "%H:%M").time()
        current_time = now.time()

        return open_time <= current_time <= close_time
