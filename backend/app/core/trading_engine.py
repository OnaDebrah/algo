"""
Main trading engine for backtesting
"""

import logging
from datetime import datetime
from typing import Dict, List

import pandas as pd

from config import DEFAULT_INITIAL_CAPITAL
from core.database import DatabaseManager
from core.risk_manager import RiskManager
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class TradingEngine:
    """Main trading engine for backtesting"""

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        risk_manager: RiskManager = None,
        db_manager: DatabaseManager = None,
    ):
        """
        Initialize trading engine

        Args:
            strategy: Trading strategy to use
            initial_capital: Starting capital
            risk_manager: Risk management system
            db_manager: Database manager
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position = None
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []

        self.risk_manager = risk_manager or RiskManager()
        self.db = db_manager or DatabaseManager()

        logger.info(f"Trading engine initialized - Strategy: {strategy.name}, " f"Capital: ${initial_capital:,.2f}")

    def execute_trade(self, symbol: str, signal: int, current_price: float, timestamp):
        """
        Execute trade based on signal

        Args:
            symbol: Stock symbol
            signal: Trading signal (1: buy, -1: sell, 0: hold)
            current_price: Current market price
            timestamp: Trade timestamp
        """
        # Buy signal
        if signal == 1 and self.position is None:
            quantity = self.risk_manager.calculate_position_size(self.cash, current_price)
            cost = quantity * current_price

            if cost <= self.cash:
                self.cash -= cost
                self.position = {
                    "symbol": symbol,
                    "quantity": quantity,
                    "entry_price": current_price,
                    "entry_time": timestamp,
                }

                trade_data = {
                    "symbol": symbol,
                    "order_type": "BUY",
                    "quantity": quantity,
                    "price": current_price,
                    "timestamp": (timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)),
                    "strategy": self.strategy.name,
                }

                self.trades.append(trade_data)
                self.db.save_trade(trade_data)

                logger.info(f"BUY: {quantity} {symbol} @ ${current_price:.2f} " f"(Cost: ${cost:.2f})")

        # Sell signal
        elif signal == -1 and self.position is not None:
            profit = (current_price - self.position["entry_price"]) * self.position["quantity"]
            profit_pct = ((current_price - self.position["entry_price"]) / self.position["entry_price"]) * 100

            self.cash += self.position["quantity"] * current_price

            trade_data = {
                "symbol": symbol,
                "order_type": "SELL",
                "quantity": self.position["quantity"],
                "price": current_price,
                "timestamp": (timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)),
                "strategy": self.strategy.name,
                "profit": profit,
                "profit_pct": profit_pct,
            }

            self.trades.append(trade_data)
            self.db.save_trade(trade_data)

            logger.info(f"SELL: {self.position['quantity']} {symbol} @ ${current_price:.2f} " f"(P&L: ${profit:.2f}, {profit_pct:.2f}%)")

            self.position = None

        # Calculate equity
        equity = self.cash
        if self.position:
            equity += self.position["quantity"] * current_price

        self.equity_curve.append({"timestamp": timestamp, "equity": equity, "cash": self.cash})

    def run_backtest(self, symbol: str, data: pd.DataFrame):
        """
        Run backtest on historical data

        Args:
            symbol: Stock symbol
            data: Historical OHLCV data
        """
        logger.info(f"Starting backtest - Symbol: {symbol}, " f"Data points: {len(data)}, Period: {data.index[0]} to {data.index[-1]}")

        for i in range(len(data)):
            current_data = data.iloc[: i + 1]
            signal = self.strategy.generate_signal(current_data)
            current_price = data["Close"].iloc[i]
            timestamp = data.index[i]

            self.execute_trade(symbol, signal, current_price, timestamp)

        logger.info(f"Backtest completed - Total trades: {len(self.trades)}, " f"Final equity: ${self.equity_curve[-1]['equity']:,.2f}")

    def get_current_position(self) -> Dict:
        """Get current position details"""
        return self.position

    def get_portfolio_value(self, current_price: float = None) -> float:
        """
        Get current portfolio value

        Args:
            current_price: Current market price (if position exists)

        Returns:
            Total portfolio value
        """
        value = self.cash
        if self.position and current_price:
            value += self.position["quantity"] * current_price
        return value

    def reset(self):
        """Reset engine to initial state"""
        self.cash = self.initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = []
        self.risk_manager.reset_peak()
        logger.info("Trading engine reset")
