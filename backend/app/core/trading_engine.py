"""
Main trading engine for backtesting
"""

import logging
from datetime import datetime
from typing import Dict, List

import pandas as pd

from backend.app.core import DatabaseManager, RiskManager
from backend.app.strategies import BaseStrategy
from config import DEFAULT_INITIAL_CAPITAL

logger = logging.getLogger(__name__)


class TradingEngine:
    """Main trading engine for backtesting"""

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        risk_manager: RiskManager = None,
        db_manager: DatabaseManager = None,
        commission_rate: float = 0.0,
        slippage_rate: float = 0.0,
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
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

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
            # Apply slippage (buy higher)
            slipped_price = current_price * (1 + self.slippage_rate)

            quantity = self.risk_manager.calculate_position_size(self.cash, slipped_price)
            trade_value = quantity * slipped_price
            commission = trade_value * self.commission_rate
            total_cost = trade_value + commission

            if total_cost <= self.cash:
                self.cash -= total_cost
                self.position = {
                    "symbol": symbol,
                    "quantity": quantity,
                    "entry_price": slipped_price,
                    "created_at": (timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)),
                }

                trade_data = {
                    "symbol": symbol,
                    "order_type": "BUY",
                    "quantity": quantity,
                    "price": slipped_price,
                    "commission": commission,
                    "slippage": slipped_price - current_price,
                    "total_value": total_cost,
                    "side": "BUY",
                    "executed_at": (timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)),
                    "strategy": self.strategy.name,
                }

                self.trades.append(trade_data)
                self.db.save_trade(trade_data)

                logger.info(f"BUY: {quantity} {symbol} @ ${slipped_price:.2f} " f"(Cost: ${total_cost:.2f}, Comm: ${commission:.2f})")

        # Sell signal
        elif signal == -1 and self.position is not None:
            # Apply slippage (sell lower)
            slipped_price = current_price * (1 - self.slippage_rate)

            revenue = self.position["quantity"] * slipped_price
            commission = revenue * self.commission_rate
            net_revenue = revenue - commission

            profit = net_revenue - (self.position["entry_price"] * self.position["quantity"])
            profit_pct = (profit / (self.position["entry_price"] * self.position["quantity"])) * 100

            self.cash += net_revenue

            trade_data = {
                "symbol": symbol,
                "order_type": "SELL",
                "quantity": self.position["quantity"],
                "price": slipped_price,
                "commission": commission,
                "slippage": current_price - slipped_price,
                "total_value": revenue,
                "side": "SELL",
                "executed_at": (timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)),
                "strategy": self.strategy.name,
                "profit": profit,
                "profit_pct": profit_pct,
            }

            self.trades.append(trade_data)
            self.db.save_trade(trade_data)

            logger.info(
                f"SELL: {self.position['quantity']} {symbol} @ ${slipped_price:.2f} "
                f"(P&L: ${profit:.2f}, {profit_pct:.2f}%, Comm: ${commission:.2f})"
            )

            self.position = None

        # Calculate equity
        equity = self.cash
        if self.position:
            equity += self.position["quantity"] * current_price

        self.equity_curve.append({"timestamp": timestamp, "equity": equity, "cash": self.cash})

    def run_backtest_loop(self, symbol: str, data: pd.DataFrame):
        """Original loop-based backtest (Fallback)"""
        logger.info(f"Starting LOOP backtest - Symbol: {symbol}, data: {len(data)}")
        for i in range(len(data)):
            current_data = data.iloc[: i + 1]
            signal = self.strategy.generate_signal(current_data)
            current_price = data["Close"].iloc[i]
            timestamp = data.index[i]
            self.execute_trade(symbol, signal, current_price, timestamp)

    def run_backtest_vectorized(self, symbol: str, data: pd.DataFrame):
        """Vectorized execution for 100x+ speedup"""
        import numpy as np

        logger.info(f"Starting VECTORIZED backtest - Symbol: {symbol}")

        # 1. Get raw signals
        signals = self.strategy.generate_signals_vectorized(data)

        # 2. Derive holding states (Long only parity with execute_trade)
        # 1 for Long, 0 for Cash
        positions = signals.replace(0, np.nan).ffill().fillna(0)
        positions = positions.clip(lower=0)

        # 3. Calculate Returns & Trades
        close = data["Close"]
        returns = close.pct_change()
        trades_mask = positions != positions.shift(1)
        entry_mask = (positions == 1) & (positions.shift(1) == 0)
        exit_mask = (positions == 0) & (positions.shift(1) == 1)

        # 4. Strategy Returns (Signal at t affects return at t+1)
        # Apply Risk Manager's position sizing factor (e.g. 0.1 for 10% exposure)
        risk_factor = getattr(self.risk_manager, "max_position_size", 1.0)
        strategy_returns = (positions.shift(1) * returns) * risk_factor

        # Apply slippage & commission approximation
        # Slippage/Commission applied on full position value (which is risk_factor % of portfolio)
        strategy_returns[entry_mask] -= self.slippage_rate * risk_factor
        strategy_returns[exit_mask] -= self.slippage_rate * risk_factor
        strategy_returns[trades_mask] -= self.commission_rate * risk_factor

        # 5. Calculate Equity
        strategy_returns = strategy_returns.fillna(0)
        cumulative_growth = (1 + strategy_returns).cumprod()
        equity_series = self.initial_capital * cumulative_growth

        # 6. Parity with UI expectations
        self.equity_curve = []
        for ts, eq in equity_series.items():
            self.equity_curve.append(
                {
                    "timestamp": ts,
                    "equity": float(eq),
                    "cash": float(self.initial_capital),  # Approximation
                }
            )

        # 7. Reconstruct trade list for UI (Only if indices exist)
        entry_indices = np.where(entry_mask)[0]
        exit_indices = np.where(exit_mask)[0]

        for i in range(min(len(entry_indices), len(exit_indices))):
            entry_idx = entry_indices[i]
            exit_idx = exit_indices[i]
            entry_price = float(close.iloc[entry_idx] * (1 + self.slippage_rate))
            exit_price = float(close.iloc[exit_idx] * (1 - self.slippage_rate))

            # Quantity based on risk manager formula: portfolio * size / price
            # We use initial capital as an approximation for vectorized trade quantity
            quantity = float(int((self.initial_capital * risk_factor) / entry_price))
            if quantity <= 0:
                quantity = 1.0

            # Reconstruct trade data for the database
            trade_data = {
                "symbol": symbol,
                "order_type": "BUY",
                "quantity": quantity,
                "price": entry_price,
                "side": "BUY",
                "executed_at": str(data.index[entry_idx]),
                "strategy": self.strategy.name,
                "total_value": entry_price * quantity,
                "commission": float(entry_price * quantity * self.commission_rate),
                "slippage": float(entry_price * quantity * self.slippage_rate),
            }
            self.trades.append(trade_data)

            profit = float((exit_price - entry_price) * quantity)
            profit_pct = float(((exit_price / entry_price) - 1) * 100)

            sell_data = {
                "symbol": symbol,
                "order_type": "SELL",
                "quantity": quantity,
                "price": exit_price,
                "side": "SELL",
                "executed_at": str(data.index[exit_idx]),
                "strategy": self.strategy.name,
                "profit": profit,
                "profit_pct": profit_pct,
                "total_value": exit_price * quantity,
                "commission": float(exit_price * quantity * self.commission_rate),
                "slippage": float(exit_price * quantity * self.slippage_rate),
            }
            self.trades.append(sell_data)

        logger.info(f"Vectorized backtest completed - Final equity: ${self.equity_curve[-1]['equity']:,.2f}")

    def run_backtest(self, symbol: str, data: pd.DataFrame):
        """
        Run backtest (dispatches to vectorized or loop)
        """
        try:
            # Check if strategy supports vectorized signals
            if hasattr(self.strategy, "generate_signals_vectorized"):
                return self.run_backtest_vectorized(symbol, data)
        except Exception as e:
            logger.warning(f"Vectorized backtest failed, falling back to loop: {e}")

        return self.run_backtest_loop(symbol, data)

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
