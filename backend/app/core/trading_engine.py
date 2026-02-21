"""
Main trading engine for backtesting
"""

import logging
from datetime import datetime
from typing import Dict, List

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import DEFAULT_INITIAL_CAPITAL
from backend.app.core import RiskManager
from backend.app.services.trading_service import TradingService
from backend.app.strategies import BaseStrategy

logger = logging.getLogger(__name__)


class TradingEngine:
    """Main trading engine for backtesting"""

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        risk_manager: RiskManager = None,
        trading_service: TradingService = None,
        commission_rate: float = 0.0,
        slippage_rate: float = 0.0,
        db: AsyncSession = None,
    ):
        """
        Initialize trading engine

        Args:
            strategy: Trading strategy to use
            initial_capital: Starting capital
            risk_manager: Risk management system
            trading_service: Database manager
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position = None
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []

        self.risk_manager = risk_manager or RiskManager()
        self.trading_service = trading_service or TradingService()
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.db = db
        logger.info(f"Trading engine initialized - Strategy: {strategy.name}, " f"Capital: ${initial_capital:,.2f}")

    async def execute_trade(self, symbol: str, signal: int, current_price: float, timestamp, start_timestamp=None):
        """
        Execute trade based on signal

        Args:
            symbol: Stock symbol
            signal: Trading signal (1: buy, -1: sell, 0: hold)
            current_price: Current market price
            timestamp: Trade timestamp
            start_timestamp: Optional timestamp to start actual trading/equity tracking
        """
        # Skip if before start_timestamp (Warm-up period)
        if start_timestamp and timestamp < start_timestamp:
            return

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
                await self.trading_service.save_trade(self.db, trade_data)

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
            await self.trading_service.save_trade(self.db, trade_data)

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

    async def run_backtest_loop(self, symbol: str, data: pd.DataFrame, start_timestamp=None):
        """Original loop-based backtest (Fallback)"""
        logger.info(f"Starting LOOP backtest - Symbol: {symbol}, data: {len(data)}, start: {start_timestamp}")
        for i in range(len(data)):
            current_data = data.iloc[: i + 1]
            signal = self.strategy.generate_signal(current_data)
            current_price = data["Close"].iloc[i]
            timestamp = data.index[i]
            await self.execute_trade(symbol, signal, current_price, timestamp, start_timestamp)

    def run_backtest_vectorized(self, symbol: str, data: pd.DataFrame, start_timestamp=None):
        """Vectorized execution for 100x+ speedup"""
        import numpy as np

        logger.info(f"Starting VECTORIZED backtest - Symbol: {symbol}, start: {start_timestamp}")

        # 1. Get raw signals
        signals = self.strategy.generate_signals_vectorized(data)

        # Apply start_timestamp mask
        if start_timestamp:
            signals.loc[signals.index < start_timestamp] = 0

        # ... (rest of vectorized logic)
        # 2. Derive holding states
        positions = signals.replace(0, np.nan).ffill().fillna(0)
        positions = positions.clip(lower=0)

        # 3. Calculate Returns & Trades
        close = data["Close"]
        returns = close.pct_change()
        trades_mask = positions != positions.shift(1)
        entry_mask = (positions == 1) & (positions.shift(1) == 0)
        exit_mask = (positions == 0) & (positions.shift(1) == 1)

        # 4. Strategy Returns
        # max_position_size may be a percentage (e.g. 20 for 20%) or fraction (e.g. 0.2)
        raw_position_size = getattr(self.risk_manager, "max_position_size", 1.0)
        risk_factor = raw_position_size / 100.0 if raw_position_size > 1 else raw_position_size
        strategy_returns = (positions.shift(1) * returns) * risk_factor

        strategy_returns[entry_mask] -= self.slippage_rate * risk_factor
        strategy_returns[exit_mask] -= self.slippage_rate * risk_factor
        strategy_returns[trades_mask] -= self.commission_rate * risk_factor

        # 5. Calculate Equity
        strategy_returns = strategy_returns.fillna(0)

        # Only start cumulative growth from start_timestamp
        if start_timestamp:
            mask = strategy_returns.index >= start_timestamp
            growth_slice = (1 + strategy_returns[mask]).cumprod()
            cumulative_growth = pd.Series(1.0, index=strategy_returns.index)
            cumulative_growth.update(growth_slice)
        else:
            cumulative_growth = (1 + strategy_returns).cumprod()

        equity_series = self.initial_capital * cumulative_growth

        # 6. Parity with UI expectations
        self.equity_curve = []
        for ts, eq in equity_series.items():
            # Only include points from start_timestamp
            if start_timestamp and ts < start_timestamp:
                continue
            self.equity_curve.append(
                {
                    "timestamp": ts,
                    "equity": float(eq),
                    "cash": float(self.initial_capital),  # Approximation
                }
            )

        # ... (reconstruct trade list based on mask)
        entry_indices = np.where(entry_mask)[0]
        exit_indices = np.where(exit_mask)[0]

        for i in range(min(len(entry_indices), len(exit_indices))):
            entry_idx = entry_indices[i]
            exit_idx = exit_indices[i]
            ts_entry = data.index[entry_idx]

            # Skip if before start_timestamp
            if start_timestamp and ts_entry < start_timestamp:
                continue

            entry_price = float(close.iloc[entry_idx] * (1 + self.slippage_rate))
            exit_price = float(close.iloc[exit_idx] * (1 - self.slippage_rate))

            quantity = float(int((self.initial_capital * risk_factor) / entry_price))
            if quantity <= 0:
                quantity = 1.0

            trade_data = {
                "symbol": symbol,
                "order_type": "BUY",
                "quantity": quantity,
                "price": entry_price,
                "side": "BUY",
                "executed_at": str(ts_entry),
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

        logger.info(f"Vectorized backtest completed - Final equity: ${self.equity_curve[-1]['equity'] if self.equity_curve else 0:,.2f}")

    async def run_backtest(self, symbol: str, data: pd.DataFrame, start_timestamp=None):
        """
        Run backtest (dispatches to vectorized or loop)
        """
        try:
            # Check if strategy supports vectorized signals
            if hasattr(self.strategy, "generate_signals_vectorized"):
                return self.run_backtest_vectorized(symbol, data, start_timestamp)
        except Exception as e:
            logger.warning(f"Vectorized backtest failed, falling back to loop: {e}")

        return await self.run_backtest_loop(symbol, data, start_timestamp)

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
