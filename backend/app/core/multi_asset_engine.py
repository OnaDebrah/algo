"""
Multi-Asset Backtesting Engine
Backtest strategies across multiple symbols simultaneously
"""

import logging
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from backend.app.analytics import calculate_performance_metrics
from backend.app.core.data_fetcher import fetch_stock_data
from backend.app.core.database import DatabaseManager
from backend.app.core.risk_manager import RiskManager
from backend.app.strategies import BaseStrategy
from backend.app.schemas.backtest import MultiAssetBacktestResult

logger = logging.getLogger(__name__)


class MultiAssetEngine:
    """Engine for backtesting across multiple assets"""

    def __init__(
            self,
            strategies: Dict[str, BaseStrategy],  # {symbol: strategy}
            initial_capital: float = 100000,
            risk_manager: RiskManager = None,
            db: DatabaseManager = None,
            allocation_method: str = "equal",  # equal, optimized, custom
            commission_rate: float = 0.05,
            slippage_rate: float = 0.03,
    ):
        """
        Initialize multi-asset backtesting engine

        Args:
            strategies: Dictionary mapping symbols to strategies
            initial_capital: Starting capital
            risk_manager: Risk manager instance
            db: Database manager
            allocation_method: How to allocate capital across assets
        """
        self.strategies = strategies
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # {symbol: position_dict}
        self.trades = []
        self.equity_curve = []
        self.risk_manager = risk_manager or RiskManager()
        self.db = db or DatabaseManager()
        self.allocation_method = allocation_method

        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

        # Calculate capital allocation per symbol
        self._calculate_allocations()

    def _calculate_allocations(self):
        """Calculate capital allocation for each symbol"""
        num_symbols = len(self.strategies)

        if self.allocation_method == "equal":
            # Equal allocation
            self.allocations = {symbol: 1.0 / num_symbols for symbol in self.strategies.keys()}
        else:
            # Default to equal if other methods not implemented
            self.allocations = {symbol: 1.0 / num_symbols for symbol in self.strategies.keys()}

        logger.info(f"Capital allocations: {self.allocations}")

    def run_backtest(self, symbols: List[str], period: str, interval: str):
        """
        Run backtest across multiple assets

        Args:
            symbols: List of symbols to backtest
            period: Time period
            interval: Data interval
        """
        logger.info(
            f"Starting multi-asset backtest: {len(symbols)} symbols, " f"period: {period}, interval: {interval}")

        # Fetch data for all symbols
        data_dict = {}
        for symbol in symbols:
            data = fetch_stock_data(symbol, period, interval)
            if not data.empty:
                data_dict[symbol] = data
            else:
                logger.warning(f"No data for {symbol}, skipping")

        if not data_dict:
            logger.error("No data available for any symbol")
            return

        # Align all data to common dates
        aligned_data = self._align_data(data_dict)

        # Run backtest for each timestamp
        for i in range(len(aligned_data["dates"])):
            timestamp = aligned_data["dates"][i]

            # Process each symbol
            for symbol in aligned_data["symbols"]:
                if symbol not in self.strategies:
                    continue

                # Get data up to current point
                symbol_data = aligned_data["data"][symbol].iloc[: i + 1]
                current_price = symbol_data["Close"].iloc[-1]

                # Generate signal
                strategy = self.strategies[symbol]
                signal = strategy.generate_signal(symbol_data)

                # Execute trade
                self._execute_trade(symbol, signal, current_price, timestamp, strategy.name)

            # Calculate portfolio equity
            self._update_equity(timestamp, aligned_data, i)

        logger.info(
            f"Backtest complete: {len(self.trades)} trades, " f"Final equity: ${self.equity_curve[-1]['equity']:,.2f}")

    def _align_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """Align data across all symbols to common timestamps"""

        # Get common dates
        all_dates = None
        for data in data_dict.values():
            if all_dates is None:
                all_dates = set(data.index)
            else:
                all_dates = all_dates.intersection(set(data.index))

        common_dates = sorted(list(all_dates))

        # Align all data
        aligned = {"dates": common_dates, "symbols": list(data_dict.keys()), "data": {}}

        for symbol, data in data_dict.items():
            aligned["data"][symbol] = data.loc[common_dates]

        logger.info(f"Aligned data: {len(common_dates)} common dates")

        return aligned

    def _execute_trade(
            self,
            symbol: str,
            signal: int,
            current_price: float,
            timestamp,
            strategy_name: str,
    ):
        """Execute trade with commission and slippage"""

        symbol_allocation = self.allocations.get(symbol, 0)
        available_capital = self.cash * symbol_allocation

        # --- BUY SIGNAL ---
        if signal == 1 and symbol not in self.positions:
            # Apply Slippage (Paying more than current_price)
            execution_price = current_price * (1 + self.slippage_rate)

            quantity = self.risk_manager.calculate_position_size(available_capital, execution_price)

            # Calculate Costs
            trade_value = quantity * execution_price
            commission = trade_value * self.commission_rate
            total_cost = trade_value + commission

            if total_cost <= self.cash:
                self.cash -= total_cost
                self.positions[symbol] = {
                    "symbol": symbol,
                    "quantity": quantity,
                    "entry_price": execution_price,  # We track the "penalized" price
                    "entry_time": timestamp,
                    "strategy": strategy_name,
                    "entry_commission": commission
                }

                trade_data = {
                    "symbol": symbol,
                    "order_type": "BUY",
                    "quantity": quantity,
                    "price": execution_price,
                    "commission": commission,
                    "slippage_impact": execution_price - current_price,
                    "timestamp": (timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)),
                    "strategy": strategy_name,
                }

                self.trades.append(trade_data)
                self.db.save_trade(trade_data)
                logger.debug(f"BUY: {quantity} {symbol} @ ${execution_price:.2f} (Comm: ${commission:.2f})")

        # --- SELL SIGNAL ---
        elif signal == -1 and symbol in self.positions:
            position = self.positions[symbol]

            # Apply Slippage (Receiving less than current_price)
            execution_price = current_price * (1 - self.slippage_rate)

            # Calculate Costs
            trade_value = position["quantity"] * execution_price
            commission = trade_value * self.commission_rate

            # Net Cash Received
            net_proceeds = trade_value - commission
            self.cash += net_proceeds

            # Calculate P&L (Total commission includes entry + exit)
            total_commissions = position.get("entry_commission", 0) + commission
            profit = (trade_value - (position["quantity"] * position["entry_price"])) - total_commissions
            profit_pct = (profit / (position["quantity"] * position["entry_price"])) * 100

            trade_data = {
                "symbol": symbol,
                "order_type": "SELL",
                "quantity": position["quantity"],
                "price": execution_price,
                "commission": commission,
                "total_fees": total_commissions,
                "timestamp": (timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)),
                "strategy": strategy_name,
                "profit": profit,
                "profit_pct": profit_pct,
            }

            self.trades.append(trade_data)
            self.db.save_trade(trade_data)

            logger.debug(f"SELL: {position['quantity']} {symbol} @ ${execution_price:.2f} "
                         f"(Net P&L: ${profit:.2f}, Fees: ${total_commissions:.2f})")

            del self.positions[symbol]

    def _update_equity(self, timestamp, aligned_data: Dict, index: int):
        """Update portfolio equity"""

        equity = self.cash

        # Add value of all positions
        for symbol, position in self.positions.items():
            current_price = aligned_data["data"][symbol]["Close"].iloc[index]
            equity += position["quantity"] * current_price

        self.equity_curve.append(
            {
                "timestamp": timestamp,
                "equity": equity,
                "cash": self.cash,
                "num_positions": len(self.positions),
            }
        )

    def get_results(self) -> MultiAssetBacktestResult:
        """Get backtest results"""

        symbol_stats = self._calculate_symbol_stats()

        metrics = calculate_performance_metrics(self.trades, self.equity_curve, self.initial_capital)
        return MultiAssetBacktestResult(
            **metrics,
            num_symbols=len(self.strategies),
            symbol_stats=symbol_stats,
        )

    def _calculate_symbol_stats(self) -> Dict:
        """Calculate per-symbol statistics"""

        stats = {}

        # Check if we have any trades
        if not self.trades:
            # Return empty stats for all symbols
            for symbol in self.strategies.keys():
                stats[symbol] = {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "total_return": 0.0,
                    "win_rate": 0.0,
                    "avg_profit": 0.0,
                }
            return stats

        trades_df = pd.DataFrame(self.trades)

        # Safety check: ensure required columns exist
        if "symbol" not in trades_df.columns:
            logger.warning("No 'symbol' column in trades DataFrame")
            # Return empty stats for all symbols
            for symbol in self.strategies.keys():
                stats[symbol] = {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "total_return": 0.0,
                    "win_rate": 0.0,
                    "avg_profit": 0.0,
                }
            return stats

        for symbol in self.strategies.keys():
            symbol_trades = trades_df[trades_df["symbol"] == symbol]

            # Check if profit column exists before filtering
            if "profit" in symbol_trades.columns:
                completed = symbol_trades[symbol_trades["profit"].notna()]
            else:
                completed = pd.DataFrame()

            if not completed.empty:
                profits = completed["profit"]
                total_trades = len(completed)
                winning_trades = (profits > 0).sum()
                losing_trades = (profits < 0).sum()
                total_return = profits.sum()
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

                stats[symbol] = {
                    "total_trades": int(total_trades),
                    "winning_trades": int(winning_trades),
                    "losing_trades": int(losing_trades),
                    "total_return": float(total_return),
                    "win_rate": float(win_rate),
                    "avg_profit": float(profits.mean()),
                }
            else:
                stats[symbol] = {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "total_return": 0.0,
                    "win_rate": 0.0,
                    "avg_profit": 0.0,
                }

        return stats


class PortfolioBacktester:
    """Backtest a portfolio with single strategy across multiple assets"""

    def __init__(
            self,
            strategy: BaseStrategy,
            symbols: List[str],
            weights: Dict[str, float] = None,
            initial_capital: float = 100000,
    ):
        """
        Initialize portfolio backtester

        Args:
            strategy: Single strategy to apply to all symbols
            symbols: List of symbols
            weights: Optional custom weights {symbol: weight}
            initial_capital: Starting capital
        """
        self.strategy = strategy
        self.symbols = symbols
        self.initial_capital = initial_capital

        # Set equal weights if not provided
        if weights is None:
            self.weights = {s: 1.0 / len(symbols) for s in symbols}
        else:
            self.weights = weights

    def run(self, period: str, interval: str) -> Dict:
        """
        Run portfolio backtest

        Args:
            period: Time period
            interval: Data interval

        Returns:
            Results dictionary
        """
        # Create strategies dict (same strategy for all symbols)
        strategies = {symbol: self.strategy for symbol in self.symbols}

        # Create multi-asset engine
        engine = MultiAssetEngine(
            strategies=strategies,
            initial_capital=self.initial_capital,
            allocation_method="custom",
        )

        # Override allocations with provided weights
        engine.allocations = self.weights

        # Run backtest
        engine.run_backtest(self.symbols, period, interval)

        return engine.get_results().model_dump()
