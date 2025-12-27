"""
Multi-Asset Backtesting Engine
Backtest strategies across multiple symbols simultaneously
"""

import logging
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from core.data_fetcher import fetch_stock_data
from core.database import DatabaseManager
from core.risk_manager import RiskManager
from strategies.base_strategy import BaseStrategy

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
        logger.info(f"Starting multi-asset backtest: {len(symbols)} symbols, " f"period: {period}, interval: {interval}")

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

        logger.info(f"Backtest complete: {len(self.trades)} trades, " f"Final equity: ${self.equity_curve[-1]['equity']:,.2f}")

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
        """Execute trade for a specific symbol"""

        # Calculate available capital for this symbol
        symbol_allocation = self.allocations.get(symbol, 0)
        available_capital = self.cash * symbol_allocation

        # Buy signal
        if signal == 1 and symbol not in self.positions:
            quantity = self.risk_manager.calculate_position_size(available_capital, current_price)
            cost = quantity * current_price

            if cost <= self.cash:
                self.cash -= cost
                self.positions[symbol] = {
                    "symbol": symbol,
                    "quantity": quantity,
                    "entry_price": current_price,
                    "entry_time": timestamp,
                    "strategy": strategy_name,
                }

                trade_data = {
                    "symbol": symbol,
                    "order_type": "BUY",
                    "quantity": quantity,
                    "price": current_price,
                    "timestamp": (timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)),
                    "strategy": strategy_name,
                }

                self.trades.append(trade_data)
                self.db.save_trade(trade_data)

                logger.debug(f"BUY: {quantity} {symbol} @ ${current_price:.2f}")

        # Sell signal
        elif signal == -1 and symbol in self.positions:
            position = self.positions[symbol]
            profit = (current_price - position["entry_price"]) * position["quantity"]
            profit_pct = ((current_price - position["entry_price"]) / position["entry_price"]) * 100

            self.cash += position["quantity"] * current_price

            trade_data = {
                "symbol": symbol,
                "order_type": "SELL",
                "quantity": position["quantity"],
                "price": current_price,
                "timestamp": (timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)),
                "strategy": strategy_name,
                "profit": profit,
                "profit_pct": profit_pct,
            }

            self.trades.append(trade_data)
            self.db.save_trade(trade_data)

            logger.debug(f"SELL: {position['quantity']} {symbol} @ ${current_price:.2f} " f"(P&L: ${profit:.2f})")

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

    # def get_results(self) -> Dict:
    #     """Get backtest results"""
    #
    #     if not self.equity_curve:
    #         return {}
    #
    #     final_equity = self.equity_curve[-1]["equity"]
    #     total_return = ((final_equity - self.initial_capital) / self.initial_capital) * 100
    #
    #     # Per-symbol statistics
    #     symbol_stats = self._calculate_symbol_stats()
    #
    #     # Overall statistics
    #     trades_df = pd.DataFrame(self.trades)
    #     completed_trades = trades_df[trades_df["profit"].notna()]
    #
    #     if not completed_trades.empty:
    #         win_rate = (completed_trades["profit"] > 0).sum() / len(completed_trades) * 100
    #         avg_profit = completed_trades["profit"].mean()
    #
    #         returns = completed_trades["profit_pct"].values
    #         if len(returns) > 1:
    #             sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
    #         else:
    #             sharpe = 0
    #     else:
    #         win_rate = 0
    #         avg_profit = 0
    #         sharpe = 0
    #
    #     # Max drawdown
    #     equity_values = [e["equity"] for e in self.equity_curve]
    #     peak = equity_values[0]
    #     max_dd = 0
    #
    #     for value in equity_values:
    #         if value > peak:
    #             peak = value
    #         dd = ((peak - value) / peak) * 100
    #         if dd > max_dd:
    #             max_dd = dd
    #
    #     return {
    #         "total_return": total_return,
    #         "win_rate": win_rate,
    #         "sharpe_ratio": sharpe,
    #         "max_drawdown": max_dd,
    #         "total_trades": len(completed_trades),
    #         "avg_profit": avg_profit,
    #         "final_equity": final_equity,
    #         "symbol_stats": symbol_stats,
    #         "num_symbols": len(self.strategies),
    #     }

    # def _calculate_symbol_stats(self) -> Dict:
    #     """Calculate per-symbol statistics"""
    #
    #     stats = {}
    #     trades_df = pd.DataFrame(self.trades)
    #
    #     for symbol in self.strategies.keys():
    #         symbol_trades = trades_df[trades_df["symbol"] == symbol]
    #         completed = symbol_trades[symbol_trades["profit"].notna()]
    #
    #         if not completed.empty:
    #             total_profit = completed["profit"].sum()
    #             num_trades = len(completed)
    #             win_rate = (completed["profit"] > 0).sum() / num_trades * 100
    #
    #             stats[symbol] = {
    #                 "total_profit": total_profit,
    #                 "num_trades": num_trades,
    #                 "win_rate": win_rate,
    #                 "avg_profit": completed["profit"].mean(),
    #                 "strategy": self.strategies[symbol].name,
    #             }
    #         else:
    #             stats[symbol] = {
    #                 "total_profit": 0,
    #                 "num_trades": 0,
    #                 "win_rate": 0,
    #                 "avg_profit": 0,
    #                 "strategy": self.strategies[symbol].name,
    #             }
    #
    #     return stats

    def get_results(self) -> Dict:
        """Get backtest results"""

        if not self.equity_curve:
            return {}

        final_equity = self.equity_curve[-1]["equity"]
        total_return = ((final_equity - self.initial_capital) / self.initial_capital) * 100

        # Per-symbol statistics
        symbol_stats = self._calculate_symbol_stats()

        # Overall statistics
        if not self.trades:
            # No trades executed
            return {
                "total_return": total_return,
                "win_rate": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "total_trades": 0,
                "avg_profit": 0,
                "final_equity": final_equity,
                "symbol_stats": symbol_stats,
                "num_symbols": len(self.strategies),
            }

        trades_df = pd.DataFrame(self.trades)

        # Check if profit column exists
        if "profit" in trades_df.columns:
            completed_trades = trades_df[trades_df["profit"].notna()]
        else:
            completed_trades = pd.DataFrame()

        if not completed_trades.empty:
            win_rate = (completed_trades["profit"] > 0).sum() / len(completed_trades) * 100
            avg_profit = completed_trades["profit"].mean()

            returns = completed_trades["profit_pct"].values
            if len(returns) > 1:
                sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
            else:
                sharpe = 0
        else:
            win_rate = 0
            avg_profit = 0
            sharpe = 0

        # Max drawdown
        equity_values = [e["equity"] for e in self.equity_curve]
        peak = equity_values[0]
        max_dd = 0

        for value in equity_values:
            if value > peak:
                peak = value
            dd = ((peak - value) / peak) * 100
            if dd > max_dd:
                max_dd = dd

        return {
            "total_return": total_return,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "total_trades": len(completed_trades),
            "avg_profit": avg_profit,
            "final_equity": final_equity,
            "symbol_stats": symbol_stats,
            "num_symbols": len(self.strategies),
        }

    def _calculate_symbol_stats(self) -> Dict:
        """Calculate per-symbol statistics"""

        stats = {}

        # Check if we have any trades
        if not self.trades:
            # Return empty stats for all symbols
            for symbol in self.strategies.keys():
                stats[symbol] = {
                    "total_profit": 0,
                    "num_trades": 0,
                    "win_rate": 0,
                    "avg_profit": 0,
                    "strategy": self.strategies[symbol].name,
                }
            return stats

        trades_df = pd.DataFrame(self.trades)

        # Safety check: ensure required columns exist
        if "symbol" not in trades_df.columns:
            logger.warning("No 'symbol' column in trades DataFrame")
            # Return empty stats for all symbols
            for symbol in self.strategies.keys():
                stats[symbol] = {
                    "total_profit": 0,
                    "num_trades": 0,
                    "win_rate": 0,
                    "avg_profit": 0,
                    "strategy": self.strategies[symbol].name,
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
                total_profit = completed["profit"].sum()
                num_trades = len(completed)
                win_rate = (completed["profit"] > 0).sum() / num_trades * 100

                stats[symbol] = {
                    "total_profit": total_profit,
                    "num_trades": num_trades,
                    "win_rate": win_rate,
                    "avg_profit": completed["profit"].mean(),
                    "strategy": self.strategies[symbol].name,
                }
            else:
                stats[symbol] = {
                    "total_profit": 0,
                    "num_trades": 0,
                    "win_rate": 0,
                    "avg_profit": 0,
                    "strategy": self.strategies[symbol].name,
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

        return engine.get_results()
