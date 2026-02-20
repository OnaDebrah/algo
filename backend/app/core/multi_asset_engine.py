"""
Multi-Asset Backtesting Engine
Backtest strategies across multiple symbols simultaneously
Enhanced to support pairs trading strategies like Kalman Filter
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Union

import pandas as pd

from backend.app.analytics import calculate_performance_metrics
from backend.app.core.data_fetcher import fetch_stock_data
from backend.app.core.risk_manager import RiskManager
from backend.app.schemas.backtest import MultiAssetBacktestResult
from backend.app.strategies import BaseStrategy

logger = logging.getLogger(__name__)


class MultiAssetEngine:
    """
    Engine for backtesting across multiple assets

    Supports two modes:
    1. Independent strategies: Each symbol has its own strategy
    2. Pairs trading: One strategy operates on a pair of symbols
    """

    def __init__(
        self,
        strategies: Union[Dict[str, BaseStrategy], BaseStrategy],  # {symbol: strategy} OR single pairs strategy
        initial_capital: float = 100000,
        risk_manager: RiskManager = None,
        allocation_method: str = "equal",  # equal, optimized, custom
        commission_rate: float = 0.05,
        slippage_rate: float = 0.03,
        pairs_mode: bool = False,  # NEW: Enable pairs trading mode
        pair_symbols: Optional[List[str]] = None,  # NEW: Symbols for pairs trading
    ):
        """
        Initialize multi-asset backtesting engine

        Args:
            strategies: Dictionary mapping symbols to strategies OR single pairs strategy
            initial_capital: Starting capital
            risk_manager: Risk manager instance
            allocation_method: How to allocate capital across assets
            pairs_mode: If True, strategy operates on multiple symbols as a pair
            pair_symbols: List of symbols for pairs trading (e.g., ['AAPL', 'MSFT'])
        """
        self.pairs_mode = pairs_mode
        self.pair_symbols = pair_symbols or []

        # Handle strategies based on mode
        if pairs_mode:
            # Single strategy for all symbols (pairs trading)
            self.pairs_strategy = strategies if isinstance(strategies, BaseStrategy) else None
            self.strategies = {}
            if not self.pairs_strategy:
                raise ValueError("In pairs_mode, strategies must be a single BaseStrategy instance")
        else:
            # Independent strategies per symbol
            self.strategies = strategies if isinstance(strategies, dict) else {}
            self.pairs_strategy = None

        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # {symbol: position_dict}
        self.trades = []
        self.equity_curve = []
        self.risk_manager = risk_manager or RiskManager()
        self.allocation_method = allocation_method

        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

        # Calculate capital allocation per symbol
        self._calculate_allocations()

    def _calculate_allocations(self):
        """Calculate capital allocation for each symbol"""
        if self.pairs_mode:
            # In pairs mode, allocate capital across the pair
            num_symbols = len(self.pair_symbols)
            if num_symbols > 0:
                self.allocations = {symbol: 1.0 / num_symbols for symbol in self.pair_symbols}
            else:
                self.allocations = {}
        else:
            # Independent mode
            num_symbols = len(self.strategies)
            if self.allocation_method == "equal":
                self.allocations = {symbol: 1.0 / num_symbols for symbol in self.strategies.keys()}
            else:
                self.allocations = {symbol: 1.0 / num_symbols for symbol in self.strategies.keys()}

        logger.info(f"Capital allocations: {self.allocations}")

    async def run_backtest(self, symbols: List[str], period: str, interval: str):
        """
        Run backtest across multiple assets

        Args:
            symbols: List of symbols to backtest
            period: Time period
            interval: Data interval
        """
        logger.info(
            f"Starting {'pairs' if self.pairs_mode else 'multi-asset'} backtest: " f"{len(symbols)} symbols, period: {period}, interval: {interval}"
        )

        # Fetch data for all symbols
        data_dict = {}
        for symbol in symbols:
            data = await fetch_stock_data(symbol, period, interval)
            if not data.empty:
                data_dict[symbol] = data
            else:
                logger.warning(f"No data for {symbol}, skipping")

        if not data_dict:
            logger.error("No data available for any symbol")
            return

        # Align all data to common dates
        aligned_data = self._align_data(data_dict)

        # Pre-compute vectorized signals for all strategies that support it
        if not self.pairs_mode:
            self._precompute_vectorized_signals(aligned_data)

        # Run backtest for each timestamp
        for i in range(len(aligned_data["dates"])):
            timestamp = aligned_data["dates"][i]

            if self.pairs_mode:
                # PAIRS TRADING MODE
                self._process_pairs_trading(aligned_data, i, timestamp)
            else:
                # INDEPENDENT STRATEGIES MODE
                self._process_independent_strategies(aligned_data, i, timestamp)

            # Calculate portfolio equity
            self._update_equity(timestamp, aligned_data, i)

        logger.info(f"Backtest complete: {len(self.trades)} trades, " f"Final equity: ${self.equity_curve[-1]['equity']:,.2f}")

    def _process_pairs_trading(self, aligned_data: Dict, index: int, timestamp):
        """Process pairs trading strategy (e.g., Kalman Filter)"""
        if not self.pairs_strategy:
            return

        # Build dataframe with all pair symbols
        pair_data = {}
        for symbol in self.pair_symbols:
            if symbol in aligned_data["data"]:
                symbol_data = aligned_data["data"][symbol].iloc[: index + 1]
                # Use 'Close' price as the main column
                pair_data[symbol] = symbol_data["Close"]

        if len(pair_data) < len(self.pair_symbols):
            logger.warning(f"Missing data for some symbols in pair at index {index}")
            return

        # Create combined DataFrame for the pair
        combined_df = pd.DataFrame(pair_data)

        # Generate signal from pairs strategy
        signal_info = self.pairs_strategy.generate_signal(combined_df)

        if not signal_info:
            return

        signal = signal_info.get("signal", 0)
        position_size = signal_info.get("position_size", 1.0)
        metadata = signal_info.get("metadata", {})

        # For Kalman Filter, the signal indicates:
        # +1: Long asset_1, Short asset_2
        # -1: Short asset_1, Long asset_2
        # 0: No position or exit

        if signal != 0:
            # Execute trades on both assets
            asset_1 = self.pair_symbols[0]
            asset_2 = self.pair_symbols[1]

            price_1 = aligned_data["data"][asset_1]["Close"].iloc[index]
            price_2 = aligned_data["data"][asset_2]["Close"].iloc[index]

            # Get hedge ratio from metadata if available
            hedge_ratio = metadata.get("hedge_ratio", 1.0)

            if signal == 1:
                # Long asset_1, Short asset_2
                self._execute_pairs_trade(asset_1, 1, price_1, timestamp, self.pairs_strategy.name, position_size, metadata)
                self._execute_pairs_trade(asset_2, -1, price_2, timestamp, self.pairs_strategy.name, position_size * hedge_ratio, metadata)
            elif signal == -1:
                # Short asset_1, Long asset_2
                self._execute_pairs_trade(asset_1, -1, price_1, timestamp, self.pairs_strategy.name, position_size, metadata)
                self._execute_pairs_trade(asset_2, 1, price_2, timestamp, self.pairs_strategy.name, position_size * hedge_ratio, metadata)

    def _execute_pairs_trade(
        self, symbol: str, signal: int, current_price: float, timestamp, strategy_name: str, position_size: float = 1.0, metadata: dict = None
    ):
        """Execute trade for pairs trading (supports both long and short)"""

        symbol_allocation = self.allocations.get(symbol, 0.5)  # Default 50% if not set
        available_capital = self.cash * symbol_allocation * position_size

        # BUY (or cover short)
        if signal == 1:
            if symbol in self.positions and self.positions[symbol].get("is_short", False):
                # Cover existing short position
                self._close_pairs_position(symbol, current_price, timestamp, strategy_name)
            elif symbol not in self.positions:
                # Open new long position
                execution_price = current_price * (1 + self.slippage_rate)
                quantity = self.risk_manager.calculate_position_size(available_capital, execution_price)

                trade_value = quantity * execution_price
                commission = trade_value * self.commission_rate
                total_cost = trade_value + commission

                if total_cost <= self.cash:
                    self.cash -= total_cost
                    self.positions[symbol] = {
                        "symbol": symbol,
                        "quantity": quantity,
                        "entry_price": execution_price,
                        "entry_time": timestamp,
                        "strategy": strategy_name,
                        "entry_commission": commission,
                        "is_short": False,
                        "metadata": metadata or {},
                    }

                    trade_data = {
                        "symbol": symbol,
                        "order_type": "BUY",
                        "quantity": quantity,
                        "price": execution_price,
                        "commission": commission,
                        "slippage_impact": execution_price - current_price,
                        "total_value": trade_value + commission,
                        "side": "BUY",
                        "executed_at": (timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)),
                        "strategy": strategy_name,
                        "notes": f"Pairs trade: {metadata.get('z_score', 'N/A')}" if metadata else None,
                    }

                    self.trades.append(trade_data)
                    logger.debug(f"BUY: {quantity} {symbol} @ ${execution_price:.2f}")

        # SELL (or open short)
        elif signal == -1:
            if symbol in self.positions and not self.positions[symbol].get("is_short", False):
                # Close existing long position
                self._close_pairs_position(symbol, current_price, timestamp, strategy_name)
            elif symbol not in self.positions:
                # Open new short position (for pairs trading)
                execution_price = current_price * (1 - self.slippage_rate)
                quantity = self.risk_manager.calculate_position_size(available_capital, execution_price)

                # For short, we receive cash (minus commission)
                trade_value = quantity * execution_price
                commission = trade_value * self.commission_rate
                net_proceeds = trade_value - commission

                self.cash += net_proceeds

                self.positions[symbol] = {
                    "symbol": symbol,
                    "quantity": quantity,
                    "entry_price": execution_price,
                    "entry_time": timestamp,
                    "strategy": strategy_name,
                    "entry_commission": commission,
                    "is_short": True,
                    "metadata": metadata or {},
                }

                trade_data = {
                    "symbol": symbol,
                    "order_type": "SHORT",
                    "quantity": quantity,
                    "price": execution_price,
                    "commission": commission,
                    "total_value": trade_value,
                    "side": "SHORT",
                    "executed_at": (timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)),
                    "strategy": strategy_name,
                    "notes": f"Pairs short: {metadata.get('z_score', 'N/A')}" if metadata else None,
                    "profit": None,
                    "profit_pct": None,
                }

                self.trades.append(trade_data)
                logger.debug(f"SHORT: {quantity} {symbol} @ ${execution_price:.2f}")

    def _close_pairs_position(self, symbol: str, current_price: float, timestamp, strategy_name: str):
        """Close a pairs trading position (long or short)"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        is_short = position.get("is_short", False)

        if is_short:
            # Cover short: Buy back at current price
            execution_price = current_price * (1 + self.slippage_rate)
            trade_value = position["quantity"] * execution_price
            commission = trade_value * self.commission_rate
            total_cost = trade_value + commission

            self.cash -= total_cost

            # P&L calculation for short
            profit = (position["quantity"] * position["entry_price"]) - trade_value - position["entry_commission"] - commission
            profit_pct = (profit / (position["quantity"] * position["entry_price"])) * 100

            order_type = "COVER"
        else:
            # Close long: Sell at current price
            execution_price = current_price * (1 - self.slippage_rate)
            trade_value = position["quantity"] * execution_price
            commission = trade_value * self.commission_rate
            net_proceeds = trade_value - commission

            self.cash += net_proceeds

            # P&L calculation for long
            total_commissions = position["entry_commission"] + commission
            profit = trade_value - (position["quantity"] * position["entry_price"]) - total_commissions
            profit_pct = (profit / (position["quantity"] * position["entry_price"])) * 100

            order_type = "SELL"

        trade_data = {
            "symbol": symbol,
            "order_type": order_type,
            "quantity": position["quantity"],
            "price": execution_price,
            "commission": commission,
            "total_value": trade_value,
            "side": order_type,
            "executed_at": (timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)),
            "strategy": strategy_name,
            "profit": profit,
            "profit_pct": profit_pct,
        }

        self.trades.append(trade_data)
        logger.debug(f"{order_type}: {position['quantity']} {symbol} @ ${execution_price:.2f} (P&L: ${profit:.2f})")

        del self.positions[symbol]

    def _process_independent_strategies(self, aligned_data: Dict, index: int, timestamp):
        """Process independent strategies with vectorized signal lookup when available"""
        for symbol in aligned_data["symbols"]:
            if symbol not in self.strategies:
                logger.debug(f"No strategy for {symbol}, skipping")
                continue

            strategy = self.strategies[symbol]
            full_data = aligned_data["data"][symbol]
            current_price = full_data["Close"].iloc[index]

            # Use pre-computed vectorized signals if available
            if hasattr(self, "_precomputed_signals") and symbol in self._precomputed_signals:
                signal = int(self._precomputed_signals[symbol].iloc[index])
            else:
                # Fallback to loop-based signal generation
                symbol_data = full_data.iloc[: index + 1]
                signal_info = strategy.generate_signal(symbol_data)

                from backend.app.strategies.base_strategy import normalize_signal

                normalized = normalize_signal(signal_info)
                signal = normalized["signal"]

            if signal != 0:
                logger.info(f"Signal for {symbol} at index {index}: {signal} (price={current_price:.2f}, cash={self.cash:.2f})")

            # Execute trade
            self._execute_trade(symbol, signal, current_price, timestamp, strategy.name)

    def _precompute_vectorized_signals(self, aligned_data: Dict):
        """Pre-compute all signals using vectorized methods where available"""
        self._precomputed_signals = {}

        for symbol in aligned_data["symbols"]:
            if symbol not in self.strategies:
                continue

            strategy = self.strategies[symbol]
            full_data = aligned_data["data"][symbol]

            if hasattr(strategy, "generate_signals_vectorized"):
                try:
                    signals = strategy.generate_signals_vectorized(full_data)
                    self._precomputed_signals[symbol] = signals
                    buy_count = (signals == 1).sum()
                    sell_count = (signals == -1).sum()
                    logger.info(
                        f"Pre-computed vectorized signals for {symbol} ({strategy.name}): {buy_count} buys, {sell_count} sells out of {len(signals)} bars"
                    )
                except Exception as e:
                    logger.warning(f"Vectorized signal failed for {symbol}: {e}, will use loop fallback")

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
        """Execute trade with commission and slippage (original implementation)"""

        symbol_allocation = self.allocations.get(symbol, 0)
        available_capital = self.cash * symbol_allocation

        # --- BUY SIGNAL ---
        if signal == 1 and symbol not in self.positions:
            execution_price = current_price * (1 + self.slippage_rate)
            quantity = self.risk_manager.calculate_position_size(available_capital, execution_price)

            trade_value = quantity * execution_price
            commission = trade_value * self.commission_rate
            total_cost = trade_value + commission

            if total_cost <= self.cash:
                self.cash -= total_cost
                self.positions[symbol] = {
                    "symbol": symbol,
                    "quantity": quantity,
                    "entry_price": execution_price,
                    "entry_time": timestamp,
                    "strategy": strategy_name,
                    "entry_commission": commission,
                }

                trade_data = {
                    "symbol": symbol,
                    "order_type": "BUY",
                    "quantity": quantity,
                    "price": execution_price,
                    "commission": commission,
                    "slippage_impact": execution_price - current_price,
                    "total_value": trade_value + commission,
                    "side": "BUY",
                    "executed_at": (timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)),
                    "strategy": strategy_name,
                }

                self.trades.append(trade_data)
                logger.debug(f"BUY: {quantity} {symbol} @ ${execution_price:.2f} (Comm: ${commission:.2f})")

        # --- SELL SIGNAL ---
        elif signal == -1 and symbol in self.positions:
            position = self.positions[symbol]
            execution_price = current_price * (1 - self.slippage_rate)

            trade_value = position["quantity"] * execution_price
            commission = trade_value * self.commission_rate
            net_proceeds = trade_value - commission
            self.cash += net_proceeds

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
                "total_value": trade_value,
                "side": "SELL",
                "executed_at": (timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)),
                "strategy": strategy_name,
                "profit": profit,
                "profit_pct": profit_pct,
            }
            self.trades.append(trade_data)
            logger.debug(
                f"SELL: {position['quantity']} {symbol} @ ${execution_price:.2f} " f"(Net P&L: ${profit:.2f}, Fees: ${total_commissions:.2f})"
            )

            del self.positions[symbol]

    def _update_equity(self, timestamp, aligned_data: Dict, index: int):
        """Update portfolio equity"""

        equity = self.cash

        # Add value of all positions
        for symbol, position in self.positions.items():
            if symbol in aligned_data["data"]:
                current_price = aligned_data["data"][symbol]["Close"].iloc[index]

                if position.get("is_short", False):
                    # For short positions: cash already includes short-sale proceeds,
                    # so we subtract the current buyback cost (obligation to return shares)
                    equity -= position["quantity"] * current_price
                else:
                    # For long positions
                    equity += position["quantity"] * current_price

        self.equity_curve.append(
            {
                "timestamp": timestamp,
                "equity": equity,
                "cash": self.cash,
                "num_positions": len(self.positions),
            }
        )

    def get_results(self, benchmark_equity: List[Dict] = None) -> MultiAssetBacktestResult:
        """Get backtest results"""
        symbol_stats = self._calculate_symbol_stats()
        metrics = calculate_performance_metrics(self.trades, self.equity_curve, self.initial_capital, benchmark_equity=benchmark_equity)
        num_symbols = len(self.pair_symbols) if self.pairs_mode else len(self.strategies)

        return MultiAssetBacktestResult(
            **metrics,
            num_symbols=num_symbols,
            symbol_stats=symbol_stats,
        )

    def _calculate_symbol_stats(self) -> Dict:
        """Calculate per-symbol statistics"""

        stats = {}

        # Determine which symbols to track
        symbols_to_track = self.pair_symbols if self.pairs_mode else list(self.strategies.keys())

        if not self.trades:
            for symbol in symbols_to_track:
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

        if "symbol" not in trades_df.columns:
            logger.warning("No 'symbol' column in trades DataFrame")
            for symbol in symbols_to_track:
                stats[symbol] = {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "total_return": 0.0,
                    "win_rate": 0.0,
                    "avg_profit": 0.0,
                }
            return stats

        for symbol in symbols_to_track:
            symbol_trades = trades_df[trades_df["symbol"] == symbol]

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
