"""
Benchmark Calculator for comparing backtest results against buy-and-hold strategies
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BenchmarkCalculator:
    """Calculate benchmark performance for comparison"""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital

    def calculate_buy_and_hold(self, symbol: str, data: pd.DataFrame, commission_rate: float = 0.001) -> Dict:
        """
        Calculate buy-and-hold performance for a single asset

        Args:
            symbol: Asset symbol (e.g., 'SPY')
            data: Historical price data with 'Close' column
            commission_rate: Commission rate for trades

        Returns:
            Dictionary with benchmark metrics and equity curve
        """
        if data.empty or "Close" not in data.columns:
            raise ValueError(f"Invalid data for {symbol}")

        # Buy at the first price
        first_price = data["Close"].iloc[0]
        commission = self.initial_capital * commission_rate
        shares = (self.initial_capital - commission) / first_price

        # Vectorized equity calculation - no iterrows loop
        close_prices = data["Close"].values
        timestamps = data.index
        equity_values = shares * close_prices

        # Vectorized drawdown calculation
        equity_series = pd.Series(equity_values)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max

        # Build equity curve from vectorized arrays
        equity_curve = [
            {
                "timestamp": timestamps[i],
                "equity": float(equity_values[i]),
                "cash": 0,
                "drawdown": float(drawdown.iloc[i]),
            }
            for i in range(len(timestamps))
        ]

        # Final metrics
        final_equity = float(equity_values[-1])
        total_return = final_equity - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        max_drawdown = float(drawdown.min())

        # Sharpe ratio (annualized)
        returns = equity_series.pct_change().dropna()
        if len(returns) > 1:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0

        return {
            "symbol": symbol,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "final_equity": final_equity,
            "initial_capital": self.initial_capital,
            "shares": shares,
            "equity_curve": equity_curve,
            "trades": 1,  # Just buy and hold
        }

    def calculate_multi_benchmark(
        self, symbols: List[str], data_dict: Dict[str, pd.DataFrame], allocations: Dict[str, float] = None, commission_rate: float = 0.001
    ) -> Dict:
        """
        Calculate buy-and-hold for a portfolio of assets

        Args:
            symbols: List of symbols
            data_dict: Dictionary mapping symbols to their price data
            allocations: Custom allocation percentages (default: equal weight)
            commission_rate: Commission rate

        Returns:
            Dictionary with benchmark metrics
        """
        if not allocations:
            # Equal weight allocation
            allocations = {symbol: 100 / len(symbols) for symbol in symbols}

        # Normalize allocations to sum to 100
        total_alloc = sum(allocations.values())
        allocations = {k: (v / total_alloc) for k, v in allocations.items()}

        # Calculate for each asset
        symbol_positions = {}
        for symbol in symbols:
            if symbol not in data_dict or data_dict[symbol].empty:
                logger.warning(f"No data for {symbol}, skipping")
                continue

            capital_for_symbol = self.initial_capital * allocations[symbol]
            data = data_dict[symbol]

            first_price = data["Close"].iloc[0]
            commission = capital_for_symbol * commission_rate
            shares = (capital_for_symbol - commission) / first_price

            symbol_positions[symbol] = {"shares": shares, "data": data}

        # Find common date range
        all_dates = None
        for symbol, pos in symbol_positions.items():
            dates = pos["data"].index
            if all_dates is None:
                all_dates = set(dates)
            else:
                all_dates = all_dates.intersection(set(dates))

        all_dates = sorted(list(all_dates))

        # Calculate portfolio equity over time
        equity_curve = []
        for date in all_dates:
            total_equity = 0
            for symbol, pos in symbol_positions.items():
                price = pos["data"].loc[date, "Close"]
                total_equity += pos["shares"] * price

            equity_curve.append({"timestamp": date, "equity": total_equity, "cash": 0, "drawdown": 0})

        # Calculate drawdown
        equity_series = pd.Series([point["equity"] for point in equity_curve])
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max

        for i, point in enumerate(equity_curve):
            point["drawdown"] = drawdown.iloc[i]

        # Final metrics
        final_equity = equity_curve[-1]["equity"]
        total_return = final_equity - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        max_drawdown = drawdown.min()

        # Sharpe ratio
        returns = equity_series.pct_change().dropna()
        if len(returns) > 1:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0

        return {
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "final_equity": final_equity,
            "initial_capital": self.initial_capital,
            "equity_curve": equity_curve,
            "allocations": allocations,
            "symbols": list(symbol_positions.keys()),
        }

    def calculate_spy_benchmark(self, period: str, interval: str, commission_rate: float = 0.001) -> Dict:
        """
        Calculate SPY buy-and-hold benchmark

        Args:
            period: Time period (e.g., '1y', '2y')
            interval: Data interval (e.g., '1d')
            commission_rate: Commission rate

        Returns:
            Dictionary with SPY benchmark metrics
        """
        from backend.app.core import fetch_stock_data

        try:
            data = fetch_stock_data("SPY", period, interval)
            if data.empty:
                raise ValueError("No SPY data available")

            return self.calculate_buy_and_hold("SPY", data, commission_rate)

        except Exception as e:
            logger.error(f"Failed to calculate SPY benchmark: {e}")
            return None

    def compare_to_benchmark(self, strategy_metrics: Dict, benchmark_metrics: Dict) -> Dict:
        """
        Compare strategy performance to benchmark

        Args:
            strategy_metrics: Strategy performance metrics
            benchmark_metrics: Benchmark performance metrics

        Returns:
            Comparison metrics
        """
        return {
            "outperformance": strategy_metrics["total_return_pct"] - benchmark_metrics["total_return_pct"],
            "alpha": strategy_metrics["total_return_pct"] - benchmark_metrics["total_return_pct"],
            "sharpe_ratio_diff": strategy_metrics["sharpe_ratio"] - benchmark_metrics["sharpe_ratio"],
            "max_drawdown_diff": strategy_metrics["max_drawdown"] - benchmark_metrics["max_drawdown"],
            "strategy_return": strategy_metrics["total_return_pct"],
            "benchmark_return": benchmark_metrics["total_return_pct"],
            "strategy_sharpe": strategy_metrics["sharpe_ratio"],
            "benchmark_sharpe": benchmark_metrics["sharpe_ratio"],
        }
