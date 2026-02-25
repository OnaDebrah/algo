"""
Options Strategy Backtesting Engine
Backtest options strategies with historical data
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from ..strategies.options_builder import OptionsStrategyBuilder, create_preset_strategy
from ..strategies.options_strategies import (
    BlackScholesCalculator,
    OptionsStrategy,
    OptionType,
)

logger = logging.getLogger(__name__)


class OptionsBacktestEngine:
    """
    Backtest options strategies using historical underlying data
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        risk_free_rate: float = 0.05,
        commission_per_contract: float = 0.65,
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.commission = commission_per_contract

        self.positions: List[Dict] = []
        self.closed_positions: List[Dict] = []
        self.equity_curve = []
        self.trades = []

    def run_strategy_backtest(
        self,
        symbol: str,
        data: pd.DataFrame,
        strategy_type: OptionsStrategy,
        entry_rules: Dict,
        exit_rules: Dict,
        volatility: float = 0.3,
    ) -> Dict:
        """
        Run backtest for a specific options strategy

        Args:
            symbol: Underlying symbol
            data: Historical price data (OHLC)
            strategy_type: Type of options strategy
            entry_rules: Entry conditions (e.g., {'min_days': 30, 'delta_target': 0.3})
            exit_rules: Exit conditions (e.g., {'profit_target': 0.5, 'loss_limit': -0.3})
            volatility: Implied volatility to use

        Returns:
            Dictionary with backtest results
        """

        logger.info(f"Starting backtest for {strategy_type.value} on {symbol}")

        # Reset state
        self.capital = self.initial_capital
        self.positions = []
        self.closed_positions = []
        self.equity_curve = []
        self.trades = []

        # Iterate through historical data
        for i in range(len(data)):
            current_date = data.index[i]
            current_price = data["Close"].iloc[i]

            # Update existing positions
            self._update_positions(current_date, current_price, volatility, exit_rules)

            # Check for new entry signals
            if self._check_entry_signal(data, i, entry_rules):
                self._enter_position(
                    symbol,
                    current_date,
                    current_price,
                    strategy_type,
                    entry_rules,
                    volatility,
                )

            # Record equity
            portfolio_value = self._calculate_portfolio_value(current_price, volatility)
            self.equity_curve.append(
                {
                    "date": current_date,
                    "equity": portfolio_value,
                    "cash": self.capital,
                    "positions_value": portfolio_value - self.capital,
                }
            )

        # Close all remaining positions at end
        final_date = data.index[-1]
        final_price = data["Close"].iloc[-1]
        self._close_all_positions(final_date, final_price)

        # Calculate metrics
        results = self._calculate_results()

        return results

    def _check_entry_signal(self, data: pd.DataFrame, index: int, entry_rules: Dict) -> bool:
        """Check if entry conditions are met"""

        # Don't enter if we have open positions (for now)
        if self.positions:
            return False

        # Check if we have enough capital
        if self.capital < entry_rules.get("min_capital", 5000):
            return False

        # Custom entry logic based on rules
        signal_type = entry_rules.get("signal", "regular")

        if signal_type == "regular":
            # Enter every N days
            entry_frequency = entry_rules.get("entry_frequency", 30)
            return index % entry_frequency == 0

        elif signal_type == "rsi":
            # Entry based on RSI
            if index < 14:
                return False

            rsi_period = entry_rules.get("rsi_period", 14)
            rsi_oversold = entry_rules.get("rsi_oversold", 30)
            rsi_overbought = entry_rules.get("rsi_overbought", 70)

            closes = data["Close"].iloc[max(0, index - rsi_period) : index + 1]
            delta = closes.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]

            # Bullish strategies on oversold, bearish on overbought
            strategy_direction = entry_rules.get("direction", "bullish")
            if strategy_direction == "bullish":
                return current_rsi < rsi_oversold
            else:
                return current_rsi > rsi_overbought

        elif signal_type == "moving_average":
            # Entry based on moving average crossover
            if index < 50:
                return False

            short_ma = data["Close"].iloc[index - 20 : index + 1].mean()
            long_ma = data["Close"].iloc[index - 50 : index + 1].mean()

            direction = entry_rules.get("direction", "bullish")
            if direction == "bullish":
                return short_ma > long_ma
            else:
                return short_ma < long_ma

        return False

    def _enter_position(
        self,
        symbol: str,
        date: datetime,
        price: float,
        strategy_type: OptionsStrategy,
        entry_rules: Dict,
        volatility: float,
    ):
        """Enter a new options position"""

        # Use specific expiration if provided, otherwise calculate from DTE
        specific_exp = entry_rules.get("expiration")
        if specific_exp:
            from datetime import datetime as dt

            expiration = dt.strptime(str(specific_exp)[:10], "%Y-%m-%d") if isinstance(specific_exp, str) else specific_exp
        else:
            days_to_exp = entry_rules.get("days_to_expiration", 30)
            expiration = date + timedelta(days=days_to_exp)

        builder = OptionsStrategyBuilder(symbol, self.risk_free_rate)
        builder.current_price = price

        # Create strategy with custom strikes based on rules
        kwargs = self._get_strategy_strikes(price, entry_rules)

        builder = create_preset_strategy(
            strategy_type, symbol, price, expiration, volatility=volatility, risk_free_rate=self.risk_free_rate, **kwargs
        )

        # Calculate cost
        cost = builder.get_initial_cost()
        # Commission: per option contract (not stock legs)
        option_legs = [leg for leg in builder.legs if leg.option_type != OptionType.STOCK]
        num_option_contracts = sum(abs(leg.quantity) for leg in option_legs)
        commission = self.commission * num_option_contracts
        total_cost = cost + commission

        # For strategies involving stock legs (covered call, protective put, collar),
        # use margin requirement instead of full stock cost to make backtesting
        # practical with typical capital levels.
        has_stock_legs = any(leg.option_type == OptionType.STOCK for leg in builder.legs)
        if has_stock_legs:
            stock_cost = sum(leg.premium * leg.quantity for leg in builder.legs if leg.option_type == OptionType.STOCK)
            options_cost = sum(leg.premium * leg.quantity * 100 for leg in builder.legs if leg.option_type != OptionType.STOCK)
            # Margin requirement: 20% of stock value + full options cost + commission
            margin_requirement = abs(stock_cost) * 0.20 + options_cost + commission
            capital_needed = margin_requirement
        else:
            capital_needed = abs(total_cost)

        # Check if we can afford it
        if capital_needed > self.capital:
            logger.warning(f"Insufficient capital for {strategy_type.value} (need ${capital_needed:.0f}, have ${self.capital:.0f})")
            return

        # Enter position
        position = {
            "symbol": symbol,
            "strategy": strategy_type.value,
            "entry_date": date,
            "entry_price": price,
            "expiration": expiration,
            "builder": builder,
            "initial_cost": cost,
            "commission": commission,
            "volatility": volatility,
            "legs": builder.legs.copy(),
        }

        self.positions.append(position)
        self.capital -= total_cost

        logger.info(f"Entered {strategy_type.value} at ${price:.2f}, cost: ${total_cost:.2f}")

        # Record trade
        self.trades.append(
            {
                "date": date,
                "type": "ENTRY",
                "strategy": strategy_type.value,
                "price": price,
                "cost": total_cost,
            }
        )

    def _get_strategy_strikes(self, price: float, entry_rules: Dict) -> Dict:
        """Generate strike prices based on entry rules"""

        kwargs = {}

        # Default percentages
        otm_call_pct = entry_rules.get("otm_call_pct", 1.05)
        otm_put_pct = entry_rules.get("otm_put_pct", 0.95)

        kwargs["strike"] = price * otm_call_pct
        kwargs["call_strike"] = price * otm_call_pct
        kwargs["put_strike"] = price * otm_put_pct
        kwargs["long_strike"] = price
        kwargs["short_strike"] = price * otm_call_pct

        # Iron Condor
        kwargs["put_long_strike"] = price * 0.90
        kwargs["put_short_strike"] = price * 0.95
        kwargs["call_short_strike"] = price * 1.05
        kwargs["call_long_strike"] = price * 1.10

        return kwargs

    def _update_positions(
        self,
        current_date: datetime,
        current_price: float,
        volatility: float,
        exit_rules: Dict,
    ):
        """Update existing positions and check exit conditions"""

        positions_to_close = []

        for i, position in enumerate(self.positions):
            # Check expiration
            if current_date >= position["expiration"]:
                positions_to_close.append(i)
                continue

            # Calculate current P&L
            builder = position["builder"]
            builder.current_price = current_price

            current_value = self._calculate_position_value(position, current_price, volatility)

            pnl = current_value - position["initial_cost"]
            pnl_pct = pnl / abs(position["initial_cost"]) if position["initial_cost"] != 0 else 0

            # Check profit target
            profit_target = exit_rules.get("profit_target", 0.5)
            if pnl_pct >= profit_target:
                positions_to_close.append(i)
                logger.info(f"Profit target reached: {pnl_pct:.2%}")
                continue

            # Check loss limit
            loss_limit = exit_rules.get("loss_limit", -0.5)
            if pnl_pct <= loss_limit:
                positions_to_close.append(i)
                logger.info(f"Loss limit hit: {pnl_pct:.2%}")
                continue

            # Check days to expiration
            dte_exit = exit_rules.get("dte_exit", 7)
            days_remaining = (position["expiration"] - current_date).days
            if days_remaining <= dte_exit:
                positions_to_close.append(i)
                logger.info(f"DTE exit: {days_remaining} days remaining")
                continue

        # Close positions
        for i in sorted(positions_to_close, reverse=True):
            self._close_position(i, current_date, current_price, volatility)

    def _close_position(
        self,
        position_index: int,
        exit_date: datetime,
        exit_price: float,
        volatility: float,
    ):
        """Close a position"""

        position = self.positions.pop(position_index)

        # Calculate exit value
        exit_value = self._calculate_position_value(position, exit_price, volatility)

        # Add commission
        num_contracts = len(position["legs"])
        commission = self.commission * num_contracts
        exit_value -= commission

        # Update capital
        self.capital += exit_value

        # Calculate P&L
        pnl = exit_value - position["initial_cost"]
        pnl_pct = pnl / abs(position["initial_cost"]) if position["initial_cost"] != 0 else 0

        # Store closed position
        closed = position.copy()
        closed.update(
            {
                "exit_date": exit_date,
                "exit_price": exit_price,
                "exit_value": exit_value,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "days_held": (exit_date - position["entry_date"]).days,
            }
        )

        self.closed_positions.append(closed)

        logger.info(f"Closed {position['strategy']} - P&L: ${pnl:.2f} ({pnl_pct:.2%})")

        # Record trade
        self.trades.append(
            {
                "date": exit_date,
                "type": "EXIT",
                "strategy": position["strategy"],
                "price": exit_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            }
        )

    def _calculate_position_value(self, position: Dict, current_price: float, volatility: float) -> float:
        """Calculate current value of a position"""

        value = 0
        current_date = pd.Timestamp.now(tz="UTC")

        for leg in position["legs"]:
            # Stock legs: value = quantity * current_price (no expiry, no Greeks)
            if leg.option_type == OptionType.STOCK:
                value += leg.quantity * current_price
                continue

            if leg.expiry is None:
                continue

            expiry = pd.to_datetime(leg.expiry)
            if expiry.tzinfo is None:
                expiry = expiry.tz_localize("UTC")

            delta = expiry - current_date
            T = max(delta.total_seconds() / (365.0 * 24 * 3600), 0)

            if T <= 0:
                # At expiration - intrinsic value only
                if leg.option_type == OptionType.CALL:
                    intrinsic = max(current_price - leg.strike, 0)
                else:
                    intrinsic = max(leg.strike - current_price, 0)

                value += leg.quantity * intrinsic * 100
            else:
                # Calculate current option price
                current_premium = BlackScholesCalculator.calculate_option_price(
                    S=current_price,
                    K=leg.strike,
                    T=T,
                    r=self.risk_free_rate,
                    sigma=volatility,
                    option_type=leg.option_type,
                )

                value += leg.quantity * current_premium * 100

        return value

    def _calculate_portfolio_value(self, current_price: float, volatility: float) -> float:
        """Calculate total portfolio value"""

        total = self.capital

        for position in self.positions:
            total += self._calculate_position_value(position, current_price, volatility)

        return total

    def _close_all_positions(self, final_date: datetime, final_price: float):
        """Close all remaining positions at end of backtest"""

        while self.positions:
            vol = self.positions[0].get("volatility", 0.3)
            self._close_position(0, final_date, final_price, vol)

    def _calculate_results(self) -> Dict:
        """Calculate backtest performance metrics"""

        if not self.closed_positions:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "total_return": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "profit_factor": 0,
                "total_profit": 0,
                "total_loss": 0,
                "equity_curve": [],
                "trades": [],
            }

        # Basic metrics
        total_trades = len(self.closed_positions)
        winning_trades = [p for p in self.closed_positions if p["pnl"] > 0]
        losing_trades = [p for p in self.closed_positions if p["pnl"] <= 0]

        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0

        total_profit = sum(p["pnl"] for p in winning_trades)
        total_loss = abs(sum(p["pnl"] for p in losing_trades))

        avg_profit = total_profit / len(winning_trades) if winning_trades else 0
        avg_loss = total_loss / len(losing_trades) if losing_trades else 0

        profit_factor = total_profit / total_loss if total_loss > 0 else min(total_profit, 9999.99)

        # Portfolio metrics
        final_equity = self.equity_curve[-1]["equity"] if self.equity_curve else self.initial_capital
        total_return = (final_equity - self.initial_capital) / self.initial_capital * 100

        # Max drawdown
        equity_values = [e["equity"] for e in self.equity_curve]
        running_max = np.maximum.accumulate(equity_values)
        drawdowns = (equity_values - running_max) / running_max * 100
        max_drawdown = np.min(drawdowns)

        # Sharpe ratio (simplified)
        if len(equity_values) > 1:
            returns = np.diff(equity_values) / equity_values[:-1]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0

        return {
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "total_return": total_return,
            "final_equity": final_equity,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "avg_days_held": np.mean([p["days_held"] for p in self.closed_positions]),
            "avg_pnl_pct": np.mean([p["pnl_pct"] for p in self.closed_positions]) * 100,
        }


def backtest_options_strategy(symbol: str, data: pd.DataFrame, strategy_type: OptionsStrategy, **kwargs) -> Dict:
    """
    Convenience function to backtest an options strategy

    Args:
        symbol: Underlying symbol
        data: Historical price data
        strategy_type: Type of options strategy
        **kwargs: Additional parameters (entry_rules, exit_rules, etc.)

    Returns:
        Backtest results dictionary
    """

    engine = OptionsBacktestEngine(
        initial_capital=kwargs.get("initial_capital", 100000),
        risk_free_rate=kwargs.get("risk_free_rate", 0.05),
        commission_per_contract=kwargs.get("commission", 0.65),
    )

    entry_rules = kwargs.get(
        "entry_rules",
        {
            "signal": "regular",
            "entry_frequency": 30,
            "days_to_expiration": 30,
            "min_capital": 5000,
        },
    )

    exit_rules = kwargs.get("exit_rules", {"profit_target": 0.5, "loss_limit": -0.5, "dte_exit": 7})

    volatility = kwargs.get("volatility", 0.3)

    results = engine.run_strategy_backtest(symbol, data, strategy_type, entry_rules, exit_rules, volatility)

    results["engine"] = engine  # Include engine for equity curve access

    return results
