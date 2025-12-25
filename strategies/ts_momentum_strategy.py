"""
Time-Series Momentum (Trend Following) Strategy
Alpha source: Persistent trends
Examples: 3-12 month momentum, Dual moving average crossover, Volatility-scaled trend
Signal: sign(P_t − MA_t)
Why add it: Extremely robust, Works across asset classes, Excellent for drawdown diversification
"""

import warnings
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from strategies.adpative_trend_ff_strategy import AdaptiveTrendFollowingStrategy

warnings.filterwarnings("ignore")


class TimeSeriesMomentumStrategy:
    """
    Time-Series Momentum (Trend Following) Strategy

    Key Features:
    1. Multiple signal generation methods
    2. Volatility scaling for position sizing
    3. Asset-class aware parameterization
    4. Drawdown protection mechanisms
    """

    def __init__(
        self,
        asset_symbol: str,
        asset_class: str = "equity",
        signal_method: str = "dual_ma",
        fast_period: int = 20,
        slow_period: int = 50,
        momentum_period: int = 252,
        volatility_lookback: int = 63,
        target_volatility: float = 0.15,
        max_position: float = 1.0,
        enable_drawdown_protection: bool = True,
        **kwargs,
    ):
        """
        Initialize Time-Series Momentum Strategy

        Args:
            asset_symbol: Asset symbol
            asset_class: Type of asset (equity, commodity, fx, rates)
            signal_method: Method to generate signals
                          'simple_momentum' - 3-12 month return
                          'dual_ma' - Dual moving average crossover
                          'vol_scaled' - Volatility-scaled trend
            fast_period: Fast moving average/lookback period
            slow_period: Slow moving average/lookback period
            momentum_period: Momentum calculation period (in days, typically 3-12 months)
            volatility_lookback: Lookback for volatility calculation
            target_volatility: Annualized target volatility for scaling
            max_position: Maximum position size (absolute value)
            enable_drawdown_protection: Enable drawdown-based position reduction
        """
        self.asset_symbol = asset_symbol
        self.asset_class = asset_class.lower()
        self.signal_method = signal_method.lower()

        # Set asset-class specific defaults if not provided
        if self.asset_class == "equity":
            self.fast_period = fast_period
            self.slow_period = slow_period or 200
            self.momentum_period = momentum_period or 252  # 12 months
        elif self.asset_class == "commodity":
            self.fast_period = fast_period or 40
            self.slow_period = slow_period or 100
            self.momentum_period = momentum_period or 126  # 6 months
        elif self.asset_class in ["fx", "currency"]:
            self.fast_period = fast_period or 10
            self.slow_period = slow_period or 30
            self.momentum_period = momentum_period or 63  # 3 months
        else:  # rates, bonds, etc.
            self.fast_period = fast_period
            self.slow_period = slow_period
            self.momentum_period = momentum_period

        self.volatility_lookback = volatility_lookback
        self.target_volatility = target_volatility
        self.max_position = max_position
        self.enable_drawdown_protection = enable_drawdown_protection

        # Risk management parameters
        self.stop_loss_pct = kwargs.get("stop_loss_pct", 0.10)
        self.trailing_stop_pct = kwargs.get("trailing_stop_pct", 0.08)
        self.max_drawdown_limit = kwargs.get("max_drawdown_limit", 0.20)

        # Strategy state
        self.current_position = 0
        self.position_size = 0
        self.entry_price = 0
        self.trailing_stop_level = 0
        self.peak_equity = 1.0
        self.current_equity = 1.0
        self.signals_history = []

        # Performance tracking
        self.equity_curve = []
        self.drawdown_history = []

    def _calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate logarithmic returns"""
        return np.log(prices / prices.shift(1))

    def _calculate_volatility(self, returns: pd.Series, annualize: bool = True) -> float:
        """Calculate volatility"""
        if len(returns) < 2:
            return 0.0

        vol = returns.std()
        if annualize:
            vol = vol * np.sqrt(252)  # Annualize assuming 252 trading days

        return vol

    def _calculate_momentum_signal(self, prices: pd.Series) -> float:
        """
        Calculate simple momentum signal: sign(P_t - P_{t-n})

        Args:
            prices: Price series

        Returns:
            Momentum signal (-1, 0, 1)
        """
        if len(prices) < self.momentum_period + 1:
            return 0

        current_price = prices.iloc[-1]
        past_price = prices.iloc[-self.momentum_period]

        # Basic momentum signal
        if current_price > past_price:
            return 1.0
        elif current_price < past_price:
            return -1.0
        else:
            return 0.0

    def _calculate_dual_ma_signal(self, prices: pd.Series) -> float:
        """
        Dual moving average crossover signal

        Args:
            prices: Price series

        Returns:
            MA crossover signal (-1, 0, 1)
        """
        if len(prices) < max(self.fast_period, self.slow_period) + 1:
            return 0

        # Calculate moving averages
        fast_ma = prices.rolling(window=self.fast_period).mean().iloc[-1]
        slow_ma = prices.rolling(window=self.slow_period).mean().iloc[-1]

        # Generate signal
        if fast_ma > slow_ma:
            return 1.0
        elif fast_ma < slow_ma:
            return -1.0
        else:
            return 0.0

    def _calculate_vol_scaled_signal(self, prices: pd.Series) -> Tuple[float, float]:
        """
        Volatility-scaled trend signal

        Args:
            prices: Price series

        Returns:
            Tuple of (raw_signal, volatility_scale)
        """
        if len(prices) < max(self.fast_period, self.volatility_lookback) + 1:
            return 0.0, 1.0

        # Calculate momentum signal
        raw_signal = self._calculate_momentum_signal(prices)

        # Calculate volatility
        returns = self._calculate_returns(prices)
        recent_returns = returns.iloc[-self.volatility_lookback :] if len(returns) >= self.volatility_lookback else returns

        if len(recent_returns) < 20:  # Minimum for meaningful volatility
            vol_scale = 1.0
        else:
            current_vol = self._calculate_volatility(recent_returns, annualize=True)

            # Avoid division by zero
            if current_vol > 0:
                # Scale position inversely to volatility (target constant risk)
                vol_scale = min(self.target_volatility / current_vol, 3.0)  # Cap at 3x
            else:
                vol_scale = 1.0

        return raw_signal, vol_scale

    def _calculate_drawdown_protection(self, current_equity: float) -> float:
        """
        Calculate position reduction factor based on drawdown

        Args:
            current_equity: Current strategy equity

        Returns:
            Drawdown reduction factor (0 to 1)
        """
        # Update peak equity
        self.peak_equity = max(self.peak_equity, current_equity)
        self.current_equity = current_equity

        # Calculate current drawdown
        current_drawdown = (self.peak_equity - current_equity) / self.peak_equity

        # Store for history
        self.drawdown_history.append(current_drawdown)

        # Apply reduction if drawdown exceeds threshold
        if current_drawdown > self.max_drawdown_limit:
            # Gradually reduce positions as drawdeepens
            reduction = 1.0 - (current_drawdown - self.max_drawdown_limit) / self.max_drawdown_limit
            return max(0.1, reduction)  # Minimum 10% position
        else:
            return 1.0

    def _check_stop_loss(self, current_price: float, signal_direction: float) -> bool:
        """
        Check if stop loss or trailing stop is triggered

        Args:
            current_price: Current asset price
            signal_direction: Current signal direction

        Returns:
            True if stop loss triggered, False otherwise
        """
        if self.current_position == 0:
            return False

        # Calculate unrealized P&L
        if self.current_position > 0:  # Long position
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            # Update trailing stop
            if pnl_pct > 0:
                self.trailing_stop_level = max(
                    self.trailing_stop_level,
                    current_price * (1 - self.trailing_stop_pct),
                )
        else:  # Short position
            pnl_pct = (self.entry_price - current_price) / self.entry_price
            # Update trailing stop
            if pnl_pct > 0:
                self.trailing_stop_level = min(
                    self.trailing_stop_level,
                    current_price * (1 + self.trailing_stop_pct),
                )

        # Check stop loss conditions
        stop_triggered = False

        # Fixed stop loss
        if abs(pnl_pct) < -self.stop_loss_pct:
            stop_triggered = True

        # Trailing stop
        if self.current_position > 0 and current_price < self.trailing_stop_level:
            stop_triggered = True
        elif self.current_position < 0 and current_price > self.trailing_stop_level:
            stop_triggered = True

        return stop_triggered

    def generate_signal(
        self,
        data: Union[pd.DataFrame, pd.Series],
        current_equity: Optional[float] = None,
    ) -> Dict:
        """
        Generate trading signal for Time-Series Momentum

        Args:
            data: Price data (DataFrame with 'close' or Series)
            current_equity: Current strategy equity for drawdown protection

        Returns:
            Dictionary with signal details
        """
        # Extract price series
        if isinstance(data, pd.DataFrame):
            if "close" in data.columns:
                prices = data["close"]
            else:
                prices = data.iloc[:, 0]
        else:
            prices = data

        if len(prices) < max(self.fast_period, self.slow_period, self.momentum_period) + 1:
            return {
                "signal": 0,
                "position_size": 0,
                "raw_signal": 0,
                "volatility_scale": 1.0,
                "drawdown_scale": 1.0,
                "metadata": {"error": "insufficient_data"},
            }

        # Check stop loss first
        current_price = prices.iloc[-1]
        stop_loss_triggered = self._check_stop_loss(current_price, 0)

        if stop_loss_triggered and self.current_position != 0:
            # Exit position due to stop loss
            self.current_position = 0
            self.position_size = 0
            self.trailing_stop_level = 0
            return {
                "signal": (-np.sign(self.current_position) if self.current_position != 0 else 0),
                "position_size": (abs(self.current_position) if self.current_position != 0 else 0),
                "raw_signal": 0,
                "volatility_scale": 1.0,
                "drawdown_scale": 1.0,
                "metadata": {
                    "action": "stop_loss_exit",
                    "entry_price": self.entry_price,
                },
            }

        # Generate raw signal based on method
        raw_signal = 0
        volatility_scale = 1.0

        if self.signal_method == "simple_momentum":
            raw_signal = self._calculate_momentum_signal(prices)

        elif self.signal_method == "dual_ma":
            raw_signal = self._calculate_dual_ma_signal(prices)

        elif self.signal_method == "vol_scaled":
            raw_signal, volatility_scale = self._calculate_vol_scaled_signal(prices)

        else:
            # Default to dual MA
            raw_signal = self._calculate_dual_ma_signal(prices)

        # Calculate drawdown protection scaling
        drawdown_scale = 1.0
        if self.enable_drawdown_protection and current_equity is not None:
            drawdown_scale = self._calculate_drawdown_protection(current_equity)

        # Determine if we should enter/exit/hold
        signal_direction = 0
        position_size = 0

        # Entry/Exit logic
        if raw_signal != 0 and self.current_position == 0:
            # Enter new position
            signal_direction = raw_signal
            position_size = min(abs(raw_signal) * volatility_scale * drawdown_scale, self.max_position)
            self.current_position = signal_direction * position_size
            self.position_size = position_size
            self.entry_price = current_price
            self.trailing_stop_level = current_price * (1 - self.trailing_stop_pct if signal_direction > 0 else 1 + self.trailing_stop_pct)

        elif self.current_position != 0 and np.sign(raw_signal) != np.sign(self.current_position):
            # Reverse position (signal changed direction)
            signal_direction = raw_signal * 2  # Double signal to indicate reversal
            position_size = min(abs(raw_signal) * volatility_scale * drawdown_scale, self.max_position)
            self.current_position = np.sign(raw_signal) * position_size
            self.position_size = position_size
            self.entry_price = current_price
            self.trailing_stop_level = current_price * (1 - self.trailing_stop_pct if raw_signal > 0 else 1 + self.trailing_stop_pct)

        elif self.current_position != 0 and np.sign(raw_signal) == np.sign(self.current_position):
            # Hold position
            signal_direction = np.sign(self.current_position)
            position_size = self.position_size

        # Store signal history
        self.signals_history.append(
            {
                "timestamp": (prices.index[-1] if hasattr(prices.index, "__len__") else len(prices)),
                "raw_signal": raw_signal,
                "position": self.current_position,
                "volatility_scale": volatility_scale,
                "drawdown_scale": drawdown_scale,
                "price": current_price,
            }
        )

        return {
            "signal": signal_direction,
            "position_size": position_size,
            "raw_signal": raw_signal,
            "volatility_scale": volatility_scale,
            "drawdown_scale": drawdown_scale,
            "metadata": {
                "asset_class": self.asset_class,
                "method": self.signal_method,
                "current_price": float(current_price),
                "in_position": self.current_position != 0,
                "entry_price": (float(self.entry_price) if self.entry_price > 0 else None),
                "stop_level": (float(self.trailing_stop_level) if self.trailing_stop_level > 0 else None),
            },
        }

    def generate_signals_batch(self, prices: pd.Series) -> pd.DataFrame:
        """
        Generate signals for entire price history (for backtesting)

        Args:
            prices: Historical price series

        Returns:
            DataFrame with signals and positions
        """
        signals = []
        positions = []
        position_sizes = []

        # Simulate rolling signal generation
        for i in range(len(prices)):
            if i < max(self.fast_period, self.slow_period, self.momentum_period):
                signals.append(0)
                positions.append(0)
                position_sizes.append(0)
                continue

            window = prices.iloc[: i + 1]
            result = self.generate_signal(window)

            signals.append(result["signal"])
            positions.append(result["position_size"] * np.sign(result["signal"]) if result["signal"] != 0 else 0)
            position_sizes.append(result["position_size"])

        # Create results DataFrame
        results = pd.DataFrame(
            {
                "price": prices,
                "signal": signals,
                "position": positions,
                "position_size": position_sizes,
            },
            index=prices.index,
        )

        # Calculate returns
        results["returns"] = self._calculate_returns(results["price"])
        results["strategy_returns"] = results["position"].shift(1) * results["returns"]
        results["cumulative_returns"] = (1 + results["strategy_returns"]).cumprod()

        return results

    def get_performance_metrics(self, strategy_returns: pd.Series) -> Dict:
        """
        Calculate performance metrics

        Args:
            strategy_returns: Strategy return series

        Returns:
            Dictionary of performance metrics
        """
        if len(strategy_returns) < 2:
            return {}

        # Basic metrics
        total_return = (1 + strategy_returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)

        # Risk metrics
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0

        # Drawdown metrics
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        wins = strategy_returns[strategy_returns > 0]
        win_rate = len(wins) / len(strategy_returns) if len(strategy_returns) > 0 else 0

        return {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "calmar_ratio": (float(annual_return / abs(max_drawdown)) if max_drawdown < 0 else 0),
        }

    def plot_signals(self, prices: pd.Series, signals_df: pd.DataFrame = None):
        """
        Plot strategy signals and performance
        """
        try:
            import matplotlib.pyplot as plt

            if signals_df is None:
                signals_df = self.generate_signals_batch(prices)

            fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [2, 1, 1]})

            # Plot 1: Price and signals
            ax1 = axes[0]
            ax1.plot(prices.index, prices, label="Price", color="black", alpha=0.7)

            # Plot buy/sell signals
            buy_signals = signals_df[signals_df["signal"] > 0]
            sell_signals = signals_df[signals_df["signal"] < 0]

            ax1.scatter(
                buy_signals.index,
                prices.loc[buy_signals.index],
                marker="^",
                color="green",
                s=100,
                label="Buy Signal",
            )
            ax1.scatter(
                sell_signals.index,
                prices.loc[sell_signals.index],
                marker="v",
                color="red",
                s=100,
                label="Sell Signal",
            )

            ax1.set_title(f"Time-Series Momentum Strategy: {self.asset_symbol}")
            ax1.set_ylabel("Price")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Positions
            ax2 = axes[1]
            ax2.fill_between(
                signals_df.index,
                0,
                signals_df["position"],
                alpha=0.3,
                color="blue",
                label="Position",
            )
            ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
            ax2.set_ylabel("Position")
            ax2.set_ylim([-self.max_position * 1.1, self.max_position * 1.1])
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Plot 3: Cumulative returns
            ax3 = axes[2]
            ax3.plot(
                signals_df.index,
                signals_df["cumulative_returns"],
                label="Strategy",
                color="blue",
            )
            ax3.plot(
                signals_df.index,
                (1 + signals_df["returns"]).cumprod(),
                label="Buy & Hold",
                color="gray",
                alpha=0.5,
            )
            ax3.set_ylabel("Cumulative Returns")
            ax3.set_xlabel("Date")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            return fig

        except ImportError:
            print("Matplotlib not installed. Install with: pip install matplotlib")
            return None


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example 1: Basic Time-Series Momentum
    print("=== Basic Time-Series Momentum Strategy ===")

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    prices = 100 + np.cumsum(np.random.randn(500) * 0.5)
    price_series = pd.Series(prices, index=dates)

    # Initialize strategy
    strategy = TimeSeriesMomentumStrategy(
        asset_symbol="SPY",
        asset_class="equity",
        signal_method="dual_ma",
        fast_period=20,
        slow_period=50,
        volatility_lookback=63,
        target_volatility=0.15,
        max_position=1.0,
    )

    # Generate signals
    results = strategy.generate_signals_batch(price_series)

    # Calculate performance
    metrics = strategy.get_performance_metrics(results["strategy_returns"].dropna())

    print("Strategy Performance:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    print(f"\nTotal trades: {(results['signal'].abs() > 0).sum()}")
    print(f"Average position size: {results['position_size'].abs().mean():.3f}")

    # Example 2: Volatility-scaled trend following
    print("\n=== Volatility-Scaled Trend Following ===")

    vol_scaled_strategy = TimeSeriesMomentumStrategy(
        asset_symbol="GC=F",  # Gold futures
        asset_class="commodity",
        signal_method="vol_scaled",
        momentum_period=126,  # 6-month momentum
        volatility_lookback=63,
        target_volatility=0.12,
        enable_drawdown_protection=True,
    )

    # Example 3: Adaptive trend following
    print("\n=== Adaptive Trend Following Strategy ===")

    adaptive_strategy = AdaptiveTrendFollowingStrategy(asset_symbol="EURUSD", asset_class="fx", signal_method="dual_ma")

    # Generate adaptive signal
    adaptive_signal = adaptive_strategy.generate_signal(price_series)
    print(f"Adaptive signal: {adaptive_signal}")
