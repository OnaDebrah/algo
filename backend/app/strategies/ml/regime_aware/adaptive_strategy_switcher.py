import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from ....analytics.market_regime_detector import MarketRegimeDetector
from ....strategies.base_strategy import BaseStrategy, normalize_signal
from ....strategies.parabolic_sar import ParabolicSARStrategy
from ....strategies.technical.bb_mean_reversion import BollingerMeanReversionStrategy
from ....strategies.technical.rsi_strategy import RSIStrategy
from ....strategies.technical.sma_crossover import SMACrossoverStrategy
from ....strategies.ts_momentum_strategy import TimeSeriesMomentumStrategy
from ....strategies.volatility.volatility_breakout import VolatilityBreakoutStrategy

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Types of strategies for different regimes"""

    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    VOLATILITY = "volatility"
    ADAPTIVE = "adaptive"


@dataclass
class StrategyConfig:
    """Configuration for a strategy"""

    strategy_class: Type[BaseStrategy]
    strategy_type: StrategyType
    default_params: Dict[str, Any]
    weight: float = 1.0
    min_confidence: float = 0.3
    max_position_size: float = 1.0
    preferred_regimes: List[str] = field(default_factory=list)
    avoided_regimes: List[str] = field(default_factory=list)


@dataclass
class StrategyPerformance:
    """Performance tracking for a strategy"""

    strategy_name: str
    returns: List[float] = field(default_factory=list)
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_trade: float = 0.0
    num_trades: int = 0
    last_update: datetime = field(default_factory=datetime.now)
    regime_performance: Dict[str, float] = field(default_factory=dict)


class AdaptiveStrategySwitcher(BaseStrategy):
    """
    Dynamically switches between strategies based on:
    1. Detected market regime
    2. Recent strategy performance
    3. Confidence scores
    4. Risk metrics
    """

    def __init__(
        self,
        name: str = "adaptive_switcher",
        lookback_days: int = 252,
        regime_update_freq: int = 5,
        performance_lookback: int = 60,
        min_regime_confidence: float = 0.6,
        use_ensemble: bool = True,
        ensemble_size: int = 3,
        rebalance_frequency: int = 20,
        risk_aversion: float = 2.0,
        params: Dict = None,
    ):
        super().__init__(name, params or {})

        self.lookback_days = lookback_days
        self.regime_update_freq = regime_update_freq
        self.performance_lookback = performance_lookback
        self.min_regime_confidence = min_regime_confidence
        self.use_ensemble = use_ensemble
        self.ensemble_size = ensemble_size
        self.rebalance_frequency = rebalance_frequency
        self.risk_aversion = risk_aversion

        self.regime_detector = MarketRegimeDetector(name=f"{name}_regime")

        self.strategy_catalog = self._initialize_strategies()

        self.performance_history: Dict[str, StrategyPerformance] = {}
        self.current_strategies: List[Tuple[str, float]] = []  # (strategy_name, weight)
        self.last_regime = None
        self.last_update = None
        self.strategy_instances: Dict[str, BaseStrategy] = {}

        # Risk metrics
        self.daily_returns: List[float] = []
        self.current_drawdown = 0.0

        # Initialize performance tracking for each strategy
        for strat_name in self.strategy_catalog:
            self.performance_history[strat_name] = StrategyPerformance(strategy_name=strat_name)

        logger.info(f"AdaptiveStrategySwitcher initialized with {len(self.strategy_catalog)} strategies")

    def _initialize_strategies(self) -> Dict[str, StrategyConfig]:
        """Initialize the catalog of available strategies"""
        return {
            # Trend Following Strategies
            "sma_crossover": StrategyConfig(
                strategy_class=SMACrossoverStrategy,
                strategy_type=StrategyType.TREND_FOLLOWING,
                default_params={"short_window": 20, "long_window": 50},
                weight=1.0,
                preferred_regimes=["bull", "strong_trend"],
                avoided_regimes=["bear", "high_volatility"],
            ),
            "ts_momentum": StrategyConfig(
                strategy_class=TimeSeriesMomentumStrategy,
                strategy_type=StrategyType.MOMENTUM,
                default_params={"lookback": 12, "holding_period": 3},
                weight=1.0,
                preferred_regimes=["bull", "recovery"],
                avoided_regimes=["bear", "crisis"],
            ),
            "bb_mean_reversion": StrategyConfig(
                strategy_class=BollingerMeanReversionStrategy,
                strategy_type=StrategyType.MEAN_REVERSION,
                default_params={"period": 20, "std_dev": 2.0},
                weight=1.0,
                preferred_regimes=["neutral", "range_bound"],
                avoided_regimes=["strong_trend", "crisis"],
            ),
            "rsi_strategy": StrategyConfig(
                strategy_class=RSIStrategy,
                strategy_type=StrategyType.MEAN_REVERSION,
                default_params={"period": 14, "oversold": 30, "overbought": 70},
                weight=0.8,
                preferred_regimes=["neutral", "range_bound"],
                avoided_regimes=["strong_trend"],
            ),
            # Breakout Strategies
            "volatility_breakout": StrategyConfig(
                strategy_class=VolatilityBreakoutStrategy,
                strategy_type=StrategyType.BREAKOUT,
                default_params={"period": 20, "std_dev": 2.0},
                weight=1.0,
                preferred_regimes=["high_volatility", "breakout"],
                avoided_regimes=["low_volatility", "range_bound"],
            ),
            "parabolic_sar": StrategyConfig(
                strategy_class=ParabolicSARStrategy,
                strategy_type=StrategyType.TREND_FOLLOWING,
                default_params={"start": 0.02, "increment": 0.02, "maximum": 0.2},
                weight=0.7,
                preferred_regimes=["trending"],
                avoided_regimes=["choppy"],
            ),
        }

    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """
        Generate adaptive signal by combining multiple strategies
        """
        try:
            # Update regime and performance
            self._update_state(data)

            # Select best strategies for current conditions
            selected_strategies = self._select_strategies(data)

            if not selected_strategies:
                logger.warning("No strategies selected")
                return {"signal": 0, "position_size": 0.0, "metadata": {}}

            # Generate signals from selected strategies
            signals = []
            weights = []

            for strategy_name, weight in selected_strategies:
                strategy = self._get_strategy_instance(strategy_name)
                if strategy:
                    raw_signal = strategy.generate_signal(data)
                    signal_dict = normalize_signal(raw_signal)

                    if signal_dict["signal"] != 0:
                        signals.append(signal_dict["signal"])
                        weights.append(weight * signal_dict["position_size"])

            if not signals:
                return {"signal": 0, "position_size": 0.0, "metadata": {}}

            # Combine signals
            combined_signal, confidence = self._combine_signals(signals, weights)

            # Apply risk management
            position_size = self._apply_risk_management(combined_signal, confidence, data)

            # Update performance tracking
            self._update_performance_tracking(signals, weights)

            # Prepare metadata
            metadata = self._prepare_metadata(selected_strategies, confidence, data)

            return {"signal": combined_signal, "position_size": position_size, "metadata": metadata}

        except Exception as e:
            logger.error(f"Error generating adaptive signal: {e}")
            return {"signal": 0, "position_size": 0.0, "metadata": {"error": str(e)}}

    def _update_state(self, data: pd.DataFrame):
        """Update regime and performance metrics"""
        current_time = data.index[-1] if hasattr(data.index, "max") else datetime.now()

        # Update regime periodically
        if self.last_update is None or (hasattr(current_time, "day") and (current_time - self.last_update).days >= self.regime_update_freq):
            if len(data) > self.lookback_days:
                recent_data = data.tail(self.lookback_days)
                regime_df = self.regime_detector.detect_regimes(recent_data)

                if not regime_df.empty:
                    self.last_regime = regime_df.iloc[-1]
                    logger.info(f"Regime updated: {self.last_regime.get('regime')}")

            self.last_update = current_time

    def _select_strategies(self, data: pd.DataFrame) -> List[Tuple[str, float]]:
        """
        Select best strategies based on regime and performance
        """
        if not self.last_regime:
            # Default to diversified set
            return self._get_default_strategies()

        current_regime = self.last_regime.get("regime")
        regime_confidence = self.last_regime.get("confidence", 0.5)

        # Score each strategy
        strategy_scores = []

        for name, config in self.strategy_catalog.items():
            score = 0.0

            # Regime alignment
            if current_regime in config.preferred_regimes:
                score += 2.0
            elif current_regime in config.avoided_regimes:
                score -= 1.0

            # Performance score
            perf = self.performance_history.get(name)
            if perf and perf.num_trades > 5:
                # Recent performance matters more
                recent_returns = perf.returns[-min(20, len(perf.returns)) :]
                if recent_returns:
                    recent_sharpe = self._calculate_sharpe(recent_returns)
                    score += recent_sharpe * 2

                # Regime-specific performance
                if current_regime in perf.regime_performance:
                    regime_perf = perf.regime_performance[current_regime]
                    score += regime_perf * 1.5

            # Confidence multiplier
            score *= regime_confidence

            # Add to scores
            strategy_scores.append((name, max(0, score), config.weight))

        # Sort by score
        strategy_scores.sort(key=lambda x: x[1], reverse=True)

        if self.use_ensemble:
            # Take top N for ensemble
            n_strategies = min(self.ensemble_size, len(strategy_scores))
            selected = strategy_scores[:n_strategies]

            # Normalize weights
            total_score = sum(s[1] for s in selected)
            if total_score > 0:
                selected = [(name, score / total_score * weight) for name, score, weight in selected]
            else:
                selected = self._get_default_strategies()
        else:
            # Take single best strategy
            if strategy_scores and strategy_scores[0][1] > self.min_regime_confidence:
                selected = [(strategy_scores[0][0], 1.0)]
            else:
                selected = self._get_default_strategies()

        self.current_strategies = selected
        return selected

    def _get_default_strategies(self) -> List[Tuple[str, float]]:
        """Get default diversified strategy set"""
        return [("sma_crossover", 0.4), ("bb_mean_reversion", 0.3), ("volatility_breakout", 0.3)]

    def _get_strategy_instance(self, strategy_name: str) -> Optional[BaseStrategy]:
        """Get or create strategy instance"""
        if strategy_name not in self.strategy_instances:
            config = self.strategy_catalog.get(strategy_name)
            if config:
                strategy = config.strategy_class(name=strategy_name, params=config.default_params)
                self.strategy_instances[strategy_name] = strategy

        return self.strategy_instances.get(strategy_name)

    def _combine_signals(self, signals: List[int], weights: List[float]) -> Tuple[int, float]:
        """
        Combine multiple signals into one
        Returns: (combined_signal, confidence)
        """
        if not signals:
            return 0, 0.0

        # Weighted average of signals
        weighted_sum = sum(s * w for s, w in zip(signals, weights))
        total_weight = sum(weights)

        if total_weight == 0:
            return 0, 0.0

        avg_signal = weighted_sum / total_weight

        # Determine final signal
        if avg_signal > 0.3:
            final_signal = 1
        elif avg_signal < -0.3:
            final_signal = -1
        else:
            final_signal = 0

        # Calculate confidence based on agreement
        if final_signal != 0:
            # How many signals agree with final direction
            agreement = sum(1 for s in signals if s == final_signal) / len(signals)
            confidence = min(agreement * (1 + abs(avg_signal)), 0.95)
        else:
            confidence = 0.3

        return final_signal, confidence

    def _apply_risk_management(self, signal: int, confidence: float, data: pd.DataFrame) -> float:
        """
        Apply risk management rules to position size
        """
        if signal == 0:
            return 0.0

        # Base position size from confidence
        base_size = confidence

        # Volatility adjustment
        if len(data) > 20:
            returns = data["Close"].pct_change().dropna()
            recent_vol = returns.iloc[-20:].std() * np.sqrt(252)

            # Scale down in high volatility
            vol_multiplier = max(0.3, min(1.0, 0.2 / (recent_vol + 0.01)))
            base_size *= vol_multiplier

        # Drawdown adjustment
        if self.current_drawdown > 0.1:  # >10% drawdown
            drawdown_multiplier = max(0.2, 1 - self.current_drawdown)
            base_size *= drawdown_multiplier

        # Risk aversion adjustment
        base_size /= self.risk_aversion

        return float(np.clip(base_size, 0, 1))

    def _update_performance_tracking(
        self, signals: List[int], weights: List[float], current_price: Optional[float] = None, timestamp: Optional[datetime] = None
    ):
        """
        Update performance metrics for all strategies with sophisticated tracking

        This method tracks:
        1. Individual strategy performance
        2. Strategy correlations
        3. Regime-specific performance
        4. Risk metrics (Sharpe, Sortino, Max DD)
        5. Signal quality metrics
        """
        # Initialize data structures if first time
        if not hasattr(self, "_price_history"):
            self._price_history = []
            self._signal_history = {name: [] for name in self.strategy_catalog}
            self._strategy_returns = {name: [] for name in self.strategy_catalog}
            self._position_history = {name: [] for name in self.strategy_catalog}
            self._signal_quality = {name: [] for name in self.strategy_catalog}
            self._last_prices = {}
            self._strategy_weights_history = {name: [] for name in self.strategy_catalog}

        # Store timestamp
        current_time = timestamp or datetime.now()

        # Update price history
        if current_price is not None:
            self._price_history.append({"timestamp": current_time, "price": current_price})

            # Keep only last 1000 prices
            if len(self._price_history) > 1000:
                self._price_history = self._price_history[-1000:]

        # ===== PART 1: Update signals for ACTIVE strategies =====
        # Map active strategies to their signals
        for i, (strategy_name, weight_info) in enumerate(self.current_strategies):
            # Handle different possible formats of current_strategies
            if isinstance(strategy_name, tuple):
                name = strategy_name[0]
                base_weight = strategy_name[1] if len(strategy_name) > 1 else 1.0
            else:
                name = strategy_name
                base_weight = 1.0

            if i < len(signals):
                signal = signals[i]
                # Use provided weight or fall back to base_weight
                effective_weight = weights[i] if i < len(weights) else base_weight

                # Store signal for active strategy
                if name in self._signal_history:
                    self._signal_history[name].append(
                        {"timestamp": current_time, "signal": signal, "weight": effective_weight, "price": current_price, "is_active": True}
                    )

                    # Store weight history
                    if name in self._strategy_weights_history:
                        self._strategy_weights_history[name].append({"timestamp": current_time, "weight": effective_weight})

                    # Keep only last 500 signals
                    if len(self._signal_history[name]) > 500:
                        self._signal_history[name] = self._signal_history[name][-500:]

        # ===== PART 2: Record that INACTIVE strategies had no signal =====
        # This is important for calculating hit rates and strategy comparisons
        active_strategy_names = {s[0] if isinstance(s, tuple) else s for s in self.current_strategies}

        for strategy_name in self.strategy_catalog:
            if strategy_name not in active_strategy_names:
                # Record that this strategy had no signal (hold position)
                if strategy_name in self._signal_history:
                    self._signal_history[strategy_name].append(
                        {
                            "timestamp": current_time,
                            "signal": 0,  # No signal = hold
                            "weight": 0.0,  # No allocation
                            "price": current_price,
                            "is_active": False,
                        }
                    )

        # ===== PART 3: Calculate returns if we have previous price =====
        if len(self._price_history) >= 2:
            prev_price = self._price_history[-2]["price"]
            current_price = self._price_history[-1]["price"]

            if prev_price > 0:
                market_return = (current_price - prev_price) / prev_price

                # Update returns for ALL strategies based on their signals
                for strategy_name in self.strategy_catalog:
                    self._update_strategy_return(strategy_name, market_return, current_time)

                    # Update correlation data
                    self._update_correlation_data(strategy_name, market_return, current_time)

        # ===== PART 4: Periodically recalculate all metrics =====
        if len(self._price_history) % 20 == 0:  # Every 20 updates
            self._recalculate_all_metrics(current_time)

            # Log summary of active strategies
            if len(self.current_strategies) > 0:
                active_names = [s[0] if isinstance(s, tuple) else s for s in self.current_strategies]
                logger.debug(f"Active strategies: {active_names}")

    def _update_correlation_data(self, strategy_name: str, market_return: float, timestamp: datetime):
        """Update data needed for correlation calculations"""
        if not hasattr(self, "_strategy_correlation_data"):
            self._strategy_correlation_data = {name: [] for name in self.strategy_catalog}

        # Get the strategy's return for this period
        returns_data = self._strategy_returns.get(strategy_name, [])
        if returns_data and len(returns_data) > 0:
            latest_return = returns_data[-1]["return"]

            self._strategy_correlation_data[strategy_name].append(
                {"timestamp": timestamp, "strategy_return": latest_return, "market_return": market_return}
            )

            # Keep only last 252 days for correlation
            if len(self._strategy_correlation_data[strategy_name]) > 252:
                self._strategy_correlation_data[strategy_name] = self._strategy_correlation_data[strategy_name][-252:]

    def get_strategy_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix between strategy returns"""
        if not hasattr(self, "_strategy_correlation_data"):
            return pd.DataFrame()

        # Build returns matrix
        all_returns = {}
        common_timestamps = None

        for strategy_name, data in self._strategy_correlation_data.items():
            if len(data) < 20:  # Need minimum data
                continue

            # Extract returns with timestamps
            returns_dict = {entry["timestamp"]: entry["strategy_return"] for entry in data}

            if common_timestamps is None:
                common_timestamps = set(returns_dict.keys())
            else:
                common_timestamps &= set(returns_dict.keys())

            all_returns[strategy_name] = returns_dict

        if not common_timestamps or not all_returns:
            return pd.DataFrame()

        # Create aligned DataFrame
        timestamps = sorted(common_timestamps)
        data = {}

        for strategy_name, returns_dict in all_returns.items():
            data[strategy_name] = [returns_dict.get(ts, 0) for ts in timestamps]

        df = pd.DataFrame(data, index=timestamps)

        # Calculate correlation
        if len(df) > 1:
            return df.corr()
        return pd.DataFrame()

    def _update_strategy_return(self, strategy_name: str, market_return: float, timestamp: datetime):
        """Update return for a single strategy based on its position"""
        # Get recent signal
        signal_history = self._signal_history.get(strategy_name, [])
        if not signal_history:
            return

        # Get the last signal (this is the position held during this return period)
        last_signal = signal_history[-1]
        signal = last_signal["signal"]
        weight = last_signal.get("weight", 1.0)

        # Calculate strategy return
        # If signal is 1 (long), return = market_return
        # If signal is -1 (short), return = -market_return
        # If signal is 0 (neutral), return = 0
        if signal == 1:
            strategy_return = market_return * weight
        elif signal == -1:
            strategy_return = -market_return * weight
        else:
            strategy_return = 0.0

        # Store return
        if strategy_name in self._strategy_returns:
            self._strategy_returns[strategy_name].append(
                {"timestamp": timestamp, "return": strategy_return, "market_return": market_return, "signal": signal, "weight": weight}
            )

            # Keep only last 500 returns
            if len(self._strategy_returns[strategy_name]) > 500:
                self._strategy_returns[strategy_name] = self._strategy_returns[strategy_name][-500:]

        # Update position history
        if strategy_name in self._position_history:
            cumulative_return = 1.0
            if self._position_history[strategy_name]:
                last_position = self._position_history[strategy_name][-1]
                cumulative_return = last_position["cumulative_return"] * (1 + strategy_return)
            else:
                cumulative_return = 1 + strategy_return

            self._position_history[strategy_name].append(
                {"timestamp": timestamp, "position": signal, "return": strategy_return, "cumulative_return": cumulative_return}
            )

        # Update signal quality (did the signal predict direction correctly?)
        self._update_signal_quality(strategy_name, signal, market_return, timestamp)

        # Update regime-specific performance
        if self.last_regime:
            regime = self.last_regime.get("regime")
            if regime and strategy_name in self.performance_history:
                perf = self.performance_history[strategy_name]

                if regime not in perf.regime_performance:
                    perf.regime_performance[regime] = []

                regime_returns = perf.regime_performance[regime]
                regime_returns.append(strategy_return)

                # Keep only recent regime returns
                if len(regime_returns) > 50:
                    regime_returns = regime_returns[-50:]

                # Update average regime performance
                perf.regime_performance[regime] = np.mean(regime_returns)

    def _update_signal_quality(self, strategy_name: str, signal: int, market_return: float, timestamp: datetime):
        """Track signal quality metrics"""
        if strategy_name not in self._signal_quality:
            self._signal_quality[strategy_name] = []

        # Determine if signal was correct
        correct = False
        if signal == 1 and market_return > 0:
            correct = True
        elif signal == -1 and market_return < 0:
            correct = True
        elif signal == 0:
            correct = None  # Neutral signals are not judged

        if correct is not None:
            self._signal_quality[strategy_name].append(
                {"timestamp": timestamp, "correct": correct, "signal": signal, "return": abs(market_return) if correct else -abs(market_return)}
            )

            # Keep only last 200 signals
            if len(self._signal_quality[strategy_name]) > 200:
                self._signal_quality[strategy_name] = self._signal_quality[strategy_name][-200:]

    def _recalculate_all_metrics(self, current_time: datetime):
        """Recalculate all performance metrics for all strategies"""

        for strategy_name in self.strategy_catalog:
            if strategy_name not in self.performance_history:
                continue

            perf = self.performance_history[strategy_name]

            # Get returns data
            returns_data = self._strategy_returns.get(strategy_name, [])
            if len(returns_data) < 10:
                continue

            # Extract return values - BOTH recent and all
            recent_returns = [r["return"] for r in returns_data[-60:]]  # Last 60 for current assessment
            all_returns = [r["return"] for r in returns_data]  # All history for stability metrics

            if not recent_returns or not all_returns:
                continue

            # Convert to numpy arrays
            recent_array = np.array(recent_returns)
            all_array = np.array(all_returns)

            # ===== SHORT-TERM METRICS (recent performance) =====

            # Sharpe ratio (annualized) - recent
            if recent_array.std() > 0:
                perf.sharpe_ratio = float(recent_array.mean() / recent_array.std() * np.sqrt(252))
            else:
                perf.sharpe_ratio = 0.0

            # Sortino ratio (downside deviation) - recent
            downside_recent = recent_array[recent_array < 0]
            if len(downside_recent) > 0 and downside_recent.std() > 0:
                perf.sortino_ratio = float(recent_array.mean() / downside_recent.std() * np.sqrt(252))
            else:
                perf.sortino_ratio = perf.sharpe_ratio if perf.sharpe_ratio > 0 else 0.0

            # Win rate - recent
            wins_recent = sum(1 for r in recent_returns if r > 0)
            perf.win_rate = wins_recent / len(recent_returns) if recent_returns else 0.0

            # Average trade - recent
            perf.avg_trade = float(np.mean(recent_returns))

            # ===== LONG-TERM METRICS (overall stability) =====

            # Long-term Sharpe (for stability assessment)
            if all_array.std() > 0:
                perf.long_term_sharpe = float(all_array.mean() / all_array.std() * np.sqrt(252))
            else:
                perf.long_term_sharpe = 0.0

            # Long-term win rate
            wins_all = sum(1 for r in all_returns if r > 0)
            perf.long_term_win_rate = wins_all / len(all_returns) if all_returns else 0.0

            # Performance trend (is strategy improving or degrading?)
            if len(all_returns) >= 120:  # Need enough data for trend
                # Split into recent vs older
                older_returns = all_returns[:-60] if len(all_returns) > 60 else all_returns
                if len(older_returns) > 20:
                    older_sharpe = np.mean(older_returns) / (np.std(older_returns) + 1e-10) * np.sqrt(252)
                    perf.performance_trend = perf.sharpe_ratio - older_sharpe  # Positive = improving
                else:
                    perf.performance_trend = 0.0
            else:
                perf.performance_trend = 0.0

            # ===== DRAWDOWN ANALYSIS (using all data) =====

            # Max drawdown from cumulative returns
            position_data = self._position_history.get(strategy_name, [])
            if len(position_data) > 20:
                cum_returns = [p["cumulative_return"] for p in position_data]
                running_max = np.maximum.accumulate(cum_returns)
                drawdown = (cum_returns - running_max) / running_max
                perf.max_drawdown = float(abs(np.min(drawdown)))

                # Current drawdown
                perf.current_drawdown = float(abs(drawdown[-1])) if len(drawdown) > 0 else 0.0

                # Drawdown duration
                in_drawdown = False
                drawdown_start = None
                max_duration = 0

                for i, dd in enumerate(drawdown):
                    if dd < -0.01:  # More than 1% drawdown
                        if not in_drawdown:
                            in_drawdown = True
                            drawdown_start = i
                    else:
                        if in_drawdown:
                            in_drawdown = False
                            duration = i - drawdown_start
                            max_duration = max(max_duration, duration)

                perf.max_drawdown_days = max_duration

            # ===== RISK-ADJUSTED METRICS =====

            # Calmar ratio (return / max drawdown)
            if perf.max_drawdown > 0:
                annual_return = np.mean(recent_returns) * 252
                perf.calmar_ratio = float(annual_return / perf.max_drawdown)
            else:
                perf.calmar_ratio = 0.0

            # Information ratio (excess return over benchmark)
            # This would need benchmark data
            if hasattr(self, "_benchmark_returns") and self._benchmark_returns:
                # Align dates and calculate
                pass

            # ===== SIGNAL QUALITY METRICS =====

            signal_quality = self._signal_quality.get(strategy_name, [])
            if len(signal_quality) > 20:
                # Recent signal accuracy
                recent_signals = signal_quality[-20:]
                correct_signals = sum(1 for s in recent_signals if s.get("correct", False))
                perf.signal_accuracy = correct_signals / len(recent_signals) if recent_signals else 0.5

                # Long-term signal accuracy
                if len(signal_quality) > 60:
                    correct_all = sum(1 for s in signal_quality if s.get("correct", False))
                    perf.long_term_accuracy = correct_all / len(signal_quality)

                # Signal consistency (how often signals are generated)
                if hasattr(self, "_signal_history") and strategy_name in self._signal_history:
                    signals = self._signal_history[strategy_name]
                    trading_days = len(set(s.get("timestamp").date() for s in signals))
                    total_days = (current_time - perf.last_update).days + 1
                    perf.signal_frequency = trading_days / total_days if total_days > 0 else 0

            # ===== STABILITY METRICS =====

            # Rolling Sharpe volatility (how stable is the performance)
            if len(all_returns) > 120:
                rolling_sharpes = []
                for i in range(60, len(all_returns)):
                    window = all_returns[i - 60 : i]
                    if np.std(window) > 0:
                        rs = np.mean(window) / np.std(window) * np.sqrt(252)
                        rolling_sharpes.append(rs)

                if rolling_sharpes:
                    perf.sharpe_stability = 1.0 - (np.std(rolling_sharpes) / (abs(np.mean(rolling_sharpes)) + 1e-10))
                    perf.sharpe_stability = np.clip(perf.sharpe_stability, 0, 1)

            # ===== DECAY ANALYSIS (is strategy losing effectiveness?) =====

            if len(all_returns) >= 120:
                # Split into 4 quarters
                quarter_size = len(all_returns) // 4
                quarters = [
                    all_returns[:quarter_size],
                    all_returns[quarter_size : 2 * quarter_size],
                    all_returns[2 * quarter_size : 3 * quarter_size],
                    all_returns[3 * quarter_size :],
                ]

                quarter_sharpes = []
                for q in quarters:
                    if len(q) > 5 and np.std(q) > 0:
                        qs = np.mean(q) / np.std(q) * np.sqrt(252)
                        quarter_sharpes.append(qs)
                    else:
                        quarter_sharpes.append(0)

                # Linear trend in Sharpe ratios
                if len(quarter_sharpes) >= 4:
                    x = np.arange(len(quarter_sharpes))
                    z = np.polyfit(x, quarter_sharpes, 1)
                    perf.performance_decay = float(z[0])  # Negative = decaying

            # ===== CONSISTENCY METRICS =====

            # Up capture / Down capture (would need benchmark)
            # Profit factor
            gross_profit = sum(r for r in all_returns if r > 0)
            gross_loss = abs(sum(r for r in all_returns if r < 0))
            perf.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

            # Expectancy
            perf.expectancy = perf.avg_trade * perf.win_rate - perf.avg_trade * (1 - perf.win_rate)

            # Update timestamp
            perf.last_update = current_time

            # ===== LOGGING =====

            # Log significant changes
            if perf.sharpe_ratio > 2.0:
                logger.info(f"Strategy {strategy_name} has exceptional recent Sharpe: {perf.sharpe_ratio:.2f}")
            elif perf.sharpe_ratio < -1.0:
                logger.warning(f"Strategy {strategy_name} performing poorly: Sharpe {perf.sharpe_ratio:.2f}")

            # Log decay warning
            if hasattr(perf, "performance_decay") and perf.performance_decay < -0.1:
                logger.warning(f"Strategy {strategy_name} showing performance decay: {perf.performance_decay:.3f}")

            # Log consistency
            if perf.profit_factor > 2.0:
                logger.info(f"Strategy {strategy_name} has excellent profit factor: {perf.profit_factor:.2f}")

    def get_strategy_performance(self, strategy_name: Optional[str] = None) -> Dict:
        """
        Get detailed performance metrics for one or all strategies

        Args:
            strategy_name: Optional specific strategy name

        Returns:
            Dictionary with performance metrics
        """
        if strategy_name:
            if strategy_name not in self.performance_history:
                return {}

            perf = self.performance_history[strategy_name]

            # Get recent returns
            returns_data = self._strategy_returns.get(strategy_name, [])
            recent_returns = [r["return"] for r in returns_data[-20:]] if returns_data else []

            # Get signal history
            signal_history = self._signal_history.get(strategy_name, [])
            recent_signals = signal_history[-10:] if signal_history else []

            # Get position history
            position_history = self._position_history.get(strategy_name, [])

            return {
                "strategy": strategy_name,
                "metrics": {
                    "sharpe_ratio": perf.sharpe_ratio,
                    "sortino_ratio": getattr(perf, "sortino_ratio", 0),
                    "calmar_ratio": getattr(perf, "calmar_ratio", 0),
                    "win_rate": perf.win_rate,
                    "avg_trade": perf.avg_trade,
                    "max_drawdown": perf.max_drawdown,
                    "signal_accuracy": getattr(perf, "signal_accuracy", 0),
                    "num_trades": perf.num_trades,
                },
                "regime_performance": perf.regime_performance,
                "recent_returns": {
                    "values": recent_returns[-10:],
                    "mean": np.mean(recent_returns) if recent_returns else 0,
                    "std": np.std(recent_returns) if recent_returns else 0,
                },
                "recent_signals": [
                    {
                        "timestamp": s["timestamp"].isoformat() if hasattr(s["timestamp"], "isoformat") else str(s["timestamp"]),
                        "signal": s["signal"],
                        "weight": s.get("weight", 1.0),
                    }
                    for s in recent_signals
                ],
                "current_position": position_history[-1] if position_history else None,
                "last_update": perf.last_update.isoformat() if hasattr(perf.last_update, "isoformat") else str(perf.last_update),
            }
        else:
            # Return summary for all strategies
            summary = {}
            for name in self.strategy_catalog:
                summary[name] = self.get_strategy_performance(name)
            return summary

    def get_best_performing_strategies(self, n: int = 3, metric: str = "sharpe_ratio", min_trades: int = 10) -> List[Tuple[str, float]]:
        """
        Get best performing strategies based on specified metric

        Args:
            n: Number of strategies to return
            metric: Metric to use for ranking ('sharpe_ratio', 'win_rate', 'signal_accuracy')
            min_trades: Minimum number of trades required

        Returns:
            List of (strategy_name, score) tuples
        """
        scores = []

        for strategy_name, perf in self.performance_history.items():
            if perf.num_trades >= min_trades:
                if metric == "sharpe_ratio":
                    score = perf.sharpe_ratio
                elif metric == "win_rate":
                    score = perf.win_rate
                elif metric == "signal_accuracy":
                    score = getattr(perf, "signal_accuracy", 0)
                elif metric == "calmar_ratio":
                    score = getattr(perf, "calmar_ratio", 0)
                else:
                    score = 0

                if score > 0:
                    scores.append((strategy_name, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:n]

    def get_worst_performing_strategies(self, n: int = 3, metric: str = "sharpe_ratio", min_trades: int = 10) -> List[Tuple[str, float]]:
        """
        Get worst performing strategies for potential removal
        """
        scores = []

        for strategy_name, perf in self.performance_history.items():
            if perf.num_trades >= min_trades:
                if metric == "sharpe_ratio":
                    score = perf.sharpe_ratio
                elif metric == "win_rate":
                    score = perf.win_rate
                else:
                    score = 0

                scores.append((strategy_name, score))

        # Sort by score ascending
        scores.sort(key=lambda x: x[1])

        return scores[:n]

    def plot_strategy_performance(self, strategy_name: str, save_path: Optional[str] = None):
        """
        Generate performance plot for a strategy

        Args:
            strategy_name: Name of strategy to plot
            save_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt

            if strategy_name not in self._strategy_returns:
                logger.error(f"No data for strategy {strategy_name}")
                return

            returns_data = self._strategy_returns[strategy_name]
            if not returns_data:
                return

            # Create figure with subplots
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))

            # Plot 1: Cumulative returns
            timestamps = [r["timestamp"] for r in returns_data]
            cumulative = []
            cum = 1.0
            for r in returns_data:
                cum *= 1 + r["return"]
                cumulative.append(cum)

            axes[0].plot(timestamps, cumulative, "b-", linewidth=2)
            axes[0].set_title(f"{strategy_name} - Cumulative Returns")
            axes[0].set_ylabel("Cumulative Return")
            axes[0].grid(True, alpha=0.3)

            # Plot 2: Individual returns
            returns = [r["return"] * 100 for r in returns_data]  # Convert to %
            colors = ["g" if r > 0 else "r" for r in returns]
            axes[1].bar(range(len(returns)), returns, color=colors, alpha=0.6)
            axes[1].set_title("Individual Returns (%)")
            axes[1].set_ylabel("Return %")
            axes[1].grid(True, alpha=0.3)

            # Plot 3: Rolling Sharpe
            window = 20
            rolling_sharpe = []
            rolling_idx = []

            for i in range(window, len(returns) + 1):
                window_returns = returns[i - window : i]
                if np.std(window_returns) > 0:
                    sharpe = np.mean(window_returns) / np.std(window_returns) * np.sqrt(252)
                    rolling_sharpe.append(sharpe)
                    rolling_idx.append(i)

            axes[2].plot(rolling_idx, rolling_sharpe, "purple", linewidth=2)
            axes[2].axhline(y=1.0, color="g", linestyle="--", alpha=0.5, label="Good")
            axes[2].axhline(y=0.0, color="k", linestyle="-", alpha=0.3)
            axes[2].axhline(y=-1.0, color="r", linestyle="--", alpha=0.5, label="Poor")
            axes[2].set_title("Rolling Sharpe Ratio (20-period)")
            axes[2].set_ylabel("Sharpe Ratio")
            axes[2].set_xlabel("Trade Number")
            axes[2].grid(True, alpha=0.3)
            axes[2].legend()

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Plot saved to {save_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting performance: {e}")

    def export_performance_report(self, filepath: str):
        """
        Export comprehensive performance report to CSV

        Args:
            filepath: Path to save CSV file
        """
        report_data = []

        for strategy_name in self.strategy_catalog:
            perf = self.get_strategy_performance(strategy_name)
            if perf:
                row = {
                    "Strategy": strategy_name,
                    "Sharpe Ratio": perf["metrics"]["sharpe_ratio"],
                    "Sortino Ratio": perf["metrics"]["sortino_ratio"],
                    "Calmar Ratio": perf["metrics"]["calmar_ratio"],
                    "Win Rate": perf["metrics"]["win_rate"],
                    "Avg Trade": perf["metrics"]["avg_trade"],
                    "Max Drawdown": perf["metrics"]["max_drawdown"],
                    "Signal Accuracy": perf["metrics"]["signal_accuracy"],
                    "Num Trades": perf["metrics"]["num_trades"],
                    "Last Update": perf["last_update"],
                }

                # Add regime performance
                for regime, regime_return in perf.get("regime_performance", {}).items():
                    if isinstance(regime_return, (int, float)):
                        row[f"Regime_{regime}"] = regime_return

                report_data.append(row)

        if report_data:
            df = pd.DataFrame(report_data)
            df.to_csv(filepath, index=False)
            logger.info(f"Performance report exported to {filepath}")

            # Also create summary stats
            summary = {
                "Best Strategy": df.loc[df["Sharpe Ratio"].idxmax(), "Strategy"],
                "Best Sharpe": df["Sharpe Ratio"].max(),
                "Worst Strategy": df.loc[df["Sharpe Ratio"].idxmin(), "Strategy"],
                "Worst Sharpe": df["Sharpe Ratio"].min(),
                "Average Sharpe": df["Sharpe Ratio"].mean(),
                "Average Win Rate": df["Win Rate"].mean(),
                "Total Strategies": len(df),
            }

            summary_df = pd.DataFrame([summary])
            summary_df.to_csv(filepath.replace(".csv", "_summary.csv"), index=False)

            return summary

        return None

    def update_strategy_performance(self, strategy_name: str, return_value: float, regime: Optional[str] = None):
        """Update performance for a specific strategy"""
        if strategy_name in self.performance_history:
            perf = self.performance_history[strategy_name]
            perf.returns.append(return_value)
            perf.num_trades += 1

            # Keep only recent returns
            if len(perf.returns) > self.performance_lookback:
                perf.returns = perf.returns[-self.performance_lookback :]

            # Update metrics
            if len(perf.returns) > 20:
                perf.sharpe_ratio = self._calculate_sharpe(perf.returns[-60:])
                perf.win_rate = sum(1 for r in perf.returns[-60:] if r > 0) / min(60, len(perf.returns))
                perf.avg_trade = np.mean(perf.returns[-60:]) if perf.returns[-60:] else 0

            # Update regime-specific performance
            if regime:
                if regime not in perf.regime_performance:
                    perf.regime_performance[regime] = []

                regime_returns = perf.regime_performance[regime]
                regime_returns.append(return_value)

                # Keep only recent
                if len(regime_returns) > 20:
                    regime_returns = regime_returns[-20:]

                perf.regime_performance[regime] = np.mean(regime_returns)

    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio from returns"""
        if len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        if returns_array.std() == 0:
            return 0.0

        return float(returns_array.mean() / returns_array.std() * np.sqrt(252))

    def _prepare_metadata(self, selected_strategies: List[Tuple[str, float]], confidence: float, data: pd.DataFrame) -> Dict:
        """Prepare metadata for signal output"""
        # Get strategy details
        strategy_details = []
        for name, weight in selected_strategies:
            config = self.strategy_catalog.get(name)
            perf = self.performance_history.get(name)

            strategy_details.append(
                {
                    "name": name,
                    "type": config.strategy_type.value if config else "unknown",
                    "weight": weight,
                    "recent_sharpe": perf.sharpe_ratio if perf else 0,
                    "win_rate": perf.win_rate if perf else 0,
                }
            )

        # Get regime info
        regime_info = {}
        if self.last_regime:
            regime_info = {
                "current_regime": self.last_regime.get("regime"),
                "regime_confidence": self.last_regime.get("confidence", 0.5),
                "regime_duration": self.last_regime.get("duration", 0),
                "next_regime_probs": self.regime_detector.predict_next_regime_probs()
                if hasattr(self.regime_detector, "predict_next_regime_probs")
                else {},
            }

        # Calculate market conditions
        market_conditions = {}
        if len(data) > 20:
            returns = data["Close"].pct_change()
            market_conditions = {
                "volatility_21d": float(returns.iloc[-20:].std() * np.sqrt(252)),
                "trend_strength": float(data["Close"].iloc[-1] / data["Close"].iloc[-20] - 1),
                "volume_ratio": float(data["Volume"].iloc[-1] / data["Volume"].iloc[-20:].mean()),
            }

        return {
            "selected_strategies": strategy_details,
            "ensemble_size": len(selected_strategies),
            "combined_confidence": confidence,
            "regime": regime_info,
            "market_conditions": market_conditions,
            "current_drawdown": self.current_drawdown,
            "rebalance_needed": self._should_rebalance(datetime.now()),
        }

    def _should_rebalance(self, current_time: datetime) -> bool:
        """Check if strategy weights should be rebalanced"""
        if self.last_update is None:
            return True

        days_since = (current_time - self.last_update).days
        return days_since >= self.rebalance_frequency

    def reset(self):
        """Reset strategy state"""
        self.performance_history = {}
        self.current_strategies = []
        self.last_regime = None
        self.last_update = None
        self.strategy_instances = {}
        self.daily_returns = []
        self.current_drawdown = 0.0

        # Re-initialize performance tracking
        for strat_name in self.strategy_catalog:
            self.performance_history[strat_name] = StrategyPerformance(strategy_name=strat_name)

        # Reset regime detector
        self.regime_detector.regime_history = []

        logger.info("AdaptiveStrategySwitcher reset")

    def get_strategy_weights(self) -> Dict[str, float]:
        """Get current strategy weights"""
        return dict(self.current_strategies)

    def get_performance_summary(self) -> Dict:
        """Get performance summary for all strategies"""
        summary = {}

        for name, perf in self.performance_history.items():
            if perf.num_trades > 0:
                summary[name] = {
                    "sharpe_ratio": perf.sharpe_ratio,
                    "win_rate": perf.win_rate,
                    "avg_trade": perf.avg_trade,
                    "num_trades": perf.num_trades,
                    "regime_performance": perf.regime_performance,
                }

        return summary
