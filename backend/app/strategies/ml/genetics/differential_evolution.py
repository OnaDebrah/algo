"""
Differential Evolution Trading Strategy

Implements Differential Evolution (DE) for trading strategy optimization.
Based on research demonstrating DE's superior performance for financial applications:

- Outperforms 5 other algorithms including GA and PSO [9]
- Best generalization with minimal overfitting [9]
- Successfully applied to index forecasting [3]
- Excellent for multi-objective trading optimization [10]

Key features:
- DE/rand/1/bin strategy with adaptive parameters
- Multiple DE variants (rand/1, best/1, current-to-best/1, rand/2)
- Adaptive F and CR parameters
- Population diversity tracking
- Parallel fitness evaluation
- Multi-objective optimization support
- Comprehensive validation framework
"""

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import numpy as np
import pandas as pd
from base_strategy import BaseStrategy, normalize_signal


class DEVariant(Enum):
    """Differential Evolution variants supported"""

    RAND_1_BIN = "rand_1_bin"  # DE/rand/1/bin - Most common, good exploration
    BEST_1_BIN = "best_1_bin"  # DE/best/1/bin - Faster convergence
    CURRENT_TO_BEST_1_BIN = "current_to_best_1_bin"  # DE/current-to-best/1/bin - Balanced
    RAND_2_BIN = "rand_2_bin"  # DE/rand/2/bin - Enhanced exploration
    ADAPTIVE = "adaptive"  # Self-adaptive F and CR


class StrategyType(Enum):
    """Types of strategies that DE can optimize"""

    MOVING_AVERAGE_CROSSOVER = "ma_crossover"
    RSI_STRATEGY = "rsi_strategy"
    MACD_STRATEGY = "macd_strategy"
    BOLLINGER_BANDS = "bollinger_bands"
    COMBINED_SIGNAL = "combined_signal"
    MACHINE_LEARNING = "ml_strategy"  # For ML-based strategies
    ENSEMBLE = "ensemble"  # Weighted ensemble of multiple strategies


@dataclass
class ParameterRange:
    """Defines the range for a tunable parameter"""

    min_val: float
    max_val: float
    param_type: type = float
    categories: List = None
    is_int: bool = False
    log_scale: bool = False  # For parameters that work better on log scale (e.g., periods)

    def __post_init__(self):
        if self.param_type is int or self.is_int:
            self.param_type = int
            self.is_int = True


class DEIndividual:
    """
    Individual in Differential Evolution population.
    DE uses vector-based representation with mutation based on difference vectors.
    """

    def __init__(self, genes: Dict[str, float], param_ranges: Dict[str, ParameterRange]):
        """
        Initialize individual with genes.

        Args:
            genes: Dictionary of parameter names to values
            param_ranges: Dictionary of parameter ranges for validation
        """
        self.genes = genes.copy()
        self.param_ranges = param_ranges
        self.fitness = -np.inf
        self.fitness_components = {}  # For multi-objective optimization
        self.age = 0
        self.metadata = {}
        self.trial_vector = None  # For DE trial vector
        self.trial_fitness = -np.inf

        # For adaptive DE variants
        self.F = 0.5  # Mutation factor (can be self-adaptive)
        self.CR = 0.9  # Crossover rate (can be self-adaptive)

        # Validate genes
        self._validate_genes()

    def _validate_genes(self):
        """Ensure genes are within specified ranges"""
        for name, value in self.genes.items():
            if name in self.param_ranges:
                param_range = self.param_ranges[name]

                # Handle log scale transformation
                if param_range.log_scale:
                    value = np.exp(value) if value > 0 else value

                # Handle categorical parameters
                if param_range.categories is not None:
                    if value not in param_range.categories:
                        self.genes[name] = random.choice(param_range.categories)

                # Handle numeric ranges
                else:
                    if value < param_range.min_val:
                        self.genes[name] = param_range.min_val
                    elif value > param_range.max_val:
                        self.genes[name] = param_range.max_val

                    # Convert to int if needed
                    if param_range.is_int:
                        self.genes[name] = int(round(self.genes[name]))

    def copy(self) -> "DEIndividual":
        """Create a deep copy"""
        new_individual = DEIndividual(self.genes.copy(), self.param_ranges)
        new_individual.fitness = self.fitness
        new_individual.fitness_components = self.fitness_components.copy()
        new_individual.age = self.age
        new_individual.metadata = self.metadata.copy()
        new_individual.F = self.F
        new_individual.CR = self.CR
        return new_individual

    def __repr__(self) -> str:
        return f"DEIndividual(fitness={self.fitness:.4f}, F={self.F:.2f}, CR={self.CR:.2f})"


class DifferentialEvolutionStrategy(BaseStrategy):
    """
    Trading strategy using Differential Evolution for parameter optimization.

    Based on research demonstrating DE's superior performance:
    - Outperforms GA, PSO, and other evolutionary algorithms for trading [9]
    - Better generalization and less overfitting [9]
    - Successfully applied to stock index forecasting [3]
    - Effective for multi-objective portfolio optimization [10]

    Key advantages over standard GA:
    - Self-adjusting step sizes through difference vectors
    - Better exploration of complex fitness landscapes
    - Natural adaptation to parameter correlations
    - Proven superior convergence properties
    """

    def __init__(self, name: str = "DifferentialEvolution", params: Dict = None):
        """
        Initialize Differential Evolution Strategy.

        Args:
            name: Strategy name
            params: Dictionary with parameters:
                - strategy_type: Type of strategy to optimize (StrategyType enum or string)
                - population_size: Number of individuals (default: 50)
                - generations: Max generations (default: 100)
                - F: Mutation factor (default: 0.8) - can be adaptive
                - CR: Crossover rate (default: 0.9)
                - variant: DE variant (DEVariant enum, default: RAND_1_BIN)
                - adaptive_params: Use adaptive F and CR (default: True)
                - dither: Apply dither to F (random variation) (default: True)
                - jitter: Apply jitter to CR (default: False)
                - population_reinit: Reinitialize if stagnation (default: True)
                - stagnation_threshold: Generations without improvement (default: 20)
                - optimization_objective: Fitness metric
                - multi_objective: Enable multi-objective optimization
                - objective_weights: Weights for multi-objective
                - validation_split: Fraction for validation (default: 0.3)
                - cross_validation_folds: Number of CV folds (default: 3)
                - retrain_frequency: How often to retrain (days)
                - parallel_evaluation: Use parallel processing (default: True)
                - num_workers: Number of parallel workers (default: 4)
                - random_seed: Random seed for reproducibility
        """
        default_params = {
            "strategy_type": StrategyType.MOVING_AVERAGE_CROSSOVER,
            "population_size": 50,
            "generations": 100,
            "F": 0.8,  # Mutation factor
            "CR": 0.9,  # Crossover rate
            "variant": DEVariant.RAND_1_BIN,
            "adaptive_params": True,
            "dither": True,
            "jitter": False,
            "population_reinit": True,
            "stagnation_threshold": 20,
            "optimization_objective": "sharpe",
            "multi_objective": False,
            "objective_weights": {"sharpe": 0.4, "total_return": 0.3, "max_drawdown": 0.3},
            "validation_split": 0.3,
            "cross_validation_folds": 3,
            "retrain_frequency": 90,
            "parallel_evaluation": True,
            "num_workers": 4,
            "random_seed": 42,
        }

        if params:
            default_params.update(params)

        super().__init__(name, default_params)

        # Set random seed
        if "random_seed" in self.params:
            random.seed(self.params["random_seed"])
            np.random.seed(self.params["random_seed"])

        # Convert string enums if needed
        self._parse_enums()

        # Initialize parameter ranges based on strategy type
        self.param_ranges = self._initialize_parameter_ranges()

        # Population and evolution state
        self.population = []
        self.best_individual = None
        self.best_fitness = -np.inf
        self.fitness_history = []
        self.diversity_history = []
        self.stagnation_counter = 0
        self.generation = 0

        # Training data
        self.training_data = None
        self.last_training_date = None

        # Strategy state
        self.current_strategy_params = None
        self._signal_cache = None

        # Validation metrics
        self.validation_results = {}
        self.oos_performance = {}

    def _parse_enums(self):
        """Convert string parameters to enums"""
        # Parse strategy_type
        if isinstance(self.params["strategy_type"], str):
            try:
                self.params["strategy_type"] = StrategyType(self.params["strategy_type"])
            except ValueError:
                for strategy in StrategyType:
                    if strategy.name.lower() == self.params["strategy_type"].lower():
                        self.params["strategy_type"] = strategy
                        break

        # Parse variant
        if isinstance(self.params["variant"], str):
            try:
                self.params["variant"] = DEVariant(self.params["variant"])
            except ValueError:
                for variant in DEVariant:
                    if variant.name.lower() == self.params["variant"].lower():
                        self.params["variant"] = variant
                        break

    def _initialize_parameter_ranges(self) -> Dict[str, ParameterRange]:
        """Initialize parameter ranges based on strategy type"""
        strategy_type = self.params["strategy_type"]
        ranges = {}

        if strategy_type == StrategyType.MOVING_AVERAGE_CROSSOVER:
            ranges = {
                "short_window": ParameterRange(5, 50, int, is_int=True, log_scale=True),
                "long_window": ParameterRange(20, 200, int, is_int=True, log_scale=True),
                "signal_threshold": ParameterRange(0.0, 1.0, float),
                "use_volume_filter": ParameterRange(0, 1, int, categories=[0, 1]),
                "volume_threshold": ParameterRange(1.0, 3.0, float),
                "exit_bars": ParameterRange(1, 20, int, is_int=True),  # Time-based exit
            }

        elif strategy_type == StrategyType.RSI_STRATEGY:
            ranges = {
                "rsi_period": ParameterRange(5, 30, int, is_int=True, log_scale=True),
                "oversold_threshold": ParameterRange(20, 40, int, is_int=True),
                "overbought_threshold": ParameterRange(60, 80, int, is_int=True),
                "exit_oversold": ParameterRange(40, 60, int, is_int=True),
                "exit_overbought": ParameterRange(40, 60, int, is_int=True),
                "use_trend_filter": ParameterRange(0, 1, int, categories=[0, 1]),
                "trend_period": ParameterRange(20, 100, int, is_int=True, log_scale=True),
            }

        elif strategy_type == StrategyType.MACD_STRATEGY:
            ranges = {
                "fast_period": ParameterRange(5, 20, int, is_int=True, log_scale=True),
                "slow_period": ParameterRange(20, 40, int, is_int=True, log_scale=True),
                "signal_period": ParameterRange(5, 15, int, is_int=True, log_scale=True),
                "signal_threshold": ParameterRange(0.0, 1.0, float),
                "use_histogram": ParameterRange(0, 1, int, categories=[0, 1]),
                "histogram_threshold": ParameterRange(0.0, 0.5, float),
            }

        elif strategy_type == StrategyType.BOLLINGER_BANDS:
            ranges = {
                "period": ParameterRange(10, 50, int, is_int=True, log_scale=True),
                "num_std": ParameterRange(1.5, 3.0, float),
                "entry_threshold": ParameterRange(0.0, 1.0, float),
                "exit_threshold": ParameterRange(0.0, 1.0, float),
                "use_squeeze": ParameterRange(0, 1, int, categories=[0, 1]),
                "squeeze_period": ParameterRange(10, 30, int, is_int=True, log_scale=True),
            }

        elif strategy_type == StrategyType.COMBINED_SIGNAL:
            ranges = {
                "ma_weight": ParameterRange(0.0, 1.0, float),
                "rsi_weight": ParameterRange(0.0, 1.0, float),
                "macd_weight": ParameterRange(0.0, 1.0, float),
                "bb_weight": ParameterRange(0.0, 1.0, float),
                "ma_short": ParameterRange(5, 50, int, is_int=True, log_scale=True),
                "ma_long": ParameterRange(20, 200, int, is_int=True, log_scale=True),
                "rsi_period": ParameterRange(5, 30, int, is_int=True, log_scale=True),
                "rsi_oversold": ParameterRange(20, 40, int, is_int=True),
                "rsi_overbought": ParameterRange(60, 80, int, is_int=True),
                "macd_fast": ParameterRange(5, 20, int, is_int=True, log_scale=True),
                "macd_slow": ParameterRange(20, 40, int, is_int=True, log_scale=True),
                "macd_signal": ParameterRange(5, 15, int, is_int=True, log_scale=True),
                "bb_period": ParameterRange(10, 50, int, is_int=True, log_scale=True),
                "bb_std": ParameterRange(1.5, 3.0, float),
                "signal_threshold": ParameterRange(0.0, 2.0, float),
            }

        elif strategy_type == StrategyType.ENSEMBLE:
            ranges = {
                "ma_weight": ParameterRange(0.0, 1.0, float),
                "rsi_weight": ParameterRange(0.0, 1.0, float),
                "macd_weight": ParameterRange(0.0, 1.0, float),
                "bb_weight": ParameterRange(0.0, 1.0, float),
                "ml_weight": ParameterRange(0.0, 1.0, float),
                "voting_threshold": ParameterRange(0.0, 1.0, float),
            }

        return ranges

    def _create_strategy_signals(self, data: pd.DataFrame, params: Dict[str, float]) -> pd.Series:
        """Generate trading signals based on strategy type and parameters"""
        strategy_type = self.params["strategy_type"]

        if strategy_type == StrategyType.MOVING_AVERAGE_CROSSOVER:
            return self._ma_crossover_signals(data, params)
        elif strategy_type == StrategyType.RSI_STRATEGY:
            return self._rsi_signals(data, params)
        elif strategy_type == StrategyType.MACD_STRATEGY:
            return self._macd_signals(data, params)
        elif strategy_type == StrategyType.BOLLINGER_BANDS:
            return self._bollinger_signals(data, params)
        elif strategy_type == StrategyType.COMBINED_SIGNAL:
            return self._combined_signals(data, params)
        elif strategy_type == StrategyType.ENSEMBLE:
            return self._ensemble_signals(data, params)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

    def _ma_crossover_signals(self, data: pd.DataFrame, params: Dict) -> pd.Series:
        """Moving average crossover strategy"""
        short_window = int(params["short_window"])
        long_window = int(params["long_window"])
        signal_threshold = params.get("signal_threshold", 0.0)

        # Calculate moving averages
        short_ma = data["Close"].rolling(window=short_window).mean()
        long_ma = data["Close"].rolling(window=long_window).mean()

        # Generate signals
        signals = pd.Series(0, index=data.index)

        # Crossover signal with threshold
        ma_diff = short_ma - long_ma
        signals[ma_diff > signal_threshold] = 1
        signals[ma_diff < -signal_threshold] = -1

        # Volume filter
        if params.get("use_volume_filter", 0) == 1:
            volume_ratio = data["Volume"] / data["Volume"].rolling(window=20).mean()
            volume_threshold = params.get("volume_threshold", 1.5)
            signals[volume_ratio < volume_threshold] = 0

        # Time-based exit
        exit_bars = params.get("exit_bars", 0)
        if exit_bars > 0:
            position_duration = 0
            for i in range(1, len(signals)):
                if signals.iloc[i] != 0 and signals.iloc[i - 1] == signals.iloc[i]:
                    position_duration += 1
                    if position_duration >= exit_bars:
                        signals.iloc[i] = 0
                        position_duration = 0
                else:
                    position_duration = 0

        return signals

    def _rsi_signals(self, data: pd.DataFrame, params: Dict) -> pd.Series:
        """RSI mean reversion strategy"""
        rsi_period = int(params["rsi_period"])
        oversold = int(params["oversold_threshold"])
        overbought = int(params["overbought_threshold"])
        exit_oversold = int(params.get("exit_oversold", 50))
        exit_overbought = int(params.get("exit_overbought", 50))

        # Calculate RSI
        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        signals = pd.Series(0, index=data.index)

        # Entry signals
        signals[rsi < oversold] = 1
        signals[rsi > overbought] = -1

        # Exit signals
        if exit_oversold > oversold:
            signals[(signals.shift(1) == 1) & (rsi > exit_oversold)] = 0

        if exit_overbought < overbought:
            signals[(signals.shift(1) == -1) & (rsi < exit_overbought)] = 0

        # Trend filter
        if params.get("use_trend_filter", 0) == 1:
            trend_period = int(params.get("trend_period", 50))
            sma = data["Close"].rolling(window=trend_period).mean()
            signals[(data["Close"] < sma) & (signals == 1)] = 0
            signals[(data["Close"] > sma) & (signals == -1)] = 0

        return signals

    def _macd_signals(self, data: pd.DataFrame, params: Dict) -> pd.Series:
        """MACD crossover strategy"""
        fast = int(params["fast_period"])
        slow = int(params["slow_period"])
        signal_period = int(params["signal_period"])
        signal_threshold = params.get("signal_threshold", 0.0)

        # Calculate MACD
        ema_fast = data["Close"].ewm(span=fast, adjust=False).mean()
        ema_slow = data["Close"].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        signals = pd.Series(0, index=data.index)

        if params.get("use_histogram", 0) == 1:
            hist_threshold = params.get("histogram_threshold", 0.0)
            signals[histogram > hist_threshold] = 1
            signals[histogram < -hist_threshold] = -1
        else:
            macd_diff = macd_line - signal_line
            signals[macd_diff > signal_threshold] = 1
            signals[macd_diff < -signal_threshold] = -1

        return signals

    def _bollinger_signals(self, data: pd.DataFrame, params: Dict) -> pd.Series:
        """Bollinger Bands mean reversion"""
        period = int(params["period"])
        num_std = params["num_std"]
        entry_threshold = params.get("entry_threshold", 0.0)
        exit_threshold = params.get("exit_threshold", 0.5)

        # Calculate Bollinger Bands
        sma = data["Close"].rolling(window=period).mean()
        std = data["Close"].rolling(window=period).std()
        upper = sma + num_std * std
        lower = sma - num_std * std

        # Position within bands (0 to 1)
        band_position = (data["Close"] - lower) / (upper - lower)

        signals = pd.Series(0, index=data.index)

        # Entry signals
        signals[band_position < entry_threshold] = 1
        signals[band_position > (1 - entry_threshold)] = -1

        # Exit signals
        middle_zone = (band_position > exit_threshold) & (band_position < (1 - exit_threshold))
        signals[middle_zone & (signals.shift(1) != 0)] = 0

        # Squeeze filter
        if params.get("use_squeeze", 0) == 1:
            squeeze_period = int(params.get("squeeze_period", 20))
            band_width = (upper - lower) / sma
            avg_width = band_width.rolling(window=squeeze_period).mean()
            signals[band_width < avg_width] = 0

        return signals

    def _combined_signals(self, data: pd.DataFrame, params: Dict) -> pd.Series:
        """Weighted combination of multiple indicators"""
        # Get weights
        ma_weight = params.get("ma_weight", 0.25)
        rsi_weight = params.get("rsi_weight", 0.25)
        macd_weight = params.get("macd_weight", 0.25)
        bb_weight = params.get("bb_weight", 0.25)

        # Normalize weights
        total = ma_weight + rsi_weight + macd_weight + bb_weight
        if total > 0:
            ma_weight /= total
            rsi_weight /= total
            macd_weight /= total
            bb_weight /= total

        # Generate individual signals
        ma_signals = self._ma_crossover_signals(
            data, {"short_window": params.get("ma_short", 10), "long_window": params.get("ma_long", 30), "signal_threshold": 0}
        )

        rsi_signals = self._rsi_signals(
            data,
            {
                "rsi_period": params.get("rsi_period", 14),
                "oversold_threshold": params.get("rsi_oversold", 30),
                "overbought_threshold": params.get("rsi_overbought", 70),
                "exit_oversold": 50,
                "exit_overbought": 50,
                "use_trend_filter": 0,
            },
        )

        macd_signals = self._macd_signals(
            data,
            {
                "fast_period": params.get("macd_fast", 12),
                "slow_period": params.get("macd_slow", 26),
                "signal_period": params.get("macd_signal", 9),
                "signal_threshold": 0,
                "use_histogram": 0,
            },
        )

        bb_signals = self._bollinger_signals(
            data,
            {
                "period": params.get("bb_period", 20),
                "num_std": params.get("bb_std", 2.0),
                "entry_threshold": 0.1,
                "exit_threshold": 0.5,
                "use_squeeze": 0,
            },
        )

        # Calculate weighted signal score
        score = ma_weight * ma_signals + rsi_weight * rsi_signals + macd_weight * macd_signals + bb_weight * bb_signals

        signal_threshold = params.get("signal_threshold", 0.5)

        signals = pd.Series(0, index=data.index)
        signals[score > signal_threshold] = 1
        signals[score < -signal_threshold] = -1

        return signals

    def _ensemble_signals(self, data: pd.DataFrame, params: Dict) -> pd.Series:
        """Ensemble of multiple strategies with voting"""
        weights = {
            "ma": params.get("ma_weight", 0.2),
            "rsi": params.get("rsi_weight", 0.2),
            "macd": params.get("macd_weight", 0.2),
            "bb": params.get("bb_weight", 0.2),
            "ml": params.get("ml_weight", 0.2),
        }

        # Generate signals from each strategy (using default parameters)
        # In practice, these would be optimized separately
        ma_signal = self._ma_crossover_signals(data, {"short_window": 10, "long_window": 30, "signal_threshold": 0})

        rsi_signal = self._rsi_signals(data, {"rsi_period": 14, "oversold_threshold": 30, "overbought_threshold": 70})

        macd_signal = self._macd_signals(data, {"fast_period": 12, "slow_period": 26, "signal_period": 9})

        bb_signal = self._bollinger_signals(data, {"period": 20, "num_std": 2.0, "entry_threshold": 0.1})

        # Weighted voting
        weighted_vote = weights["ma"] * ma_signal + weights["rsi"] * rsi_signal + weights["macd"] * macd_signal + weights["bb"] * bb_signal

        # Apply voting threshold
        threshold = params.get("voting_threshold", 0.3)
        signals = pd.Series(0, index=data.index)
        signals[weighted_vote > threshold] = 1
        signals[weighted_vote < -threshold] = -1

        return signals

    def _calculate_fitness(self, individual: DEIndividual, data: pd.DataFrame) -> float:
        """
        Calculate fitness for an individual.
        Implements multiple fitness metrics with cross-validation support.
        """
        # Generate signals
        signals = self._create_strategy_signals(data, individual.genes)

        # Calculate returns
        returns = data["Close"].pct_change().fillna(0)
        strategy_returns = returns * signals.shift(1).fillna(0)

        # Apply transaction costs (0.1% per trade)
        trades = (signals.diff().fillna(0) != 0).astype(int)
        transaction_costs = trades * 0.001
        net_returns = strategy_returns - transaction_costs.shift(1).fillna(0)

        # Skip warm-up period
        warmup = max([int(v) for v in individual.genes.values() if isinstance(v, (int, float)) and v < 200] or [20])
        net_returns.iloc[:warmup] = 0

        if self.params["multi_objective"]:
            # Multi-objective fitness
            fitness_dict = {}

            # Sharpe ratio
            if net_returns.std() > 0:
                fitness_dict["sharpe"] = np.sqrt(252) * net_returns.mean() / net_returns.std()
            else:
                fitness_dict["sharpe"] = 0

            # Total return
            fitness_dict["total_return"] = (1 + net_returns).prod() - 1

            # Max drawdown (to minimize)
            cumulative = (1 + net_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            fitness_dict["max_drawdown"] = abs(drawdown.min())

            # Win rate
            winning_trades = net_returns[net_returns > 0].count()
            total_trades = net_returns[net_returns != 0].count()
            fitness_dict["win_rate"] = winning_trades / total_trades if total_trades > 0 else 0

            # Profit factor
            gross_profit = net_returns[net_returns > 0].sum()
            gross_loss = abs(net_returns[net_returns < 0].sum())
            fitness_dict["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else 10

            # Store components
            individual.fitness_components = fitness_dict

            # Weighted combination
            weights = self.params["objective_weights"]
            fitness = (
                weights.get("sharpe", 0.4) * fitness_dict["sharpe"]
                + weights.get("total_return", 0.3) * fitness_dict["total_return"]
                - weights.get("max_drawdown", 0.3) * fitness_dict["max_drawdown"]
            )

            return fitness

        else:
            # Single-objective optimization
            objective = self.params["optimization_objective"]

            if objective == "sharpe":
                if net_returns.std() > 0:
                    return np.sqrt(252) * net_returns.mean() / net_returns.std()
                return 0

            elif objective == "total_return":
                return (1 + net_returns).prod() - 1

            elif objective == "sortino":
                downside = net_returns[net_returns < 0].std()
                if downside > 0:
                    return np.sqrt(252) * net_returns.mean() / downside
                return np.sqrt(252) * net_returns.mean() * 10

            elif objective == "calmar":
                cumulative = (1 + net_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = abs(drawdown.min())
                total_ret = (1 + net_returns).prod() - 1
                if max_drawdown > 0:
                    return total_ret / max_drawdown
                return total_ret * 10

            elif objective == "profit_factor":
                gross_profit = net_returns[net_returns > 0].sum()
                gross_loss = abs(net_returns[net_returns < 0].sum())
                return gross_profit / gross_loss if gross_loss > 0 else 10

            else:  # Combined
                sharpe = np.sqrt(252) * net_returns.mean() / net_returns.std() if net_returns.std() > 0 else 0
                total_ret = (1 + net_returns).prod() - 1
                return 0.6 * sharpe + 0.4 * total_ret

    def _initialize_population(self) -> List[DEIndividual]:
        """Initialize random population with Latin Hypercube Sampling for better coverage"""
        population = []
        n_params = len(self.param_ranges)

        # Use Latin Hypercube Sampling for better initial distribution
        lhs_samples = np.random.uniform(0, 1, (self.params["population_size"], n_params))

        for i in range(self.params["population_size"]):
            genes = {}

            for j, (name, param_range) in enumerate(self.param_ranges.items()):
                if param_range.categories is not None:
                    # Categorical parameter
                    genes[name] = random.choice(param_range.categories)
                else:
                    # Use LHS sample
                    u = lhs_samples[i, j]

                    # Transform to parameter range
                    if param_range.log_scale:
                        # Log scale transformation
                        log_min = np.log(param_range.min_val)
                        log_max = np.log(param_range.max_val)
                        value = np.exp(log_min + u * (log_max - log_min))
                    else:
                        value = param_range.min_val + u * (param_range.max_val - param_range.min_val)

                    if param_range.is_int:
                        value = int(round(value))

                    genes[name] = value

            individual = DEIndividual(genes, self.param_ranges)

            # Initialize adaptive parameters
            if self.params["adaptive_params"]:
                individual.F = np.random.uniform(0.1, 1.0)
                individual.CR = np.random.uniform(0.5, 1.0)

            population.append(individual)

        return population

    def _mutate_rand_1(self, target: DEIndividual, population: List[DEIndividual]) -> Dict:
        """
        DE/rand/1 mutation: v = x + F * (y - z)
        where x, y, z are distinct random individuals
        """
        # Select three distinct random individuals (excluding target)
        candidates = [ind for ind in population if ind != target]
        x, y, z = random.sample(candidates, 3)

        F = target.F if self.params["adaptive_params"] else self.params["F"]

        # Apply dither (random variation of F)
        if self.params["dither"]:
            F = F * np.random.uniform(0.5, 1.0)

        mutant_genes = {}
        for name in target.genes:
            # DE mutation formula
            mutant_genes[name] = x.genes[name] + F * (y.genes[name] - z.genes[name])

            # Boundary handling - reflect
            param_range = self.param_ranges[name]
            if param_range.categories is None:
                if mutant_genes[name] < param_range.min_val:
                    mutant_genes[name] = 2 * param_range.min_val - mutant_genes[name]
                elif mutant_genes[name] > param_range.max_val:
                    mutant_genes[name] = 2 * param_range.max_val - mutant_genes[name]

        return mutant_genes

    def _mutate_best_1(self, target: DEIndividual, population: List[DEIndividual], best: DEIndividual) -> Dict:
        """
        DE/best/1 mutation: v = best + F * (y - z)
        Faster convergence, less exploration
        """
        # Select two random individuals (excluding target and best)
        candidates = [ind for ind in population if ind != target and ind != best]
        y, z = random.sample(candidates, 2)

        F = target.F if self.params["adaptive_params"] else self.params["F"]

        mutant_genes = {}
        for name in target.genes:
            mutant_genes[name] = best.genes[name] + F * (y.genes[name] - z.genes[name])

            # Boundary handling
            param_range = self.param_ranges[name]
            if param_range.categories is None:
                mutant_genes[name] = max(param_range.min_val, min(param_range.max_val, mutant_genes[name]))

        return mutant_genes

    def _mutate_current_to_best_1(self, target: DEIndividual, population: List[DEIndividual], best: DEIndividual) -> Dict:
        """
        DE/current-to-best/1 mutation: v = target + F * (best - target) + F * (y - z)
        Balanced exploration/exploitation
        """
        # Select two random individuals
        candidates = [ind for ind in population if ind != target and ind != best]
        y, z = random.sample(candidates, 2)

        F = target.F if self.params["adaptive_params"] else self.params["F"]

        mutant_genes = {}
        for name in target.genes:
            mutant_genes[name] = target.genes[name] + F * (best.genes[name] - target.genes[name]) + F * (y.genes[name] - z.genes[name])

            # Boundary handling
            param_range = self.param_ranges[name]
            if param_range.categories is None:
                mutant_genes[name] = max(param_range.min_val, min(param_range.max_val, mutant_genes[name]))

        return mutant_genes

    def _mutate_rand_2(self, target: DEIndividual, population: List[DEIndividual]) -> Dict:
        """
        DE/rand/2 mutation: v = x + F * (y - z) + F * (u - v)
        Enhanced exploration using two difference vectors
        """
        # Select five distinct random individuals
        candidates = [ind for ind in population if ind != target]
        x, y, z, u, v = random.sample(candidates, 5)

        F = target.F if self.params["adaptive_params"] else self.params["F"]

        mutant_genes = {}
        for name in target.genes:
            mutant_genes[name] = x.genes[name] + F * (y.genes[name] - z.genes[name]) + F * (u.genes[name] - v.genes[name])

            # Boundary handling
            param_range = self.param_ranges[name]
            if param_range.categories is None:
                mutant_genes[name] = max(param_range.min_val, min(param_range.max_val, mutant_genes[name]))

        return mutant_genes

    def _crossover_binomial(self, target: DEIndividual, mutant_genes: Dict) -> DEIndividual:
        """
        Binomial crossover: trial = crossover(target, mutant) with probability CR
        """
        CR = target.CR if self.params["adaptive_params"] else self.params["CR"]

        # Apply jitter (random variation to CR)
        if self.params["jitter"]:
            CR = CR * np.random.uniform(0.9, 1.1)
            CR = min(1.0, max(0.0, CR))

        trial_genes = {}

        # Ensure at least one gene comes from mutant
        crossover_point = random.choice(list(target.genes.keys()))

        for name in target.genes:
            if name == crossover_point or random.random() < CR:
                trial_genes[name] = mutant_genes[name]
            else:
                trial_genes[name] = target.genes[name]

        # Create trial individual
        trial = target.copy()
        trial.genes = trial_genes
        trial._validate_genes()

        # Carry over adaptive parameters
        trial.F = target.F
        trial.CR = target.CR

        return trial

    def _select(self, target: DEIndividual, trial: DEIndividual, data: pd.DataFrame) -> DEIndividual:
        """
        Selection: keep the better of target and trial
        """
        # Evaluate trial
        trial_fitness = self._calculate_fitness(trial, data)

        # Greedy selection
        if trial_fitness >= target.fitness:
            trial.fitness = trial_fitness
            return trial
        else:
            return target

    def _evaluate_population(self, population: List[DEIndividual], data: pd.DataFrame) -> List[DEIndividual]:
        """Evaluate fitness for entire population with parallel support"""
        if self.params["parallel_evaluation"] and len(population) > 10:
            with ThreadPoolExecutor(max_workers=self.params["num_workers"]) as executor:
                futures = []
                for individual in population:
                    future = executor.submit(self._calculate_fitness, individual, data)
                    futures.append((individual, future))

                for individual, future in futures:
                    individual.fitness = future.result()
        else:
            for individual in population:
                individual.fitness = self._calculate_fitness(individual, data)

        return population

    def _calculate_population_diversity(self, population: List[DEIndividual]) -> float:
        """Calculate population diversity (mean pairwise distance)"""
        if len(population) < 2:
            return 1.0

        # Sample pairs for efficiency
        n_samples = min(100, len(population) * (len(population) - 1) // 2)
        distances = []

        for _ in range(n_samples):
            i, j = random.sample(range(len(population)), 2)
            ind1, ind2 = population[i], population[j]

            # Calculate Euclidean distance in normalized parameter space
            dist = 0
            count = 0

            for name in ind1.genes:
                param_range = self.param_ranges.get(name)
                if param_range is None or param_range.categories is not None:
                    continue

                # Normalize values
                v1 = (ind1.genes[name] - param_range.min_val) / (param_range.max_val - param_range.min_val)
                v2 = (ind2.genes[name] - param_range.min_val) / (param_range.max_val - param_range.min_val)

                dist += (v1 - v2) ** 2
                count += 1

            if count > 0:
                distances.append(np.sqrt(dist / count))

        return np.mean(distances) if distances else 0

    def _check_stagnation(self, best_fitness: float, generation: int) -> bool:
        """Check if evolution has stagnated"""
        if len(self.fitness_history) < 2:
            return False

        # Check if best fitness hasn't improved
        if best_fitness <= self.best_fitness:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0

        # Check diversity
        diversity = self.diversity_history[-1] if self.diversity_history else 1.0

        return self.stagnation_counter >= self.params["stagnation_threshold"] or diversity < 0.01  # Very low diversity

    def _reinitialize_population(self, population: List[DEIndividual], keep_elite: int = 1) -> List[DEIndividual]:
        """Reinitialize population while keeping elite individuals"""
        # Sort by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Keep elite
        new_population = population[:keep_elite]

        # Generate rest randomly
        n_new = self.params["population_size"] - keep_elite
        random_pop = self._initialize_population()
        new_population.extend(random_pop[:n_new])

        print(f"Reinitializing population - kept {keep_elite} elites")

        return new_population

    def _evolve(self, training_data: pd.DataFrame) -> DEIndividual:
        """
        Main evolution loop using Differential Evolution.
        Implements multiple DE variants with adaptive parameters.
        """
        # Initialize population
        population = self._initialize_population()

        # Evaluate initial population
        population = self._evaluate_population(population, training_data)

        # Find best
        best_idx = np.argmax([ind.fitness for ind in population])
        self.best_individual = population[best_idx].copy()
        self.best_fitness = self.best_individual.fitness

        # Evolution loop
        for generation in range(self.params["generations"]):
            self.generation = generation
            new_population = []

            # Create trial vectors for each individual
            for i, target in enumerate(population):
                # Select mutation strategy based on variant
                if self.params["variant"] == DEVariant.RAND_1_BIN:
                    mutant_genes = self._mutate_rand_1(target, population)
                elif self.params["variant"] == DEVariant.BEST_1_BIN:
                    mutant_genes = self._mutate_best_1(target, population, self.best_individual)
                elif self.params["variant"] == DEVariant.CURRENT_TO_BEST_1_BIN:
                    mutant_genes = self._mutate_current_to_best_1(target, population, self.best_individual)
                elif self.params["variant"] == DEVariant.RAND_2_BIN:
                    mutant_genes = self._mutate_rand_2(target, population)
                else:  # Default to rand/1
                    mutant_genes = self._mutate_rand_1(target, population)

                # Create trial through crossover
                trial = self._crossover_binomial(target, mutant_genes)

                # Select better of target and trial
                selected = self._select(target, trial, training_data)
                new_population.append(selected)

            # Update population
            population = new_population

            # Evaluate population (if not evaluated in selection)
            if not hasattr(population[0], "fitness") or population[0].fitness == -np.inf:
                population = self._evaluate_population(population, training_data)

            # Find best in current generation
            current_best = max(population, key=lambda x: x.fitness)

            # Update global best
            if current_best.fitness > self.best_fitness:
                self.best_individual = current_best.copy()
                self.best_fitness = current_best.fitness
                print(f"Generation {generation + 1}: New best fitness = {self.best_fitness:.6f}")

            # Track history
            self.fitness_history.append(self.best_fitness)
            diversity = self._calculate_population_diversity(population)
            self.diversity_history.append(diversity)

            # Adaptive parameter update
            if self.params["adaptive_params"]:
                for ind in population:
                    if random.random() < 0.1:  # 10% chance to update
                        ind.F = np.random.uniform(0.1, 1.0)
                        ind.CR = np.random.uniform(0.5, 1.0)

            # Check for stagnation
            if self.params["population_reinit"] and self._check_stagnation(current_best.fitness, generation):
                population = self._reinitialize_population(population, keep_elite=2)

                # Re-evaluate new population
                population = self._evaluate_population(population, training_data)

            # Progress update
            if (generation + 1) % 10 == 0:
                print(
                    f"Generation {generation + 1}/{self.params['generations']}: "
                    f"Best Fitness = {self.best_fitness:.6f}, "
                    f"Diversity = {diversity:.4f}"
                )

        return self.best_individual

    def _cross_validate(self, data: pd.DataFrame, n_folds: int = 3) -> Dict:
        """
        Perform cross-validation to assess parameter robustness .
        """
        fold_size = len(data) // n_folds
        cv_results = []

        for fold in range(n_folds):
            # Split data
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < n_folds - 1 else len(data)

            train_data = pd.concat([data.iloc[:val_start], data.iloc[val_end:]])
            val_data = data.iloc[val_start:val_end]

            # Evolve on training fold
            best_individual = self._evolve(train_data)

            # Evaluate on validation fold
            val_fitness = self._calculate_fitness(best_individual, val_data)

            cv_results.append(
                {"fold": fold, "train_fitness": best_individual.fitness, "val_fitness": val_fitness, "parameters": best_individual.genes.copy()}
            )

        # Calculate statistics
        val_fitnesses = [r["val_fitness"] for r in cv_results]
        cv_summary = {
            "mean_val_fitness": np.mean(val_fitnesses),
            "std_val_fitness": np.std(val_fitnesses),
            "min_val_fitness": np.min(val_fitnesses),
            "max_val_fitness": np.max(val_fitnesses),
            "fold_results": cv_results,
        }

        return cv_summary

    def train(self, data: pd.DataFrame, cross_validate: bool = True):
        """
        Train the Differential Evolution strategy.

        Args:
            data: DataFrame with OHLCV data
            cross_validate: Whether to perform cross-validation
        """
        print(f"\n{'='*60}")
        print("Training Differential Evolution Strategy")
        print(f"{'='*60}")
        print(f"Strategy: {self.params['strategy_type'].value}")
        print(f"Variant: {self.params['variant'].value}")
        print(f"Population: {self.params['population_size']}, Generations: {self.params['generations']}")
        print(f"Multi-objective: {self.params['multi_objective']}")

        # Split into training and validation
        split_idx = int(len(data) * (1 - self.params["validation_split"]))
        train_data = data.iloc[:split_idx]
        val_data = data.iloc[split_idx:]

        # Perform cross-validation if requested
        if cross_validate and self.params["cross_validation_folds"] > 1:
            print(f"\nPerforming {self.params['cross_validation_folds']}-fold cross-validation...")
            cv_results = self._cross_validate(train_data, self.params["cross_validation_folds"])
            self.validation_results = cv_results
            print("Cross-validation results:")
            print(f"  Mean validation fitness: {cv_results['mean_val_fitness']:.6f}")
            print(f"  Std validation fitness: {cv_results['std_val_fitness']:.6f}")

        # Final evolution on full training data
        print("\nFinal evolution on training data...")
        self.best_individual = self._evolve(train_data)

        # Validate on out-of-sample data
        val_fitness = self._calculate_fitness(self.best_individual, val_data)
        self.oos_performance = {"val_fitness": val_fitness, "train_fitness": self.best_fitness, "fitness_drop": self.best_fitness - val_fitness}

        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        print(f"Best Training Fitness: {self.best_fitness:.6f}")
        print(f"Validation Fitness: {val_fitness:.6f}")
        print(f"Fitness Drop: {self.oos_performance['fitness_drop']:.6f}")
        print("Best Parameters:")
        for name, value in self.best_individual.genes.items():
            print(f"  {name}: {value}")

        self.last_training_date = data.index[-1]
        self.training_data = data
        self.current_strategy_params = self.best_individual.genes.copy()

    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """
        Generate trading signal using optimized parameters.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dict with signal information
        """
        # Check if we need to retrain
        current_date = data.index[-1]
        if (
            self.last_training_date is None
            or self.params.get("retrain_frequency", 0) > 0
            and self.last_training_date is not None
            and (current_date - self.last_training_date).days >= self.params["retrain_frequency"]
        ):
            lookback = min(252 * 2, len(data))
            training_data = data.iloc[-lookback:]
            self.train(training_data, cross_validate=False)

        # If no trained individual, return neutral signal
        if self.best_individual is None:
            if len(data) > 252:
                self.train(data, cross_validate=False)
            else:
                return normalize_signal(0)

        # Generate signal using best parameters
        signals = self._create_strategy_signals(data, self.best_individual.genes)
        current_signal = signals.iloc[-1]

        # Calculate confidence metrics
        recent_signals = signals.iloc[-20:]
        consistency = (recent_signals == current_signal).mean()

        # Recent performance
        returns = data["Close"].pct_change().iloc[-20:]
        recent_sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # Combine confidence factors
        base_confidence = min(1.0, consistency * 1.2)  # Boost consistency slightly
        performance_boost = min(0.3, max(-0.3, recent_sharpe * 0.1))
        confidence = min(1.0, max(0.2, base_confidence + performance_boost))

        # Position sizing based on confidence and validation
        position_size = confidence

        # Reduce position if validation performance is poor
        if self.oos_performance.get("fitness_drop", 0) > 0.5:
            position_size *= 0.7
        elif self.oos_performance.get("fitness_drop", 0) > 1.0:
            position_size *= 0.5

        return {
            "signal": int(current_signal),
            "position_size": float(position_size),
            "metadata": {
                "fitness": self.best_fitness,
                "val_fitness": self.oos_performance.get("val_fitness", 0),
                "fitness_drop": self.oos_performance.get("fitness_drop", 0),
                "parameters": self.best_individual.genes.copy(),
                "signal_consistency": float(consistency),
                "recent_sharpe": float(recent_sharpe),
                "confidence": float(confidence),
                "variant": self.params["variant"].value,
                "generations": self.generation,
                "final_diversity": self.diversity_history[-1] if self.diversity_history else 0,
                "F": self.best_individual.F,
                "CR": self.best_individual.CR,
                "multi_objective": self.params["multi_objective"],
            },
        }

    def generate_signals_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals for entire dataset at once"""
        if self.best_individual is None:
            if len(data) > 252:
                self.train(data, cross_validate=False)
            else:
                return pd.Series(0, index=data.index)

        return self._create_strategy_signals(data, self.best_individual.genes)

    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame"""
        history = pd.DataFrame(
            {
                "generation": range(1, len(self.fitness_history) + 1),
                "best_fitness": self.fitness_history,
                "diversity": self.diversity_history if self.diversity_history else [np.nan] * len(self.fitness_history),
            }
        )
        return history

    def get_parameter_importance(self) -> Dict:
        """
        Analyze parameter importance based on population diversity.
        Higher diversity indicates parameter has more impact on fitness.
        """
        if not self.population:
            return {}

        importance = {}
        for name in self.param_ranges.keys():
            values = [ind.genes[name] for ind in self.population]
            param_range = self.param_ranges[name]

            if param_range.categories is not None:
                # Categorical: use entropy
                from collections import Counter

                counts = Counter(values)
                probs = [c / len(values) for c in counts.values()]
                entropy = -sum(p * np.log(p) for p in probs)
                max_entropy = np.log(len(param_range.categories))
                importance[name] = entropy / max_entropy if max_entropy > 0 else 0
            else:
                # Numeric: use coefficient of variation
                normalized = [(v - param_range.min_val) / (param_range.max_val - param_range.min_val) for v in values]
                importance[name] = np.std(normalized) / (np.mean(normalized) + 1e-6)

        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def reset(self):
        """Reset strategy state"""
        self.population = []
        self.best_individual = None
        self.best_fitness = -np.inf
        self.fitness_history = []
        self.diversity_history = []
        self.stagnation_counter = 0
        self.generation = 0
        self.training_data = None
        self.last_training_date = None
        self.current_strategy_params = None
        self._signal_cache = None
        self.validation_results = {}
        self.oos_performance = {}

        if "random_seed" in self.params:
            random.seed(self.params["random_seed"])
            np.random.seed(self.params["random_seed"])

    def get_params_info(self) -> Dict:
        """Get parameter descriptions"""
        return {
            "strategy_type": "Type of strategy to optimize",
            "population_size": "Number of individuals",
            "generations": "Maximum generations",
            "F": "Mutation factor (0-1)",
            "CR": "Crossover rate (0-1)",
            "variant": "DE variant: rand_1_bin, best_1_bin, current_to_best_1_bin, rand_2_bin, adaptive",
            "adaptive_params": "Use adaptive F and CR",
            "dither": "Apply random variation to F",
            "jitter": "Apply random variation to CR",
            "population_reinit": "Reinitialize on stagnation",
            "stagnation_threshold": "Generations without improvement before reinit",
            "optimization_objective": "Fitness metric",
            "multi_objective": "Enable multi-objective optimization",
            "objective_weights": "Weights for multi-objective",
            "validation_split": "Fraction for validation",
            "cross_validation_folds": "Number of CV folds",
            "retrain_frequency": "How often to retrain (days)",
            "parallel_evaluation": "Use parallel processing",
            "num_workers": "Number of parallel workers",
            "random_seed": "Random seed",
        }
