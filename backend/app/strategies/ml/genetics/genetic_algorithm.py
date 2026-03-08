"""
Genetic Algorithm Trading Strategy

Optimizes strategy parameters using evolutionary computation.
Based on research showing GA effectively optimizes trading parameters
and indicator combinations [citation:1][citation:7].

Key features:
- Optimizes multiple strategy types (moving averages, RSI, MACD, Bollinger Bands)
- Tournament selection with elitism
- Adaptive mutation rates
- Parallel fitness evaluation
- Population diversity tracking
- Support for both single and multi-objective optimization
"""

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from base_strategy import BaseStrategy, normalize_signal


class StrategyType(Enum):
    """Types of strategies that GA can optimize"""

    MOVING_AVERAGE_CROSSOVER = "ma_crossover"
    RSI_STRATEGY = "rsi_strategy"
    MACD_STRATEGY = "macd_strategy"
    BOLLINGER_BANDS = "bollinger_bands"
    COMBINED_SIGNAL = "combined_signal"  # Weighted combination of indicators


class OptimizationObjective(Enum):
    """Optimization objectives for fitness evaluation [citation:7]"""

    SHARPE_RATIO = "sharpe"
    TOTAL_RETURN = "total_return"
    SORTINO_RATIO = "sortino"
    CALMAR_RATIO = "calmar"
    MAX_DRAWDOWN = "max_drawdown"  # Minimize
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    COMBINED = "combined"  # Weighted combination


@dataclass
class ParameterRange:
    """Defines the range for a tunable parameter"""

    min_val: float
    max_val: float
    param_type: type = float  # int, float, or 'categorical'
    categories: List = None  # For categorical parameters
    is_int: bool = False

    def __post_init__(self):
        if self.param_type is int or self.is_int:
            self.param_type = int
            self.is_int = True


class Chromosome:
    """
    Chromosome representing a trading strategy's parameters.
    Supports binary, real, and integer encoding [citation:3].
    """

    def __init__(self, genes: Dict[str, float], param_ranges: Dict[str, ParameterRange]):
        """
        Initialize chromosome with genes.

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

        # Validate genes against ranges
        self._validate_genes()

    def _validate_genes(self):
        """Ensure genes are within specified ranges"""
        for name, value in self.genes.items():
            if name in self.param_ranges:
                param_range = self.param_ranges[name]

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

    def mutate(self, mutation_rate: float, mutation_strength: float = 0.1):
        """
        Mutate genes with given probability.

        Args:
            mutation_rate: Probability of mutating each gene
            mutation_strength: Standard deviation for Gaussian mutation (as fraction of range)
        """
        for name in self.genes:
            if random.random() < mutation_rate:
                param_range = self.param_ranges.get(name)
                if param_range is None:
                    continue

                # Categorical mutation
                if param_range.categories is not None:
                    current_idx = param_range.categories.index(self.genes[name])
                    new_idx = (current_idx + random.choice([-1, 1])) % len(param_range.categories)
                    self.genes[name] = param_range.categories[new_idx]

                # Numeric mutation
                else:
                    # Gaussian mutation
                    range_size = param_range.max_val - param_range.min_val
                    sigma = range_size * mutation_strength
                    mutation = np.random.normal(0, sigma)

                    new_value = self.genes[name] + mutation

                    # Clip to range
                    new_value = max(param_range.min_val, min(param_range.max_val, new_value))

                    # Convert to int if needed
                    if param_range.is_int:
                        new_value = int(round(new_value))

                    self.genes[name] = new_value

        self.age += 1

    def copy(self) -> "Chromosome":
        """Create a deep copy"""
        new_chromosome = Chromosome(self.genes.copy(), self.param_ranges)
        new_chromosome.fitness = self.fitness
        new_chromosome.fitness_components = self.fitness_components.copy()
        new_chromosome.age = self.age
        new_chromosome.metadata = self.metadata.copy()
        return new_chromosome

    def __repr__(self) -> str:
        return f"Chromosome(fitness={self.fitness:.4f}, genes={self.genes})"


class GeneticAlgorithmStrategy(BaseStrategy):
    """
    Trading strategy that uses Genetic Algorithm to optimize parameters.

    Based on research demonstrating GA's effectiveness for:
    - Parameter optimization [citation:1][citation:7]
    - Trading pair selection [citation:7]
    - Multi-objective optimization [citation:4]
    """

    def __init__(self, name: str = "GeneticAlgorithm", params: Dict = None):
        """
        Initialize Genetic Algorithm Strategy.

        Args:
            name: Strategy name
            params: Dictionary with parameters:
                - strategy_type: Type of strategy to optimize (StrategyType enum or string)
                - population_size: Number of individuals (default: 50)
                - generations: Number of generations to evolve (default: 30)
                - tournament_size: Tournament selection size (default: 3)
                - elitism_count: Number of elites to preserve (default: 2)
                - crossover_rate: Crossover probability (default: 0.8)
                - mutation_rate: Mutation probability (default: 0.1)
                - mutation_strength: Strength of mutations (default: 0.1)
                - optimization_objective: Fitness metric (default: 'sharpe')
                - retrain_frequency: How often to retrain (days, default: 90)
                - validation_split: Fraction for validation (default: 0.3)
                - parallel_evaluation: Use parallel processing (default: True)
                - num_workers: Number of parallel workers (default: 4)
                - adaptive_mutation: Adapt mutation rate based on diversity (default: True)
                - diversity_threshold: Min diversity before increasing mutation (default: 0.1)
                - early_stopping: Stop if no improvement for N gens (default: 10)
                - multi_objective: Enable multi-objective optimization (default: False)
                - objective_weights: Weights for multi-objective (if multi_objective=True)
                - random_seed: Random seed for reproducibility
        """
        default_params = {
            "strategy_type": StrategyType.MOVING_AVERAGE_CROSSOVER,
            "population_size": 50,
            "generations": 30,
            "tournament_size": 3,
            "elitism_count": 2,
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "mutation_strength": 0.1,
            "optimization_objective": OptimizationObjective.SHARPE_RATIO,
            "retrain_frequency": 90,
            "validation_split": 0.3,
            "parallel_evaluation": True,
            "num_workers": 4,
            "adaptive_mutation": True,
            "diversity_threshold": 0.1,
            "early_stopping": 10,
            "multi_objective": False,
            "objective_weights": {
                "sharpe": 0.4,
                "total_return": 0.3,
                "max_drawdown": 0.3,  # Negative weight for minimization
            },
            "random_seed": 42,
        }

        if params:
            default_params.update(params)

        super().__init__(name, default_params)

        # Set random seed
        if "random_seed" in self.params:
            random.seed(self.params["random_seed"])
            np.random.seed(self.params["random_seed"])

        # Convert string strategy_type to enum if needed
        if isinstance(self.params["strategy_type"], str):
            try:
                self.params["strategy_type"] = StrategyType(self.params["strategy_type"])
            except ValueError:
                # Try to match by name
                for strategy in StrategyType:
                    if strategy.name.lower() == self.params["strategy_type"].lower():
                        self.params["strategy_type"] = strategy
                        break

        # Convert string optimization_objective to enum if needed
        if isinstance(self.params["optimization_objective"], str):
            try:
                self.params["optimization_objective"] = OptimizationObjective(self.params["optimization_objective"])
            except ValueError:
                # Try to match by name
                for obj in OptimizationObjective:
                    if obj.name.lower() == self.params["optimization_objective"].lower():
                        self.params["optimization_objective"] = obj
                        break

        # Initialize parameter ranges based on strategy type
        self.param_ranges = self._initialize_parameter_ranges()

        # Population
        self.population = []
        self.best_chromosome = None
        self.best_fitness = -np.inf
        self.fitness_history = []
        self.diversity_history = []

        # Training data
        self.training_data = None
        self.last_training_date = None

        # Strategy state
        self.current_strategy_params = None
        self._signal_cache = None

    def _initialize_parameter_ranges(self) -> Dict[str, ParameterRange]:
        """
        Initialize parameter ranges based on strategy type [citation:9].
        """
        strategy_type = self.params["strategy_type"]
        ranges = {}

        if strategy_type == StrategyType.MOVING_AVERAGE_CROSSOVER:
            ranges = {
                "short_window": ParameterRange(5, 50, int, is_int=True),
                "long_window": ParameterRange(20, 200, int, is_int=True),
                "signal_threshold": ParameterRange(0.0, 1.0, float),  # For confirmation
                "use_volume_filter": ParameterRange(0, 1, int, categories=[0, 1]),
                "volume_threshold": ParameterRange(1.0, 3.0, float),
            }

        elif strategy_type == StrategyType.RSI_STRATEGY:
            ranges = {
                "rsi_period": ParameterRange(5, 30, int, is_int=True),
                "oversold_threshold": ParameterRange(20, 40, int, is_int=True),
                "overbought_threshold": ParameterRange(60, 80, int, is_int=True),
                "exit_oversold": ParameterRange(40, 60, int, is_int=True),
                "exit_overbought": ParameterRange(40, 60, int, is_int=True),
                "use_trend_filter": ParameterRange(0, 1, int, categories=[0, 1]),
                "trend_period": ParameterRange(20, 100, int, is_int=True),
            }

        elif strategy_type == StrategyType.MACD_STRATEGY:
            ranges = {
                "fast_period": ParameterRange(5, 20, int, is_int=True),
                "slow_period": ParameterRange(20, 40, int, is_int=True),
                "signal_period": ParameterRange(5, 15, int, is_int=True),
                "signal_threshold": ParameterRange(0.0, 1.0, float),  # Min MACD value
                "use_histogram": ParameterRange(0, 1, int, categories=[0, 1]),
                "histogram_threshold": ParameterRange(0.0, 0.5, float),
            }

        elif strategy_type == StrategyType.BOLLINGER_BANDS:
            ranges = {
                "period": ParameterRange(10, 50, int, is_int=True),
                "num_std": ParameterRange(1.5, 3.0, float),
                "entry_threshold": ParameterRange(0.0, 1.0, float),  # % distance from band
                "exit_threshold": ParameterRange(0.0, 1.0, float),
                "use_squeeze": ParameterRange(0, 1, int, categories=[0, 1]),
                "squeeze_period": ParameterRange(10, 30, int, is_int=True),
            }

        elif strategy_type == StrategyType.COMBINED_SIGNAL:
            ranges = {
                "ma_weight": ParameterRange(0.0, 1.0, float),
                "rsi_weight": ParameterRange(0.0, 1.0, float),
                "macd_weight": ParameterRange(0.0, 1.0, float),
                "bb_weight": ParameterRange(0.0, 1.0, float),
                "ma_short": ParameterRange(5, 50, int, is_int=True),
                "ma_long": ParameterRange(20, 200, int, is_int=True),
                "rsi_period": ParameterRange(5, 30, int, is_int=True),
                "rsi_oversold": ParameterRange(20, 40, int, is_int=True),
                "rsi_overbought": ParameterRange(60, 80, int, is_int=True),
                "macd_fast": ParameterRange(5, 20, int, is_int=True),
                "macd_slow": ParameterRange(20, 40, int, is_int=True),
                "macd_signal": ParameterRange(5, 15, int, is_int=True),
                "bb_period": ParameterRange(10, 50, int, is_int=True),
                "bb_std": ParameterRange(1.5, 3.0, float),
                "signal_threshold": ParameterRange(0.0, 2.0, float),  # Min combined score
            }

        return ranges

    def _create_strategy_signals(self, data: pd.DataFrame, params: Dict[str, float]) -> pd.Series:
        """
        Generate trading signals based on strategy type and parameters.

        Returns:
            Series with values: 1 (buy), -1 (sell), 0 (hold)
        """
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
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

    def _ma_crossover_signals(self, data: pd.DataFrame, params: Dict) -> pd.Series:
        """Moving average crossover strategy [citation:9]"""
        short_window = int(params["short_window"])
        long_window = int(params["long_window"])
        signal_threshold = params.get("signal_threshold", 0.0)

        # Calculate moving averages
        short_ma = data["Close"].rolling(window=short_window).mean()
        long_ma = data["Close"].rolling(window=long_window).mean()

        # Generate signals
        signals = pd.Series(0, index=data.index)

        # Crossover signal
        ma_diff = short_ma - long_ma
        signals[ma_diff > signal_threshold] = 1
        signals[ma_diff < -signal_threshold] = -1

        # Volume filter if enabled
        if params.get("use_volume_filter", 0) == 1:
            volume_ratio = data["Volume"] / data["Volume"].rolling(window=20).mean()
            volume_threshold = params.get("volume_threshold", 1.5)

            # Only trade on high volume
            signals[volume_ratio < volume_threshold] = 0

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
        signals[rsi < oversold] = 1  # Oversold - buy
        signals[rsi > overbought] = -1  # Overbought - sell

        # Exit signals
        if exit_oversold > oversold:
            # Exit long positions when RSI rises above exit_oversold
            signals[(signals.shift(1) == 1) & (rsi > exit_oversold)] = 0

        if exit_overbought < overbought:
            # Exit short positions when RSI falls below exit_overbought
            signals[(signals.shift(1) == -1) & (rsi < exit_overbought)] = 0

        # Trend filter if enabled
        if params.get("use_trend_filter", 0) == 1:
            trend_period = int(params.get("trend_period", 50))
            sma = data["Close"].rolling(window=trend_period).mean()

            # Only buy in uptrend, only sell in downtrend
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

        # Use MACD line vs signal line
        macd_diff = macd_line - signal_line

        if params.get("use_histogram", 0) == 1:
            # Use histogram for signals
            hist_threshold = params.get("histogram_threshold", 0.0)
            signals[histogram > hist_threshold] = 1
            signals[histogram < -hist_threshold] = -1
        else:
            # Use line crossover
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

        # Entry signals - price near bands
        signals[band_position < entry_threshold] = 1  # Near lower band - buy
        signals[band_position > (1 - entry_threshold)] = -1  # Near upper band - sell

        # Exit signals - price back to middle
        middle_zone = (band_position > exit_threshold) & (band_position < (1 - exit_threshold))
        signals[middle_zone & (signals.shift(1) != 0)] = 0

        # Squeeze filter if enabled
        if params.get("use_squeeze", 0) == 1:
            squeeze_period = int(params.get("squeeze_period", 20))
            band_width = (upper - lower) / sma
            avg_width = band_width.rolling(window=squeeze_period).mean()

            # Only trade when bands are expanding (volatility increasing)
            signals[band_width < avg_width] = 0

        return signals

    def _combined_signals(self, data: pd.DataFrame, params: Dict) -> pd.Series:
        """Weighted combination of multiple indicators [citation:1]"""
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

        # Convert to final signals
        signals = pd.Series(0, index=data.index)
        signals[score > signal_threshold] = 1
        signals[score < -signal_threshold] = -1

        return signals

    def _calculate_fitness(self, chromosome: Chromosome, data: pd.DataFrame) -> float:
        """
        Calculate fitness for a chromosome [citation:1][citation:7].

        Returns:
            Fitness value (higher is better)
        """
        # Generate signals using chromosome's parameters
        signals = self._create_strategy_signals(data, chromosome.genes)

        # Calculate returns
        returns = data["Close"].pct_change().fillna(0)
        strategy_returns = returns * signals.shift(1).fillna(0)

        # Apply transaction costs (0.1% per trade)
        trades = (signals.diff().fillna(0) != 0).astype(int)
        transaction_costs = trades * 0.001
        net_returns = strategy_returns - transaction_costs.shift(1).fillna(0)

        # Skip first few periods where indicators aren't available
        min_period = max([int(v) for v in chromosome.genes.values() if isinstance(v, (int, float)) and v < 200])
        net_returns.iloc[:min_period] = 0

        # Calculate fitness based on optimization objective
        if self.params["multi_objective"]:
            # Multi-objective optimization
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

            # Store components
            chromosome.fitness_components = fitness_dict

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

            if objective == OptimizationObjective.SHARPE_RATIO:
                if net_returns.std() > 0:
                    return np.sqrt(252) * net_returns.mean() / net_returns.std()
                return 0

            elif objective == OptimizationObjective.TOTAL_RETURN:
                return (1 + net_returns).prod() - 1

            elif objective == OptimizationObjective.SORTINO_RATIO:
                downside = net_returns[net_returns < 0].std()
                if downside > 0:
                    return np.sqrt(252) * net_returns.mean() / downside
                return np.sqrt(252) * net_returns.mean() * 10

            elif objective == OptimizationObjective.CALMAR_RATIO:
                cumulative = (1 + net_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = abs(drawdown.min())
                total_ret = (1 + net_returns).prod() - 1
                if max_drawdown > 0:
                    return total_ret / max_drawdown
                return total_ret * 10

            elif objective == OptimizationObjective.MAX_DRAWDOWN:
                cumulative = (1 + net_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                return -abs(drawdown.min())  # Negative because we minimize

            elif objective == OptimizationObjective.WIN_RATE:
                winning_trades = net_returns[net_returns > 0].count()
                total_trades = net_returns[net_returns != 0].count()
                return winning_trades / total_trades if total_trades > 0 else 0

            elif objective == OptimizationObjective.PROFIT_FACTOR:
                gross_profit = net_returns[net_returns > 0].sum()
                gross_loss = abs(net_returns[net_returns < 0].sum())
                return gross_profit / gross_loss if gross_loss > 0 else 10

            else:  # Combined
                # Simple weighted combination of multiple metrics
                sharpe = np.sqrt(252) * net_returns.mean() / net_returns.std() if net_returns.std() > 0 else 0
                total_ret = (1 + net_returns).prod() - 1
                return 0.6 * sharpe + 0.4 * total_ret

    def _initialize_population(self) -> List[Chromosome]:
        """Initialize random population [citation:9]"""
        population = []

        for _ in range(self.params["population_size"]):
            genes = {}

            for name, param_range in self.param_ranges.items():
                if param_range.categories is not None:
                    # Categorical parameter
                    genes[name] = random.choice(param_range.categories)
                else:
                    # Numeric parameter
                    value = random.uniform(param_range.min_val, param_range.max_val)
                    if param_range.is_int:
                        value = int(round(value))
                    genes[name] = value

            chromosome = Chromosome(genes, self.param_ranges)
            population.append(chromosome)

        return population

    def _evaluate_population(self, population: List[Chromosome], data: pd.DataFrame) -> List[Chromosome]:
        """
        Evaluate fitness for entire population.
        Supports parallel evaluation [citation:7].
        """
        if self.params["parallel_evaluation"] and len(population) > 10:
            # Use parallel processing
            with ThreadPoolExecutor(max_workers=self.params["num_workers"]) as executor:
                # Submit all evaluation tasks
                futures = []
                for chromosome in population:
                    future = executor.submit(self._calculate_fitness, chromosome, data)
                    futures.append((chromosome, future))

                # Collect results
                for chromosome, future in futures:
                    chromosome.fitness = future.result()
        else:
            # Sequential evaluation
            for chromosome in population:
                chromosome.fitness = self._calculate_fitness(chromosome, data)

        return population

    def _tournament_selection(self, population: List[Chromosome]) -> Chromosome:
        """Tournament selection [citation:3][citation:7]"""
        tournament = random.sample(population, self.params["tournament_size"])
        return max(tournament, key=lambda c: c.fitness)

    def _crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Single-point crossover [citation:9].
        """
        if random.random() > self.params["crossover_rate"]:
            return parent1.copy(), parent2.copy()

        # Get gene names
        gene_names = list(parent1.genes.keys())

        if len(gene_names) <= 1:
            return parent1.copy(), parent2.copy()

        # Select crossover point
        point = random.randint(1, len(gene_names) - 1)

        # Create children genes
        child1_genes = {}
        child2_genes = {}

        for i, name in enumerate(gene_names):
            if i < point:
                child1_genes[name] = parent1.genes[name]
                child2_genes[name] = parent2.genes[name]
            else:
                child1_genes[name] = parent2.genes[name]
                child2_genes[name] = parent1.genes[name]

        # Create children
        child1 = Chromosome(child1_genes, self.param_ranges)
        child2 = Chromosome(child2_genes, self.param_ranges)

        return child1, child2

    def _calculate_population_diversity(self, population: List[Chromosome]) -> float:
        """Calculate population diversity (standard deviation of genes)"""
        if len(population) < 2:
            return 1.0

        # Collect all genes
        gene_names = list(population[0].genes.keys())

        # Calculate average normalized standard deviation
        total_diversity = 0
        count = 0

        for name in gene_names:
            param_range = self.param_ranges.get(name)
            if param_range is None or param_range.categories is not None:
                continue

            values = [c.genes[name] for c in population]
            range_size = param_range.max_val - param_range.min_val

            if range_size > 0:
                # Normalize values to [0,1]
                normalized = [(v - param_range.min_val) / range_size for v in values]
                diversity = np.std(normalized)
                total_diversity += diversity
                count += 1

        return total_diversity / max(count, 1)

    def _evolve(self, training_data: pd.DataFrame, validation_data: pd.DataFrame) -> Chromosome:
        """
        Evolve population using genetic algorithm [citation:1][citation:7].

        Args:
            training_data: Training data
            validation_data: Validation data

        Returns:
            Best chromosome
        """
        # Initialize population
        population = self._initialize_population()

        best_fitness = -np.inf
        best_chromosome = None
        no_improvement_count = 0

        # Evolution loop
        for generation in range(self.params["generations"]):
            # Evaluate population on training data
            population = self._evaluate_population(population, training_data)

            # Sort by fitness
            population.sort(key=lambda c: c.fitness, reverse=True)

            # Track best
            if population[0].fitness > best_fitness:
                best_fitness = population[0].fitness
                best_chromosome = population[0].copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            self.fitness_history.append(best_fitness)

            # Calculate diversity
            diversity = self._calculate_population_diversity(population)
            self.diversity_history.append(diversity)

            # Adaptive mutation [citation:7]
            mutation_rate = self.params["mutation_rate"]
            if self.params["adaptive_mutation"]:
                if diversity < self.params["diversity_threshold"]:
                    mutation_rate = min(0.5, mutation_rate * 2)
                else:
                    mutation_rate = self.params["mutation_rate"]

            # Early stopping
            if no_improvement_count >= self.params["early_stopping"]:
                print(f"Early stopping at generation {generation}")
                break

            # Create new population
            new_population = []

            # Elitism
            elites = population[: self.params["elitism_count"]]
            new_population.extend([e.copy() for e in elites])

            # Generate rest through selection, crossover, mutation
            while len(new_population) < self.params["population_size"]:
                # Select parents
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)

                # Crossover
                child1, child2 = self._crossover(parent1, parent2)

                # Mutate
                child1.mutate(mutation_rate, self.params["mutation_strength"])
                child2.mutate(mutation_rate, self.params["mutation_strength"])

                new_population.append(child1)
                if len(new_population) < self.params["population_size"]:
                    new_population.append(child2)

            population = new_population

            # Progress update
            if (generation + 1) % 10 == 0:
                print(f"Generation {generation + 1}: Best Fitness = {best_fitness:.4f}, Diversity = {diversity:.4f}")

        # Validate best on validation data
        if best_chromosome:
            val_fitness = self._calculate_fitness(best_chromosome, validation_data)
            best_chromosome.metadata["validation_fitness"] = val_fitness
            best_chromosome.metadata["generations"] = generation + 1
            best_chromosome.metadata["final_diversity"] = diversity

            print(f"\nValidation Fitness: {val_fitness:.4f}")
            print(f"Best Parameters: {best_chromosome.genes}")

        return best_chromosome

    def train(self, data: pd.DataFrame):
        """
        Train the genetic algorithm strategy.

        Args:
            data: DataFrame with OHLCV data
        """
        # Split into training and validation
        split_idx = int(len(data) * (1 - self.params["validation_split"]))
        train_data = data.iloc[:split_idx]
        val_data = data.iloc[split_idx:]

        print(f"Training GA with {self.params['population_size']} individuals for {self.params['generations']} generations...")
        print(f"Strategy: {self.params['strategy_type'].value}")
        print(f"Objective: {self.params['optimization_objective'].value if not self.params['multi_objective'] else 'multi-objective'}")

        # Evolve
        self.best_chromosome = self._evolve(train_data, val_data)
        self.best_fitness = self.best_chromosome.fitness

        self.last_training_date = data.index[-1]
        self.training_data = data
        self.current_strategy_params = self.best_chromosome.genes.copy()

        print("\nTraining complete!")
        print(f"Best fitness: {self.best_fitness:.4f}")
        print(f"Validation fitness: {self.best_chromosome.metadata.get('validation_fitness', 0):.4f}")

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
            self.train(training_data)

        # If no trained chromosome, return neutral signal
        if self.best_chromosome is None:
            if len(data) > 252:
                self.train(data)
            else:
                return normalize_signal(0)

        # Generate signal using best parameters
        signals = self._create_strategy_signals(data, self.best_chromosome.genes)
        current_signal = signals.iloc[-1]

        # Calculate confidence based on signal strength and recent performance
        recent_signals = signals.iloc[-20:]
        consistency = (recent_signals == current_signal).mean()

        # Get recent returns for confidence adjustment
        returns = data["Close"].pct_change().iloc[-20:]
        recent_performance = returns.mean() / returns.std() if returns.std() > 0 else 0

        # Combined confidence
        confidence = min(1.0, max(0.0, consistency * (0.7 + 0.3 * min(1, max(-1, recent_performance)))))

        # Position size based on confidence
        position_size = min(1.0, max(0.2, confidence))

        # Check if validation fitness is positive (strategy adds value)
        val_fitness = self.best_chromosome.metadata.get("validation_fitness", 0)
        if val_fitness < 0:
            position_size *= 0.5  # Reduce position if strategy underperforms

        return {
            "signal": int(current_signal),
            "position_size": float(position_size),
            "metadata": {
                "fitness": self.best_fitness,
                "validation_fitness": val_fitness,
                "parameters": self.best_chromosome.genes.copy(),
                "signal_consistency": float(consistency),
                "recent_performance": float(recent_performance),
                "confidence": float(confidence),
                "generations": self.best_chromosome.metadata.get("generations", 0),
                "final_diversity": self.best_chromosome.metadata.get("final_diversity", 0),
                "strategy_type": self.params["strategy_type"].value,
                "objective": self.params["optimization_objective"].value if not self.params["multi_objective"] else "multi",
            },
        }

    def generate_signals_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate signals for entire dataset at once.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Series of signals indexed by timestamp
        """
        if self.best_chromosome is None:
            if len(data) > 252:
                self.train(data)
            else:
                return pd.Series(0, index=data.index)

        return self._create_strategy_signals(data, self.best_chromosome.genes)

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

    def reset(self):
        """Reset strategy state"""
        self.population = []
        self.best_chromosome = None
        self.best_fitness = -np.inf
        self.fitness_history = []
        self.diversity_history = []
        self.training_data = None
        self.last_training_date = None
        self.current_strategy_params = None
        self._signal_cache = None

        if "random_seed" in self.params:
            random.seed(self.params["random_seed"])
            np.random.seed(self.params["random_seed"])

    def get_params_info(self) -> Dict:
        """Get parameter descriptions"""
        return {
            "strategy_type": "Type of strategy to optimize: ma_crossover, rsi_strategy, macd_strategy, bollinger_bands, combined_signal",
            "population_size": "Number of individuals in population",
            "generations": "Number of generations to evolve",
            "tournament_size": "Tournament selection size",
            "elitism_count": "Number of elites to preserve",
            "crossover_rate": "Crossover probability (0.0-1.0)",
            "mutation_rate": "Mutation probability (0.0-1.0)",
            "mutation_strength": "Strength of mutations (as fraction of range)",
            "optimization_objective": "Fitness metric: sharpe, total_return, sortino, calmar, max_drawdown, win_rate, profit_factor, combined",
            "retrain_frequency": "How often to retrain (days)",
            "validation_split": "Fraction of data for validation (0.0-1.0)",
            "parallel_evaluation": "Use parallel processing for fitness evaluation",
            "num_workers": "Number of parallel workers",
            "adaptive_mutation": "Adapt mutation rate based on diversity",
            "diversity_threshold": "Min diversity before increasing mutation",
            "early_stopping": "Stop if no improvement for N generations",
            "multi_objective": "Enable multi-objective optimization",
            "objective_weights": "Weights for multi-objective optimization",
            "random_seed": "Random seed for reproducibility",
        }
