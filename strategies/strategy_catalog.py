"""
Strategy Catalog and Categories
Organize all trading strategies by type
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Type

from strategies.base_strategy import BaseStrategy
from strategies.macd_strategy import MACDStrategy
from strategies.ml_strategy import MLStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.sma_crossover import SMACrossoverStrategy


class StrategyCategory(Enum):
    """Strategy categories"""

    TECHNICAL = "Technical Indicators"
    MOMENTUM = "Momentum"
    TREND_FOLLOWING = "Trend Following"
    MEAN_REVERSION = "Mean Reversion"
    MACHINE_LEARNING = "Machine Learning"
    VOLATILITY = "Volatility"
    PRICE_ACTION = "Price Action"
    HYBRID = "Hybrid"


@dataclass
class StrategyInfo:
    """Information about a strategy"""

    name: str
    class_type: Type[BaseStrategy]
    category: StrategyCategory
    description: str
    complexity: str  # Beginner, Intermediate, Advanced
    time_horizon: str  # Intraday, Short-term, Medium-term, Long-term
    best_for: List[str]
    parameters: Dict
    pros: List[str]
    cons: List[str]


class StrategyCatalog:
    """Catalog of all available trading strategies"""

    def __init__(self):
        self.strategies = self._build_catalog()

    def _build_catalog(self) -> Dict[str, StrategyInfo]:
        """Build the strategy catalog"""

        catalog = {
            # Technical Indicators - Trend Following
            "sma_crossover": StrategyInfo(
                name="SMA Crossover",
                class_type=SMACrossoverStrategy,
                category=StrategyCategory.TREND_FOLLOWING,
                description="Trades based on moving average crossovers. Buy when short MA crosses above long MA, sell when it crosses below.",
                complexity="Beginner",
                time_horizon="Short to Medium-term",
                best_for=["Trending markets", "Beginner traders", "Clear trends"],
                parameters={
                    "short_window": {
                        "default": 20,
                        "range": (5, 50),
                        "description": "Fast moving average period",
                    },
                    "long_window": {
                        "default": 50,
                        "range": (20, 200),
                        "description": "Slow moving average period",
                    },
                },
                pros=[
                    "Simple to understand",
                    "Works well in trending markets",
                    "Clear entry/exit signals",
                    "Low parameter sensitivity",
                ],
                cons=[
                    "Lags in fast-moving markets",
                    "Many false signals in ranging markets",
                    "Late entries and exits",
                ],
            ),
            # Technical Indicators - Momentum
            "rsi": StrategyInfo(
                name="RSI Strategy",
                class_type=RSIStrategy,
                category=StrategyCategory.MOMENTUM,
                description="Uses Relative Strength Index to identify overbought and oversold conditions. Buy at oversold, sell at overbought.",
                complexity="Beginner",
                time_horizon="Short-term",
                best_for=["Range-bound markets", "Momentum trading", "Quick trades"],
                parameters={
                    "period": {
                        "default": 14,
                        "range": (5, 30),
                        "description": "RSI calculation period",
                    },
                    "oversold": {
                        "default": 30,
                        "range": (10, 40),
                        "description": "Oversold threshold (buy signal)",
                    },
                    "overbought": {
                        "default": 70,
                        "range": (60, 90),
                        "description": "Overbought threshold (sell signal)",
                    },
                },
                pros=[
                    "Good for range-bound markets",
                    "Identifies momentum shifts",
                    "Clear overbought/oversold levels",
                    "Works on any timeframe",
                ],
                cons=[
                    "Can stay overbought/oversold for extended periods",
                    "Less effective in strong trends",
                    "False signals common",
                ],
            ),
            # Technical Indicators - Trend/Momentum
            "macd": StrategyInfo(
                name="MACD Strategy",
                class_type=MACDStrategy,
                category=StrategyCategory.TREND_FOLLOWING,
                description="Moving Average Convergence Divergence. Trades on crossovers of MACD line and signal line.",
                complexity="Intermediate",
                time_horizon="Medium-term",
                best_for=[
                    "Trend identification",
                    "Momentum confirmation",
                    "Swing trading",
                ],
                parameters={
                    "fast": {
                        "default": 12,
                        "range": (5, 20),
                        "description": "Fast EMA period",
                    },
                    "slow": {
                        "default": 26,
                        "range": (15, 40),
                        "description": "Slow EMA period",
                    },
                    "signal": {
                        "default": 9,
                        "range": (5, 15),
                        "description": "Signal line period",
                    },
                },
                pros=[
                    "Combines trend and momentum",
                    "Fewer false signals than simple MA",
                    "Works well for swing trading",
                    "Divergence signals available",
                ],
                cons=[
                    "Lags similar to moving averages",
                    "Complex for beginners",
                    "Can whipsaw in choppy markets",
                ],
            ),
            # Machine Learning
            "ml_random_forest": StrategyInfo(
                name="ML Random Forest",
                class_type=MLStrategy,
                category=StrategyCategory.MACHINE_LEARNING,
                description="Uses Random Forest classifier trained on technical indicators to predict market direction.",
                complexity="Advanced",
                time_horizon="Adaptable",
                best_for=[
                    "Complex pattern recognition",
                    "Multi-factor analysis",
                    "Data-rich environments",
                ],
                parameters={
                    "n_estimators": {
                        "default": 100,
                        "range": (50, 500),
                        "description": "Number of trees",
                    },
                    "max_depth": {
                        "default": 10,
                        "range": (5, 30),
                        "description": "Maximum tree depth",
                    },
                    "test_size": {
                        "default": 0.2,
                        "range": (0.1, 0.4),
                        "description": "Test set size",
                    },
                },
                pros=[
                    "Learns complex patterns",
                    "Adapts to market conditions",
                    "Multi-indicator integration",
                    "Non-linear relationships",
                ],
                cons=[
                    "Requires substantial training data",
                    "Black box (hard to interpret)",
                    "Risk of overfitting",
                    "Computationally expensive",
                ],
            ),
            "ml_gradient_boosting": StrategyInfo(
                name="ML Gradient Boosting",
                class_type=MLStrategy,
                category=StrategyCategory.MACHINE_LEARNING,
                description="Uses Gradient Boosting classifier for sequential learning and improved predictions.",
                complexity="Advanced",
                time_horizon="Adaptable",
                best_for=[
                    "Complex patterns",
                    "Incremental learning",
                    "High accuracy needs",
                ],
                parameters={
                    "n_estimators": {
                        "default": 100,
                        "range": (50, 500),
                        "description": "Number of boosting stages",
                    },
                    "learning_rate": {
                        "default": 0.1,
                        "range": (0.01, 0.3),
                        "description": "Learning rate",
                    },
                    "max_depth": {
                        "default": 5,
                        "range": (3, 15),
                        "description": "Tree depth",
                    },
                },
                pros=[
                    "Often more accurate than Random Forest",
                    "Handles complex patterns well",
                    "Sequential learning",
                    "Feature importance available",
                ],
                cons=[
                    "Even more prone to overfitting",
                    "Slower to train",
                    "Requires careful tuning",
                    "Computationally intensive",
                ],
            ),
        }

        return catalog

    def get_by_category(self, category: StrategyCategory) -> Dict[str, StrategyInfo]:
        """Get all strategies in a category"""
        return {key: info for key, info in self.strategies.items() if info.category == category}

    def get_by_complexity(self, complexity: str) -> Dict[str, StrategyInfo]:
        """Get strategies by complexity level"""
        return {key: info for key, info in self.strategies.items() if info.complexity == complexity}

    def get_categories(self) -> List[StrategyCategory]:
        """Get list of all categories"""
        return list(set(info.category for info in self.strategies.values()))

    def get_strategy_names(self) -> List[str]:
        """Get list of all strategy display names"""
        return [info.name for info in self.strategies.values()]

    def get_info(self, strategy_key: str) -> StrategyInfo:
        """Get information about a specific strategy"""
        return self.strategies.get(strategy_key)

    def create_strategy(self, strategy_key: str, **kwargs) -> BaseStrategy:
        """
        Create a strategy instance

        Args:
            strategy_key: Strategy identifier
            **kwargs: Strategy parameters

        Returns:
            Instantiated strategy
        """
        info = self.strategies.get(strategy_key)
        if not info:
            raise ValueError(f"Unknown strategy: {strategy_key}")

        # Use defaults for missing parameters
        params = {}
        for param_name, param_info in info.parameters.items():
            params[param_name] = kwargs.get(param_name, param_info["default"])

        return info.class_type(**params)

    def format_for_ui(self) -> Dict[str, List[Dict]]:
        """Format strategies for UI display, grouped by category"""

        result = {}

        for category in self.get_categories():
            strategies = self.get_by_category(category)

            result[category.value] = [
                {
                    "key": key,
                    "name": info.name,
                    "description": info.description,
                    "complexity": info.complexity,
                    "time_horizon": info.time_horizon,
                    "best_for": info.best_for,
                }
                for key, info in strategies.items()
            ]

        return result

    def get_comparison_matrix(self) -> Dict:
        """Get comparison matrix of all strategies"""

        matrix = []

        for key, info in self.strategies.items():
            matrix.append(
                {
                    "Strategy": info.name,
                    "Category": info.category.value,
                    "Complexity": info.complexity,
                    "Time Horizon": info.time_horizon,
                    "Best For": ", ".join(info.best_for[:2]),
                    "Pros": len(info.pros),
                    "Cons": len(info.cons),
                }
            )

        return matrix


# Global catalog instance
strategy_catalog = StrategyCatalog()


def get_catalog() -> StrategyCatalog:
    """Get the global strategy catalog"""
    return strategy_catalog


def get_strategies_by_category() -> Dict[str, List[str]]:
    """Get strategies organized by category for UI"""
    catalog = get_catalog()

    result = {}
    for category in catalog.get_categories():
        strategies = catalog.get_by_category(category)
        result[category.value] = [info.name for info in strategies.values()]

    return result


def get_strategy_description(strategy_name: str) -> str:
    """Get strategy description by display name"""
    catalog = get_catalog()

    for info in catalog.strategies.values():
        if info.name == strategy_name:
            return info.description

    return "No description available"


def get_recommended_strategies(level: str = "Beginner") -> List[str]:
    """Get recommended strategies for a skill level"""
    catalog = get_catalog()
    strategies = catalog.get_by_complexity(level)

    return [info.name for info in strategies.values()]
