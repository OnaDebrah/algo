from enum import Enum


class StrategyType(Enum):
    """Types of strategies that DE can optimize"""

    MOVING_AVERAGE_CROSSOVER = "ma_crossover"
    RSI_STRATEGY = "rsi_strategy"
    MACD_STRATEGY = "macd_strategy"
    BOLLINGER_BANDS = "bollinger_bands"
    COMBINED_SIGNAL = "combined_signal"
    MACHINE_LEARNING = "ml_strategy"  # For ML-based strategies
    ENSEMBLE = "ensemble"  # Weighted ensemble of multiple strategies
