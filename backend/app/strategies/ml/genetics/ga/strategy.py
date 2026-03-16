from enum import Enum


class StrategyType(Enum):
    """Types of strategies that GA can optimize"""

    MOVING_AVERAGE_CROSSOVER = "ma_crossover"
    RSI_STRATEGY = "rsi_strategy"
    MACD_STRATEGY = "macd_strategy"
    BOLLINGER_BANDS = "bollinger_bands"
    COMBINED_SIGNAL = "combined_signal"  # Weighted combination of indicators
