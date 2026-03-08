from enum import Enum


class StrategyCategory(Enum):
    """Strategy categories"""

    TECHNICAL = "Technical Indicators"
    MOMENTUM = "Momentum"
    TREND_FOLLOWING = "Trend Following"
    MEAN_REVERSION = "Mean Reversion"
    MACHINE_LEARNING = "Machine Learning"
    VOLATILITY = "Volatility"
    STATISTICAL_ARBITRAGE = "Statistical Arbitrage"
    PAIRS_TRADING = "Pairs Trading"
    PRICE_ACTION = "Price Action"
    ADAPTIVE = "Adaptive Strategies"
    HYBRID = "Hybrid"
    DEEP_LEARNING = "Deep Learning"
    REINFORCEMENT_LEARNING = "Reinforcement Learning"
