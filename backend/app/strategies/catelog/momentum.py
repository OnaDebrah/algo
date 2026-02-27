from typing import Dict

from ...strategies import RSIStrategy
from ...strategies.catelog.category import StrategyCategory
from ...strategies.catelog.strategy_info import StrategyInfo
from ...strategies.cs_momentum_strategy import CrossSectionalMomentumStrategy
from ...strategies.ts_momentum_strategy import TimeSeriesMomentumStrategy


class Momentum:
    """Catalog of all available trading strategies"""

    def __init__(self):
        self.strategies = self._build_catalog()

    def _build_catalog(self) -> Dict[str, StrategyInfo]:
        """Build the strategy catalog"""

        catalog = {
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
                backtest_mode="both",
            ),
            "ts_momentum": StrategyInfo(
                name="Time Series Momentum",
                class_type=TimeSeriesMomentumStrategy,
                category=StrategyCategory.MOMENTUM,
                description="Exploits persistence in asset returns over time. Goes long when recent returns are positive, short when negative.",
                complexity="Intermediate",
                time_horizon="Short to Medium-term",
                best_for=[
                    "Trending assets",
                    "Futures trading",
                    "Systematic strategies",
                ],
                parameters={
                    "lookback": {
                        "default": 12,
                        "range": (1, 36),
                        "description": "Momentum lookback period (months)",
                    },
                    "holding_period": {
                        "default": 1,
                        "range": (1, 12),
                        "description": "Position holding period (months)",
                    },
                },
                pros=[
                    "Well-documented academic research",
                    "Works across asset classes",
                    "Simple to implement",
                    "Low turnover",
                ],
                cons=[
                    "Can reverse quickly",
                    "Drawdowns during trend reversals",
                    "Requires patience",
                    "Transaction costs matter",
                ],
                backtest_mode="multi",
            ),
            "cs_momentum": StrategyInfo(
                name="Cross-Sectional Momentum",
                class_type=CrossSectionalMomentumStrategy,
                category=StrategyCategory.MOMENTUM,
                description="Ranks assets by performance and goes long winners, short losers. Also known as relative strength.",
                complexity="Intermediate",
                time_horizon="Medium-term",
                best_for=[
                    "Stock portfolios",
                    "ETF rotation",
                    "Long-short strategies",
                ],
                parameters={
                    "lookback": {
                        "default": 6,
                        "range": (1, 12),
                        "description": "Performance lookback period (months)",
                    },
                    "top_pct": {
                        "default": 0.2,
                        "range": (0.1, 0.5),
                        "description": "Top percentile to buy",
                    },
                    "bottom_pct": {
                        "default": 0.2,
                        "range": (0.1, 0.5),
                        "description": "Bottom percentile to short",
                    },
                },
                pros=[
                    "Strong academic backing",
                    "Market-neutral variant available",
                    "Captures relative performance",
                    "Works in various markets",
                ],
                cons=[
                    "Requires multiple assets",
                    "Higher turnover than TS momentum",
                    "Crowded trade risk",
                    "Sector concentration risk",
                ],
                backtest_mode="multi",
            ),
        }
        return catalog
