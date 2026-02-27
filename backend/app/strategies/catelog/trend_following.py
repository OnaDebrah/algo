from typing import Dict

from ...strategies.catelog.strategy_info import StrategyInfo
from ...strategies.technical.sma_crossover import SMACrossoverStrategy
from .. import KAMAStrategy, MACDStrategy, MultiTimeframeKAMAStrategy
from ..adpative_trend_ff_strategy import AdaptiveTrendFollowingStrategy
from ..donchain_strategy import DonchianATRStrategy, DonchianChannelStrategy, FilteredDonchianStrategy
from ..parabolic_sar import ParabolicSARStrategy
from .category import StrategyCategory


class TrendFollowing:
    """Catalog of all available trading strategies"""

    def __init__(self):
        self.strategies = self._build_catalog()

    def _build_catalog(self) -> Dict[str, StrategyInfo]:
        """Build the strategy catalog"""

        catalog = {
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
                backtest_mode="both",
            ),
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
                backtest_mode="both",
            ),
            "adaptive_trend": StrategyInfo(
                name="Adaptive Trend Following",
                class_type=AdaptiveTrendFollowingStrategy,
                category=StrategyCategory.ADAPTIVE,
                description="Dynamically adjusts trend-following parameters based on market conditions and volatility.",
                complexity="Advanced",
                time_horizon="Medium to Long-term",
                best_for=[
                    "Changing market conditions",
                    "Volatile markets",
                    "Institutional trading",
                ],
                parameters={
                    "lookback_period": {
                        "default": 50,
                        "range": (20, 100),
                        "description": "Lookback period for adaptation",
                    },
                    "volatility_threshold": {
                        "default": 0.02,
                        "range": (0.01, 0.05),
                        "description": "Volatility threshold for adjustment",
                    },
                },
                pros=[
                    "Adapts to market conditions",
                    "Reduces false signals in choppy markets",
                    "Better risk-adjusted returns",
                    "Handles regime changes well",
                ],
                cons=[
                    "Complex to implement",
                    "Requires significant data history",
                    "Parameter optimization needed",
                    "Computationally intensive",
                ],
                backtest_mode="both",
            ),
            "kama": StrategyInfo(
                name="KAMA Strategy",
                class_type=KAMAStrategy,
                category=StrategyCategory.TREND_FOLLOWING,
                description="Kaufman's Adaptive Moving Average. Adapts speed based on market efficiency - fast during trends, slow during consolidation. Reduces whipsaws compared to traditional moving averages.",
                complexity="Intermediate",
                time_horizon="Medium-term",
                best_for=["Trending markets", "Swing trading", "Reduced whipsaws", "Adaptive trend following"],
                parameters={
                    "period": {"default": 10, "range": (5, 30), "description": "Efficiency ratio calculation period"},
                    "fast_ema": {"default": 2, "range": (2, 5), "description": "Fast EMA constant (for trending markets)"},
                    "slow_ema": {"default": 30, "range": (20, 50), "description": "Slow EMA constant (for choppy markets)"},
                    "signal_threshold": {"default": 0.0, "range": (0.0, 0.02), "description": "Minimum % above/below KAMA for signal"},
                },
                pros=[
                    "Adapts to market conditions automatically",
                    "Fewer false signals than simple MAs",
                    "Fast response during strong trends",
                    "Stable during consolidation",
                    "Works across multiple timeframes",
                ],
                cons=["More complex than simple MAs", "Still has some lag", "Parameter sensitive", "Can whipsaw during trend transitions"],
                backtest_mode="both",
            ),
            "multi_kama": StrategyInfo(
                name="MULTI KAMA Strategy",
                class_type=MultiTimeframeKAMAStrategy,
                category=StrategyCategory.TREND_FOLLOWING,
                description="Kaufman's Adaptive Moving Average. Adapts speed based on market efficiency - fast during trends, slow during consolidation. Reduces whipsaws compared to traditional moving averages.",
                complexity="Intermediate",
                time_horizon="Medium-term",
                best_for=["Trending markets", "Swing trading", "Reduced whipsaws", "Adaptive trend following"],
                parameters={
                    "short_period": {"default": 10, "range": (5, 30), "description": "Efficiency ratio calculation period"},
                    "long_period": {"default": 15, "range": (5, 45), "description": "Efficiency ratio calculation period"},
                    "fast_ema": {"default": 2, "range": (2, 5), "description": "Fast EMA constant (for trending markets)"},
                    "slow_ema": {"default": 30, "range": (20, 50), "description": "Slow EMA constant (for choppy markets)"},
                },
                pros=[
                    "Adapts to market conditions automatically",
                    "Fewer false signals than simple MAs",
                    "Fast response during strong trends",
                    "Stable during consolidation",
                    "Works across multiple timeframes",
                ],
                cons=["More complex than simple MAs", "Still has some lag", "Parameter sensitive", "Can whipsaw during trend transitions"],
                backtest_mode="both",
            ),
            "donchian": StrategyInfo(
                name="Donchian Channel Breakout",
                class_type=DonchianChannelStrategy,
                category=StrategyCategory.TREND_FOLLOWING,
                description="The classic Turtle Trader strategy. Goes long on N-day high breakouts, exits on M-day low breakouts. Simple, robust, and effective across all timeframes and asset classes.",
                complexity="Beginner",
                time_horizon="Medium to Long-term",
                best_for=["Trending markets", "Futures trading", "Long-term trend following", "Systematic trading"],
                parameters={
                    "entry_period": {"default": 20, "range": (10, 55), "description": "Period for entry breakout (N-day high/low)"},
                    "exit_period": {"default": 10, "range": (5, 20), "description": "Period for exit breakout (M-day high/low)"},
                    "use_both_sides": {"default": True, "range": [True, False], "description": "Trade both long and short positions"},
                },
                pros=[
                    "Extremely simple and objective",
                    "Works across all markets and timeframes",
                    "Catches big trends early",
                    "Proven track record (Turtle Traders)",
                    "No curve fitting",
                    "Always in the market (captures all moves)",
                ],
                cons=[
                    "Many small losses during ranging markets",
                    "Low win rate (30-40%)",
                    "Large drawdowns during whipsaws",
                    "Requires discipline to follow",
                    "Late exits can give back profits",
                ],
                backtest_mode="both",
            ),
            "donchian_atr": StrategyInfo(
                name="Donchian ATR Strategy",
                class_type=DonchianATRStrategy,
                category=StrategyCategory.TREND_FOLLOWING,
                description="Enhanced Donchian strategy with ATR-based risk management. Uses volatility for position sizing and dynamic stop losses.",
                complexity="Intermediate",
                time_horizon="Medium to Long-term",
                best_for=["Risk-managed trend following", "Volatile markets", "Professional trading", "Portfolio management"],
                parameters={
                    "entry_period": {"default": 20, "range": (10, 55), "description": "Period for entry breakout"},
                    "exit_period": {"default": 10, "range": (5, 20), "description": "Period for exit breakout"},
                    "atr_period": {"default": 14, "range": (10, 30), "description": "Period for ATR calculation"},
                    "atr_multiplier": {"default": 2.0, "range": (1.0, 4.0), "description": "ATR multiplier for stop loss"},
                },
                pros=[
                    "Risk-adjusted position sizing",
                    "Dynamic stop losses",
                    "Better risk management",
                    "Filters weak breakouts",
                    "Adapts to volatility",
                ],
                cons=["More complex than classic version", "May miss some breakouts", "Requires ATR calculation", "Parameter optimization needed"],
                backtest_mode="both",
            ),
            "filtered_donchian": StrategyInfo(
                name="Filtered Donchian Strategy",
                class_type=FilteredDonchianStrategy,
                category=StrategyCategory.TREND_FOLLOWING,
                description="Donchian breakouts with trend filter. Only takes trades in direction of longer-term trend. Reduces whipsaws.",
                complexity="Intermediate",
                time_horizon="Medium to Long-term",
                best_for=["Reducing false breakouts", "Trend-aligned trading", "Lower drawdown tolerance", "Conservative trend following"],
                parameters={
                    "entry_period": {"default": 20, "range": (10, 55), "description": "Period for entry breakout"},
                    "exit_period": {"default": 10, "range": (5, 20), "description": "Period for exit breakout"},
                    "trend_period": {"default": 50, "range": (20, 200), "description": "Period for trend filter MA"},
                },
                pros=["Fewer false breakouts", "Better win rate", "Reduced whipsaws", "Trend-aligned entries", "Lower drawdowns"],
                cons=["Misses counter-trend moves", "Later entries than classic", "More parameters to optimize", "May miss trend reversals"],
                backtest_mode="both",
            ),
            "parabolic_sar": StrategyInfo(
                name="Parabolic SAR",
                class_type=ParabolicSARStrategy,
                category=StrategyCategory.TREND_FOLLOWING,
                description="Stop And Reverse strategy. Follows price trends and reverses when price crosses the indicator.",
                complexity="Intermediate",
                time_horizon="Medium-term",
                best_for=[
                    "Trending markets",
                    "Trailing stops",
                    "Crypto/Volatile assets",
                ],
                parameters={
                    "start": {
                        "default": 0.02,
                        "range": (0.01, 0.05),
                        "description": "Start acceleration factor",
                    },
                    "increment": {
                        "default": 0.02,
                        "range": (0.01, 0.05),
                        "description": "Acceleration increment",
                    },
                    "maximum": {
                        "default": 0.2,
                        "range": (0.1, 0.5),
                        "description": "Max acceleration factor",
                    },
                },
                pros=[
                    "Good visual trend indicator",
                    "Built-in exit strategy",
                    "Always in the market",
                    "Accelerates as trend matures",
                ],
                cons=[
                    "Whipsaws in ranging markets",
                    "Always in market (can be a con)",
                    "Late entries in fast reversals",
                ],
                backtest_mode="both",
            ),
        }
        return catalog
