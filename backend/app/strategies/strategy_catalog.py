"""
Strategy Catalog and Categories
Organize all trading strategies by type - Expanded Version
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Type

from backend.app.strategies import (
    BaseStrategy,
    DynamicStrategy,
    KAMAStrategy,
    MACDStrategy,
    MLStrategy,
    MultiTimeframeKAMAStrategy,
)
from backend.app.strategies.adpative_trend_ff_strategy import AdaptiveTrendFollowingStrategy
from backend.app.strategies.cs_momentum_strategy import CrossSectionalMomentumStrategy
from backend.app.strategies.donchain_strategy import DonchianATRStrategy, DonchianChannelStrategy, FilteredDonchianStrategy
from backend.app.strategies.kalman_filter_strategy import KalmanFilterStrategy
from backend.app.strategies.lstm_strategy import LSTMStrategy

# ML Strategies
from backend.app.strategies.ml.mc_ml_sentiment_strategy import MonteCarloMLSentimentStrategy
from backend.app.strategies.pairs_trading_strategy import PairsTradingStrategy
from backend.app.strategies.parabolic_sar import ParabolicSARStrategy
from backend.app.strategies.stat_arb.base_stat_arb import RiskParityStatArb
from backend.app.strategies.stat_arb.sector_neutral import SectorNeutralStrategy
from backend.app.strategies.technical.bb_mean_reversion import BollingerMeanReversionStrategy
from backend.app.strategies.technical.rsi_strategy import RSIStrategy
from backend.app.strategies.technical.sma_crossover import SMACrossoverStrategy

# Statistical Arbitrage
from backend.app.strategies.ts_momentum_strategy import TimeSeriesMomentumStrategy

# Kalman Filter HFT (conditional on numba)
try:
    from backend.app.strategies.kalman_filter_strategy import KalmanFilterStrategyHFT

    HFT_AVAILABLE = True
except ImportError:
    HFT_AVAILABLE = False
from backend.app.strategies.volatility.dynamic_scaling import DynamicVolatilityScalingStrategy
from backend.app.strategies.volatility.variance_risk_premium import VarianceRiskPremiumStrategy

# Volatility strategies
from backend.app.strategies.volatility.volatility_breakout import VolatilityBreakoutStrategy
from backend.app.strategies.volatility.volatility_targeting import VolatilityTargetingStrategy

logger = logging.getLogger(__name__)


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
    backtest_mode: Literal["single", "multi", "both"] = "single"  # Which backtest modes this strategy supports


class StrategyCatalog:
    """Catalog of all available trading strategies"""

    def __init__(self):
        self.strategies = self._build_catalog()

    def _build_catalog(self) -> Dict[str, StrategyInfo]:
        """Build the strategy catalog"""

        catalog = {
            "visual_builder": StrategyInfo(
                name="Visual Strategy Builder",
                class_type=DynamicStrategy,
                category=StrategyCategory.HYBRID,
                description="Custom strategy constructed using the visual Block Builder. Allows combining multiple ML models with technical filters and complex logic.",
                complexity="Advanced",
                time_horizon="Adaptive",
                best_for=["Custom logic", "Multi-factor models", "Hybrid strategies"],
                parameters={
                    "blocks": {
                        "default": [],
                        "description": "JSON block configuration defining the strategy logic",
                    },
                    "root_block_id": {
                        "default": "root",
                        "description": "The ID of the block that generates the final signal",
                    },
                },
                pros=[
                    "Highly customizable",
                    "No-code / Low-code approach",
                    "Allows combining ML with classic indicators",
                ],
                cons=[
                    "Complexity scales with number of blocks",
                    "Requires careful logic design",
                ],
                backtest_mode="both",
            ),
            # ============================================================
            # TECHNICAL INDICATORS - TREND FOLLOWING
            # ============================================================
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
            # ============================================================
            # MOMENTUM STRATEGIES
            # ============================================================
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
            # ============================================================
            # MEAN REVERSION
            # ============================================================
            "bb_mean_reversion": StrategyInfo(
                name="Bollinger Band Mean Reversion",
                class_type=BollingerMeanReversionStrategy,
                category=StrategyCategory.MEAN_REVERSION,
                description="Trades reversals from Bollinger Bands. Buy below lower band, sell above upper band.",
                complexity="Intermediate",
                time_horizon="Short-term",
                best_for=[
                    "Range-bound markets",
                    "Mean reversion",
                    "Swing trading",
                ],
                parameters={
                    "period": {
                        "default": 20,
                        "range": (10, 50),
                        "description": "Moving average period",
                    },
                    "std_dev": {
                        "default": 2.0,
                        "range": (1.5, 3.0),
                        "description": "Standard deviations",
                    },
                },
                pros=[
                    "Captures overextended moves",
                    "Dynamic support/resistance",
                    "Works well in ranges",
                    "Clear entry signals",
                ],
                cons=[
                    "Fails in strong trends (band walking)",
                    "Stop loss placement tricky",
                    "Can be early to counter-trend",
                ],
                backtest_mode="both",
            ),
            "pairs_trading": StrategyInfo(
                name="Pairs Trading",
                class_type=PairsTradingStrategy,
                category=StrategyCategory.PAIRS_TRADING,
                description="Identifies cointegrated asset pairs and trades their spread reversion to mean.",
                complexity="Advanced",
                time_horizon="Short to Medium-term",
                best_for=[
                    "Market-neutral strategies",
                    "Statistical arbitrage",
                    "Hedge funds",
                ],
                parameters={
                    "lookback": {
                        "default": 60,
                        "range": (30, 252),
                        "description": "Cointegration lookback period",
                    },
                    "entry_threshold": {
                        "default": 2.0,
                        "range": (1.0, 3.0),
                        "description": "Z-score entry threshold",
                    },
                    "exit_threshold": {
                        "default": 0.5,
                        "range": (0.0, 1.0),
                        "description": "Z-score exit threshold",
                    },
                },
                pros=[
                    "Market-neutral",
                    "Lower volatility",
                    "Exploits statistical relationships",
                    "Consistent returns in stable markets",
                ],
                cons=[
                    "Pairs can decouple",
                    "Requires careful pair selection",
                    "Higher transaction costs",
                    "Relationship breakdown risk",
                ],
                backtest_mode="multi",
            ),
            # ============================================================
            # VOLATILITY STRATEGIES
            # ============================================================
            "volatility_breakout": StrategyInfo(
                name="Volatility Breakout",
                class_type=VolatilityBreakoutStrategy,
                category=StrategyCategory.VOLATILITY,
                description="Trades breakouts from volatility bands (Bollinger Bands). Enters when price breaks out of bands.",
                complexity="Intermediate",
                time_horizon="Short-term",
                best_for=[
                    "Volatile markets",
                    "Breakout trading",
                    "Crypto and commodities",
                ],
                parameters={
                    "period": {
                        "default": 20,
                        "range": (10, 50),
                        "description": "Bollinger Band period",
                    },
                    "std_dev": {
                        "default": 2.0,
                        "range": (1.0, 3.0),
                        "description": "Standard deviations for bands",
                    },
                },
                pros=[
                    "Captures volatile moves",
                    "Visual and intuitive",
                    "Works in trending markets",
                    "Clear risk definition",
                ],
                cons=[
                    "Many false breakouts",
                    "Whipsaws in ranging markets",
                    "Requires quick execution",
                    "Stop losses essential",
                ],
                backtest_mode="single",
            ),
            "volatility_targeting": StrategyInfo(
                name="Volatility Targeting",
                class_type=VolatilityTargetingStrategy,
                category=StrategyCategory.VOLATILITY,
                description="Adjusts position size to maintain constant portfolio volatility. Scales down in high vol, up in low vol.",
                complexity="Advanced",
                time_horizon="All timeframes",
                best_for=[
                    "Risk management",
                    "Portfolio optimization",
                    "Professional trading",
                ],
                parameters={
                    "target_vol": {
                        "default": 0.15,
                        "range": (0.05, 0.30),
                        "description": "Target annualized volatility",
                    },
                    "lookback": {
                        "default": 21,
                        "range": (10, 63),
                        "description": "Volatility estimation period",
                    },
                },
                pros=[
                    "Consistent risk exposure",
                    "Improves Sharpe ratio",
                    "Reduces tail risk",
                    "Professional standard",
                ],
                cons=[
                    "Can miss big moves",
                    "Realized vs implied vol mismatch",
                    "Frequent rebalancing",
                    "Implementation complexity",
                ],
                backtest_mode="single",
            ),
            "dynamic_scaling": StrategyInfo(
                name="Dynamic Position Scaling",
                class_type=DynamicVolatilityScalingStrategy,
                category=StrategyCategory.VOLATILITY,
                description="Dynamically scales position sizes based on market conditions, volatility, and momentum.",
                complexity="Advanced",
                time_horizon="All timeframes",
                best_for=[
                    "Risk-adjusted returns",
                    "Drawdown management",
                    "Adaptive trading",
                ],
                parameters={
                    "base_size": {
                        "default": 0.1,
                        "range": (0.05, 0.25),
                        "description": "Base position size (% of capital)",
                    },
                    "scaling_factor": {
                        "default": 1.5,
                        "range": (1.0, 3.0),
                        "description": "Scaling multiplier",
                    },
                },
                pros=[
                    "Adapts to market conditions",
                    "Better risk management",
                    "Maximizes favorable conditions",
                    "Reduces losses in bad conditions",
                ],
                cons=[
                    "Complex to implement",
                    "Requires careful calibration",
                    "Can underperform in stable markets",
                    "Higher turnover",
                ],
                backtest_mode="single",
            ),
            "variance_risk_premium": StrategyInfo(
                name="Variance Risk Premium",
                class_type=VarianceRiskPremiumStrategy,
                category=StrategyCategory.VOLATILITY,
                description="Captures the premium between implied and realized volatility. Typically short volatility.",
                complexity="Advanced",
                time_horizon="Medium-term",
                best_for=[
                    "Options trading",
                    "Volatility arbitrage",
                    "Institutional strategies",
                ],
                parameters={
                    "lookback": {
                        "default": 30,
                        "range": (20, 90),
                        "description": "Historical vol lookback",
                    },
                    "threshold": {
                        "default": 0.05,
                        "range": (0.01, 0.10),
                        "description": "Premium threshold for entry",
                    },
                },
                pros=[
                    "Consistent premium capture",
                    "Well-researched strategy",
                    "Diversification benefits",
                    "Works in calm markets",
                ],
                cons=[
                    "Tail risk (vol spikes)",
                    "Requires options access",
                    "Negative skewness",
                    "Can blow up in crises",
                ],
                backtest_mode="single",
            ),
            # ============================================================
            # STATISTICAL ARBITRAGE
            # ============================================================
            "sector_neutral": StrategyInfo(
                name="Sector Neutral Arbitrage",
                class_type=SectorNeutralStrategy,
                category=StrategyCategory.STATISTICAL_ARBITRAGE,
                description="Market-neutral strategy that is neutral within each sector. Exploits intra-sector relationships.",
                complexity="Advanced",
                time_horizon="Short to Medium-term",
                best_for=[
                    "Hedge funds",
                    "Market-neutral portfolios",
                    "Statistical arbitrage",
                ],
                parameters={
                    "lookback": {
                        "default": 60,
                        "range": (30, 252),
                        "description": "Ranking lookback period",
                    },
                    "rebalance_freq": {
                        "default": 20,
                        "range": (5, 60),
                        "description": "Rebalancing frequency (days)",
                    },
                },
                pros=[
                    "Market risk neutralized",
                    "Lower volatility",
                    "Sector-specific alpha",
                    "Reduced systematic risk",
                ],
                cons=[
                    "Requires many stocks",
                    "Higher complexity",
                    "Execution challenges",
                    "Lower absolute returns",
                ],
                backtest_mode="multi",
            ),
            # ============================================================
            # MACHINE LEARNING
            # ============================================================
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
                    "strategy_type": {"default": "random_forest", "range": None, "description": "Model type"},
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
                backtest_mode="single",
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
                    "strategy_type": {"default": "gradient_boosting", "range": None, "description": "Model type"},
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
                backtest_mode="single",
            ),
            "ml_svm": StrategyInfo(
                name="ML SVM Classifier",
                class_type=MLStrategy,
                category=StrategyCategory.MACHINE_LEARNING,
                description="Uses Support Vector Machine (SVM) to classify market regimes and predict direction.",
                complexity="Advanced",
                time_horizon="Adaptable",
                best_for=[
                    "Regime classification",
                    "Non-linear boundaries",
                    "Small datasets",
                ],
                parameters={
                    "strategy_type": {"default": "svm", "range": None, "description": "Model type"},
                    "test_size": {
                        "default": 0.2,
                        "range": (0.1, 0.4),
                        "description": "Test set size",
                    },
                },
                pros=[
                    "Effective in high dimensions",
                    "Robust to overfitting",
                    "Good for regime detection",
                ],
                cons=[
                    "Slow on large datasets",
                    "Sensitive to noise",
                    "Hard to interpret probability",
                ],
                backtest_mode="single",
            ),
            "ml_logistic": StrategyInfo(
                name="ML Logistic Regression",
                class_type=MLStrategy,
                category=StrategyCategory.MACHINE_LEARNING,
                description="Uses Logistic Regression for simple, interpretable market direction prediction.",
                complexity="Intermediate",
                time_horizon="Adaptable",
                best_for=[
                    "Baseline models",
                    "Interpretability",
                    "Linear relationships",
                ],
                parameters={
                    "strategy_type": {"default": "logistic_regression", "range": None, "description": "Model type"},
                    "test_size": {
                        "default": 0.2,
                        "range": (0.1, 0.4),
                        "description": "Test set size",
                    },
                },
                pros=[
                    "Highly interpretable",
                    "Fast training",
                    "Less overfitting risk",
                ],
                cons=[
                    "Linear boundaries only",
                    "Underperforms on complex data",
                    "Requires feature engineering",
                ],
                backtest_mode="single",
            ),
            "ml_lstm": StrategyInfo(
                name="ML LSTM (Deep Learning)",
                class_type=LSTMStrategy,
                category=StrategyCategory.MACHINE_LEARNING,
                description="Uses Long Short-Term Memory (LSTM) neural network for time-series forecasting.",
                complexity="Expert",
                time_horizon="Short to Medium-term",
                best_for=[
                    "Time-series forecasting",
                    "Sequence patterns",
                    "Complex temporal dependencies",
                ],
                parameters={
                    "lookback": {
                        "default": 10,
                        "range": (5, 60),
                        "description": "Sequence lookback length",
                    },
                    "classes": {
                        "default": 2,
                        "range": (2, 3),
                        "description": "Number of classes (Up/Down)",
                    },
                    "epochs": {
                        "default": 20,
                        "range": (10, 100),
                        "description": "Training epochs",
                    },
                },
                pros=[
                    "Captures temporal dependencies",
                    "State-of-the-art for sequences",
                    "Non-linear mapping",
                ],
                cons=[
                    "Requires large data",
                    "Computationally expensive",
                    "Hard to train (vanishing gradients)",
                    "Black box",
                ],
                backtest_mode="single",
            ),
            # ============================================================
            # ADAPTIVE STRATEGIES
            # ============================================================
            "kalman_filter": StrategyInfo(
                name="Kalman Filter Pairs Strategy",
                class_type=KalmanFilterStrategy,
                category=StrategyCategory.PAIRS_TRADING,
                description="Statistical arbitrage using Kalman Filtering to dynamically estimate the hedge ratio between two assets.",
                complexity="Advanced",
                time_horizon="Intraday to Medium-term",
                best_for=[
                    "Pairs Trading",
                    "Statistical Arbitrage",
                    "Mean Reversion in Cointegrated Assets",
                    "Cointegrated assets",
                    "Market-neutral portfolios",
                ],
                parameters={
                    "asset_1": {
                        "default": "AAPL",
                        "range": None,
                        "description": "First asset in the pair",
                    },
                    "asset_2": {
                        "default": "MSFT",
                        "range": None,
                        "description": "Second asset in the pair",
                    },
                    "entry_z": {
                        "default": 2.0,
                        "range": (1.0, 4.0),
                        "description": "Z-score threshold for trade entry",
                    },
                    "exit_z": {
                        "default": 0.5,
                        "range": (0.0, 1.0),
                        "description": "Z-score threshold for mean reversion exit",
                    },
                    "stop_loss_z": {
                        "default": 3.0,
                        "range": (2.0, 5.0),
                        "description": "Z-score threshold for stop loss",
                    },
                    "transitory_std": {
                        "default": 0.01,
                        "range": (0.0001, 0.1),
                        "description": "System noise: how fast the hedge ratio (Beta) can change",
                    },
                    "observation_std": {
                        "default": 0.1,
                        "range": (0.01, 1.0),
                        "description": "Measurement noise: how much price noise to ignore",
                    },
                    "decay_factor": {
                        "default": 0.99,
                        "range": (0.90, 1.0),
                        "description": "Forgetting factor for old price observations",
                    },
                    "min_obs": {
                        "default": 20,
                        "range": (10, 60),
                        "description": "Minimum observations before Kalman starts trading",
                    },
                },
                pros=[
                    "Dynamic hedge ratio (Beta) updates instantly",
                    "Superior to rolling OLS for non-stationary spreads",
                    "Mathematically optimal Bayesian estimation",
                    "Confidence-weighted position sizing",
                ],
                cons=[
                    "Highly sensitive to transitory_std parameter",
                    "Risk of 'over-adapting' to market noise",
                    "Requires cointegrated asset pairs to be effective",
                    "More complex than simple pairs trading",
                ],
                backtest_mode="multi",
            ),
            # ============================================================
            # STATISTICAL ARBITRAGE - RISK PARITY
            # ============================================================
            "risk_parity_stat_arb": StrategyInfo(
                name="Risk Parity Statistical Arbitrage",
                class_type=RiskParityStatArb,
                category=StrategyCategory.STATISTICAL_ARBITRAGE,
                description="Market-neutral statistical arbitrage that allocates based on risk contribution rather than equal weights. Builds cointegrated baskets and trades mean-reverting spreads with risk-parity position sizing.",
                complexity="Advanced",
                time_horizon="Short to Medium-term",
                best_for=[
                    "Market-neutral portfolios",
                    "Risk-adjusted stat arb",
                    "Institutional trading",
                    "Balanced risk allocation",
                ],
                parameters={
                    "universe": {
                        "default": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
                        "range": None,
                        "description": "List of asset symbols for basket construction",
                    },
                    "basket_size": {
                        "default": 3,
                        "range": (2, 10),
                        "description": "Number of assets per trading basket",
                    },
                    "lookback_period": {
                        "default": 252,
                        "range": (60, 504),
                        "description": "Period for cointegration analysis (days)",
                    },
                    "entry_threshold": {
                        "default": 2.0,
                        "range": (1.0, 3.0),
                        "description": "Z-score entry threshold",
                    },
                    "exit_threshold": {
                        "default": 0.5,
                        "range": (0.0, 1.0),
                        "description": "Z-score exit threshold",
                    },
                    "stop_loss_threshold": {
                        "default": 3.0,
                        "range": (2.0, 5.0),
                        "description": "Z-score stop loss threshold",
                    },
                    "method": {
                        "default": "cointegration",
                        "range": ["cointegration", "pca", "kalman"],
                        "description": "Basket construction method",
                    },
                },
                pros=[
                    "Risk-balanced position sizing",
                    "Market-neutral by design",
                    "Lower volatility than equal-weight stat arb",
                    "Exploits multi-asset cointegration",
                    "Institutional-grade risk management",
                ],
                cons=[
                    "Requires multiple correlated assets",
                    "Complex implementation",
                    "Cointegration relationships can break down",
                    "Higher computational requirements",
                    "Sensitive to lookback period choice",
                ],
                backtest_mode="multi",
            ),
            # ============================================================
            # MACHINE LEARNING - MONTE CARLO SENTIMENT
            # ============================================================
            "mc_ml_sentiment": StrategyInfo(
                name="Monte Carlo ML Sentiment",
                class_type=MonteCarloMLSentimentStrategy,
                category=StrategyCategory.MACHINE_LEARNING,
                description="Combines sentiment analysis, machine learning predictions, and Monte Carlo simulation for probabilistic price forecasting. Uses Kelly Criterion for risk-aware position sizing.",
                complexity="Expert",
                time_horizon="Short to Medium-term",
                best_for=[
                    "Sentiment-driven trading",
                    "Probabilistic forecasting",
                    "Risk-aware position sizing",
                    "Multi-factor alpha generation",
                ],
                parameters={
                    "ml_model_type": {
                        "default": "gradient_boosting",
                        "range": ["gradient_boosting", "random_forest"],
                        "description": "ML model for return prediction",
                    },
                    "lookback_period": {
                        "default": 252,
                        "range": (60, 504),
                        "description": "Training data lookback (days)",
                    },
                    "forecast_horizon": {
                        "default": 20,
                        "range": (5, 60),
                        "description": "Forecast horizon (days)",
                    },
                    "num_simulations": {
                        "default": 10000,
                        "range": (1000, 50000),
                        "description": "Number of Monte Carlo paths",
                    },
                    "confidence_level": {
                        "default": 0.95,
                        "range": (0.90, 0.99),
                        "description": "Confidence level for VaR/bounds",
                    },
                    "sentiment_weight": {
                        "default": 0.3,
                        "range": (0.0, 1.0),
                        "description": "Weight of sentiment in combined signal",
                    },
                },
                pros=[
                    "Combines multiple alpha sources (sentiment + technical + ML)",
                    "Probabilistic risk assessment via Monte Carlo",
                    "Kelly Criterion position sizing",
                    "Automatic model retraining",
                    "Confidence-weighted signals",
                ],
                cons=[
                    "Requires sentiment data API access (Twitter, news)",
                    "Computationally intensive (Monte Carlo sims)",
                    "Complex pipeline with multiple failure points",
                    "Sentiment data quality varies",
                    "ML model overfitting risk",
                ],
                backtest_mode="single",
            ),
        }

        # Conditionally add HFT Kalman Filter strategy
        if HFT_AVAILABLE:
            catalog["kalman_filter_hft"] = StrategyInfo(
                name="Kalman Filter Pairs (HFT/Numba)",
                class_type=KalmanFilterStrategyHFT,
                category=StrategyCategory.PAIRS_TRADING,
                description="High-frequency version of Kalman Filter pairs trading with Numba JIT acceleration. Uses pre-compiled numerical routines for ultra-fast Kalman updates suitable for intraday trading.",
                complexity="Expert",
                time_horizon="Intraday to Short-term",
                best_for=[
                    "High-frequency pairs trading",
                    "Low-latency stat arb",
                    "Intraday mean reversion",
                    "Cointegrated asset pairs",
                ],
                parameters={
                    "asset_1": {
                        "default": "AAPL",
                        "range": None,
                        "description": "First asset in the pair",
                    },
                    "asset_2": {
                        "default": "MSFT",
                        "range": None,
                        "description": "Second asset in the pair",
                    },
                    "entry_z": {
                        "default": 2.0,
                        "range": (1.0, 4.0),
                        "description": "Z-score threshold for trade entry",
                    },
                    "exit_z": {
                        "default": 0.5,
                        "range": (0.0, 1.0),
                        "description": "Z-score threshold for mean reversion exit",
                    },
                    "stop_loss_z": {
                        "default": 3.0,
                        "range": (2.0, 5.0),
                        "description": "Z-score threshold for stop loss",
                    },
                    "transitory_std": {
                        "default": 0.01,
                        "range": (0.0001, 0.1),
                        "description": "System noise for hedge ratio changes",
                    },
                    "observation_std": {
                        "default": 0.1,
                        "range": (0.01, 1.0),
                        "description": "Measurement noise",
                    },
                },
                pros=[
                    "10-100x faster than pure Python Kalman updates",
                    "Numba JIT compilation for near-C performance",
                    "Suitable for intraday/HFT timeframes",
                    "Same mathematical model as standard Kalman pairs",
                ],
                cons=[
                    "Requires numba package installation",
                    "First run has JIT compilation overhead",
                    "Same pair selection challenges as standard Kalman",
                    "Not all systems support numba",
                ],
                backtest_mode="multi",
            )

        return catalog

    def get_by_mode(self, mode: str) -> Dict[str, StrategyInfo]:
        """Get strategies compatible with a backtest mode ('single' or 'multi')"""
        return {key: info for key, info in self.strategies.items() if info.backtest_mode == mode or info.backtest_mode == "both"}

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

        # Use defaults for missing parameters and sanitize types
        params = {}
        for param_name, param_info in info.parameters.items():
            val = kwargs.get(param_name, param_info.get("default"))

            # Robust type conversion: if default is int, ensure val is int
            # This handles cases like Bayesian optimization returning floats for windows
            default_val = param_info.get("default")
            if isinstance(default_val, int) and not isinstance(val, int) and val is not None:
                try:
                    # Capture float strings or direct floats
                    val = int(float(val))
                except (ValueError, TypeError):
                    logger.warning(f"Failed to cast parameter {param_name} to int: {val}")

            params[param_name] = val

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

    def get_strategy_count_by_category(self) -> Dict[str, int]:
        """Get count of strategies in each category"""
        counts = {}
        for category in self.get_categories():
            counts[category.value] = len(self.get_by_category(category))
        return counts

    def search_strategies(self, query: str) -> List[Dict]:
        """Search strategies by name, description, or tags"""
        results = []
        query_lower = query.lower()

        for key, info in self.strategies.items():
            if (
                query_lower in info.name.lower()
                or query_lower in info.description.lower()
                or any(query_lower in tag.lower() for tag in info.best_for)
            ):
                results.append({"key": key, "name": info.name, "category": info.category.value, "description": info.description})

        return results


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


def get_strategy_summary() -> Dict:
    """Get summary statistics of the strategy catalog"""
    catalog = get_catalog()

    return {
        "total_strategies": len(catalog.strategies),
        "by_category": catalog.get_strategy_count_by_category(),
        "by_complexity": {
            "Beginner": len(catalog.get_by_complexity("Beginner")),
            "Intermediate": len(catalog.get_by_complexity("Intermediate")),
            "Advanced": len(catalog.get_by_complexity("Advanced")),
        },
        "categories": [cat.value for cat in catalog.get_categories()],
    }
