"""
Strategy Catalog and Categories
Organize all trading strategies by type - Expanded Version
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Type

from streamlit.strategies import KAMAStrategy, MultiTimeframeKAMAStrategy
from streamlit.strategies.adpative_trend_ff_strategy import AdaptiveTrendFollowingStrategy
from streamlit.strategies.base_strategy import BaseStrategy
from streamlit.strategies.bb_mean_reversion import BollingerMeanReversionStrategy
from streamlit.strategies.cs_momentum_strategy import CrossSectionalMomentumStrategy
from streamlit.strategies.donchain_strategy import DonchianATRStrategy, DonchianChannelStrategy, FilteredDonchianStrategy
from streamlit.strategies.kalman_filter_strategy import KalmanFilterStrategy
from streamlit.strategies.lstm_strategy import LSTMStrategy
from streamlit.strategies.macd_strategy import MACDStrategy
from streamlit.strategies.ml_strategy import MLStrategy
from streamlit.strategies.options_strategies import OptionsStrategy
from streamlit.strategies.pairs_trading_strategy import PairsTradingStrategy
from streamlit.strategies.parabolic_sar import ParabolicSARStrategy
from streamlit.strategies.rsi_strategy import RSIStrategy
from streamlit.strategies.sma_crossover import SMACrossoverStrategy

# Statistical Arbitrage
from streamlit.strategies.stat_arb.sector_neutral import SectorNeutralStrategy
from streamlit.strategies.ts_momentum_strategy import TimeSeriesMomentumStrategy
from streamlit.strategies.volatility.dynamic_scaling import DynamicVolatilityScalingStrategy
from streamlit.strategies.volatility.variance_risk_premium import VarianceRiskPremiumStrategy

# Volatility strategies
from streamlit.strategies.volatility.volatility_breakout import VolatilityBreakoutStrategy
from streamlit.strategies.volatility.volatility_targeting import VolatilityTargetingStrategy


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
    OPTIONS = "Options Strategies"
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


class StrategyCatalog:
    """Catalog of all available trading strategies"""

    def __init__(self):
        self.strategies = self._build_catalog()

    def _build_catalog(self) -> Dict[str, StrategyInfo]:
        """Build the strategy catalog"""

        catalog = {
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
                    "model_type": {"default": "svm", "range": None, "description": "Model type"},
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
                    "model_type": {"default": "logistic_regression", "range": None, "description": "Model type"},
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
            ),
            # ============================================================
            # ADAPTIVE STRATEGIES
            # ============================================================
            "kalman_filter": StrategyInfo(
                name="Kalman Filter Pairs Strategy",
                class_type=KalmanFilterStrategy,
                category=StrategyCategory.ADAPTIVE,
                description="Statistical arbitrage using Kalman Filtering to dynamically estimate the hedge ratio between two assets.",
                complexity="Institutional",
                time_horizon="Intraday to Medium-term",
                best_for=[
                    "Pairs Trading",
                    "Statistical Arbitrage",
                    "Mean Reversion in Cointegrated Assets",
                ],
                parameters={
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
                    "Mathematically optimal signal-to-noise separation",
                ],
                cons=[
                    "Highly sensitive to transitory_std parameter",
                    "Risk of 'over-adapting' to market noise",
                    "Requires cointegrated asset pairs to be effective",
                ],
            ),
            # ============================================================
            # OPTIONS STRATEGIES
            # ============================================================
            "covered_call": StrategyInfo(
                name="Covered Call",
                class_type=OptionsStrategy,
                category=StrategyCategory.OPTIONS,
                description="Hold stock and sell call options to generate income. Limited upside, downside protected by premium.",
                complexity="Intermediate",
                time_horizon="Short to Medium-term",
                best_for=[
                    "Income generation",
                    "Range-bound markets",
                    "Conservative traders",
                ],
                parameters={
                    "strategy_type": {"default": "covered_call", "range": None, "description": "Options strategy type"},
                    "strike_pct": {
                        "default": 0.05,
                        "range": (0.01, 0.15),
                        "description": "Strike price % above current",
                    },
                    "dte": {
                        "default": 30,
                        "range": (7, 90),
                        "description": "Days to expiration",
                    },
                },
                pros=[
                    "Generates income",
                    "Reduces cost basis",
                    "Lower risk than naked long",
                    "Consistent returns in flat markets",
                ],
                cons=[
                    "Limited upside",
                    "Still exposed to downside",
                    "Opportunity cost if stock rallies",
                    "Early assignment risk",
                ],
            ),
            "iron_condor": StrategyInfo(
                name="Iron Condor",
                class_type=OptionsStrategy,
                category=StrategyCategory.OPTIONS,
                description="Market-neutral options strategy. Profits when underlying stays within a range. Limited risk and reward.",
                complexity="Advanced",
                time_horizon="Short-term",
                best_for=[
                    "Low volatility markets",
                    "Income generation",
                    "Range-bound stocks",
                ],
                parameters={
                    "strategy_type": {"default": "iron_condor", "range": None, "description": "Options strategy type"},
                    "wing_width": {
                        "default": 0.05,
                        "range": (0.03, 0.10),
                        "description": "Width of wings (% of price)",
                    },
                    "dte": {
                        "default": 30,
                        "range": (14, 60),
                        "description": "Days to expiration",
                    },
                },
                pros=[
                    "Defined risk",
                    "High probability strategy",
                    "Profits from time decay",
                    "Market neutral",
                ],
                cons=[
                    "Limited profit potential",
                    "Requires careful management",
                    "Pin risk near expiration",
                    "Complex adjustments needed",
                ],
            ),
            "butterfly_spread": StrategyInfo(
                name="Butterfly Spread",
                class_type=OptionsStrategy,
                category=StrategyCategory.OPTIONS,
                description="Limited risk strategy with concentrated profit zone. Profits when price stays near middle strike.",
                complexity="Advanced",
                time_horizon="Short-term",
                best_for=[
                    "Neutral outlook",
                    "Low volatility expected",
                    "Precise targets",
                ],
                parameters={
                    "strategy_type": {"default": "butterfly_spread", "range": None, "description": "Options strategy type"},
                    "wing_width": {
                        "default": 0.03,
                        "range": (0.02, 0.08),
                        "description": "Distance between strikes",
                    },
                    "dte": {
                        "default": 30,
                        "range": (14, 60),
                        "description": "Days to expiration",
                    },
                },
                pros=[
                    "Low cost to enter",
                    "Defined max loss",
                    "High reward/risk ratio at target",
                    "Works in neutral markets",
                ],
                cons=[
                    "Narrow profit zone",
                    "Lower probability of max profit",
                    "Time decay works against you early",
                    "Complex to manage",
                ],
            ),
            "straddle": StrategyInfo(
                name="Long Straddle",
                class_type=OptionsStrategy,
                category=StrategyCategory.OPTIONS,
                description="Profits from large moves in either direction. Buy ATM call and put. Volatility play.",
                complexity="Intermediate",
                time_horizon="Short-term",
                best_for=[
                    "Earnings events",
                    "High expected volatility",
                    "Direction unknown",
                ],
                parameters={
                    "strategy_type": {"default": "straddle", "range": None, "description": "Options strategy type"},
                    "dte": {
                        "default": 30,
                        "range": (7, 90),
                        "description": "Days to expiration",
                    },
                    "iv_threshold": {
                        "default": 0.30,
                        "range": (0.20, 0.60),
                        "description": "Implied volatility entry threshold",
                    },
                },
                pros=[
                    "Profits from big moves",
                    "Direction doesn't matter",
                    "Defined max loss",
                    "Great for events",
                ],
                cons=[
                    "Expensive to enter",
                    "Needs significant move",
                    "Time decay hurts",
                    "IV crush risk after event",
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
