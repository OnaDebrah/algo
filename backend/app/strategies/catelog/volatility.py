from typing import Dict

from ...strategies.catelog.category import StrategyCategory
from ...strategies.catelog.strategy_info import StrategyInfo
from ..volatility.dynamic_scaling import DynamicVolatilityScalingStrategy
from ..volatility.variance_risk_premium import VarianceRiskPremiumStrategy
from ..volatility.volatility_breakout import VolatilityBreakoutStrategy
from ..volatility.volatility_targeting import VolatilityTargetingStrategy


class Volatility:
    """Catalog of all available trading strategies"""

    def __init__(self):
        self.strategies = self._build_catalog()

    def _build_catalog(self) -> Dict[str, StrategyInfo]:
        """Build the strategy catalog"""

        catalog = {
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
        }

        return catalog
