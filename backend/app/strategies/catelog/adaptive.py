from typing import Dict

from ...strategies.catelog.category import StrategyCategory
from ...strategies.catelog.strategy_info import StrategyInfo
from ...strategies.ml.regime_aware.adaptive_strategy_switcher import AdaptiveStrategySwitcher
from ...strategies.ml.regime_aware.regime_specific_models import RegimeSpecificStrategy


class Adaptive:
    def __init__(self):
        self.strategies = self._build_catalog()

    def _build_catalog(self) -> Dict[str, StrategyInfo]:
        """Build the strategy catalog"""

        catalog = {

            "regime_adaptive": StrategyInfo(
                name="Regime-Adaptive Strategy",
                class_type=RegimeSpecificStrategy,
                category=StrategyCategory.ADAPTIVE,
                description="Dynamically switches between specialist strategies based on HMM regime detection. Uses different strategy configurations for bull, bear, and neutral markets.",
                complexity="Advanced",
                time_horizon="Medium-term",
                best_for=["Regime-aware trading", "Dynamic allocation", "Multi-strategy portfolio"],
                parameters={
                    "lookback_days": {"default": 252, "range": (60, 504),
                                      "description": "Historical data lookback (days)"},
                    "regime_update_freq": {"default": 5, "range": (1, 20),
                                           "description": "Regime re-detection frequency (days)"},
                    "use_markov_chain": {"default": True, "range": [True, False],
                                         "description": "Use Markov chain regime transitions"},
                },
                pros=[
                    "Adapts to market regimes automatically",
                    "Specialist strategies per regime",
                    "Reduces drawdowns in bear markets",
                    "HMM-based regime detection",
                ],
                cons=[
                    "Complex multi-model architecture",
                    "Regime detection lag",
                    "Requires substantial historical data",
                    "Multiple strategies to maintain",
                ],
                backtest_mode="single",
            ),
            "adaptive_strategy_switcher": StrategyInfo(
                name="Adaptive Strategy Switcher",
                class_type=AdaptiveStrategySwitcher,
                category=StrategyCategory.ADAPTIVE,
                description="Multi-strategy portfolio that weights strategies by their historical performance in the current market regime. Continuously adapts allocation.",
                complexity="Advanced",
                time_horizon="Medium-term",
                best_for=["Portfolio of strategies", "Regime-aware switching", "Strategy ensemble"],
                parameters={
                    "performance_lookback": {"default": 60, "range": (20, 120),
                                             "description": "Performance evaluation window (days)"},
                    "rebalance_frequency": {"default": 20, "range": (5, 60),
                                            "description": "Rebalance frequency (days)"},
                    "min_regime_confidence": {"default": 0.6, "range": (0.3, 0.9),
                                              "description": "Minimum confidence for regime signal"},
                    "use_ensemble": {"default": True, "range": [True, False],
                                     "description": "Use ensemble of strategies"},
                },
                pros=[
                    "Combines multiple strategy signals",
                    "Performance-weighted allocation",
                    "Regime-aware switching",
                    "Diversification across strategies",
                ],
                cons=[
                    "Complexity of multi-strategy system",
                    "Performance chasing risk",
                    "Higher computational cost",
                    "Requires all sub-strategies to be functional",
                ],
                backtest_mode="single",
            ),
        }
        return catalog
