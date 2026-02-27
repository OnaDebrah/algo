from typing import Dict

from ...strategies.catelog.category import StrategyCategory
from ...strategies.catelog.strategy_info import StrategyInfo
from ...strategies.kalman_filter_strategy import KalmanFilterStrategy

try:
    from ..strategies.kalman_filter_strategy import KalmanFilterStrategyHFT

    HFT_AVAILABLE = True
except ImportError:
    HFT_AVAILABLE = False


class PairsTrading:
    def __init__(self):
        self.strategies = self._build_catalog()

    def _build_catalog(self) -> Dict[str, StrategyInfo]:
        """Build the strategy catalog"""

        catalog = {
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
        }

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
