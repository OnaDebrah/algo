from typing import Dict

from ...strategies.catelog.category import StrategyCategory
from ...strategies.catelog.strategy_info import StrategyInfo
from ...strategies.stat_arb.sector_neutral import SectorNeutralStrategy
from ..stat_arb.base_stat_arb import RiskParityStatArb


class StatisticalArbitrage:
    def __init__(self):
        self.strategies = self._build_catalog()

    def _build_catalog(self) -> Dict[str, StrategyInfo]:
        """Build the strategy catalog"""

        catalog = {
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
        }
        return catalog
