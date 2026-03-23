from typing import Dict

from ...strategies.catelog.category import StrategyCategory
from ...strategies.catelog.strategy_info import StrategyInfo
from ...strategies.pairs_trading_strategy import PairsTradingStrategy
from ...strategies.technical.bb_mean_reversion import BollingerMeanReversionStrategy


class MeanReversion:
    """Catalog of all available trading strategies"""

    def __init__(self):
        self.strategies = self._build_catalog()

    def _build_catalog(self) -> Dict[str, StrategyInfo]:
        """Build the strategy catalog"""

        catalog = {
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
                    "lookback": {
                        "default": 60,
                        "range": (30, 252),
                        "description": "Cointegration lookback period",
                    },
                    "entry_z": {
                        "default": 2.0,
                        "range": (1.0, 3.0),
                        "description": "Z-score entry threshold",
                    },
                    "exit_z": {
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
        }
        return catalog
