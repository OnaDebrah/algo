from typing import Dict

from ...strategies.catelog.category import StrategyCategory
from ...strategies.catelog.strategy_info import StrategyInfo
from .. import DynamicStrategy


class VisualBuilder:
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
            )
        }

        return catalog
