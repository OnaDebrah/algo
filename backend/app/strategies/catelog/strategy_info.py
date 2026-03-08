from dataclasses import dataclass, field
from typing import Dict, List, Literal, Type

from ...strategies import BaseStrategy
from ...strategies.catelog.category import StrategyCategory


@dataclass
class StrategyInfo:
    """Information about a strategy"""

    name: str
    class_type: Type[BaseStrategy]
    category: StrategyCategory
    description: str
    complexity: str
    time_horizon: str
    best_for: List[str]
    parameters: Dict
    pros: List[str]
    cons: List[str]
    backtest_mode: Literal["single", "multi", "both"] = "single"
    tags: List[str] = field(default_factory=list)
    requires_ml_training: bool = False
    min_data_days: int = 60
