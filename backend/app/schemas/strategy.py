"""
Strategy schemas
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class StrategyParameter(BaseModel):
    name: str
    type: str  # number, string, boolean, select
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    description: str
    options: Optional[List[str]] = None  # For select type


class StrategyInfo(BaseModel):
    key: str
    name: str
    description: str
    category: str
    complexity: Optional[str] = "Intermediate"
    time_horizon: Optional[str] = "Medium-term"
    best_for: List[str] = []
    parameters: List[StrategyParameter]


class Strategy(BaseModel):
    key: str
    name: str
    description: str
    category: str
    complexity: str
    time_horizon: str
    risk_level: str
    parameters: Dict[str, StrategyParameter]
    pros: List[str]
    cons: List[str]
