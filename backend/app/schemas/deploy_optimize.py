"""
Pydantic schemas for the pre-live portfolio optimization API.
"""

from typing import Dict, List

from pydantic import BaseModel


class OptimizePreviewRequest(BaseModel):
    symbols: List[str]
    lookback_days: int = 252
    initial_capital: float = 100000.0


class OptimizationResult(BaseModel):
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe: float
    method: str


class OptimizePreviewResponse(BaseModel):
    methods: Dict[str, OptimizationResult]
    symbols: List[str]
    equal_weight_baseline: OptimizationResult


class OptimizeApplyRequest(BaseModel):
    symbols: List[str]
    method: str
    lookback_days: int = 252
