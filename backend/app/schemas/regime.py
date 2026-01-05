"""
Market regime schemas
"""

from typing import Dict, List, Optional

from pydantic import BaseModel


class RegimeDetectionRequest(BaseModel):
    symbol: str = "SPY"
    period: str = "2y"
    interval: str = "1d"


class RegimeDetection(BaseModel):
    symbol: str
    regime: str  # bull, bear, sideways
    confidence: float
    timestamp: str


class DurationPrediction(BaseModel):
    current_regime: str
    expected_duration: float
    median_duration: float
    std_duration: float
    probability_end_next_week: float
    sample_size: int


class ChangeWarning(BaseModel):
    warning: bool
    confidence_trend: float
    disagreement_rate: float
    recommendation: str


class RegimeDetectionResult(BaseModel):
    regime: str
    confidence: float
    scores: Dict[str, float]
    method: str
    strategy_allocation: Dict[str, float]
    regime_strength: float
    duration_prediction: Optional[DurationPrediction]
    change_warning: Optional[ChangeWarning]
    statistical_regime: Optional[str]
    ml_regime: Optional[str]


class RegimeHistoryItem(BaseModel):
    start_date: str
    end_date: Optional[str] = None
    regime: str
    duration_days: int


class RegimeHistory(BaseModel):
    symbol: str
    history: List[RegimeHistoryItem]


class BatchRegimeRequest(BaseModel):
    symbols: List[str]
    period: str = "2y"


class BatchRegimeResponse(BaseModel):
    results: List[RegimeDetection]
