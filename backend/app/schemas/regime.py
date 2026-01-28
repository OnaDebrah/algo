from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime

class RegimeMetrics(BaseModel):
    volatility: float
    trend_strength: float
    liquidity_score: float
    correlation_index: float

class RegimeData(BaseModel):
    id: str
    name: str  # e.g., "Bullish Trend", "High Volatility"
    description: str
    start_date: datetime
    end_date: Optional[datetime]
    confidence: float
    metrics: RegimeMetrics
    # Could add related assets or strategy recommendations here

class CurrentRegimeResponse(BaseModel):
    symbol: str
    current_regime: RegimeData
    historical_regimes: List[RegimeData]
    market_health_score: float

class StrategyAllocation(BaseModel):
    trend_following: float
    momentum: float
    volatility_strategies: float
    mean_reversion: float
    statistical_arbitrage: float
    cash: float

class AllocationResponse(BaseModel):
    symbol: str
    current_regime: str
    confidence: float
    allocation: StrategyAllocation
    timestamp: str

class RegimeStrengthResponse(BaseModel):
    symbol: str
    current_regime: str
    strength: float  # 0-1 scale
    confirming_signals: int
    total_signals: int
    description: str
    timestamp: str

class WarningResponse(BaseModel):
    symbol: str
    current_regime: str
    warning: bool
    confidence_trend: float
    disagreement_rate: float
    recommendation: str  # maintain, increase_cash, increase_cash_significantly
    timestamp: str

class TransitionProbability(BaseModel):
    from_regime: str
    to_regime: str
    probability: float

class TransitionResponse(BaseModel):
    symbol: str
    current_regime: str
    expected_duration: float
    median_duration: float
    probability_end_next_week: float
    likely_transitions: List[TransitionProbability]
    timestamp: str

class FeatureImportance(BaseModel):
    feature: str
    importance: float
    current_value: float

class FeaturesResponse(BaseModel):
    symbol: str
    current_regime: str
    top_features: List[FeatureImportance]
    timestamp: str
