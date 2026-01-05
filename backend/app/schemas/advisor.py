"""
AI Advisor schemas
"""

from typing import List, Tuple

from pydantic import BaseModel


class UserProfile(BaseModel):
    goals: List[str]
    risk_tolerance: str
    time_horizon: str
    experience: str
    time_commitment: str
    capital: float
    market_preference: str


class StrategyRecommendation(BaseModel):
    strategy_key: str
    name: str
    fit_score: float
    why_recommended: List[str]
    personalized_insight: str
    risk_adjustment: str = None
    expected_return: Tuple[float, float]
    risk_level: str
    similar_traders_usage: str
