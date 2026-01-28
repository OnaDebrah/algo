"""
AI Advisor schemas
"""

from typing import List, Tuple, Dict, Any

from pydantic import BaseModel


class UserProfile(BaseModel):
    goals: List[str]
    risk_tolerance: str
    time_horizon: str
    experience: str
    time_commitment: str
    capital: float
    market_preference: str

class GuideRequest(BaseModel):
    goal: str
    risk: str
    experience: str
    capital: float
    timeHorizon: str
    markets: List[str]


class Theme(BaseModel):
    primary: str
    secondary: str
    bg: str
    border: str
    text: str
    icon: str  # String identifier for the icon


class Recommendation(BaseModel):
    id: int
    name: str
    tagline: str
    description: str
    fit_score: int
    risk_level: str
    theme: Theme
    expected_return: str
    similar_traders: str
    time_commitment: str
    success_rate: str
    min_capital: float
    why: List[str]
    pros: List[str]
    cons: List[str]
    best_for: List[str]
    performance_data: List[Dict[str, Any]]
    allocation_data: List[Dict[str, Any]]
    tags: List[str]
    icon: str  # String identifier


class GuideResponse(BaseModel):
    recommendations: List[Recommendation]
    radar_data: List[Dict[str, Any]]
