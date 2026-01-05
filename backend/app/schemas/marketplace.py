"""
Marketplace schemas
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel


class PerformanceMetrics(BaseModel):
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int


class MarketplaceStrategy(BaseModel):
    id: int
    name: str
    description: str
    creator_id: int
    creator_name: str
    strategy_type: str
    category: str
    complexity: str
    parameters: Dict
    performance_metrics: Optional[PerformanceMetrics]
    price: float
    is_public: bool
    is_verified: bool
    version: str
    tags: List[str]
    downloads: int
    rating: float
    num_ratings: int
    num_reviews: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PublishStrategyRequest(BaseModel):
    name: str
    description: str
    strategy_type: str
    complexity: str
    parameters: Dict
    performance_metrics: Optional[PerformanceMetrics]
    price: float = 0
    is_public: bool = True
    tags: List[str] = []


class StrategyReview(BaseModel):
    id: Optional[int]
    strategy_id: int
    user_id: int
    username: str
    rating: int
    review_text: str
    performance_achieved: Optional[PerformanceMetrics]
    created_at: datetime

    class Config:
        from_attributes = True
