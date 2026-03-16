"""
Pydantic schemas for the Admin dashboard API.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel


class AdminUserListItem(BaseModel):
    id: int
    username: str
    email: str
    tier: str
    is_active: bool
    is_superuser: bool
    created_at: Optional[str] = None
    last_login: Optional[str] = None
    backtest_count: int = 0


class UsageLogItem(BaseModel):
    id: int
    action: str
    timestamp: Optional[str] = None
    metadata: Optional[str] = None


class AdminUserDetail(AdminUserListItem):
    country: Optional[str] = None
    investor_type: Optional[str] = None
    risk_profile: Optional[str] = None
    usage_logs: List[UsageLogItem] = []


class AdminStats(BaseModel):
    total_users: int
    users_by_tier: Dict[str, int]
    active_today: int
    total_backtests: int
    backtests_by_type: Dict[str, int]
    backtests_today: int
    backtests_this_week: int
    backtests_this_month: int
    active_live_strategies: int
    models_trained: int


class TierUpdateRequest(BaseModel):
    tier: str


class StatusUpdateRequest(BaseModel):
    is_active: bool


class SubmissionListItem(BaseModel):
    id: int
    name: str
    creator_name: str
    category: str
    complexity: str
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    price: float
    status: str
    submitted_at: Optional[str] = None


class SubmissionRejectRequest(BaseModel):
    rejection_reason: str
