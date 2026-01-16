from typing import List, Optional, Dict
from pydantic import BaseModel

class BacktestResultsSchema(BaseModel):
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    calmar_ratio: float
    num_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_trade_duration: float
    volatility: float
    var_95: float
    cvar_95: float
    equity_curve: List[Dict] = []
    trades: List[Dict] = []
    daily_returns: List[Dict] = []
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_capital: float
    symbols: List[str]

class StrategyReviewSchema(BaseModel):
    id: Optional[int] = None
    strategy_id: int
    user_id: int
    username: str
    rating: int
    review_text: str
    performance_achieved: Optional[Dict] = None
    created_at: str

class StrategyListing(BaseModel):
    id: str | int
    name: str
    creator: str
    description: str
    rating: float
    reviews: int
    price: float
    category: str
    complexity: str
    time_horizon: str = "Medium-term"
    monthly_return: float
    drawdown: float
    sharpe_ratio: float
    total_downloads: int
    tags: List[str]
    best_for: List[str]
    pros: List[str]
    cons: List[str]
    is_favorite: bool
    is_verified: bool
    publish_date: str

class StrategyListingDetailed(StrategyListing):
    backtest_results: Optional[BacktestResultsSchema] = None
    reviews_list: List[StrategyReviewSchema] = []

class StrategyPublishRequest(BaseModel):
    name: str
    description: str
    category: str
    complexity: str
    price: float
    is_public: bool = True
    tags: List[str] = []
    backtest_id: Optional[int] = None  # Link to an existing backtest

class ReviewCreateRequest(BaseModel):
    rating: int
    review_text: str
    performance_achieved: Optional[Dict] = None
