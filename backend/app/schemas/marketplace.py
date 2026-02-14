from typing import Dict, List, Optional

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
    total_return: float = 0.0
    monthly_return: float = 0.0
    drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    num_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    volatility: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    var_95: float = 0.0
    initial_capital: float = 10000.0
    total_downloads: int = 0
    tags: List[str] = []
    best_for: List[str] = []
    pros: List[str] = []
    cons: List[str] = []
    is_favorite: bool = False
    is_verified: bool = False
    verification_badge: Optional[str] = None
    publish_date: str = ""


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
    pros: List[str] = []
    cons: List[str] = []
    risk_level: Optional[str] = "medium"  # low, medium, high
    recommended_capital: Optional[float] = 10000.0
    backtest_id: Optional[int] = None  # Link to an existing backtest
    strategy_key: Optional[str] = None  # Strategy type identifier


class ReviewCreateRequest(BaseModel):
    rating: int
    review_text: str
    performance_achieved: Optional[Dict] = None
