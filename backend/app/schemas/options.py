from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime

class OptionContract(BaseModel):
    strike: float
    type: str # 'call' or 'put'
    expiration: str
    premium: float

class OptionData(BaseModel):
    strike: float
    lastPrice: float
    bid: float
    ask: float
    volume: int
    openInterest: int
    impliedVolatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    inTheMoney: bool = False

class ChainRequest(BaseModel):
    symbol: str
    expiration: Optional[str] = None  # ✅ Make optional

class ChainResponse(BaseModel):
    symbol: str
    current_price: float
    expiration_dates: List[str]
    calls: List[OptionData]
    puts: List[OptionData]

class BacktestRequest(BaseModel):
    symbol: str
    strategy_type: str
    initial_capital: float
    risk_free_rate: float
    start_date: str
    end_date: str
    entry_rules: Dict[str, Any]
    exit_rules: Dict[str, Any]

class BacktestResult(BaseModel):
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    total_profit: float
    total_loss: float
    equity_curve: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]

# New schemas for enhanced endpoints

class OptionLegRequest(BaseModel):
    option_type: str  # 'CALL' or 'PUT'
    strike: float
    expiration: str  # ISO format date
    quantity: int  # Positive for long, negative for short
    premium: Optional[float] = None  # If None, will be calculated

class StrategyAnalysisRequest(BaseModel):
    symbol: str
    legs: List[OptionLegRequest]
    volatility: Optional[float] = None  # If None, will be estimated

class GreeksResponse(BaseModel):
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

class PayoffPoint(BaseModel):
    price: float
    payoff: float

class StrategyAnalysisResponse(BaseModel):
    symbol: str
    current_price: float
    initial_cost: float
    greeks: GreeksResponse
    breakeven_points: List[float]
    max_profit: float
    max_profit_condition: str
    max_loss: float
    max_loss_condition: str
    probability_of_profit: float
    payoff_diagram: List[PayoffPoint]

class GreeksRequest(BaseModel):
    symbol: str
    legs: List[OptionLegRequest]
    volatility: Optional[float] = None

class StrategyComparisonRequest(BaseModel):
    symbol: str
    strategies: List[Dict[str, Any]]  # List of strategy configurations

class StrategyComparisonResponse(BaseModel):
    symbol: str
    current_price: float
    comparisons: List[Dict[str, Any]]

# Options Analytics Schemas

class ProbabilityRequest(BaseModel):
    current_price: float
    strike: float
    days_to_expiration: int
    volatility: float
    risk_free_rate: Optional[float] = 0.05
    option_type: str  # 'call' or 'put'

class ProbabilityResponse(BaseModel):
    probability_itm: float
    probability_otm: float
    probability_touch: float
    expected_return_long: float
    expected_return_short: float

class StrikeAnalysis(BaseModel):
    strike: float
    moneyness: float
    premium_estimate: float
    prob_itm: float
    prob_otm: float
    expected_return: float

class StrikeOptimizerRequest(BaseModel):
    symbol: str
    current_price: float
    volatility: float
    days_to_expiration: int
    strategy_type: str  # 'covered_call', 'cash_secured_put', etc.
    num_strikes: Optional[int] = 10

class StrikeOptimizerResponse(BaseModel):
    symbol: str
    strategy_type: str
    current_price: float
    strikes: List[StrikeAnalysis]

class RiskMetricsRequest(BaseModel):
    portfolio_value: float
    returns: List[float]
    confidence_level: Optional[float] = 0.95

class RiskMetricsResponse(BaseModel):
    var_95: float
    cvar_95: float
    kelly_fraction: Optional[float] = None
    recommendation: str

class PortfolioPosition(BaseModel):
    pnl: float
    pnl_pct: float
    days_held: int

class PortfolioStatsRequest(BaseModel):
    positions: List[PortfolioPosition]

class PortfolioStatsResponse(BaseModel):
    total_pnl: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float
    avg_days_held: float
    avg_return_pct: float
    std_return_pct: float
    kelly_fraction: float
    expectancy: float

class MonteCarloRequest(BaseModel):
    current_price: float
    volatility: float
    days: int
    num_simulations: Optional[int] = 10000
    drift: Optional[float] = 0.0

class MonteCarloResponse(BaseModel):
    mean_final_price: float
    median_final_price: float
    std_final_price: float
    percentile_5: float
    percentile_95: float
    probability_above_current: float
    simulated_prices: List[float]  # Sample of simulations
