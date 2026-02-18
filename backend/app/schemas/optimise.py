from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# Request/Response Models
class PortfolioRequest(BaseModel):
    symbols: List[str] = Field(..., min_length=2, description="List of stock symbols")
    lookback_days: int = Field(252, ge=30, le=1000, description="Historical lookback period in days")

    @field_validator("symbols")
    def validate_symbols(cls, v):
        if len(v) < 2:
            raise ValueError("Need at least 2 symbols")
        return [s.upper().strip() for s in v]


class OptimizationResponse(BaseModel):
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    method: str


class TargetReturnRequest(PortfolioRequest):
    target_return: float = Field(..., ge=-1.0, le=5.0, description="Target annual return (e.g., 0.15 for 15%)")


class BlackLittermanRequest(PortfolioRequest):
    views: Dict[str, float] = Field(..., description="Investor views {symbol: expected_return}")
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Confidence in views (0-1)")


class EfficientFrontierRequest(PortfolioRequest):
    num_portfolios: int = Field(50, ge=10, le=200, description="Number of portfolios to generate")


class BacktestRequest(BaseModel):
    symbols: List[str] = Field(..., min_length=1)
    weights: Dict[str, float] = Field(..., description="Portfolio weights")
    start_capital: float = Field(100000, ge=1000, description="Starting capital")
    period: str = Field("1y", description="Backtest period (e.g., '1y', '6mo', '2y')")

    @field_validator("weights")
    def validate_weights(cls, v):
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError("Weights must sum to 1.0")
        return v


class BacktestResponse(BaseModel):
    total_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    final_value: float


class ParamRange(BaseModel):
    min: float
    max: float
    step: Optional[float] = None
    type: str = Field("float", description="'int' or 'float'")

    @field_validator("type")
    def validate_type(cls, v):
        if v not in ["int", "float"]:
            raise ValueError("Type must be 'int' or 'float'")
        return v

    @model_validator(mode="after")
    def validate_range(self):
        if self.min > self.max:
            # Swap values if min > max to prevent Optuna errors
            self.min, self.max = self.max, self.min
        return self


class BayesianOptimizationRequest(BaseModel):
    tickers: List[str] = Field(..., description="List of stock tickers")
    strategy_key: str = Field(..., description="Strategy name")
    param_ranges: Dict[str, ParamRange] = Field(..., description="Parameter bounds")
    n_trials: int = Field(20, ge=5, le=100, description="Number of optimization trials")
    period: str = Field("1y", description="Backtest period")
    interval: str = Field("1d", description="Data interval")
    initial_capital: float = Field(100000.0, ge=1000.0)
    metric: str = Field("sharpe_ratio", description="Metric to maximize: 'sharpe_ratio', 'total_return', 'win_rate'")


class TrialResult(BaseModel):
    trial_id: int
    params: Dict[str, Any]
    value: float
    status: str


class BayesianOptimizationResponse(BaseModel):
    best_params: Dict[str, Any]
    best_value: float
    trials: List[TrialResult]
    tickers: List[str]
    strategy_key: str
    metric: str
    n_completed: int
    n_failed: int


class OptimizationRequest(BaseModel):
    """Request schema for parameter optimization"""

    symbol: str
    strategy_key: str
    interval: str
    metric: str
    initial_capital: float
    commission_rate: float = 0.001
    slippage_rate: float = 0.001
    param_ranges: Dict[str, Dict[str, Any]] = {}
    indicator_config: Dict[str, Any] = {"returns": True, "volatility": True, "moving_averages": True}

    class Config:
        arbitrary_types_allowed = True
