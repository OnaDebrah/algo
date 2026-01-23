from typing import Dict, List

from pydantic import BaseModel, Field, field_validator


# Request/Response Models
class PortfolioRequest(BaseModel):
    symbols: List[str] = Field(..., min_length=2, description="List of stock symbols")
    lookback_days: int = Field(252, ge=30, le=1000, description="Historical lookback period in days")

    @field_validator('symbols')
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

    @field_validator('weights')
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
