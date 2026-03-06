"""
Pydantic schemas for custom strategy endpoints
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── AI Generation ──────────────────────────────────────────────


class StrategyGenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=10, max_length=2000, description="Natural language strategy description")
    style: str = Field("technical", description="Strategy style: technical, fundamental, hybrid, momentum, mean_reversion, breakout")
    timeframe: Optional[str] = Field(None, description="Target timeframe hint")


class StrategyGenerateResponse(BaseModel):
    code: str
    explanation: str
    example_usage: str
    provider: str  # "anthropic", "deepseek", or "template"


# ── Code Validation ────────────────────────────────────────────


class StrategyValidateRequest(BaseModel):
    code: str = Field(..., min_length=10, description="Python strategy code to validate")


class StrategyValidateResponse(BaseModel):
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []


# ── Custom Strategy CRUD ───────────────────────────────────────


class CustomStrategyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    code: str = Field(..., min_length=10)
    strategy_type: str = Field("custom")
    parameters: Optional[Dict[str, Any]] = None
    ai_generated: bool = False
    ai_explanation: Optional[str] = None


class CustomStrategyUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    code: Optional[str] = Field(None, min_length=10)
    strategy_type: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class CustomStrategyResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    code: str
    strategy_type: str
    parameters: Optional[Dict[str, Any]] = None
    is_validated: bool
    ai_generated: bool
    ai_explanation: Optional[str] = None
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


# ── Custom Backtest ────────────────────────────────────────────


class CustomBacktestRequest(BaseModel):
    code: str = Field(..., min_length=10, description="Python strategy code to backtest")
    symbol: str = Field(..., description="Ticker symbol (e.g. AAPL)")
    period: str = Field("1y", description="Data period: 1mo, 3mo, 6mo, 1y, 2y, 5y")
    interval: str = Field("1d", description="Data interval: 1m, 5m, 1h, 1d")
    initial_capital: float = Field(100000, gt=0)
    commission_rate: float = Field(0.001, ge=0)
    slippage_rate: float = Field(0.0005, ge=0)
    strategy_name: Optional[str] = Field("Custom Strategy")
    custom_strategy_id: Optional[int] = Field(None, description="Link to saved custom strategy")
