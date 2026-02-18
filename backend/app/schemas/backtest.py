"""
Backtest schemas
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StrategyConfig(BaseModel):
    strategy_key: str
    parameters: Dict[str, Any]


class BacktestRequest(BaseModel):
    symbol: str
    strategy_key: str
    parameters: Dict
    period: str = "1y"
    interval: str = "1d"
    initial_capital: float = 100000
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    ml_model_id: Optional[str] = None  # For ML strategies: ID of a deployed model to use
    strategy_configs: Optional[Dict[str, StrategyConfig]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class BacktestResult(BaseModel):
    total_return: float
    total_return_pct: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_profit: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    final_equity: float
    initial_capital: float

    # Advanced metrics
    sortino_ratio: Optional[float] = 0.0
    calmar_ratio: Optional[float] = 0.0
    var_95: Optional[float] = 0.0
    cvar_95: Optional[float] = 0.0
    volatility: Optional[float] = 0.0
    expectancy: Optional[float] = 0.0
    total_commission: Optional[float] = 0.0

    # Factor Analysis
    alpha: Optional[float] = 0.0
    beta: Optional[float] = 0.0
    r_squared: Optional[float] = 0.0

    # Matrix Data (Year -> Month -> PctReturn)
    monthly_returns_matrix: Optional[Dict[str, Dict[str, float]]] = None


class EquityCurvePoint(BaseModel):
    timestamp: str
    equity: float
    cash: float
    drawdown: Optional[float] = None


class Trade(BaseModel):
    id: Optional[int] = None
    symbol: str
    order_type: str
    quantity: int
    price: float
    commission: float
    executed_at: str
    strategy: str
    total_value: float
    side: Optional[str] = None
    notes: Optional[str] = None
    profit: Optional[float] = None
    profit_pct: Optional[float] = None


class BacktestResponse(BaseModel):
    result: BacktestResult
    equity_curve: List[EquityCurvePoint]
    trades: List[Trade]
    price_data: Optional[List[Dict]] = None
    benchmark: Optional[Dict] = None


class MultiAssetBacktestRequest(BaseModel):
    symbols: List[str]
    strategy_configs: Dict[str, StrategyConfig]
    allocation_method: str = "equal"
    custom_allocations: Optional[Dict[str, float]] = None
    period: str = "1y"
    interval: str = "1d"
    initial_capital: float = 100000
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005


class SymbolStats(BaseModel):
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_return: float
    win_rate: float
    avg_profit: float


class MultiAssetBacktestResult(BacktestResult):
    symbol_stats: Dict[str, SymbolStats]
    num_symbols: int


class MultiAssetBacktestResponse(BaseModel):
    result: MultiAssetBacktestResult
    equity_curve: List[EquityCurvePoint]
    trades: List[Trade]
    price_data: Optional[List[Dict]] = None
    benchmark: Optional[Dict] = None


# Options backtest
class OptionsBacktestRequest(BaseModel):
    symbol: str
    strategy_type: str
    entry_rules: Dict
    exit_rules: Dict
    period: str = "1y"
    interval: str = "1d"
    initial_capital: float = 100000
    volatility: float = 0.3
    commission: float = 0.65


class OptionsBacktestResult(BacktestResult):
    avg_days_held: float
    avg_pnl_pct: float


class OptionsBacktestResponse(BaseModel):
    result: OptionsBacktestResult
    equity_curve: List[EquityCurvePoint]
    trades: List[Trade]


class BacktestHistoryItem(BaseModel):
    """Backtest history item for list view"""

    id: int
    name: Optional[str] = None
    backtest_type: str
    symbols: List[str]
    strategy_config: Dict[str, Any]
    period: str
    interval: str
    initial_capital: float

    # Results (optional - only for completed)
    total_return_pct: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    total_trades: Optional[int] = None
    final_equity: Optional[float] = None
    equity_curve: Optional[List[EquityCurvePoint]] = None
    trades: Optional[List[Trade]] = None

    # Metadata
    status: str
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None

    class Config:
        from_attributes = True


class PairsValidationRequest(BaseModel):
    asset_1: str
    asset_2: str
    period: str = "1y"
    interval: str = "1d"


class PairsValidationResponse(BaseModel):
    asset_1: str
    asset_2: str
    sector_1: str
    sector_2: str
    correlation: float
    cointegration_pvalue: float
    cointegration_statistic: float
    is_valid: bool
    warnings: list[str]
    errors: list[str]
    lookback_days: int


# Walk-Forward Analysis
class ParamRange(BaseModel):
    min: float
    max: float
    step: Optional[float] = None
    type: str = Field("float", description="'int' or 'float'")


class WFARequest(BaseModel):
    symbol: str
    strategy_key: str
    param_ranges: Dict[str, ParamRange]
    initial_capital: float = 100000
    period: str = "2y"
    interval: str = "1d"
    is_window_days: int = 180
    oos_window_days: int = 60
    step_days: int = 60
    anchored: bool = False
    metric: str = "sharpe_ratio"
    n_trials: int = 20
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005


class WFAFoldResult(BaseModel):
    fold_index: int
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str
    best_params: Dict[str, Any]
    is_metrics: BacktestResult
    oos_metrics: BacktestResult
    optimization_metadata: Dict[str, Any]


class WFAResponse(BaseModel):
    """Enhanced WFA response with robustness metrics"""

    folds: List[WFAFoldResult]
    aggregated_oos_metrics: BacktestResult
    oos_equity_curve: List[EquityCurvePoint]
    wfe: float
    robustness_metrics: Dict[str, float] = {}
    strategy_key: str
    symbol: str
    successful_folds: int = 0
    total_folds: int = 0

    class Config:
        arbitrary_types_allowed = True
