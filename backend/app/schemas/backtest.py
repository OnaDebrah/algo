"""
Backtest schemas
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class BacktestRequest(BaseModel):
    symbol: str
    strategy_key: str
    parameters: Dict
    period: str = "1y"
    interval: str = "1d"
    initial_capital: float = 100000
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005


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
    timestamp: str
    strategy: str
    profit: Optional[float] = None
    profit_pct: Optional[float] = None


class BacktestResponse(BaseModel):
    result: BacktestResult
    equity_curve: List[EquityCurvePoint]
    trades: List[Trade]
    price_data: Optional[List[Dict]] = None
    benchmark: Optional[Dict] = None


# Multi-asset backtest
class StrategyConfig(BaseModel):
    strategy_key: str
    parameters: Dict[str, Any]


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

    # Metadata
    status: str
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None

    class Config:
        from_attributes = True