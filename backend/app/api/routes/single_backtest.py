from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from core.trading_engine import TradingEngine
from strategies.strategy_catalog import get_catalog

from core import fetch_stock_data

router = APIRouter()


class BacktestRequest(BaseModel):
    symbol: str
    strategy_name: str
    initial_capital: float = 100000.0
    period: str = "1y"
    interval: str = "1d"
    commission_rate: float = 0.0
    slippage_rate: float = 0.0
    parameters: Dict[str, Any] = {}


class BacktestResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]]
    message: Optional[str]


@router.post("/api/backtest/single", response_model=BacktestResponse)
async def run_single_backtest(request: BacktestRequest):
    """
    Run single asset backtest
    """
    try:
        data = fetch_stock_data(request.symbol, period=request.period, interval=request.interval)
        if data.empty:
            return {"status": "error", "message": f"No data found for {request.symbol}"}

        catalog = get_catalog()
        strategy = catalog.create_strategy(
            request.strategy_type,
            **request.parameters
        )

        strategy_info = catalog.strategies.get(request.strategy_name)

        if not strategy_info:
            return {"status": "error", "message": f"Strategy {request.strategy_name} not found"}

        engine = TradingEngine(
            strategy,
            initial_capital=request.initial_capital,
            commission_rate=request.commission_rate / 100.0,
            slippage_rate=request.slippage_rate / 100.0
        )

        results = engine.run_backtest(request.symbol, data)

        return BacktestResponse(
            success=True,
            data=results,
            message="Backtest completed successfully"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Backtest failed: {str(e)}"
        )
