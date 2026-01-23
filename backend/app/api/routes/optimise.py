"""
FastAPI endpoints for Portfolio Optimization
"""

import logging
from typing import Any, Hashable

from fastapi import HTTPException, Query, APIRouter

from backend.app.optimise import PortfolioOptimizer, PortfolioBacktest
from backend.app.schemas.optimise import PortfolioRequest, OptimizationResponse, TargetReturnRequest, BlackLittermanRequest, \
    EfficientFrontierRequest, BacktestRequest, BacktestResponse

router = APIRouter(prefix="/optimise", tags=["Optimise"])


logger = logging.getLogger(__name__)


@router.post("/sharpe", response_model=OptimizationResponse)
async def optimize_sharpe(
        request: PortfolioRequest,
        risk_free_rate: float = Query(0.02, ge=0.0, le=0.1, description="Risk-free rate")
):
    """
    Optimize portfolio for maximum Sharpe ratio

    - **symbols**: List of stock ticker symbols
    - **lookback_days**: Historical data period (default: 252 trading days = 1 year)
    - **risk_free_rate**: Risk-free rate for Sharpe calculation (default: 2%)
    """
    try:
        optimizer = PortfolioOptimizer(request.symbols, request.lookback_days)
        optimizer.fetch_data()
        result = optimizer.optimize_sharpe(risk_free_rate)
        return OptimizationResponse(**result)
    except Exception as e:
        logger.error(f"Sharpe optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/min-volatility", response_model=OptimizationResponse)
async def optimize_min_volatility(request: PortfolioRequest):
    """
    Optimize portfolio for minimum volatility

    - **symbols**: List of stock ticker symbols
    - **lookback_days**: Historical data period
    """
    try:
        optimizer = PortfolioOptimizer(request.symbols, request.lookback_days)
        optimizer.fetch_data()
        result = optimizer.optimize_min_volatility()
        return OptimizationResponse(**result)
    except Exception as e:
        logger.error(f"Min volatility optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/target-return", response_model=OptimizationResponse)
async def optimize_target_return(request: TargetReturnRequest):
    """
    Optimize portfolio to achieve target return with minimum volatility

    - **symbols**: List of stock ticker symbols
    - **target_return**: Target annual return (e.g., 0.15 for 15%)
    - **lookback_days**: Historical data period
    """
    try:
        optimizer = PortfolioOptimizer(request.symbols, request.lookback_days)
        optimizer.fetch_data()
        result = optimizer.optimize_target_return(request.target_return)
        return OptimizationResponse(**result)
    except Exception as e:
        logger.error(f"Target return optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/equal-weight", response_model=OptimizationResponse)
async def equal_weight_portfolio(request: PortfolioRequest):
    """
    Create equal-weighted portfolio

    - **symbols**: List of stock ticker symbols
    - **lookback_days**: Historical data period
    """
    try:
        optimizer = PortfolioOptimizer(request.symbols, request.lookback_days)
        optimizer.fetch_data()
        result = optimizer.equal_weight_portfolio()
        return OptimizationResponse(**result)
    except Exception as e:
        logger.error(f"Equal weight portfolio failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/risk-parity", response_model=OptimizationResponse)
async def risk_parity_portfolio(request: PortfolioRequest):
    """
    Create risk parity portfolio (equal risk contribution from each asset)

    - **symbols**: List of stock ticker symbols
    - **lookback_days**: Historical data period
    """
    try:
        optimizer = PortfolioOptimizer(request.symbols, request.lookback_days)
        optimizer.fetch_data()
        result = optimizer.risk_parity_portfolio()
        return OptimizationResponse(**result)
    except Exception as e:
        logger.error(f"Risk parity optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/black-litterman", response_model=OptimizationResponse)
async def black_litterman_optimization(request: BlackLittermanRequest):
    """
    Black-Litterman optimization with investor views

    - **symbols**: List of stock ticker symbols
    - **views**: Dictionary of expected returns for specific symbols
    - **confidence**: Confidence level in views (0.0 to 1.0)
    - **lookback_days**: Historical data period
    """
    try:
        optimizer = PortfolioOptimizer(request.symbols, request.lookback_days)
        optimizer.fetch_data()
        result = optimizer.black_litterman(request.views, request.confidence)
        return OptimizationResponse(**result)
    except Exception as e:
        logger.error(f"Black-Litterman optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/efficient-frontier")
async def generate_efficient_frontier(request: EfficientFrontierRequest):
    """
    Generate efficient frontier portfolios

    - **symbols**: List of stock ticker symbols
    - **num_portfolios**: Number of portfolios on the frontier
    - **lookback_days**: Historical data period

    Returns a list of portfolios with varying risk/return profiles
    """
    try:
        optimizer = PortfolioOptimizer(request.symbols, request.lookback_days)
        optimizer.fetch_data()
        frontier_df = optimizer.efficient_frontier(request.num_portfolios)

        # Convert DataFrame to list of dicts
        frontier_list: list[dict[Hashable, Any]] = frontier_df.to_dict('records')

        return {
            "num_portfolios": len(frontier_list),
            "portfolios": frontier_list
        }
    except Exception as e:
        logger.error(f"Efficient frontier generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    """
    Run portfolio backtest

    - **symbols**: List of stock ticker symbols
    - **weights**: Portfolio weights (must sum to 1.0)
    - **start_capital**: Starting capital amount
    - **period**: Backtest period (e.g., '1y', '6mo', '2y')
    """
    try:
        backtester = PortfolioBacktest(request.symbols, request.weights)
        results = backtester.run_backtest(request.start_capital, request.period)

        # Remove equity curve and returns from response (too large for API)
        return BacktestResponse(
            total_return=results["total_return"],
            volatility=results["volatility"],
            sharpe_ratio=results["sharpe_ratio"],
            max_drawdown=results["max_drawdown"],
            final_value=results["final_value"]
        )
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare-strategies")
async def compare_strategies(request: PortfolioRequest):
    """
    Compare multiple portfolio optimization strategies

    - **symbols**: List of stock ticker symbols
    - **lookback_days**: Historical data period

    Returns results for all optimization methods for easy comparison
    """
    try:
        optimizer = PortfolioOptimizer(request.symbols, request.lookback_days)
        optimizer.fetch_data()

        strategies = {
            "max_sharpe": optimizer.optimize_sharpe(),
            "min_volatility": optimizer.optimize_min_volatility(),
            "equal_weight": optimizer.equal_weight_portfolio(),
            "risk_parity": optimizer.risk_parity_portfolio(),
        }

        return {
            "symbols": optimizer.symbols,
            "lookback_days": request.lookback_days,
            "strategies": strategies
        }
    except Exception as e:
        logger.error(f"Strategy comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))