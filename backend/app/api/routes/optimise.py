"""
FastAPI endpoints for Portfolio Optimization
"""

import logging
from typing import Any, Hashable

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ...api.deps import check_permission, get_db
from ...core.permissions import Permission
from ...models.user import User
from ...optimise import PortfolioBacktest, PortfolioOptimizer
from ...optimise.optimisation_runner import OptimizationRunner
from ...schemas.optimise import (
    BacktestRequest,
    BacktestResponse,
    BayesianOptimizationRequest,
    BayesianOptimizationResponse,
    BlackLittermanRequest,
    EfficientFrontierRequest,
    OptimizationResponse,
    PortfolioRequest,
    TargetReturnRequest,
    TrialResult,
)

router = APIRouter(prefix="/optimise", tags=["Optimise"])


logger = logging.getLogger(__name__)


@router.post("/sharpe", response_model=OptimizationResponse)
async def optimize_sharpe(request: PortfolioRequest, risk_free_rate: float = Query(0.02, ge=0.0, le=0.1, description="Risk-free rate")):
    """
    Optimize portfolio for maximum Sharpe ratio

    - **symbols**: List of stock ticker symbols
    - **lookback_days**: Historical data period (default: 252 trading days = 1 year)
    - **risk_free_rate**: Risk-free rate for Sharpe calculation (default: 2%)
    """
    try:
        optimizer = PortfolioOptimizer(request.symbols, request.lookback_days)
        await optimizer.fetch_data()
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
        await optimizer.fetch_data()
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
        await optimizer.fetch_data()
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
        await optimizer.fetch_data()
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
        await optimizer.fetch_data()
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
        await optimizer.fetch_data()
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
        await optimizer.fetch_data()
        frontier_df = optimizer.efficient_frontier(request.num_portfolios)

        # Convert DataFrame to list of dicts
        frontier_list: list[dict[Hashable, Any]] = frontier_df.to_dict("records")

        return {"num_portfolios": len(frontier_list), "portfolios": frontier_list}
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
            final_value=results["final_value"],
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
        await optimizer.fetch_data()

        strategies = {
            "max_sharpe": optimizer.optimize_sharpe(),
            "min_volatility": optimizer.optimize_min_volatility(),
            "equal_weight": optimizer.equal_weight_portfolio(),
            "risk_parity": optimizer.risk_parity_portfolio(),
        }

        return {"symbols": optimizer.symbols, "lookback_days": request.lookback_days, "strategies": strategies}
    except Exception as e:
        logger.error(f"Strategy comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bayesian", response_model=BayesianOptimizationResponse)
async def bayesian_optimization(
    request: BayesianOptimizationRequest,
    current_user: User = Depends(check_permission(Permission.BASIC_BACKTEST)),
    db: AsyncSession = Depends(get_db),
):
    """
    Run Bayesian optimization on strategy parameters

    This endpoint:
    1. Creates an isolated database connection pool for optimization
    2. Runs multiple backtest trials with different parameters
    3. Uses Bayesian optimization (TPE) to find optimal parameters
    4. Returns best parameters and all trial results

    Note: This can take several minutes depending on n_trials
    """
    try:
        # Validate request
        if request.n_trials < 1:
            raise HTTPException(status_code=400, detail="n_trials must be at least 1")

        if request.n_trials > 200:
            raise HTTPException(status_code=400, detail="n_trials cannot exceed 200 (performance limit)")

        async with OptimizationRunner(request=request, user_id=current_user.id, max_workers=4) as runner:
            study = await runner.run_optimization()

        trials = []
        for t in study.trials:
            if t.value is not None and t.value != float("-inf"):
                trials.append(TrialResult(trial_id=t.number, params=t.params, value=t.value, status=str(t.state)))

        if not trials:
            raise HTTPException(status_code=500, detail="All optimization trials failed. Check strategy configuration.")

        return BayesianOptimizationResponse(
            best_params=study.best_params,
            best_value=study.best_value,
            trials=trials,
            tickers=request.tickers,
            strategy_key=request.strategy_key,
            metric=request.metric,
            n_completed=len(trials),
            n_failed=request.n_trials - len(trials),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bayesian optimization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)[:200]}")
