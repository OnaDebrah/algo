"""
Backtest routes
"""

import asyncio
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from statsmodels.tsa.stattools import coint

from ...api.deps import check_permission, get_current_active_user, get_current_user, get_db
from ...core import fetch_stock_data
from ...core.permissions import Permission
from ...models import BacktestRun
from ...models.user import User
from ...schemas.backtest import (
    BacktestHistoryItem,
    BacktestRequest,
    BacktestResponse,
    MultiAssetBacktestRequest,
    MultiAssetBacktestResponse,
    OptionsBacktestRequest,
    OptionsBacktestResponse,
    PairsValidationRequest,
    PairsValidationResponse,
    WFARequest,
    WFAResponse,
)
from ...services.auth_service import AuthService
from ...services.backtest_service import BacktestService
from ...services.market_service import get_market_service
from ...services.walk_forward_service import WalkForwardService
from ...utils.helpers import pairs

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/backtest", tags=["Backtest"])


@router.post("/single", response_model=BacktestResponse)
async def run_single_backtest(
    request: BacktestRequest, current_user: User = Depends(check_permission(Permission.BASIC_BACKTEST)), db: AsyncSession = Depends(get_db)
):
    """Run single asset backtest"""
    # Track usage
    await AuthService.track_usage(db, current_user.id, "run_backtest_single", {"symbol": request.symbol})

    service = BacktestService(db)
    result = await service.run_single_backtest(request, current_user.id)
    return result


@router.post("/multi", response_model=MultiAssetBacktestResponse)
async def run_multi_asset_backtest(
    request: MultiAssetBacktestRequest,
    current_user: User = Depends(check_permission(Permission.MULTI_ASSET_BACKTEST)),
    db: AsyncSession = Depends(get_db),
):
    """Run multi-asset backtest"""
    # Track usage
    await AuthService.track_usage(db, current_user.id, "run_backtest_multi", {"symbols": request.symbols})

    service = BacktestService(db)
    result = await service.run_multi_asset_backtest(request, current_user.id)
    return result


@router.post("/options", response_model=OptionsBacktestResponse)
async def run_options_backtest(
    request: OptionsBacktestRequest, current_user: User = Depends(check_permission(Permission.ML_STRATEGIES)), db: AsyncSession = Depends(get_db)
):
    """Run options backtest"""
    # Track usage
    await AuthService.track_usage(db, current_user.id, "run_backtest_options", {"symbol": request.symbol})

    service = BacktestService(db)
    result = await service.run_options_backtest(request, current_user.id)
    return result


@router.get("/history", response_model=List[BacktestHistoryItem])
async def get_backtest_history(
    limit: int = Query(20, ge=1, le=100, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    backtest_type: Optional[str] = Query(None, description="Filter by type: single, multi, options"),
    status: Optional[str] = Query(None, description="Filter by status: pending, running, completed, failed"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get user's backtest history with filtering and pagination

    - **limit**: Maximum number of results (1-100, default: 20)
    - **offset**: Number of results to skip for pagination
    - **backtest_type**: Filter by backtest type (single, multi, options)
    - **status**: Filter by status (pending, running, completed, failed)
    - **symbol**: Filter backtests containing this symbol
    """
    # Build query
    query = select(BacktestRun).filter(BacktestRun.user_id == current_user.id)

    # Apply filters
    if backtest_type:
        query = query.filter(BacktestRun.backtest_type == backtest_type)

    if status:
        query = query.filter(BacktestRun.status == status)

    if symbol:
        # Filter by symbol in JSON array
        query = query.filter(BacktestRun.symbols.contains([symbol]))

    # Order by creation date (newest first)
    query = query.order_by(desc(BacktestRun.created_at))

    # Apply pagination
    query = query.offset(offset).limit(limit)

    # Execute query
    result = await db.execute(query)
    backtest_runs = result.scalars().all()

    # Convert to response schema
    history = [
        BacktestHistoryItem(
            id=run.id,
            name=run.name,
            backtest_type=run.backtest_type,
            symbols=run.symbols,
            strategy_config=run.strategy_config,
            period=run.period,
            interval=run.interval,
            initial_capital=run.initial_capital,
            total_return_pct=run.total_return_pct,
            sharpe_ratio=run.sharpe_ratio,
            max_drawdown=run.max_drawdown,
            win_rate=run.win_rate,
            total_trades=run.total_trades,
            final_equity=run.final_equity,
            status=run.status,
            error_message=run.error_message,
            equity_curve=run.equity_curve,
            trades=run.trades_json,
            created_at=run.created_at.isoformat() if run.created_at else None,
            completed_at=run.completed_at.isoformat() if run.completed_at else None,
        )
        for run in backtest_runs
    ]

    return history


@router.get("/history/count")
async def get_backtest_count(
    backtest_type: Optional[str] = None,
    status: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Get total count of backtests (useful for pagination)"""
    from sqlalchemy import func

    query = select(func.count(BacktestRun.id)).filter(BacktestRun.user_id == current_user.id)

    if backtest_type:
        query = query.filter(BacktestRun.backtest_type == backtest_type)

    if status:
        query = query.filter(BacktestRun.status == status)

    result = await db.execute(query)
    count = result.scalar()

    return {"count": count}


@router.get("/history/{backtest_id}", response_model=BacktestHistoryItem)
async def get_backtest_details(backtest_id: int, current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)):
    """Get specific backtest run details"""
    result = await db.execute(select(BacktestRun).filter(BacktestRun.id == backtest_id, BacktestRun.user_id == current_user.id))
    backtest = result.scalar_one_or_none()

    if not backtest:
        raise HTTPException(status_code=404, detail="Backtest not found")

    return BacktestHistoryItem(
        id=backtest.id,
        name=backtest.name,
        backtest_type=backtest.backtest_type,
        symbols=backtest.symbols,
        strategy_config=backtest.strategy_config,
        period=backtest.period,
        interval=backtest.interval,
        initial_capital=backtest.initial_capital,
        total_return_pct=backtest.total_return_pct,
        sharpe_ratio=backtest.sharpe_ratio,
        max_drawdown=backtest.max_drawdown,
        win_rate=backtest.win_rate,
        total_trades=backtest.total_trades,
        final_equity=backtest.final_equity,
        status=backtest.status,
        error_message=backtest.error_message,
        equity_curve=backtest.equity_curve,
        trades=backtest.trades_json,
        created_at=backtest.created_at.isoformat() if backtest.created_at else None,
        completed_at=backtest.completed_at.isoformat() if backtest.completed_at else None,
    )


@router.put("/history/{backtest_id}/name")
async def update_backtest_name(
    backtest_id: int,
    name: str = Query(..., min_length=1, max_length=255),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Update backtest name"""
    result = await db.execute(select(BacktestRun).filter(BacktestRun.id == backtest_id, BacktestRun.user_id == current_user.id))
    backtest = result.scalar_one_or_none()

    if not backtest:
        raise HTTPException(status_code=404, detail="Backtest not found")

    backtest.name = name
    await db.commit()

    return {"message": "Backtest name updated", "name": name}


@router.delete("/history/{backtest_id}")
async def delete_backtest(backtest_id: int, current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)):
    """Delete a backtest run"""
    result = await db.execute(select(BacktestRun).filter(BacktestRun.id == backtest_id, BacktestRun.user_id == current_user.id))
    backtest = result.scalar_one_or_none()

    if not backtest:
        raise HTTPException(status_code=404, detail="Backtest not found")

    await db.delete(backtest)
    await db.commit()

    return {"message": "Backtest deleted successfully"}


@router.get("/history/stats/summary")
async def get_backtest_stats(current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)):
    """Get summary statistics of all user backtests"""
    from sqlalchemy import func

    # Total count by status
    status_query = (
        select(BacktestRun.status, func.count(BacktestRun.id).label("count"))
        .filter(BacktestRun.user_id == current_user.id)
        .group_by(BacktestRun.status)
    )

    status_result = await db.execute(status_query)
    status_counts = {row[0]: row[1] for row in status_result}

    # Count by type
    type_query = (
        select(BacktestRun.backtest_type, func.count(BacktestRun.id).label("count"))
        .filter(BacktestRun.user_id == current_user.id)
        .group_by(BacktestRun.backtest_type)
    )

    type_result = await db.execute(type_query)
    type_counts = {row[0]: row[1] for row in type_result}

    # Average metrics for completed backtests
    metrics_query = select(
        func.avg(BacktestRun.total_return_pct).label("avg_return"),
        func.avg(BacktestRun.sharpe_ratio).label("avg_sharpe"),
        func.avg(BacktestRun.max_drawdown).label("avg_drawdown"),
        func.avg(BacktestRun.win_rate).label("avg_win_rate"),
        func.sum(BacktestRun.total_trades).label("total_trades"),
    ).filter(BacktestRun.user_id == current_user.id, BacktestRun.status == "completed")

    metrics_result = await db.execute(metrics_query)
    metrics = metrics_result.one()

    return {
        "total_backtests": sum(status_counts.values()),
        "by_status": status_counts,
        "by_type": type_counts,
        "completed_metrics": {
            "avg_return_pct": float(metrics[0]) if metrics[0] else 0,
            "avg_sharpe_ratio": float(metrics[1]) if metrics[1] else 0,
            "avg_max_drawdown": float(metrics[2]) if metrics[2] else 0,
            "avg_win_rate": float(metrics[3]) if metrics[3] else 0,
            "total_trades": int(metrics[4]) if metrics[4] else 0,
        },
    }


@router.post("/validated", response_model=PairsValidationResponse)
async def validate_pairs(request: PairsValidationRequest):
    """
    Validate a pairs trading pair

    Checks:
    1. Sector matching
    2. Historical correlation
    3. Cointegration (Engle-Granger test)
    """
    try:
        # Fetch historical data
        asset1 = await asyncio.to_thread(fetch_stock_data, request.asset_1, period=request.period, interval=request.interval)
        asset2 = await asyncio.to_thread(fetch_stock_data, request.asset_2, period=request.period, interval=request.interval)

        if asset1.empty or asset2.empty:
            raise HTTPException(status_code=400, detail="Could not fetch data for one or both symbols")
        # Align data
        common_dates = asset1.index.intersection(asset2.index)
        if len(common_dates) < 30:
            raise HTTPException(status_code=400, detail="Insufficient common data points")

        prices_1 = asset1.loc[common_dates, "Close"]
        prices_2 = asset2.loc[common_dates, "Close"]

        # Get sectors
        service = get_market_service()
        sector_1 = await service.get_sector(request.asset_1)
        sector_2 = await service.get_sector(request.asset_2)

        # Calculate correlation
        correlation: float = prices_1.corr(prices_2)

        # Cointegration test (Engle-Granger)
        coint_stat, coint_pvalue, _ = await asyncio.to_thread(coint, prices_1, prices_2)

        # Validation logic
        warnings = []
        errors = []

        # Check sector
        if sector_1 != sector_2:
            warnings.append(f"Different sectors: {sector_1} vs {sector_2}")

        # Check correlation
        if correlation < 0.5:
            errors.append(f"Very low correlation: {correlation:.3f} (< 0.5)")
        elif correlation < 0.7:
            warnings.append(f"Moderate correlation: {correlation:.3f} (recommended > 0.7)")

        # Check cointegration
        if coint_pvalue > 0.1:
            errors.append(f"Not cointegrated: p-value {coint_pvalue:.4f} (> 0.1)")
        elif coint_pvalue > 0.05:
            warnings.append(f"Weak cointegration: p-value {coint_pvalue:.4f} (recommended < 0.05)")

        is_valid = len(errors) == 0

        return PairsValidationResponse(
            asset_1=request.asset_1,
            asset_2=request.asset_2,
            sector_1=sector_1,
            sector_2=sector_2,
            correlation=float(correlation),
            cointegration_pvalue=float(coint_pvalue),
            cointegration_statistic=float(coint_stat),
            is_valid=is_valid,
            warnings=warnings,
            errors=errors,
            lookback_days=len(common_dates),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.get("/suggested")
async def get_suggested_pairs():
    """
    Get list of pre-validated pairs
    """
    return {"pairs": pairs}


@router.post("/walk-forward", response_model=WFAResponse)
async def walk_forward(request: WFARequest, db: AsyncSession = Depends(get_db), current_user=Depends(get_current_user)):
    """
    Run Walk-Forward Analysis on a strategy.

    This performs iterative Optimization (In-Sample) and Validation (Out-of-Sample)
    to verify strategy robustness and detect overfitting.
    """
    try:
        service = WalkForwardService(db, current_user.id)
        result = await service.run_wfa(request)
        return result
    except ValueError as e:
        logger.error(f"WFA validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"WFA execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during Walk-Forward Analysis")
