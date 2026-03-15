"""
Pre-live portfolio optimization preview API.

Provides an intermediate step between multi-asset backtest results
and live deployment, letting users compare 6 optimization methods
and choose weights before going live.
"""

import logging
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ...api.deps import check_permission, get_db
from ...core.permissions import Permission
from ...core.portfolio_optimizer import PortfolioOptimizer
from ...models.user import User
from ...schemas.deploy_optimize import (
    OptimizationResult,
    OptimizeApplyRequest,
    OptimizePreviewRequest,
    OptimizePreviewResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/deploy", tags=["Deploy Optimization"])


def _run_optimizer_method(optimizer: PortfolioOptimizer, method: str) -> Dict:
    """Call a single optimizer method and return the raw dict."""
    dispatch = {
        "max_sharpe": lambda: optimizer.optimize_sharpe(),
        "min_volatility": lambda: optimizer.optimize_min_volatility(),
        "risk_parity": lambda: optimizer.risk_parity_portfolio(),
        "equal_weight": lambda: optimizer.equal_weight_portfolio(),
        "black_litterman": lambda: optimizer.black_litterman(views={}),
        "target_return": lambda: optimizer.optimize_target_return(target_return=0.15),
    }
    fn = dispatch.get(method)
    if fn is None:
        raise ValueError(f"Unknown method: {method}")
    return fn()


def _dict_to_result(raw: Dict) -> OptimizationResult:
    """Convert raw optimizer output to schema."""
    return OptimizationResult(
        weights=raw.get("weights", {}),
        expected_return=raw.get("expected_return", 0.0),
        volatility=raw.get("volatility", 0.0),
        sharpe=raw.get("sharpe_ratio", 0.0),
        method=raw.get("method", ""),
    )


@router.post("/optimize-preview", response_model=OptimizePreviewResponse)
async def optimize_preview(
    request: OptimizePreviewRequest,
    current_user: User = Depends(check_permission(Permission.LIVE_TRADING)),
    db: AsyncSession = Depends(get_db),
):
    """
    Given a list of symbols, compute optimized weights using every
    available method and return them side-by-side for comparison.
    """
    if len(request.symbols) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 symbols for portfolio optimization")

    try:
        optimizer = PortfolioOptimizer(symbols=request.symbols, lookback_days=request.lookback_days)
    except Exception as e:
        logger.error(f"PortfolioOptimizer init failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Could not initialize optimizer: {str(e)}")

    methods_to_run = ["max_sharpe", "min_volatility", "risk_parity", "equal_weight", "black_litterman", "target_return"]
    results: Dict[str, OptimizationResult] = {}

    for method in methods_to_run:
        try:
            raw = _run_optimizer_method(optimizer, method)
            results[method] = _dict_to_result(raw)
        except Exception as e:
            logger.warning(f"Optimization method '{method}' failed: {e}")
            # Skip failed methods instead of blocking the whole preview
            continue

    if not results:
        raise HTTPException(status_code=500, detail="All optimization methods failed")

    # Equal weight is always the baseline
    eq = results.get("equal_weight")
    if eq is None:
        # Manually build equal-weight baseline
        n = len(request.symbols)
        eq = OptimizationResult(
            weights={s: 1.0 / n for s in request.symbols},
            expected_return=0.0,
            volatility=0.0,
            sharpe=0.0,
            method="equal_weight",
        )

    return OptimizePreviewResponse(methods=results, symbols=request.symbols, equal_weight_baseline=eq)


@router.post("/optimize-apply")
async def optimize_apply(
    request: OptimizeApplyRequest,
    current_user: User = Depends(check_permission(Permission.LIVE_TRADING)),
    db: AsyncSession = Depends(get_db),
):
    """
    Return the final optimized weights for the chosen method,
    ready to be forwarded to the live deployment endpoint.
    """
    if len(request.symbols) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 symbols for portfolio optimization")

    try:
        optimizer = PortfolioOptimizer(symbols=request.symbols, lookback_days=request.lookback_days)
        raw = _run_optimizer_method(optimizer, request.method)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"optimize-apply failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

    return {
        "method": request.method,
        "weights": raw.get("weights", {}),
        "expected_return": raw.get("expected_return", 0.0),
        "volatility": raw.get("volatility", 0.0),
        "sharpe_ratio": raw.get("sharpe_ratio", 0.0),
    }
