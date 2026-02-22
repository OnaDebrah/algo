"""
Analytics routes
"""

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import get_current_active_user
from backend.app.database import get_db
from backend.app.models.user import User
from backend.app.services.trading_service import TradingService

router = APIRouter(prefix="/analytics", tags=["Analytics"])

trading_service = TradingService


@router.get("/performance/{portfolio_id}")
async def get_performance_analytics(
    portfolio_id: int,
    period: str = Query("1M", regex="^(1D|1W|1M|3M|6M|1Y|ALL)$"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Get performance analytics for a portfolio"""
    # Calculate date range
    end_date = datetime.now()
    if period == "1D":
        start_date = end_date - timedelta(days=1)
    elif period == "1W":
        start_date = end_date - timedelta(weeks=1)
    elif period == "1M":
        start_date = end_date - timedelta(days=30)
    elif period == "3M":
        start_date = end_date - timedelta(days=90)
    elif period == "6M":
        start_date = end_date - timedelta(days=180)
    elif period == "1Y":
        start_date = end_date - timedelta(days=365)
    else:  # ALL
        start_date = None

    equity_curve = await trading_service.get_equity_curve(db, portfolio_id, start_date, end_date)

    from backend.app.analytics.performance import calculate_performance_metrics

    if equity_curve:
        initial_equity = equity_curve[0]["equity"]
        final_equity = equity_curve[-1]["equity"]
        total_return = final_equity - initial_equity
        total_return_pct = (total_return / initial_equity) * 100

        trades = await trading_service.get_trades(db, portfolio_id, start_date=start_date, end_date=end_date)
        trades_dicts = [trade.to_dict() for trade in trades]

        metrics = calculate_performance_metrics(trades_dicts, equity_curve, initial_equity)

        return {
            "period": period,
            "metrics": metrics,
            "equity_curve": equity_curve,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
        }

    return {"period": period, "metrics": {}, "equity_curve": []}


@router.get("/returns/{portfolio_id}")
async def get_returns_analysis(portfolio_id: int, current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)):
    """Get returns analysis"""

    trades = await trading_service.get_trades(db, portfolio_id)

    if not trades:
        return {"daily_returns": [], "monthly_returns": [], "cumulative_returns": []}

    from backend.app.analytics.performance import calculate_returns

    returns = calculate_returns(trades)

    return returns


@router.get("/risk/{portfolio_id}")
async def get_risk_metrics(portfolio_id: int, current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)):
    """Get risk metrics"""
    trades = await trading_service.get_trades(db, portfolio_id)
    equity_curve = await trading_service.get_equity_curve(db, portfolio_id)

    if not trades or not equity_curve:
        return {"volatility": 0, "beta": 0, "var_95": 0, "cvar_95": 0, "max_drawdown": 0, "sharpe_ratio": 0, "sortino_ratio": 0}

    from backend.app.analytics.performance import calculate_risk_metrics

    metrics = calculate_risk_metrics(trades, equity_curve)

    return metrics


@router.get("/drawdown/{portfolio_id}")
async def get_drawdown_analysis(portfolio_id: int, current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)):
    """Get drawdown analysis"""
    equity_curve = await trading_service.get_equity_curve(db, portfolio_id)

    if not equity_curve:
        return {"drawdowns": [], "max_drawdown": 0, "max_drawdown_duration": 0}

    from backend.app.analytics.performance import calculate_max_drawdown

    analysis = calculate_max_drawdown(equity_curve)

    return analysis
