"""
Paper trading routes — virtual portfolio with manual + strategy-driven trading.
"""

import json

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ...api.deps import get_current_active_user
from ...database import get_db
from ...models.user import User
from ...schemas.paper_trading import (
    PaperAttachStrategy,
    PaperEquitySnapshotOut,
    PaperPerformanceOut,
    PaperPortfolioCreate,
    PaperPortfolioOut,
    PaperTradeOut,
    PaperTradeRequest,
    StrategySignalResult,
)
from ...services.paper_trading_service import PaperTradingService, is_market_open
from ...strategies.catelog.strategy_catalog import get_catalog

router = APIRouter(prefix="/paper", tags=["Paper Trading"])
service = PaperTradingService()


@router.get("/market-status")
async def market_status():
    """Check if US stock market is currently open."""
    from datetime import datetime

    import pytz

    et = pytz.timezone("US/Eastern")
    now_et = datetime.now(et)
    return {
        "market_open": is_market_open(),
        "current_time_et": now_et.strftime("%Y-%m-%d %H:%M:%S ET"),
        "next_action": "Market is open — live prices available" if is_market_open() else "Market is closed — prices reflect last close",
    }


@router.get("/strategies")
async def list_available_strategies(
    current_user: User = Depends(get_current_active_user),
):
    """List all strategies available for paper trading."""
    catalog = get_catalog()
    strategies = []
    for key, info in catalog.strategies.items():
        strategies.append(
            {
                "key": key,
                "name": info.name,
                "category": info.category,
                "description": getattr(info, "description", ""),
                "parameters": {
                    pname: {
                        "default": pinfo.get("default"),
                        "type": pinfo.get("type", "float"),
                        "description": pinfo.get("description", ""),
                    }
                    for pname, pinfo in info.parameters.items()
                }
                if info.parameters
                else {},
            }
        )
    return strategies


@router.post("/portfolios", response_model=PaperPortfolioOut)
async def create_portfolio(
    req: PaperPortfolioCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new paper trading portfolio."""
    portfolio = await service.create_portfolio(
        db,
        current_user.id,
        req.name,
        req.initial_cash,
        strategy_key=req.strategy_key,
        strategy_params=req.strategy_params,
        strategy_symbol=req.strategy_symbol,
        trade_quantity=req.trade_quantity,
        data_interval=req.data_interval or "1d",
    )
    return _portfolio_response(portfolio)


@router.get("/portfolios", response_model=list[PaperPortfolioOut])
async def list_portfolios(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """List all active paper portfolios."""
    portfolios = await service.get_portfolios(db, current_user.id)
    return [_portfolio_response(p) for p in portfolios]


@router.get("/portfolios/{portfolio_id}", response_model=PaperPortfolioOut)
async def get_portfolio(
    portfolio_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Get portfolio with positions enriched with live prices."""
    portfolio = await service.get_portfolio(db, portfolio_id, current_user.id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    positions, positions_value = await service.enrich_positions(portfolio)
    equity = portfolio.current_cash + positions_value

    strat_params = None
    if portfolio.strategy_params:
        try:
            strat_params = json.loads(portfolio.strategy_params)
        except (json.JSONDecodeError, TypeError):
            strat_params = None

    return {
        "id": portfolio.id,
        "name": portfolio.name,
        "initial_cash": portfolio.initial_cash,
        "current_cash": round(portfolio.current_cash, 2),
        "is_active": portfolio.is_active,
        "strategy_key": portfolio.strategy_key,
        "strategy_symbol": portfolio.strategy_symbol,
        "strategy_params": strat_params,
        "trade_quantity": portfolio.trade_quantity,
        "data_interval": portfolio.data_interval or "1d",
        "created_at": portfolio.created_at,
        "updated_at": portfolio.updated_at,
        "positions": positions,
        "equity": round(equity, 2),
        "total_return_pct": round((equity - portfolio.initial_cash) / portfolio.initial_cash * 100, 2) if portfolio.initial_cash > 0 else 0,
    }


@router.post("/portfolios/{portfolio_id}/trade", response_model=PaperTradeOut)
async def place_trade(
    portfolio_id: int,
    req: PaperTradeRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Place a manual paper trade."""
    try:
        trade = await service.place_trade(db, portfolio_id, current_user.id, req.symbol, req.side, req.quantity)
        return trade
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/portfolios/{portfolio_id}/trades", response_model=list[PaperTradeOut])
async def get_trades(
    portfolio_id: int,
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Get trade history for a portfolio."""
    try:
        trades = await service.get_trades(db, portfolio_id, current_user.id, limit)
        return trades
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/portfolios/{portfolio_id}/performance", response_model=PaperPerformanceOut)
async def get_performance(
    portfolio_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Get performance metrics for a portfolio."""
    try:
        return await service.get_performance(db, portfolio_id, current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/portfolios/{portfolio_id}/equity", response_model=list[PaperEquitySnapshotOut])
async def get_equity_history(
    portfolio_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Get equity curve data."""
    try:
        return await service.get_equity_history(db, portfolio_id, current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/portfolios/{portfolio_id}/attach-strategy", response_model=PaperPortfolioOut)
async def attach_strategy(
    portfolio_id: int,
    req: PaperAttachStrategy,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Attach a strategy to a paper portfolio for signal-driven trading."""
    try:
        portfolio = await service.attach_strategy(
            db,
            portfolio_id,
            current_user.id,
            req.strategy_key,
            req.strategy_symbol,
            req.strategy_params,
            req.trade_quantity,
            data_interval=req.data_interval or "1d",
        )
        return _portfolio_response(portfolio)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/portfolios/{portfolio_id}/detach-strategy", response_model=PaperPortfolioOut)
async def detach_strategy(
    portfolio_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Remove strategy from a portfolio."""
    try:
        portfolio = await service.detach_strategy(db, portfolio_id, current_user.id)
        return _portfolio_response(portfolio)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/portfolios/{portfolio_id}/run-signal", response_model=StrategySignalResult)
async def run_strategy_signal(
    portfolio_id: int,
    auto_execute: bool = Query(True, description="Automatically place trade if signal is actionable"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Run the attached strategy on current data and optionally execute the trade."""
    try:
        return await service.run_strategy_signal(db, portfolio_id, current_user.id, auto_execute)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/portfolios/{portfolio_id}/refresh-equity", response_model=PaperEquitySnapshotOut)
async def refresh_equity(
    portfolio_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Take a mark-to-market equity snapshot with live prices (no trade needed)."""
    try:
        return await service.refresh_equity(db, portfolio_id, current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/portfolios/{portfolio_id}")
async def deactivate_portfolio(
    portfolio_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Deactivate a paper portfolio."""
    deleted = await service.deactivate_portfolio(db, portfolio_id, current_user.id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return {"status": "deactivated"}


def _portfolio_response(portfolio) -> dict:
    """Convert a portfolio ORM object to a response dict."""
    positions_value = sum((p.current_price or p.avg_entry_price) * p.quantity for p in portfolio.positions)
    equity = portfolio.current_cash + positions_value

    # Parse strategy_params JSON string
    strat_params = None
    if portfolio.strategy_params:
        try:
            strat_params = json.loads(portfolio.strategy_params)
        except (json.JSONDecodeError, TypeError):
            strat_params = None

    return {
        "id": portfolio.id,
        "name": portfolio.name,
        "initial_cash": portfolio.initial_cash,
        "current_cash": round(portfolio.current_cash, 2),
        "is_active": portfolio.is_active,
        "strategy_key": portfolio.strategy_key,
        "strategy_symbol": portfolio.strategy_symbol,
        "strategy_params": strat_params,
        "trade_quantity": portfolio.trade_quantity,
        "data_interval": portfolio.data_interval or "1d",
        "created_at": portfolio.created_at,
        "updated_at": portfolio.updated_at,
        "positions": [
            {
                "id": p.id,
                "symbol": p.symbol,
                "quantity": p.quantity,
                "avg_entry_price": round(p.avg_entry_price, 2),
                "current_price": round(p.current_price, 2) if p.current_price else None,
                "unrealized_pnl": round((p.current_price - p.avg_entry_price) * p.quantity, 2) if p.current_price else None,
                "unrealized_pnl_pct": round((p.current_price - p.avg_entry_price) / p.avg_entry_price * 100, 2)
                if p.current_price and p.avg_entry_price > 0
                else None,
                "market_value": round(p.current_price * p.quantity, 2) if p.current_price else None,
                "opened_at": p.opened_at,
            }
            for p in portfolio.positions
        ],
        "equity": round(equity, 2),
        "total_return_pct": round((equity - portfolio.initial_cash) / portfolio.initial_cash * 100, 2) if portfolio.initial_cash > 0 else 0,
    }
