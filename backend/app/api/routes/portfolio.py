"""
Updated Portfolio routes with full CRUD
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import get_current_active_user
from backend.app.database import get_db
from backend.app.models.user import User
from backend.app.schemas.portfolio import Portfolio, PortfolioCreate, PortfolioMetrics, PortfolioUpdate, Position, Trade
from backend.app.services.portfolio_service import PortfolioService

router = APIRouter(prefix="/portfolio", tags=["Portfolio"])


@router.get("/", response_model=List[Portfolio])
async def get_portfolios(current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)):
    """Get user's portfolios"""
    service = PortfolioService(db)
    portfolios = service.get_portfolios(current_user.id)
    return portfolios


@router.post("/", response_model=Portfolio)
async def create_portfolio(
    portfolio_data: PortfolioCreate, current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)
):
    """Create new portfolio"""
    service = PortfolioService(db)
    portfolio = service.create_portfolio(current_user.id, portfolio_data)
    return portfolio


@router.get("/{portfolio_id}", response_model=Portfolio)
async def get_portfolio(portfolio_id: int, current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)):
    """Get portfolio by ID"""
    service = PortfolioService(db)
    portfolio = service.get_portfolio(portfolio_id, current_user.id)

    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    return portfolio


@router.put("/{portfolio_id}", response_model=Portfolio)
async def update_portfolio(
    portfolio_id: int, update_data: PortfolioUpdate, current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)
):
    """Update portfolio"""
    service = PortfolioService(db)
    portfolio = service.update_portfolio(portfolio_id, current_user.id, update_data)

    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    return portfolio


@router.delete("/{portfolio_id}")
async def delete_portfolio(portfolio_id: int, current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)):
    """Delete portfolio"""
    service = PortfolioService(db)
    success = service.delete_portfolio(portfolio_id, current_user.id)

    if not success:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    return {"message": "Portfolio deleted successfully"}


@router.get("/{portfolio_id}/metrics", response_model=PortfolioMetrics)
async def get_portfolio_metrics(portfolio_id: int, current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)):
    """Get portfolio metrics"""
    from backend.app.models.portfolio import Portfolio
    from backend.app.models.position import Position

    portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id, Portfolio.user_id == current_user.id).first()

    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    # Calculate metrics
    positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()

    exposure = sum(pos.quantity * (pos.current_price or pos.avg_entry_price) for pos in positions)

    unrealized_pnl = sum(pos.unrealized_pnl or 0 for pos in positions)

    nav = portfolio.current_capital + exposure
    prev_nav = portfolio.initial_capital
    cash = portfolio.current_capital

    return PortfolioMetrics(
        nav=nav,
        prev_nav=prev_nav,
        exposure=exposure,
        unrealized_pnl=unrealized_pnl,
        cash=cash,
        total_value=nav,
        daily_return=nav - prev_nav,
        daily_return_pct=((nav - prev_nav) / prev_nav * 100) if prev_nav > 0 else 0,
    )


@router.get("/{portfolio_id}/positions", response_model=List[Position])
async def get_positions(portfolio_id: int, current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)):
    """Get portfolio positions"""
    service = PortfolioService(db)
    positions = service.get_positions(portfolio_id)
    return positions


@router.get("/{portfolio_id}/trades", response_model=List[Trade])
async def get_trades(
    portfolio_id: int, limit: int = 100, offset: int = 0, current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)
):
    """Get portfolio trade history"""
    service = PortfolioService(db)
    trades = service.get_trades(portfolio_id, limit=limit, offset=offset)
    return trades
