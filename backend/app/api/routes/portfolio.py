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
from backend.app.services.auth_service import AuthService
from backend.app.services.portfolio_service import PortfolioService

router = APIRouter(prefix="/portfolio", tags=["Portfolio"])


@router.get("/", response_model=List[Portfolio])
async def get_portfolios(current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)):
    """Get user's portfolios"""
    await AuthService.track_usage(db, current_user.id, "get_portfolios")
    service = PortfolioService(db)
    portfolios = await service.get_portfolios(current_user.id)
    return portfolios


@router.post("/", response_model=Portfolio)
async def create_portfolio(
    portfolio_data: PortfolioCreate, current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)
):
    """Create new portfolio"""
    await AuthService.track_usage(db, current_user.id, "create_portfolios", {"portfolio_name": portfolio_data.name})
    service = PortfolioService(db)
    portfolio = await service.create_portfolio(current_user.id, portfolio_data)
    return portfolio


@router.get("/{portfolio_id}", response_model=Portfolio)
async def get_portfolio(portfolio_id: int, current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)):
    """Get portfolio by ID"""
    await AuthService.track_usage(db, current_user.id, "get_portfolio", {"portfolio_key": portfolio_id})
    service = PortfolioService(db)
    portfolio = await service.get_portfolio(portfolio_id, current_user.id)

    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    return portfolio


@router.put("/{portfolio_id}", response_model=Portfolio)
async def update_portfolio(
    portfolio_id: int, update_data: PortfolioUpdate, current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)
):
    """Update portfolio"""
    await AuthService.track_usage(db, current_user.id, "get_portfolios", {"portfolio_key": portfolio_id})

    service = PortfolioService(db)
    portfolio = await service.update_portfolio(portfolio_id, current_user.id, update_data)

    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    return portfolio


@router.delete("/{portfolio_id}")
async def delete_portfolio(portfolio_id: int, current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)):
    """Delete portfolio"""
    await AuthService.track_usage(db, current_user.id, "delete_portfolio", {"portfolio_key": portfolio_id})

    service = PortfolioService(db)
    success = await service.delete_portfolio(portfolio_id, current_user.id)

    if not success:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    return {"message": "Portfolio deleted successfully"}


@router.get("/{portfolio_id}/metrics", response_model=PortfolioMetrics)
async def get_portfolio_metrics(portfolio_id: int, current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)):
    """Get portfolio metrics"""
    await AuthService.track_usage(db, current_user.id, "get_portfolio_metrics", {"portfolio_key": portfolio_id})
    service = PortfolioService(db)
    success = await service.get_portfolio_metrics(portfolio_id, current_user.id)

    if not success:
        raise HTTPException(status_code=404, detail="Portfolio metrics not found")


@router.get("/{portfolio_id}/positions", response_model=List[Position])
async def get_positions(portfolio_id: int, current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)):
    """Get portfolio positions"""
    await AuthService.track_usage(db, current_user.id, "get_positions", {"portfolio_key": portfolio_id})

    service = PortfolioService(db)
    positions = await service.get_positions(portfolio_id)
    return positions


@router.get("/{portfolio_id}/trades", response_model=List[Trade])
async def get_trades(
    portfolio_id: int, limit: int = 100, offset: int = 0, current_user: User = Depends(get_current_active_user), db: AsyncSession = Depends(get_db)
):
    """Get portfolio trade history"""
    await AuthService.track_usage(db, current_user.id, "get_portfolios", {"portfolio_key": portfolio_id})

    service = PortfolioService(db)
    trades = await service.get_trades(portfolio_id, limit=limit, offset=offset)
    return trades
