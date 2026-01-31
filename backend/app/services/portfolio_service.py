"""
Portfolio service with full CRUD operations
"""

from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models import Portfolio
from backend.app.models.position import Position
from backend.app.models.trade import Trade
from backend.app.schemas import PortfolioMetrics
from backend.app.schemas.portfolio import (
    Portfolio as PortfolioSchema,
    PortfolioCreate,
    PortfolioUpdate,
    Position as PositionSchema,
    Trade as TradeSchema,
)


class PortfolioService:
    """Service for managing portfolios"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_portfolio(self, user_id: int, portfolio_data: PortfolioCreate) -> PortfolioSchema:
        """Create a new portfolio"""
        portfolio = Portfolio(
            user_id=user_id,
            name=portfolio_data.name,
            description=portfolio_data.description,
            initial_capital=portfolio_data.initial_capital,
            current_capital=portfolio_data.initial_capital,
        )

        self.db.add(portfolio)
        await self.db.commit()
        await self.db.refresh(portfolio)

        return PortfolioSchema(**portfolio.to_dict())

    async def get_portfolios(self, user_id: int) -> List[PortfolioSchema]:
        """Get all portfolios for a user"""
        stmt = select(Portfolio).where(Portfolio.user_id == user_id).where(Portfolio.is_active)

        result = await self.db.execute(stmt)
        portfolios = result.scalars().all()
        return [PortfolioSchema(**p.to_dict()) for p in portfolios]

    async def get_portfolio(self, portfolio_id: int, user_id: int) -> Optional[PortfolioSchema]:
        """Get a specific portfolio"""
        stmt = select(Portfolio).where(Portfolio.id == portfolio_id).where(Portfolio.user_id == user_id)

        result = await self.db.execute(stmt)
        portfolio = result.scalars().first()

        if not portfolio:
            return None

        positions = self.get_positions(portfolio_id)
        trades = self.get_trades(portfolio_id, limit=10)

        portfolio_dict = portfolio.to_dict()
        portfolio_dict["positions"] = positions
        portfolio_dict["recent_trades"] = trades

        return PortfolioSchema(**portfolio_dict)

    async def update_portfolio(self, portfolio_id: int, user_id: int, update_data: PortfolioUpdate) -> Optional[PortfolioSchema]:
        """Update a portfolio"""
        stmt = select(Portfolio).where(Portfolio.id == portfolio_id).where(Portfolio.user_id == user_id)

        result = await self.db.execute(stmt)
        portfolio = result.scalars().first()
        if not portfolio:
            return None

        update_dict = update_data.model_dump(exclude_unset=True)
        for key, value in update_dict.items():
            setattr(portfolio, key, value)

        await self.db.commit()
        await self.db.refresh(portfolio)

        return PortfolioSchema(**portfolio.to_dict())

    async def delete_portfolio(self, portfolio_id: int, user_id: int) -> bool:
        """Soft delete a portfolio"""
        stmt = select(Portfolio).where(Portfolio.id == portfolio_id).where(Portfolio.user_id == user_id)

        result = await self.db.execute(stmt)
        portfolio = result.scalars().first()

        if not portfolio:
            return False

        portfolio.is_active = False
        await self.db.commit()

        return True

    async def get_positions(self, portfolio_id: int) -> List[PositionSchema]:
        """Get all positions for a portfolio"""
        stmt = select(Position).where(Position.portfolio_id == portfolio_id)

        result = await self.db.execute(stmt)
        positions = result.scalars().all()

        return [PositionSchema(**p.to_dict()) for p in positions]

    async def get_trades(self, portfolio_id: int, limit: int = 100, offset: int = 0) -> List[TradeSchema]:
        """Get trades for a portfolio"""
        stmt = select(Trade).where(Trade.portfolio_id == portfolio_id).order_by(Trade.executed_at.desc()).limit(limit).offset(offset)

        result = await self.db.execute(stmt)
        trades = result.scalars().all()

        return [TradeSchema(**t.to_dict()) for t in trades]

    async def get_portfolio_metrics(self, portfolio_id: int, user_id: int) -> PortfolioMetrics:
        """Get all metrics for a portfolio"""
        stmt = select(Portfolio).where(Portfolio.id == portfolio_id).where(Portfolio.user_id == user_id)

        result = await self.db.execute(stmt)
        portfolio = result.scalars().first()

        stmt = select(Position).where(Position.portfolio_id == portfolio_id)

        result = await self.db.execute(stmt)
        positions = result.scalars().all()

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
