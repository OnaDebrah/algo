"""
Portfolio service with full CRUD operations
"""

from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models import Portfolio
from backend.app.models.position import Position
from backend.app.models.trade import Trade
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

    def create_portfolio(self, user_id: int, portfolio_data: PortfolioCreate) -> PortfolioSchema:
        """Create a new portfolio"""
        portfolio = Portfolio(
            user_id=user_id,
            name=portfolio_data.name,
            description=portfolio_data.description,
            initial_capital=portfolio_data.initial_capital,
            current_capital=portfolio_data.initial_capital,
        )

        self.db.add(portfolio)
        self.db.commit()
        self.db.refresh(portfolio)

        return PortfolioSchema(**portfolio.to_dict())

    def get_portfolios(self, user_id: int) -> List[PortfolioSchema]:
        """Get all portfolios for a user"""
        portfolios = self.db.query(Portfolio).filter(Portfolio.user_id == user_id, Portfolio.is_active).all()

        return [PortfolioSchema(**p.to_dict()) for p in portfolios]

    def get_portfolio(self, portfolio_id: int, user_id: int) -> Optional[PortfolioSchema]:
        """Get a specific portfolio"""
        portfolio = self.db.query(Portfolio).filter(Portfolio.id == portfolio_id, Portfolio.user_id == user_id).first()

        if not portfolio:
            return None

        # Include positions and recent trades
        positions = self.get_positions(portfolio_id)
        trades = self.get_trades(portfolio_id, limit=10)

        portfolio_dict = portfolio.to_dict()
        portfolio_dict["positions"] = positions
        portfolio_dict["recent_trades"] = trades

        return PortfolioSchema(**portfolio_dict)

    def update_portfolio(self, portfolio_id: int, user_id: int, update_data: PortfolioUpdate) -> Optional[PortfolioSchema]:
        """Update a portfolio"""
        portfolio = self.db.query(Portfolio).filter(Portfolio.id == portfolio_id, Portfolio.user_id == user_id).first()

        if not portfolio:
            return None

        update_dict = update_data.dict(exclude_unset=True)
        for key, value in update_dict.items():
            setattr(portfolio, key, value)

        self.db.commit()
        self.db.refresh(portfolio)

        return PortfolioSchema(**portfolio.to_dict())

    def delete_portfolio(self, portfolio_id: int, user_id: int) -> bool:
        """Soft delete a portfolio"""
        portfolio = self.db.query(Portfolio).filter(Portfolio.id == portfolio_id, Portfolio.user_id == user_id).first()

        if not portfolio:
            return False

        portfolio.is_active = False
        self.db.commit()

        return True

    def get_positions(self, portfolio_id: int) -> List[PositionSchema]:
        """Get all positions for a portfolio"""
        positions = self.db.query(Position).filter(Position.portfolio_id == portfolio_id).all()

        return [PositionSchema(**p.to_dict()) for p in positions]

    def get_trades(self, portfolio_id: int, limit: int = 100, offset: int = 0) -> List[TradeSchema]:
        """Get trades for a portfolio"""
        trades = self.db.query(Trade).filter(Trade.portfolio_id == portfolio_id).order_by(Trade.executed_at.desc()).limit(limit).offset(offset).all()

        return [TradeSchema(**t.to_dict()) for t in trades]
