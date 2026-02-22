from datetime import datetime
from typing import Dict, List

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.database import bulk_create
from backend.app.models import Portfolio, Position, Trade
from backend.app.models.performance_history import PerformanceHistory


class TradingService:
    @staticmethod
    async def save_trade(db: AsyncSession, trade_data: Dict, portfolio_id: int = 1) -> int:
        """Save trade to database using AsyncSession"""
        """Save trade to database using AsyncSession"""
        trade_data = trade_data.copy()

        # Calculate total value if missing
        if "total_value" not in trade_data:
            trade_data["total_value"] = trade_data["quantity"] * trade_data["price"]

        # Ensure portfolio_id is set
        trade_data["portfolio_id"] = trade_data.get("portfolio_id") or portfolio_id

        # Mapping legacy keys if necessary
        if "timestamp" in trade_data and "executed_at" not in trade_data:
            trade_data["executed_at"] = trade_data["timestamp"]

        new_trade = Trade(**trade_data)
        db.add(new_trade)
        await db.flush()  # flush to get the ID before commit
        return new_trade.id

    @staticmethod
    async def save_trades_bulk(db: AsyncSession, trades: List[Dict], portfolio_id: int):
        """Save multiple trades using the bulk_create helper from database.py"""
        for t in trades:
            t["portfolio_id"] = portfolio_id
        await bulk_create(db, Trade, trades)

    @staticmethod
    async def get_trades(
        db: AsyncSession, portfolio_id: int = 1, limit: int = 100, start_date: datetime = None, end_date: datetime = None
    ) -> List[Trade]:
        """Retrieve trades using SQLAlchemy select"""
        stmt = select(Trade).where(Trade.portfolio_id == portfolio_id)

        if start_date:
            stmt = stmt.where(Trade.executed_at >= start_date)
        if end_date:
            stmt = stmt.where(Trade.executed_at <= end_date)

        stmt = stmt.order_by(Trade.executed_at.desc()).limit(limit)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def get_equity_curve(
        db: AsyncSession, portfolio_id: int, start_date: datetime = None, end_date: datetime = None
    ) -> List[PerformanceHistory]:
        """Retrieve equity curve from database"""
        stmt = select(PerformanceHistory).where(PerformanceHistory.portfolio_id == portfolio_id)

        if start_date:
            stmt = stmt.where(PerformanceHistory.timestamp >= start_date)
        if end_date:
            stmt = stmt.where(PerformanceHistory.timestamp <= end_date)

        stmt = stmt.order_by(PerformanceHistory.timestamp.asc())
        result = await db.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def save_performance(db: AsyncSession, portfolio_id: int, equity: float, cash: float, total_return: float):
        """Save performance snapshot"""
        perf = PerformanceHistory(portfolio_id=portfolio_id, timestamp=datetime.now(), equity=equity, cash=cash, total_return=total_return)
        db.add(perf)

    @staticmethod
    async def create_portfolio(
        db: AsyncSession,
        name: str,
        initial_capital: float,
        user_id: int = 1,
    ) -> int:
        """Create new portfolio with error handling for existing names"""
        # Try to find existing
        stmt = select(Portfolio).where(Portfolio.name == name)
        result = await db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            return existing.id

        new_portfolio = Portfolio(user_id=user_id, name=name, initial_capital=initial_capital, current_capital=initial_capital)
        db.add(new_portfolio)
        await db.flush()
        return new_portfolio.id

    @staticmethod
    async def get_header_metrics(db: AsyncSession, portfolio_id: int = 1) -> Dict:
        """Gather dashboard metrics using async aggregation functions"""
        res_cap = await db.execute(select(Portfolio.current_capital).where(Portfolio.id == portfolio_id))
        nav = res_cap.scalar() or 0.0

        res_exp = await db.execute(
            select(func.coalesce(func.sum(Position.entry_price * Position.quantity), 0.0)).where(Position.portfolio_id == portfolio_id)
        )
        exposure = res_exp.scalar()

        res_pnl = await db.execute(select(func.coalesce(func.sum(Position.unrealized_pnl), 0.0)).where(Position.portfolio_id == portfolio_id))
        unrealized = res_pnl.scalar()

        res_prev = await db.execute(
            select(PerformanceHistory.equity)
            .where(PerformanceHistory.portfolio_id == portfolio_id)
            .order_by(PerformanceHistory.timestamp.desc())
            .limit(1)
        )
        prev_nav = res_prev.scalar() or nav

        return {
            "nav": float(nav),
            "prev_nav": float(prev_nav),
            "exposure": float(exposure),
            "unrealized_pnl": float(unrealized),
        }
