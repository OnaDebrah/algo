"""
Paper trading models — virtual portfolio with optional strategy automation.
"""

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..database import Base


class PaperPortfolio(Base):
    __tablename__ = "paper_portfolios"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(100), nullable=False)
    initial_cash = Column(Float, nullable=False, default=100000.0)
    current_cash = Column(Float, nullable=False, default=100000.0)
    is_active = Column(Boolean, default=True)

    # Strategy automation (optional — null means manual-only portfolio)
    strategy_key = Column(String(100), nullable=True)  # e.g. "sma_crossover"
    strategy_params = Column(Text, nullable=True)  # JSON string of params
    strategy_symbol = Column(String(20), nullable=True)  # e.g. "AAPL"
    trade_quantity = Column(Float, nullable=True, default=100)  # shares per strategy signal
    data_interval = Column(String(10), nullable=True, default="1d")  # e.g. "1h", "30m", "1d"

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    positions = relationship("PaperPosition", back_populates="portfolio", cascade="all, delete-orphan")
    trades = relationship("PaperTrade", back_populates="portfolio", cascade="all, delete-orphan", order_by="PaperTrade.executed_at.desc()")
    equity_snapshots = relationship(
        "PaperEquitySnapshot", back_populates="portfolio", cascade="all, delete-orphan", order_by="PaperEquitySnapshot.timestamp"
    )

    __table_args__ = (Index("ix_paper_portfolios_user_id", "user_id"),)


class PaperPosition(Base):
    __tablename__ = "paper_positions"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("paper_portfolios.id", ondelete="CASCADE"), nullable=False)
    symbol = Column(String(20), nullable=False)
    quantity = Column(Float, nullable=False, default=0)
    avg_entry_price = Column(Float, nullable=False, default=0)
    current_price = Column(Float, nullable=True)
    opened_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    portfolio = relationship("PaperPortfolio", back_populates="positions")

    __table_args__ = (Index("ix_paper_positions_portfolio_symbol", "portfolio_id", "symbol", unique=True),)


class PaperTrade(Base):
    __tablename__ = "paper_trades"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("paper_portfolios.id", ondelete="CASCADE"), nullable=False)
    symbol = Column(String(20), nullable=False)
    side = Column(String(4), nullable=False)  # "buy" or "sell"
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    slippage = Column(Float, nullable=False, default=0)
    total_cost = Column(Float, nullable=False)
    realized_pnl = Column(Float, nullable=True)
    source = Column(String(20), nullable=False, default="manual")  # "manual" or "strategy"
    executed_at = Column(DateTime(timezone=True), server_default=func.now())

    portfolio = relationship("PaperPortfolio", back_populates="trades")

    __table_args__ = (Index("ix_paper_trades_portfolio_id", "portfolio_id"),)


class PaperEquitySnapshot(Base):
    __tablename__ = "paper_equity_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("paper_portfolios.id", ondelete="CASCADE"), nullable=False)
    equity = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    positions_value = Column(Float, nullable=False, default=0)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    portfolio = relationship("PaperPortfolio", back_populates="equity_snapshots")

    __table_args__ = (Index("ix_paper_equity_snapshots_portfolio_id", "portfolio_id"),)
