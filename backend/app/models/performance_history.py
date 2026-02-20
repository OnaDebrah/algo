from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, Numeric, func

from backend.app.database import Base


class PerformanceHistory(Base):
    __tablename__ = "performance_history"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"))
    timestamp = Column(DateTime, nullable=False)
    equity = Column(Numeric, nullable=False)
    cash = Column(Numeric, nullable=False)
    total_return = Column(Numeric)
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (Index("idx_performance_portfolio_timestamp", "portfolio_id", timestamp.desc()),)
