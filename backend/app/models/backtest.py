"""
Backtest models
"""

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..database import Base


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=True)
    backtest_type = Column(String, nullable=False)  # single, multi, options

    # Configuration
    symbols = Column(JSON, nullable=False)  # List of symbols
    strategy_config = Column(JSON, nullable=False)
    period = Column(String, nullable=False)
    interval = Column(String, nullable=False)
    initial_capital = Column(Float, nullable=False)

    # Results
    total_return = Column(Float)
    total_return_pct = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    total_trades = Column(Integer)
    final_equity = Column(Float)

    # Results Data (Stored as JSON)
    equity_curve = Column(JSON, nullable=True)
    trades_json = Column(JSON, nullable=True)

    # Metadata
    status = Column(String, default="pending")  # pending, running, completed, failed
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User", back_populates="backtest_runs")

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "backtest_type": self.backtest_type,
            "symbols": self.symbols,
            "strategy_config": self.strategy_config,
            "period": self.period,
            "interval": self.interval,
            "initial_capital": self.initial_capital,
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "final_equity": self.final_equity,
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
