"""Scheduled/recurring backtest models."""

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB

from ..database import Base


class ScheduledBacktest(Base):
    __tablename__ = "scheduled_backtests"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    strategy_key = Column(String(100), nullable=False)
    strategy_params = Column(JSONB, default={})
    symbols = Column(JSONB, nullable=False)  # list of symbols
    interval = Column(String(10), default="1d")
    period = Column(String(10), default="1y")
    initial_capital = Column(Float, default=100000.0)
    schedule_cron = Column(String(50), nullable=False)  # e.g. "0 9 * * 1" = Monday 9am
    is_active = Column(Boolean, default=True, index=True)
    last_run_at = Column(DateTime(timezone=True), nullable=True)
    next_run_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ScheduledBacktestRun(Base):
    __tablename__ = "scheduled_backtest_runs"

    id = Column(Integer, primary_key=True, index=True)
    scheduled_backtest_id = Column(Integer, ForeignKey("scheduled_backtests.id", ondelete="CASCADE"), nullable=False, index=True)
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    result_summary = Column(JSONB, nullable=True)  # key metrics snapshot
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
