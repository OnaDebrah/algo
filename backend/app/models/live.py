"""
Live Trading Database Models
SQLAlchemy models for live strategy deployment and monitoring
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlalchemy import ARRAY, JSON, Boolean, Column, DateTime, Enum as SQLEnum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database import Base


class DeploymentMode(str, Enum):
    """Trading mode"""

    PAPER = "paper"
    LIVE = "live"


class StrategyStatus(str, Enum):
    """Strategy execution status"""

    PENDING = "pending"  # Deployed but not started
    RUNNING = "running"  # Actively trading
    PAUSED = "paused"  # Temporarily stopped
    STOPPED = "stopped"  # Permanently stopped
    ERROR = "error"  # Error state


class LiveStrategy(Base):
    """
    Live trading strategy instance
    Represents a deployed strategy with its configuration and performance
    """

    __tablename__ = "live_strategies"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Strategy Definition
    name = Column(String(255), nullable=False)
    strategy_key = Column(String(100), nullable=False)  # e.g., 'sma_crossover'
    parameters = Column(JSON, nullable=False)  # Strategy parameters
    symbols = Column(ARRAY(String), nullable=False)  # ['AAPL'] or ['AAPL', 'MSFT']

    # Deployment Info
    backtest_id = Column(Integer, ForeignKey("backtest_runs.id"), nullable=True)
    deployment_mode = Column(SQLEnum(DeploymentMode), nullable=False)
    status = Column(SQLEnum(StrategyStatus), nullable=False, default=StrategyStatus.PENDING)
    is_deleted = Column(Boolean, default=False, nullable=False)

    # Capital & Risk Management
    initial_capital = Column(Float, nullable=False)
    current_equity = Column(Float, nullable=True)
    max_position_pct = Column(Float, default=20.0)  # Max 20% per position
    stop_loss_pct = Column(Float, default=5.0)  # 5% stop loss
    daily_loss_limit = Column(Float, nullable=True)  # Daily loss limit in $
    max_drawdown_limit = Column(Float, default=20.0)  # Max drawdown %

    # Performance Tracking
    total_return = Column(Float, default=0.0)
    total_return_pct = Column(Float, default=0.0)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, default=0.0)

    # Daily Performance
    daily_pnl = Column(Float, default=0.0)
    daily_trades = Column(Integer, default=0)

    # Backtest Comparison (for reference)
    backtest_return_pct = Column(Float, nullable=True)
    backtest_sharpe = Column(Float, nullable=True)
    backtest_max_drawdown = Column(Float, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    deployed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    stopped_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_trade_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_equity_update: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    # Metadata
    broker = Column(String(50), nullable=True)  # 'alpaca', 'ib', 'paper'
    version = Column(Integer, default=1)  # Strategy version
    notes = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)

    # Relationships
    equity_snapshots = relationship("LiveEquitySnapshot", back_populates="strategy", cascade="all, delete-orphan")
    parameter_snapshots = relationship("LiveStrategySnapshot", back_populates="strategy", cascade="all, delete-orphan")
    trades = relationship("LiveTrade", back_populates="strategy", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<LiveStrategy(id={self.id}, name='{self.name}', status='{self.status}')>"

    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "strategy_key": self.strategy_key,
            "parameters": self.parameters,
            "symbols": self.symbols,
            "backtest_id": self.backtest_id,
            "deployment_mode": self.deployment_mode.value,
            "status": self.status.value,
            "initial_capital": self.initial_capital,
            "current_equity": self.current_equity,
            "max_position_pct": self.max_position_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "daily_loss_limit": self.daily_loss_limit,
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "daily_pnl": self.daily_pnl,
            "backtest_return_pct": self.backtest_return_pct,
            "backtest_sharpe": self.backtest_sharpe,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "last_trade_at": self.last_trade_at.isoformat() if self.last_trade_at else None,
            "broker": self.broker,
            "version": self.version,
            "notes": self.notes,
            "error_message": self.error_message,
        }


class LiveEquitySnapshot(Base):
    """
    Equity snapshots for live strategies
    Captured every 60 seconds during market hours
    """

    __tablename__ = "live_equity_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, ForeignKey("live_strategies.id"), nullable=False)

    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    equity = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    positions_value = Column(Float, default=0.0)

    # Metrics
    daily_pnl = Column(Float, default=0.0)
    total_pnl = Column(Float, default=0.0)
    drawdown_pct = Column(Float, default=0.0)

    # Relationship
    strategy = relationship("LiveStrategy", back_populates="equity_snapshots")

    def __repr__(self):
        return f"<LiveEquitySnapshot(strategy_id={self.strategy_id}, timestamp={self.timestamp}, equity={self.equity})>"

    def to_dict(self):
        return {
            "timestamp": self.timestamp.isoformat(),
            "equity": self.equity,
            "cash": self.cash,
            "positions_value": self.positions_value,
            "daily_pnl": self.daily_pnl,
            "total_pnl": self.total_pnl,
            "drawdown_pct": self.drawdown_pct,
        }


class TradeStatus(str, Enum):
    """Trade status"""

    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class TradeSide(str, Enum):
    """Trade side"""

    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"


class LiveTrade(Base):
    """
    Live trade execution record
    """

    __tablename__ = "live_trades"

    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, ForeignKey("live_strategies.id"), nullable=False)

    # Trade Details
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(SQLEnum(TradeSide), nullable=False)
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=True)
    exit_price = Column(Float, nullable=True)

    # Execution
    order_id = Column(String(100), nullable=True)  # Broker order ID
    status = Column(SQLEnum(TradeStatus), nullable=False, default=TradeStatus.OPEN)

    # P&L
    profit = Column(Float, nullable=True)
    profit_pct = Column(Float, nullable=True)
    commission = Column(Float, default=0.0)
    total_fees = Column(Float, default=0.0)

    # Timestamps
    opened_at = Column(DateTime, nullable=False, default=datetime.now(timezone.utc))
    closed_at = Column(DateTime, nullable=True)

    # Metadata
    strategy_signal = Column(JSON, nullable=True)  # Signal metadata from strategy
    notes = Column(Text, nullable=True)

    # Relationship
    strategy = relationship("LiveStrategy", back_populates="trades")

    def __repr__(self):
        return f"<LiveTrade(id={self.id}, symbol={self.symbol}, side={self.side}, status={self.status})>"

    def to_dict(self):
        return {
            "id": self.id,
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "order_id": self.order_id,
            "status": self.status.value,
            "profit": self.profit,
            "profit_pct": self.profit_pct,
            "commission": self.commission,
            "total_fees": self.total_fees,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "strategy_signal": self.strategy_signal,
            "notes": self.notes,
        }


class StrategyMarketplace(Base):
    """
    Strategy marketplace for verified strategies
    """

    __tablename__ = "strategy_marketplace"

    id = Column(Integer, primary_key=True, index=True)

    # Strategy Info
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    strategy_key = Column(String(100), nullable=False)
    default_parameters = Column(JSON, nullable=True)
    category = Column(String(100), nullable=True)

    # Creator
    creator_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    creator_name = Column(String(255), nullable=True)
    is_verified = Column(Boolean, default=False)

    # Performance (from verified backtests)
    avg_return_pct = Column(Float, nullable=True)
    avg_sharpe = Column(Float, nullable=True)
    avg_win_rate = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    total_deployments = Column(Integer, default=0)

    # Pricing
    price_monthly = Column(Float, default=0.0)
    has_free_trial = Column(Boolean, default=False)
    trial_days = Column(Integer, default=0)

    # Ratings
    rating = Column(Float, default=0.0)  # 0.00 to 5.00
    num_ratings = Column(Integer, default=0)

    # Status
    status = Column(String(20), default="active")  # 'active', 'deprecated'

    # Timestamps
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))

    def __repr__(self):
        return f"<StrategyMarketplace(id={self.id}, name='{self.name}', rating={self.rating})>"

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "strategy_key": self.strategy_key,
            "default_parameters": self.default_parameters,
            "category": self.category,
            "creator_id": self.creator_id,
            "creator_name": self.creator_name,
            "is_verified": self.is_verified,
            "avg_return_pct": self.avg_return_pct,
            "avg_sharpe": self.avg_sharpe,
            "avg_win_rate": self.avg_win_rate,
            "max_drawdown": self.max_drawdown,
            "total_deployments": self.total_deployments,
            "price_monthly": self.price_monthly,
            "has_free_trial": self.has_free_trial,
            "trial_days": self.trial_days,
            "rating": self.rating,
            "num_ratings": self.num_ratings,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class LiveStrategySnapshot(Base):
    """
    Historical snapshots of strategy parameters
    Allows for version tracking and rollbacks
    """

    __tablename__ = "live_strategy_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, ForeignKey("live_strategies.id"), nullable=False)
    version = Column(Integer, nullable=False)
    parameters = Column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    notes = Column(Text, nullable=True)

    # Relationship
    strategy = relationship("LiveStrategy", back_populates="parameter_snapshots")

    def __repr__(self):
        return f"<LiveStrategySnapshot(strategy_id={self.strategy_id}, version={self.version})>"

    def to_dict(self):
        return {
            "id": self.id,
            "strategy_id": self.strategy_id,
            "version": self.version,
            "parameters": self.parameters,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "notes": self.notes,
        }
