from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB

from ..database import Base


class MarketplaceStrategy(Base):
    __tablename__ = "marketplace_strategies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    creator_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    creator_name = Column(String, nullable=False)
    strategy_type = Column(String, nullable=False)
    category = Column(String, nullable=False)
    complexity = Column(String, nullable=False)

    # Strategy configuration
    parameters = Column(JSONB, nullable=False)  # Changed to JSONB

    # Performance metrics
    sharpe_ratio = Column(Float, default=0.0)
    total_return = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    num_trades = Column(Integer, default=0)

    # Access control
    is_proprietary = Column(Boolean, default=False)
    status = Column(String, default="approved", index=True)  # pending_review, approved, rejected
    rejection_reason = Column(Text, nullable=True)
    custom_strategy_id = Column(Integer, ForeignKey("custom_strategies.id"), nullable=True)

    # Marketplace details
    price = Column(Float, default=0.0)
    is_public = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    verification_badge = Column(Text)
    version = Column(String, default="1.0.0")
    tags = Column(JSONB, default=[])
    pros = Column(JSONB, default=[])
    cons = Column(JSONB, default=[])
    risk_level = Column(String, default="medium")
    recommended_capital = Column(Float, default=10000.0)

    # Cloning / Forking
    parent_strategy_id = Column(Integer, ForeignKey("marketplace_strategies.id"), nullable=True)
    fork_count = Column(Integer, default=0)

    # Social
    downloads = Column(Integer, default=0)
    rating = Column(Float, default=0.0)
    num_ratings = Column(Integer, default=0)
    num_reviews = Column(Integer, default=0)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class StrategyBacktest(Base):
    __tablename__ = "strategy_backtests"

    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, ForeignKey("marketplace_strategies.id"), nullable=False)
    version = Column(String, nullable=False)

    backtest_data = Column(JSONB, nullable=False)  # The serialized results
    equity_curve = Column(LargeBinary)  # Pickle blob
    trades_history = Column(LargeBinary)  # Pickle blob
    daily_returns = Column(LargeBinary)  # Pickle blob

    start_date = Column(String)
    end_date = Column(String)
    initial_capital = Column(Float)
    symbols = Column(JSONB)

    created_at = Column(DateTime, server_default=func.now())


class StrategyReview(Base):
    __tablename__ = "strategy_reviews"
    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer, ForeignKey("marketplace_strategies.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    username = Column(String)
    rating = Column(Integer)
    review_text = Column(Text)
    performance_achieved = Column(JSONB)
    created_at = Column(DateTime, server_default=func.now())
    __table_args__ = (UniqueConstraint("strategy_id", "user_id", name="_strategy_user_review_uc"),)


class StrategyFavorite(Base):
    __tablename__ = "strategy_favorites"
    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer, ForeignKey("marketplace_strategies.id", ondelete="CASCADE"))
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    created_at = Column(DateTime, server_default=func.now())
    __table_args__ = (UniqueConstraint("strategy_id", "user_id", name="_strategy_user_fav_uc"),)


class StrategyDownload(Base):
    __tablename__ = "strategy_downloads"
    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer, ForeignKey("marketplace_strategies.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    downloaded_at = Column(DateTime, server_default=func.now())


class StrategyComment(Base):
    """Threaded discussion comments on marketplace strategies."""

    __tablename__ = "strategy_comments"

    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, ForeignKey("marketplace_strategies.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    username = Column(String(100), nullable=False)
    content = Column(Text, nullable=False)
    parent_comment_id = Column(Integer, ForeignKey("strategy_comments.id", ondelete="CASCADE"), nullable=True)
    is_edited = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class StrategyPurchase(Base):
    __tablename__ = "strategy_purchases"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    strategy_id = Column(Integer, ForeignKey("marketplace_strategies.id"), nullable=False)
    stripe_checkout_session_id = Column(String, nullable=True)
    amount_paid = Column(Float, nullable=False)
    currency = Column(String, default="usd")
    status = Column(String, default="completed")  # completed, refunded
    purchased_at = Column(DateTime, server_default=func.now())
    __table_args__ = (UniqueConstraint("user_id", "strategy_id", name="_user_strategy_purchase_uc"),)
