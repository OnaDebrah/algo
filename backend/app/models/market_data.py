from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, Index, Integer, LargeBinary, String
from sqlalchemy.dialects.postgresql import JSONB

from ..database import Base


class MarketDataCacheModel(Base):
    __tablename__ = "market_data_cache"

    cache_key = Column(String, primary_key=True)
    data_type = Column(String, nullable=False)
    data_json = Column(JSONB, nullable=False)  # JSONB is faster for reads/writes
    created_at = Column(Float, nullable=False)
    expires_at = Column(Float, nullable=False, index=True)
    access_count = Column(Integer, default=0)
    last_accessed = Column(Float)

    # Index for the cleanup task
    __table_args__ = (Index("idx_market_cache_expires_at", expires_at),)


class MacroCacheEntry(Base):
    """PostgreSQL model for macro cache entries"""

    __tablename__ = "macro_cache"

    cache_key = Column(String(64), primary_key=True, index=True)
    data = Column(LargeBinary, nullable=False)  # Pickled DataFrame
    meta_data = Column(JSON, nullable=False, default={})
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True, index=True)
    data_size = Column(Integer, nullable=False)
    compression = Column(String(10), nullable=True)  # 'zlib' or None
    version = Column(Integer, nullable=False, default=1)

    # Indexes for efficient cleanup
    __table_args__ = (
        Index("ix_macro_cache_expires", expires_at),
        Index("ix_macro_cache_created", created_at),
    )

    def __repr__(self):
        return f"<MacroCacheEntry(key={self.cache_key}, size={self.data_size}, expires={self.expires_at})>"


class RateLimitEntry(Base):
    """PostgreSQL model for API rate limit tracking"""

    __tablename__ = "rate_limits"

    provider = Column(String(20), primary_key=True, index=True)
    remaining = Column(Integer, nullable=False)
    reset_at = Column(DateTime, nullable=False)
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f"<RateLimitEntry(provider={self.provider}, remaining={self.remaining}, reset={self.reset_at})>"
