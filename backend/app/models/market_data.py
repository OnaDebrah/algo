from sqlalchemy import Column, Float, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB

from backend.app.database import Base


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
