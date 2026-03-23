"""Strategy version history model."""

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB

from ..database import Base


class StrategyVersion(Base):
    __tablename__ = "strategy_versions"

    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, nullable=False, index=True)
    strategy_type = Column(String(20), nullable=False)  # "marketplace" or "live"
    version_number = Column(Integer, nullable=False)
    version_label = Column(String(20), nullable=False)  # e.g. "1.0.0"
    parameters_snapshot = Column(JSONB, nullable=False)
    performance_snapshot = Column(JSONB, nullable=True)  # metrics at version time
    change_description = Column(Text, nullable=True)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
