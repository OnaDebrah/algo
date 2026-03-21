"""Audit trail / trade journal model."""

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB

from ..database import Base


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)  # trade, login, strategy_deploy, settings_change
    category = Column(String(30), nullable=False, index=True)  # trade_journal, system, security
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    metadata_ = Column("metadata", JSONB, nullable=True)  # flexible payload
    tags = Column(JSONB, nullable=True)  # user-defined tags like ["swing", "earnings_play"]
    notes = Column(Text, nullable=True)  # user journal notes
    prev_hash = Column(String(64), nullable=True)  # chain integrity
    event_hash = Column(String(64), nullable=False)  # SHA256 of content
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
