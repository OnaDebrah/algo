"""API Key model for programmatic user access."""

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB

from ..database import Base


class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    key_prefix = Column(String(8), nullable=False)  # First 8 chars shown to user
    key_hash = Column(String(128), nullable=False)  # bcrypt hash of full key
    name = Column(String(100), nullable=False)
    permissions = Column(JSONB, default=["read"])  # e.g. ["read", "trade", "write"]
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
