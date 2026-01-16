from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from backend.app.database import Base

class UsageTracking(Base):
    """Model for tracking user actions and feature usage"""
    __tablename__ = "usage_tracking"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    action = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    metadata_json = Column(Text, nullable=True)  # Using metadata_json to avoid conflict with 'metadata' reserved word

    user = relationship("User", backref="usage_logs")

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "action": self.action,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata_json
        }
