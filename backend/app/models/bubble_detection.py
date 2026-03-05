from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..database import Base


class BubbleDetection(Base):
    __tablename__ = "bubble_detection"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    detected = Column(Boolean, default=True)
    confidence = Column(Float, nullable=True)
    crash_probability = Column(Float, nullable=True)
    timestamp = Column(DateTime, server_default=func.now())

    # Relationships
    user = relationship("User")
