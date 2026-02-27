from sqlalchemy import Boolean, Column, DateTime, Integer, func

from ..database import Base


class BubbleDetection(Base):
    __tablename__ = "bubble_detection"
    id = Column(Integer, primary_key=True, index=True)
    detected = Column(Boolean, default=True)
    timestamp = Column(DateTime, server_default=func.now())
