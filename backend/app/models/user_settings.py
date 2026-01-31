"""
User Settings Model
"""

from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from backend.app.database import Base


class UserSettings(Base):
    """User settings model"""

    __tablename__ = "user_settings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)

    # Backtest settings
    data_source = Column(String, default="yahoo")
    slippage = Column(Float, default=0.001)
    commission = Column(Float, default=0.002)
    initial_capital = Column(Float, default=100000.0)

    # General settings
    theme = Column(String, default="dark")
    notifications = Column(Boolean, default=True)
    auto_refresh = Column(Boolean, default=True)
    refresh_interval = Column(Integer, default=30)

    # Relationship
    user = relationship("User", back_populates="settings")
