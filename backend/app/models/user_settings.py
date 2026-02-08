"""
Database Migration: Add Live Trading Settings Columns

Add this to your UserSettings model or create a new migration
"""

from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from backend.app.database import Base


class UserSettings(Base):
    """
    User Settings Model
    """

    __tablename__ = "user_settings"

    # Existing columns
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)

    # Backtest settings (existing)
    data_source = Column(String, default="yahoo")
    slippage = Column(Float, default=0.001)
    commission = Column(Float, default=0.002)
    initial_capital = Column(Float, default=10000.0)

    # Live trading settings
    live_data_source = Column(String, default="alpaca", nullable=True)
    default_broker = Column(String, default="paper", nullable=True)
    auto_connect_broker = Column(Boolean, default=False, nullable=True)

    # Broker credentials (encrypted in production!)
    broker_api_key = Column(String, nullable=True)
    broker_api_secret = Column(String, nullable=True)
    broker_base_url = Column(String, nullable=True)

    # General settings
    theme = Column(String, default="dark")
    notifications = Column(Boolean, default=True)
    auto_refresh = Column(Boolean, default=True)
    refresh_interval = Column(Integer, default=30)

    # Relationship
    user = relationship("User", back_populates="settings")
