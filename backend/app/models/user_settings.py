"""
Database Migration: Add Live Trading Settings Columns

Add this to your UserSettings model or create a new migration
"""

from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from backend.app.config import settings
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
    slippage = Column(Float, default=settings.DEFAULT_SLIPPAGE_RATE)
    commission = Column(Float, default=settings.DEFAULT_COMMISSION_RATE)
    initial_capital = Column(Float, default=settings.DEFAULT_INITIAL_CAPITAL)

    # Live trading settings
    live_data_source = Column(String, default="alpaca", nullable=True)
    default_broker = Column(String, default="paper", nullable=True)
    auto_connect_broker = Column(Boolean, default=False, nullable=True)

    # Broker credentials (encrypted in production!)
    broker_api_key = Column(String, nullable=True)
    broker_api_secret = Column(String, nullable=True)
    broker_base_url = Column(String, nullable=True)

    broker_host = Column(String, nullable=True)
    broker_port = Column(Integer, nullable=True)
    broker_client_id = Column(Integer, nullable=True)
    user_ib_account_id = Column(String, nullable=True)

    # General settings
    theme = Column(String, default="dark")
    notifications = Column(Boolean, default=True)
    auto_refresh = Column(Boolean, default=True)
    refresh_interval = Column(Integer, default=30)

    # Relationship
    user = relationship("User", back_populates="settings")
