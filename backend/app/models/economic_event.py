"""Economic calendar event model."""

from sqlalchemy import Column, DateTime, Integer, String, func

from ..database import Base


class EconomicEvent(Base):
    __tablename__ = "economic_events"

    id = Column(Integer, primary_key=True, index=True)
    event_name = Column(String(200), nullable=False)
    country = Column(String(5), default="US")
    event_date = Column(DateTime(timezone=True), nullable=False, index=True)
    impact = Column(String(10), nullable=False, index=True)  # high, medium, low
    previous_value = Column(String(50), nullable=True)
    forecast_value = Column(String(50), nullable=True)
    actual_value = Column(String(50), nullable=True)
    category = Column(String(50), nullable=True)  # GDP, CPI, Employment, Fed, Earnings
    source = Column(String(50), default="manual")
    fetched_at = Column(DateTime(timezone=True), server_default=func.now())
