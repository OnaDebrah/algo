from sqlalchemy import Column, DateTime, Integer, Numeric, String, func

from backend.app.database import Base


class OptionsPosition(Base):
    __tablename__ = "options_positions"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20))
    strategy = Column(String(100))
    entry_date = Column(DateTime)
    expiration = Column(DateTime)
    initial_cost = Column(Numeric)
    status = Column(String(20))  # e.g., 'open', 'closed'
    pnl = Column(Numeric)
    created_at = Column(DateTime, server_default=func.now())
