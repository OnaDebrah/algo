from sqlalchemy import Column, DateTime, ForeignKey, Integer, Numeric, String, func

from backend.app.database import Base


class OptionsLeg(Base):
    __tablename__ = "options_legs"

    id = Column(Integer, primary_key=True, index=True)
    position_id = Column(Integer, ForeignKey("options_positions.id"))
    option_type = Column(String(10))  # e.g., 'CALL', 'PUT'
    strike = Column(Numeric)
    quantity = Column(Integer)
    premium = Column(Numeric)
    created_at = Column(DateTime, server_default=func.now())
