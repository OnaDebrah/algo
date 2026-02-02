from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.app.database import Base


class Position(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)  # Added to match raw SQL side
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)  # Renamed from avg_entry_price
    current_price = Column(Float, nullable=True)
    unrealized_pnl = Column(Float, nullable=True)
    unrealized_pnl_pct = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    portfolio = relationship("Portfolio", back_populates="positions")

    def to_dict(self):
        return {
            "id": self.id,
            "portfolio_id": self.portfolio_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "market_value": self.quantity * (self.current_price or self.entry_price),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
