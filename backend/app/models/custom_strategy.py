"""
Custom strategy model for user-created AI-generated or manually-written strategies
"""

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..database import Base


class CustomStrategy(Base):
    __tablename__ = "custom_strategies"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    code = Column(Text, nullable=False)
    strategy_type = Column(String(50), nullable=False, default="custom")
    parameters = Column(JSONB, nullable=True, default=dict)
    is_validated = Column(Boolean, default=False)
    is_encrypted = Column(Boolean, default=False)
    ai_generated = Column(Boolean, default=False)
    ai_explanation = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="custom_strategies")

    # ------------------------------------------------------------------
    # Encryption helpers
    # ------------------------------------------------------------------

    def set_code(self, plaintext: str) -> None:
        """Encrypt *plaintext* and store it in the ``code`` column."""
        from ..services.encryption_service import get_encryption_service

        svc = get_encryption_service()
        self.code = svc.encrypt_code(plaintext)
        self.is_encrypted = True

    def get_code(self) -> str:
        """Return the decrypted strategy code.

        If the stored value is not encrypted (legacy row), it is returned as-is.
        """
        from ..services.encryption_service import get_encryption_service

        svc = get_encryption_service()
        return svc.decrypt_code(self.code)

    # ------------------------------------------------------------------

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "description": self.description,
            "code": self.get_code(),
            "strategy_type": self.strategy_type,
            "parameters": self.parameters,
            "is_validated": self.is_validated,
            "ai_generated": self.ai_generated,
            "ai_explanation": self.ai_explanation,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
