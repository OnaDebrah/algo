"""
Crash prediction model for storing ML-based crash prediction results
"""

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..database import Base


class CrashPrediction(Base):
    __tablename__ = "crash_predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Core prediction outputs
    crash_probability = Column(Float, nullable=False)  # 0.0 - 1.0 combined
    intensity = Column(String(20))  # 'mild', 'moderate', 'severe'
    confidence = Column(Float)  # Overall model confidence

    # LPPLS model signals
    lppls_confidence = Column(Float)
    lppls_crash_probability = Column(Float)
    lppls_bubble_detected = Column(Boolean, default=False)
    lppls_critical_date = Column(DateTime(timezone=True), nullable=True)

    # LSTM stress model signals
    lstm_stress_index = Column(Float)
    lstm_confidence = Column(Float)
    lstm_stress_trend = Column(String(20))  # 'increasing', 'decreasing', 'stable'

    # RF ensemble
    rf_ensemble_probability = Column(Float, nullable=True)

    # Combined / derived score
    combined_score = Column(Float)

    # Hedge recommendation snapshot
    hedge_strategy = Column(String(50), nullable=True)  # 'covered_calls', 'put_spread', 'tail_risk', 'collar'
    hedge_cost = Column(Float, nullable=True)
    hedge_coverage = Column(Float, nullable=True)
    hedge_details = Column(JSON, nullable=True)

    # Alert tracking
    alert_sent = Column(Boolean, default=False)
    alert_level = Column(String(20), nullable=True)

    # Full metadata blob
    meta_data = Column(JSON, nullable=True)

    # Relationships
    user = relationship("User")

    @property
    def probability(self):
        """Alias for crash_probability (used by scheduler)"""
        return self.crash_probability

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "crash_probability": self.crash_probability,
            "intensity": self.intensity,
            "confidence": self.confidence,
            "lppls_confidence": self.lppls_confidence,
            "lppls_crash_probability": self.lppls_crash_probability,
            "lppls_bubble_detected": self.lppls_bubble_detected,
            "lppls_critical_date": self.lppls_critical_date.isoformat() if self.lppls_critical_date else None,
            "lstm_stress_index": self.lstm_stress_index,
            "lstm_confidence": self.lstm_confidence,
            "lstm_stress_trend": self.lstm_stress_trend,
            "rf_ensemble_probability": self.rf_ensemble_probability,
            "combined_score": self.combined_score,
            "hedge_strategy": self.hedge_strategy,
            "hedge_cost": self.hedge_cost,
            "hedge_coverage": self.hedge_coverage,
            "hedge_details": self.hedge_details,
            "alert_sent": self.alert_sent,
            "alert_level": self.alert_level,
            "metadata": self.meta_data,
        }
