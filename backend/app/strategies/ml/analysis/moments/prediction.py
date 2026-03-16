from dataclasses import dataclass
from typing import Dict


@dataclass
class IntegratedPrediction:
    """Container for integrated predictions"""

    expected_return: float
    expected_volatility: float
    risk_adjusted_return: float
    conviction_score: float
    position_size: float
    signal: int  # -1, 0, 1
    metadata: Dict
