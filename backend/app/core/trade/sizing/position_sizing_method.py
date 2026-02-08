from enum import Enum


class PositionSizingMethod(str, Enum):
    """Position sizing methods"""

    FIXED_SHARES = "fixed_shares"
    FIXED_DOLLAR = "fixed_dollar"
    FIXED_PERCENT = "fixed_percent"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_BASED = "volatility_based"
    RISK_PARITY = "risk_parity"
    ATR_BASED = "atr_based"
