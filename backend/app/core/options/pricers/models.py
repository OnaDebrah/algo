from dataclasses import dataclass
from enum import Enum
from typing import Optional


class OptionType(Enum):
    """Option types"""

    CALL = "Call"
    PUT = "Put"


class ExerciseType(Enum):
    """Option exercise style"""

    EUROPEAN = "European"
    AMERICAN = "American"


class PricingModel(Enum):
    """Available pricing models"""

    BLACK_SCHOLES = "Black-Scholes"
    BINOMIAL_TREE = "Binomial Tree"
    MONTE_CARLO = "Monte Carlo"
    FINITE_DIFFERENCE = "Finite Difference"
    ANALYTIC = "Analytic"


@dataclass
class Greeks:
    """Option Greeks"""

    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    charm: Optional[float] = None
    vanna: Optional[float] = None
    vomma: Optional[float] = None


@dataclass
class PricingResult:
    """Result of option pricing"""

    price: float
    model: PricingModel
    greeks: Optional[Greeks] = None
    convergence: Optional[dict] = None
    error_estimate: Optional[float] = None
