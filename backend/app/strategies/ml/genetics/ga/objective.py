from enum import Enum


class OptimizationObjective(Enum):
    """Optimization objectives for fitness evaluation [citation:7]"""

    SHARPE_RATIO = "sharpe"
    TOTAL_RETURN = "total_return"
    SORTINO_RATIO = "sortino"
    CALMAR_RATIO = "calmar"
    MAX_DRAWDOWN = "max_drawdown"  # Minimize
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    COMBINED = "combined"  # Weighted combination
