"""Analytics and performance metrics"""

from .performance import (
    calculate_max_drawdown,
    calculate_performance_metrics,
    format_metrics_for_display,
)

__all__ = [
    "calculate_performance_metrics",
    "calculate_max_drawdown",
    "format_metrics_for_display",
]
