from dataclasses import dataclass
from typing import List


@dataclass
class ParameterRange:
    """Defines the range for a tunable parameter"""

    min_val: float
    max_val: float
    param_type: type = float
    categories: List = None
    is_int: bool = False
    log_scale: bool = False  # For parameters that work better on log scale (e.g., periods)

    def __post_init__(self):
        if self.param_type is int or self.is_int:
            self.param_type = int
            self.is_int = True
