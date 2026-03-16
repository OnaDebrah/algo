from enum import Enum


class DEVariant(Enum):
    """Differential Evolution variants supported"""

    RAND_1_BIN = "rand_1_bin"  # DE/rand/1/bin - Most common, good exploration
    BEST_1_BIN = "best_1_bin"  # DE/best/1/bin - Faster convergence
    CURRENT_TO_BEST_1_BIN = "current_to_best_1_bin"  # DE/current-to-best/1/bin - Balanced
    RAND_2_BIN = "rand_2_bin"  # DE/rand/2/bin - Enhanced exploration
    ADAPTIVE = "adaptive"  # Self-adaptive F and CR
