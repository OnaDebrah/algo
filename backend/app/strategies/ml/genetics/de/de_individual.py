import random
from typing import Dict

import numpy as np
from strategies.ml.genetics.de.parameter_range import ParameterRange


class DEIndividual:
    """
    Individual in Differential Evolution population.
    DE uses vector-based representation with mutation based on difference vectors.
    """

    def __init__(self, genes: Dict[str, float], param_ranges: Dict[str, ParameterRange]):
        """
        Initialize individual with genes.

        Args:
            genes: Dictionary of parameter names to values
            param_ranges: Dictionary of parameter ranges for validation
        """
        self.genes = genes.copy()
        self.param_ranges = param_ranges
        self.fitness = -np.inf
        self.fitness_components = {}  # For multi-objective optimization
        self.age = 0
        self.metadata = {}
        self.trial_vector = None  # For DE trial vector
        self.trial_fitness = -np.inf

        # For adaptive DE variants
        self.F = 0.5  # Mutation factor (can be self-adaptive)
        self.CR = 0.9  # Crossover rate (can be self-adaptive)

        # Validate genes
        self._validate_genes()

    def _validate_genes(self):
        """Ensure genes are within specified ranges"""
        for name, value in self.genes.items():
            if name in self.param_ranges:
                param_range = self.param_ranges[name]

                # Handle log scale transformation
                if param_range.log_scale:
                    value = np.exp(value) if value > 0 else value

                # Handle categorical parameters
                if param_range.categories is not None:
                    if value not in param_range.categories:
                        self.genes[name] = random.choice(param_range.categories)

                # Handle numeric ranges
                else:
                    if value < param_range.min_val:
                        self.genes[name] = param_range.min_val
                    elif value > param_range.max_val:
                        self.genes[name] = param_range.max_val

                    # Convert to int if needed
                    if param_range.is_int:
                        self.genes[name] = int(round(self.genes[name]))

    def copy(self) -> "DEIndividual":
        """Create a deep copy"""
        new_individual = DEIndividual(self.genes.copy(), self.param_ranges)
        new_individual.fitness = self.fitness
        new_individual.fitness_components = self.fitness_components.copy()
        new_individual.age = self.age
        new_individual.metadata = self.metadata.copy()
        new_individual.F = self.F
        new_individual.CR = self.CR
        return new_individual

    def __repr__(self) -> str:
        return f"DEIndividual(fitness={self.fitness:.4f}, F={self.F:.2f}, CR={self.CR:.2f})"
