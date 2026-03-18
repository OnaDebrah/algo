import random
from typing import Dict

import numpy as np

from .parameter_range import ParameterRange


class Chromosome:
    """
    Chromosome representing a trading strategy's parameters.
    Supports binary, real, and integer encoding [citation:3].
    """

    def __init__(self, genes: Dict[str, float], param_ranges: Dict[str, ParameterRange]):
        """
        Initialize chromosome with genes.

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

        # Validate genes against ranges
        self._validate_genes()

    def _validate_genes(self):
        """Ensure genes are within specified ranges"""
        for name, value in self.genes.items():
            if name in self.param_ranges:
                param_range = self.param_ranges[name]

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

    def mutate(self, mutation_rate: float, mutation_strength: float = 0.1):
        """
        Mutate genes with given probability.

        Args:
            mutation_rate: Probability of mutating each gene
            mutation_strength: Standard deviation for Gaussian mutation (as fraction of range)
        """
        for name in self.genes:
            if random.random() < mutation_rate:
                param_range = self.param_ranges.get(name)
                if param_range is None:
                    continue

                # Categorical mutation
                if param_range.categories is not None:
                    current_idx = param_range.categories.index(self.genes[name])
                    new_idx = (current_idx + random.choice([-1, 1])) % len(param_range.categories)
                    self.genes[name] = param_range.categories[new_idx]

                # Numeric mutation
                else:
                    # Gaussian mutation
                    range_size = param_range.max_val - param_range.min_val
                    sigma = range_size * mutation_strength
                    mutation = np.random.normal(0, sigma)

                    new_value = self.genes[name] + mutation

                    # Clip to range
                    new_value = max(param_range.min_val, min(param_range.max_val, new_value))

                    # Convert to int if needed
                    if param_range.is_int:
                        new_value = int(round(new_value))

                    self.genes[name] = new_value

        self.age += 1

    def copy(self) -> "Chromosome":
        """Create a deep copy"""
        new_chromosome = Chromosome(self.genes.copy(), self.param_ranges)
        new_chromosome.fitness = self.fitness
        new_chromosome.fitness_components = self.fitness_components.copy()
        new_chromosome.age = self.age
        new_chromosome.metadata = self.metadata.copy()
        return new_chromosome

    def __repr__(self) -> str:
        return f"Chromosome(fitness={self.fitness:.4f}, genes={self.genes})"
