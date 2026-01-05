import numpy as np

from ....strategies.options_strategies import OptionType
from ...options.pricers.models import ExerciseType


class FiniteDifferenceModel:
    """Finite difference method for option pricing"""

    @staticmethod
    def price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        exercise_type: ExerciseType = ExerciseType.EUROPEAN,
        q: float = 0.0,
        S_max: float = None,
        num_S: int = 100,
        num_t: int = 100,
    ) -> float:
        """Price option using finite difference method (Crank-Nicolson)"""

        if S_max is None:
            S_max = S * 3

        # Grid parameters
        dS = S_max / num_S
        dt = T / num_t

        # Grid for asset price
        S_grid = np.linspace(0, S_max, num_S + 1)

        # Initialize option values at maturity
        if option_type == OptionType.CALL:
            V = np.maximum(S_grid - K, 0)
        else:
            V = np.maximum(K - S_grid, 0)

        # Coefficients for Crank-Nicolson
        alpha = 0.25 * dt * (sigma**2 * np.arange(num_S + 1) ** 2 - (r - q) * np.arange(num_S + 1))
        beta = -0.5 * dt * (sigma**2 * np.arange(num_S + 1) ** 2 + r)
        gamma = 0.25 * dt * (sigma**2 * np.arange(num_S + 1) ** 2 + (r - q) * np.arange(num_S + 1))

        # Time stepping backwards
        for _ in range(num_t):
            if exercise_type == ExerciseType.AMERICAN:
                # American option - need to check early exercise
                V_old = V.copy()
                if option_type == OptionType.CALL:
                    V = np.maximum(V, S_grid - K)
                else:
                    V = np.maximum(V, K - S_grid)

                # NOW USING V_old to check convergence
                max_change = np.max(np.abs(V - V_old))
                if max_change < 1e-6 * dS:  # Convergence threshold
                    break  # Exit time-stepping early if converged

            # Set up tridiagonal system
            A = np.zeros((num_S - 1, num_S - 1))
            B = np.zeros((num_S - 1, num_S - 1))

            for i in range(1, num_S):
                # Interior points
                A[i - 1, i - 1] = 1 - beta[i]
                if i > 1:
                    A[i - 1, i - 2] = -alpha[i]
                if i < num_S - 1:
                    A[i - 1, i] = -gamma[i]

                B[i - 1, i - 1] = 1 + beta[i]
                if i > 1:
                    B[i - 1, i - 2] = alpha[i]
                if i < num_S - 1:
                    B[i - 1, i] = gamma[i]

            # Boundary conditions
            rhs = np.dot(B, V[1:-1])

            # Apply boundary conditions
            if option_type == OptionType.CALL:
                # At S=0, option value = 0
                rhs[0] += alpha[1] * V[0]
                # At S=S_max, option value follows linearity
                rhs[-1] += gamma[num_S - 1] * V[num_S]
            else:
                # At S=0, option value = K * exp(-r*(T-t))
                V[0] = K * np.exp(-r * dt)
                rhs[0] += alpha[1] * V[0]
                # At S=S_max, option value = 0
                rhs[-1] += gamma[num_S - 1] * V[num_S]

            # Solve system
            V[1:-1] = np.linalg.solve(A, rhs)

            if exercise_type == ExerciseType.AMERICAN:
                # Check for early exercise
                if option_type == OptionType.CALL:
                    V = np.maximum(V, S_grid - K)
                else:
                    V = np.maximum(V, K - S_grid)

        # Interpolate to get price at given S
        price = np.interp(S, S_grid, V)

        return price
