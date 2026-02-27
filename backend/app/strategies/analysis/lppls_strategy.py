"""
LPPLS (Log-Periodic Power Law Singularity) strategy for bubble detection
with AI-enhanced confidence scoring based on 2025 research
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict

import numpy as np
from scipy.optimize import differential_evolution

logger = logging.getLogger(__name__)


@dataclass
class LPPLParameters:
    """Parameters for LPPL model"""

    A: float  # Linear coefficient
    B: float  # Power law amplitude
    C1: float  # Cosine amplitude
    C2: float  # Sine amplitude (alternative formulation)
    tc: float  # Critical time (crash date)
    m: float  # Power law exponent (0 < m < 1)
    omega: float  # Angular log-frequency


class LPPLSModel:
    """
    Core LPPLS mathematical model for bubble detection

    The LPPLS model describes the trajectory of a bubble as:
    E[ln p(t)] = A + B(tc - t)^m + C1(tc - t)^m cos(ω ln(tc - t) + φ)

    or alternatively:
    ln p(t) = A + B(tc - t)^m[1 + C cos(ω ln(tc - t) + φ)]
    """

    def __init__(self):
        self.params = None
        self.fit_quality = None
        self.converged = False
        self.initial_guess_used = None
        self.optimization_path = None

    def _objective(self, params: np.ndarray, t: np.ndarray, y: np.ndarray) -> float:
        """
        Objective function for LPPLS fitting
        Minimize sum of squared residuals
        """
        tc, m, omega, A, B, C1, C2 = params

        # Parameter bounds check
        if m <= 0 or m >= 1:
            return 1e10
        if omega <= 0:
            return 1e10
        if tc <= t[-1]:  # Critical time must be after last data point
            return 1e10

        # Compute (tc - t)^m
        tc_minus_t = tc - t
        tc_minus_t_m = np.power(tc_minus_t, m)

        # Compute cos(ω ln(tc - t))
        log_tc_minus_t = np.log(tc_minus_t)
        cos_term = np.cos(omega * log_tc_minus_t)
        sin_term = np.sin(omega * log_tc_minus_t)

        # Model prediction
        y_pred = A + B * tc_minus_t_m + C1 * tc_minus_t_m * cos_term + C2 * tc_minus_t_m * sin_term

        # Sum of squared residuals
        ssr = np.sum((y - y_pred) ** 2)

        return ssr

    def fit(self, prices: np.ndarray, times: np.ndarray, max_search_days: int = 30, verbose: bool = False) -> Dict:
        """
        Fit LPPLS model to price data

        Args:
            prices: Array of prices (will be log-transformed)
            times: Array of time indices (normalized to 0-1 range)
            max_search_days: Maximum days ahead to search for critical time
            verbose: Print fitting progress

        Returns:
            Dictionary with fitted parameters and fit quality
        """
        # Log transform prices
        y = np.log(prices)
        t = times.copy()

        # Normalize time to [0, 1] range
        t_min, t_max = t.min(), t.max()
        t = (t - t_min) / (t_max - t_min)

        # Parameter bounds
        # tc must be > 1 (after normalized time)
        # m in (0, 1)
        # omega in (1, 50) typical range
        bounds = [
            (1.0, 1.0 + max_search_days / 252),  # tc: up to max_search_days ahead
            (0.01, 0.99),  # m: power law exponent
            (1.0, 50.0),  # omega: angular frequency
            (-10, 10),  # A: intercept
            (-10, 10),  # B: amplitude
            (-10, 10),  # C1: cosine amplitude
            (-10, 10),  # C2: sine amplitude
        ]

        # Initial guess - now actually used!
        init_params = [
            1.1,  # tc slightly ahead
            0.5,  # m mid-range
            6.0,  # omega typical value
            y.mean(),  # A around mean log price
            0.0,  # B zero initially
            0.0,  # C1 zero initially
            0.0,  # C2 zero initially
        ]

        # Create initial guess array within bounds
        init_array = np.array(init_params)

        # Ensure initial guess respects bounds
        for i, (low, high) in enumerate(bounds):
            if init_array[i] < low:
                init_array[i] = low + 0.01
            elif init_array[i] > high:
                init_array[i] = high - 0.01

        try:
            # Use differential evolution with initial guess as a seed
            # Method 1: Use initial guess as part of the initial population
            populations = [init_array]

            # Generate diverse initial population around the guess
            np.random.seed(42)
            for _ in range(14):  # popsize=15 total, one is our guess
                perturbed = init_array.copy()
                # Add random perturbations within bounds
                for j, (low, high) in enumerate(bounds):
                    scale = (high - low) * 0.1
                    perturbed[j] += np.random.normal(0, scale)
                    perturbed[j] = np.clip(perturbed[j], low, high)
                populations.append(perturbed)

            # Use differential evolution with custom initial population
            result = differential_evolution(
                self._objective,
                bounds=bounds,
                args=(t, y),
                strategy="best1bin",
                maxiter=1000,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1.5),
                recombination=0.7,
                seed=42,
                init=populations,  # Pass our custom initial population!
                disp=verbose,
            )

            # Alternative Method 2: Use local optimization with initial guess
            # Uncomment this if you prefer a hybrid approach
            """
            # First try local optimization from initial guess
            local_result = minimize(
                self._objective,
                init_array,
                args=(t, y),
                method='L-BFGS-B',
                bounds=bounds
            )

            # If local optimization does well, use it; otherwise use global
            if local_result.success and local_result.fun < 1.0:
                result = local_result
            else:
                result = differential_evolution(
                    self._objective,
                    bounds=bounds,
                    args=(t, y),
                    strategy='best1bin',
                    maxiter=1000,
                    popsize=15,
                    tol=1e-6,
                    seed=42,
                    disp=verbose
                )
            """

            if result.success:
                self.converged = True
                self.initial_guess_used = init_params
                self.optimization_path = result

                # Calculate fit quality (R-squared)
                y_pred = self.predict(t)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r_squared = 1 - (ss_res / ss_tot)

                self.fit_quality = {
                    "r_squared": r_squared,
                    "rmse": np.sqrt(ss_res / len(y)),
                    "aic": len(y) * np.log(ss_res / len(y)) + 2 * len(bounds),
                    "bic": len(y) * np.log(ss_res / len(y)) + len(bounds) * np.log(len(y)),
                    "objective_value": result.fun,
                    "converged": result.success,
                    "iterations": result.nit if hasattr(result, "nit") else None,
                    "initial_guess": init_params,
                    "improvement": init_params[0] - result.x[0] if result.success else 0,
                }

                # Extract parameters
                tc_norm, m, omega, A, B, C1, C2 = result.x

                # Denormalize tc back to original time scale
                tc_actual = tc_norm * (t_max - t_min) + t_min

                param_dict = {
                    "tc": tc_actual,
                    "tc_days_ahead": (tc_actual - t_max) * 252,  # Convert to trading days
                    "m": m,
                    "omega": omega,
                    "A": A,
                    "B": B,
                    "C1": C1,
                    "C2": C2,
                    "amplitude": np.sqrt(C1**2 + C2**2),  # Combined amplitude
                    "phase": np.arctan2(C2, C1),  # Phase angle
                    "optimization_success": result.success,
                    "initial_tc_guess": init_params[0],
                    "tc_shift": init_params[0] - tc_norm,
                }

                # Calculate confidence metrics based on 2025 research
                self._calculate_confidence_metrics(param_dict)

                # Store as dict so predict() and detect_bubble_regime() can access all keys
                self.params = param_dict

                if verbose:
                    logger.info(
                        f"LPPLS optimization: initial tc={init_params[0]:.3f}, " f"final tc={tc_norm:.3f}, improvement={init_params[0]-tc_norm:.3f}"
                    )

                return param_dict
            else:
                logger.warning("LPPLS optimization did not converge")
                return {}

        except Exception as e:
            logger.error(f"LPPLS fitting error: {e}")
            return {}

    def _calculate_confidence_metrics(self, params: Dict):
        """Calculate confidence metrics based on AI-enhanced methodology"""
        if not self.fit_quality:
            return

        # R-squared confidence
        r2 = self.fit_quality["r_squared"]
        r2_conf = min(1.0, max(0.0, (r2 - 0.7) / 0.3))  # 0.7 threshold, 1.0 at 1.0

        # Parameter stability confidence
        m = params.get("m", 0.5)
        omega = params.get("omega", 6)

        # m should be between 0.2 and 0.8 for stable fits
        m_conf = 1.0 - min(1.0, abs(m - 0.5) * 2)

        # omega should be reasonable (not extreme)
        omega_conf = 1.0 - min(1.0, abs(omega - 10) / 40)

        # Combined confidence score (DTCAI - Distance-to-Crash with AI)
        dtcai = 0.4 * r2_conf + 0.3 * m_conf + 0.3 * omega_conf

        self.fit_quality["confidence_score"] = dtcai
        self.fit_quality["r2_confidence"] = r2_conf
        self.fit_quality["m_confidence"] = m_conf
        self.fit_quality["omega_confidence"] = omega_conf

        # Crash probability based on confidence and time to critical
        days_ahead = params.get("tc_days_ahead", 0)
        if days_ahead > 0 and days_ahead < 60:  # Only consider crashes within 60 days
            time_factor = max(0, 1 - days_ahead / 60)  # Higher probability if closer
            params["crash_probability"] = dtcai * time_factor
        else:
            params["crash_probability"] = 0.0

    def predict(self, t: np.ndarray) -> np.ndarray:
        """Generate predictions for given time points"""
        if self.params is None:
            return np.zeros_like(t)

        tc = self.params["tc"]
        m = self.params["m"]
        omega = self.params["omega"]
        A = self.params["A"]
        B = self.params["B"]
        C1 = self.params["C1"]
        C2 = self.params["C2"]
        tc_minus_t = tc - t
        tc_minus_t_m = np.power(tc_minus_t, m)
        log_tc_minus_t = np.log(tc_minus_t)

        return A + B * tc_minus_t_m + C1 * tc_minus_t_m * np.cos(omega * log_tc_minus_t) + C2 * tc_minus_t_m * np.sin(omega * log_tc_minus_t)

    def detect_bubble_regime(self, window_size: int = 60) -> Dict:
        """
        Detect if current regime exhibits bubble characteristics
        Based on LPPLS parameters and residuals
        """
        if self.params is None or not self.fit_quality:
            return {"is_bubble": False, "confidence": 0.0}

        # Bubble indicators
        is_bubble = True
        reasons = []
        confidence = 0.0

        # Condition 1: m should be between 0.2 and 0.8 (not too extreme)
        m = self.params["m"]
        if 0.2 <= m <= 0.8:
            reasons.append(f"m={m:.3f} in acceptable range")
            confidence += 0.3
        else:
            is_bubble = False
            reasons.append(f"m={m:.3f} outside acceptable range")

        # Condition 2: omega should be between 2 and 15 (reasonable oscillation)
        omega = self.params["omega"]
        if 2 <= omega <= 15:
            reasons.append(f"omega={omega:.1f} in typical range")
            confidence += 0.3
        else:
            reasons.append(f"omega={omega:.1f} unusual")

        # Condition 3: Good fit (R² > 0.8)
        if self.fit_quality.get("r_squared", 0) > 0.8:
            reasons.append(f"R²={self.fit_quality['r_squared']:.3f} > 0.8")
            confidence += 0.2
        else:
            is_bubble = False
            reasons.append(f"R²={self.fit_quality.get('r_squared', 0):.3f} < 0.8")

        # Condition 4: Critical time within reasonable future (10-60 days)
        tc_days = self.params["tc_days_ahead"]
        if 10 <= tc_days <= 60:
            reasons.append(f"Critical time {tc_days:.0f} days ahead")
            confidence += 0.2
        elif tc_days > 60:
            reasons.append(f"Critical time {tc_days:.0f} days ahead (too far)")
        else:
            reasons.append("Critical time in past")

        # Normalize confidence
        confidence = min(1.0, confidence)

        return {
            "is_bubble": is_bubble,
            "confidence": confidence,
            "reasons": reasons,
            "crash_probability": self.params.get("crash_probability", 0),
            "critical_date": datetime.now() + timedelta(days=self.params["tc_days_ahead"]),
        }
