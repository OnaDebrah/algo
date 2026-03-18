from typing import Dict


class GreekCalculator:
    """Calculates option Greeks using Black-Scholes model"""

    def __init__(self, risk_free_rate: float = 0.03):
        self.r = risk_free_rate

    def calculate_all(self, S: float, K: float, T: float, sigma: float, option_type: str) -> Dict:
        """Calculate all Greeks"""

        if T <= 0 or sigma <= 0:
            return self._zero_greeks()

        import math

        from scipy.stats import norm

        d1 = (math.log(S / K) + (self.r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if option_type == "call":
            delta = norm.cdf(d1)
            theta = -S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) - self.r * K * math.exp(-self.r * T) * norm.cdf(d2)
        else:
            delta = norm.cdf(d1) - 1
            theta = -S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) + self.r * K * math.exp(-self.r * T) * norm.cdf(-d2)

        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
        vega = S * norm.pdf(d1) * math.sqrt(T) / 100  # Per 1% vol change
        rho = K * T * math.exp(-self.r * T) * (norm.cdf(d2) if option_type == "call" else -norm.cdf(-d2))

        return {
            "delta": delta,
            "gamma": gamma,
            "theta": theta / 365,  # Daily theta
            "vega": vega,
            "rho": rho,
        }

    def _zero_greeks(self) -> Dict:
        """Return zero Greeks for edge cases"""
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}
