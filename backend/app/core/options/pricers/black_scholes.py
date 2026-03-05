import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from ...options.pricers.models import Greeks, OptionType


class BlackScholesModel:
    """Black-Scholes option pricing model for European options"""

    @staticmethod
    def price(
        S: float,  # Current stock price
        K: float,  # Strike price
        T: float,  # Time to expiration (years)
        r: float,  # Risk-free rate
        sigma: float,  # Volatility
        option_type: OptionType,
        q: float = 0.0,  # Dividend yield
    ) -> float:
        """Calculate option price using Black-Scholes"""

        if T <= 0:
            # At expiration
            if option_type == OptionType.CALL:
                return max(S - K, 0)
            else:
                return max(K - S, 0)

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == OptionType.CALL:
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

        return price

    @staticmethod
    def greeks(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        q: float = 0.0,
    ) -> Greeks:
        """Calculate option Greeks"""

        if T <= 0:
            return Greeks(delta=0, gamma=0, theta=0, vega=0, rho=0)

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Common calculations
        pdf_d1 = norm.pdf(d1)

        # Delta
        if option_type == OptionType.CALL:
            delta = np.exp(-q * T) * norm.cdf(d1)
        else:
            delta = np.exp(-q * T) * (norm.cdf(d1) - 1)

        # Gamma (same for calls and puts)
        gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * np.sqrt(T))

        # Theta
        term1 = -(S * pdf_d1 * sigma * np.exp(-q * T)) / (2 * np.sqrt(T))
        if option_type == OptionType.CALL:
            term2 = q * S * norm.cdf(d1) * np.exp(-q * T)
            term3 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            term2 = -q * S * norm.cdf(-d1) * np.exp(-q * T)
            term3 = r * K * np.exp(-r * T) * norm.cdf(-d2)

        theta = (term1 + term2 + term3) / 365  # Per day

        # Vega (per 1% change)
        vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T) / 100

        # Rho (per 1% change)
        if option_type == OptionType.CALL:
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

        # Higher-order Greeks
        d2_d1 = d2 - d1  # NOW USED in charm, vanna, vomma

        # Charm (Delta decay) - NOW USING d2_d1
        if option_type == OptionType.CALL:
            charm = -np.exp(-q * T) * (pdf_d1 * (d2_d1 / (sigma * np.sqrt(T)) - d1 / np.sqrt(T)) + q * norm.cdf(d1)) / 365
        else:
            charm = -np.exp(-q * T) * (-pdf_d1 * (d2_d1 / (sigma * np.sqrt(T)) - d1 / np.sqrt(T)) + q * norm.cdf(-d1)) / 365

        # Vanna (Delta sensitivity to vol) - NOW USING d2_d1
        vanna = np.exp(-q * T) * pdf_d1 * (d2_d1 + 1 / sigma) / sigma

        # Vomma (Vega convexity) - NOW USING d2_d1
        vomma = vega * 100 * (d1 * d2 + d2_d1) / sigma

        return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho, charm=charm, vanna=vanna, vomma=vomma)

    @staticmethod
    def implied_volatility(
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: OptionType,
        q: float = 0.0,
        initial_guess: float = 0.3,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> float:
        """Calculate implied volatility using Newton-Raphson method"""

        def price_diff(sigma):
            return BlackScholesModel.price(S, K, T, r, sigma, option_type, q) - market_price

        def vega_func(sigma):
            return BlackScholesModel.greeks(S, K, T, r, sigma, option_type, q).vega * 100

        try:
            # Use brentq for robust root finding
            iv = brentq(
                price_diff,
                0.001,  # Lower bound
                5.0,  # Upper bound
                xtol=tolerance,
                maxiter=max_iterations,
            )
            return iv
        except (ValueError, RuntimeError):
            # Fall back to Newton-Raphson
            sigma = initial_guess
            for i in range(max_iterations):
                diff = price_diff(sigma)
                if abs(diff) < tolerance:
                    return sigma

                vega = vega_func(sigma)
                if abs(vega) < 1e-10:
                    return sigma

                sigma = sigma - diff / vega

                if sigma <= 0:
                    sigma = 0.001

            return sigma
