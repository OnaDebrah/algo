"""
Probability of Profit Calculator for Options Strategies
Provides comprehensive probability analysis using multiple methodologies
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

from ....strategies.options_strategies import OptionsChain

logger = logging.getLogger(__name__)


class ProbabilityMethod(Enum):
    """Available probability calculation methods"""

    MONTE_CARLO = "monte_carlo"
    ANALYTICAL = "analytical"
    HYBRID = "hybrid"
    BOOTSTRAP = "bootstrap"


@dataclass
class ProbabilityResult:
    """Comprehensive probability of profit results"""

    pop: float  # Probability of profit
    method: ProbabilityMethod
    confidence_interval: Tuple[float, float] = None
    expected_return: float = None
    std_dev_return: float = None
    sharpe_ratio: float = None
    var: float = None  # Value at Risk
    cvar: float = None  # Conditional VaR
    max_loss: float = None
    max_profit: float = None
    expected_price: float = None
    price_range: Tuple[float, float] = None
    skewness: float = None
    kurtosis: float = None
    num_simulations: int = None
    convergence: Dict = field(default_factory=dict)
    error_estimate: float = None
    breakeven_probabilities: Dict[float, float] = field(default_factory=dict)
    scenarios: Dict = field(default_factory=dict)
    error: Optional[str] = None
    note: Optional[str] = None


class ProbabilityOfProfit:
    """
    Probability of Profit Calculator for Options Strategies

    Provides comprehensive probability analysis using:
    - Monte Carlo simulation
    - Analytical methods (log-normal distribution)
    - Historical bootstrapping
    - Hybrid approaches

    Features:
    - Confidence intervals for POP estimates
    - Risk metrics (VaR, CVaR, Sharpe ratio)
    - Multiple breakeven point handling
    - Strategy-type detection
    - Volatility estimation
    """

    def __init__(
        self,
        strategy,  # OptionsStrategy instance
        risk_free_rate: float = 0.05,
        default_volatility: float = 0.3,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize Probability of Profit calculator

        Args:
            strategy: OptionsStrategy instance with legs
            risk_free_rate: Risk-free interest rate
            default_volatility: Default volatility if none available
            random_seed: Random seed for reproducibility
        """
        self.strategy = strategy
        self.risk_free_rate = risk_free_rate
        self.default_volatility = default_volatility
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize cache
        self._price_cache = {}
        self._volatility_cache = {}

    def calculate(
        self,
        volatility: Optional[float] = None,
        days_to_expiration: Optional[int] = None,
        method: Union[str, ProbabilityMethod] = ProbabilityMethod.HYBRID,
        num_simulations: int = 10000,
        confidence_level: float = 0.95,
        use_geometric_brownian: bool = True,
        dividend_yield: float = 0.0,
        num_std_devs: float = 3.0,
        bootstrap_history_days: int = 252,
        parallel: bool = False,
        return_distribution: bool = False,
    ) -> ProbabilityResult:
        """
        Calculate probability of profit

        Args:
            volatility: Implied volatility (if None, estimates from market)
            days_to_expiration: Days to expiration (if None, uses first leg expiry)
            method: Calculation method (monte_carlo, analytical, hybrid, bootstrap)
            num_simulations: Number of simulations for Monte Carlo
            confidence_level: Confidence level for intervals
            use_geometric_brownian: Use GBM (True) or simple log-normal (False)
            dividend_yield: Continuous dividend yield
            num_std_devs: Number of std devs for price range
            bootstrap_history_days: Days of history for bootstrap
            parallel: Use parallel processing for simulations
            return_distribution: Return full distribution in result

        Returns:
            ProbabilityResult object with all metrics
        """

        # Validate strategy
        if not self.strategy.legs:
            return ProbabilityResult(pop=0.0, method=self._get_method_enum(method), error="No legs in strategy")

        # Get time to expiration
        T, days = self._get_time_to_expiration(days_to_expiration)
        if T <= 0:
            return ProbabilityResult(pop=self._calculate_expiration_pop(), method=self._get_method_enum(method), note="At expiration")

        # Get or estimate volatility
        if volatility is None:
            volatility = self._estimate_implied_volatility() or self.default_volatility

        # Get current price
        current_price = self._get_current_price()

        # Get breakeven points
        breakevens = self._get_breakeven_points()

        # Calculate based on method
        method_enum = self._get_method_enum(method)

        if method_enum == ProbabilityMethod.ANALYTICAL:
            result = self._calculate_analytical(
                current_price=current_price,
                volatility=volatility,
                T=T,
                breakevens=breakevens,
                num_std_devs=num_std_devs,
                dividend_yield=dividend_yield,
            )

        elif method_enum == ProbabilityMethod.MONTE_CARLO:
            result = self._calculate_monte_carlo(
                current_price=current_price,
                volatility=volatility,
                T=T,
                num_simulations=num_simulations,
                confidence_level=confidence_level,
                use_geometric_brownian=use_geometric_brownian,
                dividend_yield=dividend_yield,
                parallel=parallel,
                return_distribution=return_distribution,
            )

        elif method_enum == ProbabilityMethod.BOOTSTRAP:
            result = self._calculate_bootstrap(
                current_price=current_price,
                days=days,
                num_simulations=num_simulations,
                history_days=bootstrap_history_days,
                confidence_level=confidence_level,
            )

        else:  # HYBRID
            result = self._calculate_hybrid(
                current_price=current_price,
                volatility=volatility,
                T=T,
                days=days,
                breakevens=breakevens,
                num_simulations=num_simulations,
                confidence_level=confidence_level,
                dividend_yield=dividend_yield,
            )

        # Add common metadata
        result.method = method_enum
        result.convergence.update(
            {
                "volatility_used": volatility,
                "days_to_expiration": days,
                "current_price": current_price,
                "num_breakevens": len(breakevens),
                "risk_free_rate": self.risk_free_rate,
                "dividend_yield": dividend_yield,
            }
        )

        return result

    def _calculate_monte_carlo(
        self,
        current_price: float,
        volatility: float,
        T: float,
        num_simulations: int,
        confidence_level: float,
        use_geometric_brownian: bool,
        dividend_yield: float,
        parallel: bool,
        return_distribution: bool,
    ) -> ProbabilityResult:
        """Monte Carlo simulation for probability of profit"""

        # Generate random price paths
        if use_geometric_brownian:
            # Geometric Brownian Motion
            drift = self.risk_free_rate - dividend_yield - 0.5 * volatility**2
            random_shock = volatility * np.sqrt(T) * np.random.randn(num_simulations)
            prices = current_price * np.exp(drift * T + random_shock)
        else:
            # Simple log-normal
            prices = current_price * np.exp(
                (self.risk_free_rate - dividend_yield - 0.5 * volatility**2) * T + volatility * np.sqrt(T) * np.random.randn(num_simulations)
            )

        # Calculate payoffs
        payoffs = self._calculate_payoff(prices)

        # Calculate probabilities
        profitable_paths = payoffs > 0
        pop = np.sum(profitable_paths) / num_simulations

        # Confidence interval for POP
        z_score = norm.ppf((1 + confidence_level) / 2)
        pop_std = np.sqrt(pop * (1 - pop) / num_simulations)
        pop_ci_lower = max(0, pop - z_score * pop_std)
        pop_ci_upper = min(1, pop + z_score * pop_std)

        # Calculate return metrics
        expected_return = np.mean(payoffs).item()
        std_dev_return = np.std(payoffs).item()

        # Sharpe-like ratio
        sharpe_ratio = expected_return / std_dev_return if std_dev_return > 0 else 0

        # VaR and CVaR
        sorted_payoffs = np.sort(payoffs)
        var_index = int((1 - confidence_level) * num_simulations)
        var = sorted_payoffs[var_index].item()
        cvar = np.mean(sorted_payoffs[:var_index]) if var_index > 0 else 0

        # Distribution moments
        if len(payoffs) > 1:
            skewness = pd.Series(payoffs).skew()
            kurtosis = pd.Series(payoffs).kurtosis()
        else:
            skewness = kurtosis = 0

        # Breakeven probabilities
        breakeven_probs = {}
        for be in self._get_breakeven_points():
            prob_above = np.sum(prices > be) / num_simulations
            prob_below = np.sum(prices < be) / num_simulations
            breakeven_probs[be] = {"above": prob_above, "below": prob_below, "exact": np.sum(np.abs(prices - be) < be * 0.01) / num_simulations}

        # Scenario analysis
        scenarios = {
            "best_case": np.max(payoffs),
            "worst_case": np.min(payoffs),
            "median_case": np.median(payoffs),
            "profit_scenarios": {
                "profit_gt_0": pop,
                "profit_gt_10pct": np.sum(payoffs > expected_return * 0.1) / num_simulations,
                "profit_gt_25pct": np.sum(payoffs > expected_return * 0.25) / num_simulations,
                "profit_gt_50pct": np.sum(payoffs > expected_return * 0.5) / num_simulations,
            },
            "loss_scenarios": {
                "loss_any": np.sum(payoffs < 0) / num_simulations,
                "loss_gt_10pct": np.sum(payoffs < expected_return * -0.1) / num_simulations,
                "loss_gt_25pct": np.sum(payoffs < expected_return * -0.25) / num_simulations,
                "loss_gt_50pct": np.sum(payoffs < expected_return * -0.5) / num_simulations,
            },
        }

        result = ProbabilityResult(
            pop=pop,
            method=ProbabilityMethod.MONTE_CARLO,
            confidence_interval=(pop_ci_lower, pop_ci_upper),
            expected_return=expected_return,
            std_dev_return=std_dev_return,
            sharpe_ratio=sharpe_ratio,
            var=var,
            cvar=cvar,
            max_loss=np.min(payoffs),
            max_profit=np.max(payoffs),
            expected_price=np.mean(prices).item(),
            price_range=(np.percentile(prices, 5), np.percentile(prices, 95)),
            skewness=skewness,
            kurtosis=kurtosis,
            num_simulations=num_simulations,
            error_estimate=pop_std,
            breakeven_probabilities=breakeven_probs,
            scenarios=scenarios,
            convergence={"simulation_converged": pop_std < 0.01, "standard_error": pop_std, "confidence_level": confidence_level},
        )

        if return_distribution:
            result.convergence["price_distribution"] = prices
            result.convergence["payoff_distribution"] = payoffs

        return result

    def _calculate_analytical(
        self, current_price: float, volatility: float, T: float, breakevens: List[float], num_std_devs: float, dividend_yield: float
    ) -> ProbabilityResult:
        """Analytical calculation using log-normal distribution"""

        # Parameters for log-normal distribution
        mu = np.log(current_price) + (self.risk_free_rate - dividend_yield - 0.5 * volatility**2) * T
        sigma = volatility * np.sqrt(T)

        # Calculate POP based on breakeven structure
        if len(breakevens) == 1:
            pop = self._analytical_single_breakeven(breakevens[0], mu, sigma)
        elif len(breakevens) == 2:
            pop = self._analytical_two_breakevens(breakevens, mu, sigma)
        else:
            pop = self._analytical_multiple_breakevens(breakevens, mu, sigma)

        # Expected price
        expected_price = current_price * np.exp(self.risk_free_rate * T)

        # Price range
        price_range = (current_price * np.exp(mu - num_std_devs * sigma), current_price * np.exp(mu + num_std_devs * sigma))

        # Calculate theoretical Greeks for probability
        delta_pop = self._analytical_delta_pop(mu, sigma, breakevens)

        return ProbabilityResult(
            pop=pop,
            method=ProbabilityMethod.ANALYTICAL,
            expected_return=expected_price - current_price,
            expected_price=expected_price,
            price_range=price_range,
            convergence={"log_mean": mu, "log_std": sigma, "delta_pop": delta_pop, "num_breakevens": len(breakevens)},
        )

    async def _calculate_bootstrap(
        self, current_price: float, days: int, num_simulations: int, history_days: int, confidence_level: float
    ) -> ProbabilityResult:
        """Bootstrap using historical returns"""

        # Fetch historical data
        historical_returns = await self._fetch_historical_returns(history_days)

        if len(historical_returns) < 30:
            logger.warning("Insufficient historical data, falling back to Monte Carlo")
            return self._calculate_monte_carlo(
                current_price=current_price,
                volatility=self.default_volatility,
                T=days / 365,
                num_simulations=num_simulations,
                confidence_level=confidence_level,
                use_geometric_brownian=True,
                dividend_yield=0.0,
                parallel=False,
                return_distribution=False,
            )

        # Bootstrap price paths
        prices_list = []
        payoffs_list = []

        for _ in range(num_simulations):
            # Sample random returns with replacement
            sampled_returns = np.random.choice(historical_returns, size=days, replace=True)

            # Calculate cumulative return
            cumulative_return = np.exp(np.sum(sampled_returns))

            # Final price
            final_price = current_price * cumulative_return
            prices_list.append(final_price)

            # Calculate payoff
            payoff = self._calculate_payoff(final_price)
            payoffs_list.append(payoff)

        prices = np.array(prices_list)
        payoffs = np.array(payoffs_list)

        # Calculate metrics
        profitable_paths = payoffs > 0
        pop = np.sum(profitable_paths) / num_simulations

        # Confidence interval
        z_score = norm.ppf((1 + confidence_level) / 2)
        pop_std = np.sqrt(pop * (1 - pop) / num_simulations)
        pop_ci_lower = max(0, pop - z_score * pop_std)
        pop_ci_upper = min(1, pop + z_score * pop_std)

        return ProbabilityResult(
            pop=pop,
            method=ProbabilityMethod.BOOTSTRAP,
            confidence_interval=(pop_ci_lower, pop_ci_upper),
            expected_return=np.mean(payoffs).item(),
            std_dev_return=np.std(payoffs).item(),
            max_loss=np.min(payoffs),
            max_profit=np.max(payoffs),
            expected_price=np.mean(prices).item(),
            num_simulations=num_simulations,
            error_estimate=pop_std,
            convergence={
                "historical_volatility": np.std(historical_returns) * np.sqrt(252),
                "historical_skew": pd.Series(historical_returns).skew(),
                "num_historical_days": len(historical_returns),
            },
        )

    def _calculate_hybrid(
        self,
        current_price: float,
        volatility: float,
        T: float,
        days: int,
        breakevens: List[float],
        num_simulations: int,
        confidence_level: float,
        dividend_yield: float,
    ) -> ProbabilityResult:
        """
        Hybrid approach: Use analytical for main POP, Monte Carlo for risk metrics

        Combines speed of analytical with richness of Monte Carlo
        """

        # Get analytical POP
        analytical = self._calculate_analytical(
            current_price=current_price, volatility=volatility, T=T, breakevens=breakevens, num_std_devs=3.0, dividend_yield=dividend_yield
        )

        # Get Monte Carlo for detailed metrics (with fewer simulations)
        mc = self._calculate_monte_carlo(
            current_price=current_price,
            volatility=volatility,
            T=T,
            num_simulations=min(num_simulations, 5000),  # Reduce for speed
            confidence_level=confidence_level,
            use_geometric_brownian=True,
            dividend_yield=dividend_yield,
            parallel=False,
            return_distribution=False,
        )

        # Combine results
        return ProbabilityResult(
            pop=analytical.pop,
            method=ProbabilityMethod.HYBRID,
            confidence_interval=mc.confidence_interval,
            expected_return=mc.expected_return,
            std_dev_return=mc.std_dev_return,
            sharpe_ratio=mc.sharpe_ratio,
            var=mc.var,
            cvar=mc.cvar,
            max_loss=mc.max_loss,
            max_profit=mc.max_profit,
            expected_price=analytical.expected_price,
            price_range=analytical.price_range,
            skewness=mc.skewness,
            kurtosis=mc.kurtosis,
            num_simulations=mc.num_simulations,
            error_estimate=abs(analytical.pop - mc.pop) / 2,  # Conservative error estimate
            breakeven_probabilities=mc.breakeven_probabilities,
            scenarios=mc.scenarios,
            convergence={
                "analytical_pop": analytical.pop,
                "monte_carlo_pop": mc.pop,
                "pop_difference": analytical.pop - mc.pop,
                "analytical_convergence": analytical.convergence,
                "monte_carlo_convergence": mc.convergence,
            },
        )

    def _analytical_single_breakeven(self, breakeven: float, mu: float, sigma: float) -> float:
        """Analytical POP for single breakeven point"""
        if self._is_bullish_strategy():
            # Bullish: profit when price > breakeven
            return 1 - norm.cdf((np.log(breakeven) - mu) / sigma)
        else:
            # Bearish: profit when price < breakeven
            return norm.cdf((np.log(breakeven) - mu) / sigma)

    def _analytical_two_breakevens(self, breakevens: List[float], mu: float, sigma: float) -> float:
        """Analytical POP for two breakeven points"""
        z1 = (np.log(breakevens[0]) - mu) / sigma
        z2 = (np.log(breakevens[1]) - mu) / sigma

        if self._is_range_bound_strategy():
            # Range-bound: profit between breakevens
            return norm.cdf(z2) - norm.cdf(z1)
        else:
            # Directional: profit outside breakevens
            return norm.cdf(z1) + (1 - norm.cdf(z2))

    def _analytical_multiple_breakevens(self, breakevens: List[float], mu: float, sigma: float, num_points: int = 1000) -> float:
        """Numerical integration for multiple breakevens"""

        # Create price grid
        min_price = min(breakevens) * 0.5
        max_price = max(breakevens) * 2.0

        prices = np.linspace(min_price, max_price, num_points)

        # Log-normal PDF
        pdf = norm.pdf((np.log(prices) - mu) / sigma) / (prices * sigma)

        # Payoffs
        payoffs = self._calculate_payoff(prices)

        # Integrate where payoff > 0
        profitable_mask = payoffs > 0
        if np.any(profitable_mask):
            dx = prices[1] - prices[0]
            pop = np.trapezoid(pdf[profitable_mask], dx=dx[0])
        else:
            pop = 0.0

        return min(pop, 1.0)

    def _analytical_delta_pop(self, mu: float, sigma: float, breakevens: List[float]) -> float:
        """Sensitivity of POP to price changes"""
        if not breakevens:
            return 0.0

        # Approximate derivative using finite difference
        epsilon = 0.001

        pop_up = self._analytical_single_breakeven(breakevens[0], mu + epsilon, sigma) if len(breakevens) == 1 else 0
        pop_down = self._analytical_single_breakeven(breakevens[0], mu - epsilon, sigma) if len(breakevens) == 1 else 0

        return (pop_up - pop_down) / (2 * epsilon) if len(breakevens) == 1 else 0

    def _calculate_payoff(self, prices: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate strategy payoff at given prices"""
        if not hasattr(self.strategy, "calculate_payoff"):
            # Implement basic payoff calculation if not available
            return self._basic_payoff_calculation(prices)

        return self.strategy.calculate_payoff(prices)

    def _basic_payoff_calculation(self, prices: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Basic payoff calculation if strategy doesn't have its own"""
        total_payoff = 0

        for leg in self.strategy.legs:
            if leg.option_type.value == "Stock":
                # Stock position
                if isinstance(prices, np.ndarray):
                    payoff = leg.quantity * (prices - leg.underlying_price)
                else:
                    payoff = leg.quantity * (prices - leg.underlying_price)

            elif leg.option_type.value == "Call":
                # Call option
                if isinstance(prices, np.ndarray):
                    payoff = leg.quantity * np.maximum(prices - leg.strike, 0)
                else:
                    payoff = leg.quantity * max(prices - leg.strike, 0)

            elif leg.option_type.value == "Put":
                # Put option
                if isinstance(prices, np.ndarray):
                    payoff = leg.quantity * np.maximum(leg.strike - prices, 0)
                else:
                    payoff = leg.quantity * max(leg.strike - prices, 0)

            # Adjust for premium
            if leg.premium:
                payoff -= leg.quantity * leg.premium

            total_payoff += payoff

        return total_payoff

    def _get_time_to_expiration(self, days_to_expiration: Optional[int]) -> Tuple[float, int]:
        """Get time to expiration in years and days"""
        if days_to_expiration is not None:
            days = max(0, days_to_expiration)
        else:
            # Find earliest expiry
            expiries = [leg.expiry for leg in self.strategy.legs if hasattr(leg, "expiry") and leg.expiry is not None]

            if not expiries:
                return 0.0, 0

            earliest_expiry = min(expiries)
            days = max(0, (earliest_expiry - datetime.now()).days)

        return max(days / 365.0, 1 / 365), days

    def _get_current_price(self) -> float:
        """Get current underlying price"""
        # Try to get from strategy
        if hasattr(self.strategy, "current_price") and self.strategy.current_price:
            return self.strategy.current_price

        # Try to get from first leg
        if self.strategy.legs and hasattr(self.strategy.legs[0], "underlying_price"):
            return self.strategy.legs[0].underlying_price

        # Default
        return 100.0

    def _get_breakeven_points(self) -> List[float]:
        """Get breakeven points from strategy"""
        if hasattr(self.strategy, "get_breakeven_points"):
            return self.strategy.get_breakeven_points()

        # Simple calculation for basic strategies
        return self._calculate_basic_breakevens()

    def _calculate_basic_breakevens(self) -> List[float]:
        """Calculate breakevens for basic strategies"""
        breakevens = []

        if len(self.strategy.legs) == 1:
            leg = self.strategy.legs[0]
            if leg.option_type.value == "Stock":
                breakevens.append(leg.underlying_price)
            elif leg.option_type.value == "Call":
                if leg.quantity > 0:  # Long call
                    breakevens.append(leg.strike + leg.premium)
                else:  # Short call
                    breakevens.append(leg.strike + leg.premium)
            elif leg.option_type.value == "Put":
                if leg.quantity > 0:  # Long put
                    breakevens.append(leg.strike - leg.premium)
                else:  # Short put
                    breakevens.append(leg.strike - leg.premium)

        return breakevens

    def _estimate_implied_volatility(self) -> Optional[float]:
        """Estimate implied volatility from strategy legs"""
        # Check if legs have IV
        for leg in self.strategy.legs:
            if hasattr(leg, "implied_volatility") and leg.implied_volatility:
                return leg.implied_volatility

        # Try to fetch from market
        if hasattr(self.strategy, "symbol") and self.strategy.symbol:
            try:
                chain = OptionsChain(self.strategy.symbol)

                # Get IV from ATM option
                expiries = self._get_expirations()
                if expiries:
                    iv = chain.get_implied_volatility(expiries[0])
                    if iv:
                        return iv
            except Exception:
                pass

        return None

    async def _fetch_historical_returns(self, days: int) -> np.ndarray:
        """Fetch historical returns for bootstrap"""
        # This would integrate with your data provider
        # Placeholder implementation
        if hasattr(self.strategy, "symbol") and self.strategy.symbol:
            try:
                from ...data.providers.providers import ProviderFactory

                factory = ProviderFactory()
                hist = await factory.fetch_data(self.strategy.symbol, f"{days}d", "1d")

                if not hist.empty and "Close" in hist.columns:
                    returns = hist["Close"].pct_change().dropna().values
                    return returns[~np.isnan(returns) & ~np.isinf(returns)]
            except Exception:
                pass

        # Return synthetic returns if no data
        return np.random.normal(0.0001, 0.02, days)

    def _get_expirations(self) -> List[str]:
        """Get available expirations for the symbol"""
        if hasattr(self.strategy, "legs"):
            expiries = [leg.expiry.strftime("%Y-%m-%d") for leg in self.strategy.legs if hasattr(leg, "expiry") and leg.expiry]
            if expiries:
                return expiries

        return []

    def _calculate_expiration_pop(self) -> float:
        """Calculate POP at expiration"""
        current_price = self._get_current_price()
        payoff = self._calculate_payoff(current_price)
        return 1.0 if payoff > 0 else 0.0

    def _is_bullish_strategy(self) -> bool:
        """Determine if strategy is bullish"""
        # Simple heuristic based on net delta
        total_delta = 0
        for leg in self.strategy.legs:
            if hasattr(leg, "delta"):
                total_delta += leg.delta * leg.quantity

        return total_delta > 0

    def _is_range_bound_strategy(self) -> bool:
        """Determine if strategy profits from range-bound movement"""
        # Check strategy type if available
        if hasattr(self.strategy, "name"):
            strategy_name = self.strategy.name.lower()
            range_patterns = ["iron condor", "butterfly", "calendar", "diagonal"]
            return any(pattern in strategy_name for pattern in range_patterns)

        # Check legs structure
        if len(self.strategy.legs) >= 4:
            # Likely an iron condor or butterfly
            strikes = [leg.strike for leg in self.strategy.legs if hasattr(leg, "strike")]
            if len(strikes) >= 4:
                strikes.sort()
                # Check if strikes are symmetric around current price
                mid_point = (strikes[0] + strikes[-1]) / 2
                current_price = self._get_current_price()
                return abs(mid_point - current_price) / current_price < 0.1

        return False

    def _get_method_enum(self, method: Union[str, ProbabilityMethod]) -> ProbabilityMethod:
        """Convert string method to enum"""
        if isinstance(method, ProbabilityMethod):
            return method

        method_map = {
            "monte_carlo": ProbabilityMethod.MONTE_CARLO,
            "analytical": ProbabilityMethod.ANALYTICAL,
            "hybrid": ProbabilityMethod.HYBRID,
            "bootstrap": ProbabilityMethod.BOOTSTRAP,
        }

        return method_map.get(method.lower(), ProbabilityMethod.HYBRID)

    def compare_methods(
        self, volatility: Optional[float] = None, days_to_expiration: Optional[int] = None, num_simulations: int = 10000
    ) -> pd.DataFrame:
        """Compare POP across all methods"""

        results = []

        for method in ProbabilityMethod:
            try:
                result = self.calculate(volatility=volatility, days_to_expiration=days_to_expiration, method=method, num_simulations=num_simulations)

                results.append(
                    {
                        "Method": method.value,
                        "POP": result.pop,
                        "Expected Return": result.expected_return,
                        "Std Dev": result.std_dev_return,
                        "Sharpe": result.sharpe_ratio,
                        "VaR (95%)": result.var,
                        "CVaR (95%)": result.cvar,
                        "Error Estimate": result.error_estimate,
                    }
                )
            except Exception as e:
                logger.error(f"Error with method {method}: {e}")
                results.append({"Method": method.value, "POP": None, "Error": str(e)})

        return pd.DataFrame(results)

    def sensitivity_analysis(
        self, volatility_range: Tuple[float, float] = (0.1, 0.5), price_range: Tuple[float, float] = (0.7, 1.3), num_points: int = 20
    ) -> Dict[str, np.ndarray]:
        """
        Perform sensitivity analysis on POP

        Args:
            volatility_range: (min_vol, max_vol)
            price_range: (min_price_factor, max_price_factor)
            num_points: Number of points in each dimension

        Returns:
            Dictionary with sensitivity matrices
        """

        current_price = self._get_current_price()
        base_vol = self._estimate_implied_volatility() or self.default_volatility

        vol_values = np.linspace(volatility_range[0], volatility_range[1], num_points)
        price_values = np.linspace(current_price * price_range[0], current_price * price_range[1], num_points)

        # Create sensitivity matrices
        pop_vs_vol = np.zeros(num_points)
        pop_vs_price = np.zeros(num_points)
        pop_2d = np.zeros((num_points, num_points))

        # POP vs volatility
        for i, vol in enumerate(vol_values):
            result = self.calculate(volatility=vol, method=ProbabilityMethod.ANALYTICAL)
            pop_vs_vol[i] = result.pop

        # POP vs price
        # This requires temporarily modifying current price
        original_price = current_price
        for i, price in enumerate(price_values):
            # Temporarily override price
            self.strategy.current_price = price
            result = self.calculate(volatility=base_vol, method=ProbabilityMethod.ANALYTICAL)
            pop_vs_price[i] = result.pop

        # Restore original price
        self.strategy.current_price = original_price

        # 2D sensitivity
        for i, vol in enumerate(vol_values):
            for j, price in enumerate(price_values):
                self.strategy.current_price = price
                result = self.calculate(volatility=vol, method=ProbabilityMethod.ANALYTICAL)
                pop_2d[i, j] = result.pop

        self.strategy.current_price = original_price

        return {
            "volatility_values": vol_values,
            "price_values": price_values,
            "pop_vs_volatility": pop_vs_vol,
            "pop_vs_price": pop_vs_price,
            "pop_2d_surface": pop_2d,
            "volatility_elasticity": np.gradient(pop_vs_vol) / pop_vs_vol * vol_values,
            "price_elasticity": np.gradient(pop_vs_price) / pop_vs_price * price_values,
        }
