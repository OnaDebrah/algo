"""
Portfolio Optimization Module
Implements Modern Portfolio Theory and various optimization strategies
"""

import logging
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class PortfolioOptimizer:
    """Portfolio optimization using Modern Portfolio Theory"""

    def __init__(self, symbols: List[str], lookback_days: int = 252):
        """
        Initialize portfolio optimizer

        Args:
            symbols: List of stock symbols
            lookback_days: Historical data period for calculations (default: 1 year)
        """
        self.symbols = symbols
        self.lookback_days = lookback_days
        self.returns = None
        self.cov_matrix = None
        self.mean_returns = None

    def fetch_data(self) -> pd.DataFrame:
        """Fetch historical data for all symbols"""
        from core.data_fetcher import fetch_stock_data

        prices = pd.DataFrame()

        for symbol in self.symbols:
            data = fetch_stock_data(symbol, "2y", "1d")
            if not data.empty:
                prices[symbol] = data["Close"]
            else:
                logger.warning(f"Could not fetch data for {symbol}")

        # Drop any symbols with missing data
        prices = prices.dropna(axis=1)
        self.symbols = list(prices.columns)

        if len(self.symbols) < 2:
            raise ValueError("Need at least 2 symbols with valid data")

        # Calculate returns
        self.returns = prices.pct_change().dropna().tail(self.lookback_days)
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()

        logger.info(f"Data fetched for {len(self.symbols)} symbols")

        return prices

    def calculate_portfolio_performance(
        self, weights: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate portfolio expected return and volatility

        Args:
            weights: Portfolio weights

        Returns:
            Tuple of (expected_return, volatility)
        """
        # Annualized return
        returns = np.sum(self.mean_returns * weights) * 252

        # Annualized volatility
        volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))

        return returns, volatility

    def negative_sharpe_ratio(
        self, weights: np.ndarray, risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate negative Sharpe ratio (for minimization)

        Args:
            weights: Portfolio weights
            risk_free_rate: Risk-free rate (default: 2%)

        Returns:
            Negative Sharpe ratio
        """
        returns, volatility = self.calculate_portfolio_performance(weights)
        sharpe = (returns - risk_free_rate) / volatility
        return -sharpe

    def optimize_sharpe(self, risk_free_rate: float = 0.02) -> Dict:
        """
        Optimize for maximum Sharpe ratio

        Args:
            risk_free_rate: Risk-free rate

        Returns:
            Dictionary with optimal weights and metrics
        """
        num_assets = len(self.symbols)

        # Constraints
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

        # Bounds (0 to 1 for each weight, no short selling)
        bounds = tuple((0, 1) for _ in range(num_assets))

        # Initial guess (equal weights)
        initial_weights = np.array([1.0 / num_assets] * num_assets)

        # Optimize
        result = minimize(
            self.negative_sharpe_ratio,
            initial_weights,
            args=(risk_free_rate,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        optimal_weights = result.x
        returns, volatility = self.calculate_portfolio_performance(optimal_weights)
        sharpe = (returns - risk_free_rate) / volatility

        logger.info(f"Sharpe optimization complete - Sharpe: {sharpe:.2f}")

        return {
            "weights": dict(zip(self.symbols, optimal_weights)),
            "expected_return": returns,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "method": "Maximum Sharpe Ratio",
        }

    def optimize_min_volatility(self) -> Dict:
        """
        Optimize for minimum volatility

        Returns:
            Dictionary with optimal weights and metrics
        """
        num_assets = len(self.symbols)

        def portfolio_volatility(weights):
            return self.calculate_portfolio_performance(weights)[1]

        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_weights = np.array([1.0 / num_assets] * num_assets)

        result = minimize(
            portfolio_volatility,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        optimal_weights = result.x
        returns, volatility = self.calculate_portfolio_performance(optimal_weights)
        sharpe = (returns - 0.02) / volatility

        logger.info(f"Min volatility optimization complete - Vol: {volatility:.2%}")

        return {
            "weights": dict(zip(self.symbols, optimal_weights)),
            "expected_return": returns,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "method": "Minimum Volatility",
        }

    def optimize_target_return(self, target_return: float) -> Dict:
        """
        Optimize for target return with minimum volatility

        Args:
            target_return: Target annual return

        Returns:
            Dictionary with optimal weights and metrics
        """
        num_assets = len(self.symbols)

        def portfolio_volatility(weights):
            return self.calculate_portfolio_performance(weights)[1]

        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},
            {
                "type": "eq",
                "fun": lambda x: self.calculate_portfolio_performance(x)[0]
                - target_return,
            },
        ]

        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_weights = np.array([1.0 / num_assets] * num_assets)

        result = minimize(
            portfolio_volatility,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if not result.success:
            logger.warning(f"Target return {target_return:.2%} may not be achievable")
            return self.optimize_sharpe()

        optimal_weights = result.x
        returns, volatility = self.calculate_portfolio_performance(optimal_weights)
        sharpe = (returns - 0.02) / volatility

        logger.info(
            f"Target return optimization complete - "
            f"Return: {returns:.2%}, Vol: {volatility:.2%}"
        )

        return {
            "weights": dict(zip(self.symbols, optimal_weights)),
            "expected_return": returns,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "method": f"Target Return ({target_return:.1%})",
        }

    def efficient_frontier(self, num_portfolios: int = 50) -> pd.DataFrame:
        """
        Generate efficient frontier

        Args:
            num_portfolios: Number of portfolios to generate

        Returns:
            DataFrame with efficient frontier portfolios
        """
        # Get range of returns
        min_vol_port = self.optimize_min_volatility()
        max_sharpe_port = self.optimize_sharpe()

        min_return = min_vol_port["expected_return"]
        max_return = max_sharpe_port["expected_return"] * 1.5

        target_returns = np.linspace(min_return, max_return, num_portfolios)

        frontier = []
        for target_return in target_returns:
            try:
                result = self.optimize_target_return(target_return)
                frontier.append(
                    {
                        "return": result["expected_return"],
                        "volatility": result["volatility"],
                        "sharpe": result["sharpe_ratio"],
                        "weights": result["weights"],
                    }
                )
            except Exception as e:
                logger.exception(f"Failed to create efficient frontier: {e}")
                continue

        df = pd.DataFrame(frontier)
        logger.info(f"Efficient frontier generated with {len(df)} portfolios")

        return df

    def equal_weight_portfolio(self) -> Dict:
        """
        Create equal-weighted portfolio

        Returns:
            Dictionary with equal weights and metrics
        """
        num_assets = len(self.symbols)
        weights = np.array([1.0 / num_assets] * num_assets)

        returns, volatility = self.calculate_portfolio_performance(weights)
        sharpe = (returns - 0.02) / volatility

        return {
            "weights": dict(zip(self.symbols, weights)),
            "expected_return": returns,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "method": "Equal Weight",
        }

    def risk_parity_portfolio(self) -> Dict:
        """
        Create risk parity portfolio (equal risk contribution)

        Returns:
            Dictionary with risk parity weights and metrics
        """
        num_assets = len(self.symbols)

        def risk_contribution(weights):
            portfolio_vol = self.calculate_portfolio_performance(weights)[1]
            marginal_contrib = np.dot(self.cov_matrix * 252, weights)
            contrib = weights * marginal_contrib / portfolio_vol

            # Minimize variance of risk contributions
            target_risk = 1.0 / num_assets
            return np.sum((contrib - target_risk) ** 2)

        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0.001, 1) for _ in range(num_assets))
        initial_weights = np.array([1.0 / num_assets] * num_assets)

        result = minimize(
            risk_contribution,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        optimal_weights = result.x
        returns, volatility = self.calculate_portfolio_performance(optimal_weights)
        sharpe = (returns - 0.02) / volatility

        logger.info("Risk parity optimization complete")

        return {
            "weights": dict(zip(self.symbols, optimal_weights)),
            "expected_return": returns,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "method": "Risk Parity",
        }

    def black_litterman(self, views: Dict[str, float], confidence: float = 0.5) -> Dict:
        """
        Black-Litterman model with investor views

        Args:
            views: Dictionary of {symbol: expected_return}
            confidence: Confidence in views (0 to 1)

        Returns:
            Dictionary with Black-Litterman weights and metrics
        """
        # Market equilibrium returns
        market_weights = np.array([1.0 / len(self.symbols)] * len(self.symbols))
        risk_aversion = 2.5
        pi = risk_aversion * np.dot(self.cov_matrix * 252, market_weights)

        # Views matrix
        P = np.zeros((len(views), len(self.symbols)))
        Q = np.zeros(len(views))

        for i, (symbol, view_return) in enumerate(views.items()):
            if symbol in self.symbols:
                idx = self.symbols.index(symbol)
                P[i, idx] = 1
                Q[i] = view_return

        # Uncertainty in views
        omega = confidence * np.dot(np.dot(P, self.cov_matrix * 252), P.T)

        # Black-Litterman formula
        tau = 0.025
        M_inverse = np.linalg.inv(tau * self.cov_matrix * 252)
        omega_inverse = np.linalg.inv(omega)

        bl_returns = np.linalg.inv(M_inverse + np.dot(np.dot(P.T, omega_inverse), P))
        bl_returns = np.dot(
            bl_returns, np.dot(M_inverse, pi) + np.dot(np.dot(P.T, omega_inverse), Q)
        )

        # Optimize with Black-Litterman returns
        num_assets = len(self.symbols)

        def neg_utility(weights):
            portfolio_return = np.sum(bl_returns * weights)
            portfolio_vol = np.sqrt(
                np.dot(weights.T, np.dot(self.cov_matrix * 252, weights))
            )
            return -(portfolio_return - risk_aversion * portfolio_vol**2)

        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_weights = market_weights

        result = minimize(
            neg_utility,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        optimal_weights = result.x
        returns, volatility = self.calculate_portfolio_performance(optimal_weights)
        sharpe = (returns - 0.02) / volatility

        logger.info("Black-Litterman optimization complete")

        return {
            "weights": dict(zip(self.symbols, optimal_weights)),
            "expected_return": returns,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "method": "Black-Litterman",
            "views": views,
        }


class PortfolioBacktest:
    """Backtest portfolio strategies"""

    def __init__(self, symbols: List[str], weights: Dict[str, float]):
        """
        Initialize portfolio backtest

        Args:
            symbols: List of symbols
            weights: Portfolio weights
        """
        self.symbols = symbols
        self.weights = weights

    def run_backtest(self, start_capital: float = 100000, period: str = "1y") -> Dict:
        """
        Run portfolio backtest

        Args:
            start_capital: Starting capital
            period: Backtest period

        Returns:
            Backtest results
        """
        from core.data_fetcher import fetch_stock_data

        # Fetch data for all symbols
        prices = pd.DataFrame()
        for symbol in self.symbols:
            data = fetch_stock_data(symbol, period, "1d")
            if not data.empty:
                prices[symbol] = data["Close"]

        prices = prices.dropna()

        if prices.empty:
            raise ValueError("No price data available")

        # Calculate returns
        returns = prices.pct_change().dropna()

        # Portfolio returns
        weights_array = np.array([self.weights[s] for s in prices.columns])
        portfolio_returns = (returns * weights_array).sum(axis=1)

        # Equity curve
        equity = (1 + portfolio_returns).cumprod() * start_capital

        # Calculate metrics
        total_return = (equity.iloc[-1] / start_capital - 1) * 100
        volatility = portfolio_returns.std() * np.sqrt(252) * 100
        sharpe = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)

        # Max drawdown
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_drawdown = drawdown.min() * 100

        return {
            "total_return": total_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "final_value": equity.iloc[-1],
            "equity_curve": equity,
            "returns": portfolio_returns,
        }
