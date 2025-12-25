"""
Statistical Arbitrage Strategy (Beyond Pairs)
Alpha source: Mean reversion in cointegrated portfolios
Extensions: Multi-asset cointegration baskets, PCA residual trading, Kalman filter hedge ratios
Why add it: Market-neutral, Low correlation to trend strategies, Institutional credibility
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from statsmodels.tsa.vector_ar.vecm import coint_johansen

warnings.filterwarnings("ignore")


class StatisticalArbitrageStrategy:
    """
    Advanced Statistical Arbitrage Strategy

    Features:
    1. Multi-asset cointegration baskets (3+ assets)
    2. PCA-based residual trading
    3. Dynamic hedge ratios with Kalman Filter
    4. Portfolio optimization for basket construction
    """

    def __init__(
        self,
        universe: List[str],
        basket_size: int = 3,
        lookback_period: int = 252,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        stop_loss_threshold: float = 3.0,
        max_basket_weight: float = 0.5,
        method: str = "cointegration",  # "cointegration", "pca", "kalman"
        rebalancing_freq: str = "monthly",
        min_half_life: int = 5,
        **kwargs,
    ):
        """
        Initialize Statistical Arbitrage Strategy

        Args:
            universe: List of asset symbols
            basket_size: Number of assets in each trading basket
            lookback_period: Period for calculating statistics
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit
            stop_loss_threshold: Z-score threshold for stop loss
            max_basket_weight: Maximum weight for any single asset
            method: Method for basket construction
                   "cointegration" - Johansen cointegration test
                   "pca" - Principal Component Analysis
                   "kalman" - Kalman filter for dynamic relationships
            rebalancing_freq: How often to re-evaluate baskets
            min_half_life: Minimum half-life for mean reversion (days)
        """
        self.universe = universe
        self.basket_size = min(basket_size, len(universe))
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.max_basket_weight = max_basket_weight
        self.method = method.lower()
        self.rebalancing_freq = rebalancing_freq
        self.min_half_life = min_half_life

        # Strategy state
        self.active_baskets = []  # List of trading baskets
        self.basket_weights = {}  # Weights for each basket
        self.basket_metadata = {}  # Metadata for each basket
        self.position_history = []

        # Kalman filter parameters
        if self.method == "kalman":
            self.kf_transition_cov = kwargs.get("kf_transition_cov", 1e-4)
            self.kf_observation_cov = kwargs.get("kf_observation_cov", 1e-3)
            self.kalman_states = {}

        # PCA parameters
        if self.method == "pca":
            self.n_components = kwargs.get("n_components", 3)
            self.residual_threshold = kwargs.get("residual_threshold", 0.1)

        # Performance tracking
        self.performance = {
            "total_trades": 0,
            "winning_trades": 0,
            "total_pnl": 0,
            "sharpe_ratio": 0,
        }

    # ============================================================================
    # CORE BASKET CONSTRUCTION METHODS
    # ============================================================================

    def _construct_cointegration_basket(self, prices: pd.DataFrame, test_period: Optional[int] = None) -> Optional[Tuple[List[str], np.ndarray]]:
        """
        Construct cointegrated basket using Johansen test

        Args:
            prices: DataFrame with price data (assets as columns)
            test_period: Period to use for cointegration test

        Returns:
            Tuple of (selected_assets, cointegration_vector) or None
        """
        if test_period is None:
            test_period = self.lookback_period

        if len(prices) < test_period:
            return None

        # Use most recent data for cointegration test
        test_data = prices.iloc[-test_period:]

        # Try to find cointegrated combinations
        best_score = -np.inf
        best_basket = None
        best_vector = None

        # For simplicity, test random combinations
        # In production, use more systematic search
        n_attempts = min(100, 2 ** len(self.universe))

        for _ in range(n_attempts):
            # Randomly select assets for basket
            selected = np.random.choice(
                self.universe,
                size=min(self.basket_size, len(self.universe)),
                replace=False,
            )

            if len(selected) < 2:
                continue

            # Prepare data for Johansen test
            basket_data = test_data[selected].dropna()
            if len(basket_data) < test_period // 2:
                continue

            # Perform Johansen cointegration test
            try:
                result = coint_johansen(
                    basket_data,
                    det_order=0,  # No deterministic terms
                    k_ar_diff=1,  # Lag order
                )

                # Check cointegration (trace test)
                trace_stat = result.lr1[0]  # First eigenvalue
                critical_value = result.cvt[0, 1]  # 95% critical value

                if trace_stat > critical_value:
                    # Get cointegration vector
                    eigenvector = result.evec[:, 0]

                    # Normalize (first asset coefficient = 1)
                    if eigenvector[0] != 0:
                        eigenvector = eigenvector / eigenvector[0]

                    # Calculate half-life of mean reversion
                    spread = self._calculate_spread(basket_data, eigenvector)
                    half_life = self._calculate_half_life(spread)

                    # Score based on half-life and cointegration strength
                    score = trace_stat / critical_value - max(0, half_life - 30) / 100

                    if score > best_score and half_life > self.min_half_life:
                        best_score = score
                        best_basket = list(selected)
                        best_vector = eigenvector

            except Exception:
                continue

        if best_basket is not None:
            # Ensure weights sum to 0 (market neutral)
            if np.sum(best_vector) != 0:
                best_vector = best_vector - np.mean(best_vector)

            # Normalize for risk parity
            best_vector = self._normalize_weights(best_vector)

            return best_basket, best_vector

        return None

    def _construct_pca_basket(self, prices: pd.DataFrame) -> Optional[Tuple[List[str], Dict]]:
        """
        Construct basket using PCA residual trading

        Args:
            prices: DataFrame with price data

        Returns:
            Tuple of (selected_assets, pca_metadata)
        """
        if len(prices) < self.lookback_period:
            return None

        # Calculate returns
        returns = np.log(prices / prices.shift(1)).dropna()

        if len(returns) < 30:
            return None

        # Standardize returns
        returns_standardized = (returns - returns.mean()) / returns.std()

        # Perform PCA
        pca = PCA(n_components=self.n_components)
        pca.fit(returns_standardized)

        # Get principal components
        components = pca.components_
        explained_variance = pca.explained_variance_ratio_

        # Construct residual portfolio
        # Residual = Actual returns - predicted by first n-1 components
        if self.n_components > 1:
            # Use first n-1 components as factors
            factor_components = components[: self.n_components - 1]

            # Project returns onto factor space
            factor_returns = returns_standardized @ factor_components.T
            reconstructed_returns = factor_returns @ factor_components

            # Calculate residuals
            residuals = returns_standardized - reconstructed_returns

            # Find assets with largest residual variance
            residual_variance = residuals.var(axis=0)
            selected_indices = np.argsort(residual_variance)[-self.basket_size :]
            selected_assets = [self.universe[i] for i in selected_indices]

            # Construct mean-reverting portfolio
            # Weights proportional to residual loading on last component
            last_component = components[-1]
            weights = last_component[selected_indices]

            # Make market neutral
            weights = weights - np.mean(weights)
            weights = self._normalize_weights(weights)

            metadata = {
                "components": components,
                "explained_variance": explained_variance,
                "residual_variance": residual_variance[selected_indices],
                "method": "pca",
            }

            return selected_assets, weights, metadata

        return None

    def _construct_kalman_basket(self, prices: pd.DataFrame) -> Optional[Tuple[List[str], Dict]]:
        """
        Construct basket using Kalman filter for dynamic relationships
        """
        # Select a subset of assets
        selected = np.random.choice(
            self.universe,
            size=min(self.basket_size + 2, len(self.universe)),
            replace=False,
        )
        selected = list(selected)

        # Use first asset as dependent variable
        dependent = selected[0]
        independents = selected[1:]

        # Initialize Kalman filter for each independent asset
        kalman_models = {}
        hedge_ratios = []

        for i, asset in enumerate(independents):
            # Simple Kalman filter implementation
            # In production, use pykalman or similar
            kf_state = self._initialize_kalman(prices[dependent].values, prices[asset].values)
            kalman_models[asset] = kf_state

            # Get current hedge ratio
            hedge_ratio = kf_state["current_hedge_ratio"]
            hedge_ratios.append(hedge_ratio)

        # Construct basket weights
        weights = np.ones(len(selected))
        weights[0] = 1.0  # Dependent asset
        weights[1:] = -np.array(hedge_ratios)  # Hedge with independents

        # Normalize
        weights = weights / np.sum(np.abs(weights))

        metadata = {
            "dependent": dependent,
            "independents": independents,
            "kalman_models": kalman_models,
            "hedge_ratios": hedge_ratios,
            "method": "kalman",
        }

        return selected, weights, metadata

    # ============================================================================
    # CORE TRADING LOGIC
    # ============================================================================

    def _calculate_spread(self, prices: pd.DataFrame, weights: np.ndarray) -> pd.Series:
        """
        Calculate portfolio spread

        Args:
            prices: Price DataFrame
            weights: Portfolio weights

        Returns:
            Spread series
        """
        log_prices = np.log(prices)
        spread = log_prices @ weights
        return spread

    def _calculate_half_life(self, series: pd.Series) -> float:
        """
        Calculate half-life of mean reversion

        Args:
            series: Time series

        Returns:
            Half-life in periods
        """
        if len(series) < 2:
            return np.inf

        # Fit OU process: dX = -θ(X - μ)dt + σdW
        delta_x = np.diff(series.values)
        x_lag = series.values[:-1]

        # Remove NaN
        mask = ~np.isnan(delta_x) & ~np.isnan(x_lag)
        delta_x = delta_x[mask]
        x_lag = x_lag[mask]

        if len(delta_x) < 10:
            return np.inf

        # Linear regression: ΔX = α + βX + ε
        beta = np.polyfit(x_lag, delta_x, 1)[0]

        if beta >= 0:  # Not mean-reverting
            return np.inf

        # Half-life = ln(2) / θ, where θ = -β
        half_life = -np.log(2) / beta

        return half_life

    def _calculate_zscore(self, spread: pd.Series, lookback: int = None) -> pd.Series:
        """
        Calculate rolling z-score of spread
        """
        if lookback is None:
            lookback = self.lookback_period

        if len(spread) < lookback:
            return pd.Series([np.nan] * len(spread), index=spread.index)

        rolling_mean = spread.rolling(window=lookback, min_periods=lookback // 2).mean()
        rolling_std = spread.rolling(window=lookback, min_periods=lookback // 2).std()

        zscore = (spread - rolling_mean) / rolling_std
        return zscore

    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Normalize weights for risk parity
        """
        # Ensure sum of absolute weights = 1
        abs_sum = np.sum(np.abs(weights))
        if abs_sum > 0:
            weights = weights / abs_sum

        # Apply max weight constraint
        weights = np.clip(weights, -self.max_basket_weight, self.max_basket_weight)

        # Renormalize
        abs_sum = np.sum(np.abs(weights))
        if abs_sum > 0:
            weights = weights / abs_sum

        return weights

    def _initialize_kalman(self, series_1: np.ndarray, series_2: np.ndarray) -> Dict:
        """
        Initialize Kalman filter for two series
        """
        # Simple implementation - in production use pykalman
        n = len(series_1)

        # Initial state (hedge ratio from OLS)
        if n > 10:
            beta = np.polyfit(series_2, series_1, 1)[0]
        else:
            beta = 1.0

        state = {
            "current_hedge_ratio": beta,
            "hedge_ratio_history": [beta],
            "state_covariance": self.kf_transition_cov,
            "last_update": n,
        }

        return state

    def _update_kalman(self, kalman_state: Dict, new_price_1: float, new_price_2: float) -> Dict:
        """
        Update Kalman filter with new prices
        """
        # Simplified Kalman update
        # In production, use proper Kalman filter implementation

        old_beta = kalman_state["current_hedge_ratio"]

        # Prediction
        predicted_price = old_beta * new_price_2
        prediction_error = new_price_1 - predicted_price

        # Kalman gain (simplified)
        kalman_gain = kalman_state["state_covariance"] / (kalman_state["state_covariance"] + self.kf_observation_cov)

        # Update
        new_beta = old_beta + kalman_gain * prediction_error / new_price_2

        # Update covariance
        new_cov = (1 - kalman_gain) * kalman_state["state_covariance"]

        # Add process noise
        new_cov += self.kf_transition_cov

        kalman_state["current_hedge_ratio"] = new_beta
        kalman_state["hedge_ratio_history"].append(new_beta)
        kalman_state["state_covariance"] = new_cov
        kalman_state["last_update"] += 1

        return kalman_state

    # ============================================================================
    # MAIN STRATEGY METHODS
    # ============================================================================

    def update_baskets(self, prices: pd.DataFrame, force_update: bool = False) -> List[Dict]:
        """
        Update trading baskets based on new data

        Args:
            prices: Latest price data
            force_update: Force re-evaluation of all baskets

        Returns:
            List of updated baskets
        """
        new_baskets = []

        # Check if it's time to rebalance
        if not force_update and len(self.active_baskets) > 0:
            # Check rebalancing frequency
            if self.rebalancing_freq == "monthly":
                # Only rebalance at month end
                if not prices.index[-1].is_month_end:
                    return self.active_baskets

        # Construct baskets based on selected method
        if self.method == "cointegration":
            basket_info = self._construct_cointegration_basket(prices)
            if basket_info:
                assets, weights = basket_info
                basket = {
                    "assets": assets,
                    "weights": weights,
                    "method": "cointegration",
                    "created_at": prices.index[-1],
                    "last_spread": None,
                    "current_position": 0,
                }
                new_baskets.append(basket)

        elif self.method == "pca":
            result = self._construct_pca_basket(prices)
            if result:
                assets, weights, metadata = result
                basket = {
                    "assets": assets,
                    "weights": weights,
                    "metadata": metadata,
                    "method": "pca",
                    "created_at": prices.index[-1],
                    "last_spread": None,
                    "current_position": 0,
                }
                new_baskets.append(basket)

        elif self.method == "kalman":
            result = self._construct_kalman_basket(prices)
            if result:
                assets, weights, metadata = result
                basket = {
                    "assets": assets,
                    "weights": weights,
                    "metadata": metadata,
                    "method": "kalman",
                    "created_at": prices.index[-1],
                    "last_spread": None,
                    "current_position": 0,
                }
                new_baskets.append(basket)

        # Update active baskets
        if new_baskets:
            self.active_baskets = new_baskets

        return self.active_baskets

    def generate_signals(self, prices: pd.DataFrame) -> Dict[str, Dict]:
        """
        Generate trading signals for all active baskets

        Args:
            prices: Latest price data

        Returns:
            Dictionary mapping basket_id to signal details
        """
        signals = {}

        if not self.active_baskets:
            return signals

        for i, basket in enumerate(self.active_baskets):
            basket_id = f"basket_{i}"
            basket_signals = self._generate_basket_signal(basket, prices)
            signals[basket_id] = basket_signals

            # Update basket state
            self.active_baskets[i].update(
                {
                    "last_spread": basket_signals.get("current_spread"),
                    "current_position": basket_signals.get("position", 0),
                    "last_zscore": basket_signals.get("current_zscore"),
                }
            )

        return signals

    def _generate_basket_signal(self, basket: Dict, prices: pd.DataFrame) -> Dict:
        """
        Generate signal for a single basket
        """
        assets = basket["assets"]
        weights = basket["weights"]

        # Get prices for basket assets
        basket_prices = prices[assets].dropna()
        if len(basket_prices) < self.lookback_period // 2:
            return {"signal": 0, "position": 0, "reason": "insufficient_data"}

        # Calculate spread
        spread = self._calculate_spread(basket_prices, weights)

        # Calculate z-score
        zscore = self._calculate_zscore(spread, self.lookback_period)
        current_z = zscore.iloc[-1] if len(zscore) > 0 else 0

        if np.isnan(current_z):
            return {"signal": 0, "position": 0, "reason": "nan_zscore"}

        current_position = basket.get("current_position", 0)
        signal = 0
        position = 0

        # Entry logic
        if current_position == 0:
            if current_z > self.entry_threshold:
                # Spread too wide: short the basket (sell spread)
                signal = -1
                position = -1.0
            elif current_z < -self.entry_threshold:
                # Spread too narrow: long the basket (buy spread)
                signal = 1
                position = 1.0

        # Exit logic
        elif current_position != 0:
            # Stop loss
            if abs(current_z) > self.stop_loss_threshold:
                signal = -np.sign(current_position)
                position = 0

            # Mean reversion exit
            elif abs(current_z) < self.exit_threshold:
                signal = -np.sign(current_position)
                position = 0

            # Hold
            else:
                signal = np.sign(current_position)
                position = current_position

        # Update Kalman models if applicable
        if basket["method"] == "kalman" and "metadata" in basket:
            metadata = basket["metadata"]
            if "kalman_models" in metadata:
                dependent = metadata["dependent"]
                independents = metadata["independents"]

                for asset in independents:
                    if asset in metadata["kalman_models"]:
                        kalman_state = metadata["kalman_models"][asset]
                        new_price_dep = prices[dependent].iloc[-1]
                        new_price_ind = prices[asset].iloc[-1]

                        updated_state = self._update_kalman(kalman_state, new_price_dep, new_price_ind)
                        metadata["kalman_models"][asset] = updated_state

        return {
            "signal": signal,
            "position": position,
            "current_zscore": float(current_z),
            "current_spread": float(spread.iloc[-1]) if len(spread) > 0 else 0,
            "assets": assets,
            "weights": weights.tolist(),
            "method": basket["method"],
        }

    def get_portfolio_weights(self, signals: Dict[str, Dict]) -> Dict[str, float]:
        """
        Convert basket signals to individual asset weights

        Args:
            signals: Dictionary of basket signals

        Returns:
            Dictionary mapping asset symbols to target weights
        """
        portfolio_weights = {asset: 0.0 for asset in self.universe}

        for basket_id, basket_signal in signals.items():
            if basket_signal["signal"] == 0:
                continue

            position = basket_signal["position"]
            assets = basket_signal["assets"]
            weights = basket_signal["weights"]

            # Scale basket position to individual assets
            for asset, weight in zip(assets, weights):
                if asset in portfolio_weights:
                    portfolio_weights[asset] += position * weight

        # Normalize to ensure market neutrality
        long_exposure = sum(w for w in portfolio_weights.values() if w > 0)
        short_exposure = abs(sum(w for w in portfolio_weights.values() if w < 0))

        # Scale to maintain dollar neutrality
        if long_exposure > 0 and short_exposure > 0:
            scale_factor = min(1.0, short_exposure / long_exposure)
            for asset in portfolio_weights:
                if portfolio_weights[asset] > 0:
                    portfolio_weights[asset] *= scale_factor

        return portfolio_weights

    def run_backtest(self, prices: pd.DataFrame, initial_capital: float = 1000000) -> pd.DataFrame:
        """
        Run backtest on historical data

        Args:
            prices: Historical price data
            initial_capital: Initial capital

        Returns:
            DataFrame with backtest results
        """
        results = []
        capital = initial_capital
        positions = {asset: 0 for asset in self.universe}

        # Run strategy day by day
        for i in range(self.lookback_period, len(prices)):
            current_date = prices.index[i]
            historical_data = prices.iloc[: i + 1]

            # Update baskets periodically
            if i % 63 == 0:  # Quarterly
                self.update_baskets(historical_data, force_update=True)

            # Generate signals
            signals = self.generate_signals(historical_data)

            # Get portfolio weights
            target_weights = self.get_portfolio_weights(signals)

            # Calculate returns
            if i > self.lookback_period:
                daily_returns = {}
                portfolio_return = 0

                for asset in self.universe:
                    if positions[asset] != 0:
                        asset_return = prices[asset].iloc[i] / prices[asset].iloc[i - 1] - 1
                        position_return = positions[asset] * asset_return
                        daily_returns[asset] = position_return
                        portfolio_return += position_return * capital

                capital *= 1 + portfolio_return

                # Record results
                results.append(
                    {
                        "date": current_date,
                        "capital": capital,
                        "return": portfolio_return,
                        "num_baskets": len(self.active_baskets),
                        "num_trades": sum(1 for s in signals.values() if s["signal"] != 0),
                    }
                )

            # Update positions (simplified - no transaction costs)
            positions = target_weights

        results_df = pd.DataFrame(results)
        results_df.set_index("date", inplace=True)

        # Calculate performance metrics
        self._calculate_performance_metrics(results_df)

        return results_df

    def _calculate_performance_metrics(self, results: pd.DataFrame):
        """Calculate and store performance metrics"""
        if len(results) < 2:
            return

        returns = results["return"]

        # Basic metrics
        total_return = (results["capital"].iloc[-1] / results["capital"].iloc[0]) - 1
        annual_return = (1 + total_return) ** (252 / len(results)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0

        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Update performance tracking
        self.performance.update(
            {
                "total_return": total_return,
                "annual_return": annual_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_drawdown,
                "calmar_ratio": (annual_return / abs(max_drawdown) if max_drawdown < 0 else 0),
                "win_rate": (len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0),
            }
        )


class RiskParityStatArb(StatisticalArbitrageStrategy):
    """
    Risk-Parity Statistical Arbitrage
    Allocates based on risk contribution rather than equal weights
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Override for risk parity weighting"""
        # In production, calculate risk contributions
        # For now, use inverse volatility weighting
        return weights / np.sum(np.abs(weights))


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="D")

    # Simulate correlated assets
    n_assets = 10
    base_returns = np.random.randn(500)

    asset_returns = []
    for i in range(n_assets):
        # Create correlated returns
        correlation = 0.3 + 0.6 * i / n_assets
        asset_return = correlation * base_returns + np.sqrt(1 - correlation**2) * np.random.randn(500)
        asset_return = asset_return * 0.01  # Scale to 1% daily vol

        # Add some mean reversion
        if i % 3 == 0:
            asset_return = -0.1 * asset_return + 0.9 * asset_return

        asset_returns.append(100 * np.exp(np.cumsum(asset_return)))

    # Create price DataFrame
    prices = pd.DataFrame(
        np.column_stack(asset_returns),
        index=dates,
        columns=[f"ASSET_{i}" for i in range(n_assets)],
    )

    print("=== Statistical Arbitrage Strategy Backtest ===")

    # Initialize strategy
    statarb = StatisticalArbitrageStrategy(
        universe=[f"ASSET_{i}" for i in range(n_assets)],
        basket_size=4,
        lookback_period=252,
        entry_threshold=2.0,
        exit_threshold=0.5,
        stop_loss_threshold=3.0,
        method="cointegration",
        rebalancing_freq="monthly",
    )

    # Run backtest
    results = statarb.run_backtest(prices, initial_capital=1000000)

    print("\nPerformance Metrics:")
    for metric, value in statarb.performance.items():
        print(f"  {metric}: {value:.4f}")

    print(f"\nActive Baskets: {len(statarb.active_baskets)}")

    # Example with PCA method
    print("\n=== PCA-Based Statistical Arbitrage ===")

    pca_statarab = StatisticalArbitrageStrategy(
        universe=[f"ASSET_{i}" for i in range(n_assets)],
        basket_size=5,
        method="pca",
        n_components=3,
    )

    # Example with Kalman filter
    print("\n=== Kalman Filter Statistical Arbitrage ===")

    kalman_statarab = StatisticalArbitrageStrategy(
        universe=[f"ASSET_{i}" for i in range(n_assets)],
        basket_size=3,
        method="kalman",
        kf_transition_cov=1e-4,
        kf_observation_cov=1e-3,
    )
