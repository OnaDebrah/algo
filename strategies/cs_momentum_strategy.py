"""
Enhanced Cross-Sectional Momentum Strategy
Alpha source: Relative strength within universe with robust implementation
Features: Skip periods, sector neutrality, momentum crashes protection, transaction cost-aware
Used by: AQR, Fama-French research, systematic hedge funds
"""

import logging
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class CrossSectionalMomentumStrategy:
    """
    Cross-Sectional Momentum Strategy

    Key Improvements over basic version:
    1. Skip-period momentum (avoid short-term reversal)
    2. Sector/industry neutrality
    3. Momentum crash protection
    4. Volatility scaling
    5. Transaction cost optimization
    6. Multi-horizon momentum combination
    """

    def __init__(
        self,
        universe: List[str],
        formation_period: int = 252,  # 12-month formation
        skip_period: int = 21,  # Skip last month (avoid reversal)
        holding_period: int = 21,  # Monthly rebalancing
        top_quantile: float = 0.3,  # Long top 30%
        bottom_quantile: float = 0.3,  # Short bottom 30%
        sector_mapping: Optional[Dict] = None,
        volatility_adjustment: bool = True,
        momentum_crash_protection: bool = True,
        max_position_size: float = 0.05,  # 5% max per position
        transaction_cost_bps: float = 5.0,  # 5 bps per trade
        **kwargs,
    ):
        """
        Initialize Cross-Sectional Momentum

        Args:
            universe: List of asset symbols
            formation_period: Momentum calculation period (days)
            skip_period: Period to skip at end (avoid reversal)
            holding_period: Holding period between rebalances
            top_quantile: Top quantile to go long
            bottom_quantile: Bottom quantile to go short
            sector_mapping: Dict mapping assets to sectors/industries
            volatility_adjustment: Scale positions by volatility
            momentum_crash_protection: Implement crash protection rules
            max_position_size: Maximum position size per asset
            transaction_cost_bps: Transaction cost in basis points
        """
        self.universe = universe
        self.formation_period = formation_period
        self.skip_period = skip_period
        self.holding_period = holding_period
        self.top_quantile = top_quantile
        self.bottom_quantile = bottom_quantile
        self.sector_mapping = sector_mapping
        self.volatility_adjustment = volatility_adjustment
        self.momentum_crash_protection = momentum_crash_protection
        self.max_position_size = max_position_size
        self.transaction_cost_bps = transaction_cost_bps

        # Additional parameters
        self.momentum_horizons = kwargs.get("momentum_horizons", [21, 63, 126, 252])
        self.min_data_points = kwargs.get("min_data_points", formation_period // 2)
        self.zero_cost_portfolio = kwargs.get("zero_cost_portfolio", True)
        self.long_only_mode = kwargs.get("long_only_mode", False)

        # State tracking
        self.current_positions = {}
        self.rebalance_dates = []
        self.momentum_scores_history = []
        self.turnover_history = []

        # Performance tracking
        self.performance_metrics = {
            "total_trades": 0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "information_ratio": 0.0,
        }

    # ============================================================================
    # CORE MOMENTUM CALCULATION METHODS
    # ============================================================================

    def calculate_momentum(self, price_data: pd.DataFrame, method: str = "skip_period") -> pd.DataFrame:
        """
        Calculate momentum scores with different methodologies

        Args:
            price_data: DataFrame with prices (assets as columns)
            method: Momentum calculation method

        Returns:
            DataFrame with momentum scores for each asset
        """
        if len(price_data) < self.formation_period + self.skip_period:
            return pd.DataFrame()

        momentum_scores = pd.DataFrame(index=price_data.columns)

        if method == "skip_period":
            # Standard academic approach: skip most recent period
            for asset in price_data.columns:
                if len(price_data[asset].dropna()) >= self.formation_period:
                    past_price = price_data[asset].iloc[-self.formation_period - self.skip_period]
                    recent_price = price_data[asset].iloc[-self.skip_period - 1]
                    momentum = (recent_price / past_price) - 1
                    momentum_scores.loc[asset, "momentum"] = momentum

        elif method == "multi_horizon":
            # Combine multiple momentum horizons
            momentum_series = pd.Series(index=price_data.columns, dtype=float)

            for horizon in self.momentum_horizons:
                if len(price_data) >= horizon + self.skip_period:
                    horizon_momentum = price_data.iloc[-self.skip_period - 1] / price_data.iloc[-horizon - self.skip_period] - 1
                    # Weight by horizon (longer horizons typically more predictive)
                    weight = np.log(horizon) / np.log(max(self.momentum_horizons))
                    momentum_series = momentum_series.add(horizon_momentum * weight, fill_value=0)

            momentum_scores["momentum"] = momentum_series

        elif method == "risk_adjusted":
            # Momentum adjusted by volatility
            for asset in price_data.columns:
                prices = price_data[asset].dropna()
                if len(prices) >= self.formation_period:
                    # Calculate return
                    momentum_return = prices.iloc[-self.skip_period - 1] / prices.iloc[-self.formation_period - self.skip_period] - 1

                    # Calculate volatility
                    returns = np.log(prices / prices.shift(1))
                    if len(returns) >= 63:
                        volatility = returns.rolling(63).std().iloc[-1] * np.sqrt(252)
                    else:
                        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.15

                    # Risk-adjusted momentum (Sharpe-like)
                    if volatility > 0:
                        momentum_scores.loc[asset, "momentum"] = momentum_return / volatility
                    else:
                        momentum_scores.loc[asset, "momentum"] = momentum_return

        else:
            # Simple momentum (no skip period)
            momentum_scores["momentum"] = price_data.iloc[-1] / price_data.iloc[-self.formation_period] - 1

        return momentum_scores.dropna()

    def calculate_volatility(self, price_data: pd.DataFrame, lookback: int = 63) -> pd.Series:
        """
        Calculate volatility for each asset

        Args:
            price_data: Price DataFrame
            lookback: Volatility lookback period

        Returns:
            Series of annualized volatilities
        """
        volatilities = {}

        for asset in price_data.columns:
            prices = price_data[asset].dropna()
            if len(prices) >= lookback:
                returns = np.log(prices / prices.shift(1)).dropna()
                if len(returns) >= lookback:
                    vol = returns.rolling(lookback).std().iloc[-1] * np.sqrt(252)
                else:
                    vol = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.15
            else:
                vol = 0.15  # Default

            volatilities[asset] = vol

        return pd.Series(volatilities)

    # ============================================================================
    # PORTFOLIO CONSTRUCTION METHODS
    # ============================================================================

    def construct_sector_neutral_portfolio(self, momentum_scores: pd.DataFrame, sector_mapping: Dict[str, str]) -> Dict[str, float]:
        """
        Construct sector-neutral momentum portfolio

        Args:
            momentum_scores: DataFrame with momentum scores
            sector_mapping: Dict mapping assets to sectors

        Returns:
            Dict of position weights
        """
        positions = {}

        # Group assets by sector
        sectors = {}
        for asset in momentum_scores.index:
            if asset in sector_mapping:
                sector = sector_mapping[asset]
                if sector not in sectors:
                    sectors[sector] = []
                sectors[sector].append(asset)

        # Within each sector, rank by momentum
        sector_positions = {}

        for sector, sector_assets in sectors.items():
            # Get momentum scores for this sector
            sector_scores = momentum_scores.loc[momentum_scores.index.intersection(sector_assets)]

            if len(sector_scores) >= 2:  # Need at least 2 assets
                # Rank within sector
                sector_ranks = sector_scores["momentum"].rank(ascending=False)

                # Determine long/short cutoff
                n_sector = len(sector_scores)
                top_cutoff = max(1, int(n_sector * self.top_quantile))
                bottom_cutoff = max(1, int(n_sector * (1 - self.bottom_quantile)))

                # Assign positions
                for asset, rank in sector_ranks.items():
                    if rank <= top_cutoff:
                        sector_positions[asset] = 1.0 / top_cutoff  # Equal weight long
                    elif rank >= bottom_cutoff:
                        sector_positions[asset] = -1.0 / (n_sector - bottom_cutoff + 1)  # Equal weight short

        # Normalize to equal dollar long/short
        long_exposure = sum(w for w in sector_positions.values() if w > 0)
        short_exposure = abs(sum(w for w in sector_positions.values() if w < 0))

        if long_exposure > 0 and short_exposure > 0:
            scale = min(1.0, short_exposure / long_exposure)
            for asset in sector_positions:
                if sector_positions[asset] > 0:
                    sector_positions[asset] *= scale

        positions.update(sector_positions)

        return positions

    def apply_volatility_scaling(self, positions: Dict[str, float], volatilities: pd.Series) -> Dict[str, float]:
        """
        Scale positions inversely by volatility

        Args:
            positions: Raw position weights
            volatilities: Asset volatilities

        Returns:
            Volatility-scaled positions
        """
        scaled_positions = {}

        for asset, weight in positions.items():
            if asset in volatilities.index and volatilities[asset] > 0:
                # Scale inversely to volatility
                vol_scale = 0.15 / volatilities[asset]  # Target 15% vol
                scaled_weight = weight * np.clip(vol_scale, 0.5, 2.0)  # Bound scaling
            else:
                scaled_weight = weight

            # Apply position size limit
            scaled_weight = np.clip(scaled_weight, -self.max_position_size, self.max_position_size)
            scaled_positions[asset] = scaled_weight

        return scaled_positions

    def apply_momentum_crash_protection(
        self,
        positions: Dict[str, float],
        market_data: pd.DataFrame,
        crash_threshold: float = -0.05,
    ) -> Dict[str, float]:
        """
        Implement momentum crash protection (Daniel & Moskowitz, 2016)

        Args:
            positions: Current positions
            market_data: Market index data
            crash_threshold: Market return threshold for crash detection

        Returns:
            Adjusted positions with crash protection
        """
        if not self.momentum_crash_protection or len(market_data) < 252:
            return positions

        # Calculate market conditions
        market_returns = np.log(market_data.iloc[-1] / market_data.iloc[-252]) - 1
        market_vol = np.log(market_data / market_data.shift(1)).std() * np.sqrt(252)

        # Check for momentum crash conditions
        # 1. Market down significantly
        # 2. High market volatility
        # 3. Momentum had been strong

        crash_conditions = (
            market_returns.iloc[0] < crash_threshold  # Market down >5%
            and market_vol.iloc[0] > 0.20  # High volatility (>20%)
        )

        if crash_conditions:
            # Reduce or reverse momentum positions
            protection_factor = 0.3  # Reduce to 30% of normal size
            protected_positions = {}

            for asset, weight in positions.items():
                protected_positions[asset] = weight * protection_factor

            return protected_positions

        return positions

    # ============================================================================
    # SIGNAL GENERATION & PORTFOLIO MANAGEMENT
    # ============================================================================

    def generate_signals(
        self,
        price_data: pd.DataFrame,
        market_index: Optional[pd.Series] = None,
        current_positions: Optional[Dict] = None,
        rebalance: bool = True,
    ) -> Dict:
        """
        Generate enhanced cross-sectional momentum signals

        Args:
            price_data: Price DataFrame (assets as columns)
            market_index: Market index series for crash protection
            current_positions: Current portfolio positions
            rebalance: Whether to rebalance or hold

        Returns:
            Dictionary with signals and metadata
        """
        if current_positions is None:
            current_positions = self.current_positions

        # Check if it's rebalance day
        if not rebalance and not self._is_rebalance_day(price_data):
            return {
                "signals": current_positions,
                "action": "hold",
                "metadata": {"reason": "not_rebalance_day"},
            }

        # Calculate momentum scores
        momentum_scores = self.calculate_momentum(price_data, method="skip_period")

        if momentum_scores.empty:
            return {
                "signals": current_positions,
                "action": "hold",
                "metadata": {"reason": "insufficient_data"},
            }

        # Store momentum scores for analysis
        self.momentum_scores_history.append(
            {
                "date": (price_data.index[-1] if hasattr(price_data.index, "__len__") else len(price_data)),
                "scores": momentum_scores["momentum"].to_dict(),
            }
        )

        # Construct initial portfolio
        if self.sector_mapping:
            # Sector-neutral construction
            raw_positions = self.construct_sector_neutral_portfolio(momentum_scores, self.sector_mapping)
        else:
            # Simple cross-sectional ranking
            raw_positions = self._simple_cross_sectional(momentum_scores)

        # Apply volatility scaling if enabled
        if self.volatility_adjustment:
            volatilities = self.calculate_volatility(price_data)
            positions = self.apply_volatility_scaling(raw_positions, volatilities)
        else:
            positions = raw_positions

        # Apply momentum crash protection
        if self.momentum_crash_protection and market_index is not None:
            positions = self.apply_momentum_crash_protection(positions, pd.DataFrame({"market": market_index}))

        # Ensure zero-cost portfolio if required
        if self.zero_cost_portfolio:
            positions = self._ensure_zero_cost(positions)

        # Long-only mode if specified
        if self.long_only_mode:
            positions = {k: max(0, v) for k, v in positions.items()}

        # Calculate turnover and transaction costs
        turnover_info = self._calculate_turnover(current_positions, positions)

        # Update current positions
        self.current_positions = positions
        self.rebalance_dates.append(price_data.index[-1] if hasattr(price_data.index, "__len__") else len(price_data))

        return {
            "signals": positions,
            "action": "rebalance",
            "metadata": {
                "momentum_scores": momentum_scores["momentum"].to_dict(),
                "turnover": turnover_info["turnover"],
                "estimated_cost": turnover_info["estimated_cost"],
                "num_long": sum(1 for v in positions.values() if v > 0),
                "num_short": sum(1 for v in positions.values() if v < 0),
                "gross_exposure": sum(abs(v) for v in positions.values()),
                "net_exposure": sum(positions.values()),
                "date": (price_data.index[-1] if hasattr(price_data.index, "__len__") else len(price_data)),
            },
        }

    def _simple_cross_sectional(self, momentum_scores: pd.DataFrame) -> Dict[str, float]:
        """Simple cross-sectional ranking without sector constraints"""
        ranks = momentum_scores["momentum"].rank(ascending=False)
        n_assets = len(ranks)

        top_cutoff = max(1, int(n_assets * self.top_quantile))
        bottom_cutoff = max(1, int(n_assets * (1 - self.bottom_quantile)))

        positions = {}
        for asset, rank in ranks.items():
            if rank <= top_cutoff:
                positions[asset] = 1.0 / top_cutoff
            elif rank >= bottom_cutoff:
                positions[asset] = -1.0 / (n_assets - bottom_cutoff + 1)
            else:
                positions[asset] = 0.0

        return positions

    def _ensure_zero_cost(self, positions: Dict[str, float]) -> Dict[str, float]:
        """Ensure portfolio is dollar neutral"""
        long_exposure = sum(w for w in positions.values() if w > 0)
        short_exposure = abs(sum(w for w in positions.values() if w < 0))

        if long_exposure > 0 and short_exposure > 0:
            # Scale to achieve dollar neutrality
            scale = min(1.0, short_exposure / long_exposure)

            adjusted_positions = {}
            for asset, weight in positions.items():
                if weight > 0:
                    adjusted_positions[asset] = weight * scale
                else:
                    adjusted_positions[asset] = weight

            return adjusted_positions

        return positions

    def _calculate_turnover(self, old_positions: Dict[str, float], new_positions: Dict[str, float]) -> Dict:
        """Calculate portfolio turnover and estimated transaction costs"""
        all_assets = set(old_positions.keys()).union(set(new_positions.keys()))

        turnover = 0.0
        for asset in all_assets:
            old_weight = old_positions.get(asset, 0.0)
            new_weight = new_positions.get(asset, 0.0)
            turnover += abs(new_weight - old_weight)

        # One-way turnover (half of total turnover)
        one_way_turnover = turnover / 2

        # Estimated transaction cost
        estimated_cost = one_way_turnover * (self.transaction_cost_bps / 10000)

        # Store for history
        self.turnover_history.append(
            {
                "turnover": turnover,
                "one_way_turnover": one_way_turnover,
                "estimated_cost": estimated_cost,
            }
        )

        return {
            "turnover": turnover,
            "one_way_turnover": one_way_turnover,
            "estimated_cost": estimated_cost,
        }

    def _is_rebalance_day(self, price_data: pd.DataFrame) -> bool:
        """Check if today is a rebalance day based on holding period"""
        if len(self.rebalance_dates) == 0:
            return True

        # Simple implementation: rebalance every holding_period days
        if isinstance(price_data.index, pd.DatetimeIndex):
            days_since_rebalance = (price_data.index[-1] - self.rebalance_dates[-1]).days
        else:
            days_since_rebalance = len(price_data) - self.rebalance_dates[-1]

        return days_since_rebalance >= self.holding_period

    # ============================================================================
    # PERFORMANCE ANALYSIS & OPTIMIZATION
    # ============================================================================

    def backtest(
        self,
        price_data: pd.DataFrame,
        market_index: Optional[pd.Series] = None,
        initial_capital: float = 1000000,
    ) -> pd.DataFrame:
        """
        Run backtest on historical data

        Args:
            price_data: Historical price data
            market_index: Market index for crash protection
            initial_capital: Initial capital

        Returns:
            DataFrame with backtest results
        """
        results = []
        capital = initial_capital
        positions = {}

        # Run strategy
        for i in range(self.formation_period + self.skip_period, len(price_data)):
            current_date = price_data.index[i]
            historical_data = price_data.iloc[: i + 1]

            # Market data for crash protection
            if market_index is not None and len(market_index) > i:
                market_slice = market_index.iloc[: i + 1]
            else:
                market_slice = None

            # Check if rebalance day
            days_since_start = i - (self.formation_period + self.skip_period)
            rebalance = days_since_start % self.holding_period == 0

            # Generate signals
            signal_result = self.generate_signals(historical_data, market_slice, positions, rebalance)

            # Update positions
            positions = signal_result["signals"]

            # Calculate daily P&L
            daily_returns = {}
            portfolio_return = 0.0

            for asset, weight in positions.items():
                if i > 0 and asset in price_data.columns:
                    # Calculate asset return
                    asset_return = price_data[asset].iloc[i] / price_data[asset].iloc[i - 1] - 1
                    position_return = weight * asset_return * capital
                    daily_returns[asset] = position_return
                    portfolio_return += position_return

            # Update capital
            capital += portfolio_return

            # Store results
            results.append(
                {
                    "date": current_date,
                    "capital": capital,
                    "return": portfolio_return / capital if capital > 0 else 0,
                    "num_positions": len([v for v in positions.values() if abs(v) > 0.001]),
                    "gross_exposure": sum(abs(v) for v in positions.values()),
                    "net_exposure": sum(positions.values()),
                    "turnover": signal_result["metadata"].get("turnover", 0),
                    "action": signal_result["action"],
                }
            )

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

        # Information ratio (vs market if available)
        information_ratio = 0.0

        # Turnover metrics
        avg_turnover = results["turnover"].mean()
        avg_annual_turnover = avg_turnover * (252 / self.holding_period)

        # Win rate
        win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0

        # Update metrics
        self.performance_metrics.update(
            {
                "total_return": total_return,
                "annual_return": annual_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_drawdown,
                "information_ratio": information_ratio,
                "avg_annual_turnover": avg_annual_turnover,
                "win_rate": win_rate,
                "calmar_ratio": (annual_return / abs(max_drawdown) if max_drawdown < 0 else 0),
            }
        )

    def optimize_parameters(self, price_data: pd.DataFrame, parameter_grid: Optional[Dict] = None) -> Dict:
        """
        Optimize strategy parameters using walk-forward optimization

        Args:
            price_data: Historical price data
            parameter_grid: Grid of parameters to test

        Returns:
            Optimal parameters
        """
        if parameter_grid is None:
            parameter_grid = {
                "formation_period": [63, 126, 252],
                "skip_period": [0, 5, 21],
                "top_quantile": [0.2, 0.3, 0.4],
                "bottom_quantile": [0.2, 0.3, 0.4],
            }

        # Walk-forward optimization
        # Split data into in-sample and out-of-sample
        split_idx = int(len(price_data) * 0.7)
        train_data = price_data.iloc[:split_idx]
        test_data = price_data.iloc[split_idx:]

        best_params = {}
        best_sharpe = -np.inf

        # Grid search
        from itertools import product

        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())

        for params in product(*param_values):
            param_dict = dict(zip(param_names, params))

            # Test on training data
            temp_strategy = CrossSectionalMomentumStrategy(universe=self.universe, **param_dict)

            results = temp_strategy.backtest(train_data)

            if len(results) > 0:
                sharpe = temp_strategy.performance_metrics["sharpe_ratio"]

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = param_dict

        # Validate on test data
        if best_params:
            test_strategy = CrossSectionalMomentumStrategy(universe=self.universe, **best_params)
            test_results = test_strategy.backtest(test_data)

            logger.info(f"TEST RESULTS SIZE: {test_results.size}")
        return {
            "best_params": best_params,
            "best_sharpe": best_sharpe,
            "test_sharpe": (test_strategy.performance_metrics["sharpe_ratio"] if best_params else 0),
        }

    def generate_report(self) -> Dict:
        """
        Generate comprehensive strategy report
        """
        report = {
            "parameters": {
                "universe_size": len(self.universe),
                "formation_period": self.formation_period,
                "skip_period": self.skip_period,
                "holding_period": self.holding_period,
                "top_quantile": self.top_quantile,
                "bottom_quantile": self.bottom_quantile,
                "sector_neutral": self.sector_mapping is not None,
                "volatility_adjusted": self.volatility_adjustment,
                "crash_protection": self.momentum_crash_protection,
            },
            "performance": self.performance_metrics,
            "portfolio_characteristics": {
                "avg_gross_exposure": (
                    np.mean([r.get("gross_exposure", 0) for r in self.momentum_scores_history]) if self.momentum_scores_history else 0
                ),
                "avg_net_exposure": (
                    np.mean([r.get("net_exposure", 0) for r in self.momentum_scores_history]) if self.momentum_scores_history else 0
                ),
                "num_rebalances": len(self.rebalance_dates),
            },
            "recent_signals": (self.momentum_scores_history[-5:] if len(self.momentum_scores_history) > 5 else self.momentum_scores_history),
        }

        return report


# ============================================================================
# SPECIALIZED MOMENTUM VARIANTS
# ============================================================================


class FactorMomentumStrategy(CrossSectionalMomentumStrategy):
    """
    Factor Momentum Strategy

    Applies cross-sectional momentum to factor portfolios
    rather than individual stocks
    """

    def __init__(self, factor_model, **kwargs):
        super().__init__(**kwargs)
        self.factor_model = factor_model

    def generate_factor_momentum_signals(self, stock_data: pd.DataFrame) -> Dict:
        """
        Generate momentum signals for factors
        """
        # Extract factor returns
        factor_returns = self.factor_model.calculate_factor_returns(stock_data)

        # Apply cross-sectional momentum to factors
        factor_signals = self.generate_signals(factor_returns)

        # Map factor signals back to stocks
        stock_signals = self._map_factors_to_stocks(factor_signals, stock_data)

        return stock_signals

    def _map_factors_to_stocks(self, factor_signals: Dict, stock_data: pd.DataFrame) -> Dict:
        """
        Map factor momentum signals to individual stocks
        """
        # Get factor exposures
        factor_exposures = self.factor_model.get_factor_exposures(stock_data)

        # Weight stocks by their factor exposures and factor momentum
        stock_weights = {}

        for stock in stock_data.columns:
            weight = 0.0
            for factor, factor_signal in factor_signals["signals"].items():
                exposure = factor_exposures.get(stock, {}).get(factor, 0.0)
                weight += exposure * factor_signal

            stock_weights[stock] = weight

        # Normalize
        total_weight = sum(abs(w) for w in stock_weights.values())
        if total_weight > 0:
            stock_weights = {k: v / total_weight for k, v in stock_weights.items()}

        return stock_weights


class AdaptiveMomentumStrategy(CrossSectionalMomentumStrategy):
    """
    Adaptive Momentum Strategy

    Dynamically adjusts momentum parameters based on market conditions
    """

    def __init__(self, regime_detector, **kwargs):
        super().__init__(**kwargs)
        self.regime_detector = regime_detector

    def detect_market_regime(self, market_data: pd.Series) -> str:
        """
        Detect current market regime
        """
        return self.regime_detector.detect_regime(market_data)

    def adapt_parameters(self, regime: str):
        """
        Adjust momentum parameters based on regime
        """
        regime_params = {
            "trending": {
                "formation_period": 126,  # Shorter in trends
                "skip_period": 0,  # No skip in trends
                "top_quantile": 0.2,  # More concentrated
                "holding_period": 10,  # More frequent rebalancing
            },
            "mean_reverting": {
                "formation_period": 252,  # Longer lookback
                "skip_period": 21,  # Skip to avoid reversal
                "top_quantile": 0.3,  # Standard
                "holding_period": 21,  # Monthly
            },
            "high_vol": {
                "formation_period": 63,  # Shorter in high vol
                "skip_period": 5,  # Small skip
                "top_quantile": 0.4,  # Broader selection
                "holding_period": 5,  # Very frequent
            },
            "low_vol": {
                "formation_period": 252,  # Full year
                "skip_period": 21,  # Standard skip
                "top_quantile": 0.3,  # Standard
                "holding_period": 21,  # Monthly
            },
        }

        if regime in regime_params:
            for param, value in regime_params[regime].items():
                setattr(self, param, value)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=== Enhanced Cross-Sectional Momentum Strategy ===")

    # Create sample universe
    np.random.seed(42)
    n_assets = 50
    dates = pd.date_range("2010-01-01", periods=1000, freq="D")

    # Simulate asset prices with momentum factor
    asset_prices = {}
    for i in range(n_assets):
        # Base returns with momentum persistence
        base_returns = np.random.randn(1000) * 0.01

        # Add momentum factor
        momentum_factor = np.random.randn(1000) * 0.005
        momentum_factor = np.convolve(momentum_factor, np.ones(30) / 30, mode="same")

        # Combine
        total_returns = base_returns + momentum_factor
        prices = 100 * np.exp(np.cumsum(total_returns))
        asset_prices[f"ASSET_{i:02d}"] = prices

    price_data = pd.DataFrame(asset_prices, index=dates)

    # Create sector mapping
    sectors = ["TECH", "FINANCIAL", "HEALTHCARE", "INDUSTRIAL", "CONSUMER"]
    sector_mapping = {f"ASSET_{i:02d}": sectors[i % len(sectors)] for i in range(n_assets)}

    print(f"Created universe of {n_assets} assets across {len(sectors)} sectors")

    # Initialize strategy
    momentum_strategy = CrossSectionalMomentumStrategy(
        universe=list(asset_prices.keys()),
        formation_period=252,
        skip_period=21,
        holding_period=21,
        top_quantile=0.3,
        bottom_quantile=0.3,
        sector_mapping=sector_mapping,
        volatility_adjustment=True,
        momentum_crash_protection=True,
        max_position_size=0.05,
        transaction_cost_bps=5.0,
    )

    # Run backtest
    print("\nRunning backtest...")
    results = momentum_strategy.backtest(price_data)

    print("\nPerformance Metrics:")
    for metric, value in momentum_strategy.performance_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Generate report
    report = momentum_strategy.generate_report()
    print("\nStrategy Characteristics:")
    print(f"  Average gross exposure: {report['portfolio_characteristics']['avg_gross_exposure']:.2f}")
    print(f"  Number of rebalances: {report['portfolio_characteristics']['num_rebalances']}")

    # Test parameter optimization
    print("\nOptimizing parameters...")
    optimal_params = momentum_strategy.optimize_parameters(
        price_data.iloc[:700],  # Use first 70% for optimization
        parameter_grid={
            "formation_period": [126, 252],
            "skip_period": [5, 21],
            "top_quantile": [0.2, 0.3],
        },
    )

    print(f"Optimal parameters: {optimal_params}")
