import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from ....analytics.market_regime_detector import MarketRegimeDetector
from ....core import fetch_financials, fetch_stock_data
from ....core.data.fetchers.fundamentals_fetcher import FundamentalsFetcher
from ....utils.helpers import SECTOR_ETFS, SECTOR_MAPPINGS
from ... import BaseStrategy

logger = logging.getLogger(__name__)


class SectorRotationStrategy(BaseStrategy):
    """
    Advanced sector rotation strategy that:
    1. Predicts top-performing sectors using macro + fundamental data
    2. Selects best stocks within predicted sectors
    3. Dynamically switches between models based on regime
    4. Provides explainable predictions with confidence scores
    """

    def __init__(
        self,
        name: str = "sector_rotation",
        lookback_years: int = 5,
        forecast_horizon_days: int = 60,
        top_sectors: int = 3,
        stocks_per_sector: int = 5,
        model_type: str = "random_forest",  # random_forest, gradient_boosting, ensemble
        use_regime_detection: bool = True,
        use_sector_neutral: bool = True,
        confidence_threshold: float = 0.6,
        rebalance_frequency_days: int = 30,
        max_sector_exposure: float = 0.3,
        enable_ml: bool = True,
        params: Dict = None,
    ):
        super().__init__(name, params or {})

        # Core parameters
        self.lookback_years = lookback_years
        self.forecast_horizon = forecast_horizon_days
        self.top_sectors = top_sectors
        self.stocks_per_sector = stocks_per_sector
        self.model_type = model_type
        self.use_regime_detection = use_regime_detection
        self.use_sector_neutral = use_sector_neutral
        self.confidence_threshold = confidence_threshold
        self.rebalance_frequency = rebalance_frequency_days
        self.max_sector_exposure = max_sector_exposure
        self.enable_ml = enable_ml

        # Initialize components
        self.fundamental_provider = FundamentalsFetcher()
        self.regime_detector = MarketRegimeDetector(name=f"{name}_regime") if use_regime_detection else None

        # Model storage - start with simple Random Forest, add complexity gradually
        self.sector_models = {}  # sector -> model
        self.stock_models = {}  # sector -> {symbol -> model}
        self.scalers = {}  # sector -> scaler
        self.feature_importance = {}  # sector -> feature importance

        # Performance tracking
        self.model_performance = defaultdict(list)
        self.prediction_history = []
        self.last_rebalance = None
        self.current_positions = {}

        # Feature engineering cache
        self.feature_cache = {}

        # Sector universe
        self.SECTORS = SECTOR_MAPPINGS

        logger.info(f"SectorRotationStrategy initialized with model_type={model_type}")

    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """
        Generate trading signal - main entry point
        """
        try:
            # Check if rebalance is needed
            current_time = data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else datetime.now()

            if self._should_rebalance(current_time):
                logger.info("Rebalancing triggered")
                self.current_positions = self._generate_positions(data)
                self.last_rebalance = current_time

            # Generate signal based on current positions
            if self.current_positions:
                # For simplicity in single-asset context, return hold
                # In multi-asset context, this would return allocations
                return {
                    "signal": 0,  # Hold - positions are managed separately
                    "position_size": 1.0,
                    "metadata": {
                        "strategy": "sector_rotation",
                        "positions": self.current_positions,
                        "last_rebalance": self.last_rebalance.isoformat() if self.last_rebalance else None,
                        "regime": self._get_current_regime(data) if self.regime_detector else None,
                    },
                }
            else:
                return {"signal": 0, "position_size": 0.0, "metadata": {}}

        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return {"signal": 0, "position_size": 0.0, "metadata": {"error": str(e)}}

    async def generate_signals_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate multiple signals at once (for multi-asset backtesting)
        """
        if not self._should_rebalance(data.index[-1]):
            # Return existing allocations
            if self.current_positions:
                return pd.Series(self.current_positions)
            return pd.Series()

        # Rebalance - generate new allocations
        positions = await self._generate_positions(data)
        self.last_rebalance = data.index[-1]
        self.current_positions = positions

        return pd.Series(positions)

    # ==================== PHASE 1: SECTOR PREDICTION ====================

    async def _predict_top_sectors(self, data: pd.DataFrame) -> List[Dict]:
        """
        Phase 1: Predict top-performing sectors
        Start with Random Forest, gradually add complexity
        """
        if not self.enable_ml:
            return self._baseline_sector_prediction()

        # Extract macro features
        macro_features = await self._extract_macro_features(data)

        if macro_features.empty:
            logger.warning("No macro features available, using baseline")
            return self._baseline_sector_prediction()

        # Get current regime if enabled
        regime = None
        if self.regime_detector and len(data) > 50:
            regime_df = self.regime_detector.markov_chain.detect_regimes(data)
            regime = regime_df["regime"].iloc[-1] if not regime_df.empty else None

        # Predict each sector
        sector_predictions = []

        for sector in self.SECTORS.keys():
            try:
                # Get or train sector model
                if sector not in self.sector_models:
                    await self._train_sector_model(sector, data)

                if sector in self.sector_models:
                    model = self.sector_models[sector]
                    scaler = self.scalers.get(sector)

                    # Prepare features
                    latest_features = macro_features.iloc[-1:].values
                    if scaler:
                        latest_features = scaler.transform(latest_features)

                    # Make prediction
                    expected_return = model.predict(latest_features)[0]

                    # Calculate confidence
                    confidence = self._calculate_prediction_confidence(model, latest_features, sector, regime)

                    # Get feature importance if available
                    top_factors = self._get_top_factors(sector)

                    sector_predictions.append(
                        {
                            "sector": sector,
                            "expected_return": float(expected_return),
                            "confidence": confidence,
                            "top_factors": top_factors,
                            "regime": regime,
                            "model_type": self.model_type,
                        }
                    )

            except Exception as e:
                logger.error(f"Error predicting sector {sector}: {e}")
                continue

        # Sort by expected return * confidence
        for p in sector_predictions:
            p["score"] = p["expected_return"] * p["confidence"]

        sector_predictions.sort(key=lambda x: x["score"], reverse=True)

        # Store prediction history
        self.prediction_history.append(
            {
                "date": data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else datetime.now(),
                "predictions": sector_predictions.copy(),
                "regime": regime,
            }
        )

        return sector_predictions[: self.top_sectors]

    async def _train_sector_model(self, sector: str, data: pd.DataFrame):
        """
        Train sector prediction model - start with Random Forest
        Gradually add complexity based on model_type
        """
        logger.info(f"Training model for sector: {sector} with type={self.model_type}")

        # Prepare training data
        X, y = await self._prepare_sector_training_data(sector, data)

        if X is None or y is None or len(X) < 20:
            logger.warning(f"Insufficient training data for sector {sector}")
            return

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Select model based on complexity level
        if self.model_type == "random_forest":
            # Level 1: Simple Random Forest
            model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)

        elif self.model_type == "gradient_boosting":
            # Level 2: Gradient Boosting (more complex)
            model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8, random_state=42)

        elif self.model_type == "ensemble":
            # Level 3: Ensemble of multiple models (most complex)
            # We'll use Random Forest as primary, but store multiple
            model = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)

            # Also train a gradient boosting model for ensemble
            gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
            gb_model.fit(X_scaled, y)
            self.sector_models[f"{sector}_gb"] = gb_model

        else:
            # Default to Random Forest
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

        # Train model
        model.fit(X_scaled, y)

        # Store model and scaler
        self.sector_models[sector] = model
        self.scalers[sector] = scaler

        # Calculate and store feature importance
        if hasattr(model, "feature_importances_"):
            importance = dict(zip(X.columns, model.feature_importances_))
            self.feature_importance[sector] = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]  # Top 10 features

        # Cross-validation score
        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []

        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            cv_scores.append(score)

        self.model_performance[sector].append(np.mean(cv_scores))
        logger.info(f"Sector {sector} CV R² = {np.mean(cv_scores):.4f}")

    async def _prepare_sector_training_data(self, sector: str, data: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Prepare training data for sector prediction
        """
        try:
            # Get macro data (you'll need to implement this)
            macro_data = await self._fetch_historical_macro()

            if macro_data.empty:
                return None, None

            # Calculate sector returns
            sector_returns = await self._calculate_sector_returns(sector, macro_data.index)

            if sector_returns.empty:
                return None, None

            # Align dates
            common_dates = macro_data.index.intersection(sector_returns.index)
            if len(common_dates) < 20:
                return None, None

            X = macro_data.loc[common_dates]
            y = sector_returns.loc[common_dates]

            # Create lagged features for additional predictive power
            for lag in [1, 3, 6]:
                lagged = X.shift(lag)
                lagged.columns = [f"{col}_lag{lag}" for col in lagged.columns]
                X = pd.concat([X, lagged], axis=1)

            # Drop NaN rows
            combined = pd.concat([X, y], axis=1).dropna()
            X = combined[list(X.columns)]
            y = combined[y.name]

            return X, y

        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None, None

    async def _calculate_sector_returns(self, sector: str, dates: pd.DatetimeIndex) -> pd.Series:
        """
        Calculate historical sector returns
        """
        stocks = self.SECTORS.get(sector, [])
        if not stocks:
            return pd.Series()

        # Get top 5 stocks for efficiency
        top_stocks = stocks[:5]

        all_returns = []
        for symbol in top_stocks:
            try:
                data = await fetch_stock_data(symbol, period=f"{self.lookback_years+1}y", interval="1mo")

                if data is not None and not data.empty:
                    # Calculate forward returns
                    returns = data["Close"].pct_change(self.forecast_horizon).shift(-self.forecast_horizon)
                    returns.name = symbol
                    all_returns.append(returns)

            except Exception as e:
                logger.debug(f"Error fetching {symbol}: {e}")
                continue

        if not all_returns:
            return pd.Series()

        # Combine and average
        returns_df = pd.concat(all_returns, axis=1)
        sector_returns = returns_df.mean(axis=1)
        sector_returns.name = f"{sector}_returns"

        return sector_returns

    async def _fetch_historical_macro(self) -> pd.DataFrame:
        """
        Fetch historical macro data from sector ETF prices and VIX.
        Uses real market data via fetch_stock_data.
        """
        try:
            macro_data = pd.DataFrame()

            # Fetch sector ETF monthly returns as macro proxies
            for sector, etf in SECTOR_ETFS.items():
                try:
                    data = await fetch_stock_data(etf, period=f"{self.lookback_years}y", interval="1mo")
                    if data is not None and not data.empty:
                        macro_data[f"{sector}_return"] = data["Close"].pct_change()
                except Exception as e:
                    logger.debug(f"Failed to fetch ETF {etf} for {sector}: {e}")
                    continue

            # Fetch VIX as volatility proxy
            try:
                vix = await fetch_stock_data("^VIX", period=f"{self.lookback_years}y", interval="1mo")
                if vix is not None and not vix.empty:
                    macro_data["vix"] = vix["Close"]
            except Exception:
                pass

            # Fetch SPY for market-level features
            try:
                spy = await fetch_stock_data("SPY", period=f"{self.lookback_years}y", interval="1mo")
                if spy is not None and not spy.empty:
                    macro_data["market_return"] = spy["Close"].pct_change()
                    macro_data["market_momentum_3m"] = spy["Close"].pct_change(3)
                    macro_data["market_momentum_6m"] = spy["Close"].pct_change(6)
            except Exception:
                pass

            # Fetch Treasury proxy (TLT) for interest rate environment
            try:
                tlt = await fetch_stock_data("TLT", period=f"{self.lookback_years}y", interval="1mo")
                if tlt is not None and not tlt.empty:
                    macro_data["bond_return"] = tlt["Close"].pct_change()
            except Exception:
                pass

            return macro_data.dropna() if not macro_data.empty else pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to fetch macro data: {e}")
            return pd.DataFrame()

    def _baseline_sector_prediction(self) -> List[Dict]:
        """
        Baseline sector prediction when ML is disabled
        Uses momentum and simple trends
        """
        # Simple momentum-based sector ranking
        sectors = list(self.SECTORS.keys())

        # Simulate some predictable patterns
        predictions = []
        for i, sector in enumerate(sectors):
            # Cyclical pattern based on sector index
            expected_return = 0.05 * np.sin(i * 0.5) + 0.02

            predictions.append(
                {
                    "sector": sector,
                    "expected_return": expected_return,
                    "confidence": 0.5,
                    "top_factors": [{"factor": "momentum", "importance": 1.0}],
                    "regime": None,
                    "model_type": "baseline",
                }
            )

        predictions.sort(key=lambda x: x["expected_return"], reverse=True)
        return predictions[: self.top_sectors]

    # ==================== PHASE 2: STOCK SELECTION ====================

    async def _select_top_stocks(self, sector: str, n_stocks: int, data: pd.DataFrame) -> List[Dict]:
        """
        Phase 2: Select best stocks within a sector
        """
        stocks = self.SECTORS.get(sector, [])
        if not stocks:
            return []

        # Get fundamental data for stocks
        fundamentals = await self._fetch_sector_fundamentals(sector, stocks)

        if fundamentals.empty:
            # Fallback to price momentum
            return await self._momentum_based_selection(sector, stocks, n_stocks, data)

        # Score each stock
        stock_scores = []

        for symbol in stocks:
            try:
                # Get stock data
                stock_data = data[data["symbol"] == symbol] if "symbol" in data.columns else data

                if stock_data.empty:
                    continue

                # Get fundamental factors for this stock
                stock_factors = fundamentals[fundamentals["symbol"] == symbol] if "symbol" in fundamentals.columns else fundamentals

                if stock_factors.empty:
                    continue

                # Calculate composite score
                score = self._calculate_stock_score(symbol, stock_data, stock_factors.iloc[-1] if len(stock_factors) > 0 else None)

                # Get regime context
                regime_context = None
                if self.regime_detector and len(data) > 50:
                    regime_df = self.regime_detector.detect_regimes(data)
                    regime_context = regime_df["regime"].iloc[-1] if not regime_df.empty else None

                # Apply sector-neutral adjustment if enabled
                if self.use_sector_neutral and regime_context:
                    score = self._sector_neutral_adjustment(score, sector, regime_context)

                stock_scores.append(
                    {
                        "symbol": symbol,
                        "score": score,
                        "expected_return": score * 0.1,  # Simplified mapping
                        "confidence": min(abs(score) * 2, 0.95),
                        "factors": self._get_stock_factors(symbol),
                    }
                )

            except Exception as e:
                logger.error(f"Error scoring {symbol}: {e}")
                continue

        # Sort by score
        stock_scores.sort(key=lambda x: x["score"], reverse=True)

        return stock_scores[:n_stocks]

    async def _fetch_sector_fundamentals(self, sector: str, stocks: List[str]) -> pd.DataFrame:
        """
        Fetch fundamental data for all stocks in a sector
        """
        all_fundamentals = []

        for symbol in stocks[:10]:  # Limit for performance
            try:
                financials = await fetch_financials(symbol)

                if financials:
                    # Extract key metrics
                    fundamentals = {
                        "symbol": symbol,
                        "sector": sector,
                        "pe_ratio": financials.get("trailingPE", 0),
                        "forward_pe": financials.get("forwardPE", 0),
                        "peg_ratio": financials.get("pegRatio", 0),
                        "pb_ratio": financials.get("priceToBook", 0),
                        "debt_to_equity": financials.get("debtToEquity", 0),
                        "roe": financials.get("returnOnEquity", 0),
                        "operating_margin": financials.get("operatingMargins", 0),
                        "revenue_growth": financials.get("revenueGrowth", 0),
                        "eps_growth": financials.get("earningsGrowth", 0),
                    }
                    all_fundamentals.append(fundamentals)

            except Exception as e:
                logger.debug(f"Error fetching fundamentals for {symbol}: {e}")
                continue

        if not all_fundamentals:
            return pd.DataFrame()

        return pd.DataFrame(all_fundamentals)

    def _calculate_stock_score(self, symbol: str, price_data: pd.DataFrame, fundamentals: Optional[pd.Series]) -> float:
        """
        Calculate composite stock score
        Combines multiple factors
        """
        score = 0.0

        # Factor 1: Price momentum (30% weight)
        if len(price_data) > 20:
            returns_1m = price_data["Close"].pct_change(21).iloc[-1] if len(price_data) > 21 else 0
            returns_3m = price_data["Close"].pct_change(63).iloc[-1] if len(price_data) > 63 else 0
            momentum_score = returns_1m * 0.6 + returns_3m * 0.4
            score += 0.3 * np.clip(momentum_score, -0.5, 0.5) / 0.5

        # Factor 2: Volatility (20% weight) - lower is better
        if len(price_data) > 20:
            volatility = price_data["Close"].pct_change().std() * np.sqrt(252)
            vol_score = -np.clip(volatility - 0.2, -0.3, 0.3) / 0.3
            score += 0.2 * vol_score

        # Factor 3: Fundamentals (50% weight)
        if fundamentals is not None and not fundamentals.empty:
            # P/E score (lower is better, but not negative)
            pe = fundamentals.get("pe_ratio", 20)
            if pe > 0:
                pe_score = -np.clip((pe - 20) / 20, -1, 1)
                score += 0.1 * pe_score

            # Growth score
            rev_growth = fundamentals.get("revenue_growth", 0)
            eps_growth = fundamentals.get("eps_growth", 0)
            growth_score = np.clip((rev_growth + eps_growth) / 0.2, -1, 1)
            score += 0.2 * growth_score

            # Profitability score
            roe = fundamentals.get("roe", 0.1)
            margin = fundamentals.get("operating_margin", 0.1)
            profit_score = np.clip((roe + margin) / 0.3, -1, 1)
            score += 0.2 * profit_score

        return np.clip(score, -1, 1)

    async def _momentum_based_selection(self, sector: str, stocks: List[str], n_stocks: int, data: pd.DataFrame) -> List[Dict]:
        """
        Fallback: momentum-based stock selection
        """
        stock_momentum = []

        for symbol in stocks:
            try:
                stock_data = data[data["symbol"] == symbol] if "symbol" in data.columns else data

                if len(stock_data) > 20:
                    returns = stock_data["Close"].pct_change(21).iloc[-1]
                    stock_momentum.append(
                        {
                            "symbol": symbol,
                            "score": returns,
                            "expected_return": returns,
                            "confidence": 0.5,
                            "factors": [{"factor": "momentum", "importance": 1.0}],
                        }
                    )

            except Exception:
                continue

        stock_momentum.sort(key=lambda x: x["score"], reverse=True)
        return stock_momentum[:n_stocks]

    def _sector_neutral_adjustment(self, score: float, sector: str, regime: str) -> float:
        """
        Apply sector-neutral adjustments based on regime
        """
        # Different regimes favor different adjustments
        regime_multipliers = {
            "bull": 1.2,  # Amplify scores in bull markets
            "neutral": 1.0,  # Neutral
            "bear": 0.8,  # Reduce scores in bear markets
        }

        multiplier = regime_multipliers.get(regime, 1.0)

        # Sector-specific adjustments
        sector_biases = {
            "Technology": 1.1 if regime == "bull" else 0.9,
            "Utilities": 1.1 if regime == "bear" else 0.9,
            "Consumer Defensive": 1.1 if regime == "bear" else 0.9,
            "Financials": 1.1 if regime == "bull" else 0.9,
        }

        sector_multiplier = sector_biases.get(sector, 1.0)

        return score * multiplier * sector_multiplier

    # ==================== PHASE 3: POSITION GENERATION ====================

    async def _generate_positions(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Generate position allocations
        """
        # Phase 1: Predict top sectors
        top_sectors = await self._predict_top_sectors(data)

        if not top_sectors:
            logger.warning("No sector predictions available")
            return {}

        logger.info(f"Top sectors: {[s['sector'] for s in top_sectors]}")

        # Phase 2: Select stocks from top sectors
        all_positions = {}
        sector_weights = {}

        # Calculate sector weights based on confidence
        total_confidence = sum(s["confidence"] for s in top_sectors)

        for sector_pred in top_sectors:
            sector = sector_pred["sector"]
            confidence = sector_pred["confidence"]

            # Base sector weight
            sector_weight = (confidence / total_confidence) if total_confidence > 0 else 1.0 / len(top_sectors)
            sector_weight = min(sector_weight, self.max_sector_exposure)

            # Select stocks in this sector
            stocks = await self._select_top_stocks(sector, self.stocks_per_sector, data)

            if stocks:
                # Equal weight within sector
                stock_weight = sector_weight / len(stocks)

                for stock in stocks:
                    symbol = stock["symbol"]

                    # Apply stock-level confidence adjustment
                    final_weight = stock_weight * stock["confidence"]

                    all_positions[symbol] = final_weight

                    # Store metadata
                    if symbol not in self.current_positions:
                        self.current_positions[symbol] = {}

                    self.current_positions[symbol].update(
                        {
                            "sector": sector,
                            "sector_confidence": confidence,
                            "stock_confidence": stock["confidence"],
                            "expected_return": stock["expected_return"],
                            "allocation": final_weight,
                        }
                    )

                sector_weights[sector] = sector_weight

        # Normalize to ensure sum <= 1.0
        total_weight = sum(all_positions.values())
        if total_weight > 1.0:
            for symbol in all_positions:
                all_positions[symbol] /= total_weight

        logger.info(f"Generated {len(all_positions)} positions with total weight {sum(all_positions.values()):.2f}")

        return all_positions

    # ==================== UTILITY FUNCTIONS ====================

    async def _extract_macro_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract macro features from data
        """
        # Cache key based on data end date
        cache_key = data.index[-1].strftime("%Y-%m-%d") if hasattr(data.index, "strftime") else "latest"

        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        # Fetch macro data
        macro_data = await self._fetch_historical_macro()

        if macro_data.empty:
            return pd.DataFrame()

        # Get latest macro data
        latest = macro_data.iloc[-1:].copy()

        # Add technical indicators from price data
        if not data.empty:
            # Volatility
            latest["vix"] = data["Close"].pct_change().std() * np.sqrt(252)

            # Trend
            sma_50 = data["Close"].rolling(50).mean()
            sma_200 = data["Close"].rolling(200).mean()
            latest["trend_strength"] = (sma_50.iloc[-1] / sma_200.iloc[-1] - 1) if not pd.isna(sma_50.iloc[-1]) else 0

            # Momentum
            returns_3m = data["Close"].pct_change(63).iloc[-1] if len(data) > 63 else 0
            latest["market_momentum"] = returns_3m

        self.feature_cache[cache_key] = latest
        return latest

    def _calculate_prediction_confidence(self, model: Any, features: np.ndarray, sector: str, regime: Optional[str]) -> float:
        """
        Calculate confidence in prediction
        """
        confidence = 0.7  # Base confidence

        # Factor 1: Model type confidence
        if self.model_type == "random_forest":
            # Random Forest: use tree variance
            if hasattr(model, "estimators_"):
                tree_preds = np.array([tree.predict(features)[0] for tree in model.estimators_])
                variance_confidence = 1.0 - (tree_preds.std() / (abs(tree_preds.mean()) + 1e-6))
                variance_confidence = np.clip(variance_confidence, 0.5, 0.95)
                confidence *= variance_confidence

        elif self.model_type == "ensemble":
            # Ensemble: check agreement between models
            if f"{sector}_gb" in self.sector_models:
                rf_pred = model.predict(features)[0]
                gb_pred = self.sector_models[f"{sector}_gb"].predict(features)[0]

                # Higher confidence if models agree
                agreement = 1.0 - abs(rf_pred - gb_pred) / (abs(rf_pred) + abs(gb_pred) + 1e-6)
                confidence *= agreement

        # Factor 2: Regime confidence
        if regime and self.regime_detector:
            regime_history = self.regime_detector.regime_history
            if regime_history:
                # Higher confidence if regime is stable
                recent_regimes = [r["regime"] for r in regime_history[-10:]]
                stability = recent_regimes.count(regime) / len(recent_regimes)
                confidence *= 0.8 + 0.2 * stability

        # Factor 3: Historical performance
        if sector in self.model_performance and self.model_performance[sector]:
            recent_perf = np.mean(self.model_performance[sector][-3:])
            confidence *= 0.7 + 0.3 * max(0, recent_perf)

        return float(np.clip(confidence, 0.3, 0.95))

    def _get_top_factors(self, sector: str) -> List[Dict]:
        """
        Get top contributing factors for a sector
        """
        if sector in self.feature_importance:
            return [{"factor": factor, "importance": importance} for factor, importance in self.feature_importance[sector]]

        # Default factors
        return [
            {"factor": "gdp_growth", "importance": 0.2},
            {"factor": "interest_rates", "importance": 0.15},
            {"factor": "inflation", "importance": 0.15},
            {"factor": "market_momentum", "importance": 0.25},
            {"factor": "volatility", "importance": 0.25},
        ]

    def _get_stock_factors(self, symbol: str) -> List[Dict]:
        """
        Get key factors for a stock
        """
        # Simplified - in production, use SHAP
        return [
            {"factor": "momentum", "importance": 0.3},
            {"factor": "valuation", "importance": 0.25},
            {"factor": "growth", "importance": 0.25},
            {"factor": "quality", "importance": 0.2},
        ]

    def _get_current_regime(self, data: pd.DataFrame) -> Optional[str]:
        """
        Get current market regime
        """
        if not self.regime_detector or len(data) < 50:
            return None

        try:
            regime_df = self.regime_detector.detect_regimes(data)
            return regime_df["regime"].iloc[-1] if not regime_df.empty else None
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return None

    def _should_rebalance(self, current_time: datetime) -> bool:
        """
        Check if rebalancing is needed
        """
        if self.last_rebalance is None:
            return True

        days_since = (current_time - self.last_rebalance).days
        return days_since >= self.rebalance_frequency

    def get_prediction_history(self) -> List[Dict]:
        """
        Get historical predictions for analysis
        """
        return self.prediction_history

    def get_model_performance(self) -> Dict:
        """
        Get model performance metrics
        """
        return {
            "sector_models": {
                sector: {
                    "performance": self.model_performance[sector],
                    "avg_performance": np.mean(self.model_performance[sector]) if self.model_performance[sector] else 0,
                    "features": self.feature_importance.get(sector, []),
                }
                for sector in self.model_performance
            },
            "current_regime": self._get_current_regime(pd.DataFrame()) if self.regime_detector else None,
            "last_rebalance": self.last_rebalance.isoformat() if self.last_rebalance else None,
        }

    def reset(self):
        """Reset strategy state"""
        self.sector_models = {}
        self.stock_models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = defaultdict(list)
        self.prediction_history = []
        self.last_rebalance = None
        self.current_positions = {}
        self.feature_cache = {}

        if self.regime_detector:
            self.regime_detector.regime_history = []
