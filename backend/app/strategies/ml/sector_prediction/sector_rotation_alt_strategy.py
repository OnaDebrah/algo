import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from ....analytics.market_regime_detector import MarketRegimeDetector
from ....core import fetch_financials, fetch_stock_data
from ....core.data.fetchers.alternative_data_fetcher import AlternativeDataSource
from ....core.data.fetchers.fundamentals_fetcher import FundamentalsFetcher
from ....strategies import BaseStrategy
from ....strategies.ml.sector_prediction.factor_explainer import FactorExplainer
from ....utils.async_helper import AsyncHelper
from ....utils.helpers import SECTOR_MAPPINGS

logger = logging.getLogger(__name__)


class SectorRotationAltStrategy(BaseStrategy):
    """
    Advanced sector rotation strategy with:
    1. SHAP-based explainability
    2. Alternative data integration (sentiment, news)
    3. Progressive ML complexity
    4. Regime-aware predictions
    """

    def __init__(
        self,
        name: str = "enhanced_sector_rotation",
        lookback_years: int = 5,
        forecast_horizon_days: int = 60,
        top_sectors: int = 3,
        stocks_per_sector: int = 5,
        model_type: str = "random_forest",
        use_regime_detection: bool = True,
        use_alternative_data: bool = True,
        use_shap_explanations: bool = True,
        confidence_threshold: float = 0.6,
        rebalance_frequency_days: int = 30,
        max_sector_exposure: float = 0.3,
        sentiment_weight: float = 0.2,
        enable_ml: bool = True,
        params: Dict = None,
    ):
        super().__init__(name, params or {})

        self.lookback_years = lookback_years
        self.forecast_horizon = forecast_horizon_days
        self.top_sectors = top_sectors
        self.stocks_per_sector = stocks_per_sector
        self.model_type = model_type
        self.use_regime_detection = use_regime_detection
        self.use_alternative_data = use_alternative_data
        self.use_shap_explanations = use_shap_explanations
        self.confidence_threshold = confidence_threshold
        self.rebalance_frequency = rebalance_frequency_days
        self.max_sector_exposure = max_sector_exposure
        self.sentiment_weight = sentiment_weight
        self.enable_ml = enable_ml

        # Initialize components
        self.fundamentals_fetcher = FundamentalsFetcher()
        self.alternative_data = AlternativeDataSource() if use_alternative_data else None
        self.regime_detector = MarketRegimeDetector(name=f"{name}_regime") if use_regime_detection else None

        # Model storage
        self.sector_models = {}  # sector -> model
        self.sector_explainers = {}  # sector -> SHAPExplainer
        self.stock_models = {}  # sector -> {symbol -> model}
        self.scalers = {}  # sector -> scaler
        self.feature_names = {}  # sector -> list of feature names

        # Performance tracking
        self.model_performance = defaultdict(list)
        self.prediction_history = []
        self.explanation_history = []
        self.last_rebalance = None
        self.current_positions = {}

        # Feature engineering cache
        self.feature_cache = {}
        self.sentiment_cache = {}

        self.SECTORS = SECTOR_MAPPINGS

        logger.info(
            f"SectorRotationStrategy initialized with model={model_type}, " f"alternative_data={use_alternative_data}, shap={use_shap_explanations}"
        )

    def _run_async(self, coro):
        """Run an async coroutine from sync context"""
        return AsyncHelper.run_async(coro)

    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """Generate trading signal with explanations"""
        try:
            current_time = data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else datetime.now()

            if self._should_rebalance(current_time):
                logger.info("Rebalancing triggered")
                positions, explanations = self._run_async(self._generate_positions_with_explanations(data))
                self.current_positions = positions
                self.explanation_history.append({"date": current_time, "explanations": explanations})
                self.last_rebalance = current_time

            if self.current_positions:
                return {
                    "signal": 0,
                    "position_size": 1.0,
                    "metadata": {
                        "strategy": "enhanced_sector_rotation",
                        "positions": self.current_positions,
                        "explanations": self.explanation_history[-1] if self.explanation_history else {},
                        "last_rebalance": self.last_rebalance.isoformat() if self.last_rebalance else None,
                        "regime": self._get_current_regime(data) if self.regime_detector else None,
                    },
                }
            else:
                return {"signal": 0, "position_size": 0.0, "metadata": {}}

        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return {"signal": 0, "position_size": 0.0, "metadata": {"error": str(e)}}

    # ==================== SECTOR PREDICTION WITH ALTERNATIVE DATA ====================

    async def _predict_top_sectors_with_explanations(self, data: pd.DataFrame) -> Tuple[List[Dict], Dict]:
        """
        Predict top sectors with SHAP explanations
        """
        if not self.enable_ml:
            return self._baseline_sector_prediction(), {}

        macro_features = await self._extract_macro_features(data)
        sentiment_features = await self._extract_sentiment_features() if self.alternative_data else pd.DataFrame()

        all_features = pd.concat([macro_features, sentiment_features], axis=1) if not sentiment_features.empty else macro_features

        if all_features.empty:
            return self._baseline_sector_prediction(), {}

        feature_names = list(all_features.columns)

        self.current_feature_names = feature_names

        # Get current regime
        regime = None
        if self.regime_detector and len(data) > 50:
            try:
                regime_result = self.regime_detector.detect_current_regime(data["Close"], data.get("Volume"))
                regime = regime_result.get("regime") if regime_result else None
            except Exception as e:
                logger.debug(f"Regime detection failed: {e}")
                regime = None

        sector_predictions = []
        sector_explanations = {}

        for sector in self.SECTORS.keys():
            try:
                if sector not in self.sector_models:
                    await self._train_sector_model_with_alternative(sector, data)

                if sector in self.sector_models:
                    model = self.sector_models[sector]
                    scaler = self.scalers.get(sector)
                    explainer = self.sector_explainers.get(sector)

                    # Prepare features
                    latest_features = all_features.iloc[-1:].values

                    # Create feature values Series for context
                    feature_values = pd.Series(latest_features[0], index=feature_names)

                    if scaler:
                        latest_features = scaler.transform(latest_features)

                    # Make prediction
                    expected_return = model.predict(latest_features)[0]

                    # Get SHAP explanation - NOW WITH FEATURE NAMES!
                    explanation = None
                    if explainer and self.use_shap_explanations:
                        try:
                            # Pass feature_names to the explainer
                            if hasattr(explainer, "feature_names"):
                                # If explainer already has feature names, just pass features
                                top_factors_result = await explainer.get_top_factors(
                                    symbol=sector, features=latest_features[0], feature_values=feature_values, n_factors=5
                                )
                            else:
                                # Need to initialize explainer with feature names
                                # This assumes your FactorExplainer can be updated
                                explainer.feature_names = feature_names
                                top_factors_result = await explainer.get_top_factors(
                                    symbol=sector, features=latest_features[0], feature_values=feature_values, n_factors=5
                                )

                            explanation = {
                                "top_factors": top_factors_result,
                                "confidence": self._calculate_prediction_confidence(model, latest_features, sector, regime),
                                "feature_names": feature_names,
                            }

                        except Exception as e:
                            logger.debug(f"SHAP explanation failed for {sector}: {e}")
                            explanation = None

                        if explanation:
                            sector_explanations[sector] = explanation
                            confidence = explanation.get("confidence", 0.5)
                        else:
                            confidence = self._calculate_prediction_confidence(model, latest_features, sector, regime)
                    else:
                        # Calculate confidence from model
                        confidence = self._calculate_prediction_confidence(model, latest_features, sector, regime)

                    # Get feature importance
                    if explanation and explanation.get("top_factors"):
                        top_factors = explanation["top_factors"]
                    else:
                        top_factors = self._get_top_factors(sector, feature_names)

                    sector_predictions.append(
                        {
                            "sector": sector,
                            "expected_return": float(expected_return),
                            "confidence": confidence,
                            "top_factors": top_factors,
                            "regime": regime,
                            "model_type": self.model_type,
                            "sentiment_score": float(all_features.get("market_sentiment", 0)) if "market_sentiment" in all_features.columns else 0,
                            "feature_count": len(feature_names),  # Useful for debugging
                        }
                    )

            except Exception as e:
                logger.error(f"Error predicting sector {sector}: {e}")
                continue

        # Sort by expected_return * confidence
        for p in sector_predictions:
            p["score"] = p["expected_return"] * p["confidence"]

        sector_predictions.sort(key=lambda x: x["score"], reverse=True)

        # Store prediction history
        self.prediction_history.append(
            {
                "date": data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else datetime.now(),
                "predictions": sector_predictions.copy(),
                "regime": regime,
                "sentiment": await self._get_market_sentiment() if self.alternative_data else None,
                "feature_names": feature_names,  # Store for analysis
            }
        )

        return sector_predictions[: self.top_sectors], sector_explanations

    async def _train_sector_model_with_alternative(self, sector: str, data: pd.DataFrame):
        """
        Train sector model with alternative data integration
        """
        logger.info(f"Training enhanced model for sector: {sector}")

        # Prepare training data with alternative features
        X, y = await self._prepare_enhanced_training_data(sector, data)

        if X is None or y is None or len(X) < 20:
            logger.warning(f"Insufficient training data for sector {sector}")
            return

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Select model based on complexity
        if self.model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_split=5, random_state=42, n_jobs=-1)
        elif self.model_type == "gradient_boosting":
            model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=6, subsample=0.8, random_state=42)
        else:  # ensemble
            model = RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)

        # Train model
        model.fit(X_scaled, y)

        # Store model and scaler
        self.sector_models[sector] = model
        self.scalers[sector] = scaler
        self.feature_names[sector] = list(X.columns)

        # Initialize SHAP explainer
        if self.use_shap_explanations:
            try:
                # Use subset of training data for background
                background = X_scaled[np.random.choice(len(X_scaled), min(100, len(X_scaled)), replace=False)]
                self.sector_explainers[sector] = FactorExplainer(model, list(X.columns), background)
            except Exception as e:
                logger.error(f"Error initializing SHAP explainer for {sector}: {e}")

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

        # Log feature importance
        if hasattr(model, "feature_importances_"):
            importance = dict(zip(X.columns, model.feature_importances_))
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"Top features for {sector}: {top_features}")

    async def _prepare_enhanced_training_data(self, sector: str, data: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Prepare training data with alternative data features
        """
        try:
            # Get macro data
            macro_data = await self._fetch_historical_macro()

            # Get sentiment data if enabled
            sentiment_data = None
            if self.alternative_data:
                sentiment_data = await self._fetch_historical_sentiment(sector)

            # Calculate sector returns
            sector_returns = await self._calculate_sector_returns(sector, macro_data.index)

            if sector_returns.empty:
                return None, None

            # Combine features
            features = macro_data.copy()

            if sentiment_data is not None and not sentiment_data.empty:
                # Align dates
                common_dates = features.index.intersection(sentiment_data.index)
                features = features.loc[common_dates]
                features = pd.concat([features, sentiment_data.loc[common_dates]], axis=1)

            # Align with returns
            common_dates = features.index.intersection(sector_returns.index)
            if len(common_dates) < 20:
                return None, None

            X = features.loc[common_dates]
            y = sector_returns.loc[common_dates]

            # Create lagged features
            for lag in [1, 3, 6]:
                lagged = X.shift(lag)
                lagged.columns = [f"{col}_lag{lag}" for col in lagged.columns]
                X = pd.concat([X, lagged], axis=1)

            # Drop NaN rows
            combined = pd.concat([X, y], axis=1).dropna()
            X = combined[list(X.columns)]
            y = combined[y.name]

            logger.info(f"Prepared enhanced training data: X shape {X.shape}, y shape {y.shape}")

            return X, y

        except Exception as e:
            logger.error(f"Error preparing enhanced training data: {e}")
            return None, None

    async def _fetch_historical_sentiment(self, sector: str) -> pd.DataFrame:
        """
        Fetch historical sentiment data for a sector using AlternativeDataSource.
        Falls back to sector ETF price-derived sentiment proxies.
        """
        if not self.alternative_data:
            return pd.DataFrame()

        stocks = self.SECTORS.get(sector, [])[:3]  # Top 3 for efficiency

        try:
            # Get current sector sentiment from AlternativeDataSource
            sector_sent = await self.alternative_data.get_sector_sentiment(sector, stocks)

            # Build a simple sentiment DataFrame from available data
            # For training, we use price-derived sentiment proxies
            from ....utils.helpers import SECTOR_ETFS

            etf = SECTOR_ETFS.get(sector)
            if not etf:
                return pd.DataFrame()

            etf_data = await fetch_stock_data(etf, period=f"{self.lookback_years}y", interval="1wk")
            if etf_data is None or etf_data.empty:
                return pd.DataFrame()

            sentiment_data = pd.DataFrame(index=etf_data.index)

            # Price-derived sentiment proxies
            returns = etf_data["Close"].pct_change()
            sentiment_data["sector_sentiment"] = returns.rolling(4).mean()  # 4-week avg return as sentiment proxy
            sentiment_data["news_volume"] = etf_data["Volume"].rolling(4).mean() / etf_data["Volume"].rolling(20).mean()  # Relative volume
            sentiment_data["analyst_rating"] = returns.rolling(12).mean()  # 12-week trend as analyst proxy

            # Add actual current sentiment score if available
            current_sent = sector_sent.get("sector_sentiment", 0.0)
            if current_sent != 0.0:
                sentiment_data.iloc[-1, sentiment_data.columns.get_loc("sector_sentiment")] = current_sent

            return sentiment_data.dropna()

        except Exception as e:
            logger.warning(f"Sentiment fetch failed for {sector}: {e}")
            return pd.DataFrame()

    async def _extract_sentiment_features(self) -> pd.DataFrame:
        """
        Extract current sentiment features
        """
        if not self.alternative_data:
            return pd.DataFrame()

        features = {}

        # Get market sentiment
        market_sent = await self.alternative_data.get_market_sentiment()
        features["market_sentiment"] = market_sent.get("market_sentiment", 0)
        features["fear_greed_index"] = market_sent.get("fear_greed_index", 50)

        # Get sector sentiments
        for sector in self.SECTORS.keys():
            sector_sent = await self.alternative_data.get_sector_sentiment(sector, self.SECTORS[sector])
            features[f"{sector.lower()}_sentiment"] = sector_sent.get("sector_sentiment", 0)

        return pd.DataFrame([features])

    async def _get_market_sentiment(self) -> Dict:
        """Get current market sentiment"""
        if not self.alternative_data:
            return {}
        return await self.alternative_data.get_market_sentiment()

    # ==================== ENHANCED STOCK SELECTION WITH SENTIMENT ====================

    async def _select_top_stocks_with_explanations(self, sector: str, n_stocks: int, data: pd.DataFrame) -> Tuple[List[Dict], Dict]:
        """
        Select best stocks with sentiment integration and explanations
        """
        stocks = self.SECTORS.get(sector, [])
        if not stocks:
            return [], {}

        # Get fundamental data
        fundamentals = await self._fetch_sector_fundamentals(sector, stocks)

        # Get sentiment data for each stock
        sentiment_scores = {}
        if self.alternative_data:
            for symbol in stocks[:10]:  # Limit for performance
                sentiment_scores[symbol] = await self.alternative_data.get_sentiment_score(symbol)

        # Score each stock
        stock_scores = []
        stock_explanations = {}

        for symbol in stocks:
            try:
                # Get fundamental factors
                stock_factors = fundamentals[fundamentals["symbol"] == symbol] if not fundamentals.empty else pd.DataFrame()

                # Get sentiment
                sentiment = sentiment_scores.get(symbol, {})

                # Calculate composite score with explanation
                score, explanation = self._calculate_enhanced_stock_score(symbol, data, stock_factors, sentiment)

                if explanation:
                    stock_explanations[symbol] = explanation

                stock_scores.append(
                    {
                        "symbol": symbol,
                        "score": score,
                        "expected_return": score * 0.1,
                        "confidence": min(abs(score) * 2, 0.95),
                        "sentiment": sentiment.get("overall_sentiment", 0),
                        "explanation": explanation,
                    }
                )

            except Exception as e:
                logger.error(f"Error scoring {symbol}: {e}")
                continue

        # Sort by score
        stock_scores.sort(key=lambda x: x["score"], reverse=True)

        return stock_scores[:n_stocks], stock_explanations

    def _calculate_enhanced_stock_score(
        self, symbol: str, price_data: pd.DataFrame, fundamentals: pd.DataFrame, sentiment: Dict
    ) -> Tuple[float, Dict]:
        """
        Calculate composite stock score with sentiment and explanation
        """
        score = 0.0
        factors = {}

        # Factor 1: Price momentum (25% weight)
        if len(price_data) > 20:
            returns_1m = price_data["Close"].pct_change(21).iloc[-1] if len(price_data) > 21 else 0
            returns_3m = price_data["Close"].pct_change(63).iloc[-1] if len(price_data) > 63 else 0
            momentum = returns_1m * 0.6 + returns_3m * 0.4
            momentum_score = np.clip(momentum, -0.3, 0.3) / 0.3
            score += 0.25 * momentum_score
            factors["momentum"] = {"value": float(momentum), "score": float(momentum_score), "weight": 0.25}

        # Factor 2: Volatility (20% weight) - lower is better
        if len(price_data) > 20:
            volatility = price_data["Close"].pct_change().std() * np.sqrt(252)
            vol_score = -np.clip(volatility - 0.2, -0.3, 0.3) / 0.3
            score += 0.2 * vol_score
            factors["volatility"] = {"value": float(volatility), "score": float(vol_score), "weight": 0.2}

        # Factor 3: Fundamentals (35% weight)
        if not fundamentals.empty:
            # P/E score
            pe = fundamentals.get("pe_ratio", 20).iloc[0] if len(fundamentals) > 0 else 20
            if pe > 0 and pe < 50:
                pe_score = -np.clip((pe - 20) / 20, -1, 1)
                score += 0.15 * pe_score
                factors["pe_ratio"] = {"value": float(pe), "score": float(pe_score), "weight": 0.15}

            # Growth score
            rev_growth = fundamentals.get("revenue_growth", 0).iloc[0] if len(fundamentals) > 0 else 0
            eps_growth = fundamentals.get("eps_growth", 0).iloc[0] if len(fundamentals) > 0 else 0
            growth = (rev_growth + eps_growth) / 2
            growth_score = np.clip(growth / 0.2, -1, 1)
            score += 0.2 * growth_score
            factors["growth"] = {"value": float(growth), "score": float(growth_score), "weight": 0.2}

        # Factor 4: Sentiment (20% weight)
        if sentiment:
            sent_score = sentiment.get("overall_sentiment", 0)
            score += self.sentiment_weight * sent_score
            factors["sentiment"] = {
                "value": float(sent_score),
                "score": float(sent_score),
                "weight": self.sentiment_weight,
                "components": {
                    "news": sentiment.get("news_sentiment", 0),
                    "analyst": sentiment.get("analyst_sentiment", 0),
                    "momentum": sentiment.get("momentum_score", 0),
                },
            }

        explanation = {
            "total_score": float(score),
            "factors": factors,
            "primary_driver": max(factors.items(), key=lambda x: abs(x[1]["score"] * x[1]["weight"]))[0] if factors else "unknown",
        }

        return float(np.clip(score, -1, 1)), explanation

    # ==================== POSITION GENERATION WITH EXPLANATIONS ====================

    async def _generate_positions_with_explanations(self, data: pd.DataFrame) -> Tuple[Dict, Dict]:
        """
        Generate positions with full explanations
        """
        # Phase 1: Predict top sectors with explanations
        top_sectors, sector_explanations = await self._predict_top_sectors_with_explanations(data)

        if not top_sectors:
            logger.warning("No sector predictions available")
            return {}, {}

        logger.info(f"Top sectors with explanations: {[s['sector'] for s in top_sectors]}")

        # Phase 2: Select stocks from top sectors with explanations
        all_positions = {}
        all_explanations = {"sector_predictions": top_sectors, "sector_explanations": sector_explanations, "stock_selections": {}}

        sector_weights = {}
        total_confidence = sum(s["confidence"] for s in top_sectors)

        for sector_pred in top_sectors:
            sector = sector_pred["sector"]
            confidence = sector_pred["confidence"]

            # Base sector weight
            sector_weight = (confidence / total_confidence) if total_confidence > 0 else 1.0 / len(top_sectors)
            sector_weight = min(sector_weight, self.max_sector_exposure)

            # Select stocks in this sector with explanations
            stocks, stock_explanations = await self._select_top_stocks_with_explanations(sector, self.stocks_per_sector, data)

            if stocks:
                stock_weight = sector_weight / len(stocks)

                sector_stocks = []
                for stock in stocks:
                    symbol = stock["symbol"]

                    # Apply stock-level confidence
                    final_weight = stock_weight * stock["confidence"]
                    all_positions[symbol] = final_weight

                    # Store detailed explanation
                    all_explanations["stock_selections"][symbol] = {
                        "sector": sector,
                        "sector_confidence": confidence,
                        "stock_confidence": stock["confidence"],
                        "expected_return": stock["expected_return"],
                        "sentiment": stock.get("sentiment", 0),
                        "score_explanation": stock.get("explanation", {}),
                        "allocation": final_weight,
                    }

                    sector_stocks.append({"symbol": symbol, "allocation": final_weight, "confidence": stock["confidence"]})

                sector_weights[sector] = {"weight": sector_weight, "stocks": sector_stocks, "explanation": sector_explanations.get(sector, {})}

        all_explanations["sector_weights"] = sector_weights

        # Normalize positions
        total_weight = sum(all_positions.values())
        if total_weight > 1.0:
            for symbol in all_positions:
                all_positions[symbol] /= total_weight
                if symbol in all_explanations["stock_selections"]:
                    all_explanations["stock_selections"][symbol]["allocation"] = all_positions[symbol]

        logger.info(f"Generated {len(all_positions)} positions with explanations")

        return all_positions, all_explanations

    # ==================== UTILITY FUNCTIONS ====================

    async def _extract_macro_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract macro features with caching"""
        cache_key = data.index[-1].strftime("%Y-%m-%d") if hasattr(data.index, "strftime") else "latest"

        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        # Fetch macro data
        macro_data = await self._fetch_historical_macro()

        if macro_data.empty:
            return pd.DataFrame()

        # Get latest macro data
        latest = macro_data.iloc[-1:].copy()

        # Add technical indicators
        if not data.empty:
            latest["vix"] = data["Close"].pct_change().std() * np.sqrt(252)
            latest["trend_strength"] = self._calculate_trend_strength(data)
            latest["market_momentum"] = data["Close"].pct_change(63).iloc[-1] if len(data) > 63 else 0

            # Add volume indicators
            latest["volume_ratio"] = data["Volume"].iloc[-1] / data["Volume"].rolling(20).mean().iloc[-1]

        self.feature_cache[cache_key] = latest
        return latest

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength indicator"""
        if len(data) < 200:
            return 0

        sma_50 = data["Close"].rolling(50).mean()
        sma_200 = data["Close"].rolling(200).mean()

        if pd.isna(sma_50.iloc[-1]) or pd.isna(sma_200.iloc[-1]):
            return 0

        # Trend strength based on MA alignment
        price_vs_sma50 = data["Close"].iloc[-1] / sma_50.iloc[-1] - 1
        sma50_vs_sma200 = sma_50.iloc[-1] / sma_200.iloc[-1] - 1

        return float(price_vs_sma50 * 0.5 + sma50_vs_sma200 * 0.5)

    def _calculate_prediction_confidence(self, model, features, sector, regime):
        """Calculate prediction confidence"""
        confidence = 0.7

        # Model confidence
        if hasattr(model, "estimators_"):
            tree_preds = np.array([tree.predict(features)[0] for tree in model.estimators_])
            model_conf = 1.0 - (tree_preds.std() / (abs(tree_preds.mean()) + 1e-6))
            confidence *= np.clip(model_conf, 0.5, 0.95)

        # Regime confidence
        if regime and self.regime_detector and self.regime_detector.regime_history:
            recent = self.regime_detector.regime_history[-10:]
            stability = sum(1 for r in recent if r["regime"] == regime) / len(recent)
            confidence *= 0.8 + 0.2 * stability

        # Historical performance
        if sector in self.model_performance and self.model_performance[sector]:
            recent_perf = np.mean(self.model_performance[sector][-3:])
            confidence *= 0.7 + 0.3 * max(0, recent_perf)

        return float(np.clip(confidence, 0.3, 0.95))

    def _get_top_factors(self, sector: str, feature_names) -> List[Dict]:
        """Get top factors for a sector"""
        if sector in feature_names and sector in self.sector_models:
            model = self.sector_models[sector]
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                features = feature_names[sector]

                # Create factor list
                factors = []
                for feat, imp in zip(features, importances):
                    factors.append({"factor": feat, "importance": float(imp), "impact": "positive" if imp > 0 else "negative"})

                # Sort by importance
                factors.sort(key=lambda x: x["importance"], reverse=True)
                return factors[:5]

        return [
            {"factor": "market_momentum", "importance": 0.3, "impact": "positive"},
            {"factor": "interest_rates", "importance": 0.25, "impact": "negative"},
            {"factor": "sector_sentiment", "importance": 0.2, "impact": "positive"},
            {"factor": "volatility", "importance": 0.15, "impact": "negative"},
            {"factor": "growth_outlook", "importance": 0.1, "impact": "positive"},
        ]

    def _get_current_regime(self, data: pd.DataFrame) -> Optional[str]:
        """Get current market regime"""
        if not self.regime_detector or len(data) < 50:
            return None
        try:
            regime_result = self.regime_detector.detect_current_regime(data["Close"], data.get("Volume"))
            return regime_result.get("regime") if regime_result else None
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return None

    def _should_rebalance(self, current_time: datetime) -> bool:
        """Check if rebalancing is needed"""
        if self.last_rebalance is None:
            return True
        days_since = (current_time - self.last_rebalance).days
        return days_since >= self.rebalance_frequency

    def _baseline_sector_prediction(self) -> List[Dict]:
        """Baseline prediction when ML is disabled"""
        sectors = list(self.SECTORS.keys())
        predictions = []
        for i, sector in enumerate(sectors):
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

    async def _fetch_historical_macro(self) -> pd.DataFrame:
        """
        Fetch historical macro data from sector ETF prices and VIX.
        Uses real market data via fetch_stock_data.
        """
        from ....utils.helpers import SECTOR_ETFS

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
            except Exception as e:
                logger.debug(f"Failed to fetch VIX: {e}")

            # Fetch SPY for market-level features
            try:
                spy = await fetch_stock_data("SPY", period=f"{self.lookback_years}y", interval="1mo")
                if spy is not None and not spy.empty:
                    macro_data["market_return"] = spy["Close"].pct_change()
                    macro_data["market_momentum_3m"] = spy["Close"].pct_change(3)
                    macro_data["market_momentum_6m"] = spy["Close"].pct_change(6)
            except Exception as e:
                logger.debug(f"Failed to fetch SPY: {e}")

            # Fetch Treasury proxy (TLT) for interest rate environment
            try:
                tlt = await fetch_stock_data("TLT", period=f"{self.lookback_years}y", interval="1mo")
                if tlt is not None and not tlt.empty:
                    macro_data["bond_return"] = tlt["Close"].pct_change()
            except Exception as e:
                logger.debug(f"Failed to fetch TLT: {e}")

            return macro_data.dropna() if not macro_data.empty else pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to fetch macro data: {e}")
            return pd.DataFrame()

    async def _fetch_sector_fundamentals(self, sector: str, stocks: List[str]) -> pd.DataFrame:
        """Fetch fundamental data for sector stocks"""
        all_fundamentals = []

        for symbol in stocks[:10]:
            try:
                financials = await fetch_financials(symbol)
                if financials:
                    fundamentals = {
                        "symbol": symbol,
                        "sector": sector,
                        "pe_ratio": financials.get("trailingPE", 20),
                        "forward_pe": financials.get("forwardPE", 20),
                        "peg_ratio": financials.get("pegRatio", 1),
                        "pb_ratio": financials.get("priceToBook", 3),
                        "debt_to_equity": financials.get("debtToEquity", 50),
                        "roe": financials.get("returnOnEquity", 0.15),
                        "operating_margin": financials.get("operatingMargins", 0.15),
                        "revenue_growth": financials.get("revenueGrowth", 0.1),
                        "eps_growth": financials.get("earningsGrowth", 0.1),
                    }
                    all_fundamentals.append(fundamentals)
            except Exception:
                continue

        if not all_fundamentals:
            return pd.DataFrame()

        return pd.DataFrame(all_fundamentals)

    async def _calculate_sector_returns(self, sector: str, dates: pd.DatetimeIndex) -> pd.Series:
        """Calculate historical sector returns"""
        stocks = self.SECTORS.get(sector, [])[:5]

        all_returns = []
        for symbol in stocks:
            try:
                data = await fetch_stock_data(symbol, period=f"{self.lookback_years + 1}y", interval="1mo")
                if data is not None and not data.empty:
                    returns = data["Close"].pct_change(self.forecast_horizon).shift(-self.forecast_horizon)
                    returns.name = symbol
                    all_returns.append(returns)
            except Exception:
                continue

        if not all_returns:
            return pd.Series()

        returns_df = pd.concat(all_returns, axis=1)
        sector_returns = returns_df.mean(axis=1)
        sector_returns.name = f"{sector}_returns"

        return sector_returns

    def get_explanation_summary(self) -> Dict:
        """Get summary of latest explanations"""
        if not self.explanation_history:
            return {}

        latest = self.explanation_history[-1]

        summary = {
            "date": latest["date"].isoformat() if hasattr(latest["date"], "isoformat") else str(latest["date"]),
            "sector_predictions": [],
            "top_stocks": [],
        }

        # Summarize sector predictions
        for pred in latest["explanations"].get("sector_predictions", []):
            summary["sector_predictions"].append(
                {
                    "sector": pred["sector"],
                    "expected_return": f"{pred['expected_return'] * 100:.1f}%",
                    "confidence": f"{pred['confidence'] * 100:.1f}%",
                    "top_factors": pred["top_factors"][:3] if pred["top_factors"] else [],
                }
            )

        # Summarize top stocks
        for symbol, info in latest["explanations"].get("stock_selections", {}).items():
            summary["top_stocks"].append(
                {
                    "symbol": symbol,
                    "sector": info["sector"],
                    "allocation": f"{info['allocation'] * 100:.1f}%",
                    "confidence": f"{info['stock_confidence'] * 100:.1f}%",
                    "primary_driver": info.get("score_explanation", {}).get("primary_driver", "unknown"),
                }
            )

        return summary

    def reset(self):
        """Reset strategy state"""
        self.sector_models = {}
        self.sector_explainers = {}
        self.stock_models = {}
        self.scalers = {}
        self.feature_names = {}
        self.model_performance = defaultdict(list)
        self.prediction_history = []
        self.explanation_history = []
        self.last_rebalance = None
        self.current_positions = {}
        self.feature_cache = {}
        self.sentiment_cache = {}

        if self.regime_detector:
            self.regime_detector.regime_history = []

        if self.alternative_data:
            self.alternative_data.cache = {}
