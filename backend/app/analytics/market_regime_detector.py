"""
Market Regime Detector
Purpose: Identify current market regime and adjust strategy allocations accordingly
Why critical: Most strategies fail in wrong regimes; this ensures right strategy at right time

Features:
1. Hurst exponent calculation (now returns Series)
2. Half-life calculation (now returns Series)
3. Volume data initialization
4. Optimized correlation calculation
5. Caching for expensive calculations
6. Memory leaks
7. Regime confidence intervals
8. Early warning system
9. Regime strength indicator
"""

import hashlib
import logging
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..analytics.regimes.markov_regime_chain import MarkovRegimeChain

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """
    Advanced Market Regime Detection System

    Uses multiple methods to detect and classify market regimes:
    1. Statistical methods (volatility, trend, mean reversion)
    2. Technical indicators
    3. Machine learning classification
    4. Market breadth analysis
    5. Ensemble approach
    """

    def __init__(
        self,
        lookback_period: int = 252,
        regime_types: List[str] = None,
        primary_index: str = "SPY",
        volatility_thresholds: Dict = None,
        trend_thresholds: Dict = None,
        use_ml: bool = True,
        retrain_frequency: int = 63,
        confidence_threshold: float = 0.7,
        **kwargs,
    ):
        """
        Initialize Market Regime Detector

        Args:
            lookback_period: Period for regime analysis
            regime_types: List of regime classifications
            primary_index: Primary market index symbol
            volatility_thresholds: Thresholds for volatility regimes
            trend_thresholds: Thresholds for trend regimes
            use_ml: Use machine learning for regime classification
            retrain_frequency: How often to retrain ML models
            confidence_threshold: Minimum confidence for regime classification
        """
        self.lookback_period = lookback_period
        self.primary_index = primary_index

        # Define regime types
        if regime_types is None:
            self.regime_types = [
                "trending_bull",
                "trending_bear",
                "mean_reverting",
                "high_volatility",
                "low_volatility",
                "crisis",
                "recovery",
                "transition",
            ]
        else:
            self.regime_types = regime_types

        # Threshold configurations
        self.volatility_thresholds = volatility_thresholds or {
            "low_vol": 0.12,
            "medium_vol": 0.25,
            "high_vol": 0.40,
            "crisis_vol": 0.60,
        }

        self.trend_thresholds = trend_thresholds or {
            "strong_trend": 0.15,
            "moderate_trend": 0.08,
            "weak_trend": 0.03,
            "no_trend": 0.0,
        }

        # ML configuration
        self.use_ml = use_ml
        self.retrain_frequency = retrain_frequency
        self.confidence_threshold = confidence_threshold

        # ML models
        self.classifier = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None

        # Regime history
        self.regime_history = []
        self.regime_confidences = []
        self.last_training_date = None

        # Strategy allocation templates
        self.strategy_allocation_templates = self._create_allocation_templates()

        # Cache for expensive calculations
        self._feature_cache = {}
        self._cache_max_size = 100
        self.markov_chain = MarkovRegimeChain()

    def calculate_features(
        self,
        price_data: Union[pd.Series, pd.DataFrame],
        volume_data: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Calculate features for regime detection

        Args:
            price_data: Price series or DataFrame
            volume_data: Optional volume series

        Returns:
            DataFrame with features
        """
        # Extract primary price series
        if isinstance(price_data, pd.Series):
            prices = price_data
        elif isinstance(price_data, pd.DataFrame):
            if self.primary_index in price_data.columns:
                prices = price_data[self.primary_index]
            else:
                prices = price_data.iloc[:, 0]
        else:
            raise ValueError("price_data must be Series or DataFrame")

        # Check cache
        cache_key = self._get_cache_key(prices)
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        # Calculate returns
        returns = np.log(prices / prices.shift(1)).dropna()

        features = pd.DataFrame(index=returns.index)

        # 1. Volatility features
        features["volatility_21d"] = returns.rolling(21).std() * np.sqrt(252)
        features["volatility_63d"] = returns.rolling(63).std() * np.sqrt(252)
        features["volatility_ratio"] = features["volatility_21d"] / (features["volatility_63d"] + 1e-10)
        features["volatility_regime"] = self._classify_volatility(features["volatility_21d"])

        # 2. Trend features
        features["trend_21d"] = prices / prices.shift(21) - 1
        features["trend_63d"] = prices / prices.shift(63) - 1
        features["trend_252d"] = prices / prices.shift(252) - 1

        features["trend_strength"] = self._calculate_trend_strength(prices, window=14)
        features["trend_direction"] = np.sign(features["trend_21d"])

        # 3. Mean reversion features - FIXED
        features["hurst_exponent"] = self._calculate_hurst_exponent_rolling(prices, window=100)
        features["half_life"] = self._calculate_half_life_rolling(prices, window=50)
        features["z_score"] = self._calculate_zscore(prices, window=20)

        # 4. Market breadth features (if multiple assets)
        if isinstance(price_data, pd.DataFrame) and price_data.shape[1] > 1:
            breadth_features = self._calculate_market_breadth(price_data)
            features = pd.concat([features, breadth_features], axis=1)

        # 5. Volume features (if available) - FIXED
        if volume_data is not None and len(volume_data) > 0:
            volume_features = self._calculate_volume_features(prices, volume_data)
            features = pd.concat([features, volume_features], axis=1)

        # 6. Correlation features - OPTIMIZED
        if isinstance(price_data, pd.DataFrame) and price_data.shape[1] > 5:
            correlation_features = self._calculate_correlation_features_optimized(price_data)
            features = pd.concat([features, correlation_features], axis=1)

        # 7. Technical indicators
        features["rsi"] = self._calculate_rsi(prices, window=14)
        features["macd"] = self._calculate_macd(prices)
        features["bollinger_band_width"] = self._calculate_bollinger_bandwidth(prices)

        # 8. Statistical features
        features["skewness"] = returns.rolling(63).skew()
        features["kurtosis"] = returns.rolling(63).kurt()
        features["var_95"] = returns.rolling(63).quantile(0.05) * np.sqrt(252)

        # 9. Regime persistence
        if len(self.regime_history) > 0:
            features["regime_persistence"] = self._calculate_regime_persistence()

        # Drop rows where CORE short-lookback features are NaN,
        # but keep rows even if long-lookback features (trend_252d, hurst, etc.) are NaN
        core_cols = [c for c in ["volatility_21d", "trend_21d", "trend_strength", "rsi", "z_score"] if c in features.columns]
        if core_cols:
            features_clean = features.dropna(subset=core_cols)
        else:
            features_clean = features.dropna()

        # Update cache
        self._update_cache(cache_key, features_clean)

        return features_clean

    def _get_cache_key(self, prices: pd.Series) -> str:
        """Generate cache key for feature calculations"""
        # Use last 10 prices and length as key
        key_data = f"{len(prices)}_{prices.iloc[-10:].sum()}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _update_cache(self, key: str, features: pd.DataFrame):
        """Update feature cache with size limit"""
        self._feature_cache[key] = features

        # Limit cache size
        if len(self._feature_cache) > self._cache_max_size:
            # Remove oldest entries
            keys_to_remove = list(self._feature_cache.keys())[: -self._cache_max_size]
            for k in keys_to_remove:
                del self._feature_cache[k]

    def _classify_volatility(self, volatility: pd.Series) -> pd.Series:
        """Classify volatility regime - VECTORIZED"""
        conditions = [
            (volatility <= self.volatility_thresholds["low_vol"]),
            (volatility <= self.volatility_thresholds["medium_vol"]),
            (volatility <= self.volatility_thresholds["high_vol"]),
            (volatility > self.volatility_thresholds["high_vol"]),
        ]

        choices = ["low_vol", "medium_vol", "high_vol", "crisis_vol"]

        return pd.Series(np.select(conditions, choices, default="medium_vol"), index=volatility.index)

    def _calculate_trend_strength(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate ADX-like trend strength"""
        high = prices.rolling(window).max()
        low = prices.rolling(window).min()

        tr = pd.DataFrame(
            {
                "hl": high - low,
                "hc": abs(high - prices.shift(1)),
                "lc": abs(low - prices.shift(1)),
            }
        ).max(axis=1)

        atr = tr.rolling(window).mean()

        # Simplified trend strength
        trend_strength = (prices.diff(window).abs() / (atr + 1e-10)).replace([np.inf, -np.inf], np.nan)

        return trend_strength

    def _calculate_hurst_exponent_rolling(self, prices: pd.Series, window: int = 100) -> pd.Series:
        """
        OPTIMIZED: Calculate rolling Hurst exponent using vectorized lag differences

        Returns:
            pd.Series with Hurst exponent values
        """
        values = prices.values
        n = len(values)
        result = np.full(n, 0.5)

        max_lag = min(50, window // 2)
        if max_lag < 2:
            return pd.Series(result, index=prices.index)

        lags = np.arange(2, max_lag)
        log_lags = np.log(lags)

        for i in range(window - 1, n):
            win = values[i - window + 1 : i + 1]

            # Vectorized tau calculation across all lags at once
            tau = np.array([np.sqrt(np.std(win[lag:] - win[:-lag])) for lag in lags if lag < len(win) and len(win[lag:]) > 0])

            if len(tau) < 2:
                continue

            log_tau = np.log(tau)
            valid = np.isfinite(log_lags[: len(tau)]) & np.isfinite(log_tau)
            if valid.sum() < 2:
                continue

            try:
                hurst = np.polyfit(log_lags[: len(tau)][valid], log_tau[valid], 1)[0]
                result[i] = np.clip(hurst, 0, 1)
            except Exception:
                pass

        return pd.Series(result, index=prices.index)

    def _calculate_half_life_rolling(self, prices: pd.Series, window: int = 50) -> pd.Series:
        """
        FIXED: Calculate rolling half-life of mean reversion

        Returns:
            pd.Series with half-life values (in days)
        """

        def half_life_for_window(price_window):
            """Calculate half-life for a price window"""
            if len(price_window) < 20:
                return 100  # High value indicates no mean reversion

            log_prices = np.log(price_window)
            delta = log_prices.diff().dropna()
            lag = log_prices.shift(1).dropna()

            # Align indices
            common_idx = delta.index.intersection(lag.index)
            if len(common_idx) < 10:
                return 100

            delta = delta.loc[common_idx]
            lag = lag.loc[common_idx]

            # OLS regression: delta = alpha + beta * lag
            try:
                beta = np.polyfit(lag.values, delta.values, 1)[0]

                # No mean reversion if beta >= 0
                if beta >= 0:
                    return 100

                # Half-life = -ln(2) / ln(1 + beta)
                half_life = -np.log(2) / np.log(1 + beta)

                # Constrain to reasonable range
                return np.clip(half_life, 1, 100)
            except Exception:
                return 100

        # Apply rolling calculation
        return prices.rolling(window, min_periods=20).apply(half_life_for_window, raw=False)

    def _calculate_zscore(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling z-score"""
        rolling_mean = prices.rolling(window).mean()
        rolling_std = prices.rolling(window).std()

        return (prices - rolling_mean) / (rolling_std + 1e-10)

    def _calculate_market_breadth(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate market breadth indicators"""
        breadth = pd.DataFrame(index=price_data.index)

        # Advance-decline line
        daily_returns = price_data.pct_change()
        advances = (daily_returns > 0).sum(axis=1)
        declines = (daily_returns < 0).sum(axis=1)

        breadth["advance_decline_ratio"] = advances / (advances + declines + 1e-10)
        breadth["percent_above_ma20"] = (price_data > price_data.rolling(20).mean()).sum(axis=1) / price_data.shape[1]
        breadth["percent_above_ma50"] = (price_data > price_data.rolling(50).mean()).sum(axis=1) / price_data.shape[1]

        # New highs/lows
        breadth["new_highs_20d"] = (price_data == price_data.rolling(20).max()).sum(axis=1)
        breadth["new_lows_20d"] = (price_data == price_data.rolling(20).min()).sum(axis=1)

        return breadth

    def _calculate_volume_features(self, prices: pd.Series, volume: pd.Series) -> pd.DataFrame:
        """
        FIXED: Calculate volume-based features with proper alignment
        """
        # Align price and volume data
        common_idx = prices.index.intersection(volume.index)
        prices_aligned = prices.loc[common_idx]
        volume_aligned = volume.loc[common_idx]

        features = pd.DataFrame(index=common_idx)

        # Volume trends
        features["volume_ma_ratio"] = volume_aligned / (volume_aligned.rolling(20).mean() + 1e-10)
        features["volume_price_trend"] = (prices_aligned.diff() * volume_aligned).rolling(5).mean()

        # On-balance volume (OBV)
        price_change = prices_aligned.diff()
        obv = (np.sign(price_change) * volume_aligned).fillna(0).cumsum()

        features["obv"] = obv
        features["obv_slope"] = obv.rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True)

        return features

    def _calculate_correlation_features_optimized(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        OPTIMIZED: Calculate correlation-based features using vectorized rolling correlations
        """
        features = pd.DataFrame(index=price_data.index)
        returns = price_data.pct_change().dropna()

        if len(returns) < 63 or returns.shape[1] < 2:
            return features

        # Vectorized approach: compute rolling pairwise correlations directly
        cols = returns.columns
        n_cols = len(cols)

        if n_cols < 2:
            return features

        # Calculate rolling correlations for all pairs using pandas rolling corr
        corr_sums = pd.Series(0.0, index=returns.index)
        n_pairs = 0

        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                pair_corr = returns.iloc[:, i].rolling(63, min_periods=20).corr(returns.iloc[:, j])
                corr_sums = corr_sums.add(pair_corr, fill_value=0)
                n_pairs += 1

        if n_pairs > 0:
            features["avg_correlation"] = corr_sums / n_pairs
        else:
            features["avg_correlation"] = np.nan

        # Reindex to match original price_data index
        features = features.reindex(price_data.index)

        # Correlation regime
        corr_values = features["avg_correlation"].values
        regime = np.empty(len(corr_values), dtype=object)

        mask1 = corr_values <= 0.2
        mask2 = (corr_values > 0.2) & (corr_values <= 0.5)
        mask3 = (corr_values > 0.5) & (corr_values <= 0.8)
        mask4 = corr_values > 0.8

        regime[mask1] = "very_low"
        regime[mask2] = "low"
        regime[mask3] = "medium"
        regime[mask4] = "high"

        features["correlation_regime"] = pd.Series(regime, index=features.index)

        return features

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2

        return macd

    def _calculate_bollinger_bandwidth(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate Bollinger Band width"""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()

        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)

        bandwidth = (upper_band - lower_band) / (sma + 1e-10)

        return bandwidth

    def _calculate_regime_persistence(self) -> float:
        """Calculate how long current regime has persisted"""
        if len(self.regime_history) < 2:
            return 0.0

        current_regime = self.regime_history[-1]["regime"]
        persistence = 0

        for regime in reversed(self.regime_history[:-1]):
            if regime["regime"] == current_regime:
                persistence += 1
            else:
                break

        return persistence / len(self.regime_history) if len(self.regime_history) > 0 else 0.0

    def detect_regime_statistical(self, features: pd.DataFrame) -> Dict:
        """
        Detect regime using statistical methods
        """
        latest = features.iloc[-1] if len(features) > 0 else pd.Series()

        if len(latest) == 0:
            return {"regime": "unknown", "confidence": 0.0, "scores": {}}

        # NaN-safe accessor: pandas .get() returns NaN rather than the default
        def _safe(key, default=0):
            val = latest.get(key, default)
            if pd.notna(val):
                return float(val)
            return float(default)

        # Initialize scores
        regime_scores = {regime: 0.0 for regime in self.regime_types}

        # 1. Trend-based classification
        trend_21d = _safe("trend_21d", 0)
        trend_strength = _safe("trend_strength", 0)
        trend_direction = _safe("trend_direction", 0)

        if abs(trend_21d) > self.trend_thresholds["strong_trend"] and trend_strength > 0.5:
            if trend_direction > 0:
                regime_scores["trending_bull"] += 0.4
            else:
                regime_scores["trending_bear"] += 0.4
        elif abs(trend_21d) < self.trend_thresholds["no_trend"] and trend_strength < 0.3:
            regime_scores["mean_reverting"] += 0.3

        # 2. Volatility-based classification
        volatility = _safe("volatility_21d", 0.15)

        if volatility > self.volatility_thresholds["crisis_vol"]:
            regime_scores["crisis"] += 0.5
        elif volatility > self.volatility_thresholds["high_vol"]:
            regime_scores["high_volatility"] += 0.4
        elif volatility < self.volatility_thresholds["low_vol"]:
            regime_scores["low_volatility"] += 0.4

        # 3. Mean reversion indicators
        hurst = _safe("hurst_exponent", 0.5)
        half_life = _safe("half_life", 100)
        z_score = _safe("z_score", 0)

        if hurst < 0.5 and half_life < 20:
            regime_scores["mean_reverting"] += 0.3

        if abs(z_score) > 2.0:
            regime_scores["mean_reverting"] += 0.2

        # 4. Market breadth
        breadth = _safe("advance_decline_ratio", 0.5)
        if 0.4 < breadth < 0.6:
            regime_scores["mean_reverting"] += 0.1
        elif breadth > 0.7:
            regime_scores["trending_bull"] += 0.1
        elif breadth < 0.3:
            regime_scores["trending_bear"] += 0.1

        # 5. Technical indicators
        rsi = _safe("rsi", 50)
        if rsi > 70:
            regime_scores["trending_bull"] += 0.1
        elif rsi < 30:
            regime_scores["trending_bear"] += 0.1

        # 6. Crisis detection
        crisis_score = 0
        if volatility > self.volatility_thresholds["crisis_vol"]:
            crisis_score += 0.3
        if _safe("var_95", 0) < -0.20:
            crisis_score += 0.2
        if _safe("kurtosis", 0) > 3:
            crisis_score += 0.1

        if crisis_score > 0.4:
            regime_scores["crisis"] = max(regime_scores["crisis"], crisis_score)

        # 7. Recovery detection
        if len(self.regime_history) >= 3:
            recent_regimes = [h["regime"] for h in self.regime_history[-3:]]
            if "crisis" in recent_regimes and volatility < self.volatility_thresholds["high_vol"]:
                regime_scores["recovery"] += 0.3

        # Normalize scores
        total_score = sum(regime_scores.values())
        if total_score > 0:
            regime_scores = {k: v / total_score for k, v in regime_scores.items()}

        # Find best regime
        best_regime = max(regime_scores, key=regime_scores.get)
        confidence = regime_scores[best_regime]

        return {
            "regime": best_regime,
            "confidence": confidence,
            "scores": regime_scores,
            "method": "statistical",
        }

    def detect_regime_ml(self, features: pd.DataFrame) -> Dict:
        """Detect regime using machine learning"""
        if self.classifier is None or len(features) < self.lookback_period:
            return self.detect_regime_statistical(features)

        if self.feature_columns is None:
            return self.detect_regime_statistical(features)

        try:
            # Prepare features
            available_features = [col for col in self.feature_columns if col in features.columns]
            if not available_features:
                return self.detect_regime_statistical(features)

            X = features[available_features].iloc[-1:].values

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Predict
            probabilities = self.classifier.predict_proba(X_scaled)[0]
            predicted_class = self.classifier.predict(X_scaled)[0]

            # Map back to regime
            regime = self.label_encoder.inverse_transform([predicted_class])[0]
            confidence = probabilities[predicted_class]

            # Create scores
            scores = {self.label_encoder.inverse_transform([i])[0]: prob for i, prob in enumerate(probabilities)}

            return {
                "regime": regime,
                "confidence": confidence,
                "scores": scores,
                "method": "ml",
            }

        except Exception as e:
            logger.error(f"ML detection error: {e}")
            return self.detect_regime_statistical(features)

    def detect_regime_ensemble(self, features: pd.DataFrame) -> Dict:
        """Ensemble of statistical and ML methods"""
        # Get predictions
        statistical_result = self.detect_regime_statistical(features)
        ml_result = self.detect_regime_ml(features) if self.use_ml else statistical_result

        # Weight based on confidence
        stat_weight = statistical_result["confidence"]
        ml_weight = ml_result["confidence"] if self.use_ml else 0

        total_weight = stat_weight + ml_weight

        if total_weight > 0:
            stat_weight /= total_weight
            ml_weight /= total_weight
        else:
            stat_weight = 0.5
            ml_weight = 0.5

        # Combine scores
        combined_scores = {}
        for regime in self.regime_types:
            stat_score = statistical_result["scores"].get(regime, 0)
            ml_score = ml_result["scores"].get(regime, 0) if self.use_ml else 0
            combined_scores[regime] = stat_score * stat_weight + ml_score * ml_weight

        # Find best regime
        best_regime = max(combined_scores, key=combined_scores.get)
        confidence = combined_scores[best_regime]

        # Check for transition
        if len(self.regime_history) > 0:
            last_regime = self.regime_history[-1]["regime"]
            if best_regime != last_regime and confidence < 0.6:
                best_regime = "transition"
                confidence = 0.8

        return {
            "regime": best_regime,
            "confidence": confidence,
            "scores": combined_scores,
            "method": "ensemble",
            "statistical_regime": statistical_result["regime"],
            "ml_regime": ml_result["regime"] if self.use_ml else None,
        }

    def get_regime_with_confidence_interval(self, features: pd.DataFrame, n_bootstrap: int = 100) -> Dict:
        """
        Bootstrap confidence intervals for regime detection

        Args:
            features: Feature DataFrame
            n_bootstrap: Number of bootstrap samples

        Returns:
            Dict with regime and confidence distribution
        """
        if len(features) < 50:
            return self.detect_regime_ensemble(features)

        regime_samples = []

        # Bootstrap samples
        for _ in range(n_bootstrap):
            # Resample features (last 50 rows)
            sample_idx = np.random.choice(len(features) - 50, size=min(50, len(features)), replace=True)
            sampled = features.iloc[sample_idx]

            regime_info = self.detect_regime_statistical(sampled)
            regime_samples.append(regime_info["regime"])

        # Calculate distribution
        regime_dist = pd.Series(regime_samples).value_counts(normalize=True)

        return {
            "regime": regime_dist.idxmax(),
            "confidence": regime_dist.max(),
            "distribution": regime_dist.to_dict(),
            "method": "bootstrap",
        }

    def detect_regime_change_warning(self, features: pd.DataFrame, lookback: int = 20) -> Dict:
        """
        Detect early signs of regime change

        Args:
            features: Current features
            lookback: Lookback period for analysis

        Returns:
            Dict with warning information
        """
        if len(self.regime_history) < lookback:
            return {
                "warning": False,
                "confidence_trend": 0,
                "disagreement_rate": 0,
                "recommendation": "maintain",
            }

        # Analyze recent confidences
        recent_confidences = [h["confidence"] for h in self.regime_history[-lookback:]]

        # Confidence trend (negative = declining)
        confidence_trend = np.polyfit(range(len(recent_confidences)), recent_confidences, 1)[0]

        # Method disagreement
        recent_regimes = self.regime_history[-lookback:]
        disagreements = []

        for h in recent_regimes:
            if "statistical_regime" in h and "ml_regime" in h:
                if h["statistical_regime"] != h["ml_regime"]:
                    disagreements.append(1)
                else:
                    disagreements.append(0)

        disagreement_rate = np.mean(disagreements) if disagreements else 0

        # Warning triggers
        warning = confidence_trend < -0.01 or disagreement_rate > 0.3

        # Recommendation
        if warning:
            if confidence_trend < -0.02:
                recommendation = "increase_cash_significantly"
            else:
                recommendation = "increase_cash"
        else:
            recommendation = "maintain"

        return {
            "warning": warning,
            "confidence_trend": confidence_trend,
            "disagreement_rate": disagreement_rate,
            "recommendation": recommendation,
        }

    def calculate_regime_strength(self, features: pd.DataFrame) -> float:
        """
        Calculate how strongly the current regime is presenting itself

        Args:
            features: Current features

        Returns:
            Strength score (average z-score of key features)
        """
        if len(features) == 0:
            return 0.0

        regime_info = self.detect_regime_ensemble(features)
        regime = regime_info["regime"]

        # Define key features for each regime
        regime_feature_map = {
            "trending_bull": ["trend_21d", "trend_strength", "rsi"],
            "trending_bear": ["trend_21d", "trend_strength", "rsi"],
            "high_volatility": ["volatility_21d", "volatility_ratio"],
            "low_volatility": ["volatility_21d", "volatility_ratio"],
            "mean_reverting": ["hurst_exponent", "half_life", "z_score"],
            "crisis": ["volatility_21d", "var_95", "kurtosis"],
            "recovery": ["trend_21d", "volatility_21d"],
            "transition": ["volatility_ratio", "trend_strength"],
        }

        key_features = regime_feature_map.get(regime, [])

        if not key_features:
            return 0.0

        # Calculate z-scores
        z_scores = []
        for feat in key_features:
            if feat in features.columns:
                feat_values = features[feat].dropna()
                if len(feat_values) > 1 and feat_values.std() > 0:
                    latest_val = feat_values.iloc[-1]
                    if pd.notna(latest_val):
                        z = (latest_val - feat_values.mean()) / feat_values.std()
                        z_scores.append(abs(z))

        return np.mean(z_scores) if z_scores else 0.0

    def predict_regime_duration(self) -> Optional[Dict]:
        """
        Predict how long current regime will last

        Returns:
            Dict with duration predictions or None
        """
        if len(self.regime_history) < 10:
            return self._predict_regime_duration()

        current_regime = self.regime_history[-1]["regime"]

        # Get historical durations
        durations = self._get_historical_durations(current_regime)

        if not durations or len(durations) < 3:
            return self._predict_regime_duration()

        # Calculate statistics
        mean_duration = np.mean(durations)
        median_duration = np.median(durations)
        std_duration = np.std(durations)

        # Simple exponential decay probability
        # P(regime ends in next N days) = 1 - exp(-N/mean_duration)
        prob_end_next_week = 1 - np.exp(-5 / mean_duration) if mean_duration > 0 else 0.5

        return {
            "current_regime": current_regime,
            "expected_duration": mean_duration,
            "median_duration": median_duration,
            "std_duration": std_duration,
            "probability_end_next_week": round(prob_end_next_week),
            "sample_size": len(durations),
        }

    def get_transition_probabilities(self) -> Dict[str, Dict[str, float]]:
        """
        Get transition probabilities using Markov chain analysis

        Returns:
            Dictionary mapping from_regime -> to_regime -> probability
        """
        if len(self.regime_history) < 2:
            logger.warning("Insufficient regime history for transition probabilities")
            return self.markov_chain._empty_transition_matrix()

        # Extract regime sequence
        regime_sequence = [entry["regime"] for entry in self.regime_history]

        # Fit Markov chain if needed or use cached
        if self.markov_chain.transition_matrix is None:
            self.markov_chain.fit(regime_sequence)
        else:
            # Update with new data
            self.markov_chain.fit(regime_sequence)

        return self.markov_chain.get_transition_probabilities()

    def _predict_regime_duration(self) -> Dict[str, float]:
        """
        Predict expected duration of current regime using Markov transitions

        Returns:
            Dictionary with duration statistics
        """
        if len(self.regime_history) < 1:
            return {"expected_duration": 0, "median_duration": 0, "probability_end_next_week": 0}
        current_regime = self.regime_history[-1]["regime"]

        # Calculate/update Markov chain if necessary
        regime_sequence = [entry["regime"] for entry in self.regime_history]
        self.markov_chain.fit(regime_sequence)

        # Get expected duration from Markov chain
        expected_duration = self.markov_chain.get_expected_duration(current_regime)

        if expected_duration == float("inf"):
            expected_duration = 30  # Cap at reasonable maximum

        # Calculate current run length
        current_run = 0
        for entry in reversed(self.regime_history):
            if entry["regime"] == current_regime:
                current_run += 1
            else:
                break

        # Calculate probability of ending within next week
        if expected_duration <= current_run:
            prob_end_next_week = 0.9
        else:
            # Simple geometric probability
            p_self = self.markov_chain.get_regime_persistence().get(current_regime, 0.8)
            prob_end_next_week = 1 - (p_self**5)

        return {
            "current_regime": current_regime,
            "expected_duration": round(expected_duration, 2),
            "median_duration": round(expected_duration * 0.7, 2),  # Approximate median
            "probability_end_next_week": round(min(prob_end_next_week, 0.95), 3),
        }

    def _get_historical_durations(self, regime: str) -> List[int]:
        """Get historical durations for a specific regime"""
        durations = []
        current_duration = 0
        current_regime = None

        for entry in self.regime_history:
            if entry["regime"] == regime:
                if current_regime == regime:
                    current_duration += 1
                else:
                    if current_duration > 0:
                        durations.append(current_duration)
                    current_regime = regime
                    current_duration = 1
            else:
                if current_regime == regime and current_duration > 0:
                    durations.append(current_duration)
                current_regime = entry["regime"]
                current_duration = 0

        # Add final duration
        if current_regime == regime and current_duration > 0:
            durations.append(current_duration)

        return durations

    def _create_allocation_templates(self) -> Dict:
        """Create strategy allocation templates for each regime"""
        templates = {
            "trending_bull": {
                "trend_following": 0.40,
                "momentum": 0.30,
                "volatility_strategies": 0.10,
                "mean_reversion": 0.05,
                "statistical_arbitrage": 0.05,
                "cash": 0.10,
            },
            "trending_bear": {
                "trend_following": 0.30,
                "momentum": 0.20,
                "volatility_strategies": 0.25,
                "mean_reversion": 0.05,
                "statistical_arbitrage": 0.10,
                "cash": 0.10,
            },
            "mean_reverting": {
                "trend_following": 0.10,
                "momentum": 0.15,
                "volatility_strategies": 0.15,
                "mean_reversion": 0.40,
                "statistical_arbitrage": 0.15,
                "cash": 0.05,
            },
            "high_volatility": {
                "trend_following": 0.20,
                "momentum": 0.15,
                "volatility_strategies": 0.35,
                "mean_reversion": 0.10,
                "statistical_arbitrage": 0.10,
                "cash": 0.10,
            },
            "low_volatility": {
                "trend_following": 0.25,
                "momentum": 0.30,
                "volatility_strategies": 0.10,
                "mean_reversion": 0.25,
                "statistical_arbitrage": 0.05,
                "cash": 0.05,
            },
            "crisis": {
                "trend_following": 0.10,
                "momentum": 0.05,
                "volatility_strategies": 0.50,
                "mean_reversion": 0.00,
                "statistical_arbitrage": 0.05,
                "cash": 0.30,
            },
            "recovery": {
                "trend_following": 0.30,
                "momentum": 0.35,
                "volatility_strategies": 0.15,
                "mean_reversion": 0.10,
                "statistical_arbitrage": 0.05,
                "cash": 0.05,
            },
            "transition": {
                "trend_following": 0.15,
                "momentum": 0.15,
                "volatility_strategies": 0.20,
                "mean_reversion": 0.15,
                "statistical_arbitrage": 0.15,
                "cash": 0.20,
            },
        }

        return templates

    def get_strategy_allocation(self, regime: str, confidence: float) -> Dict:
        """Get strategy allocation for detected regime"""
        if regime not in self.strategy_allocation_templates:
            regime = "transition"

        allocation = self.strategy_allocation_templates[regime].copy()

        # Adjust based on confidence
        if confidence < self.confidence_threshold:
            cash_adjustment = (self.confidence_threshold - confidence) * 0.5
            for strategy in allocation:
                if strategy != "cash":
                    allocation[strategy] *= 1 - cash_adjustment
            allocation["cash"] += cash_adjustment

        # Normalize
        total = sum(allocation.values())
        if total > 0:
            allocation = {k: v / total for k, v in allocation.items()}

        return allocation

    def _backfill_history(self, features: pd.DataFrame):
        """Backfill regime history using pre-calculated features"""
        if len(self.regime_history) > 0 or len(features) == 0:
            return

        # Take up to 500 days of history
        historical_features = features.iloc[-500:-1]  # Exclude current day

        for i in range(len(historical_features)):
            row_df = historical_features.iloc[i : i + 1]
            if self.use_ml and self.classifier is not None:
                # We skip ML ensemble for backfill as it needs full scaled features,
                # statistical is sufficient for historical transition probabilities
                regime_info = self.detect_regime_statistical(row_df)
            else:
                regime_info = self.detect_regime_statistical(row_df)

            allocation = self.get_strategy_allocation(regime_info["regime"], regime_info["confidence"])

            self.regime_history.append(
                {
                    "timestamp": row_df.index[0],
                    "regime": regime_info["regime"],
                    "confidence": regime_info["confidence"],
                    "allocation": allocation,
                    "strength": 0.0,
                    "statistical_regime": regime_info.get("statistical_regime"),
                    "ml_regime": regime_info.get("ml_regime"),
                }
            )

    def detect_current_regime(
        self,
        price_data: Union[pd.Series, pd.DataFrame],
        volume_data: Optional[pd.Series] = None,
        update_history: bool = True,
    ) -> Dict:
        """
        Detect current market regime

        Args:
            price_data: Current price data
            volume_data: Optional volume data
            update_history: Whether to update regime history

        Returns:
            Dictionary with regime information
        """
        # Calculate features
        features = self.calculate_features(price_data, volume_data)

        if len(features) == 0:
            return {
                "regime": "unknown",
                "confidence": 0.0,
                "scores": {},
                "method": "insufficient_data",
            }

        # Backfill history if empty (requires at least 10 rows to build Markov chain)
        if update_history and len(self.regime_history) == 0:
            self._backfill_history(features)

        # Detect regime
        if self.use_ml and self.classifier is not None:
            regime_info = self.detect_regime_ensemble(features)
        else:
            regime_info = self.detect_regime_statistical(features)

        # Get strategy allocation
        allocation = self.get_strategy_allocation(regime_info["regime"], regime_info["confidence"])

        # Update history before calculating duration/warning so they can use the latest info
        current_ts = price_data.index[-1] if hasattr(price_data, "index") else len(price_data)

        if update_history:
            # Prevent duplicate entries for the same timestamp from parallel API calls
            if not self.regime_history or self.regime_history[-1]["timestamp"] != current_ts:
                history_entry = {
                    "timestamp": current_ts,
                    "regime": regime_info["regime"],
                    "confidence": regime_info["confidence"],
                    "allocation": allocation,
                    # calculate_regime_strength is slow, but we do it for the current day
                    "strength": self.calculate_regime_strength(features),
                    "statistical_regime": regime_info.get("statistical_regime"),
                    "ml_regime": regime_info.get("ml_regime"),
                }
                self.regime_history.append(history_entry)

                if len(self.regime_history) > 1000:
                    self.regime_history = self.regime_history[-1000:]

        # Calculate additional metrics
        regime_strength = self.calculate_regime_strength(features)
        duration_pred = self.predict_regime_duration()
        warning = self.detect_regime_change_warning(features)

        # Compile full results
        regime_info["strategy_allocation"] = allocation
        regime_info["regime_strength"] = regime_strength
        regime_info["duration_prediction"] = duration_pred
        regime_info["change_warning"] = warning

        return regime_info

    def train_ml_model(
        self,
        historical_data: pd.DataFrame,
        volume_data: Optional[pd.Series] = None,
        labels: Optional[pd.Series] = None,
    ):
        """Train ML model for regime classification"""
        if not self.use_ml:
            return

        # Calculate features
        features = self.calculate_features(historical_data, volume_data)

        if labels is None:
            # Generate labels using statistical method
            labels = self._generate_labels(features)

        # Prepare data
        X = features.dropna()
        y = labels.loc[X.index]

        if len(X) < 100 or len(np.unique(y)) < 3:
            logger.info("Insufficient data for ML training")
            return

        # Store feature columns
        self.feature_columns = X.columns.tolist()

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Train classifier
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
        )

        self.classifier.fit(X_scaled, y_encoded)
        self.last_training_date = historical_data.index[-1]

        logger.info(f"ML model trained on {len(X)} samples")

    def _generate_labels(self, features: pd.DataFrame) -> pd.Series:
        """Generate regime labels for training"""
        labels = pd.Series(index=features.index, dtype="object")

        for i in range(len(features)):
            if i < self.lookback_period:
                continue

            feature_slice = features.iloc[: i + 1]
            regime_info = self.detect_regime_statistical(feature_slice)
            labels.iloc[i] = regime_info["regime"]

        return labels.dropna()

    def generate_regime_report(self) -> Dict:
        """Generate comprehensive regime analysis report"""
        if len(self.regime_history) == 0:
            return {"error": "No regime history available"}

        current = self.regime_history[-1]
        regimes = [entry["regime"] for entry in self.regime_history]

        # Statistics
        regime_counts = pd.Series(regimes).value_counts().to_dict()

        # Average durations
        avg_durations = {}
        for regime in set(regimes):
            durations = self._get_historical_durations(regime)
            avg_durations[regime] = np.mean(durations) if durations else 0

        return {
            "current_regime": current["regime"],
            "current_confidence": current["confidence"],
            "current_strength": current.get("strength", 0),
            "current_allocation": current.get("allocation", {}),
            "regime_statistics": {
                "counts": regime_counts,
                "percentages": {k: v / len(regimes) for k, v in regime_counts.items()},
                "avg_durations": avg_durations,
            },
            "duration_prediction": current.get("duration_prediction"),
            "change_warning": current.get("change_warning"),
            "history_length": len(self.regime_history),
        }
