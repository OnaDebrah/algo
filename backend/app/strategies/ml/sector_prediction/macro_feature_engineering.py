import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MacroFeatureEngine:
    """
    Macro feature engineering for sector prediction
    Transforms raw macro data into predictive features
    """

    def __init__(self, lookback_years: int = 10, use_pca: bool = False, n_pca_components: int = 5, use_regime_features: bool = True):
        self.lookback_years = lookback_years
        self.use_pca = use_pca
        self.n_pca_components = n_pca_components
        self.use_regime_features = use_regime_features

        self.feature_cache = {}
        self.pca_model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}

    async def engineer_features(self, macro_data: pd.DataFrame, market_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Transform raw macro data into predictive features

        Args:
            macro_data: DataFrame with macro indicators
            market_data: Optional market data for additional features

        Returns:
            DataFrame with engineered features
        """
        if macro_data.empty:
            return pd.DataFrame()

        features = macro_data.copy()

        momentum_features = self._calculate_momentum_features(features)
        features = pd.concat([features, momentum_features], axis=1)

        vol_features = self._calculate_volatility_features(features)
        features = pd.concat([features, vol_features], axis=1)

        interaction_features = self._calculate_interaction_features(features)
        features = pd.concat([features, interaction_features], axis=1)

        if self.use_regime_features:
            regime_features = self._calculate_regime_features(features)
            features = pd.concat([features, regime_features], axis=1)

        leading_features = self._calculate_leading_indicators(features)
        features = pd.concat([features, leading_features], axis=1)

        if market_data is not None:
            market_features = self._derive_from_market(market_data)
            features = pd.concat([features, market_features], axis=1, join="inner")

        if self.use_pca:
            features = self._apply_pca(features)

        features = self._handle_missing(features)

        return features

    def _calculate_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum features for macro indicators"""
        features = pd.DataFrame(index=data.index)

        # Momentum for key indicators
        momentum_windows = [1, 3, 6, 12]  # months

        for col in data.columns:
            # Skip if not numeric
            if not np.issubdtype(data[col].dtype, np.number):
                continue

            for window in momentum_windows:
                if len(data) > window:
                    # Simple momentum
                    momentum = data[col].pct_change(window)
                    features[f"{col}_mom_{window}m"] = momentum

                    # Acceleration (change in momentum)
                    if len(data) > window * 2:
                        accel = momentum.diff(window)
                        features[f"{col}_accel_{window}m"] = accel

                    # Z-score of momentum
                    if len(data) > window * 3:
                        zscore = (momentum - momentum.rolling(window * 3).mean()) / momentum.rolling(window * 3).std()
                        features[f"{col}_mom_zscore_{window}m"] = zscore

        return features

    def _calculate_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility features"""
        features = pd.DataFrame(index=data.index)

        vol_windows = [3, 6, 12]

        for col in data.columns:
            if not np.issubdtype(data[col].dtype, np.number):
                continue

            # Calculate rolling volatility
            for window in vol_windows:
                if len(data) > window:
                    vol = data[col].rolling(window).std()
                    features[f"{col}_vol_{window}m"] = vol

                    # Volatility percentile
                    if len(data) > window * 2:
                        vol_percentile = vol.rolling(window * 2).apply(lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100)
                        features[f"{col}_vol_percentile_{window}m"] = vol_percentile

        # Volatility regime (high/low)
        if "vix" in data.columns:
            vix = data["vix"]

            # VIX percentile
            features["vix_percentile"] = vix.rolling(252).apply(lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100)

            # VIX regime
            features["vix_high"] = (vix > vix.rolling(252).quantile(0.75)).astype(int)
            features["vix_low"] = (vix < vix.rolling(252).quantile(0.25)).astype(int)

        return features

    def _calculate_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate interaction features between indicators"""
        features = pd.DataFrame(index=data.index)

        # Key economic interactions
        if all(col in data.columns for col in ["gdp_growth", "unemployment_rate"]):
            # Okun's law relationship
            features["okun_coefficient"] = data["gdp_growth"] / (data["unemployment_rate"] + 0.01)

        if all(col in data.columns for col in ["cpi_yoy", "fed_funds_rate"]):
            # Real interest rate
            features["real_rate"] = data["fed_funds_rate"] - data["cpi_yoy"]

        if all(col in data.columns for col in ["2y10y_spread", "gdp_growth"]):
            # Yield curve vs growth
            features["yield_curve_growth_ratio"] = data["2y10y_spread"] / (data["gdp_growth"] + 0.01)

        if all(col in data.columns for col in ["manufacturing_pmi", "services_pmi"]):
            # PMI spread
            features["pmi_spread"] = data["manufacturing_pmi"] - data["services_pmi"]

        if all(col in data.columns for col in ["vix", "fed_funds_rate"]):
            # Risk vs policy
            features["risk_policy_ratio"] = data["vix"] / (data["fed_funds_rate"] * 100 + 1)

        # Rolling correlations
        corr_windows = [6, 12]
        for window in corr_windows:
            if len(data) > window:
                # GDP vs inflation correlation
                if "gdp_growth" in data.columns and "cpi_yoy" in data.columns:
                    features[f"gdp_cpi_corr_{window}m"] = data["gdp_growth"].rolling(window).corr(data["cpi_yoy"])

                # Rates vs inflation correlation
                if "fed_funds_rate" in data.columns and "cpi_yoy" in data.columns:
                    features[f"rates_inflation_corr_{window}m"] = data["fed_funds_rate"].rolling(window).corr(data["cpi_yoy"])

        return features

    def _calculate_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate market regime features"""
        features = pd.DataFrame(index=data.index)

        # Growth regime
        if "gdp_growth" in data.columns:
            gdp = data["gdp_growth"]

            # Above/below trend
            if len(data) > 36:
                gdp_trend = gdp.rolling(36).mean()
                features["gdp_above_trend"] = (gdp > gdp_trend).astype(int)
                features["gdp_gap"] = gdp - gdp_trend

            # Growth accelerating/decelerating
            if len(data) > 6:
                gdp_momentum = gdp.diff(3)
                features["gdp_accelerating"] = (gdp_momentum > 0).astype(int)

        # Inflation regime
        if "cpi_yoy" in data.columns:
            cpi = data["cpi_yoy"]

            # Inflation regime classification
            features["inflation_high"] = (cpi > 0.03).astype(int)  # >3%
            features["inflation_low"] = (cpi < 0.01).astype(int)  # <1%
            features["inflation_moderate"] = ((cpi >= 0.01) & (cpi <= 0.03)).astype(int)

            # Inflation trend
            if len(data) > 12:
                cpi_trend = cpi.rolling(12).mean()
                features["inflation_rising"] = (cpi > cpi_trend).astype(int)

        # Policy regime
        if "fed_funds_rate" in data.columns:
            rates = data["fed_funds_rate"]

            # Tightening/Easing
            if len(data) > 6:
                rate_change = rates.diff(3)
                features["tightening"] = (rate_change > 0.0025).astype(int)  # >25bps
                features["easing"] = (rate_change < -0.0025).astype(int)

        # Combined regimes
        if "gdp_above_trend" in features.columns and "inflation_high" in features.columns:
            # Stagflation: low growth + high inflation
            features["stagflation"] = ((~features["gdp_above_trend"].astype(bool)) & features["inflation_high"].astype(bool)).astype(int)

            # Goldilocks: good growth + moderate inflation
            features["goldilocks"] = (features["gdp_above_trend"].astype(bool) & features["inflation_moderate"].astype(bool)).astype(int)

        return features

    def _calculate_leading_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate leading indicator composites"""
        features = pd.DataFrame(index=data.index)

        # Composite leading indicator (CLI)
        cli_components = []

        if "manufacturing_pmi" in data.columns:
            cli_components.append(data["manufacturing_pmi"].fillna(50))

        if "consumer_confidence" in data.columns:
            cli_components.append(data["consumer_confidence"].fillna(100) / 2)

        if "housing_starts" in data.columns:
            cli_components.append(data["housing_starts"].pct_change(12).fillna(0) * 100)

        if cli_components:
            features["composite_leading_indicator"] = np.mean(cli_components, axis=0)

            # CLI momentum
            if len(features) > 3:
                features["cli_momentum"] = features["composite_leading_indicator"].diff(3)

        # Yield curve signals
        if "2y10y_spread" in data.columns:
            spread = data["2y10y_spread"]

            # Yield curve inversion signal
            features["yield_curve_inverted"] = (spread < 0).astype(int)

            # Depth of inversion
            features["inversion_depth"] = np.minimum(spread, 0)

            # Steepening/Flattening
            if len(spread) > 3:
                features["curve_steepening"] = (spread.diff(3) > 0).astype(int)
                features["curve_flattening"] = (spread.diff(3) < 0).astype(int)

        # Credit spread signals
        if "corporate_spread" in data.columns:
            cs = data["corporate_spread"]

            # Credit stress
            if len(cs) > 252:
                cs_percentile = cs.rolling(252).apply(lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100)
                features["credit_stress"] = (cs_percentile > 0.9).astype(int)

        return features

    def _derive_from_market(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Derive macro features from market data"""
        features = pd.DataFrame(index=market_data.index)

        # Market-implied growth
        if "Close" in market_data.columns:
            # Cyclical vs defensive ratio
            # TODO This would require sector indices - simplified version
            features["market_momentum_3m"] = market_data["Close"].pct_change(63)
            features["market_momentum_12m"] = market_data["Close"].pct_change(252)

            # Market volatility regime
            returns = market_data["Close"].pct_change()
            features["market_vol_21d"] = returns.rolling(21).std() * np.sqrt(252)

            # Trend strength
            sma_50 = market_data["Close"].rolling(50).mean()
            sma_200 = market_data["Close"].rolling(200).mean()
            features["trend_strength"] = sma_50 / sma_200 - 1

        return features

    def _apply_pca(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply PCA for dimensionality reduction"""
        # Select numeric columns
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        X = features[numeric_cols].fillna(method="ffill").fillna(0)

        # Scale
        X_scaled = self.scaler.fit_transform(X)

        # Fit or transform PCA
        if self.pca_model is None:
            self.pca_model = PCA(n_components=min(self.n_pca_components, X.shape[1]))
            X_pca = self.pca_model.fit_transform(X_scaled)

            # Store explained variance
            logger.info(f"PCA explained variance: {self.pca_model.explained_variance_ratio_.cumsum()}")
        else:
            X_pca = self.pca_model.transform(X_scaled)

        # Create PCA features
        pca_df = pd.DataFrame(X_pca, index=features.index, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])

        return pca_df

    def _handle_missing(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values intelligently"""
        # Forward fill then backward fill
        features = features.fillna(method="ffill").fillna(method="bfill")

        # Fill any remaining with 0
        features = features.fillna(0)

        return features

    def get_feature_importance(self, target_returns: pd.Series, features: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate feature importance relative to target
        """
        # Align data
        common_idx = features.index.intersection(target_returns.index)
        if len(common_idx) == 0:
            return {}

        X = features.loc[common_idx]
        y = target_returns.loc[common_idx]

        # Calculate correlations
        correlations = {}
        for col in X.columns:
            if X[col].std() > 0:
                corr = X[col].corr(y)
                if not np.isnan(corr):
                    correlations[col] = abs(corr)

        # Sort by importance
        sorted_importance = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

        self.feature_importance = dict(sorted_importance[:20])

        return self.feature_importance

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Group features by type"""
        groups = {"momentum": [], "volatility": [], "interaction": [], "regime": [], "leading": [], "pca": []}

        for feature in self.feature_cache.get("last_features", []):
            if "mom_" in feature or "accel_" in feature:
                groups["momentum"].append(feature)
            elif "vol_" in feature or "vix" in feature:
                groups["volatility"].append(feature)
            elif any(x in feature for x in ["corr_", "spread", "ratio"]):
                groups["interaction"].append(feature)
            elif any(x in feature for x in ["regime", "_high", "_low"]):
                groups["regime"].append(feature)
            elif any(x in feature for x in ["leading", "inverted", "steep"]):
                groups["leading"].append(feature)
            elif feature.startswith("PC"):
                groups["pca"].append(feature)

        return groups
