import logging
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.data.fetchers.fundamentals_fetcher import FundamentalsFetcher
from ....core.data.fetchers.macro.macro_manager import MacroManager
from ....core.data_fetcher import fetch_stock_data
from ....utils.helpers import MACRO_INDICATORS
from .macro_feature_engineering import MacroFeatureEngine

logger = logging.getLogger(__name__)


class SectorPredictor:
    """
    Predicts top-performing sectors using macro data and fundamental factors
    """

    def __init__(
        self,
        lookback_years: int = 5,
        forecast_horizon_days: int = 60,
        model_type: str = "random_forest",
        update_frequency_days: int = 30,
        db: AsyncSession = None,
    ):
        self.lookback_years = lookback_years
        self.macro_feature_engine = MacroFeatureEngine(
            lookback_years=self.lookback_years,
            use_pca=True,  # Reduce dimensionality
            use_regime_features=True,
        )
        self.forecast_horizon = forecast_horizon_days
        self.update_frequency = update_frequency_days
        self.model_type = model_type

        self.fundamentals_fetcher = FundamentalsFetcher()
        self.models = {}  # sector -> model
        self.scalers = {}  # sector -> scaler
        self.last_update = None

        self.macro_manager = MacroManager(db)
        self.MACRO_INDICATORS = MACRO_INDICATORS

    async def predict_top_sectors(self, n_sectors: int = 5, include_probabilities: bool = True) -> List[Dict]:
        """
        Predict top N performing sectors

        Returns:
            List of dicts with sector, expected_return, probability, confidence
        """
        await self._maybe_update_models()

        current_features = await self._get_current_features()

        predictions = []
        for sector in await self.fundamentals_fetcher.get_all_sectors():
            if sector in self.models:
                features_scaled = self.scalers[sector].transform(current_features[sector].reshape(1, -1))

                expected_return = self.models[sector].predict(features_scaled)[0]

                # Calculate confidence based on model's prediction variance
                if hasattr(self.models[sector], "estimators_"):
                    # Random Forest: use std dev of trees
                    tree_preds = np.array([tree.predict(features_scaled)[0] for tree in self.models[sector].estimators_])
                    confidence = 1.0 - (tree_preds.std() / abs(expected_return + 1e-6))
                    confidence = np.clip(confidence, 0, 1)
                else:
                    # For other models, use simpler confidence metric
                    confidence = 0.7  # default

                predictions.append(
                    {
                        "sector": sector,
                        "expected_return": expected_return,
                        "confidence": float(confidence),
                        "probability_outperform": float(self._estimate_outperform_probability(sector, expected_return)),
                    }
                )

        # Sort by expected return and take top N
        predictions.sort(key=lambda x: x["expected_return"], reverse=True)
        top_sectors = predictions[:n_sectors]

        return top_sectors

    async def _maybe_update_models(self):
        """Update models if needed based on update frequency"""
        if self.last_update is None:
            await self._train_models()
        elif (datetime.now() - self.last_update).days >= self.update_frequency:
            await self._train_models()

    async def _train_models(self):
        """Train sector prediction models"""
        logger.info("Training sector prediction models...")

        # Get training data
        X, y = await self._prepare_training_data()

        # Train one model per sector (or a multi-output model)
        for sector in X.columns.levels[1] if isinstance(X.columns, pd.MultiIndex) else X.columns:
            sector_X = X.xs(sector, axis=1, level=1) if isinstance(X.columns, pd.MultiIndex) else X
            sector_y = y[sector]

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(sector_X)

            # Train model
            if self.model_type == "random_forest":
                model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
            else:  # gradient_boosting
                model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = []

            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = sector_y.iloc[train_idx], sector_y.iloc[val_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                cv_scores.append(mean_squared_error(y_val, y_pred))

            logger.info(f"Sector {sector}: CV MSE = {np.mean(cv_scores):.4f}")

            # Retrain on full data
            model.fit(X_scaled, sector_y)

            self.models[sector] = model
            self.scalers[sector] = scaler

        self.last_update = datetime.now()
        logger.info("Sector model training complete")

    async def _prepare_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare features and targets for training

        Features: Macro indicators + sector fundamentals
        Targets: Future sector returns
        """
        # Get macro data
        macro_data = await self.fundamentals_fetcher.fetch_macro_data(
            lookback_years=self.lookback_years + 1  # Extra for target creation
        )

        # Get all sectors
        sectors = await self.fundamentals_fetcher.get_all_sectors()

        # Calculate sector returns for each period
        sector_returns = pd.DataFrame(index=macro_data.index)

        for sector in sectors:
            stocks = await self.fundamentals_fetcher.get_sector_universe(sector)
            if not stocks:
                continue

            # Get average sector return
            sector_rets = []
            for stock in stocks[:5]:  # Use top 5 for efficiency
                try:
                    data = await fetch_stock_data(stock, period=f"{self.lookback_years+1}y", interval="1mo")
                    if not data.empty:
                        rets = data["Close"].pct_change(self.forecast_horizon).shift(-self.forecast_horizon)
                        sector_rets.append(rets)
                except Exception as e:
                    logger.debug(f"Error fetching {stock}: {e}")
                    continue

            if sector_rets:
                sector_returns[sector] = pd.concat(sector_rets, axis=1).mean(axis=1)

        # Align macro data with returns
        aligned_data = macro_data.align(sector_returns, join="inner", axis=0)
        X = aligned_data[0]
        y = aligned_data[1]

        # Create lagged features
        for lag in [1, 3, 6]:
            lagged = X.shift(lag)
            lagged.columns = [f"{col}_lag{lag}" for col in lagged.columns]
            X = pd.concat([X, lagged], axis=1)

        # Drop NaN rows
        combined = pd.concat([X, y], axis=1).dropna()
        X = combined[list(X.columns)]
        y = combined[list(y.columns)]

        return X, y

    async def _get_current_features(self) -> Dict[str, np.ndarray]:
        """Get current feature values for prediction"""
        macro_data = await self.fundamentals_fetcher.fetch_macro_data(lookback_years=1)
        latest_features = macro_data.iloc[-1:].copy()

        # Add lagged features
        for lag in [1, 3, 6]:
            lagged = macro_data.shift(lag).iloc[-1:]
            lagged.columns = [f"{col}_lag{lag}" for col in lagged.columns]
            latest_features = pd.concat([latest_features, lagged], axis=1)

        # Return as dict of sector -> features
        sectors = await self.fundamentals_fetcher.get_all_sectors()
        return {sector: latest_features.values.flatten() for sector in sectors}

    def _estimate_outperform_probability(self, sector: str, expected_return: float) -> float:
        """Estimate probability that sector outperforms market"""
        # Simple logistic function based on expected return
        # In practice, this would be calibrated from historical data
        return 1.0 / (1.0 + np.exp(-5 * expected_return))
