import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ....core.data.fetchers.fundamentals_fetcher import FundamentalsFetcher
from ....core.data_fetcher import fetch_stock_data
from ....strategies.ml.stock_selection.dynamic_model_selector import DynamicModelSelector
from ....strategies.ml.stock_selection.ranking_pipeline import FactorEngine, RankingConfig, RankingMethod, RankingPipeline

logger = logging.getLogger(__name__)


class SectorNeutralStockSelector:
    """
    Selects best stocks within a sector using dynamic model selection
    """

    def __init__(
        self,
        lookback_years: int = 5,
        forecast_horizon_days: int = 90,  # Quarterly forecast
        top_percentile: float = 0.2,
        min_stocks: int = 10,
    ):
        self.lookback_years = lookback_years
        self.forecast_horizon = forecast_horizon_days
        self.top_percentile = top_percentile
        self.min_stocks = min_stocks

        self.fundamentals_fetcher = FundamentalsFetcher()
        self.model_selector = DynamicModelSelector()
        self.sector_models = {}  # sector -> model
        self.sector_scalers = {}  # sector -> scaler

        self.ranking_config = RankingConfig(
            method=RankingMethod.ENSEMBLE,
            top_percentile=0.2,
            use_factor_neutralization=True,
            use_winsorization=True,
            score_combination_method="weighted",
            factor_weights={
                "momentum_weighted": 0.3,
                "quality_composite": 0.25,
                "valuation_composite": 0.2,
                "growth_composite": 0.15,
                "sentiment_score": 0.1,
            },
        )

        self.ranking_pipeline = RankingPipeline(config=self.ranking_config, factor_engine=FactorEngine())

    async def select_top_stocks(self, sector: str, n_stocks: Optional[int] = None) -> List[Dict]:
        """
        Select top stocks in a sector based on predicted returns

        Returns:
            List of dicts with symbol, expected_return, confidence, factors
        """
        # Get all stocks in sector
        stocks = await self.fundamentals_fetcher.get_sector_universe(sector)

        if len(stocks) < self.min_stocks:
            logger.warning(f"Sector {sector} has fewer than {self.min_stocks} stocks")
            return []

        fundamentals = await self.fundamentals_fetcher.fetch_fundamentals_batch(stocks, lookback_years=self.lookback_years)

        if fundamentals.empty:
            logger.warning(f"No fundamental data for sector {sector}")
            return []

        # Prepare features and train model if needed
        if sector not in self.sector_models:
            await self._train_sector_model(sector, stocks, fundamentals)

        # Get current features
        current_features = await self._get_current_features(sector, stocks, fundamentals)

        # Generate predictions
        predictions = []
        model = self.sector_models[sector]
        scaler = self.sector_scalers[sector]

        for symbol, features in current_features.items():
            features_scaled = scaler.transform(features.reshape(1, -1))
            expected_return = model.predict(features_scaled)[0]

            # Calculate confidence based on model type
            if hasattr(model, "estimators_"):
                # Random Forest confidence
                tree_preds = np.array([tree.predict(features_scaled)[0] for tree in model.estimators_])
                confidence = 1.0 - (tree_preds.std() / abs(expected_return + 1e-6))
                confidence = np.clip(confidence, 0, 1)
            else:
                # Use prediction interval or simpler metric
                confidence = 0.7

            # Get key contributing factors (simplified SHAP)
            top_factors = await self._get_top_factors(symbol, features)

            predictions.append(
                {
                    "symbol": symbol,
                    "expected_return": float(expected_return),
                    "confidence": float(confidence),
                    "top_factors": top_factors,
                    "rank_score": float(expected_return * confidence),  # Combined score
                }
            )

        # Sort by rank score
        predictions.sort(key=lambda x: x["rank_score"], reverse=True)

        # Select top N
        n_select = n_stocks or max(int(len(stocks) * self.top_percentile), 5)
        top_stocks = predictions[:n_select]

        return top_stocks

    async def _train_sector_model(self, sector: str, stocks: List[str], fundamentals: pd.DataFrame):
        """Train prediction model for a sector"""
        logger.info(f"Training model for sector: {sector}")

        # Prepare features and targets
        X_list = []
        y_list = []

        for symbol in stocks:
            try:
                # Get features for this symbol
                symbol_data = fundamentals.xs(symbol, level="symbol") if "symbol" in fundamentals.index.names else fundamentals

                if symbol_data.empty:
                    continue

                # Calculate forward returns
                price_data = await fetch_stock_data(symbol, period=f"{self.lookback_years+1}y", interval="1d")

                if price_data.empty:
                    continue

                # Align dates
                common_dates = symbol_data.index.intersection(price_data.index)
                if len(common_dates) < 20:
                    continue

                symbol_features = symbol_data.loc[common_dates]

                # Calculate forward returns
                forward_returns = []
                for date in common_dates:
                    date_idx = price_data.index.get_loc(date)
                    future_idx = min(date_idx + self.forecast_horizon, len(price_data) - 1)
                    ret = price_data.iloc[future_idx]["Close"] / price_data.iloc[date_idx]["Close"] - 1
                    forward_returns.append(ret)

                X_list.append(symbol_features)
                y_list.extend(forward_returns)

            except Exception as e:
                logger.debug(f"Error processing {symbol}: {e}")
                continue

        if not X_list:
            raise ValueError(f"No valid training data for sector {sector}")

        # Combine features
        X = pd.concat(X_list, axis=0)
        y = pd.Series(y_list)

        # Remove any remaining NaN
        valid_idx = ~(y.isna() | X.isna().any(axis=1))
        X = X[valid_idx]
        y = y[valid_idx]

        if len(X) < 50:
            raise ValueError(f"Insufficient training samples for sector {sector}")

        best_model_name, best_score = self.model_selector.select_best_model(X, y)

        # Train final model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model_class = self.model_selector.MODELS[best_model_name]
        model = model_class(**self.model_selector.model_configs.get(best_model_name, {}))
        model.fit(X_scaled, y)

        self.sector_models[sector] = model
        self.sector_scalers[sector] = scaler

        logger.info(f"Sector {sector} model trained with {best_model_name}, MSE: {best_score:.6f}")

    async def _get_current_features(self, sector: str, stocks: List[str], fundamentals: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get current features for each stock"""
        current_features = {}

        for symbol in stocks:
            try:
                # Get latest fundamental data
                symbol_data = fundamentals.xs(symbol, level="symbol") if "symbol" in fundamentals.index.names else fundamentals

                if symbol_data.empty:
                    continue

                latest = symbol_data.iloc[-1:].values.flatten()
                current_features[symbol] = latest

            except Exception as e:
                logger.debug(f"Error getting current features for {symbol}: {e}")
                continue

        return current_features

    async def _get_top_factors(self, symbol: str, features: np.ndarray, n: int = 5) -> List[Dict]:
        """Get top contributing factors from trained model's feature importances"""
        # Try to get feature importances from the sector model
        for sector, model in self.sector_models.items():
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                # Get feature names from the scaler or use generic names
                scaler = self.sector_scalers.get(sector)
                if scaler and hasattr(scaler, "feature_names_in_"):
                    feature_names = list(scaler.feature_names_in_)
                else:
                    feature_names = [f"factor_{i}" for i in range(len(importances))]

                # Sort by importance
                sorted_idx = np.argsort(importances)[::-1][:n]
                factors = [{"factor": feature_names[i], "importance": float(importances[i])} for i in sorted_idx]
                return factors

        # Fallback to default ranking factors
        return [
            {"factor": "momentum_weighted", "importance": 0.3},
            {"factor": "quality_composite", "importance": 0.25},
            {"factor": "valuation_composite", "importance": 0.2},
            {"factor": "growth_composite", "importance": 0.15},
            {"factor": "volatility_21d", "importance": 0.1},
        ]
