import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

from ....analytics.market_regime_detector import MarketRegimeDetector

logger = logging.getLogger(__name__)

# Feature tiers — progressively added as sample count grows
_CORE_FEATURES = ["returns_1d", "returns_5d", "volatility_21d", "volume_ratio"]
_EXTENDED_FEATURES = _CORE_FEATURES + ["returns_21d", "price_vs_sma50"]
_FULL_FEATURES = _EXTENDED_FEATURES + ["price_vs_sma100", "high_vol"]

_N_RESTARTS = 5
_MIN_SAMPLES_PER_PARAM = 10


class HMMRegimeDetector(MarketRegimeDetector):
    """
    Regime detector using Hidden Markov Models.
    Extends MarketRegimeDetector with HMM-based regime classification.

    Adaptations for robustness:
    - Rolling windows capped at 100 bars (was 252) to preserve data after dropna
    - Feature count and covariance type adapt to available sample size
    - Multiple random restarts to avoid local optima
    - Falls back to 2-component model if 3-component fails to converge
    - Regime labels assigned by learned state characteristics, not index order
    """

    def __init__(
        self,
        name: str = "hmm_regime_detector",
        n_regimes: int = 3,
        covariance_type: str = "full",
        n_iter: int = 100,
    ):
        super().__init__(name=name)
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.hmm_model = None
        self.scaler = StandardScaler()
        self.regime_names = ["bull", "neutral", "bear"][:n_regimes]
        self._fitted_n_components: int = 0  # actual components used after fallback

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _extract_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Extract features for regime detection.

        Max rolling window is 100 bars, so dropna removes ~100 leading rows.
        """
        features = pd.DataFrame(index=prices.index)

        features["returns_1d"] = prices["Close"].pct_change()
        features["returns_5d"] = prices["Close"].pct_change(5)
        features["returns_21d"] = prices["Close"].pct_change(21)

        features["volatility_21d"] = features["returns_1d"].rolling(21).std()

        features["volume_ratio"] = prices["Volume"] / prices["Volume"].rolling(21).mean()

        sma_50 = prices["Close"].rolling(50).mean()
        sma_100 = prices["Close"].rolling(100).mean()
        features["price_vs_sma50"] = prices["Close"] / sma_50 - 1
        features["price_vs_sma100"] = prices["Close"] / sma_100 - 1

        features["high_vol"] = (features["volatility_21d"] > features["volatility_21d"].rolling(63).quantile(0.75)).astype(int)

        features = features.dropna()
        return features

    def _select_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Select feature subset based on available sample count.

        Fewer samples → fewer features to keep parameter-to-sample ratio healthy.
        """
        n = len(features)
        if n >= 500:
            cols = [c for c in _FULL_FEATURES if c in features.columns]
        elif n >= 200:
            cols = [c for c in _EXTENDED_FEATURES if c in features.columns]
        else:
            cols = [c for c in _CORE_FEATURES if c in features.columns]
        return features[cols]

    # ------------------------------------------------------------------
    # Adaptive model configuration
    # ------------------------------------------------------------------

    def _select_covariance_type(self, n_samples: int, n_features: int, n_components: int) -> str:
        """Choose covariance complexity that the data can support."""

        def _param_count(cov_type: str) -> int:
            # means + covariance params + transition matrix
            base = n_components * n_features + n_components**2
            if cov_type == "full":
                return base + n_components * n_features * (n_features + 1) // 2
            elif cov_type == "diag":
                return base + n_components * n_features
            else:  # spherical
                return base + n_components

        for cov in ("full", "diag", "spherical"):
            if n_samples >= _param_count(cov) * _MIN_SAMPLES_PER_PARAM:
                return cov
        return "spherical"

    # ------------------------------------------------------------------
    # HMM fitting with restarts and fallback
    # ------------------------------------------------------------------

    def _fit_single(self, features_scaled: np.ndarray, n_components: int, cov_type: str, seed: int) -> Optional[hmm.GaussianHMM]:
        """Attempt a single HMM fit. Returns model or None on failure."""
        model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=cov_type,
            n_iter=self.n_iter,
            random_state=seed,
        )
        try:
            model.fit(features_scaled)
            return model
        except (ValueError, np.linalg.LinAlgError) as e:
            logger.debug(f"HMM fit failed (seed={seed}, n={n_components}, cov={cov_type}): {e}")
            return None

    def _fit_with_restarts(self, features_scaled: np.ndarray, n_components: int, cov_type: str) -> Optional[hmm.GaussianHMM]:
        """Try multiple random seeds and return model with best log-likelihood."""
        best_model: Optional[hmm.GaussianHMM] = None
        best_score = -np.inf

        for i in range(_N_RESTARTS):
            model = self._fit_single(features_scaled, n_components, cov_type, seed=42 + i)
            if model is None:
                continue
            try:
                score = model.score(features_scaled)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if score > best_score:
                best_score = score
                best_model = model

        if best_model is not None:
            converged = getattr(getattr(best_model, "monitor_", None), "converged", None)
            if converged is False:
                logger.warning(
                    f"Best HMM (n={n_components}, cov={cov_type}) did not converge "
                    f"in {self.n_iter} iters (score={best_score:.2f}). Using best attempt."
                )
            else:
                logger.info(f"HMM converged (n={n_components}, cov={cov_type}, score={best_score:.2f})")

        return best_model

    def _fit_with_fallback(self, features_scaled: np.ndarray, n_features: int):
        """Try 3-component fit, fall back to 2-component if needed.

        Returns (model, n_components) or (None, 0).
        """
        for n_comp in [self.n_regimes, 2]:
            if n_comp < 2:
                continue
            cov_type = self._select_covariance_type(len(features_scaled), n_features, n_comp)
            logger.debug(f"Attempting HMM fit: n_components={n_comp}, cov={cov_type}, samples={len(features_scaled)}, features={n_features}")
            model = self._fit_with_restarts(features_scaled, n_comp, cov_type)
            if model is not None:
                return model, n_comp

        return None, 0

    # ------------------------------------------------------------------
    # Regime label assignment
    # ------------------------------------------------------------------

    def _assign_regime_labels(self, model: hmm.GaussianHMM, n_components: int) -> Dict[int, str]:
        """Assign bull/neutral/bear labels based on learned state means.

        States are sorted by the mean of the first feature (returns_1d).
        Lowest return → bear, highest → bull.
        """
        # Column 0 is returns_1d (always present as first core feature)
        means = model.means_[:, 0]
        sorted_indices = np.argsort(means)  # ascending: bear, neutral, bull

        if n_components >= 3:
            return {
                int(sorted_indices[0]): "bear",
                int(sorted_indices[1]): "neutral",
                int(sorted_indices[2]): "bull",
            }
        else:  # 2-component
            return {
                int(sorted_indices[0]): "bear",
                int(sorted_indices[1]): "bull",
            }

    # ------------------------------------------------------------------
    # Transition matrix safety
    # ------------------------------------------------------------------

    def _fix_transmat(self, model: hmm.GaussianHMM) -> hmm.GaussianHMM:
        """Fix zero-sum rows in HMM transition matrix.

        After fitting, some rows of transmat_ may sum to 0 if a state was
        never observed.  Replace those rows with a uniform distribution so
        predict() / predict_proba() don't crash with a ValueError.
        """
        transmat = model.transmat_.copy()
        row_sums = transmat.sum(axis=1)

        bad_rows = np.where(np.isclose(row_sums, 0))[0]
        if len(bad_rows) > 0:
            logger.warning(
                f"HMM transmat_ has {len(bad_rows)} zero-sum row(s) "
                f"(states {bad_rows.tolist()} never observed). "
                f"Replacing with uniform distribution."
            )
            n = model.n_components
            for row_idx in bad_rows:
                transmat[row_idx] = 1.0 / n

        # Re-normalize all rows to ensure they sum to exactly 1
        row_sums = transmat.sum(axis=1, keepdims=True)
        transmat = transmat / row_sums
        model.transmat_ = transmat

        return model

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def detect_regimes(self, prices: pd.DataFrame, features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Detect market regimes using HMM.

        Args:
            prices: OHLCV data
            features: Optional pre-computed features

        Returns:
            DataFrame with columns ``["regime", "prob_bull", "prob_neutral", "prob_bear"]``.
            Returns empty DataFrame if insufficient data or all fits fail.
        """
        all_regime_names = ["bull", "neutral", "bear"]
        empty_result = pd.DataFrame(columns=["regime"] + [f"prob_{r}" for r in all_regime_names])

        # --- Feature extraction ---
        if features is None:
            features = self._extract_features(prices)

        if features.empty or len(features) < 10:
            return empty_result

        # --- Adaptive feature selection ---
        features = self._select_features(features)
        n_features = features.shape[1]

        features_scaled = self.scaler.fit_transform(features)

        # --- Fit with restarts + fallback ---
        model, n_components = self._fit_with_fallback(features_scaled, n_features)

        if model is None:
            logger.warning("All HMM fit attempts failed. Returning empty regime result.")
            self.hmm_model = None
            return empty_result

        # Fix transmat safety
        model = self._fix_transmat(model)
        self.hmm_model = model
        self._fitted_n_components = n_components

        # --- Predict ---
        try:
            hidden_states = model.predict(features_scaled)
            state_probs = model.predict_proba(features_scaled)
        except (ValueError, np.linalg.LinAlgError) as e:
            logger.warning(f"HMM predict failed after successful fit: {e}")
            self.hmm_model = None
            return empty_result

        # --- Build result with intelligent labels ---
        label_map = self._assign_regime_labels(model, n_components)

        result = pd.DataFrame(index=features.index)
        result["regime"] = [label_map[s] for s in hidden_states]

        # Always emit all 3 probability columns
        for r in all_regime_names:
            result[f"prob_{r}"] = 0.0

        for state_idx, regime_name in label_map.items():
            result[f"prob_{regime_name}"] = state_probs[:, state_idx]

        # Update regime history
        for idx, row in result.iterrows():
            self.regime_history.append(
                {
                    "date": idx,
                    "regime": row["regime"],
                    "probabilities": {r: row[f"prob_{r}"] for r in all_regime_names},
                }
            )

        return result

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    def predict_next_regime_probs(self) -> Dict[str, float]:
        """Predict probabilities for next regime using HMM transition matrix."""
        all_regime_names = ["bull", "neutral", "bear"]

        if self.hmm_model is None or len(self.regime_history) < 1:
            return {r: 1.0 / len(all_regime_names) for r in all_regime_names}

        current_regime = self.regime_history[-1]["regime"]

        # Build label → state index map
        label_map = self._assign_regime_labels(self.hmm_model, self._fitted_n_components)
        reverse_map = {v: k for k, v in label_map.items()}

        if current_regime not in reverse_map:
            return {r: 1.0 / len(all_regime_names) for r in all_regime_names}

        current_idx = reverse_map[current_regime]
        transition_row = self.hmm_model.transmat_[current_idx]

        probs = {r: 0.0 for r in all_regime_names}
        for state_idx, regime_name in label_map.items():
            probs[regime_name] = float(transition_row[state_idx])

        return probs
