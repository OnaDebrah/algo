"""
Historical Crash Prediction Accuracy Service

Runs LPPLS + LSTM models across 18+ years of historical data,
compares signals against known crash events, and computes
accuracy metrics (sensitivity, specificity, lead time, margin of error).
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.data.providers.providers import ProviderFactory
from ...models.user import User
from ...strategies.analysis.lppls_bubbles_strategy import LPPLSBubbleStrategy
from ...strategies.analysis.lstm_stress_strategy import LSTMStressStrategy

logger = logging.getLogger(__name__)

# ── Known Historical Crashes (S&P 500 / SPY) ────────────────────────

KNOWN_CRASHES = [
    {
        "name": "Global Financial Crisis",
        "peak_date": "2007-10-09",
        "trough_date": "2009-03-09",
        "drawdown_pct": -56.8,
        "warning_window_days": 90,
        "category": "systemic",
    },
    {
        "name": "Flash Crash",
        "peak_date": "2010-04-26",
        "trough_date": "2010-07-02",
        "drawdown_pct": -16.0,
        "warning_window_days": 30,
        "category": "shock",
    },
    {
        "name": "EU Debt Crisis",
        "peak_date": "2011-07-07",
        "trough_date": "2011-10-03",
        "drawdown_pct": -19.4,
        "warning_window_days": 60,
        "category": "systemic",
    },
    {
        "name": "China Devaluation",
        "peak_date": "2015-07-20",
        "trough_date": "2015-08-25",
        "drawdown_pct": -12.4,
        "warning_window_days": 30,
        "category": "shock",
    },
    {
        "name": "2018 Q4 Correction",
        "peak_date": "2018-09-20",
        "trough_date": "2018-12-24",
        "drawdown_pct": -19.8,
        "warning_window_days": 60,
        "category": "correction",
    },
    {
        "name": "COVID-19 Crash",
        "peak_date": "2020-02-19",
        "trough_date": "2020-03-23",
        "drawdown_pct": -33.9,
        "warning_window_days": 30,
        "category": "shock",
    },
    {
        "name": "2022 Bear Market",
        "peak_date": "2022-01-03",
        "trough_date": "2022-10-12",
        "drawdown_pct": -25.4,
        "warning_window_days": 90,
        "category": "correction",
    },
]

# ── Cache ────────────────────────────────────────────────────────────

_accuracy_cache: Dict[str, Dict] = {}
_cache_timestamps: Dict[str, datetime] = {}
CACHE_TTL_HOURS = 24


class HistoricalAccuracyService:
    """
    Backtests LPPLS + LSTM crash prediction models across historical
    data and computes accuracy metrics against known crash events.
    """

    def __init__(self):
        self.provider_factory = ProviderFactory()

    async def run_historical_accuracy(
        self,
        symbol: str,
        user: Optional[User] = None,
        db: Optional[AsyncSession] = None,
        stride_days: int = 20,
        threshold: float = 0.33,
        use_proxy: bool = True,
    ) -> Dict:
        """
        Main entry point: run historical backtest and return accuracy data.

        Args:
            symbol: Ticker to backtest (e.g., "SPY")
            user: User for provider selection
            db: Database session
            stride_days: Days between evaluation points (5 = weekly)
            threshold: Probability threshold for crash signal (default 0.33)

        Returns:
            Full accuracy analysis with timeseries, metrics, and per-event breakdown
        """
        cache_key = f"{symbol}:{stride_days}:{threshold}:{use_proxy}"

        # Check cache
        if cache_key in _accuracy_cache:
            cached_time = _cache_timestamps.get(cache_key)
            if cached_time and (datetime.now() - cached_time).total_seconds() < CACHE_TTL_HOURS * 3600:
                logger.info(f"Returning cached accuracy for {cache_key}")
                return _accuracy_cache[cache_key]

        logger.info(f"Running historical accuracy backtest for {symbol} (stride={stride_days}, threshold={threshold})")

        # Step 1: Fetch historical data
        data = await self._fetch_historical_data(symbol, user, db)
        if data.empty or len(data) < 252:
            return {"error": "Insufficient historical data", "symbol": symbol}

        logger.info(f"Fetched {len(data)} bars from {data.index[0]} to {data.index[-1]}")

        # Step 2: Compute ground truth
        ground_truth = self._compute_ground_truth(data)

        # Step 3-4: Run models (CPU-heavy, run in thread pool)
        result = await asyncio.to_thread(
            self._run_backtest_sync,
            data, ground_truth, stride_days, threshold, symbol, use_proxy,
        )

        # Cache result
        _accuracy_cache[cache_key] = result
        _cache_timestamps[cache_key] = datetime.now()

        return result

    async def _fetch_historical_data(
        self, symbol: str, user: Optional[User], db: Optional[AsyncSession]
    ) -> pd.DataFrame:
        """Fetch 18+ years of daily data."""
        start_date = datetime(2007, 1, 1)
        end_date = datetime.now()

        data = await self.provider_factory.fetch_data(
            symbol=symbol,
            period=None,
            interval="1d",
            start=start_date,
            end=end_date,
            user=user,
            db=db,
        )
        return data

    def _compute_ground_truth(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Label each date as crash zone / pre-crash window based on known events.
        Also computes rolling drawdown from 60-day peak.
        """
        # Normalize index to timezone-naive (Yahoo returns tz-aware timestamps)
        idx = data.index.tz_localize(None) if data.index.tz is not None else data.index
        gt = pd.DataFrame(index=idx)
        gt["price"] = data["Close"].values

        # Rolling drawdown
        rolling_peak = gt["price"].rolling(60, min_periods=1).max()
        gt["drawdown"] = (gt["price"] - rolling_peak) / rolling_peak

        # Label crash zones
        gt["is_crash_zone"] = False
        gt["is_pre_crash"] = False
        gt["crash_event"] = ""

        for crash in KNOWN_CRASHES:
            peak = pd.Timestamp(crash["peak_date"])
            trough = pd.Timestamp(crash["trough_date"])
            warning_start = peak - pd.Timedelta(days=crash["warning_window_days"])

            # Crash zone: peak to trough
            crash_mask = (gt.index >= peak) & (gt.index <= trough)
            gt.loc[crash_mask, "is_crash_zone"] = True
            gt.loc[crash_mask, "crash_event"] = crash["name"]

            # Pre-crash warning window: warning_start to peak
            warning_mask = (gt.index >= warning_start) & (gt.index <= peak)
            gt.loc[warning_mask, "is_pre_crash"] = True
            gt.loc[warning_mask, "crash_event"] = crash["name"]

        return gt

    def _run_backtest_sync(
        self,
        data: pd.DataFrame,
        ground_truth: pd.DataFrame,
        stride_days: int,
        threshold: float,
        symbol: str,
        use_proxy: bool = True,
    ) -> Dict:
        """Synchronous backtest (runs in thread pool)."""
        lookback = 252  # 1 year of trading days
        start_idx = lookback

        # Evaluation indices (every stride_days from start_idx)
        eval_indices = list(range(start_idx, len(data), stride_days))
        total_evals = len(eval_indices)
        logger.info(f"Running {total_evals} evaluation points (stride={stride_days})")

        # Results storage
        results = []

        # Normalize price to 100 at start
        price_start = float(data["Close"].iloc[start_idx])

        for count, idx in enumerate(eval_indices):
            if count % 100 == 0:
                logger.info(f"  Progress: {count}/{total_evals} ({count * 100 / total_evals:.0f}%)")

            window = data.iloc[idx - lookback : idx + 1].copy()
            date = data.index[idx]

            # LPPLS prediction (proxy is ~200x faster than full optimization)
            lppls_prob = self._run_lppls_proxy(window) if use_proxy else self._run_lppls(window)

            # LSTM prediction (simplified — use feature-based stress proxy)
            lstm_stress = self._run_lstm_proxy(window)

            # Combined signal (matching hedge_service.py weights)
            combined = 0.6 * lppls_prob + 0.4 * lstm_stress

            price = float(data["Close"].iloc[idx])
            drawdown = float(ground_truth["drawdown"].iloc[idx])
            is_crash = bool(ground_truth["is_pre_crash"].iloc[idx] or ground_truth["is_crash_zone"].iloc[idx])

            results.append({
                "date": date.strftime("%Y-%m-%d"),
                "price": round(price, 2),
                "price_normalized": round((price / price_start) * 100, 2),
                "lppls_prob": round(lppls_prob, 4),
                "lstm_stress": round(lstm_stress, 4),
                "combined_prob": round(combined, 4),
                "drawdown": round(drawdown, 4),
                "is_crash_zone": is_crash,
            })

        logger.info(f"Backtest complete: {len(results)} evaluation points")

        # Calculate metrics
        df_results = pd.DataFrame(results)
        metrics = self._calculate_metrics(df_results, threshold)
        model_comparison = self._calculate_model_comparison(df_results, threshold)
        crash_events = self._per_event_analysis(df_results, threshold)

        return {
            "symbol": symbol,
            "period": {
                "start": results[0]["date"] if results else "",
                "end": results[-1]["date"] if results else "",
            },
            "stride_days": stride_days,
            "threshold": threshold,
            "metrics": metrics,
            "model_comparison": model_comparison,
            "crash_events": crash_events,
            "timeseries": results,
        }

    def _run_lppls(self, window: pd.DataFrame) -> float:
        """Run LPPLS bubble detection on a data window."""
        try:
            strategy = LPPLSBubbleStrategy()
            signal = strategy.generate_signal(window)
            metadata = signal.get("metadata", {})
            prob = metadata.get("crash_probability") or 0.0
            return float(min(max(prob, 0.0), 1.0))
        except Exception as e:
            logger.debug(f"LPPLS failed for window ending {window.index[-1]}: {e}")
            return 0.0

    def _run_lstm_proxy(self, window: pd.DataFrame) -> float:
        """
        Compute stress proxy from market data features.

        Uses a feature-based stress indicator instead of the full LSTM model
        to avoid the cold-start training problem on historical data.
        Combines: realized volatility, drawdown severity, volume spike,
        and momentum breakdown — the same features the LSTM would learn.
        """
        try:
            if "Close" not in window.columns or len(window) < 60:
                return 0.5

            close = window["Close"]
            returns = close.pct_change().dropna()

            # 1. Realized volatility (20-day, annualized)
            vol_20 = returns.tail(20).std() * np.sqrt(252)
            vol_norm = min(vol_20 / 0.4, 1.0)  # Normalize: 40% vol = max stress

            # 2. Drawdown severity
            peak = close.rolling(60, min_periods=1).max()
            drawdown = float(((close - peak) / peak).iloc[-1])
            dd_stress = min(abs(drawdown) / 0.2, 1.0)  # 20% drawdown = max stress

            # 3. Volume spike (if available)
            vol_stress = 0.0
            if "Volume" in window.columns:
                vol_ratio = window["Volume"].iloc[-1] / window["Volume"].rolling(20).mean().iloc[-1]
                vol_stress = min(max(vol_ratio - 1, 0) / 3, 1.0)  # 4x volume = max stress

            # 4. Momentum breakdown (price below 50-day SMA)
            sma_50 = close.rolling(50, min_periods=1).mean().iloc[-1]
            momentum_stress = 1.0 if close.iloc[-1] < sma_50 else 0.0

            # Weighted combination
            stress = 0.35 * vol_norm + 0.30 * dd_stress + 0.15 * vol_stress + 0.20 * momentum_stress
            return float(min(max(stress, 0.0), 1.0))

        except Exception as e:
            logger.debug(f"LSTM proxy failed: {e}")
            return 0.5

    def _run_lppls_proxy(self, window: pd.DataFrame) -> float:
        """
        Fast LPPLS bubble proxy using market features instead of differential_evolution.

        Approximates bubble probability in [0, 1] compatible with real LPPLS output.
        ~0.05s per call vs ~10s for full LPPLS optimization.

        Features approximate what the LPPLS model detects:
        - Super-exponential growth (power law with m < 1)
        - Price far above fundamental trend
        - Log-periodic-like volatility clustering
        - Accelerating returns (approaching critical time tc)
        - Rising volatility with rising price
        """
        try:
            if "Close" not in window.columns or len(window) < 60:
                return 0.0

            close = window["Close"].values.astype(float)
            log_price = np.log(close)
            n = len(close)

            # 1. Super-exponential growth detection (weight: 0.30)
            # Fit quadratic to log(price): log(p) = a*t^2 + b*t + c
            # Positive curvature (a > 0) means price is growing faster than exponential
            t = np.arange(n, dtype=float)
            coeffs = np.polyfit(t, log_price, 2)
            curvature = coeffs[0]
            # Typical bubble curvature: ~1e-5 to 1e-4 per day^2
            super_exp_score = float(min(max(curvature / 5e-5, 0.0), 1.0))

            # 2. Price deviation from trend (weight: 0.25)
            # Z-score of price above EMA — bubbles push price far above trend
            if n >= 200:
                ema = pd.Series(close).ewm(span=200, min_periods=100).mean().values
                rolling_std = pd.Series(close).rolling(60, min_periods=30).std().values
            else:
                ema_span = max(n // 2, 30)
                ema = pd.Series(close).ewm(span=ema_span, min_periods=20).mean().values
                rolling_std = pd.Series(close).rolling(30, min_periods=15).std().values

            std_val = rolling_std[-1] if rolling_std[-1] > 0 else 1e-10
            z_score = (close[-1] - ema[-1]) / std_val
            deviation_score = float(min(max(z_score / 2.0, 0.0), 1.0))

            # 3. Log-periodic oscillation proxy (weight: 0.20)
            # Ratio of short-term to long-term volatility — volatility clustering
            returns = np.diff(log_price)
            if len(returns) >= 60:
                vol_20 = np.std(returns[-20:])
                vol_60 = np.std(returns[-60:])
                vol_ratio = vol_20 / (vol_60 + 1e-10)
                oscillation_score = float(min(max((vol_ratio - 1.0) / 0.5, 0.0), 1.0))
            else:
                oscillation_score = 0.0

            # 4. Return acceleration (weight: 0.15)
            # Short-term momentum exceeding long-term = approaching singularity
            if n >= 60:
                ret_20 = (close[-1] / close[-20] - 1) * (252 / 20)  # Annualized
                ret_60 = (close[-1] / close[-60] - 1) * (252 / 60)  # Annualized
                if ret_60 > 0 and ret_20 > ret_60:
                    accel = (ret_20 - ret_60) / (ret_60 + 1e-10)
                    acceleration_score = float(min(max(accel, 0.0), 1.0))
                else:
                    acceleration_score = 0.0
            else:
                acceleration_score = 0.0

            # 5. Volatility regime (weight: 0.10)
            # Rising vol + rising price = classic bubble indicator
            if n >= 40:
                vol_recent = np.std(returns[-20:]) * np.sqrt(252)
                vol_prior = np.std(returns[-40:-20]) * np.sqrt(252)
                price_up = close[-1] > close[-20]
                vol_rising = vol_recent > vol_prior * 1.1
                vol_regime_score = 1.0 if (price_up and vol_rising) else 0.0
            else:
                vol_regime_score = 0.0

            # Weighted combination
            proxy_prob = (
                0.30 * super_exp_score
                + 0.25 * deviation_score
                + 0.20 * oscillation_score
                + 0.15 * acceleration_score
                + 0.10 * vol_regime_score
            )
            return float(min(max(proxy_prob, 0.0), 1.0))

        except Exception as e:
            logger.debug(f"LPPLS proxy failed: {e}")
            return 0.0

    def _calculate_metrics(self, df: pd.DataFrame, threshold: float) -> Dict:
        """Calculate binary classification metrics."""
        predicted = df["combined_prob"] >= threshold
        actual = df["is_crash_zone"]

        tp = int(((predicted) & (actual)).sum())
        fp = int(((predicted) & (~actual)).sum())
        tn = int(((~predicted) & (~actual)).sum())
        fn = int(((~predicted) & (actual)).sum())

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

        # Lead time analysis per crash event
        lead_times = []
        for crash in KNOWN_CRASHES:
            peak = pd.Timestamp(crash["peak_date"])
            warning_start = peak - pd.Timedelta(days=crash["warning_window_days"])

            # Find predictions in the warning window
            mask = (
                (pd.to_datetime(df["date"]) >= warning_start)
                & (pd.to_datetime(df["date"]) <= peak)
                & (df["combined_prob"] >= threshold)
            )

            if mask.any():
                first_signal = pd.to_datetime(df.loc[mask, "date"].iloc[0])
                lead_days = (peak - first_signal).days
                lead_times.append(lead_days)

        avg_lead = float(np.mean(lead_times)) if lead_times else 0.0
        std_lead = float(np.std(lead_times)) if len(lead_times) > 1 else 0.0

        return {
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "precision": round(precision, 4),
            "false_positive_rate": round(fpr, 4),
            "f1_score": round(f1, 4),
            "avg_lead_time_days": round(avg_lead, 1),
            "lead_time_std_days": round(std_lead, 1),
            "total_predictions": len(df),
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        }

    def _calculate_model_comparison(self, df: pd.DataFrame, threshold: float) -> Dict:
        """Compare individual models (LPPLS, LSTM, Combined)."""
        actual = df["is_crash_zone"]
        comparisons = {}

        for model, col in [("lppls", "lppls_prob"), ("lstm", "lstm_stress"), ("combined", "combined_prob")]:
            predicted = df[col] >= threshold
            tp = int(((predicted) & (actual)).sum())
            fp = int(((predicted) & (~actual)).sum())
            tn = int(((~predicted) & (~actual)).sum())
            fn = int(((~predicted) & (actual)).sum())

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            # Lead time for this model
            lead_times = []
            for crash in KNOWN_CRASHES:
                peak = pd.Timestamp(crash["peak_date"])
                warning_start = peak - pd.Timedelta(days=crash["warning_window_days"])
                mask = (
                    (pd.to_datetime(df["date"]) >= warning_start)
                    & (pd.to_datetime(df["date"]) <= peak)
                    & (df[col] >= threshold)
                )
                if mask.any():
                    first_signal = pd.to_datetime(df.loc[mask, "date"].iloc[0])
                    lead_times.append((peak - first_signal).days)

            comparisons[model] = {
                "sensitivity": round(sensitivity, 4),
                "specificity": round(specificity, 4),
                "avg_lead_time": round(float(np.mean(lead_times)), 1) if lead_times else 0.0,
            }

        return comparisons

    def _per_event_analysis(self, df: pd.DataFrame, threshold: float) -> List[Dict]:
        """Analyze each known crash event individually."""
        events = []
        dates = pd.to_datetime(df["date"])

        for crash in KNOWN_CRASHES:
            peak = pd.Timestamp(crash["peak_date"])
            trough = pd.Timestamp(crash["trough_date"])
            warning_start = peak - pd.Timedelta(days=crash["warning_window_days"])

            # Check if we have data for this event
            if dates.max() < warning_start or dates.min() > trough:
                continue

            # Predictions in warning window
            warning_mask = (dates >= warning_start) & (dates <= peak)
            crash_zone_mask = (dates >= peak) & (dates <= trough)
            full_mask = warning_mask | crash_zone_mask

            if not full_mask.any():
                events.append({
                    "name": crash["name"],
                    "peak_date": crash["peak_date"],
                    "trough_date": crash["trough_date"],
                    "drawdown_pct": crash["drawdown_pct"],
                    "detected": False,
                    "lead_time_days": None,
                    "avg_probability_in_window": 0.0,
                    "peak_probability": 0.0,
                    "lppls_detected": False,
                    "lstm_detected": False,
                    "combined_detected": False,
                })
                continue

            window_data = df[full_mask]

            # Was crash detected (any signal >= threshold in warning window)?
            warning_data = df[warning_mask]
            combined_detected = bool((warning_data["combined_prob"] >= threshold).any()) if len(warning_data) > 0 else False
            lppls_detected = bool((warning_data["lppls_prob"] >= threshold).any()) if len(warning_data) > 0 else False
            lstm_detected = bool((warning_data["lstm_stress"] >= threshold).any()) if len(warning_data) > 0 else False

            # Lead time: first signal in warning window
            lead_time = None
            if combined_detected and len(warning_data) > 0:
                signals = warning_data[warning_data["combined_prob"] >= threshold]
                if len(signals) > 0:
                    first_signal = pd.to_datetime(signals["date"].iloc[0])
                    lead_time = (peak - first_signal).days

            events.append({
                "name": crash["name"],
                "peak_date": crash["peak_date"],
                "trough_date": crash["trough_date"],
                "drawdown_pct": crash["drawdown_pct"],
                "detected": combined_detected,
                "lead_time_days": lead_time,
                "avg_probability_in_window": round(float(window_data["combined_prob"].mean()), 4),
                "peak_probability": round(float(window_data["combined_prob"].max()), 4),
                "lppls_detected": lppls_detected,
                "lstm_detected": lstm_detected,
                "combined_detected": combined_detected,
            })

        return events
