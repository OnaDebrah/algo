import logging
from datetime import datetime
from typing import Dict

import pandas as pd

from ... import BaseStrategy
from ...parabolic_sar import ParabolicSARStrategy
from ...technical.bb_mean_reversion import BollingerMeanReversionStrategy
from ...technical.rsi_strategy import RSIStrategy
from ...technical.sma_crossover import SMACrossoverStrategy
from ...ts_momentum_strategy import TimeSeriesMomentumStrategy
from ...volatility.volatility_breakout import VolatilityBreakoutStrategy
from .hmm_regime_detector import HMMRegimeDetector

logger = logging.getLogger(__name__)


class RegimeSpecificStrategy(BaseStrategy):
    """
    Strategy that switches between specialist models based on detected regime
    """

    REGIME_STRATEGY_MAP = {
        "bull": {"primary": "ts_momentum", "secondary": "sma_crossover", "position_sizing": 1.0},
        "neutral": {"primary": "bb_mean_reversion", "secondary": "rsi", "position_sizing": 0.7},
        "bear": {"primary": "volatility_breakout", "secondary": "parabolic_sar", "position_sizing": 0.5},
    }

    def __init__(
        self, name: str = "regime_adaptive", lookback_days: int = 252, regime_update_freq: int = 5, use_markov_chain: bool = True, params: Dict = None
    ):
        super().__init__(name, params or {})

        self.lookback_days = lookback_days
        self.regime_update_freq = regime_update_freq
        self.use_markov_chain = use_markov_chain

        self.regime_detector = HMMRegimeDetector(name=f"{name}_hmm", n_regimes=3)

        self.specialist_strategies = self._init_specialist_strategies()

        self.current_regime = None
        self.last_regime_update = None
        self.regime_history = []

    def _init_specialist_strategies(self, symbol: str = "") -> Dict[str, Dict[str, BaseStrategy]]:
        """Initialize specialist strategies for each regime"""
        strategies = {
            "bull": {
                "primary": TimeSeriesMomentumStrategy(asset_symbol=symbol),
                "secondary": SMACrossoverStrategy(short_window=20, long_window=50),
            },
            "neutral": {"primary": BollingerMeanReversionStrategy(), "secondary": RSIStrategy(period=14, oversold=30, overbought=70)},
            "bear": {
                "primary": VolatilityBreakoutStrategy(name="bear_volatility", params={"period": 20, "std_dev": 2.5}),
                "secondary": ParabolicSARStrategy(),
            },
        }

        return strategies

    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """
        Generate trading signal based on current regime
        """

        self._update_regime(data)

        if self.current_regime is None:
            return {"signal": 0, "position_size": 0.0, "metadata": {}}

        # Get regime configuration
        regime_config = self.REGIME_STRATEGY_MAP.get(self.current_regime, self.REGIME_STRATEGY_MAP["neutral"])

        primary_signal = self.specialist_strategies[self.current_regime]["primary"].generate_signal(data)
        secondary_signal = self.specialist_strategies[self.current_regime]["secondary"].generate_signal(data)

        from ...base_strategy import normalize_signal

        primary = normalize_signal(primary_signal)
        secondary = normalize_signal(secondary_signal)

        # Combine signals (primary gets 70% weight)
        combined_signal = 0
        if primary["signal"] != 0 and secondary["signal"] != 0:
            if primary["signal"] == secondary["signal"]:
                combined_signal = primary["signal"]
            else:
                # Conflict - trust primary
                combined_signal = primary["signal"]
        else:
            combined_signal = primary["signal"] if primary["signal"] != 0 else secondary["signal"]

        # Get regime persistence from Markov chain
        if self.use_markov_chain and hasattr(self.regime_detector, "markov_chain"):
            persistence = self.regime_detector.markov_chain.get_regime_persistence()
            current_persistence = persistence.get(self.current_regime, 0.5)
        else:
            current_persistence = 0.5

        trans_probs = self.regime_detector.get_transition_probabilities() if len(self.regime_detector.regime_history) >= 2 else {}

        duration_pred = self.regime_detector.predict_regime_duration() if len(self.regime_detector.regime_history) >= 10 else {}
        duration_pred = duration_pred or {}

        # Adjust position size based on regime confidence and persistence
        base_size = regime_config["position_sizing"]
        confidence = self._get_regime_confidence()

        # Higher position size when regime is persistent
        position_size = base_size * (0.5 + 0.5 * current_persistence) * confidence

        return {
            "signal": combined_signal,
            "position_size": min(position_size, 1.0),
            "metadata": {
                "current_regime": self.current_regime,
                "regime_confidence": confidence,
                "regime_persistence": current_persistence,
                "expected_duration": duration_pred.get("expected_duration", 0),
                "regime_end_prob": duration_pred.get("probability_end_next_week", 0),
                "transition_probabilities": trans_probs,
                "primary_strategy": type(self.specialist_strategies[self.current_regime]["primary"]).__name__,
                "secondary_strategy": type(self.specialist_strategies[self.current_regime]["secondary"]).__name__,
                "primary_signal": primary["signal"],
                "secondary_signal": secondary["signal"],
            },
        }

    def _update_regime(self, data: pd.DataFrame):
        """Update detected regime"""
        current_time = data.index[-1] if hasattr(data.index, "max") else datetime.now()

        if self.last_regime_update is None or (
            hasattr(current_time, "day") and (current_time - self.last_regime_update).days >= self.regime_update_freq
        ):
            # Use enough data for regime detection (need 100+ for rolling features)
            min_bars = max(self.lookback_days, 150)
            recent_data = data.tail(min_bars) if len(data) > min_bars else data

            if len(recent_data) >= 150:  # Need 100+ bars for rolling features in HMM
                regime_df = self.regime_detector.detect_regimes(recent_data)
                if not regime_df.empty:
                    self.current_regime = regime_df["regime"].iloc[-1]
                    self.last_regime_update = current_time
                    logger.info(f"Regime updated: {self.current_regime}")

    def _get_regime_confidence(self) -> float:
        """Get confidence in current regime prediction"""
        if not self.regime_history:
            return 0.5

        latest = self.regime_history[-1]
        if "probabilities" in latest:
            return latest["probabilities"].get(self.current_regime, 0.5)

        return 0.5

    def reset(self):
        """Reset strategy state"""
        self.current_regime = None
        self.last_regime_update = None
        self.regime_history = []
        self.regime_detector = HMMRegimeDetector(name=f"{self.name}_hmm")

        # Reset specialist strategies
        for regime, strategies in self.specialist_strategies.items():
            for strat in strategies.values():
                strat.reset()
