import logging
from typing import Dict, Union

import pandas as pd

from ..strategies.base_strategy import BaseStrategy
from ..strategies.ml_strategy import MLStrategy

logger = logging.getLogger(__name__)


class DynamicStrategy(BaseStrategy):
    """
    Visual Strategy that executes logic defined by connected blocks.
    Supports combining ML models, technical indicators, and logical gates.
    """

    def __init__(self, blocks: list = None, root_block_id: str = "root", name: str = "Visual Strategy Builder", **kwargs):
        params = {"blocks": blocks or [], "root_block_id": root_block_id}
        super().__init__(name, params)
        self.blocks = {b["id"]: b for b in (blocks or [])}
        self.root_block_id = root_block_id
        self._ml_models = {}  # Cache for loaded ML engines

    def _get_ml_strategy(self, model_id: str) -> MLStrategy:
        """Fetch and instantiate an ML strategy instance for a specific model ID"""
        if not model_id:
            return None

        if model_id in self._ml_models:
            return self._ml_models[model_id]

        try:
            from ..api.routes.mlstudio import load_model

            strat = load_model(model_id)
            self._ml_models[model_id] = strat
            return strat
        except Exception as e:
            logger.error(f"Failed to load ML Model {model_id} for DynamicStrategy: {e}")
            return None

    # ── Scalar (loop-based) path ────────────────────────────────────────

    def generate_signal(self, data: pd.DataFrame) -> Union[int, Dict]:
        """Iterative block resolution for a single bar"""
        if not self.root_block_id:
            return 0

        results = {}

        def resolve_block(block_id):
            if block_id in results:
                return results[block_id]

            block = self.blocks.get(block_id)
            if not block:
                return 0

            b_type = block["type"]
            b_params = block.get("params", {})

            res = 0

            if b_type == "output":
                # Root output block — resolve the single input wired into it
                input_id = b_params.get("input")
                if input_id:
                    res = resolve_block(input_id)

            elif b_type == "ml_model":
                ml_strat = self._get_ml_strategy(b_params.get("model_id"))
                if ml_strat:
                    res = ml_strat.generate_signal(data)

            elif b_type == "indicator":
                res = self._eval_indicator(data, b_params)

            elif b_type == "logic":
                op = b_params.get("op", "AND")
                inputs = b_params.get("inputs", [])
                input_vals = [resolve_block(i) for i in inputs]

                if not input_vals:
                    res = 0
                elif op == "AND":
                    res = 1 if all(v > 0 for v in input_vals) else (-1 if all(v < 0 for v in input_vals) else 0)
                elif op == "OR":
                    res = 1 if any(v > 0 for v in input_vals) else (-1 if any(v < 0 for v in input_vals) else 0)
                elif op == "NOT":
                    res = -resolve_block(inputs[0]) if inputs else 0

            elif b_type == "risk":
                # Risk blocks pass the child signal through (stop-loss/take-profit
                # logic is handled at the engine level via metadata)
                input_id = b_params.get("input")
                if input_id:
                    res = resolve_block(input_id)

            results[block_id] = res
            return res

        final_signal = resolve_block(self.root_block_id)
        return final_signal

    def _eval_indicator(self, data: pd.DataFrame, params: Dict) -> int:
        """Simplified indicator evaluator for a single bar"""
        i_type = params.get("type", "rsi").lower()
        period = int(params.get("period", 14))
        op = params.get("op", ">")
        val = float(params.get("val", 50))

        if len(data) < period + 2:
            return 0

        if i_type == "rsi":
            delta = data["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current = rsi.iloc[-1]

            if op == ">":
                return 1 if current > val else 0
            if op == "<":
                return 1 if current < val else 0
            if op == "cross_above":
                return 1 if rsi.iloc[-1] > val and rsi.iloc[-2] <= val else 0
            if op == "cross_below":
                return -1 if rsi.iloc[-1] < val and rsi.iloc[-2] >= val else 0

        elif i_type == "sma":
            sma = data["Close"].rolling(window=period).mean()
            close = data["Close"]

            if op == ">":
                return 1 if close.iloc[-1] > sma.iloc[-1] else -1
            if op == "<":
                return 1 if close.iloc[-1] < sma.iloc[-1] else -1
            if op == "cross_above":
                return 1 if close.iloc[-1] > sma.iloc[-1] and close.iloc[-2] <= sma.iloc[-2] else 0
            if op == "cross_below":
                return -1 if close.iloc[-1] < sma.iloc[-1] and close.iloc[-2] >= sma.iloc[-2] else 0

        elif i_type == "macd":
            close = data["Close"]
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()

            if op == ">" or op == "cross_above":
                if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
                    return 1
                return 1 if op == ">" and macd_line.iloc[-1] > signal_line.iloc[-1] else 0
            if op == "<" or op == "cross_below":
                if macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
                    return -1
                return -1 if op == "<" and macd_line.iloc[-1] < signal_line.iloc[-1] else 0

        return 0

    # ── Vectorized path ─────────────────────────────────────────────────

    def generate_signals_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """Vectorized execution of the block graph for performance"""
        if not self.root_block_id:
            return pd.Series(0, index=data.index)

        block_series = {}

        def resolve_vectorized(block_id):
            if block_id in block_series:
                return block_series[block_id]

            block = self.blocks.get(block_id)
            if not block:
                return pd.Series(0, index=data.index)

            b_type = block["type"]
            b_params = block.get("params", {})

            res = pd.Series(0, index=data.index)

            if b_type == "output":
                input_id = b_params.get("input")
                if input_id:
                    res = resolve_vectorized(input_id)

            elif b_type == "ml_model":
                ml_strat = self._get_ml_strategy(b_params.get("model_id"))
                if ml_strat:
                    if hasattr(ml_strat, "generate_signals_vectorized"):
                        res = ml_strat.generate_signals_vectorized(data)
                    else:
                        # Fallback to scalar
                        res = pd.Series(
                            [ml_strat.generate_signal(data.iloc[: i + 1]) if i >= 30 else 0 for i in range(len(data))],
                            index=data.index,
                        )

            elif b_type == "indicator":
                res = self._eval_indicator_vectorized(data, b_params)

            elif b_type == "logic":
                op = b_params.get("op", "AND")
                inputs = b_params.get("inputs", [])
                input_series = [resolve_vectorized(i) for i in inputs]

                if not input_series:
                    res = pd.Series(0, index=data.index)
                elif op == "AND":
                    buy = pd.Series(True, index=data.index)
                    sell = pd.Series(True, index=data.index)
                    for s in input_series:
                        buy &= s == 1
                        sell &= s == -1
                    res = buy.astype(int) - sell.astype(int)
                elif op == "OR":
                    buy = pd.Series(False, index=data.index)
                    sell = pd.Series(False, index=data.index)
                    for s in input_series:
                        buy |= s == 1
                        sell |= s == -1
                    res = buy.astype(int) - sell.astype(int)
                elif op == "NOT":
                    if inputs:
                        res = -resolve_vectorized(inputs[0])

            elif b_type == "risk":
                input_id = b_params.get("input")
                if input_id:
                    res = resolve_vectorized(input_id)

            block_series[block_id] = res
            return res

        return resolve_vectorized(self.root_block_id)

    def _eval_indicator_vectorized(self, data: pd.DataFrame, params: Dict) -> pd.Series:
        """Vectorized indicator evaluator"""
        i_type = params.get("type", "rsi").lower()
        period = int(params.get("period", 14))
        op = params.get("op", ">")
        val = float(params.get("val", 50))

        signals = pd.Series(0, index=data.index)

        if i_type == "rsi":
            delta = data["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            if op == ">":
                signals[rsi > val] = 1
            elif op == "<":
                signals[rsi < val] = 1
            elif op == "cross_above":
                signals[(rsi > val) & (rsi.shift(1) <= val)] = 1
            elif op == "cross_below":
                signals[(rsi < val) & (rsi.shift(1) >= val)] = -1

        elif i_type == "sma":
            sma = data["Close"].rolling(window=period).mean()
            close = data["Close"]

            if op == ">":
                signals[close > sma] = 1
                signals[close < sma] = -1
            elif op == "<":
                signals[close < sma] = 1
                signals[close > sma] = -1
            elif op == "cross_above":
                signals[(close > sma) & (close.shift(1) <= sma.shift(1))] = 1
            elif op == "cross_below":
                signals[(close < sma) & (close.shift(1) >= sma.shift(1))] = -1

        elif i_type == "macd":
            close = data["Close"]
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()

            if op == ">":
                signals[macd_line > signal_line] = 1
                signals[macd_line < signal_line] = -1
            elif op == "<":
                signals[macd_line < signal_line] = 1
                signals[macd_line > signal_line] = -1
            elif op == "cross_above":
                signals[(macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))] = 1
            elif op == "cross_below":
                signals[(macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))] = -1

        return signals
