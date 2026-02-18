from types import SimpleNamespace
from typing import Any, Dict, Optional

from backend.app.schemas.backtest import BacktestRequest, StrategyConfig


class RequestAdapter:
    """Adapts different request types to a common interface"""

    @staticmethod
    def to_backtest_request(
        request: Any, params: Dict[str, Any], capital: float, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> BacktestRequest:
        """Convert any request type to BacktestRequest"""

        if isinstance(request, BacktestRequest):
            from copy import deepcopy

            test_request = deepcopy(request)
            if request.symbol in test_request.strategy_configs:
                test_request.strategy_configs[request.symbol].parameters = params
            else:
                test_request.strategy_configs[request.symbol] = StrategyConfig(strategy_key=request.strategy_key, parameters=params)
            test_request.start_date = start_date
            test_request.end_date = end_date
            return test_request

        if hasattr(request, "symbol") and hasattr(request, "strategy_key"):
            return BacktestRequest(
                symbol=request.symbol,
                strategy_key=request.strategy_key,
                parameters=params,
                interval=getattr(request, "interval", "1d"),
                period="1y",
                initial_capital=capital,
                commission_rate=getattr(request, "commission_rate", 0.001),
                slippage_rate=getattr(request, "slippage_rate", 0.001),
                start_date=start_date,
                end_date=end_date,
                strategy_configs={request.symbol: StrategyConfig(strategy_key=request.strategy_key, parameters=params)},
            )

        # If it's a SimpleNamespace or dict-like
        if isinstance(request, (SimpleNamespace, dict)):
            symbol = request.symbol if hasattr(request, "symbol") else request.get("symbol")
            strategy_key = request.strategy_key if hasattr(request, "strategy_key") else request.get("strategy_key")

            return BacktestRequest(
                symbol=symbol,
                strategy_key=strategy_key,
                parameters=params,
                interval=getattr(request, "interval", request.get("interval", "1d")),
                period="1y",
                initial_capital=capital,
                commission_rate=getattr(request, "commission_rate", request.get("commission_rate", 0.001)),
                slippage_rate=getattr(request, "slippage_rate", request.get("slippage_rate", 0.001)),
                start_date=start_date,
                end_date=end_date,
                strategy_configs={symbol: StrategyConfig(strategy_key=strategy_key, parameters=params)},
            )

        raise ValueError(f"Cannot adapt request type: {type(request)}")
