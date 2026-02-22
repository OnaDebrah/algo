import asyncio
import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import logging

from backend.app.schemas.backtest import MultiAssetBacktestRequest
from backend.app.services.backtest_service import BacktestService

logging.basicConfig(level=logging.INFO)


async def test_engine_outputs():
    service = BacktestService()

    request = MultiAssetBacktestRequest(
        symbols=["AAPL", "MSFT", "GOOGL"],
        strategy_configs={
            "AAPL": {"strategy_key": "sma_crossover", "parameters": {"short_window": 10, "long_window": 50}},
            "MSFT": {"strategy_key": "sma_crossover", "parameters": {"short_window": 10, "long_window": 50}},
            "GOOGL": {"strategy_key": "sma_crossover", "parameters": {"short_window": 10, "long_window": 50}},
        },
        allocation_method="equal",
        period="2y",
        interval="1d",
        initial_capital=100000,
        commission_rate=0.001,
        slippage_rate=0.0005,
    )

    engine = await service._create_independent_engine(request)
    await engine.run_backtest(request.symbols, request.period, request.interval)

    print(f"\nEngine completed. Equity curve points: {len(engine.equity_curve)}")
    print("First 3 items of equity curve:")
    for i in range(min(3, len(engine.equity_curve))):
        print(engine.equity_curve[i])

    print("\nLast 3 items of equity curve:")
    for i in range(max(0, len(engine.equity_curve) - 3), len(engine.equity_curve)):
        print(engine.equity_curve[i])

    from backend.app.analytics.performance import calculate_performance_metrics

    metrics = calculate_performance_metrics(engine.trades, engine.equity_curve, request.initial_capital)

    print("\nMetrics Output:")
    for k in ["var_95", "cvar_95", "sortino_ratio", "calmar_ratio", "total_return_pct"]:
        print(f"{k}: {metrics.get(k)}")


if __name__ == "__main__":
    asyncio.run(test_engine_outputs())
