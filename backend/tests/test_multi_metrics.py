import asyncio
import sys
import os

# Add backend to path so imports work
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.app.services.backtest_service import BacktestService
from backend.app.schemas.backtest import MultiAssetBacktestRequest
import logging

logging.basicConfig(level=logging.DEBUG)

async def test_multi_asset_zeros():
    service = BacktestService()

    request = MultiAssetBacktestRequest(
        symbols=["AAPL", "MSFT", "GOOGL"],
        strategy_configs={
            "AAPL": {"strategy_key": "sma_crossover", "parameters": {"short_window": 10, "long_window": 50}},
            "MSFT": {"strategy_key": "sma_crossover", "parameters": {"short_window": 10, "long_window": 50}},
            "GOOGL": {"strategy_key": "sma_crossover", "parameters": {"short_window": 10, "long_window": 50}}
        },
        allocation_method="equal",
        period="2y",
        interval="1d",
        initial_capital=100000,
        commission_rate=0.001,
        slippage_rate=0.0005
    )

    try:
        response = await service.run_multi_asset_backtest(request, user_id=1)
        
        print("\n=== MULTI-ASSET METRICS ===")
        metrics = response.result
        print(f"Total Return: {metrics.total_return:.2f}%")
        print(f"Number of Trades: {metrics.total_trades}")
        print(f"Sortino Ratio: {metrics.sortino_ratio}")
        print(f"Calmar Ratio: {metrics.calmar_ratio}")
        print(f"VaR (95%): {metrics.var_95}")
        print(f"Portfolio Volatility: {metrics.volatility}")
        
        if len(response.trades) > 0:
            print("\nFirst 3 trades:")
            for t in response.trades[:3]:
                print(f"{t.order_type} {t.quantity} {t.symbol} @ {t.price}")
        else:
            print("\nNO TRADES GENERATED!")
            
    except Exception as e:
        print(f"Error running backtest: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_multi_asset_zeros())
