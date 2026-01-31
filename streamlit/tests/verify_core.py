import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

import pandas as pd

from streamlit.core.analyst_agent import FinancialAnalystAgent
from streamlit.core.portfolio_optimizer import PortfolioOptimizer
from streamlit.core.trading_engine import TradingEngine
from streamlit.strategies.sma_crossover import SMACrossoverStrategy


async def verify_optimizer():
    print("Testing PortfolioOptimizer...")
    symbols = ["AAPL", "MSFT", "GOOGL"]
    opt = PortfolioOptimizer(symbols)
    try:
        opt.fetch_data()
        sharpe_results = opt.optimize_sharpe()
        print(f"  Sharpe Optimization Weights: {sharpe_results['weights']}")
        print(f"  Expected Return: {sharpe_results['expected_return']:.2%}")
        print("✅ PortfolioOptimizer looks good!")
    except Exception as e:
        print(f"❌ PortfolioOptimizer failed: {e}")


async def verify_analyst():
    print("Testing AnalystAgent Forecasts...")
    agent = FinancialAnalystAgent()
    ticker = "AAPL"
    try:
        data = await agent._gather_market_data(ticker)
        print(f"  Revenue Forecast: {data.get('revenue_forecast')}")
        print(f"  Earnings Forecast: {data.get('earnings_forecast')}")
        if data.get("revenue_forecast"):
            print("✅ AnalystAgent Data Fetching looks good!")
        else:
            print("⚠️ AnalystAgent Data Fetching: No forecasts found (might be normal for some tickers)")
    except Exception as e:
        print(f"❌ AnalystAgent failed: {e}")


def verify_trading_engine():
    print("Testing TradingEngine Costs...")
    strategy = SMACrossoverStrategy()
    engine = TradingEngine(strategy, initial_capital=100000, commission_rate=0.01, slippage_rate=0.01)

    # Mock data
    data = pd.DataFrame(
        {
            "Close": [100, 101, 102, 103, 104, 105],
            "Open": [100, 101, 102, 103, 104, 105],
            "High": [100, 101, 102, 103, 104, 105],
            "Low": [100, 101, 102, 103, 104, 105],
            "Volume": [1000] * 6,
        },
        index=pd.date_range("2023-01-01", periods=6),
    )

    # Force a buy
    engine.execute_trade("AAPL", 1, 100.0, data.index[0])
    buy_trade = engine.trades[0]
    print(f"  Buy Price (after 1% slippage): {buy_trade['price']}")
    print(f"  Buy Commission: {buy_trade['commission']}")

    # Force a sell
    engine.execute_trade("AAPL", -1, 110.0, data.index[1])
    sell_trade = engine.trades[1]
    print(f"  Sell Price (after 1% slippage): {sell_trade['price']}")
    print(f"  Sell Commission: {sell_trade['commission']}")
    print(f"  Profit: {sell_trade['profit']}")

    if buy_trade["price"] == 101.0 and sell_trade["price"] == 108.9:
        print("✅ TradingEngine costs look good!")
    else:
        print(f"❌ TradingEngine costs mismatch. Buy: {buy_trade['price']}, Sell: {sell_trade['price']}")


async def main():
    await verify_optimizer()
    await verify_analyst()
    verify_trading_engine()


if __name__ == "__main__":
    asyncio.run(main())
