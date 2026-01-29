import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backend.app.core.trading_engine import TradingEngine
from backend.app.core.multi_asset_engine import MultiAssetEngine
from backend.app.core.risk_manager import RiskManager
from backend.app.strategies.base_strategy import BaseStrategy

class MockStrategy(BaseStrategy):
    def __init__(self, name="Mock"):
        super().__init__(name)
    def generate_signal(self, data):
        # Buy on day 2, Sell on day 4
        if len(data) == 2: return 1
        if len(data) == 4: return -1
        return 0

def test_trading_engine():
    print("Testing TradingEngine with commission and slippage...")
    data = pd.DataFrame({
        "Open": [100, 100, 100, 100, 100],
        "High": [105, 105, 105, 105, 105],
        "Low": [95, 95, 95, 95, 95],
        "Close": [100, 100, 100, 100, 100],
        "Volume": [1000, 1000, 1000, 1000, 1000]
    }, index=[datetime.now() - timedelta(days=i) for i in range(5)][::-1])

    strategy = MockStrategy()
    # 1% commission, 1% slippage
    engine = TradingEngine(
        strategy=strategy,
        initial_capital=10000,
        commission_rate=0.01,
        slippage_rate=0.01
    )

    engine.run_backtest("TEST", data)

    print(f"Trades executed: {len(engine.trades)}")
    for t in engine.trades:
        print(f"Type: {t['order_type']}, Price: {t['price']}, Comm: {t['commission']}, Slippage: {t['slippage']}")
    
    # Buy at 100 + 1% slippage = 101. Value: 101 * quantity. Comm: 1% of 101*quantity.
    # Quantity: 10000 * 0.1 (max pos size) / 101 = 1000 / 101 = 9 shares
    # Total cost: 9 * 101 + (9 * 101 * 0.01) = 909 + 9.09 = 918.09
    
    # Sell at 100 - 1% slippage = 99. Revenue: 9 * 99 = 891. Comm: 1% of 891 = 8.91.
    # Net revenue: 891 - 8.91 = 882.09
    
    # Profit: 882.09 - 918.09 = -36.00

    if len(engine.trades) == 2:
        print("✓ Basic trade execution works")
        if engine.trades[0]['commission'] > 0 and engine.trades[0]['slippage'] > 0:
            print("✓ Commission and slippage applied")
    else:
        print("✗ Trade execution failed")

def test_multi_asset_engine():
    print("\nTesting MultiAssetEngine with commission and slippage...")
    strategies = {"AAPL": MockStrategy(), "MSFT": MockStrategy()}
    engine = MultiAssetEngine(
        strategies=strategies,
        initial_capital=20000,
        commission_rate=0.01,
        slippage_rate=0.01
    )
    
    # Mock aligned data
    dates = [datetime.now() - timedelta(days=i) for i in range(5)][::-1]
    data = pd.DataFrame({
        "Open": [100, 100, 100, 100, 100],
        "High": [105, 105, 105, 105, 105],
        "Low": [95, 95, 95, 95, 95],
        "Close": [100, 100, 100, 100, 100],
        "Volume": [1000, 1000, 1000, 1000, 1000]
    }, index=dates)
    
    aligned_data = {
        "dates": dates,
        "symbols": ["AAPL", "MSFT"],
        "data": {"AAPL": data, "MSFT": data}
    }
    
    for i in range(len(dates)):
        timestamp = dates[i]
        for symbol in ["AAPL", "MSFT"]:
            symbol_data = aligned_data["data"][symbol].iloc[: i + 1]
            current_price = symbol_data["Close"].iloc[-1]
            strategy = strategies[symbol]
            signal = strategy.generate_signal(symbol_data)
            engine._execute_trade(symbol, signal, current_price, timestamp, strategy.name)
        engine._update_equity(timestamp, aligned_data, i)

    print(f"Trades executed: {len(engine.trades)}")
    for t in engine.trades:
        print(f"Symbol: {t['symbol']}, Type: {t['order_type']}, Price: {t['price']}, Comm: {t['commission']}")

    if len(engine.trades) == 4:
        print("✓ Multi-asset trade execution works")
    else:
        print("✗ Multi-asset trade execution failed")

if __name__ == "__main__":
    test_trading_engine()
    test_multi_asset_engine()
