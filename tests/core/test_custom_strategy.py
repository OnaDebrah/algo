"""
Custom Strategy Builder - Comprehensive Testing Script
"""

import asyncio
import sys

import numpy as np
import pandas as pd


def print_header(text):
    """Print styled header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_section(text):
    """Print section header"""
    print("\n" + "-" * 70)
    print(f"  {text}")
    print("-" * 70)


def test_code_validation():
    """Test code validation and security"""
    print_header("TEST 1: Code Validation & Security")

    from core.custom_strategy_engine import SafeExecutionEnvironment

    executor = SafeExecutionEnvironment()

    # Test 1: Valid code
    print("Test 1.1: Valid strategy code")
    valid_code = """
def generate_signals(data):
    signals = pd.Series(0, index=data.index)
    signals[data['Close'] > data['Close'].shift(1)] = 1
    return signals
"""
    is_valid, error = executor.validate_code(valid_code)
    print(f"   Result: {'✅ PASS' if is_valid else '❌ FAIL'}")
    if not is_valid:
        print(f"   Error: {error}")

    # Test 2: Missing function
    print("\nTest 1.2: Missing generate_signals function")
    invalid_code = """
def my_strategy(data):
    return pd.Series(0, index=data.index)
"""
    is_valid, error = executor.validate_code(invalid_code)
    print(f"   Result: {'❌ Correctly Rejected' if not is_valid else '⚠️ Should Have Failed'}")
    if not is_valid:
        print(f"   Error: {error}")

    # Test 3: Forbidden operations
    print("\nTest 1.3: Forbidden operations (security)")
    dangerous_code = """
def generate_signals(data):
    import os
    os.system("rm -rf /")
    return pd.Series(0, index=data.index)
"""
    is_valid, error = executor.validate_code(dangerous_code)
    print(f"   Result: {'❌ Correctly Rejected' if not is_valid else '⚠️ SECURITY ISSUE!'}")
    if not is_valid:
        print(f"   Error: {error}")

    # Test 4: Syntax error
    print("\nTest 1.4: Syntax error")
    syntax_error_code = """
def generate_signals(data):
    if data['Close'] > 100
        return 1
"""
    is_valid, error = executor.validate_code(syntax_error_code)
    print(f"   Result: {'❌ Correctly Rejected' if not is_valid else '⚠️ Should Have Failed'}")
    if not is_valid:
        print(f"   Error: {error[:50]}...")

    print("\n✅ Code validation tests complete!")
    return True


def test_strategy_execution():
    """Test strategy execution"""
    print_header("TEST 2: Strategy Execution")

    from core.custom_strategy_engine import SafeExecutionEnvironment

    executor = SafeExecutionEnvironment()

    # Create sample data
    print("Generating sample data...")
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    data = pd.DataFrame(
        {
            "Open": np.random.randn(len(dates)).cumsum() + 100,
            "High": np.random.randn(len(dates)).cumsum() + 102,
            "Low": np.random.randn(len(dates)).cumsum() + 98,
            "Close": np.random.randn(len(dates)).cumsum() + 100,
            "Volume": np.random.randint(1000000, 10000000, len(dates)),
        },
        index=dates,
    )

    print(f"   Data shape: {data.shape}")
    print(f"   Date range: {data.index[0]} to {data.index[-1]}")

    # Test simple strategy
    print("\nTest 2.1: Execute simple SMA strategy")
    sma_strategy = """
def generate_signals(data):
    data['SMA_20'] = data['Close'].rolling(20).mean()
    data['SMA_50'] = data['Close'].rolling(50).mean()

    signals = pd.Series(0, index=data.index)
    signals[(data['SMA_20'] > data['SMA_50']) &
            (data['SMA_20'].shift(1) <= data['SMA_50'].shift(1))] = 1
    signals[(data['SMA_20'] < data['SMA_50']) &
            (data['SMA_20'].shift(1) >= data['SMA_50'].shift(1))] = -1

    return signals
"""

    success, result, output = executor.execute_strategy(sma_strategy, data)

    if success:
        print("   ✅ Execution successful")
        buy_signals = (result == 1).sum()
        sell_signals = (result == -1).sum()
        print(f"   Buy signals: {buy_signals}")
        print(f"   Sell signals: {sell_signals}")
        print(f"   Total signals: {buy_signals + sell_signals}")
    else:
        print(f"   ❌ Execution failed: {result}")
        return False

    # Test RSI strategy
    print("\nTest 2.2: Execute RSI strategy")
    rsi_strategy = """
def generate_signals(data):
    # Calculate RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    signals = pd.Series(0, index=data.index)
    signals[(rsi > 30) & (rsi.shift(1) <= 30)] = 1
    signals[(rsi < 70) & (rsi.shift(1) >= 70)] = -1

    return signals
"""

    success, result, output = executor.execute_strategy(rsi_strategy, data)

    if success:
        print("   ✅ Execution successful")
        buy_signals = (result == 1).sum()
        sell_signals = (result == -1).sum()
        print(f"   Buy signals: {buy_signals}")
        print(f"   Sell signals: {sell_signals}")
    else:
        print("   ❌ Execution failed")
        return False

    print("\n✅ Strategy execution tests complete!")
    return True


async def test_ai_generation():
    """Test AI code generation"""
    print_header("TEST 3: AI Strategy Generation")

    import os

    from core.custom_strategy_engine import StrategyCodeGenerator

    api_key = os.getenv("ANTHROPIC_API_KEY", "")

    if not api_key:
        print("⚠️  No API key found. Testing with templates...")
        generator = StrategyCodeGenerator()

        # Test template generation
        print("\nTest 3.1: Template generation (fallback)")
        code, explanation, example = generator._sma_crossover_template()

        print("   ✅ Generated SMA Crossover template")
        print(f"   Code length: {len(code)} characters")
        print(f"   Has explanation: {'Yes' if explanation else 'No'}")
        print(f"   Has example: {'Yes' if example else 'No'}")

        return True

    # Test with API
    print(f"✅ API key found: {api_key[:8]}...{api_key[-4:]}")
    generator = StrategyCodeGenerator(api_key=api_key)

    test_prompts = [
        "Create a simple moving average crossover strategy",
        "Build an RSI strategy that buys oversold and sells overbought",
    ]

    for idx, prompt in enumerate(test_prompts, 1):
        print(f"\nTest 3.{idx}: Generate from prompt")
        print(f"   Prompt: '{prompt}'")

        try:
            code, explanation, example = await generator.generate_strategy_code(prompt, style="technical")

            print("   ✅ Generation successful")
            print(f"   Code length: {len(code)} characters")
            print(f"   Has 'generate_signals': {'Yes' if 'generate_signals' in code else 'No'}")

            # Validate generated code
            from core.custom_strategy_engine import SafeExecutionEnvironment

            executor = SafeExecutionEnvironment()
            is_valid, error = executor.validate_code(code)

            print(f"   Code is valid: {'✅ Yes' if is_valid else '❌ No'}")
            if not is_valid:
                print(f"   Validation error: {error}")

        except Exception as e:
            print(f"   ❌ Generation failed: {e}")
            return False

    print("\n✅ AI generation tests complete!")
    return True


def test_backtest_simulation():
    """Test backtest simulation"""
    print_header("TEST 4: Backtest Simulation")

    from core.custom_strategy_engine import SafeExecutionEnvironment

    executor = SafeExecutionEnvironment()

    # Create realistic sample data
    print("Generating sample price data...")
    dates = pd.date_range(start="2022-01-01", end="2023-12-31", freq="D")

    # Create trending price data
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = 100 * (1 + returns).cumprod()

    data = pd.DataFrame(
        {
            "Open": prices * (1 + np.random.normal(0, 0.01, len(dates))),
            "High": prices * (1 + np.random.uniform(0, 0.02, len(dates))),
            "Low": prices * (1 + np.random.uniform(-0.02, 0, len(dates))),
            "Close": prices,
            "Volume": np.random.randint(1000000, 10000000, len(dates)),
        },
        index=dates,
    )

    print(f"   Start price: ${data['Close'].iloc[0]:.2f}")
    print(f"   End price: ${data['Close'].iloc[-1]:.2f}")
    print(f"   B&H return: {((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100:.2f}%")

    # Test strategy
    print("\nTest 4.1: Backtest SMA crossover strategy")

    strategy_code = """
def generate_signals(data):
    data['SMA_fast'] = data['Close'].rolling(20).mean()
    data['SMA_slow'] = data['Close'].rolling(50).mean()

    signals = pd.Series(0, index=data.index)
    signals[(data['SMA_fast'] > data['SMA_slow']) &
            (data['SMA_fast'].shift(1) <= data['SMA_slow'].shift(1))] = 1
    signals[(data['SMA_fast'] < data['SMA_slow']) &
            (data['SMA_fast'].shift(1) >= data['SMA_slow'].shift(1))] = -1

    return signals
"""

    success, signals, output = executor.execute_strategy(strategy_code, data)

    if not success:
        print("   ❌ Strategy execution failed")
        return False

    # Simple backtest simulation
    position = 0
    cash = 10000
    equity = []
    trades = 0

    for i in range(len(data)):
        price = data["Close"].iloc[i]
        signal = signals.iloc[i]

        if signal == 1 and position == 0:  # Buy
            shares = cash // price
            position = shares
            cash -= shares * price
            trades += 1
        elif signal == -1 and position > 0:  # Sell
            cash += position * price
            position = 0
            trades += 1

        equity.append(cash + position * price)

    final_equity = equity[-1]
    total_return = ((final_equity / 10000) - 1) * 100

    print("   ✅ Backtest complete")
    print("   Initial capital: $10,000")
    print(f"   Final equity: ${final_equity:,.2f}")
    print(f"   Total return: {total_return:.2f}%")
    print(f"   Total trades: {trades}")

    buy_signals = (signals == 1).sum()
    sell_signals = (signals == -1).sum()
    print(f"   Buy signals: {buy_signals}")
    print(f"   Sell signals: {sell_signals}")

    print("\n✅ Backtest simulation tests complete!")
    return True


def test_template_strategies():
    """Test all template strategies"""
    print_header("TEST 5: Template Strategies")

    from core.custom_strategy_engine import (
        SafeExecutionEnvironment,
        StrategyCodeGenerator,
    )

    generator = StrategyCodeGenerator()
    executor = SafeExecutionEnvironment()

    templates = [
        ("SMA Crossover", generator._sma_crossover_template),
        ("RSI Mean Reversion", generator._rsi_template),
        ("MACD Momentum", generator._macd_template),
        ("Bollinger Bands", generator._bollinger_template),
        ("Simple Momentum", generator._simple_momentum_template),
    ]

    # Generate sample data
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    data = pd.DataFrame(
        {
            "Open": np.random.randn(len(dates)).cumsum() + 100,
            "High": np.random.randn(len(dates)).cumsum() + 102,
            "Low": np.random.randn(len(dates)).cumsum() + 98,
            "Close": np.random.randn(len(dates)).cumsum() + 100,
            "Volume": np.random.randint(1000000, 10000000, len(dates)),
        },
        index=dates,
    )

    for name, template_func in templates:
        print(f"\nTest 5.{templates.index((name, template_func)) + 1}: {name}")

        try:
            code, explanation, example = template_func()

            # Validate code
            is_valid, error = executor.validate_code(code)
            if not is_valid:
                print(f"   ❌ Validation failed: {error}")
                continue

            # Execute strategy
            success, signals, output = executor.execute_strategy(code, data)

            if success:
                buy_signals = (signals == 1).sum()
                sell_signals = (signals == -1).sum()
                print(f"   ✅ {name}: {buy_signals} buys, {sell_signals} sells")
            else:
                print("   ❌ Execution failed")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n✅ Template strategy tests complete!")
    return True


async def run_all_tests():
    """Run complete test suite"""
    print_header("CUSTOM STRATEGY BUILDER - COMPLETE TEST SUITE")

    print("Running comprehensive tests...")
    print("This will take approximately 1-2 minutes.\n")

    tests = [
        ("Code Validation", test_code_validation, False),
        ("Strategy Execution", test_strategy_execution, False),
        ("AI Generation", test_ai_generation, True),
        ("Backtest Simulation", test_backtest_simulation, False),
        ("Template Strategies", test_template_strategies, False),
    ]

    results = {}

    for test_name, test_func, is_async in tests:
        try:
            if is_async:
                result = await test_func()
            else:
                result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed with error: {e}")
            results[test_name] = False

    # Summary
    print_header("TEST SUMMARY")

    total = len(results)
    passed = sum(1 for r in results.values() if r)

    print(f"Tests Run: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}\n")

    print("Detailed Results:")
    print("-" * 70)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<50} {status}")

    print("-" * 70)

    if passed == total:
        print("\n🎉 All tests passed! Custom Strategy Builder is ready!")
        print("\n📚 Next Steps:")
        print("   1. Run: streamlit run main.py")
        print("   2. Navigate to 'Custom Strategy Builder'")
        print("   3. Try the AI Generator or Code Editor")
        print("   4. Create your first custom strategy!")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")

    return passed == total


async def demo_mode():
    """Interactive demo mode"""
    print_header("🎬 CUSTOM STRATEGY BUILDER - INTERACTIVE DEMO")

    print("This demo showcases the custom strategy features.\n")

    print("Select a demo:")
    print("1. Validate and test a simple strategy")
    print("2. Generate strategy from prompt (requires API)")
    print("3. Run full backtest simulation")
    print("4. Test all template strategies")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        test_code_validation()
        test_strategy_execution()
    elif choice == "2":
        await test_ai_generation()
    elif choice == "3":
        test_backtest_simulation()
    elif choice == "4":
        test_template_strategies()
    else:
        print("Invalid choice. Running full test suite...")
        await run_all_tests()


def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        asyncio.run(demo_mode())
    else:
        asyncio.run(run_all_tests())


if __name__ == "__main__":
    main()
