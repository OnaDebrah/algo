"""
Custom Strategy Builder UI - Write or generate custom trading strategies
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from streamlit.core.custom_strategy_engine import SafeExecutionEnvironment, StrategyCodeGenerator


class CustomStrategyBuilderUI:
    """UI for custom strategy creation and testing"""

    def __init__(self):
        self.executor = SafeExecutionEnvironment()
        self.generator = None
        self._initialize_generator()

        # Initialize session state
        if "custom_strategies" not in st.session_state:
            st.session_state["custom_strategies"] = {}
        if "current_strategy_code" not in st.session_state:
            st.session_state["current_strategy_code"] = ""
        if "strategy_results" not in st.session_state:
            st.session_state["strategy_results"] = None

    def _initialize_generator(self):
        """Initialize the code generator"""
        import os

        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.generator = StrategyCodeGenerator(api_key=api_key)

    def render(self):
        """Render the custom strategy builder interface"""
        st.title("⚡ Custom Strategy Builder")
        st.markdown("Create and backtest your own trading strategies with code or natural language")

        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "🤖 AI Strategy Generator",
                "💻 Code Editor",
                "📊 Backtest Results",
                "📚 Strategy Library",
            ]
        )

        with tab1:
            self._render_ai_generator()

        with tab2:
            self._render_code_editor()

        with tab3:
            self._render_backtest_results()

        with tab4:
            self._render_strategy_library()

    def _render_ai_generator(self):
        """Render AI strategy generator tab"""
        st.markdown("### 🤖 Describe Your Strategy in Plain English")
        st.markdown("The AI will generate complete, executable Python code for your strategy")

        # Example prompts
        with st.expander("💡 Example Prompts", expanded=False):
            st.markdown(
                """
            **Trend Following:**
            - "Create a strategy that buys when price crosses above 50-day moving average"
            - "Make a MACD crossover strategy with signal line"

            **Mean Reversion:**
            - "Buy when RSI drops below 30, sell when it rises above 70"
            - "Trade Bollinger Band bounces - buy at lower band, sell at upper band"

            **Momentum:**
            - "Buy stocks with strong momentum over last 20 days"
            - "Create a breakout strategy for 52-week highs"

            **Custom:**
            - "Combine RSI and MACD - only take trades when both agree"
            - "Buy on golden cross (50MA crosses 200MA) with volume confirmation"
            """
            )

        # Input form
        with st.form("ai_generator_form"):
            col1, col2 = st.columns([3, 1])

            with col1:
                user_prompt = st.text_area(
                    "Describe your strategy:",
                    height=100,
                    placeholder="Example: Create a strategy that buys when RSI is below 30 and price is above 200-day "
                    "moving average, and sells when RSI rises above 70...",
                )

            with col2:
                st.markdown("####")  # Spacing
                strategy_style = st.selectbox(
                    "Style:",
                    ["technical", "fundamental", "hybrid"],
                    help="Type of analysis to focus on",
                )

                strategy_name = st.text_input("Name:", placeholder="My Custom Strategy")

            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                generate_btn = st.form_submit_button("🚀 Generate Code", type="primary", use_container_width=True)
            with col2:
                if st.form_submit_button("Clear", use_container_width=True):
                    st.session_state["current_strategy_code"] = ""
                    st.rerun()

        # Generate strategy
        if generate_btn:
            if not user_prompt:
                st.error("⚠️ Please describe your strategy")
                return

            with st.spinner("🤖 AI is generating your strategy code..."):
                code, explanation, example = asyncio.run(self.generator.generate_strategy_code(user_prompt, strategy_style))

                # Store generated code
                st.session_state["current_strategy_code"] = code

                # Display results
                st.success("✅ Strategy code generated successfully!")

                # Show explanation
                st.markdown("### 📖 Strategy Explanation")
                st.markdown(explanation)

                # Show code
                st.markdown("### 💻 Generated Code")
                st.code(code, language="python")

                # Show example usage
                with st.expander("📚 Example Usage"):
                    st.code(example, language="python")

                # Save option
                if strategy_name:
                    if st.button("💾 Save to Library", key="save_generated"):
                        self._save_strategy(strategy_name, code, explanation)
                        st.success(f"✅ Saved '{strategy_name}' to library!")

                # Quick test button
                st.markdown("### 🧪 Quick Test")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("▶️ Test with Sample Data", use_container_width=True):
                        st.session_state["test_custom_strategy"] = True
                        st.rerun()
                with col2:
                    if st.button("📊 Full Backtest", use_container_width=True):
                        st.session_state["page"] = "custom_strategy_backtest"
                        st.rerun()

    def _render_code_editor(self):
        """Render code editor tab"""
        st.markdown("### 💻 Write Your Own Strategy Code")

        # Instructions
        with st.expander("📚 How to Write a Strategy", expanded=False):
            st.markdown(
                """
            ### Strategy Structure

            Your strategy must define a `generate_signals(data)` function:

            ```python
            def generate_signals(data):
                # data is a pandas DataFrame with: Open, High, Low, Close, Volume

                # Your strategy logic here
                # Calculate indicators, conditions, etc.

                # Return a pandas Series with signals:
                # 1 = Buy signal
                # -1 = Sell signal
                # 0 = Hold (no signal)

                return signals
            ```

            ### Available Libraries
            - `pandas` (as pd)
            - `numpy` (as np)
            - `math`
            - `datetime`

            ### Tips
            - Use vectorized operations (avoid loops)
            - Handle NaN values properly
            - Test with `.head()` before full backtest
            - Add comments to explain logic
            """
            )

        # Load template
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("#### 📝 Code Editor")
        with col2:
            template = st.selectbox(
                "Load Template:",
                [
                    "",
                    "SMA Crossover",
                    "RSI Mean Reversion",
                    "MACD Momentum",
                    "Bollinger Bands",
                    "Custom Momentum",
                ],
                label_visibility="collapsed",
            )

            if template:
                if st.button("Load", use_container_width=True):
                    code = self._get_template_code(template)
                    st.session_state["current_strategy_code"] = code
                    st.rerun()

        # Code editor
        code = st.text_area(
            "Strategy Code:",
            value=st.session_state.get("current_strategy_code", ""),
            height=400,
            placeholder=self._get_placeholder_code(),
            label_visibility="collapsed",
        )

        st.session_state["current_strategy_code"] = code

        # Validation and testing
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("✅ Validate Code", use_container_width=True):
                self._validate_strategy_code(code)

        with col2:
            if st.button("🧪 Test on Sample", use_container_width=True):
                self._test_strategy_sample(code)

        with col3:
            if st.button("📊 Full Backtest", type="primary", use_container_width=True):
                if code.strip():
                    st.session_state["backtest_custom_code"] = code
                    st.rerun()
                else:
                    st.error("⚠️ Please write some code first")

        with col4:
            strategy_name = st.text_input("Name:", placeholder="Strategy name", label_visibility="collapsed")
            if st.button("💾 Save", use_container_width=True):
                if strategy_name and code:
                    self._save_strategy(strategy_name, code, "Custom strategy")
                    st.success(f"✅ Saved '{strategy_name}'!")
                else:
                    st.warning("⚠️ Enter name and code")

        # Show backtest results if triggered
        if st.session_state.get("backtest_custom_code"):
            self._run_full_backtest(st.session_state["backtest_custom_code"])
            st.session_state["backtest_custom_code"] = None

    def _render_backtest_results(self):
        """Render backtest results tab"""
        if not st.session_state.get("strategy_results"):
            st.info("📊 No backtest results yet. Create and test a strategy first!")
            return

        results = st.session_state["strategy_results"]

        st.markdown("### 📊 Backtest Results")

        # Performance metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Total Return",
                f"{results['total_return']:.2f}%",
                delta=f"{results['total_return'] - results['benchmark_return']:.2f}% vs benchmark",
            )

        with col2:
            st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")

        with col3:
            st.metric("Max Drawdown", f"{results['max_drawdown']:.2f}%")

        with col4:
            st.metric("Win Rate", f"{results['win_rate']:.1f}%")

        with col5:
            st.metric("Total Trades", results["total_trades"])

        # Equity curve
        st.markdown("### 📈 Equity Curve")
        self._plot_equity_curve(results)

        # Detailed metrics
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Performance Metrics")
            metrics_df = pd.DataFrame(
                {
                    "Metric": [
                        "Total Return",
                        "Annualized Return",
                        "Volatility",
                        "Sharpe Ratio",
                        "Sortino Ratio",
                        "Max Drawdown",
                        "Calmar Ratio",
                    ],
                    "Value": [
                        f"{results['total_return']:.2f}%",
                        f"{results.get('annual_return', 0):.2f}%",
                        f"{results.get('volatility', 0):.2f}%",
                        f"{results['sharpe_ratio']:.2f}",
                        f"{results.get('sortino_ratio', 0):.2f}",
                        f"{results['max_drawdown']:.2f}%",
                        f"{results.get('calmar_ratio', 0):.2f}",
                    ],
                }
            )
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("### 📈 Trade Statistics")
            trade_stats_df = pd.DataFrame(
                {
                    "Metric": [
                        "Total Trades",
                        "Winning Trades",
                        "Losing Trades",
                        "Win Rate",
                        "Average Win",
                        "Average Loss",
                        "Profit Factor",
                    ],
                    "Value": [
                        results["total_trades"],
                        results.get("winning_trades", 0),
                        results.get("losing_trades", 0),
                        f"{results['win_rate']:.1f}%",
                        f"{results.get('avg_win', 0):.2f}%",
                        f"{results.get('avg_loss', 0):.2f}%",
                        f"{results.get('profit_factor', 0):.2f}",
                    ],
                }
            )
            st.dataframe(trade_stats_df, use_container_width=True, hide_index=True)

        # Trade log
        if "trades" in results and not results["trades"].empty:
            st.markdown("### 📋 Trade Log")
            st.dataframe(results["trades"].tail(20), use_container_width=True, height=300)

        # Export options
        st.markdown("### 📥 Export Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("💾 Save Strategy", use_container_width=True):
                st.info("Enter name in Code Editor and click Save")

        with col2:
            # Export results as CSV
            csv = results.get("equity_curve", pd.DataFrame()).to_csv()
            st.download_button(
                "📊 Download CSV",
                csv,
                f"backtest_results_{datetime.now().strftime('%Y%m%d')}.csv",
                use_container_width=True,
            )

        with col3:
            if st.button("🔄 Run Again", use_container_width=True):
                st.session_state["strategy_results"] = None
                st.rerun()

    def _render_strategy_library(self):
        """Render saved strategies library"""
        st.markdown("### 📚 Your Strategy Library")

        strategies = st.session_state.get("custom_strategies", {})

        if not strategies:
            st.info("💡 No saved strategies yet. Create and save strategies from other tabs!")
            return

        # Display strategies as cards
        for name, strategy_data in strategies.items():
            with st.expander(f"📈 {name}", expanded=False):
                st.markdown(f"**Created:** {strategy_data.get('created', 'Unknown')}")
                st.markdown(f"**Description:** {strategy_data.get('description', 'No description')}")

                # Show code preview
                st.code(strategy_data["code"][:300] + "...", language="python")

                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("📝 Load", key=f"load_{name}"):
                        st.session_state["current_strategy_code"] = strategy_data["code"]
                        st.success(f"✅ Loaded '{name}' into editor")

                with col2:
                    if st.button("▶️ Test", key=f"test_{name}"):
                        self._test_strategy_sample(strategy_data["code"])

                with col3:
                    if st.button("🗑️ Delete", key=f"delete_{name}"):
                        del st.session_state["custom_strategies"][name]
                        st.success(f"✅ Deleted '{name}'")
                        st.rerun()

    def _validate_strategy_code(self, code: str):
        """Validate strategy code"""
        if not code.strip():
            st.warning("⚠️ No code to validate")
            return

        is_valid, error = self.executor.validate_code(code)

        if is_valid:
            st.success("✅ Code is valid and safe to execute!")
        else:
            st.error(f"❌ Validation Error:\n{error}")

    def _test_strategy_sample(self, code: str):
        """Test strategy on sample data"""
        if not code.strip():
            st.warning("⚠️ No code to test")
            return

        # Generate sample data
        st.info("🧪 Testing on sample data (AAPL, last 100 days)...")

        try:
            # Fetch sample data
            ticker = yf.Ticker("AAPL")
            data = ticker.history(period="100d")

            if data.empty:
                st.error("❌ Could not fetch sample data")
                return

            # Execute strategy
            success, result, output = self.executor.execute_strategy(code, data)

            if success:
                st.success("✅ Strategy executed successfully!")

                # Show output if any
                if output:
                    with st.expander("📄 Execution Output"):
                        st.text(output)

                # Show signal counts
                if isinstance(result, pd.Series):
                    buy_signals = (result == 1).sum()
                    sell_signals = (result == -1).sum()

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Buy Signals", buy_signals)
                    with col2:
                        st.metric("Sell Signals", sell_signals)
                    with col3:
                        st.metric("Total Signals", buy_signals + sell_signals)

                    # Show last few signals
                    if buy_signals + sell_signals > 0:
                        st.markdown("**Recent Signals:**")
                        signal_dates = data.index[result != 0][-5:]
                        signal_types = result[result != 0][-5:]

                        for date, signal in zip(signal_dates, signal_types):
                            signal_text = "🟢 BUY" if signal == 1 else "🔴 SELL"
                            st.text(f"{date.strftime('%Y-%m-%d')}: {signal_text}")
                else:
                    st.warning("⚠️ Strategy didn't return expected signal format")
            else:
                st.error(f"❌ Execution Error:\n{result}")

        except Exception as e:
            st.error(f"❌ Test Error: {str(e)}")

    def _run_full_backtest(self, code: str):
        """Run full backtest on strategy"""
        st.markdown("### 🔬 Running Full Backtest")

        # Backtest configuration
        col1, col2, col3 = st.columns(3)

        with col1:
            ticker = st.text_input("Ticker:", value="AAPL")
        with col2:
            start_date = st.date_input("Start Date:", value=datetime.now() - timedelta(days=365))
        with col3:
            initial_capital = st.number_input("Initial Capital:", value=10000, step=1000)

        if st.button("🚀 Run Backtest", type="primary"):
            with st.spinner("📊 Running backtest..."):
                try:
                    # Fetch data
                    data = yf.download(ticker, start=start_date, progress=False)

                    if data.empty:
                        st.error(f"❌ No data available for {ticker}")
                        return

                    # Execute strategy
                    success, signals, output = self.executor.execute_strategy(code, data)

                    if not success:
                        st.error(f"❌ Strategy execution failed:\n{signals}")
                        return

                    # Run backtest simulation
                    results = self._simulate_backtest(data, signals, initial_capital)

                    # Store results
                    st.session_state["strategy_results"] = results

                    st.success("✅ Backtest complete! Check the 'Backtest Results' tab")

                except Exception as e:
                    st.error(f"❌ Backtest error: {str(e)}")

    def _simulate_backtest(self, data: pd.DataFrame, signals: pd.Series, initial_capital: float) -> Dict:
        """Simulate backtest and calculate metrics"""

        # Initialize
        position = 0
        cash = initial_capital
        equity = []
        trades = []

        # Simulate trading
        for i in range(len(data)):
            date = data.index[i]
            price = data["Close"].iloc[i]
            signal = signals.iloc[i] if i < len(signals) else 0

            # Execute signal
            if signal == 1 and position == 0:  # Buy
                shares = cash // price
                if shares > 0:
                    position = shares
                    cash -= shares * price
                    trades.append({"date": date, "type": "BUY", "price": price, "shares": shares})

            elif signal == -1 and position > 0:  # Sell
                cash += position * price
                trades.append({"date": date, "type": "SELL", "price": price, "shares": position})
                position = 0

            # Calculate equity
            current_equity = cash + (position * price)
            equity.append(current_equity)

        # Calculate metrics
        equity_curve = pd.Series(equity, index=data.index)
        returns = equity_curve.pct_change().dropna()

        # Performance metrics
        total_return = ((equity_curve.iloc[-1] / initial_capital) - 1) * 100
        volatility = returns.std() * np.sqrt(252) * 100
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        # Drawdown
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax * 100
        max_drawdown = drawdown.min()

        # Trade statistics
        trades_df = pd.DataFrame(trades)
        total_trades = len(trades)

        # Win/loss analysis
        if total_trades > 0 and "SELL" in trades_df["type"].values:
            buy_trades = trades_df[trades_df["type"] == "BUY"]
            sell_trades = trades_df[trades_df["type"] == "SELL"]

            if len(buy_trades) > 0 and len(sell_trades) > 0:
                # Match buys with sells
                wins = 0
                losses = 0
                for i in range(min(len(buy_trades), len(sell_trades))):
                    buy_price = buy_trades.iloc[i]["price"]
                    sell_price = sell_trades.iloc[i]["price"]
                    if sell_price > buy_price:
                        wins += 1
                    else:
                        losses += 1

                win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
            else:
                win_rate = 0
        else:
            win_rate = 0

        # Benchmark (buy and hold)
        bh_return = ((data["Close"].iloc[-1] / data["Close"].iloc[0]) - 1) * 100

        return {
            "total_return": total_return,
            "benchmark_return": bh_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "equity_curve": equity_curve,
            "trades": trades_df,
        }

    def _plot_equity_curve(self, results: Dict):
        """Plot equity curve"""
        equity = results["equity_curve"]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity.values,
                mode="lines",
                name="Strategy",
                line=dict(color="blue", width=2),
            )
        )

        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode="x unified",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

    def _save_strategy(self, name: str, code: str, description: str):
        """Save strategy to library"""
        if "custom_strategies" not in st.session_state:
            st.session_state["custom_strategies"] = {}

        st.session_state["custom_strategies"][name] = {
            "code": code,
            "description": description,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

    def _get_template_code(self, template_name: str) -> str:
        """Get template code by name"""
        templates = {
            "SMA Crossover": self.generator._sma_crossover_template(),
            "RSI Mean Reversion": self.generator._rsi_template(),
            "MACD Momentum": self.generator._macd_template(),
            "Bollinger Bands": self.generator._bollinger_template(),
            "Custom Momentum": self.generator._simple_momentum_template(),
        }

        if template_name in templates:
            code, _, _ = templates[template_name]
            return code
        return ""

    def _get_placeholder_code(self) -> str:
        """Get placeholder code for editor"""
        return """def generate_signals(data):
    \"\"\"
    Your custom trading strategy.

    Args:
        data: pandas DataFrame with columns: Open, High, Low, Close, Volume

    Returns:
        pandas Series with signals: 1 (buy), -1 (sell), 0 (hold)
    \"\"\"
    # Example: Simple moving average crossover
    data['SMA_fast'] = data['Close'].rolling(window=20).mean()
    data['SMA_slow'] = data['Close'].rolling(window=50).mean()

    signals = pd.Series(0, index=data.index)

    # Buy when fast MA crosses above slow MA
    signals[(data['SMA_fast'] > data['SMA_slow']) &
            (data['SMA_fast'].shift(1) <= data['SMA_slow'].shift(1))] = 1

    # Sell when fast MA crosses below slow MA
    signals[(data['SMA_fast'] < data['SMA_slow']) &
            (data['SMA_fast'].shift(1) >= data['SMA_slow'].shift(1))] = -1

    return signals
"""


def render_custom_strategy_builder():
    """Main function to render custom strategy builder"""
    ui = CustomStrategyBuilderUI()
    ui.render()
