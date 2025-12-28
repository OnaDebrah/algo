"""
Options Backtesting UI Component - Integrated Version
Connects UI with OptionsBacktestEngine
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import fetch_stock_data
from core.database import DatabaseManager
from core.options_engine import backtest_options_strategy
from core.risk_manager import RiskManager
from strategies.options_strategies import OptionsStrategy


def render_options_backtest(db: DatabaseManager, risk_manager: RiskManager):
    """Main options backtesting interface"""

    st.header("📈 Options Strategy Backtesting")

    st.markdown("""
    Backtest options strategies on historical data with customizable entry/exit rules,
    Greeks tracking, and comprehensive performance analytics.
    """)

    # Strategy Selection
    st.markdown("### 🎯 Select Strategy to Backtest")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Income Strategies**")
        if st.button("Covered Call", key="bt_covered_call", use_container_width=True):
            st.session_state["selected_strategy"] = OptionsStrategy.COVERED_CALL
        if st.button("Cash Secured Put", key="bt_csp", use_container_width=True):
            st.session_state["selected_strategy"] = OptionsStrategy.CASH_SECURED_PUT
        if st.button("Iron Condor", key="bt_iron_condor", use_container_width=True):
            st.session_state["selected_strategy"] = OptionsStrategy.IRON_CONDOR

    with col2:
        st.markdown("**Directional Strategies**")
        if st.button("Vertical Call Spread", key="bt_call_spread", use_container_width=True):
            st.session_state["selected_strategy"] = OptionsStrategy.VERTICAL_CALL_SPREAD
        if st.button("Vertical Put Spread", key="bt_put_spread", use_container_width=True):
            st.session_state["selected_strategy"] = OptionsStrategy.VERTICAL_PUT_SPREAD
        if st.button("Butterfly Spread", key="bt_butterfly", use_container_width=True):
            st.session_state["selected_strategy"] = OptionsStrategy.BUTTERFLY_SPREAD

    with col3:
        st.markdown("**Volatility Strategies**")
        if st.button("Long Straddle", key="bt_straddle", use_container_width=True):
            st.session_state["selected_strategy"] = OptionsStrategy.STRADDLE
        if st.button("Long Strangle", key="bt_strangle", use_container_width=True):
            st.session_state["selected_strategy"] = OptionsStrategy.STRANGLE
        if st.button("Protective Put", key="bt_protective_put", use_container_width=True):
            st.session_state["selected_strategy"] = OptionsStrategy.PROTECTIVE_PUT

    # Display selected strategy configuration
    if "selected_strategy" in st.session_state:
        st.markdown("---")
        _render_backtest_configuration(st.session_state["selected_strategy"], db, risk_manager)
    else:
        st.info("👆 Select a strategy above to begin backtesting")


def _render_backtest_configuration(strategy: OptionsStrategy, db: DatabaseManager, risk_manager: RiskManager):
    """Render backtest configuration and execution"""

    st.subheader(f"📊 {strategy.value} Backtest Configuration")

    # Basic Settings
    st.markdown("### 🎯 Basic Settings")

    col1, col2, col3 = st.columns(3)

    with col1:
        symbol = st.text_input("Underlying Symbol", value="SPY", help="Stock ticker to backtest").upper()

    with col2:
        period = st.selectbox("Historical Period", ["6mo", "1y", "2y", "5y"], index=2, help="Amount of historical data to use")

    with col3:
        interval = st.selectbox("Data Interval", ["1d", "1wk"], index=0, help="Data granularity")

    # Entry Rules
    st.markdown("### 📥 Entry Rules")

    col1, col2, col3 = st.columns(3)

    with col1:
        signal_type = st.selectbox(
            "Entry Signal",
            ["regular", "rsi", "moving_average"],
            format_func=lambda x: {"regular": "Regular Intervals", "rsi": "RSI Based", "moving_average": "Moving Average Crossover"}[x],
            help="When to enter new positions",
        )

    with col2:
        if signal_type == "regular":
            entry_frequency = st.number_input("Entry Frequency (days)", min_value=1, max_value=90, value=30, help="Days between new positions")
        else:
            entry_frequency = 30

    with col3:
        dte = st.number_input("Days to Expiration", min_value=7, max_value=365, value=30, help="Option expiration period")

    # Strategy-specific parameters
    st.markdown("### ⚙️ Strategy Parameters")
    strategy_params = _get_strategy_parameters(strategy)

    # Exit Rules
    st.markdown("### 📤 Exit Rules")

    col1, col2, col3 = st.columns(3)

    with col1:
        profit_target = st.slider("Profit Target (%)", min_value=10, max_value=200, value=50, help="Close position at this % profit") / 100

    with col2:
        loss_limit = st.slider("Max Loss (%)", min_value=10, max_value=200, value=50, help="Close position at this % loss") / 100

    with col3:
        dte_exit = st.number_input("DTE Exit", min_value=1, max_value=30, value=7, help="Close position with days remaining")

    # Risk Management
    st.markdown("### 🛡️ Risk Management")

    col1, col2, col3 = st.columns(3)

    with col1:
        initial_capital = st.number_input("Initial Capital ($)", min_value=10000, max_value=1000000, value=100000, step=10000)

    with col2:
        volatility = st.slider(
            "Implied Volatility", min_value=0.1, max_value=1.0, value=0.3, step=0.05, format="%.0f%%", help="Expected volatility for pricing"
        )

    with col3:
        commission = st.number_input("Commission per Contract ($)", min_value=0.0, max_value=10.0, value=0.65, step=0.05)

    # Run Backtest Button
    if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
        with st.spinner(f"Running backtest on {symbol}..."):
            try:
                # Fetch historical data
                data = fetch_stock_data(symbol, period, interval)

                if data.empty:
                    st.error(f"❌ No data available for {symbol}")
                    return

                st.success(f"✅ Fetched {len(data)} data points for {symbol}")

                # Build entry rules
                entry_rules = {
                    "signal": signal_type,
                    "entry_frequency": entry_frequency,
                    "days_to_expiration": dte,
                    "min_capital": 5000,
                    **strategy_params,
                }

                # Build exit rules
                exit_rules = {"profit_target": profit_target, "loss_limit": -loss_limit, "dte_exit": dte_exit}

                # Run backtest
                results = backtest_options_strategy(
                    symbol=symbol,
                    data=data,
                    strategy_type=strategy,
                    initial_capital=initial_capital,
                    risk_free_rate=0.05,
                    commission=commission,
                    entry_rules=entry_rules,
                    exit_rules=exit_rules,
                    volatility=volatility,
                )

                # Display results
                st.success("✅ Backtest completed!")
                _display_backtest_results(results, symbol, strategy)

            except Exception as e:
                st.error(f"❌ Error during backtest: {str(e)}")
                with st.expander("Show detailed error"):
                    import traceback

                    st.code(traceback.format_exc())


def _get_strategy_parameters(strategy: OptionsStrategy) -> dict:
    """Get strategy-specific parameters"""

    params = {}

    if strategy == OptionsStrategy.COVERED_CALL:
        otm_pct = st.slider("Call Strike (% OTM)", min_value=0, max_value=20, value=5, help="How far out-of-the-money")
        params["otm_call_pct"] = 1 + (otm_pct / 100)

    elif strategy == OptionsStrategy.CASH_SECURED_PUT:
        otm_pct = st.slider("Put Strike (% OTM)", min_value=0, max_value=20, value=5, help="How far out-of-the-money")
        params["otm_put_pct"] = 1 - (otm_pct / 100)

    elif strategy == OptionsStrategy.IRON_CONDOR:
        col1, col2 = st.columns(2)
        with col1:
            wing_width = st.slider("Wing Width (%)", min_value=2, max_value=10, value=5, help="Distance between strikes") / 100
        with col2:
            center_width = st.slider("Center Width (%)", min_value=5, max_value=15, value=10, help="Width of profit zone") / 100

        params.update(
            {
                "put_long_strike": 1 - (center_width / 2 + wing_width),
                "put_short_strike": 1 - (center_width / 2),
                "call_short_strike": 1 + (center_width / 2),
                "call_long_strike": 1 + (center_width / 2 + wing_width),
            }
        )

    elif strategy in [OptionsStrategy.VERTICAL_CALL_SPREAD, OptionsStrategy.VERTICAL_PUT_SPREAD]:
        col1, col2 = st.columns(2)
        with col1:
            long_pct = (
                st.slider(
                    "Long Strike (% from current)",
                    min_value=-10,
                    max_value=10,
                    value=0,
                )
                / 100
            )
        with col2:
            width = (
                st.slider(
                    "Spread Width (%)",
                    min_value=2,
                    max_value=15,
                    value=5,
                )
                / 100
            )

        if strategy == OptionsStrategy.VERTICAL_CALL_SPREAD:
            params["long_strike"] = 1 + long_pct
            params["short_strike"] = 1 + long_pct + width
        else:
            params["long_strike"] = 1 + long_pct
            params["short_strike"] = 1 + long_pct - width

    return params


def _display_backtest_results(results: dict, symbol: str, strategy: OptionsStrategy):
    """Display comprehensive backtest results"""

    engine = results.get("engine")

    st.markdown("---")
    st.subheader("📊 Backtest Results")

    # Key Performance Metrics
    st.markdown("### 💰 Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Return", f"{results['total_return']:.2f}%", delta=f"${results['final_equity'] - results.get('initial_capital', 100000):,.0f}"
        )

    with col2:
        st.metric("Win Rate", f"{results['win_rate']:.1f}%", delta=f"{results['winning_trades']}/{results['total_trades']} wins")

    with col3:
        st.metric("Profit Factor", f"{results['profit_factor']:.2f}", delta="Good" if results["profit_factor"] > 1.5 else "Needs Improvement")

    with col4:
        st.metric("Max Drawdown", f"{results['max_drawdown']:.2f}%", delta="Risk Metric", delta_color="inverse")

    # Additional Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Trades", results["total_trades"])

    with col2:
        st.metric("Avg Days Held", f"{results['avg_days_held']:.1f}")

    with col3:
        st.metric("Avg P&L %", f"{results['avg_pnl_pct']:.1f}%")

    with col4:
        st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")

    # Profit/Loss Breakdown
    st.markdown("### 📈 Profit/Loss Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Profit", f"${results['total_profit']:,.2f}")

    with col2:
        st.metric("Total Loss", f"${results['total_loss']:,.2f}")

    with col3:
        st.metric("Average Profit", f"${results['avg_profit']:,.2f}")

    # Equity Curve
    if engine and engine.equity_curve:
        _plot_equity_curve(engine.equity_curve, results.get("initial_capital", 100000))

    # Trade Analysis
    if engine and engine.closed_positions:
        _display_trade_analysis(engine.closed_positions)

    # Trade Log
    if engine and engine.trades:
        _display_trade_log(engine.trades, symbol, strategy)


def _plot_equity_curve(equity_curve: list, initial_capital: float):
    """Plot equity curve"""

    st.markdown("### 📈 Equity Curve")

    df = pd.DataFrame(equity_curve)

    fig = go.Figure()

    # Portfolio value
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["equity"],
            mode="lines",
            name="Portfolio Value",
            line=dict(color="#00ff88", width=2),
            fill="tonexty",
            fillcolor="rgba(0, 255, 136, 0.1)",
        )
    )

    # Initial capital line
    fig.add_hline(y=initial_capital, line_dash="dash", line_color="gray", annotation_text="Initial Capital")

    # Cash component
    fig.add_trace(go.Scatter(x=df["date"], y=df["cash"], mode="lines", name="Cash", line=dict(color="cyan", width=1, dash="dot"), opacity=0.5))

    fig.update_layout(template="plotly_dark", height=400, xaxis_title="Date", yaxis_title="Value ($)", hovermode="x unified", showlegend=True)

    st.plotly_chart(fig, use_container_width=True)


def _display_trade_analysis(closed_positions: list):
    """Display detailed trade analysis"""

    st.markdown("### 📊 Trade Analysis")

    df = pd.DataFrame(closed_positions)

    # Profit distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Profit Distribution**")

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df["pnl"], nbinsx=30, marker_color="#00ff88", opacity=0.7))

        fig.update_layout(template="plotly_dark", height=300, xaxis_title="Profit/Loss ($)", yaxis_title="Frequency", showlegend=False)

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**P&L % Distribution**")

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df["pnl_pct"] * 100, nbinsx=30, marker_color="cyan", opacity=0.7))

        fig.update_layout(template="plotly_dark", height=300, xaxis_title="P&L %", yaxis_title="Frequency", showlegend=False)

        st.plotly_chart(fig, use_container_width=True)

    # Days held analysis
    st.markdown("**Days Held Analysis**")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["days_held"],
            y=df["pnl"],
            mode="markers",
            marker=dict(size=8, color=df["pnl"], colorscale="RdYlGn", showscale=True, colorbar=dict(title="P&L ($)")),
            text=df["strategy"],
            hovertemplate="<b>%{text}</b><br>Days: %{x}<br>P&L: $%{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(template="plotly_dark", height=300, xaxis_title="Days Held", yaxis_title="Profit/Loss ($)", showlegend=False)

    st.plotly_chart(fig, use_container_width=True)


def _display_trade_log(trades: list, symbol: str, strategy: OptionsStrategy):
    """Display trade log"""

    st.markdown("### 📋 Trade Log")

    df = pd.DataFrame(trades)

    # Format for display
    display_df = df.copy()
    display_df["date"] = pd.to_datetime(display_df["date"]).dt.strftime("%Y-%m-%d")

    if "cost" in display_df.columns:
        display_df["cost"] = display_df["cost"].apply(lambda x: f"${x:,.2f}")
    if "pnl" in display_df.columns:
        display_df["pnl"] = display_df["pnl"].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "-")
    if "pnl_pct" in display_df.columns:
        display_df["pnl_pct"] = display_df["pnl_pct"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "-")
    if "price" in display_df.columns:
        display_df["price"] = display_df["price"].apply(lambda x: f"${x:.2f}")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Download button
    csv = pd.DataFrame(trades).to_csv(index=False)
    st.download_button(
        label="📥 Download Trade Log",
        data=csv,
        file_name=f"options_backtest_{symbol}_{strategy.value}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
