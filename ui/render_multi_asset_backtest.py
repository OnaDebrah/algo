import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from alerts.alert_manager import AlertManager
from config import DEFAULT_INITIAL_CAPITAL
from core import fetch_stock_data
from core.database import DatabaseManager
from core.multi_asset_engine import MultiAssetEngine
from core.risk_manager import RiskManager
from strategies.strategy_catalog import get_catalog
from ui.asset_selector import render_portfolio_builder


def render_multi_asset_backtest(
    db: DatabaseManager,
    risk_manager: RiskManager,
    ml_models: dict,
    alert_manager: AlertManager,
):
    """Enhanced multi-asset backtest with portfolio builder"""

    st.markdown(
        """
    ### 🎯 Multi-Asset Portfolio Backtesting
    Build and test portfolios across multiple asset classes:
    - Stocks, ETFs, Crypto, Forex, Commodities, Bonds, Indices
    - Portfolio diversification analysis
    - Asset allocation optimization
    - Cross-asset correlation analysis
    """
    )

    # Portfolio Builder with Quick Input Option
    st.markdown("### 📝 Portfolio Input")

    input_mode = st.radio(
        "Input Mode",
        ["Portfolio Builder", "Quick Input (Comma-Separated)"],
        horizontal=True,
        key="portfolio_input_mode",
        help="Use Portfolio Builder for guided selection or Quick Input for fast entry",
    )

    portfolio = []

    if input_mode == "Quick Input (Comma-Separated)":
        col1, col2 = st.columns([4, 1])

        with col1:
            symbols_input = st.text_input(
                "Enter Symbols (comma-separated)",
                placeholder="e.g., AAPL, MSFT, GOOGL, SPY, BTC-USD",
                help="Enter multiple symbols separated by commas",
                key="quick_symbols_input",
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            validate_btn = st.button("✓ Validate", key="validate_symbols", type="secondary")

        if symbols_input:
            # Parse and clean symbols
            raw_symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

            if validate_btn or "validated_symbols" in st.session_state:
                # Validate symbols
                try:
                    from core.asset_classes import get_asset_manager

                    asset_mgr = get_asset_manager()

                    validated = []
                    invalid = []

                    for symbol in raw_symbols:
                        is_valid, message = asset_mgr.validate_symbol(symbol)
                        if is_valid:
                            validated.append(symbol)
                        else:
                            invalid.append((symbol, message))

                    if validated:
                        st.session_state.validated_symbols = validated
                        portfolio = validated

                        st.success(f"✅ Validated {len(validated)} symbols: {', '.join(validated)}")

                        # Show asset info
                        with st.expander("📊 Symbol Details"):
                            asset_data = []
                            for symbol in validated:
                                asset_info = asset_mgr.get_asset_info(symbol)
                                asset_data.append(
                                    {
                                        "Symbol": symbol,
                                        "Name": asset_info.name[:30],
                                        "Type": asset_info.asset_class.value,
                                        "Currency": asset_info.currency,
                                    }
                                )

                            df = pd.DataFrame(asset_data)
                            st.dataframe(df, use_container_width=True, hide_index=True)

                    if invalid:
                        st.warning(f"⚠️ {len(invalid)} invalid symbols:")
                        for sym, msg in invalid:
                            st.write(f"- {sym}: {msg}")

                except ImportError:
                    # Fallback if asset manager not available
                    portfolio = raw_symbols
                    st.session_state.validated_symbols = raw_symbols
                    st.info(f"Added {len(raw_symbols)} symbols: {', '.join(raw_symbols)}")

    else:
        # Original Portfolio Builder
        portfolio = render_portfolio_builder(key_prefix="backtest_portfolio")

    if not portfolio or len(portfolio) < 2:
        st.info("👆 Add at least 2 assets to your portfolio to begin backtesting")

        # Show example portfolios
        with st.expander("📚 Example Portfolio Ideas"):
            st.markdown(
                """
            **Conservative Portfolio:**
            - 50% SPY (S&P 500)
            - 30% TLT (Long-term Treasury)
            - 20% GLD (Gold)

            **Aggressive Growth:**
            - 40% QQQ (Tech)
            - 30% NVDA (Growth Stock)
            - 20% BTC-USD (Crypto)
            - 10% TLT (Bonds)

            **Global Diversified:**
            - 30% SPY (US Stocks)
            - 20% IWM (Small Cap)
            - 20% GLD (Gold)
            - 15% EURUSD=X (Forex)
            - 15% BTC-USD (Crypto)
            """
            )
        return

    symbols = portfolio

    # Time period configuration
    col1, col2 = st.columns(2)

    with col1:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=2, key="multi_period")

    with col2:
        interval = st.selectbox("Interval", ["1h", "1d", "1wk"], index=1, key="multi_interval")

    # Strategy Configuration
    st.markdown("### 📊 Strategy Configuration")

    catalog = get_catalog()

    strategy_mode = st.radio(
        "Strategy Mode",
        ["Same strategy for all symbols", "Different strategy per symbol"],
        horizontal=True,
        key="multi_strategy_mode",
    )

    strategies_config = {}

    if strategy_mode == "Same strategy for all symbols":
        # Single strategy TYPE for all (but create separate instances)
        col1, col2 = st.columns(2)

        with col1:
            categories = catalog.get_categories()
            category_names = [cat.value for cat in categories]
            selected_category = st.selectbox("Category", category_names, key="multi_cat")

        with col2:
            category_enum = next(cat for cat in categories if cat.value == selected_category)
            strategies_in_category = catalog.get_by_category(category_enum)
            strategy_names = [info.name for info in strategies_in_category.values()]
            strategy_type = st.selectbox("Strategy", strategy_names, key="multi_strat")

        # Get strategy key
        strategy_key = next(
            (key for key, info in strategies_in_category.items() if info.name == strategy_type),
            None,
        )

        # Configure parameters (just once for display)
        if strategy_key:
            params = _get_strategy_parameters(catalog, strategy_key, "multi")

            # IMPORTANT: Create SEPARATE strategy instances for each symbol
            # Each symbol needs its own instance to avoid state confusion
            for symbol in symbols:
                strategies_config[symbol] = catalog.create_strategy(strategy_key, **params)

    else:
        # Different strategy per symbol
        st.markdown("Configure strategy for each symbol:")

        for symbol in symbols:
            with st.expander(f"⚙️ {symbol} Strategy Configuration"):
                col1, col2 = st.columns(2)

                with col1:
                    categories = catalog.get_categories()
                    category_names = [cat.value for cat in categories]
                    selected_category = st.selectbox("Category", category_names, key=f"multi_cat_{symbol}")

                with col2:
                    category_enum = next(cat for cat in categories if cat.value == selected_category)
                    strategies_in_category = catalog.get_by_category(category_enum)
                    strategy_names = [info.name for info in strategies_in_category.values()]
                    strategy_type = st.selectbox("Strategy", strategy_names, key=f"multi_strat_{symbol}")

                strategy_key = next(
                    (key for key, info in strategies_in_category.items() if info.name == strategy_type),
                    None,
                )

                if strategy_key:
                    params = _get_strategy_parameters(catalog, strategy_key, f"multi_{symbol}")
                    strategies_config[symbol] = catalog.create_strategy(strategy_key, **params)

    # Capital allocation
    st.markdown("### 💰 Capital Allocation")

    allocation_method = st.radio(
        "Allocation Method",
        ["Equal Weight", "Custom Weights", "Risk Parity"],
        horizontal=True,
        key="multi_allocation",
    )

    allocations = {}
    if allocation_method == "Custom Weights":
        st.markdown("Set allocation for each symbol (must sum to 100%):")

        cols = st.columns(min(len(symbols), 4))
        for i, symbol in enumerate(symbols):
            with cols[i % len(cols)]:
                weight = st.number_input(
                    f"{symbol} %",
                    min_value=0,
                    max_value=100,
                    value=int(100 / len(symbols)),
                    key=f"weight_{symbol}",
                )
                allocations[symbol] = weight / 100

        total_weight = sum(allocations.values())
        if abs(total_weight - 1.0) > 0.01:
            st.warning(f"⚠️ Weights sum to {total_weight * 100:.1f}%, should be 100%")

    # Risk management
    st.markdown("### 🛡️ Risk Management")
    col1, col2 = st.columns(2)

    with col1:
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000,
            max_value=1000000,
            value=DEFAULT_INITIAL_CAPITAL,
            step=10000,
            key="multi_capital",
        )

    with col2:
        max_position_pct = st.slider(
            "Max Position per Symbol (%)",
            min_value=5,
            max_value=50,
            value=20,
            help="Maximum position size per symbol as % of portfolio",
            key="multi_position",
        )

    # Run backtest
    if st.button("🚀 Run Multi-Asset Backtest", type="primary", key="multi_run"):
        if not strategies_config or len(strategies_config) != len(symbols):
            st.error("❌ Please configure strategies for all symbols")
            return

        # ML Training Phase
        with st.spinner("Training ML models..."):
            for symbol, strategy in strategies_config.items():
                if hasattr(strategy, "is_trained") and not strategy.is_trained:
                    try:
                        # Fetch more data for training
                        train_data = fetch_stock_data(symbol, period="2y", interval=interval)
                        if len(train_data) > 100:  # Need sufficient data
                            train_score, test_score = strategy.train(train_data)
                            st.info(f"✓ {symbol}: Train={train_score:.1%}, Test={test_score:.1%}")
                        else:
                            st.warning(f"⚠️ {symbol}: Insufficient training data ({len(train_data)} rows)")
                    except Exception as e:
                        st.error(f"❌ {symbol}: Training failed - {str(e)}")

        # Backtest Execution Phase
        with st.spinner(f"Running backtest on {len(symbols)} assets..."):
            try:
                # Update risk manager with max position
                risk_manager.max_position_size = max_position_pct / 100

                # Create multi-asset engine
                engine = MultiAssetEngine(
                    strategies=strategies_config,
                    initial_capital=initial_capital,
                    risk_manager=risk_manager,
                    db=db,
                    allocation_method=("custom" if allocation_method == "Custom Weights" else "equal"),
                )

                if allocation_method == "Custom Weights":
                    engine.allocations = allocations

                # Run backtest
                engine.run_backtest(symbols, period, interval)

                # Get results
                results = engine.get_results()

                st.success("✅ Multi-asset backtest completed!")

                # Display overall metrics
                _display_multi_asset_metrics(results)

                # Display per-symbol performance
                _display_symbol_breakdown(results)

                # Display equity curve
                _display_multi_asset_equity_curve(engine.equity_curve, initial_capital)

                # Display trade log
                _display_multi_asset_trades(engine.trades)

            except Exception as e:
                st.error(f"❌ Error during backtest: {str(e)}")
                with st.expander("Show detailed error"):
                    import traceback

                    st.code(traceback.format_exc())


def _get_strategy_parameters(catalog, strategy_key: str, prefix: str) -> dict:
    """
    Get strategy parameters from UI without creating strategy instance

    Args:
        catalog: Strategy catalog
        strategy_key: Strategy key
        prefix: Widget key prefix

    Returns:
        Dictionary of parameters
    """
    info = catalog.get_info(strategy_key)
    params = {}

    if info.parameters:
        st.markdown("**Parameters:**")
        cols = st.columns(min(len(info.parameters), 3))

        for i, (param_name, param_info) in enumerate(info.parameters.items()):
            with cols[i % len(cols)]:
                if param_info["range"]:
                    min_val, max_val = param_info["range"]
                    params[param_name] = st.slider(
                        param_name.replace("_", " ").title(),
                        min_value=min_val,
                        max_value=max_val,
                        value=param_info["default"],
                        help=param_info.get("description", ""),
                        key=f"{prefix}_{param_name}_param",
                    )
                else:
                    params[param_name] = st.number_input(
                        param_name.replace("_", " ").title(),
                        value=param_info["default"],
                        help=param_info.get("description", ""),
                        key=f"{prefix}_{param_name}_param",
                    )

    return params


def _display_multi_asset_metrics(results: dict):
    """Display multi-asset backtest metrics"""

    st.subheader("📊 Portfolio Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Return", f"{results['total_return']:.2f}%")
    with col2:
        st.metric("Win Rate", f"{results['win_rate']:.1f}%")
    with col3:
        st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
    with col4:
        st.metric("Max Drawdown", f"{results['max_drawdown']:.2f}%")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Trades", results["total_trades"])
    with col2:
        st.metric("Symbols Traded", results["num_symbols"])
    with col3:
        st.metric("Avg Profit", f"${results['avg_profit']:.2f}")
    with col4:
        st.metric("Final Equity", f"${results['final_equity']:,.2f}")


def _display_symbol_breakdown(results: dict):
    """Display per-symbol performance breakdown"""

    st.subheader("📈 Per-Symbol Performance")

    symbol_stats = results.get("symbol_stats", {})

    if symbol_stats:
        df = pd.DataFrame(
            [
                {
                    "Symbol": symbol,
                    "Strategy": stats["strategy"],
                    "Total Profit": f"${stats['total_profit']:.2f}",
                    "Trades": stats["num_trades"],
                    "Win Rate": f"{stats['win_rate']:.1f}%",
                    "Avg Profit": f"${stats['avg_profit']:.2f}",
                }
                for symbol, stats in symbol_stats.items()
            ]
        )

        st.dataframe(df, use_container_width=True, hide_index=True)

        # Profit contribution chart
        profit_data = pd.DataFrame([{"Symbol": symbol, "Profit": stats["total_profit"]} for symbol, stats in symbol_stats.items()])

        fig = go.Figure(
            data=[
                go.Bar(
                    x=profit_data["Symbol"],
                    y=profit_data["Profit"],
                    marker_color=["green" if p > 0 else "red" for p in profit_data["Profit"]],
                )
            ]
        )

        fig.update_layout(
            title="Profit Contribution by Symbol",
            xaxis_title="Symbol",
            yaxis_title="Total Profit ($)",
            template="plotly_dark",
        )

        st.plotly_chart(fig, use_container_width=True)


def _display_multi_asset_equity_curve(equity_curve: list, initial_capital: float):
    """Display equity curve for multi-asset backtest"""

    st.subheader("📈 Portfolio Equity Curve")

    if not equity_curve:
        st.warning("No equity curve data available")
        return

    equity_df = pd.DataFrame(equity_curve)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=equity_df["timestamp"],
            y=equity_df["equity"],
            mode="lines",
            name="Portfolio Value",
            line=dict(color="#00ff88", width=2),
            fill="tonexty",
            fillcolor="rgba(0, 255, 136, 0.1)",
        )
    )

    fig.add_hline(
        y=initial_capital,
        line_dash="dash",
        line_color="gray",
        annotation_text="Initial Capital",
    )

    fig.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="Time",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Number of positions over time
    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            x=equity_df["timestamp"],
            y=equity_df["num_positions"],
            mode="lines",
            name="Active Positions",
            line=dict(color="cyan", width=2),
        )
    )

    fig2.update_layout(
        title="Active Positions Over Time",
        template="plotly_dark",
        height=300,
        xaxis_title="Time",
        yaxis_title="Number of Positions",
        hovermode="x unified",
    )

    st.plotly_chart(fig2, use_container_width=True)


def _display_multi_asset_trades(trades: list):
    """Display trade log for multi-asset backtest"""

    st.subheader("📋 Trade Log")

    if not trades:
        st.info("No trades executed during backtest")
        return

    trades_df = pd.DataFrame(trades)

    # Summary by symbol
    st.markdown("**Trades by Symbol:**")
    summary = trades_df.groupby("symbol").size().reset_index(name="Count")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # Recent trades
    st.markdown("**Recent Trades:**")
    display_df = pd.DataFrame(
        [
            {
                "Timestamp": t["timestamp"],
                "Symbol": t["symbol"],
                "Type": t["order_type"],
                "Quantity": t["quantity"],
                "Price": f"${t['price']:.2f}",
                "Strategy": t["strategy"],
                "Profit": f"${t.get('profit', 0):.2f}" if t.get("profit") else "-",
                "Profit %": (f"{t.get('profit_pct', 0):.2f}%" if t.get("profit_pct") else "-"),
            }
            for t in trades[-50:]
        ]
    )  # Last 50 trades

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Download option
    csv = pd.DataFrame(trades).to_csv(index=False)
    st.download_button(
        label="📥 Download All Trades",
        data=csv,
        file_name=f"multi_asset_trades_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
