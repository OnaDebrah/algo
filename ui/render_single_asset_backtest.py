import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from alerts.alert_manager import AlertManager
from analytics.performance import calculate_performance_metrics
from config import DEFAULT_INITIAL_CAPITAL
from core.data_fetcher import fetch_stock_data, validate_interval_period
from core.database import DatabaseManager
from core.risk_manager import RiskManager
from core.trading_engine import TradingEngine
from strategies.macd_strategy import MACDStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.sma_crossover import SMACrossoverStrategy
from strategies.strategy_catalog import get_catalog

try:
    from core.asset_classes import AssetClass, get_asset_manager

    MULTI_ASSET_ENABLED = True
except ImportError:
    MULTI_ASSET_ENABLED = False

logger = logging.getLogger(__name__)


def render_single_asset_backtest(
    db: DatabaseManager,
    risk_manager: RiskManager,
    ml_models: dict,
    alert_manager: AlertManager,
):
    """Render single asset backtest interface with asset class support"""

    # Get strategy catalog
    catalog = get_catalog()

    # Asset Selection
    if MULTI_ASSET_ENABLED:
        st.markdown("### 🎯 Asset Selection")

        # Use asset selector for enhanced symbol selection
        col1, col2 = st.columns([3, 1])

        with col1:
            asset_mgr = get_asset_manager()

            # Quick asset class selector
            asset_class_filter = st.selectbox(
                "Asset Class",
                ["Stock"] + [ac.value for ac in AssetClass if ac != AssetClass.STOCK],
                help="Select asset class to trade",
                key="backtest_asset_class",
            )

            # Get appropriate asset class
            if asset_class_filter == "Stock":
                selected_class = AssetClass.STOCK
            else:
                selected_class = next(ac for ac in AssetClass if ac.value == asset_class_filter)

            # Symbol input with suggestions
            popular = asset_mgr.get_popular_symbols(selected_class)

            input_method = st.radio(
                "Input Method",
                ["Manual Entry", "Popular Symbols"],
                horizontal=True,
                key="backtest_input_method",
            )

            if input_method == "Manual Entry":
                symbol = st.text_input(
                    "Symbol",
                    value="AAPL" if selected_class == AssetClass.STOCK else popular[0],
                    help=f"Enter {asset_class_filter} symbol",
                    key="backtest_symbol",
                ).upper()

                # Validate symbol
                if symbol:
                    is_valid, message = asset_mgr.validate_symbol(symbol)
                    if is_valid:
                        detected_class = asset_mgr.detect_asset_class(symbol)
                        st.success(f"✅ {symbol} - {detected_class.value}")
                    else:
                        st.error(f"❌ {message}")
            else:
                symbol = st.selectbox(
                    f"Popular {asset_class_filter} Symbols",
                    popular,
                    key="backtest_popular_symbol",
                )

        with col2:
            # Show asset info
            if symbol:
                asset_info = asset_mgr.get_asset_info(symbol)
                with st.container():
                    st.metric("Current Symbol", symbol)
                    st.caption(f"Type: {asset_info.asset_class.value}")
                    st.caption(f"Exchange: {asset_info.exchange or 'N/A'}")
                    st.caption(f"Currency: {asset_info.currency}")
    else:
        # Fallback to simple input
        col1, col2 = st.columns([2, 1])
        with col1:
            symbol = st.text_input(
                "Symbol",
                value="AAPL",
                help="Enter ticker symbol",
                key="backtest_symbol",
            ).upper()

    # Configuration Section
    col1, col2 = st.columns([2, 1])

    with col1:
        # Strategy selection by category
        st.markdown("### 📊 Select Strategy")

        # Show strategy categories
        categories = catalog.get_categories()
        category_names = [cat.value for cat in categories]

        selected_category = st.selectbox(
            "Strategy Category",
            category_names,
            help="Filter strategies by category",
            key="backtest_category",
        )

        # Get strategies in selected category
        category_enum = next(cat for cat in categories if cat.value == selected_category)
        strategies_in_category = catalog.get_by_category(category_enum)
        strategy_names = [info.name for info in strategies_in_category.values()]

        strategy_type = st.selectbox(
            "Strategy",
            strategy_names + (["Machine Learning"] if ml_models else []),
            help="Select trading strategy to backtest",
            key="backtest_strategy_type",
        )

        # Show strategy info
        if strategy_type != "ML Model":
            strategy_key = next(
                (key for key, info in strategies_in_category.items() if info.name == strategy_type),
                None,
            )
            if strategy_key:
                info = catalog.get_info(strategy_key)
                with st.expander("ℹ️ Strategy Details"):
                    st.markdown(f"**Description:** {info.description}")
                    st.markdown(f"**Complexity:** {info.complexity}")
                    st.markdown(f"**Time Horizon:** {info.time_horizon}")
                    st.markdown(f"**Best For:** {', '.join(info.best_for)}")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**Pros:**")
                        for pro in info.pros:
                            st.markdown(f"- {pro}")
                    with col_b:
                        st.markdown("**Cons:**")
                        for con in info.cons:
                            st.markdown(f"- {con}")

    with col2:
        period = st.selectbox(
            "Period",
            ["1mo", "3mo", "6mo", "1y", "2y"],
            help="Historical data period",
            key="single_asset_backtest_period",
        )
        interval = st.selectbox(
            "Interval",
            ["1h", "1d", "1wk"],
            help="Data interval/timeframe",
            key="single_asset_backtest_interval",
        )

    # Strategy-specific parameters
    if strategy_type == "SMA Crossover":
        st.subheader("SMA Parameters")
        col1, col2 = st.columns(2)
        with col1:
            short_window = st.slider("Short Window", 5, 50, 20)
        with col2:
            long_window = st.slider("Long Window", 20, 200, 50)

    elif strategy_type == "RSI Strategy":
        st.subheader("RSI Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            rsi_period = st.slider("RSI Period", 5, 30, 14)
        with col2:
            oversold = st.slider("Oversold", 10, 40, 30)
        with col3:
            overbought = st.slider("Overbought", 60, 90, 70)

    elif strategy_type == "MACD Strategy":
        st.subheader("MACD Parameters")
        st.info("Using default MACD parameters (12, 26, 9)")

    # Risk Management Section
    st.subheader("Risk Management")
    col1, col2 = st.columns(2)

    with col1:
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000,
            max_value=1000000,
            value=DEFAULT_INITIAL_CAPITAL,
            step=10000,
        )
    with col2:
        _ = st.slider(
            "Max Position (%)",
            min_value=5,
            max_value=50,
            value=10,
            help="Maximum position size as % of portfolio",
        )

    # Run Backtest Button
    if st.button("🚀 Run Backtest", type="primary"):
        with st.spinner(f"Running backtest on {symbol}..."):
            try:
                # Validate and fetch data
                interval, period = validate_interval_period(interval, period)
                data = fetch_stock_data(symbol, period, interval)

                if data.empty:
                    st.error("❌ No data returned")
                    st.write("**Possible reasons:**")
                    st.write("- Invalid ticker symbol")
                    st.write("- Market closed/no trading data available")
                    st.write("- Yahoo Finance API temporarily unavailable")
                    st.write("- Network/firewall blocking the request")

                    # Suggest alternatives if using asset selector
                    if MULTI_ASSET_ENABLED:
                        try:
                            asset_mgr = get_asset_manager()
                            detected_class = asset_mgr.detect_asset_class(symbol)
                            popular = asset_mgr.get_popular_symbols(detected_class)
                            st.info(f"💡 Try these {detected_class.value} symbols: {', '.join(popular[:5])}")
                        except Exception as e:
                            logger.error(e)
                            pass
                    return

                st.success(f"✅ Fetched {len(data)} data points for {symbol}")

                STRATEGY_FACTORY = {
                    "SMA Crossover": lambda: SMACrossoverStrategy(short_window, long_window),
                    "RSI Strategy": lambda: RSIStrategy(rsi_period, oversold, overbought),
                    "MACD Strategy": lambda: MACDStrategy(),
                }

                ML_STRATEGIES = {
                    "ML Random Forest",
                    "ML Gradient Boosting",
                }

                strategy = None

                if strategy_type in ML_STRATEGIES:
                    if symbol not in ml_models:
                        st.error("❌ ML model not trained for this symbol. Please train in ML tab first.")
                        return
                    strategy = ml_models[symbol]

                else:
                    try:
                        strategy = STRATEGY_FACTORY[strategy_type]()
                    except KeyError:
                        st.error(f"❌ Unknown strategy type: {strategy_type}")
                        return

                # Run backtest
                engine = TradingEngine(strategy, initial_capital, risk_manager, db)
                engine.run_backtest(symbol, data)

                # Calculate metrics
                metrics = calculate_performance_metrics(engine.trades, engine.equity_curve, initial_capital)

                st.success("✅ Backtest completed!")

                # Display Metrics
                _display_metrics(metrics)

                # Display Equity Curve
                _display_equity_curve(engine.equity_curve, initial_capital)

                # Display Price Chart with Signals
                _display_price_chart(
                    data,
                    strategy,
                    strategy_type,
                    engine,
                    short_window if strategy_type == "SMA Crossover" else None,
                    long_window if strategy_type == "SMA Crossover" else None,
                )

                # Display Trade Log
                _display_trade_log(engine.trades)

            except Exception as e:
                st.error(f"❌ Error during backtest: {str(e)}")
                st.write("**Troubleshooting tips:**")
                st.write("1. Check if the ticker symbol is correct")
                st.write("2. Try a well-known symbol like 'AAPL' or 'MSFT'")
                st.write("3. Check your internet connection")
                st.write("4. Yahoo Finance may be temporarily unavailable")

                with st.expander("Show detailed error"):
                    st.code(str(e))


def _display_metrics(metrics: dict):
    """Display performance metrics"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Return", f"{metrics['total_return']:.2f}%")
    with col2:
        st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
    with col3:
        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    with col4:
        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")

    # Additional metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Trades", metrics["total_trades"])
    with col2:
        st.metric("Winning Trades", metrics["winning_trades"])
    with col3:
        st.metric("Avg Profit", f"${metrics['avg_profit']:.2f}")
    with col4:
        st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")


def _display_equity_curve(equity_curve: list, initial_capital: float):
    """Display equity curve chart"""
    st.subheader("📈 Equity Curve")

    if equity_curve:
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
            showlegend=True,
        )

        st.plotly_chart(fig, use_container_width=True)


def _display_price_chart(
    data: pd.DataFrame,
    strategy,
    strategy_type: str,
    engine,
    short_window=None,
    long_window=None,
):
    """Display price chart with trading signals"""
    st.subheader("📊 Price Chart & Trading Signals")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    # Add strategy indicators
    if strategy_type == "SMA Crossover" and short_window and long_window:
        sma_short = data["Close"].rolling(window=short_window).mean()
        sma_long = data["Close"].rolling(window=long_window).mean()

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=sma_short,
                name=f"SMA {short_window}",
                line=dict(color="cyan", width=1),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=sma_long,
                name=f"SMA {long_window}",
                line=dict(color="orange", width=1),
            ),
            row=1,
            col=1,
        )

    # Mark buy/sell signals
    buy_signals = [t for t in engine.trades if t["order_type"] == "BUY"]
    sell_signals = [t for t in engine.trades if t["order_type"] == "SELL"]

    if buy_signals:
        buy_times = [pd.Timestamp(t["timestamp"]) for t in buy_signals]
        buy_prices = [t["price"] for t in buy_signals]

        fig.add_trace(
            go.Scatter(
                x=buy_times,
                y=buy_prices,
                mode="markers",
                name="Buy Signal",
                marker=dict(color="lime", size=12, symbol="triangle-up"),
            ),
            row=1,
            col=1,
        )

    if sell_signals:
        sell_times = [pd.Timestamp(t["timestamp"]) for t in sell_signals]
        sell_prices = [t["price"] for t in sell_signals]

        fig.add_trace(
            go.Scatter(
                x=sell_times,
                y=sell_prices,
                mode="markers",
                name="Sell Signal",
                marker=dict(color="red", size=12, symbol="triangle-down"),
            ),
            row=1,
            col=1,
        )

    # Volume chart
    colors = ["red" if row["Close"] < row["Open"] else "green" for _, row in data.iterrows()]
    fig.add_trace(
        go.Bar(x=data.index, y=data["Volume"], name="Volume", marker_color=colors),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)


def _display_trade_log(trades: list):
    """Display trade log table"""
    if trades:
        st.subheader("📋 Trade Log")

        trades_display = pd.DataFrame(
            [
                {
                    "Timestamp": t["timestamp"],
                    "Type": t["order_type"],
                    "Quantity": t["quantity"],
                    "Price": f"${t['price']:.2f}",
                    "Profit": f"${t.get('profit', 0):.2f}" if t.get("profit") else "-",
                    "Profit %": (f"{t.get('profit_pct', 0):.2f}%" if t.get("profit_pct") else "-"),
                }
                for t in trades
            ]
        )

        st.dataframe(trades_display, use_container_width=True)

        # Download trade log
        csv = pd.DataFrame(trades).to_csv(index=False)
        st.download_button(
            label="📥 Download Trade Log",
            data=csv,
            file_name=f"trade_log_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
