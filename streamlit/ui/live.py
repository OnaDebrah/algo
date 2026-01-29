"""
Live Trading UI Component
"""

import pandas as pd
import streamlit as st

from streamlit.alerts.alert_manager import AlertManager
from streamlit.core.database import DatabaseManager
from streamlit.core.risk_manager import RiskManager
from streamlit.live.alpaca_broker import AlpacaBroker
from streamlit.live.ib_broker import IBBroker, check_ib_connection
from streamlit.live.live_engine import LiveTradingEngine, ScheduledTrading
from streamlit.live.paper_broker import PaperBroker
from streamlit.strategies.macd_strategy import MACDStrategy
from streamlit.strategies.rsi_strategy import RSIStrategy
from streamlit.strategies.sma_crossover import SMACrossoverStrategy


def render_live_trading(
    db: DatabaseManager,
    risk_manager: RiskManager,
    ml_models: dict,
    alert_manager: AlertManager,
):
    """
    Render live trading tab

    Args:
        db: Database manager
        risk_manager: Risk manager
        ml_models: ML models dictionary
        alert_manager: Alert manager
    """
    st.header("⚡ Live Trading")

    st.warning(
        """
    ⚠️ **WARNING: Live Trading Involves Real Money Risk**

    - Start with paper trading to test your strategies
    - Never risk more than you can afford to lose
    - Monitor your positions regularly
    - Use stop losses and risk management
    """
    )

    # Initialize session state
    if "broker" not in st.session_state:
        st.session_state.broker = None
    if "live_engine" not in st.session_state:
        st.session_state.live_engine = None

    # Broker Configuration
    st.subheader("🔌 Broker Configuration")

    broker_type = st.selectbox("Select Broker", ["Interactive Brokers", "Paper Trading (Simulated)", "Alpaca"])

    if broker_type == "Interactive Brokers":
        _configure_ib_broker()
    elif broker_type == "Paper Trading (Simulated)":
        _configure_paper_broker()
    elif broker_type == "Alpaca":
        _configure_alpaca_broker()

    st.divider()

    # Check if broker is connected
    if st.session_state.broker and st.session_state.broker.connected:
        _render_trading_interface(db, risk_manager, ml_models, alert_manager)
    else:
        st.info("👆 Connect to a broker to start live trading")


def _configure_paper_broker():
    """Configure paper trading broker"""
    col1, col2 = st.columns(2)

    with col1:
        initial_cash = st.number_input(
            "Initial Cash ($)",
            min_value=1000,
            max_value=1000000,
            value=100000,
            step=10000,
        )

    with col2:
        st.write("")
        st.write("")
        if st.button("🔌 Connect to Paper Trading", type="primary"):
            broker = PaperBroker(initial_cash=initial_cash)
            if broker.connect():
                st.session_state.broker = broker
                st.success("✅ Connected to Paper Trading!")
                st.rerun()


def _configure_ib_broker():
    """Configure Interactive Brokers"""
    st.markdown(
        """
    **Interactive Brokers Setup:**
    1. Install IB Gateway or TWS (Trader Workstation)
    2. Enable API connections in settings
    3. Start IB Gateway/TWS
    4. Configure connection below

    [📥 Download IB Gateway](https://www.interactivebrokers.com/en/index.php?f=16457)
    """
    )

    # Connection test first
    with st.expander("🔍 Test Connection First"):
        st.markdown("Test if IB Gateway/TWS is running and accessible:")

        test_col1, test_col2 = st.columns(2)
        with test_col1:
            test_host = st.text_input("Test Host", value="127.0.0.1", key="test_host")
            test_port = st.number_input("Test Port", value=7497, key="test_port")

        if st.button("🧪 Test Connection"):
            with st.spinner("Testing connection..."):
                result = check_ib_connection(test_host, test_port)

                if result["connected"]:
                    st.success(f"✅ {result['message']}")
                    st.info(f"Accounts: {result['accounts']}")
                else:
                    st.error(f"❌ {result['message']}")

    st.divider()

    # Main configuration
    col1, col2 = st.columns(2)

    with col1:
        host = st.text_input("Host", value="127.0.0.1", help="Usually 127.0.0.1 for local connection")

        connection_type = st.selectbox(
            "Connection Type",
            [
                "TWS Paper Trading (Port 7497)",
                "TWS Live Trading (Port 7496)",
                "IB Gateway Paper (Port 4002)",
                "IB Gateway Live (Port 4001)",
                "Custom Port",
            ],
        )

        # Set port based on selection
        port_map = {
            "TWS Paper Trading (Port 7497)": 7497,
            "TWS Live Trading (Port 7496)": 7496,
            "IB Gateway Paper (Port 4002)": 4002,
            "IB Gateway Live (Port 4001)": 4001,
            "Custom Port": 7497,
        }

        default_port = port_map[connection_type]

        if connection_type == "Custom Port":
            port = st.number_input("Port", min_value=1000, max_value=9999, value=7497)
        else:
            port = default_port
            st.info(f"Port: {port}")

    with col2:
        client_id = st.number_input(
            "Client ID",
            min_value=0,
            max_value=999,
            value=1,
            help="Unique ID for this connection (0-999)",
        )

        paper_mode = "Paper" in connection_type

        if paper_mode:
            st.success("🟢 Paper Trading Mode")
        else:
            st.error("🔴 LIVE Trading Mode - Real Money!")
            st.warning("⚠️ You will be trading with real money!")

    # Important notes
    st.info(
        """
    **Before Connecting:**
    - ✅ IB Gateway or TWS is running
    - ✅ API connections enabled in settings
    - ✅ Socket port matches above
    - ✅ "Enable ActiveX and Socket Clients" is checked
    - ✅ Correct paper/live mode selected
    """
    )

    if st.button("🔌 Connect to Interactive Brokers", type="primary"):
        with st.spinner("Connecting to IB..."):
            try:
                broker = IBBroker(host=host, port=port, client_id=client_id, paper=paper_mode)

                if broker.connect():
                    st.session_state.broker = broker
                    st.success(f"✅ Connected to Interactive Brokers ({'Paper' if paper_mode else 'Live'} mode)!")

                    # Show account info
                    account = broker.get_account()
                    st.write("**Account Info:**")
                    st.write(f"- Account ID: {account['account_id']}")
                    st.write(f"- Portfolio Value: ${account['portfolio_value']:,.2f}")
                    st.write(f"- Cash: ${account['cash']:,.2f}")
                    st.write(f"- Buying Power: ${account['buying_power']:,.2f}")

                    st.rerun()
                else:
                    st.error("Failed to connect. Check the error message above.")

            except Exception as e:
                st.error(f"Connection failed: {str(e)}")
                st.write("**Troubleshooting:**")
                st.write("1. Is IB Gateway/TWS running?")
                st.write("2. Is API enabled in settings?")
                st.write("3. Is the port correct?")
                st.write("4. Try restarting IB Gateway/TWS")


def _configure_paper_broker_old():
    """Configure paper trading broker"""
    col1, col2 = st.columns(2)

    with col1:
        initial_cash = st.number_input(
            "Initial Cash ($)",
            min_value=1000,
            max_value=1000000,
            value=100000,
            step=10000,
        )

    with col2:
        st.write("")
        st.write("")
        if st.button("🔌 Connect to Paper Trading", type="primary"):
            broker = PaperBroker(initial_cash=initial_cash)
            if broker.connect():
                st.session_state.broker = broker
                st.success("✅ Connected to Paper Trading!")
                st.rerun()


def _configure_alpaca_broker():
    """Configure Alpaca broker"""
    st.markdown(
        """
    **Alpaca Setup:**
    1. Create account at [alpaca.markets](https://alpaca.markets)
    2. Get your API keys from the dashboard
    3. Enter your credentials below
    """
    )

    col1, col2 = st.columns(2)

    with col1:
        api_key = st.text_input("API Key", type="password")
        paper_mode = st.checkbox("Paper Trading Mode", value=True)

    with col2:
        secret_key = st.text_input("Secret Key", type="password")

    if st.button("🔌 Connect to Alpaca", type="primary"):
        if not api_key or not secret_key:
            st.error("Please enter both API Key and Secret Key")
            return

        try:
            broker = AlpacaBroker(api_key, secret_key, paper=paper_mode)
            if broker.connect():
                st.session_state.broker = broker
                st.success(f"✅ Connected to Alpaca ({'Paper' if paper_mode else 'Live'} mode)!")
                st.rerun()
        except Exception as e:
            st.error(f"Failed to connect: {str(e)}")


def _render_trading_interface(
    db: DatabaseManager,
    risk_manager: RiskManager,
    ml_models: dict,
    alert_manager: AlertManager,
):
    """Render main trading interface"""

    # Account Overview
    st.subheader("💼 Account Overview")

    account = st.session_state.broker.get_account()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Portfolio Value", f"${account['equity']:,.2f}")
    with col2:
        st.metric("Cash", f"${account['cash']:,.2f}")
    with col3:
        st.metric("Buying Power", f"${account['buying_power']:,.2f}")
    with col4:
        pnl = account["equity"] - (account["equity"] - (account["equity"] - account.get("equity", account["equity"])))
        st.metric("Day P&L", f"${pnl:0:.2f}")  # Would need to track this

    # Tabs.tsx for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Start Trading", "📊 Positions", "📋 Orders", "⚙️ Settings"])

    with tab1:
        _render_start_trading(db, risk_manager, ml_models, alert_manager)

    with tab2:
        _render_positions()

    with tab3:
        _render_orders()

    with tab4:
        _render_trading_settings(risk_manager)


def _render_start_trading(
    db: DatabaseManager,
    risk_manager: RiskManager,
    ml_models: dict,
    alert_manager: AlertManager,
):
    """Render start trading section"""

    st.subheader("Configure Strategy")

    col1, col2 = st.columns(2)

    with col1:
        symbols_input = st.text_input(
            "Symbols (comma-separated)",
            value="AAPL,MSFT,GOOGL",
            help="Enter stock symbols separated by commas",
            key="live_symbols_input",
        )
        symbols = [s.strip().upper() for s in symbols_input.split(",")]

        strategy_type = st.selectbox(
            "Strategy",
            ["SMA Crossover", "RSI", "MACD", "ML Model"],
            key="live_strategy_type",
        )

    with col2:
        check_interval = st.number_input(
            "Check Interval (seconds)",
            min_value=30,
            max_value=3600,
            value=300,
            help="How often to check for signals",
            key="live_check_interval",
        )

        auto_schedule = st.checkbox(
            "Auto Schedule (Market Hours Only)",
            value=True,
            help="Automatically start/stop during market hours",
            key="live_auto_schedule",
        )

    # Strategy parameters
    if strategy_type == "SMA Crossover":
        col1, col2 = st.columns(2)
        with col1:
            short_window = st.slider("Short Window", 5, 50, 20, key="live_sma_short")
        with col2:
            long_window = st.slider("Long Window", 20, 200, 50, key="live_sma_long")
        strategy = SMACrossoverStrategy(short_window, long_window)

    elif strategy_type == "RSI":
        col1, col2, col3 = st.columns(3)
        with col1:
            rsi_period = st.slider("RSI Period", 5, 30, 14, key="live_rsi_period")
        with col2:
            oversold = st.slider("Oversold", 10, 40, 30, key="live_rsi_oversold")
        with col3:
            overbought = st.slider("Overbought", 60, 90, 70, key="live_rsi_overbought")
        strategy = RSIStrategy(rsi_period, oversold, overbought)

    elif strategy_type == "MACD":
        strategy = MACDStrategy()
        st.info("Using default MACD parameters (12, 26, 9)")

    else:  # ML Model
        if not ml_models:
            st.error("No ML models trained. Train a model in the ML Strategy Builder tab first.")
            return

        model_symbol = st.selectbox("Select trained model", list(ml_models.keys()))
        strategy = ml_models[model_symbol]

    # Control buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("▶️ Start Trading", type="primary"):
            if st.session_state.live_engine and st.session_state.live_engine.running:
                st.warning("Trading already running!")
            else:
                engine = LiveTradingEngine(
                    broker=st.session_state.broker,
                    strategy=strategy,
                    symbols=symbols,
                    risk_manager=risk_manager,
                    db=db,
                    alert_manager=alert_manager,
                    check_interval=check_interval,
                )

                if auto_schedule:
                    scheduler = ScheduledTrading(engine)
                    scheduler.start_scheduler()
                    st.session_state.scheduler = scheduler
                else:
                    engine.start()

                st.session_state.live_engine = engine
                st.success("✅ Live trading started!")
                st.rerun()

    with col2:
        if st.button("⏸️ Stop Trading", type="secondary"):
            if st.session_state.live_engine:
                if hasattr(st.session_state, "scheduler"):
                    st.session_state.scheduler.stop_scheduler()
                else:
                    st.session_state.live_engine.stop()
                st.success("Trading stopped")
                st.rerun()

    with col3:
        if st.button("🛑 Emergency Stop & Close All", type="secondary"):
            if st.session_state.live_engine:
                st.session_state.live_engine.close_all_positions()
                st.session_state.live_engine.stop()
                st.warning("All positions closed and trading stopped")
                st.rerun()

    # Status
    if st.session_state.live_engine:
        status = st.session_state.live_engine.get_status()

        st.divider()
        st.subheader("📊 Trading Status")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Status", "🟢 RUNNING" if status["running"] else "🔴 STOPPED")
        with col2:
            st.metric("Strategy", status["strategy"])
        with col3:
            st.metric("Symbols", len(status["symbols"]))
        with col4:
            st.metric("Active Positions", status["active_positions"])


def _render_positions():
    """Render positions section"""
    st.subheader("Current Positions")

    positions = st.session_state.broker.get_positions()

    if positions:
        df = pd.DataFrame(positions)

        # Format for display
        display_df = df[
            [
                "symbol",
                "quantity",
                "entry_price",
                "current_price",
                "market_value",
                "unrealized_pl",
                "unrealized_plpc",
            ]
        ].copy()

        display_df.columns = [
            "Symbol",
            "Qty",
            "Entry Price",
            "Current Price",
            "Market Value",
            "Unrealized P&L",
            "P&L %",
        ]

        # Format currency columns
        for col in ["Entry Price", "Current Price", "Market Value", "Unrealized P&L"]:
            display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}")

        display_df["P&L %"] = display_df["P&L %"].apply(lambda x: f"{x:.2%}")

        st.dataframe(display_df, use_container_width=True)

        # Summary metrics
        total_value = sum(p["market_value"] for p in positions)
        total_pl = sum(p["unrealized_pl"] for p in positions)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Position Value", f"${total_value:,.2f}")
        with col2:
            st.metric("Total Unrealized P&L", f"${total_pl:,.2f}")
    else:
        st.info("No open positions")


def _render_orders():
    """Render orders section"""
    st.subheader("Order History")

    orders = st.session_state.broker.get_orders()

    if orders:
        df = pd.DataFrame(orders)

        # Format for display
        display_df = df[
            [
                "order_id",
                "symbol",
                "side",
                "quantity",
                "order_type",
                "status",
                "filled_qty",
                "filled_avg_price",
                "submitted_at",
            ]
        ].copy()

        display_df.columns = [
            "Order ID",
            "Symbol",
            "Side",
            "Qty",
            "Type",
            "Status",
            "Filled",
            "Avg Price",
            "Time",
        ]

        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No orders yet")


def _render_trading_settings(risk_manager: RiskManager):
    """Render trading settings"""
    st.subheader("Trading Settings")

    # Disconnect button
    if st.button("🔌 Disconnect Broker"):
        st.session_state.broker.disconnect()
        st.session_state.broker = None
        if st.session_state.live_engine:
            st.session_state.live_engine.stop()
            st.session_state.live_engine = None
        st.success("Disconnected from broker")
        st.rerun()

    st.divider()

    # Risk settings display
    st.markdown("**Current Risk Settings:**")
    st.write(f"- Max Position Size: {risk_manager.max_position_size * 100:.0f}%")
    st.write(f"- Stop Loss: {risk_manager.stop_loss_pct * 100:.0f}%")
    st.write(f"- Max Drawdown: {risk_manager.max_drawdown * 100:.0f}%")

    st.info("💡 Update risk settings in the Configuration tab")
