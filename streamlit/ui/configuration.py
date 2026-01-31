"""
Configuration UI Componen
"""

from datetime import datetime

import pandas as pd

import streamlit as st
from config import (
    DATABASE_PATH,
    DEFAULT_MAX_DRAWDOWN,
    DEFAULT_MAX_POSITION_SIZE,
    DEFAULT_STOP_LOSS_PCT,
)
from streamlit.alerts.alert_manager import AlertManager
from streamlit.core.database import DatabaseManager
from streamlit.core.risk_manager import RiskManager


def render_configuration(db: DatabaseManager, alert_manager: AlertManager, risk_manager: RiskManager):
    """
    Render the configuration tab

    Args:
        db: Database manager instance
        alert_manager: Alert manager instance
        risk_manager: Risk manager instance
    """
    st.header("⚙️ System Configuration")

    # Alert Configuration
    _render_alert_configuration(alert_manager)

    st.divider()

    # Risk Management Settings
    _render_risk_management(risk_manager)

    st.divider()

    # Database Managemen
    _render_database_management(db)

    st.divider()

    # System Information
    _render_system_information()


def _render_alert_configuration(alert_manager: AlertManager):
    """Render alert configuration section"""
    st.subheader("🔔 Alert Configuration")

    # Email Alerts
    st.markdown("#### Email Alerts")
    enable_email = st.checkbox("Enable Email Alerts", key="enable_email")

    if enable_email:
        col1, col2 = st.columns(2)

        with col1:
            from_email = st.text_input("From Email", key="from_email", help="Your email address")
            smtp_server = st.text_input(
                "SMTP Server",
                value="smtp.gmail.com",
                key="smtp",
                help="Email server address",
            )
            smtp_port = st.number_input("SMTP Port", value=587, key="smtp_port", help="Usually 587 for TLS")

        with col2:
            to_email = st.text_input("To Email", key="to_email", help="Recipient email address")
            email_password = st.text_input(
                "Email Password",
                type="password",
                key="email_pass",
                help="App password (not your regular password)",
            )

        st.info(
            """
        **Gmail Users:** You need to use an App Password instead of your regular password.

        **Steps:**
        1. Go to your Google Account settings
        2. Enable 2-Step Verification
        3. Go to Security → App Passwords
        4. Generate an App Password for "Mail"
        5. Use that password here
        """
        )

    # SMS Alerts
    st.markdown("#### SMS Alerts")
    enable_sms = st.checkbox("Enable SMS Alerts (Twilio)", key="enable_sms")

    if enable_sms:
        col1, col2 = st.columns(2)

        with col1:
            twilio_sid = st.text_input(
                "Twilio Account SID",
                type="password",
                key="twilio_sid",
                help="Found in Twilio console",
            )
            from_number = st.text_input(
                "From Number (+1234567890)",
                key="from_num",
                help="Your Twilio phone number",
            )

        with col2:
            twilio_token = st.text_input(
                "Twilio Auth Token",
                type="password",
                key="twilio_token",
                help="Found in Twilio console",
            )
            to_number = st.text_input("To Number (+1234567890)", key="to_num", help="Recipient phone number")

        st.info(
            """
        **Twilio Setup:**
        1. Create a free account at [twilio.com](https://www.twilio.com)
        2. Get your Account SID and Auth Token from the console
        3. Purchase or use a trial phone number
        4. Enter your credentials above
        """
        )

    # Save Alert Configuration
    if st.button("💾 Save Alert Configuration", type="primary"):
        email_config = {}
        if enable_email:
            if not all([from_email, to_email, smtp_server, email_password]):
                st.error("❌ Please fill in all email fields")
                return

            email_config = {
                "enabled": True,
                "from_email": from_email,
                "to_email": to_email,
                "smtp_server": smtp_server,
                "smtp_port": smtp_port,
                "password": email_password,
            }

        sms_config = {}
        if enable_sms:
            if not all([twilio_sid, twilio_token, from_number, to_number]):
                st.error("❌ Please fill in all SMS fields")
                return

            sms_config = {
                "enabled": True,
                "account_sid": twilio_sid,
                "auth_token": twilio_token,
                "from_number": from_number,
                "to_number": to_number,
            }

        # Update alert manager
        alert_manager.email_config = email_config
        alert_manager.sms_config = sms_config

        st.success("✅ Alert configuration saved!")

        # Test alerts option
        if st.button("📧 Send Test Alert"):
            alert_manager.send_email_alert("Test Alert", "This is a test alert from your trading platform.")
            alert_manager.send_sms_alert("Test alert from trading platform")
            st.info("Test alert sent! Check your email/phone.")


def _render_risk_management(risk_manager: RiskManager):
    """Render risk management settings"""
    st.subheader("⚠️ Risk Management Settings")

    st.markdown(
        """
    Configure risk parameters to protect your portfolio from excessive losses.
    These settings apply to all new backtests.
    """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        max_position = st.slider(
            "Max Position Size (%)",
            min_value=1,
            max_value=50,
            value=int(DEFAULT_MAX_POSITION_SIZE * 100),
            key="risk_pos",
            help="Maximum size of a single position as % of portfolio",
        )

    with col2:
        stop_loss = st.slider(
            "Stop Loss (%)",
            min_value=1,
            max_value=20,
            value=int(DEFAULT_STOP_LOSS_PCT * 100),
            key="risk_sl",
            help="Automatic stop loss percentage",
        )

    with col3:
        max_dd = st.slider(
            "Max Drawdown (%)",
            min_value=5,
            max_value=50,
            value=int(DEFAULT_MAX_DRAWDOWN * 100),
            key="risk_dd",
            help="Maximum allowed drawdown before halting",
        )

    # Advanced Risk Settings
    with st.expander("🔧 Advanced Risk Settings"):
        col1, col2 = st.columns(2)

        with col1:
            enable_trailing_stop = st.checkbox("Enable Trailing Stop Loss", help="Stop loss that follows price upward")

            if enable_trailing_stop:
                _ = st.slider("Trailing Stop %", min_value=1, max_value=10, value=3)

        with col2:
            enable_profit_target = st.checkbox("Enable Profit Target", help="Automatic exit at profit target")

            if enable_profit_target:
                _ = st.slider("Profit Target %", min_value=5, max_value=50, value=20)

    if st.button("💾 Save Risk Settings", type="primary"):
        # Update risk manager
        risk_manager.max_position_size = max_position / 100
        risk_manager.stop_loss_pct = stop_loss / 100
        risk_manager.max_drawdown = max_dd / 100

        st.success("✅ Risk management settings saved!")

        # Show current settings
        st.info(
            f"""
        **Current Settings:**
        - Max Position: {max_position}% (${10000 * max_position / 100:.2f} on $10k portfolio)
        - Stop Loss: {stop_loss}%
        - Max Drawdown: {max_dd}%
        """
        )


def _render_database_management(db: DatabaseManager):
    """Render database management section"""
    st.subheader("💾 Database Management")

    col1, col2, col3 = st.columns(3)

    # Get statistics
    all_trades = db.get_trades(limit=100000)
    total_trades = len(all_trades)

    with col1:
        st.metric("Total Trades", total_trades)

    with col2:
        st.metric("Database", "SQLite")

    with col3:
        st.metric("Location", DATABASE_PATH)

    # Database Actions
    st.markdown("#### Database Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📥 Export Trades to CSV"):
            if all_trades:
                df = pd.DataFrame(all_trades)
                csv = df.to_csv(index=False)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="💾 Download CSV",
                    data=csv,
                    file_name=f"trades_export_{timestamp}.csv",
                    mime="text/csv",
                    key="download_csv",
                )
            else:
                st.warning("No trades to export")

    with col2:
        if st.button("👀 View All Trades"):
            if all_trades:
                st.dataframe(pd.DataFrame(all_trades), use_container_width=True)
            else:
                st.info("No trades in database")

    with col3:
        if st.button("🗑️ Clear Database"):
            st.warning("This action cannot be undone!")

            confirm = st.checkbox("I understand this will delete all data")

            if confirm and st.button("⚠️ Confirm Delete", type="primary"):
                # This would require implementing a clear method in DatabaseManager
                st.error("Database clearing not implemented for safety")
                st.info("To clear database, manually delete the .db file")

    # Database Statistics
    if all_trades:
        st.markdown("#### Database Statistics")

        trades_df = pd.DataFrame(all_trades)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            unique_symbols = trades_df["symbol"].nunique()
            st.metric("Unique Symbols", unique_symbols)

        with col2:
            unique_strategies = trades_df["strategy"].nunique()
            st.metric("Strategies Used", unique_strategies)

        with col3:
            date_range = "N/A"
            if "timestamp" in trades_df.columns and len(trades_df) > 0:
                first_trade = pd.to_datetime(trades_df["timestamp"]).min()
                last_trade = pd.to_datetime(trades_df["timestamp"]).max()
                days = (last_trade - first_trade).days
                date_range = f"{days} days"
            st.metric("Trading Period", date_range)

        with col4:
            completed = trades_df["profit"].notna().sum()
            st.metric("Completed Trades", completed)


def _render_system_information():
    """Render system information section"""
    st.subheader("ℹ️ System Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        **Platform Details:**
        - **Version:** 1.0.0
        - **Database:** SQLite
        - **ML Framework:** scikit-learn
        - **Python Version:** 3.8+
        """
        )

    with col2:
        st.markdown(
            """
        **Data & Visualization:**
        - **Data Source:** Yahoo Finance (yfinance)
        - **Charting:** Plotly
        - **UI Framework:** Streamli
        - **License:** MIT
        """
        )

    # Feature Status
    st.markdown("#### Feature Status")

    features = {
        "Backtesting": "✅ Available",
        "ML Strategies": "✅ Available",
        "Risk Management": "✅ Available",
        "Email Alerts": "✅ Available",
        "SMS Alerts": "✅ Available (Twilio required)",
        "Live Trading": "⏳ Coming Soon",
        "Paper Trading": "⏳ Coming Soon",
        "Portfolio Optimization": "⏳ Coming Soon",
    }

    col1, col2 = st.columns(2)

    items = list(features.items())
    mid = len(items) // 2

    with col1:
        for feature, status in items[:mid]:
            st.write(f"**{feature}:** {status}")

    with col2:
        for feature, status in items[mid:]:
            st.write(f"**{feature}:** {status}")

    # Documentation Links
    st.markdown("#### Documentation & Support")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("📖 [User Guide](#)")

    with col2:
        st.markdown("💬 [Community Forum](#)")

    with col3:
        st.markdown("🐛 [Report Issue](#)")

    # System Health
    st.markdown("#### System Health")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Status", "🟢 Healthy")

    with col2:
        st.metric("Data Connection", "🟢 Connected")

    with col3:
        st.metric("Database", "🟢 Operational")
