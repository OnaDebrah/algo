"""
Advanced Algorithmic Trading Platform
===================================================================
"""

import logging

import streamlit as st

from alerts.alert_manager import AlertManager
from config import LAYOUT, PAGE_ICON, PAGE_TITLE
from core.database import DatabaseManager
from core.risk_manager import RiskManager
from ui.backtest import render_backtest
from ui.configuration import render_configuration
from ui.dashboard import render_dashboard
from ui.live import render_live_trading
from ui.ml_builder import render_ml_builder
from ui.portfolio_optimisation import render_portfolio_optimization

logger = logging.getLogger(__name__)


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "db" not in st.session_state:
        st.session_state.db = DatabaseManager()

    if "alert_manager" not in st.session_state:
        st.session_state.alert_manager = AlertManager()

    if "ml_models" not in st.session_state:
        st.session_state.ml_models = {}

    if "risk_manager" not in st.session_state:
        st.session_state.risk_manager = RiskManager()


def configure_page():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title=PAGE_TITLE,
        layout=LAYOUT,
        page_icon=PAGE_ICON,
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown(
        """
        <style>
        .main {background-color: #0e1117;}
        .stMetric {
            background-color: #1e2130;
            padding: 15px;
            border-radius: 10px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render sidebar with platform info"""
    with st.sidebar:
        st.title("🚀 Trading Platform")

        st.markdown("---")

        st.markdown("### 📊 Features")
        st.markdown(
            """
        - ✅ Backtesting
        - ✅ ML Strategies
        - ✅ **Live Trading**
        - ✅ **Portfolio Optimization**
        - ✅ Risk Management
        - ✅ Alerts
        """
        )

        st.markdown("---")

        st.markdown("### 🔗 Quick Links")
        st.markdown("[📖 Documentation](#)")
        st.markdown("[💬 Community](#)")
        st.markdown("[🐛 Report Bug](#)")

        st.markdown("---")

        st.caption("v1.1.0 - With Live Trading & Portfolio Optimization")


def main():
    """Main application entry point"""
    configure_page()
    initialize_session_state()
    render_sidebar()

    # Header
    st.title("📈 Advanced Algorithmic Trading Platform")
    st.markdown(
        "**Full-Featured Trading System: Backtesting • Live Trading • "
        "Portfolio Optimization • ML Strategies**"
    )

    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📊 Dashboard",
            "🔬 Backtest",
            "🤖 ML Builder",
            "⚡ Live Trading",
            "📊 Portfolio Optimization",
            "⚙️ Configuration",
        ]
    )

    # Render each tab
    with tab1:
        render_dashboard(st.session_state.db)

    with tab2:
        render_backtest(
            st.session_state.db,
            st.session_state.risk_manager,
            st.session_state.ml_models,
            st.session_state.alert_manager,
        )

    with tab3:
        render_ml_builder(st.session_state.ml_models)

    with tab4:
        render_live_trading(
            st.session_state.db,
            st.session_state.risk_manager,
            st.session_state.ml_models,
            st.session_state.alert_manager,
        )

    with tab5:
        render_portfolio_optimization()

    with tab6:
        render_configuration(
            st.session_state.db,
            st.session_state.alert_manager,
            st.session_state.risk_manager,
        )


if __name__ == "__main__":
    main()
