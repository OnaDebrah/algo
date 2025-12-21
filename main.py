"""
Algorithmic Trading Platform
========================================================
A trading system with backtesting, ML strategies,
multi-symbol support, and comprehensive analytics.

Installation:
pip install -r requirements.txt

Usage:
streamlit run main.py
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
from ui.ml_builder import render_ml_builder

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
    st.set_page_config(page_title=PAGE_TITLE, layout=LAYOUT, page_icon=PAGE_ICON)

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


def main():
    """Main application entry point"""
    configure_page()
    initialize_session_state()

    # Header
    st.title("📈 Advanced Algorithmic Trading Platform")
    st.markdown(
        "**System with Backtesting, ML Strategies & " "Comprehensive Analytics**"
    )

    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Portfolio Dashboard",
            "🔬 Backtest",
            "🤖 ML Strategy Builder",
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
        render_configuration(
            st.session_state.db,
            st.session_state.alert_manager,
            st.session_state.risk_manager,
        )


if __name__ == "__main__":
    main()
