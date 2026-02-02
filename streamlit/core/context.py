"""
Shared application context for all pages.
This ensures consistent state across page navigation.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import streamlit as st
from streamlit.alerts.alert_manager import AlertManager
from streamlit.core.database import DatabaseManager
from streamlit.core.risk_manager import RiskManager


@dataclass
class AppContext:
    """Centralized application context"""

    db: DatabaseManager
    risk_manager: RiskManager
    alert_manager: AlertManager
    ml_models: Dict
    regime_detector: Optional[object] = None
    current_regime: Optional[Dict] = None


def get_app_context() -> AppContext:
    """
    Get or initialize application context.
    This is called by every page to access shared state.

    Returns:
        AppContext: Initialized context
    """
    if "app_context" not in st.session_state:
        from streamlit.analytics.market_regime_detector import MarketRegimeDetector

        st.session_state.app_context = AppContext(
            db=DatabaseManager(),
            risk_manager=RiskManager(),
            alert_manager=AlertManager(),
            ml_models={},
            regime_detector=MarketRegimeDetector(),
            current_regime=None,
        )

    return st.session_state.app_context


def configure_page(page_title: str, page_icon: str = "📈"):
    """
    Standard page configuration for all pages.

    Args:
        page_title: Page title
        page_icon: Page icon
    """
    st.set_page_config(page_title=f"{page_title} | Trading Platform", page_icon=page_icon, layout="wide", initial_sidebar_state="expanded")

    st.markdown(
        """
        <style>
        /* Professional styling for all pages */
        :root {
            --bg-primary: #0E1117;
            --bg-secondary: #1E2130;
            --accent-blue: #4C78FF;
            --accent-success: #00C853;
            --accent-danger: #FF5252;
            --text-primary: #E6EAF2;
            --text-secondary: #9AA4B2;
        }

        .main {
            background-color: var(--bg-primary);
            color: var(--text-primary);
            padding: 24px;
        }

        .stMetric {
            background: linear-gradient(135deg, var(--bg-secondary) 0%, #262B3D 100%);
            padding: 18px;
            border-radius: 12px;
            border: 1px solid #2D3748;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }

        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Page header styling */
        .page-header {
            background: linear-gradient(135deg, #1a1d29 0%, #252937 100%);
            padding: 24px;
            border-radius: 12px;
            margin-bottom: 24px;
            border-left: 4px solid var(--accent-blue);
        }
        </style>
    """,
        unsafe_allow_html=True,
    )
