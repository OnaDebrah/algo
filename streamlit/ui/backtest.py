"""
Backtesting UI Component
"""

import streamlit as st

from streamlit.alerts.alert_manager import AlertManager
from streamlit.core.database import DatabaseManager
from streamlit.core.risk_manager import RiskManager
from streamlit.ui.render_multi_asset_backtest import render_multi_asset_backtest
from streamlit.ui.render_options_backtest import render_options_backtest
from streamlit.ui.render_single_asset_backtest import render_single_asset_backtest

try:
    MULTI_ASSET_ENABLED = True
except ImportError:
    MULTI_ASSET_ENABLED = False


def render_backtest(
    db: DatabaseManager,
    risk_manager: RiskManager,
    ml_models: dict,
    alert_manager: AlertManager,
):
    """
    Render the backtesting tab

    Args:
        db: Database manager instance
        risk_manager: Risk manager instance
        ml_models: Dictionary of trained ML models
        alert_manager: Alert manager instance
    """
    st.header("🔬 Strategy Backtesting")

    # Backtest Mode Selection
    mode_options = ["Single Asset", "Multi-Asset Portfolio"]

    # Add Options mode if available
    try:
        mode_options.append("Options Strategies")
        OPTIONS_ENABLED = True
    except ImportError:
        OPTIONS_ENABLED = False

    backtest_mode = st.radio(
        "Backtest Mode",
        mode_options,
        horizontal=True,
        help="Single: One symbol. Multi: Portfolio of symbols. Options: Options strategies.",
    )

    if backtest_mode == "Single Asset":
        render_single_asset_backtest(db, risk_manager, ml_models, alert_manager)
    elif backtest_mode == "Multi-Asset Portfolio":
        render_multi_asset_backtest(db, risk_manager, ml_models, alert_manager)
    elif backtest_mode == "Options Strategies" and OPTIONS_ENABLED:
        render_options_backtest(db, risk_manager)
