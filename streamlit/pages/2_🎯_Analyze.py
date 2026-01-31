"""
Analyze Page - AI-driven analysis and market intelligence
"""

import streamlit as st
from streamlit.auth import UserTier, require_tier
from streamlit.core.context import configure_page, get_app_context
from streamlit.ui import OracleTheme, render_analyst, render_regime_detector, render_strategy_advisor
from streamlit.ui.components.sidebar import render_page_sidebar


@require_tier(UserTier.FREE)
def render_analyze():
    OracleTheme.apply_theme()

    configure_page("Analyze", "🎯")
    render_page_sidebar()
    # Get context
    context = get_app_context()

    # Page header
    st.markdown(
        """
        <div class="page-header">
            <h1>🤖 AI Strategy Advisor</h1>
            <p style="color: var(--text-secondary);">
                AI-Powered Analysis, Market Regime Detection, and Intelligent Recommendations.
                Get personalized strategy recommendations based on your goals and preferences.
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Navigation tabs
    tab1, tab2, tab3 = st.tabs(["🤖 AI Strategy Advisor", "🎯 Market Regime", "🏦 AI Analyst"])

    with tab1:
        render_strategy_advisor()

    with tab2:
        render_regime_detector(context.db)

        # Update context with current regime
        if "current_regime" in st.session_state:
            context.current_regime = st.session_state.current_regime

    with tab3:
        render_analyst()


render_analyze()
