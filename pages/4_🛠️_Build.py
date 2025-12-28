"""
Build Page - Strategy development and ML model creation
"""

import streamlit as st

from auth import UserTier, require_tier
from core.context import configure_page, get_app_context
from ui.custom_strategy_builder import render_custom_strategy_builder
from ui.ml_builder import render_ml_builder
from ui.sidebar import render_page_sidebar
from ui.theme import OracleTheme


@require_tier(UserTier.PRO)
def render_build():
    OracleTheme.apply_theme()

    configure_page("Build", "🛠️")
    render_page_sidebar()
    # Get context
    context = get_app_context()

    # Page header
    st.markdown(
        """
        <div class="page-header">
            <h1>🛠️ Build</h1>
            <p style="color: var(--text-secondary);">
                Create custom strategies and train machine learning models
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Navigation tabs
    tab1, tab2 = st.tabs(["🤖 ML Strategy Studio", "🧪 Custom Strategy Builder"])

    with tab1:
        render_ml_builder(context.ml_models)

    with tab2:
        render_custom_strategy_builder()


render_build()
