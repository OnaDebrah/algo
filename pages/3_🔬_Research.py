"""
Research Page - Backtesting, options, and portfolio optimization
"""

import streamlit as st

from auth import UserTier, require_tier
from core.context import configure_page, get_app_context
from ui.backtest import render_backtest
from ui.portfolio_optimisation import render_portfolio_optimization
from ui.sidebar import render_page_sidebar
from ui.theme import OracleTheme


@require_tier(UserTier.FREE)
def render_research():
    OracleTheme.apply_theme()

    configure_page("Research", "🔬")
    render_page_sidebar()
    # Get context
    context = get_app_context()

    # Page header
    st.markdown(
        """
        <div class="page-header">
            <h1>🔬 Research</h1>
            <p style="color: var(--text-secondary);">
                Advanced backtesting, options strategies, and portfolio optimization
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Navigation tabs
    tab1, tab2, tab3 = st.tabs(["🔬 Backtesting Lab", "📈 Options Desk", "📊 Portfolio Optimization"])

    with tab1:
        render_backtest(context.db, context.risk_manager, context.ml_models, context.alert_manager)

    with tab2:
        try:
            from ui.options_builder_ui import render_options_strategy_builder

            render_options_strategy_builder()
        except ImportError:
            st.info("📦 Options module available after installation")
            st.code("pip install scipy")

    with tab3:
        render_portfolio_optimization()


render_research()
