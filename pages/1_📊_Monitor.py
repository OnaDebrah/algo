"""
Monitor Page - Portfolio tracking and live execution
"""

import streamlit as st

from auth import require_auth
from core.context import configure_page, get_app_context
from ui.components.dashboard import render_dashboard
from ui.components.sidebar import render_page_sidebar
from ui.components.theme import OracleTheme
from ui.live import render_live_trading


@require_auth
def render_monitor():
    OracleTheme.apply_theme()

    configure_page("Monitor", "📊")

    render_page_sidebar()

    context = get_app_context()

    st.markdown(
        """
        <div class="page-header">
            <h1>📊 Monitor</h1>
            <p style="color: var(--text-secondary);">
                Real-time portfolio tracking and live execution desk
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["📊 Dashboard", "⚡ Live Execution"])

    with tab1:
        render_dashboard(context.db)

    with tab2:
        render_live_trading(context.db, context.risk_manager, context.ml_models, context.alert_manager)


render_monitor()
