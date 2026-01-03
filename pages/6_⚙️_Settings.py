"""
Settings Page - System configuration and management
"""

import streamlit as st

from auth import require_auth
from core.context import configure_page, get_app_context
from ui.components.sidebar import render_page_sidebar
from ui.components.theme import OracleTheme
from ui.configuration import render_configuration


@require_auth
def render_settings():
    OracleTheme.apply_theme()

    configure_page("Settings", "⚙️")
    render_page_sidebar()
    # Get context
    context = get_app_context()

    # Page header
    st.markdown(
        """
        <div class="page-header">
            <h1>⚙️ Settings</h1>
            <p style="color: var(--text-secondary);">
                System configuration, risk management, and alert settings
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Render configuration
    render_configuration(context.db, context.alert_manager, context.risk_manager)


render_settings()
