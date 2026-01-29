"""
ORACULUM - Institutional Landing Page
"""

import streamlit as st

from streamlit.auth import init_auth_state
from streamlit.ui import (
    render_authenticated_home,
    render_cta_section,
    render_features_section,
    render_footer,
    render_landing_hero,
    render_navbar,
    render_pricing_section,
    render_social_proof,
    render_stats_section,
)
from streamlit.ui import OracleTheme

st.set_page_config(page_title="ORACULUM - Institutional Trading Platform", page_icon="🏛️", layout="wide", initial_sidebar_state="collapsed")

init_auth_state()

OracleTheme.apply_theme()

is_authenticated = st.session_state.get("authenticated", False)

if is_authenticated:
    render_navbar()
    render_authenticated_home()
else:
    render_landing_hero()
    render_stats_section()
    render_features_section()
    render_pricing_section()
    render_social_proof()
    render_cta_section()
    render_footer()
