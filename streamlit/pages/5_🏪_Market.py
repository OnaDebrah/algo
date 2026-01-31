from streamlit.auth import UserTier, require_tier
from streamlit.core.context import configure_page
from streamlit.ui import OracleTheme, marketplace
from streamlit.ui.components.sidebar import render_page_sidebar


@require_tier(UserTier.FREE)
def render_marketplace():
    OracleTheme.apply_theme()

    configure_page("Marketplace", "🏪")
    render_page_sidebar()

    marketplace()


render_marketplace()
