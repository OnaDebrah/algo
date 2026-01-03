from auth import UserTier, require_tier
from core.context import configure_page
from ui.components.sidebar import render_page_sidebar
from ui.components.theme import OracleTheme
from ui.marketplace import marketplace


@require_tier(UserTier.FREE)
def render_marketplace():
    OracleTheme.apply_theme()

    configure_page("Marketplace", "🏪")
    render_page_sidebar()

    marketplace()


render_marketplace()
