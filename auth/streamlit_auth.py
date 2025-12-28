"""
Streamlit authentication integration
"""

from functools import wraps
from typing import Callable

import streamlit as st

from auth.auth_manager import AuthManager, Permission, UserTier


def init_auth_state():
    """Initialize authentication state"""
    if "auth_manager" not in st.session_state:
        st.session_state.auth_manager = AuthManager()
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user" not in st.session_state:
        st.session_state.user = None
    if "token" not in st.session_state:
        st.session_state.token = None


def render_login_page():
    """Render login/registration page"""
    init_auth_state()
    st.title("🔐 ORACULUM")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        st.subheader("Login to Your Account")
        with st.form("login_form"):
            username = st.text_input("Username or Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login", use_container_width=True)

            if submit:
                if not username or not password:
                    st.error("Please enter both username and password")
                else:
                    result = st.session_state.auth_manager.login(username, password)
                    if result["success"]:
                        st.session_state.authenticated = True
                        st.session_state.user = result["user"]
                        st.session_state.token = result["token"]
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error(f"❌ {result['message']}")

    with tab2:
        st.subheader("Create New Account")
        with st.form("register_form"):
            new_username = st.text_input("Username")
            new_email = st.text_input("Email")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            st.info("💡 New accounts start with a FREE tier. Upgrade anytime!")
            register = st.form_submit_button("Create Account", use_container_width=True)

            if register:
                if not all([new_username, new_email, new_password, confirm_password]):
                    st.error("Please fill in all fields")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    result = st.session_state.auth_manager.register_user(new_username, new_email, new_password, UserTier.FREE)
                    if result["success"]:
                        st.success("✅ Account created! Please login.")
                    else:
                        st.error(f"❌ {result['message']}")


def render_user_menu():
    """Render user menu in sidebar"""
    if not st.session_state.get("authenticated"):
        return

    user = st.session_state.user
    with st.sidebar:
        st.divider()
        st.markdown(f"### 👤 {user['username']}")
        tier = user["tier"].upper()
        tier_colors = {"FREE": "🆓", "BASIC": "🥉", "PRO": "⭐", "ENTERPRISE": "💎"}
        st.markdown(f"**Tier:** {tier_colors.get(tier, '')} {tier}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("⚙️ Settings", use_container_width=True):
                st.session_state.show_settings = True
        with col2:
            if st.button("🚪 Logout", use_container_width=True):
                logout()


def logout():
    """Logout current user"""
    if st.session_state.get("token"):
        st.session_state.auth_manager.logout(st.session_state.token)
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.token = None
    st.rerun()


def require_auth(func: Callable) -> Callable:
    """Decorator to require authentication"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        init_auth_state()
        if not st.session_state.authenticated:
            render_login_page()
            st.stop()
        return func(*args, **kwargs)

    return wrapper


def require_permission(permission: Permission):
    """Decorator to require specific permission"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            init_auth_state()
            if not st.session_state.authenticated:
                render_login_page()
                st.stop()
            if not st.session_state.auth_manager.has_permission(st.session_state.user["tier"], permission):
                render_upgrade_page(permission)
                st.stop()
            return func(*args, **kwargs)

        return wrapper

    return decorator


def require_tier(min_tier: UserTier):
    """Decorator to require minimum tier"""
    tier_order = [UserTier.FREE, UserTier.BASIC, UserTier.PRO, UserTier.ENTERPRISE]

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            init_auth_state()
            if not st.session_state.authenticated:
                render_login_page()
                st.stop()
            user_tier = UserTier(st.session_state.user["tier"])
            if tier_order.index(user_tier) < tier_order.index(min_tier):
                render_upgrade_page_for_tier(min_tier)
                st.stop()
            return func(*args, **kwargs)

        return wrapper

    return decorator


def render_upgrade_page(permission: Permission):
    """Render upgrade prompt"""
    st.title("🔒 Upgrade Required")
    st.warning(f"This feature requires **{permission.value.replace('_', ' ').title()}** permission.")
    render_pricing_table()


def render_upgrade_page_for_tier(min_tier: UserTier):
    """Render upgrade prompt for tier"""
    st.title("🔒 Upgrade Required")
    st.warning(f"This feature requires **{min_tier.value.upper()}** tier or higher.")
    render_pricing_table()


def render_pricing_table():
    """Render pricing table"""
    st.markdown("---")
    st.subheader("💎 Choose Your Plan")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### 🆓 FREE")
        st.markdown("**€0/month**")
        st.markdown("- Basic dashboard\n- Simple backtests\n- 10 backtests/month")

    with col2:
        st.markdown("### 🥉 BASIC")
        st.markdown("**€0/month**")
        st.markdown("- Multi-asset backtests\n- Advanced analytics\n- 100 backtests/month")
        if st.button("Upgrade", key="basic_btn", type="primary"):
            st.info("Contact sales to upgrade")

    with col3:
        st.markdown("### ⭐ PRO")
        st.markdown("**€0/month**")
        st.markdown("- ML strategies\n- Unlimited backtests\n- Custom strategies")
        if st.button("Upgrade", key="pro_btn", type="primary"):
            st.info("Contact sales to upgrade")

    with col4:
        st.markdown("### 💎 ENTERPRISE")
        st.markdown("**Custom**")
        st.markdown("- Live trading\n- API access\n- Dedicated support")
        if st.button("Contact Sales", key="ent_btn", type="primary"):
            st.info("Contact sales for pricing")


def check_permission(permission: Permission) -> bool:
    """Check if current user has permission"""
    if not st.session_state.get("authenticated"):
        return False
    return st.session_state.auth_manager.has_permission(st.session_state.user["tier"], permission)


def show_feature_locked(feature_name: str, required_permission: Permission):
    """Show locked feature message"""
    st.info(f"🔒 **{feature_name}** is locked. Upgrade to access this feature.")
    if st.button("View Upgrade Options"):
        render_upgrade_page(required_permission)


def track_action(action: str, metadata: str = None):
    """Track user action"""
    if not st.session_state.get("authenticated"):
        return
    st.session_state.auth_manager.track_usage(st.session_state.user["id"], action, metadata)
