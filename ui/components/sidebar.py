"""
Professional Sidebar for ORACULUM
Clean, minimal, and functional
"""

import streamlit as st

from auth.streamlit_auth import init_auth_state


def render_page_sidebar():
    """Optimized professional sidebar hierarchy"""
    init_auth_state()

    # Pre-render logic for admin hiding
    if st.session_state.get("authenticated"):
        user = st.session_state.user
        if user.get("username") != "admin":
            st.markdown('<style>[data-testid="stSidebarNav"] a[href*="Admin"] {display: none !important;}</style>', unsafe_allow_html=True)

    st.markdown(
        """
        <style>
            /* 1. Remove padding from the main sidebar container */
            [data-testid="stSidebarContent"] {
                padding-top: 0rem !important;
            }

            /* 2. Reduce gap between the vertical blocks */
            [data-testid="stSidebarVerticalBlock"] {
                gap: 0rem !important;
            }

            /* 3. Ensure the logo container has zero top margin */
            .sidebar-logo-container {
                margin-top: 0 !important;
                padding-top: 1rem !important;
            }
        </style>
    """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        # 1. BRANDING & VERSIONING
        st.markdown(
            """
            <div class="sidebar-logo-container" style="text-align: center; padding: 1rem 0; border-bottom: 1px solid var(--border-subtle); margin-bottom: 1rem;">
                <div style="font-size: 1.8rem;">🏛️</div>
                <div style="font-size: 1.1rem; font-weight: 800; letter-spacing: 0.1em; color: white;">ORACULUM</div>
                <div style="font-size: 0.65rem; color: var(--text-muted); opacity: 0.7;">STABLE v1.0.0</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. USER CONTEXT (Who is here?)
        if st.session_state.get("authenticated"):
            render_user_profile(st.session_state.user)

            st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)

            # 3. INTERACTION (What can I do quickly?)
            render_quick_actions()
        else:
            render_guest_actions()

        # 4. MONITORING (Are we healthy?)
        st.markdown("---")
        render_system_status()

        # 5. UTILITY
        render_sidebar_footer()


def render_user_profile(user: dict):
    """Render user profile section"""
    tier_emoji = get_tier_emoji(user["tier"])

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            text-align: center;
        ">
            <div style="
                width: 60px;
                height: 60px;
                background: white;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 2rem;
                margin: 0 auto 1rem;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            ">
                👤
            </div>
            <div style="color: white; font-weight: 700; font-size: 1.125rem; margin-bottom: 0.25rem;">
                {user['username']}
            </div>
            <div style="color: rgba(255,255,255,0.8); font-size: 0.875rem; margin-bottom: 1rem;">
                {user['email']}
            </div>
            <div style="
                background: rgba(255,255,255,0.2);
                backdrop-filter: blur(10px);
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-size: 0.75rem;
                font-weight: 700;
                color: white;
                letter-spacing: 0.05em;
            ">
                {tier_emoji} {user['tier'].upper()}
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Action buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("⚙️", help="Settings", use_container_width=True, key="sidebar_settings"):
            st.switch_page("pages/6_⚙️_Settings.py")

    with col2:
        if st.button("🚪", help="Logout", use_container_width=True, key="sidebar_logout"):
            logout_user()

    st.markdown('<div style="margin: 1.5rem 0; border-top: 1px solid var(--border-subtle);"></div>', unsafe_allow_html=True)


def render_guest_actions():
    """Render actions for non-authenticated users"""
    st.markdown(
        """
        <div style="
            text-align: center;
            padding: 1rem;
            background: var(--bg-card);
            border-radius: 12px;
            margin-bottom: 1.5rem;
        ">
            <div style="font-size: 2rem; margin-bottom: 0.75rem;">👋</div>
            <div style="font-size: 0.875rem; color: var(--text-secondary); margin-bottom: 1rem;">
                Sign in to access your account
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    if st.button("🔐 Sign In", use_container_width=True, type="primary", key="sidebar_signin"):
        st.switch_page("pages/_Auth.py")

    if st.button("📝 Create Account", use_container_width=True, key="sidebar_signup"):
        st.switch_page("pages/_Auth.py")

    st.markdown('<div style="margin: 1.5rem 0; border-top: 1px solid var(--border-subtle);"></div>', unsafe_allow_html=True)


def render_system_status():
    """Render system status indicators"""
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <div style="
                font-size: 0.75rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                color: var(--text-muted);
                margin-bottom: 0.75rem;
            ">
                System Status
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    statuses = [
        ("🟢", "Platform", "Operational"),
        ("🟢", "Market Data", "Live"),
        ("🟢", "Execution", "Ready"),
    ]

    for icon, label, status in statuses:
        st.markdown(
            f"""
            <div style="
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 0.5rem 0;
                font-size: 0.875rem;
            ">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span>{icon}</span>
                    <span style="color: var(--text-secondary);">{label}</span>
                </div>
                <span style="color: var(--text-muted); font-size: 0.75rem;">{status}</span>
            </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown('<div style="margin: 1.5rem 0; border-top: 1px solid var(--border-subtle);"></div>', unsafe_allow_html=True)


def render_quick_actions():
    """Render quick action buttons"""
    st.markdown(
        """
        <div style="
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-muted);
            margin-bottom: 0.75rem;
        ">
            Quick Access
        </div>
    """,
        unsafe_allow_html=True,
    )

    if st.button("📊 Dashboard", use_container_width=True, key="quick_dashboard"):
        st.switch_page("pages/1_📊_Monitor.py")

    if st.button("🎯 Analysis", use_container_width=True, key="quick_analyze"):
        st.switch_page("pages/2_🎯_Analyze.py")

    if st.button("🔬 Research", use_container_width=True, key="quick_research"):
        st.switch_page("pages/3_🔬_Research.py")

    if st.button("🏪 Marketplace", use_container_width=True, key="quick_marketplace"):
        st.switch_page("pages/5_🏪_Market.py")


st.markdown('<div style="margin: 1.5rem 0; border-top: 1px solid var(--border-subtle);"></div>', unsafe_allow_html=True)


def render_sidebar_footer():
    """Render sidebar footer"""
    st.markdown(
        """
        <div style="
            margin-top: auto;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border-subtle);
        ">
            <div style="
                font-size: 0.75rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                color: var(--text-muted);
                margin-bottom: 0.75rem;
            ">
                Resources
            </div>
            <div style="font-size: 0.875rem; color: var(--text-secondary); line-height: 2;">
                <a href="https://docs.oraculum.com" style="color: inherit; text-decoration: none; display: block;">
                    📖 Documentation
                </a>
                <a href="https://community.oraculum.com" style="color: inherit; text-decoration: none; display: block;">
                    💬 Community
                </a>
                <a href="https://github.com/oraculum" style="color: inherit; text-decoration: none; display: block;">
                    🐛 Report Issue
                </a>
            </div>
            <div style="
                margin-top: 1.5rem;
                padding-top: 1.5rem;
                border-top: 1px solid var(--border-subtle);
                text-align: center;
                font-size: 0.75rem;
                color: var(--text-muted);
            ">
                © 2026 ORACULUM
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )


def get_tier_emoji(tier: str) -> str:
    """Get emoji for tier"""
    tier_emojis = {"free": "🆓", "basic": "🥉", "pro": "⭐", "enterprise": "💎"}
    return tier_emojis.get(tier.lower(), "🎯")


def logout_user():
    """Logout current user"""
    if st.session_state.get("token"):
        from auth.streamlit_auth import logout

        logout()
    else:
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.token = None
        st.rerun()
