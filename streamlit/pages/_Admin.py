"""
Admin panel for user management
Only accessible to admin users
"""

import pandas as pd

import streamlit as st
from streamlit.auth.auth_manager import UserTier
from streamlit.auth.streamlit_auth import render_user_menu, require_auth

st.set_page_config(page_title="Admin Panel", page_icon="⚙️", layout="wide")


@require_auth
def render_admin_panel():
    """Render admin panel"""

    render_user_menu()

    # Check if user is admin (you can add is_admin field to users table)
    user = st.session_state.user

    # For now, check if username is admin (implement proper admin check in production)
    if user["username"] != "admin":
        st.error("🚫 Access Denied - Admin privileges required")
        st.stop()

    st.title("⚙️ Admin Panel")

    tab1, tab2, tab3, tab4 = st.tabs(["👥 Users", "📊 Usage Stats", "💰 Subscriptions", "⚙️ System"])

    with tab1:
        render_user_management()

    with tab2:
        render_usage_stats()

    with tab3:
        render_subscription_management()

    with tab4:
        render_system_settings()


def render_user_management():
    """User management interface"""
    st.subheader("User Management")

    auth_manager = st.session_state.auth_manager

    # Get all users
    import sqlite3

    conn = sqlite3.connect(auth_manager.db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, username, email, tier, created_at, last_login, is_active
        FROM users
        ORDER BY created_at DESC
    """)

    users = cursor.fetchall()
    conn.close()

    if users:
        users_df = pd.DataFrame(users, columns=["ID", "Username", "Email", "Tier", "Created", "Last Page", "Active"])

        st.dataframe(users_df, use_container_width=True, hide_index=True)

        # User actions
        st.subheader("User Actions")

        col1, col2 = st.columns(2)

        with col1:
            user_id = st.number_input("User ID", min_value=1, step=1)

        with col2:
            new_tier = st.selectbox("Change Tier To", [tier.value for tier in UserTier])

        if st.button("Update Tier"):
            tier_enum = UserTier(new_tier)
            if auth_manager.update_user_tier(user_id, tier_enum):
                st.success(f"✅ Updated user {user_id} to {new_tier}")
                st.rerun()
            else:
                st.error("❌ Failed to update tier")

        # Deactivate user
        if st.button("⚠️ Deactivate User", type="secondary"):
            conn = sqlite3.connect(auth_manager.db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET is_active = 0 WHERE id = ?", (user_id,))
            conn.commit()
            conn.close()
            st.success(f"✅ User {user_id} deactivated")
            st.rerun()

    else:
        st.info("No users found")


def render_usage_stats():
    """Usage statistics dashboard"""
    st.subheader("Usage Statistics")

    auth_manager = st.session_state.auth_manager

    # Get usage data
    import sqlite3

    conn = sqlite3.connect(auth_manager.db_path)

    # Total users by tier
    df_tiers = pd.read_sql_query(
        """
        SELECT tier, COUNT(*) as count
        FROM users
        WHERE is_active = 1
        GROUP BY tier
    """,
        conn,
    )

    # Recent activity
    df_activity = pd.read_sql_query(
        """
        SELECT
            DATE(timestamp) as date,
            action,
            COUNT(*) as count
        FROM usage_tracking
        WHERE timestamp > date('now', '-30 days')
        GROUP BY DATE(timestamp), action
        ORDER BY date DESC
    """,
        conn,
    )

    conn.close()

    # Display metrics
    col1, col2, col3 = st.columns(3)

    total_users = df_tiers["count"].sum()

    with col1:
        st.metric("Total Active Users", total_users)

    with col2:
        if not df_activity.empty:
            st.metric("Actions (30 days)", df_activity["count"].sum())
        else:
            st.metric("Actions (30 days)", 0)

    with col3:
        paying_users = df_tiers[df_tiers["tier"] != "free"]["count"].sum()
        st.metric("Paying Users", paying_users)

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Users by Tier")
        if not df_tiers.empty:
            st.bar_chart(df_tiers.set_index("tier"))

    with col2:
        st.subheader("Recent Activity")
        if not df_activity.empty:
            st.dataframe(df_activity, use_container_width=True, hide_index=True)


def render_subscription_management():
    """Subscription and billing management"""
    st.subheader("Subscription Management")

    st.info("💡 Integrate with Stripe, Paddle, or your payment processor here")

    # Revenue metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("MRR (Monthly Recurring Revenue)", "$2,450")

    with col2:
        st.metric("Churn Rate", "3.2%")

    with col3:
        st.metric("LTV (Lifetime Value)", "$840")

    # Pricing management
    st.subheader("Pricing Configuration")

    pricing = {"FREE": 0, "BASIC": 29, "PRO": 99, "ENTERPRISE": "Custom"}

    for tier, price in pricing.items():
        col1, col2 = st.columns([3, 1])

        with col1:
            st.text(f"{tier} Tier")

        with col2:
            if isinstance(price, int):
                st.text(f"${price}/mo")
            else:
                st.text(price)


def render_system_settings():
    """System settings and configuration"""
    st.subheader("System Settings")

    # Database info
    st.markdown("**Database**")
    auth_manager = st.session_state.auth_manager
    st.code(f"Path: {auth_manager.db_path}")

    # Rate limiting
    st.markdown("**Rate Limiting**")
    st.info("Configure rate limits per tier here")

    rate_limits = {"FREE": "10 backtests/month", "BASIC": "100 backtests/month", "PRO": "Unlimited", "ENTERPRISE": "Unlimited + Priority"}

    for tier, limit in rate_limits.items():
        st.text(f"{tier}: {limit}")

    # System health
    st.markdown("**System Health**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Uptime", "99.9%", "0.1%")

    with col2:
        st.metric("Avg Response Time", "120ms", "-5ms")

    with col3:
        st.metric("Error Rate", "0.02%", "0.01%")

    # Clear sessions
    if st.button("🗑️ Clear All Sessions", type="secondary"):
        import sqlite3

        conn = sqlite3.connect(auth_manager.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sessions")
        conn.commit()
        conn.close()
        st.success("✅ All sessions cleared")


render_admin_panel()
