"""
Authentication Page - Page/Register
Professional design with ORACULUM theme
"""

import streamlit as st
from streamlit.auth.auth_manager import UserTier
from streamlit.auth.streamlit_auth import init_auth_state
from streamlit.ui import OracleTheme

# Page config
st.set_page_config(page_title="ORACULUM - Sign In", page_icon="🔐", layout="centered", initial_sidebar_state="collapsed")

# Apply theme
OracleTheme.apply_theme()

# Initialize auth
init_auth_state()

# Check if already authenticated
if st.session_state.get("authenticated"):
    st.success("✅ You're already logged in!")
    st.info("👉 Redirecting to home...")
    if st.button("🏠 Go to Home", use_container_width=True, type="primary"):
        st.switch_page("main.py")
    st.stop()

# Custom CSS for auth page
st.markdown(
    """
    <style>
    /* Center the auth container */
    .main .block-container {
        max-width: 600px;
        padding-top: 3rem;
    }

    /* Auth header */
    .auth-header {
        text-align: center;
        margin-bottom: 3rem;
    }

    .auth-logo {
        font-size: 4rem;
        margin-bottom: 1rem;
    }

    .auth-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }

    .auth-subtitle {
        color: var(--text-secondary);
        font-size: 1.125rem;
    }

    /* Auth container */
    .auth-container {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    }

    /* Input styling */
    .stTextInput > div > div > input {
        background: var(--bg-secondary);
        border: 2px solid var(--border-subtle);
        border-radius: 12px;
        padding: 0.875rem 1.25rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
    }

    /* Button styling */
    .stButton > button {
        border-radius: 12px;
        padding: 0.875rem 1.5rem;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.3s ease;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--bg-secondary);
        border-radius: 12px;
        padding: 0.375rem;
        margin-bottom: 2rem;
    }

    .stTabs [data-baseweb="tab"] {
        flex: 1;
        border-radius: 8px;
        padding: 0.875rem 1.5rem;
        font-weight: 600;
    }

    /* Alert styling */
    .stAlert {
        border-radius: 12px;
        border-left-width: 4px;
    }

    /* Checkbox */
    .stCheckbox {
        padding: 0.5rem 0;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class="auth-header animate-fade-in">
        <div class="auth-logo">🏛️</div>
        <div class="auth-title">Welcome to ORACULUM</div>
        <div class="auth-subtitle">Institutional-grade algorithmic trading platform</div>
    </div>
""",
    unsafe_allow_html=True,
)

# Tabs.tsx for Page/Register
tab1, tab2 = st.tabs(["🔐 Sign In", "📝 Create Account"])

with tab1:
    st.markdown("### Sign in to your account")
    st.markdown('<div style="margin-bottom: 1.5rem;"></div>', unsafe_allow_html=True)

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username or Email", placeholder="Enter your username or email", key="login_username", label_visibility="collapsed")

        st.markdown('<div style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)

        password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password", label_visibility="collapsed")

        st.markdown('<div style="margin-bottom: 1.5rem;"></div>', unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])

        with col1:
            submit = st.form_submit_button("Sign In →", use_container_width=True, type="primary")

        with col2:
            back = st.form_submit_button("Back", use_container_width=True)

        if back:
            st.switch_page("main.py")

        if submit:
            if not username or not password:
                st.error("❌ Please enter both username and password")
            else:
                with st.spinner("Signing in..."):
                    auth_manager = st.session_state.auth_manager
                    result = auth_manager.login(username, password)

                    if result["success"]:
                        st.session_state.authenticated = True
                        st.session_state.user = result["user"]
                        st.session_state.token = result["token"]

                        st.success("✅ Page successful!")
                        st.balloons()

                        # Small delay to show success message
                        import time

                        time.sleep(1)

                        # Redirect to home
                        st.switch_page("main.py")
                    else:
                        st.error(f"❌ {result['message']}")

    # Forgot password link
    st.markdown(
        """
        <div style="text-align: center; margin-top: 1.5rem;">
            <a href="#" style="
                color: var(--primary);
                text-decoration: none;
                font-size: 0.875rem;
                font-weight: 600;
            ">Forgot password?</a>
        </div>
    """,
        unsafe_allow_html=True,
    )

with tab2:
    st.markdown("### Create your ORACULUM account")
    st.markdown('<div style="margin-bottom: 1.5rem;"></div>', unsafe_allow_html=True)

    # Benefits callout
    st.info("💡 **Start with FREE tier** - No credit card required. Upgrade anytime.")

    with st.form("register_form", clear_on_submit=False):
        new_username = st.text_input("Username", placeholder="Choose a unique username", key="reg_username", label_visibility="collapsed")

        st.markdown('<div style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)

        new_email = st.text_input("Email", placeholder="your.email@example.com", key="reg_email", label_visibility="collapsed")

        st.markdown('<div style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)

        new_password = st.text_input(
            "Password", type="password", placeholder="Create a strong password (min 8 characters)", key="reg_password", label_visibility="collapsed"
        )

        st.markdown('<div style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)

        confirm_password = st.text_input(
            "Confirm Password", type="password", placeholder="Confirm your password", key="reg_confirm", label_visibility="collapsed"
        )

        st.markdown('<div style="margin-bottom: 1.5rem;"></div>', unsafe_allow_html=True)

        # Terms acceptance
        accept_terms = st.checkbox("I agree to the Terms of Service and Privacy Policy", key="terms")

        st.markdown('<div style="margin-bottom: 1.5rem;"></div>', unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])

        with col1:
            register = st.form_submit_button("Create Account →", use_container_width=True, type="primary")

        with col2:
            back_reg = st.form_submit_button("Back", use_container_width=True)

        if back_reg:
            st.switch_page("main.py")

        if register:
            # Validation
            errors = []

            if not all([new_username, new_email, new_password, confirm_password]):
                errors.append("Please fill in all fields")

            if not accept_terms:
                errors.append("Please accept the Terms of Service")

            if new_password != confirm_password:
                errors.append("Passwords do not match")

            if len(new_password) < 8:
                errors.append("Password must be at least 8 characters")

            if "@" not in new_email:
                errors.append("Please enter a valid email address")

            # Password strength check
            if new_password and len(new_password) >= 8:
                has_upper = any(c.isupper() for c in new_password)
                has_lower = any(c.islower() for c in new_password)
                has_digit = any(c.isdigit() for c in new_password)

                if not (has_upper and has_lower and has_digit):
                    errors.append("Password should contain uppercase, lowercase, and numbers")

            if errors:
                for error in errors:
                    st.error(f"❌ {error}")
            else:
                with st.spinner("Creating your account..."):
                    auth_manager = st.session_state.auth_manager
                    result = auth_manager.register_user(new_username, new_email, new_password, UserTier.FREE)

                    if result["success"]:
                        st.success("✅ Account created successfully!")
                        st.balloons()
                        st.info("👉 Please switch to the **Sign In** tab to log in")
                    else:
                        st.error(f"❌ {result['message']}")

    # What you get
    st.markdown(
        """
        <div style="
            margin-top: 2rem;
            padding: 1.5rem;
            background: var(--bg-secondary);
            border-radius: 12px;
            border-left: 4px solid var(--primary);
        ">
            <div style="font-weight: 700; margin-bottom: 0.75rem; color: var(--text-primary);">
                ✨ What you get with FREE tier:
            </div>
            <ul style="color: var(--text-secondary); font-size: 0.9rem; line-height: 1.8; margin: 0;">
                <li>Portfolio Dashboard</li>
                <li>Basic Backtesting</li>
                <li>Market Data Access</li>
                <li>Community Support</li>
            </ul>
        </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <div style="
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid var(--border-subtle);
    ">
        <p style="color: var(--text-muted); font-size: 0.875rem; margin-bottom: 0.5rem;">
            Need help getting started?
        </p>
        <div style="display: flex; gap: 1.5rem; justify-content: center; font-size: 0.875rem;">
            <a href="#" style="color: var(--primary); text-decoration: none; font-weight: 600;">
                📖 Documentation
            </a>
            <a href="#" style="color: var(--primary); text-decoration: none; font-weight: 600;">
                💬 Support
            </a>
            <a href="#" style="color: var(--primary); text-decoration: none; font-weight: 600;">
                📧 Contact Sales
            </a>
        </div>
    </div>
""",
    unsafe_allow_html=True,
)

# Security notice
st.markdown(
    """
    <div style="
        text-align: center;
        color: var(--text-muted);
        font-size: 0.75rem;
        padding: 1rem;
    ">
        🔒 Your data is encrypted and secure. We never share your information.
    </div>
""",
    unsafe_allow_html=True,
)
