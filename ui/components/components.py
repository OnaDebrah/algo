"""
Reusable UI Components for ORACULUM
Professional, production-ready components
"""

from typing import Dict, List

import streamlit as st

from ui.theme import OracleTheme

PRICING_CONFIG = [
    {
        "name": "FREE",
        "icon": "🆓",
        "price": "€0",
        "period": "forever",
        "features": ["Portfolio Dashboard", "Basic Backtesting", "10 Backtests/month", "Paper Trading"],
        "cta": "Start Free",
        "highlight": False,
    },
    {
        "name": "BASIC",
        "icon": "🥉",
        "price": "€0",
        "period": "per month",
        "features": ["Everything in FREE", "Multi-Asset Trading", "Advanced Analytics", "Market Data API"],
        "cta": "Start Trial",
        "highlight": False,
    },
    {
        "name": "PRO",
        "icon": "⭐",
        "price": "€0",
        "period": "per month",
        "features": ["Everything in BASIC", "ML Strategies", "Options Trading", "AI Strategy Advisor"],
        "cta": "Start Trial",
        "highlight": True,
    },
    {
        "name": "ENTERPRISE",
        "icon": "💎",
        "price": "Custom",
        "period": "contact sales",
        "features": ["Everything in PRO", "Live Trading", "Dedicated API", "Dedicated Support"],
        "cta": "Contact Sales",
        "highlight": False,
    },
]


@st.cache_data
def get_hero_html():
    return """
    <div class="hero-section animate-fade-in">
        <h1 style="font-size: 4rem; font-weight: 900; color: white; margin-bottom: 1.5rem; letter-spacing: -0.03em;">🏛️ ORACULUM</h1>
        <p style="font-size: 1.5rem; color: rgba(255,255,255,0.95); max-width: 800px; margin: 0 auto 2rem; line-height: 1.6;">
            Institutional-grade algorithmic trading platform.<br>
            Harness AI, quantitative analysis, and real-time execution.
        </p>
        <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
            <a href="#get-started" style="text-decoration: none;"><div class="stButton"><button kind="primary" style="padding: 1rem 2rem;">Get Started Free →</button></div></a>
        </div>
    </div>
    """


def render_stats_section():
    st.markdown('<div id="stats" style="margin: 4rem 0;"></div>', unsafe_allow_html=True)
    C = OracleTheme.COLORS

    st.markdown(
        f"""
        <div style="text-align: center; margin-bottom: 3rem;">
            <p style="color: {C['text_muted']}; font-size: 0.85rem; font-style: italic;">
                * Technical Specifications: Engineering targets for the Oraculum 2.0 Simulation Core.
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    stats = [
        ("50+", "ML Pipelines", "🧬"),
        ("10,000+", "Target User Base", "👥"),
        ("99.9%", "Backtest Fidelity", "⚖️"),
        ("< 50ms", "Target Latency", "⚡"),
    ]

    cols = st.columns(4)
    for col, (val, label, icon) in zip(cols, stats):
        with col:
            st.markdown(
                f"""
                <div class="stat-card animate-fade-in" style="background: {C['bg_card']}; border: 1px solid {C['border_subtle']}; border-radius: 16px; padding: 2rem 1rem; text-align: center;">
                    <div class="glow-icon" style="font-size: 2.5rem; margin-bottom: 1rem;">{icon}</div>
                    <div class="stat-value" style="font-size: 2.2rem; font-weight: 800; margin-bottom: 0.25rem;">{val}</div>
                    <div style="color: {C['primary']}; font-weight: 700; font-size: 0.75rem; text-transform: uppercase;">{label}</div>
                </div>
            """,
                unsafe_allow_html=True,
            )


def render_pricing_section():
    C = OracleTheme.COLORS
    st.markdown(f'<h2 style="text-align: center; color: {C["text_primary"]};">Plans for Every Trader</h2>', unsafe_allow_html=True)

    cols = st.columns(4)
    for col, tier in zip(cols, PRICING_CONFIG):
        with col:
            # Theme Variable Extraction to prevent Python 3.10 f-string errors
            border_style = f"2px solid {C['primary']}" if tier["highlight"] else f"1px solid {C['border_subtle']}"
            bg_color = C["bg_tertiary"] if tier["highlight"] else C["bg_card"]

            badge_html = (
                f'<div style="background: {C["gradient_primary"]}; color: white; padding: 0.5rem; border-radius: 12px; font-size: 0.75rem; font-weight: 700; margin-bottom: 1rem; text-align: center;">MOST POPULAR</div>'
                if tier["highlight"]
                else '<div style="height: 48px;"></div>'
            )

            feat_rows = "".join(
                [
                    f'<div style="display: flex; margin-bottom: 0.6rem;"><span style="color: {C["success"]}; margin-right: 10px;">✓</span><span style="color: {C["text_secondary"]}; font-size: 0.85rem;">{f}</span></div>'
                    for f in tier["features"]
                ]
            )

            st.markdown(
                f"""
                <div style="border: {border_style}; background: {bg_color}; padding: 2rem 1.5rem; border-radius: 20px; min-height: 680px; display: flex; flex-direction: column;">
                    {badge_html}
                    <div class="glow-icon" style="font-size: 2.5rem; text-align: center;">{tier['icon']}</div>
                    <h3 style="text-align: center; color: white;">{tier['name']}</h3>
                    <div style="text-align: center; margin: 1rem 0;">
                        <span style="font-size: 2.5rem; font-weight: 800; color: white;">{tier['price']}</span>
                        <p style="color: {C['text_muted']}; font-size: 0.8rem;">{tier['period']}</p>
                    </div>
                    <div style="flex-grow: 1;">{feat_rows}</div>
                </div>
                <div style="margin-top: -60px; padding: 0 1.5rem 2rem;">
            """,
                unsafe_allow_html=True,
            )

            if st.button(tier["cta"], key=f"tier_{tier['name']}", use_container_width=True, type="primary" if tier["highlight"] else "secondary"):
                st.switch_page("pages/_Auth.py")
            st.markdown("</div>", unsafe_allow_html=True)


def render_features_section():
    """Key features grid"""
    st.markdown(
        """
        <div id="features" style="margin: 5rem 0 3rem;">
            <h2 style="text-align: center; font-size: 2.5rem; margin-bottom: 1rem;">
                Built for Professional Traders
            </h2>
            <p style="text-align: center; color: var(--text-secondary); font-size: 1.125rem; max-width: 700px; margin: 0 auto 3rem;">
                Everything you need to research, build, test, and execute sophisticated trading strategies.
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Features grid
    col1, col2, col3 = st.columns(3)

    features = [
        {
            "icon": "🤖",
            "title": "AI Strategy Advisor",
            "desc": "GPT-4 powered analysis recommends optimal strategies based on market conditions, your portfolio, and risk profile.",
            "color": "#667eea",
        },
        {
            "icon": "📊",
            "title": "Advanced Analytics",
            "desc": "Institutional-grade portfolio analytics, risk metrics, and performance attribution with real-time updates.",
            "color": "#00c853",
        },
        {
            "icon": "🔬",
            "title": "Research Lab",
            "desc": "Backtest strategies with historical data, optimize parameters, and validate performance before going live.",
            "color": "#ff1744",
        },
        {
            "icon": "⚡",
            "title": "High-Speed Execution",
            "desc": "Sub-50ms latency with smart order routing, real-time market data, and professional-grade risk controls.",
            "color": "#ffb300",
        },
        {
            "icon": "🎯",
            "title": "Market Regime Detection",
            "desc": "Machine learning identifies market conditions (trending, mean-reverting, volatile) to adapt strategies automatically.",
            "color": "#9c27b0",
        },
        {
            "icon": "📈",
            "title": "Options Strategies",
            "desc": "13+ preset strategies, real-time Greeks, probability analysis, and interactive payoff diagrams.",
            "color": "#00bcd4",
        },
    ]

    for idx, feature in enumerate(features):
        col = [col1, col2, col3][idx % 3]
        with col:
            feature_id = f"feat-{idx}"

            st.markdown(
                f"""
                <style>
                    /* Dynamic glow for this specific feature */
                    .{feature_id}:hover .glow-icon {{
                        filter: drop-shadow(0 0 20px {feature['color']}) !important;
                        transition: filter 0.3s ease;
                    }}
                </style>

                <div class="feature-card {feature_id}" style="border-left: 4px solid {feature['color']}; margin-bottom: 1.5rem;">
                    <div class="glow-icon" style="font-size: 3rem; margin-bottom: 1rem;">
                        {feature['icon']}
                    </div>
                    <h3 style="font-size: 1.25rem; margin-bottom: 0.75rem;">{feature['title']}</h3>
                    <p style="color: var(--text-secondary); line-height: 1.6;">{feature['desc']}</p>
                </div>
            """,
                unsafe_allow_html=True,
            )


def render_social_proof():
    """Testimonials and trust badges"""
    st.markdown(
        """
        <div style="margin: 5rem 0; text-align: center;">
            <h2 style="font-size: 2rem; margin-bottom: 3rem;">Trusted by Traders Worldwide</h2>
        </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    testimonials = [
        {
            "quote": "ORACULUM transformed how I approach trading. The AI advisor is like having a hedge fund analyst on demand.",
            "author": "Sarah Chen",
            "role": "Quantitative Trader",
            "rating": "⭐⭐⭐⭐⭐",
        },
        {
            "quote": "Best backtesting platform I've used. The multi-asset support and ML integration are game-changers.",
            "author": "Michael Rodriguez",
            "role": "Portfolio Manager",
            "rating": "⭐⭐⭐⭐⭐",
        },
        {
            "quote": "Finally, a platform that combines institutional tools with modern UX. Worth every penny. Good for market research",
            "author": "David Kim",
            "role": "Algorithmic Trader",
            "rating": "⭐⭐⭐⭐⭐",
        },
    ]

    for col, testimonial in zip([col1, col2, col3], testimonials):
        with col:
            st.markdown(
                f"""
                <div class="card" style="height: 100%; padding: 1.5rem;">
                    <div style="font-size: 1.5rem; margin-bottom: 1rem;">{testimonial['rating']}</div>
                    <p style="font-style: italic; color: var(--text-secondary); margin-bottom: 1.5rem; line-height: 1.6;">
                        "{testimonial['quote']}"
                    </p>
                    <div style="border-top: 1px solid var(--border-subtle); padding-top: 1rem;">
                        <p style="font-weight: 700; margin-bottom: 0.25rem;">{testimonial['author']}</p>
                        <p style="font-size: 0.875rem; color: var(--text-muted);">{testimonial['role']}</p>
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )


def render_cta_section():
    """Final call to action"""
    st.markdown(
        """
        <div id="get-started" style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px;
            padding: 4rem 2rem;
            text-align: center;
            margin: 5rem 0;
            box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        ">
            <h2 style="color: white; font-size: 2.5rem; margin-bottom: 1rem;">
                Ready to Transform Your Trading?
            </h2>
            <p style="color: rgba(255,255,255,0.9); font-size: 1.25rem; max-width: 700px; margin: 0 auto 2rem;">
                Join thousands of traders using ORACULUM to make smarter, data-driven decisions.
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Start Free Trial", use_container_width=True, type="primary", key="cta_final"):
            st.switch_page("pages/_Auth.py")


def render_authenticated_home():
    """Home dashboard for logged-in users"""
    from core.context import get_app_context
    from ui.newsfeed import render_market_ticker

    ctx = get_app_context()
    user = st.session_state.user
    C = OracleTheme.COLORS

    # 1. Welcome Header
    OracleTheme.render_page_header(title=f"Welcome back, {user['username']}", subtitle="Your institutional command center is ready.", icon="👋")

    # 2. Market Ticker (Live Feed)
    @st.fragment(run_every="30s")
    def sync_ticker():
        render_market_ticker(style="scrolling", include_news=False, include_calendar=False)

    sync_ticker()

    st.markdown("---")

    # 3. Portfolio Overview Section
    st.markdown("## 📊 Portfolio Overview")

    # Data fetching with safe defaults
    data = ctx.db.get_header_metrics(portfolio_id=1)
    nav = data.get("nav", 0.0)
    exposure = data.get("exposure", 0.0)
    unrealized = data.get("unrealized_pnl", 0.0)
    prev_nav = data.get("prev_nav", nav)

    risk_util = ctx.risk_manager.get_current_risk_utilization(current_exposure=exposure, portfolio_value=nav)

    nav_change = ((nav - prev_nav) / prev_nav * 100) if prev_nav > 0 else 0.0

    # Layout Metrics
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.metric("Net Asset Value", f"€{nav:,.2f}", f"{nav_change:+.2f}%")
    with m_col2:
        exposure_pct = (exposure / nav * 100) if nav > 0 else 0
        st.metric("Market Exposure", f"{exposure_pct:.1f}%", f"€{exposure:,.0f}")
    with m_col3:
        unrealized_pct = (unrealized / nav * 100) if nav > 0 else 0
        st.metric("Unrealized P&L", f"€{unrealized:,.2f}", f"{unrealized_pct:+.2f}%", delta_color="normal" if unrealized >= 0 else "inverse")
    with m_col4:
        risk_status = "Nominal" if risk_util < 75 else "Warning" if risk_util < 90 else "CRITICAL"
        st.metric("Risk Utilization", f"{risk_util}%", risk_status)

    st.markdown("---")

    # 4. Workspace Navigation with Enhanced Cards
    st.markdown("## 🚀 Your Workspaces")

    # First row: Monitor, Analyze, Research
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, {C['success']}22 0%, {C['success']}11 100%);
                         padding: 24px; border-radius: 12px; border-left: 4px solid {C['success']};">
                <div class="glow-icon" style="font-size: 2.5rem; margin-bottom: 0.5rem;">📊</div>
                <h3>Monitor</h3>
                <p style="color: {C['text_secondary']};">Real-time portfolio tracking, live execution, and position management</p>
                <ul style="color: {C['text_muted']}; font-size: 0.9rem; line-height: 1.6;">
                    <li>Portfolio Dashboard</li>
                    <li>Live Trading Desk</li>
                    <li>Performance Metrics</li>
                </ul>
            </div>
        """,
            unsafe_allow_html=True,
        )

        if st.button("Go to Monitor →", key="nav_monitor", use_container_width=True):
            st.switch_page("pages/1_📊_Monitor.py")

    with col2:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, {C['primary']}22 0%, {C['primary']}11 100%);
                         padding: 24px; border-radius: 12px; border-left: 4px solid {C['primary']};">
                <div class="glow-icon" style="font-size: 2.5rem; margin-bottom: 0.5rem;">🎯</div>
                <h3>Analyze</h3>
                <p style="color: {C['text_secondary']};">AI-powered analysis, market regime detection, and strategy recommendations</p>
                <ul style="color: {C['text_muted']}; font-size: 0.9rem; line-height: 1.6;">
                    <li>AI Strategy Advisor</li>
                    <li>Market Regime Detector</li>
                    <li>Financial Analyst</li>
                </ul>
            </div>
        """,
            unsafe_allow_html=True,
        )

        if st.button("Go to Analyze →", key="nav_analyze", use_container_width=True):
            st.switch_page("pages/2_🎯_Analyze.py")

    with col3:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, {C['error']}22 0%, {C['error']}11 100%);
                         padding: 24px; border-radius: 12px; border-left: 4px solid {C['error']};">
                <div class="glow-icon" style="font-size: 2.5rem; margin-bottom: 0.5rem;">🔬</div>
                <h3>Research</h3>
                <p style="color: {C['text_secondary']};">Backtesting, options strategies, and portfolio optimization</p>
                <ul style="color: {C['text_muted']}; font-size: 0.9rem; line-height: 1.6;">
                    <li>Backtesting Lab</li>
                    <li>Options Desk</li>
                    <li>Portfolio Optimization</li>
                </ul>
            </div>
        """,
            unsafe_allow_html=True,
        )

        if st.button("Go to Research →", key="nav_research", use_container_width=True):
            st.switch_page("pages/3_🔬_Research.py")

    # Second row: Build and Settings
    col4, col5 = st.columns(2)

    with col4:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, {C['warning']}22 0%, {C['warning']}11 100%);
                         padding: 24px; border-radius: 12px; border-left: 4px solid {C['warning']};">
                <div class="glow-icon" style="font-size: 2.5rem; margin-bottom: 0.5rem;">🛠️</div>
                <h3>Build</h3>
                <p style="color: {C['text_secondary']};">Create custom strategies and ML models</p>
                <ul style="color: {C['text_muted']}; font-size: 0.9rem; line-height: 1.6;">
                    <li>ML Strategy Studio</li>
                    <li>Custom Strategy Builder</li>
                    <li>Prompt Engineering</li>
                </ul>
            </div>
        """,
            unsafe_allow_html=True,
        )

        if st.button("Go to Build →", key="nav_build", use_container_width=True):
            st.switch_page("pages/4_🛠️_Build.py")

    with col5:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #9c27b022 0%, #9c27b011 100%);
                         padding: 24px; border-radius: 12px; border-left: 4px solid #9c27b0;">
                <div class="glow-icon" style="font-size: 2.5rem; margin-bottom: 0.5rem;">⚙️</div>
                <h3>Settings</h3>
                <p style="color: {C['text_secondary']};">Configuration, risk management, and alerts</p>
                <ul style="color: {C['text_muted']}; font-size: 0.9rem; line-height: 1.6;">
                    <li>System Configuration</li>
                    <li>Risk Parameters</li>
                    <li>Alert Settings</li>
                </ul>
            </div>
        """,
            unsafe_allow_html=True,
        )

        if st.button("Go to Settings →", key="nav_settings", use_container_width=True):
            st.switch_page("pages/5_⚙️_Settings.py")


@st.cache_data
def render_landing_hero():
    """Hero section"""
    st.markdown(
        """
        <div class="hero-section animate-fade-in">
            <h1 style="
                font-size: 4rem;
                font-weight: 900;
                color: white;
                margin-bottom: 1.5rem;
                letter-spacing: -0.03em;
                text-shadow: 0 4px 12px rgba(0,0,0,0.2);
            ">
                🏛️ ORACULUM
            </h1>
            <p style="
                font-size: 1.5rem;
                color: rgba(255,255,255,0.95);
                max-width: 800px;
                margin: 0 auto 2rem;
                line-height: 1.6;
                font-weight: 500;
            ">
                Institutional-grade algorithmic trading platform.<br>
                Harness AI, quantitative analysis, and real-time execution.
            </p>
            <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
                <a href="#get-started" style="text-decoration: none;">
                    <div style="
                        background: white;
                        color: #667eea;
                        padding: 1rem 2.5rem;
                        border-radius: 12px;
                        font-weight: 700;
                        font-size: 1.125rem;
                        cursor: pointer;
                        transition: all 0.3s ease;
                        display: inline-block;
                        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
                    ">
                        Get Started Free →
                    </div>
                </a>
                <a href="#features" style="text-decoration: none;">
                    <div style="
                        background: rgba(255,255,255,0.1);
                        backdrop-filter: blur(10px);
                        color: white;
                        padding: 1rem 2.5rem;
                        border-radius: 12px;
                        font-weight: 700;
                        font-size: 1.125rem;
                        border: 2px solid rgba(255,255,255,0.3);
                        cursor: pointer;
                        transition: all 0.3s ease;
                        display: inline-block;
                    ">
                        View Demo
                    </div>
                </a>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )


def render_navbar():
    """Professional navigation bar for authenticated users"""
    user = st.session_state.get("user", {})

    st.markdown(
        f"""
        <div style="
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border-subtle);
            padding: 1rem 2rem;
            margin: -2rem -3rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        ">
            <div style="display: flex; align-items: center; gap: 2rem;">
                <div style="font-size: 1.5rem; font-weight: 800;">
                    🏛️ ORACULUM
                </div>
                <nav style="display: flex; gap: 1.5rem;">
                    <a href="/" style="
                        color: var(--text-secondary);
                        text-decoration: none;
                        font-weight: 600;
                        transition: color 0.2s;
                    ">Home</a>
                    <a href="/Monitor" style="
                        color: var(--text-secondary);
                        text-decoration: none;
                        font-weight: 600;
                        transition: color 0.2s;
                    ">Monitor</a>
                    <a href="/Research" style="
                        color: var(--text-secondary);
                        text-decoration: none;
                        font-weight: 600;
                        transition: color 0.2s;
                    ">Research</a>
                </nav>
            </div>
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="
                    padding: 0.5rem 1rem;
                    background: var(--bg-card);
                    border-radius: 20px;
                    font-size: 0.875rem;
                    font-weight: 600;
                ">
                    {get_tier_badge(user.get('tier', 'free'))}
                </div>
                <div style="
                    width: 40px;
                    height: 40px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 1.25rem;
                    cursor: pointer;
                ">
                    👤
                </div>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )


def render_footer():
    """Professional footer"""

    C = OracleTheme.COLORS

    text_muted = C["text_muted"]
    text_sec = C["text_secondary"]
    primary = C["primary"]
    border = C["border_subtle"]
    bg_sec = C["bg_secondary"]
    text_pri = C["text_primary"]

    st.markdown(
        f"""
        <style>
            .footer-link {{
                color: {text_sec} !important;
                text-decoration: none !important;
                transition: color 0.2s ease, transform 0.2s ease;
                display: inline-block;
            }}
            .footer-link:hover {{
                color: {primary} !important;
                transform: translateX(3px);
            }}
            .footer-header {{
                color: {text_pri};
                font-size: 0.85rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                margin-bottom: 1.25rem;
            }}
        </style>

        <div style="
            margin-top: 6rem;
            padding: 4rem 2rem 2rem;
            background: linear-gradient(to bottom, transparent, {bg_sec}44);
            border-top: 1px solid {border};
            border-radius: 40px 40px 0 0;
        ">
            <div style="
                display: grid;
                grid-template-columns: 2fr 1fr 1fr 1fr;
                gap: 3rem;
                margin-bottom: 4rem;
                max-width: 1200px;
                margin-left: auto;
                margin-right: auto;
            ">
                <div>
                    <h4 style="margin-bottom: 1rem; letter-spacing: -0.02em; ">ORACULUM</h4>
                    <p style="color: {text_muted}; font-size: 0.875rem; line-height: 1.6; max-width: 280px;">
                        Institutional-grade algorithmic trading platform powered by AI.
                    </p>
                </div>
                <div style="display: flex; flex-direction: column; align-items: flex-start;">
                    <h4 style="
                        margin: 0 0 1rem 0;
                        font-size: 0.875rem;
                        font-weight: 700;
                        text-transform: uppercase;
                        letter-spacing: 0.05em;
                        color: white;
                        text-align: left;
                    ">
                        Product
                    </h4>
                    <ul style="list-style: none; padding: 0; margin: 0; color: {text_sec}; font-size: 0.875rem; text-align: left;">
                        <li style="margin-bottom: 0.5rem;">
                            <a href="#" style="color: inherit; text-decoration: none;">Features</a>
                        </li>
                        <li style="margin-bottom: 0.5rem;">
                            <a href="#" style="color: inherit; text-decoration: none;">Pricing</a>
                        </li>
                        <li style="margin-bottom: 0.5rem;">
                            <a href="#" style="color: inherit; text-decoration: none;">API</a>
                        </li>
                        <li style="margin-bottom: 0.5rem;">
                            <a href="#" style="color: inherit; text-decoration: none;">Changelog</a>
                        </li>
                    </ul>
                </div>
                <div>
                    <h4 style="margin-bottom: 1rem; font-size: 0.875rem; font-weight: 700; text-transform: uppercase;">
                        Resources
                    </h4>
                    <ul style="list-style: none; padding: 0; color: {text_sec}; font-size: 0.875rem;">
                        <li style="margin-bottom: 0.5rem;"><a href="#" style="color: inherit; text-decoration: none;">Documentation</a></li>
                        <li style="margin-bottom: 0.5rem;"><a href="#" style="color: inherit; text-decoration: none;">Guides</a></li>
                        <li style="margin-bottom: 0.5rem;"><a href="#" style="color: inherit; text-decoration: none;">Blog</a></li>
                        <li style="margin-bottom: 0.5rem;"><a href="#" style="color: inherit; text-decoration: none;">Community</a></li>
                    </ul>
                </div>
                <div>
                    <h4 style="margin-bottom: 1rem; font-size: 0.875rem; font-weight: 700; text-transform: uppercase;">
                        Company
                    </h4>
                    <ul style="list-style: none; padding: 0; color: {text_sec}; font-size: 0.875rem;">
                        <li style="margin-bottom: 0.5rem;"><a href="#" style="color: inherit; text-decoration: none;">About</a></li>
                        <li style="margin-bottom: 0.5rem;"><a href="#" style="color: inherit; text-decoration: none;">Careers</a></li>
                        <li style="margin-bottom: 0.5rem;"><a href="#" style="color: inherit; text-decoration: none;">Contact</a></li>
                        <li style="margin-bottom: 0.5rem;"><a href="#" style="color: inherit; text-decoration: none;">Legal</a></li>
                    </ul>
                </div>
            </div>
            <div style="
                padding-top: 2rem;
                border-top: 1px solid {border};
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 1rem;
                color: {text_muted};
                font-size: 0.875rem;
            ">
                <div>
                    © 2026 ORACULUM. All rights reserved.
                </div>
                <div style="display: flex; gap: 1.5rem;">
                    <a href="#" style="color: inherit; text-decoration: none;">Privacy</a>
                    <a href="#" style="color: inherit; text-decoration: none;">Terms</a>
                    <a href="#" style="color: inherit; text-decoration: none;">Security</a>
                </div>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str, change: str = None, icon: str = None, color: str = "#667eea"):
    """Render a professional metric card"""
    change_html = (
        f'<div style="color: {"#00c853" if "+" in change else "#ff5252"}; font-size: 0.875rem; font-weight: 600; margin-top: 0.5rem;">{change}</div>'
        if change
        else ""
    )
    icon_html = f'<div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>' if icon else ""

    st.markdown(
        f"""
        <div class="card" style="border-left: 4px solid {color};">
            {icon_html}
            <div class="card-header">{title}</div>
            <div style="font-size: 2rem; font-weight: 800; color: var(--text-primary);">{value}</div>
            {change_html}
        </div>
    """,
        unsafe_allow_html=True,
    )


def render_status_badge(status: str, label: str = None):
    """Render a status badge"""
    colors = {
        "success": ("#00c853", "#1b3a2f"),
        "warning": ("#ffb300", "#3d3420"),
        "error": ("#ff5252", "#3d2020"),
        "info": ("#2196f3", "#1e2940"),
        "neutral": ("#9e9e9e", "#2a2a2a"),
    }

    color, bg = colors.get(status.lower(), colors["neutral"])
    display_label = label or status.title()

    st.markdown(
        f"""
        <span style="
            display: inline-block;
            padding: 0.375rem 0.875rem;
            background: {bg};
            color: {color};
            border: 1px solid {color};
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        ">
            {display_label}
        </span>
    """,
        unsafe_allow_html=True,
    )


def render_progress_bar(value: float, max_value: float = 100, label: str = None, color: str = "#667eea"):
    """Render a professional progress bar"""
    percentage = (value / max_value) * 100

    label_html = f'<div style="margin-bottom: 0.5rem; font-size: 0.875rem; font-weight: 600;">{label}</div>' if label else ""

    st.markdown(
        f"""
        <div>
            {label_html}
            <div style="
                width: 100%;
                height: 8px;
                background: var(--bg-secondary);
                border-radius: 10px;
                overflow: hidden;
            ">
                <div style="
                    width: {percentage}%;
                    height: 100%;
                    background: {color};
                    border-radius: 10px;
                    transition: width 0.3s ease;
                "></div>
            </div>
            <div style="
                margin-top: 0.5rem;
                font-size: 0.75rem;
                color: var(--text-muted);
                display: flex;
                justify-content: space-between;
            ">
                <span>{value}</span>
                <span>{percentage:.1f}%</span>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )


def render_alert(message: str, type: str = "info", dismissible: bool = False):
    """Render a professional alert"""
    colors = {
        "success": ("#00c853", "#1b3a2f", "✓"),
        "warning": ("#ffb300", "#3d3420", "⚠"),
        "error": ("#ff5252", "#3d2020", "✕"),
        "info": ("#2196f3", "#1e2940", "ℹ"),
    }

    color, bg, icon = colors.get(type, colors["info"])

    st.markdown(
        f"""
        <div style="
            background: {bg};
            border-left: 4px solid {color};
            padding: 1rem 1.25rem;
            border-radius: 8px;
            display: flex;
            align-items: center;
            gap: 1rem;
            margin: 1rem 0;
        ">
            <div style="
                font-size: 1.5rem;
                color: {color};
            ">{icon}</div>
            <div style="flex: 1; color: var(--text-primary);">{message}</div>
        </div>
    """,
        unsafe_allow_html=True,
    )


def render_data_table(data: List[Dict], columns: List[str] = None):
    """Render a professional data table"""
    if not data:
        st.info("No data available")
        return

    import pandas as pd

    df = pd.DataFrame(data)

    if columns:
        df = df[columns]

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            col: st.column_config.Column(
                width="medium",
            )
            for col in df.columns
        },
    )


def render_empty_state(title: str, message: str, icon: str = "📭", action_label: str = None, action_callback=None):
    """Render an empty state message"""
    if action_label and action_callback:
        if st.button(action_label, type="primary"):
            action_callback()

    st.markdown(
        f"""
        <div style="
            text-align: center;
            padding: 4rem 2rem;
            color: var(--text-muted);
        ">
            <div style="font-size: 4rem; margin-bottom: 1rem;">{icon}</div>
            <h3 style="margin-bottom: 0.5rem; color: var(--text-primary);">{title}</h3>
            <p style="max-width: 400px; margin: 0 auto;">{message}</p>
        </div>
    """,
        unsafe_allow_html=True,
    )


def render_loading_spinner(message: str = "Loading..."):
    """Render a loading state"""
    st.markdown(
        f"""
        <div style="
            text-align: center;
            padding: 3rem 2rem;
        ">
            <div style="
                display: inline-block;
                width: 40px;
                height: 40px;
                border: 4px solid var(--bg-secondary);
                border-top-color: var(--primary);
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-bottom: 1rem;
            "></div>
            <p style="color: var(--text-muted);">{message}</p>
        </div>
        <style>
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
        </style>
    """,
        unsafe_allow_html=True,
    )


def get_tier_badge(tier: str) -> str:
    """Get formatted tier badge"""
    tier_config = {"free": ("🆓", "FREE"), "basic": ("🥉", "BASIC"), "pro": ("⭐", "PRO"), "enterprise": ("💎", "ENTERPRISE")}

    icon, label = tier_config.get(tier.lower(), ("🎯", tier.upper()))
    return f"{icon} {label}"


def render_workspace_card(icon: str, title: str, description: str, page: str, color: str = "#667eea"):
    """Render a workspace navigation card"""
    st.markdown(
        f"""
        <div class="feature-card" style="border-left: 4px solid {color}; margin-bottom: 1.5rem;">
            <div style="font-size: 3rem; margin-bottom: 0.75rem;">{icon}</div>
            <h3 style="font-size: 1.25rem; margin-bottom: 0.5rem;">{title}</h3>
            <p style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 1rem;">{description}</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    if st.button(f"Open {title} →", key=f"nav_{title}", use_container_width=True):
        st.switch_page(f"pages/{page}")
