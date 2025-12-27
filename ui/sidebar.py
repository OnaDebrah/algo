"""
Sidebar navigation:
"""

import streamlit as st

from auth.streamlit_auth import render_user_menu


def render_page_sidebar():
    """Render consistent sidebar for all pages"""
    with st.sidebar:
        # st.markdown("# 🚀 Trading Platform")
        # st.markdown("*Institutional Edition*")

        # st.markdown("---")

        # Quick navigation
        # st.markdown("### 🧭 Quick Navigation")
        #
        # if st.button("🏠 Home", use_container_width=True):
        #     st.switch_page("main.py")
        #
        # if st.button("📊 Monitor", use_container_width=True):
        #     st.switch_page("pages/1_📊_Monitor.py")
        #
        # if st.button("🎯 Analyze", use_container_width=True):
        #     st.switch_page("pages/2_🎯_Analyze.py")
        #
        # if st.button("🔬 Research", use_container_width=True):
        #     st.switch_page("pages/3_🔬_Research.py")
        #
        # if st.button("🛠️ Build", use_container_width=True):
        #     st.switch_page("pages/4_🛠️_Build.py")
        #
        # if st.button("⚙️ Settings", use_container_width=True):
        #     st.switch_page("pages/5_⚙️_Settings.py")

        # st.markdown("---")

        # Context info (if available)
        # context = get_app_context()

        # System status
        st.markdown("### 🟢 Status")
        st.success("✓ Connected")
        st.info("✓ Data Live")

        # # Quick stats
        # st.markdown("### 📊 Quick Stats")
        # try:
        #     trades = len(context.db.get_trades())
        #     st.metric("Trades", trades)
        # except:
        #     st.metric("Trades", "N/A")
        #
        # # Market regime (if available)
        # if context.current_regime:
        #     st.markdown("---")
        #     regime = context.current_regime.get('regime', 'unknown')
        #     confidence = context.current_regime.get('confidence', 0)
        #
        #     st.markdown("### 🎯 Market Regime")
        #     st.info(f"{regime.replace('_', ' ').title()}\n{confidence:.1%} confidence")

        render_user_menu()
        st.markdown("---")
        st.caption("ORACULUM - v1.0.0 ")
