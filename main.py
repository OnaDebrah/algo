# """
# Advanced Algorithmic Trading Platform
# ===================================================================
# """
#
# import logging
#
# import streamlit as st
#
# from alerts.alert_manager import AlertManager
# from config import LAYOUT, PAGE_ICON, PAGE_TITLE
# from core.database import DatabaseManager
# from core.risk_manager import RiskManager
# from ui.backtest import render_backtest
# from ui.configuration import render_configuration
# from ui.dashboard import render_dashboard
# from ui.live import render_live_trading
# from ui.ml_builder import render_ml_builder
# from ui.portfolio_optimisation import render_portfolio_optimization
#
# logger = logging.getLogger(__name__)
#
#
# def initialize_session_state():
#     """Initialize Streamlit session state variables"""
#     if "db" not in st.session_state:
#         st.session_state.db = DatabaseManager()
#
#     if "alert_manager" not in st.session_state:
#         st.session_state.alert_manager = AlertManager()
#
#     if "ml_models" not in st.session_state:
#         st.session_state.ml_models = {}
#
#     if "risk_manager" not in st.session_state:
#         st.session_state.risk_manager = RiskManager()
#
#
# def configure_page():
#     """Configure Streamlit page settings"""
#     st.set_page_config(
#         page_title=PAGE_TITLE,
#         layout=LAYOUT,
#         page_icon=PAGE_ICON,
#         initial_sidebar_state="expanded",
#     )
#
#     # Custom CSS
#     st.markdown(
#         """
#         <style>
#         .main {background-color: #0e1117;}
#         .stMetric {
#             background-color: #1e2130;
#             padding: 15px;
#             border-radius: 10px;
#         }
#         .stTabs [data-baseweb="tab-list"] {
#             gap: 8px;
#         }
#         .stTabs [data-baseweb="tab"] {
#             padding: 10px 20px;
#         }
#         </style>
#     """,
#         unsafe_allow_html=True,
#     )
#
#
# def render_sidebar():
#     """Render sidebar with platform info"""
#     with st.sidebar:
#         st.title("🚀 Trading Platform")
#
#         st.markdown("---")
#
#         st.markdown("### 📊 Features")
#         st.markdown(
#             """
#         - ✅ Backtesting
#         - ✅ ML Strategies
#         - ✅ **Live Trading**
#         - ✅ **Portfolio Optimization**
#         - ✅ Risk Management
#         - ✅ Alerts
#         """
#         )
#
#         st.markdown("---")
#
#         st.markdown("### 🔗 Quick Links")
#         st.markdown("[📖 Documentation](#)")
#         st.markdown("[💬 Community](#)")
#         st.markdown("[🐛 Report Bug](#)")
#
#         st.markdown("---")
#
#         st.caption("v1.1.0 - With Live Trading & Portfolio Optimization")
#
#
# def main():
#     """Main application entry point"""
#     configure_page()
#     initialize_session_state()
#     render_sidebar()
#
#     # Header
#     st.title("📈 Advanced Algorithmic Trading Platform")
#     st.markdown(
#         "**Full-Featured Trading System: Backtesting • Live Trading • "
#         "Portfolio Optimization • ML Strategies**"
#     )
#
#     # Navigation tabs
#     tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
#         [
#             "📊 Dashboard",
#             "🔬 Backtest",
#             "🤖 ML Builder",
#             "⚡ Live Trading",
#             "📊 Portfolio Optimization",
#             "⚙️ Configuration",
#         ]
#     )
#
#     # Render each tab
#     with tab1:
#         render_dashboard(st.session_state.db)
#
#     with tab2:
#         render_backtest(
#             st.session_state.db,
#             st.session_state.risk_manager,
#             st.session_state.ml_models,
#             st.session_state.alert_manager,
#         )
#
#     with tab3:
#         render_ml_builder(st.session_state.ml_models)
#
#     with tab4:
#         render_live_trading(
#             st.session_state.db,
#             st.session_state.risk_manager,
#             st.session_state.ml_models,
#             st.session_state.alert_manager,
#         )
#
#     with tab5:
#         render_portfolio_optimization()
#
#     with tab6:
#         render_configuration(
#             st.session_state.db,
#             st.session_state.alert_manager,
#             st.session_state.risk_manager,
#         )
#
#
# if __name__ == "__main__":
#     main()

"""
Advanced Algorithmic Trading Platform
===================================================================
"""

import logging

import streamlit as st

from alerts.alert_manager import AlertManager
from config import LAYOUT, PAGE_ICON, PAGE_TITLE
from core.database import DatabaseManager
from core.risk_manager import RiskManager
from ui.analyst import render_analyst
from ui.backtest import render_backtest
from ui.configuration import render_configuration
from ui.dashboard import render_dashboard
from ui.live import render_live_trading
from ui.ml_builder import render_ml_builder
from ui.portfolio_optimisation import render_portfolio_optimization
from ui.strategy_advisor import render_strategy_advisor

logger = logging.getLogger(__name__)


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "db" not in st.session_state:
        st.session_state.db = DatabaseManager()

    if "alert_manager" not in st.session_state:
        st.session_state.alert_manager = AlertManager()

    if "ml_models" not in st.session_state:
        st.session_state.ml_models = {}

    if "risk_manager" not in st.session_state:
        st.session_state.risk_manager = RiskManager()


def configure_page():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title=PAGE_TITLE,
        layout=LAYOUT,
        page_icon=PAGE_ICON,
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown(
        """
        <style>
        .main {background-color: #0e1117;}
        .stMetric {
            background-color: #1e2130;
            padding: 15px;
            border-radius: 10px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
        }
        /* Options-specific styling */
        .options-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .greeks-container {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render sidebar with platform info"""
    with st.sidebar:
        st.title("🚀 Trading Platform")

        st.markdown("---")

        st.markdown("### 📊 Features")
        st.markdown(
            """
        - ✅ **Backtesting**
        - ✅ **ML Strategies**
        - ✅ **Live Trading**
        - ✅ **Portfolio Optimization**
        - ✅ **Options Strategies** 🆕
        - ✅ **Risk Management**
        - ✅ **Alerts**
        """
        )

        st.markdown("---")

        # Quick stats
        if "db" in st.session_state:
            try:
                db = st.session_state.db
                total_trades = len(db.get_all_trades())

                st.markdown("### 📈 Quick Stats")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Trades", total_trades)
                with col2:
                    # Get options positions count if table exists
                    try:
                        options_count = db.cursor.execute(
                            "SELECT COUNT(*) FROM options_positions WHERE status='OPEN'"
                        ).fetchone()[0]
                        st.metric("Options", options_count)
                    except Exception as e:
                        logger.error(e)
                        st.metric("Options", "N/A")
            except Exception as e:
                logger.warning(f"Could not load stats: {e}")

        st.markdown("---")

        st.markdown("### 🔗 Quick Links")
        st.markdown("[📖 Documentation](#)")
        st.markdown("[💬 Community](#)")
        st.markdown("[🐛 Report Bug](#)")

        st.markdown("---")

        st.caption("v1.2.0 - With Options Trading & Multi-Asset Support")


def render_options_tab():
    """Render the options strategy builder tab"""
    st.header("📈 Options Strategy Builder")

    st.markdown(
        """
    **Professional options trading tools with:**
    - 13+ preset strategies (Covered Calls, Iron Condors, Straddles, etc.)
    - Real-time Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
    - Probability of profit analysis
    - Interactive payoff diagrams
    - Historical backtesting
    - Options chain viewer
    """
    )

    # Create sub-tabs for options features
    options_tabs = st.tabs(
        [
            "🎯 Strategy Builder",
            "🔬 Backtest Options",
            "🔍 Options Chain",
            "📚 Learn Options",
        ]
    )

    with options_tabs[0]:
        try:
            from ui.options_builder_ui import render_options_strategy_builder

            render_options_strategy_builder()
        except ImportError as e:
            st.error(f"⚠️ Options module not yet installed {e}")
            st.info(
                """
            **To enable options trading:**
            1. Install required dependencies: `pip install scipy`
            2. Create the options module files in your project
            3. Restart the application

            See the integration guide for detailed instructions.
            """
            )

            # Show a preview/demo
            st.markdown("### 📊 Preview: Options Strategy Example")
            st.image(
                "https://via.placeholder.com/800x400/1e2130/00ff88?text=Options+Payoff+Diagram",
                use_container_width=True,
            )

            with st.expander("📋 Available Strategies (Coming Soon)"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**Income Strategies**")
                    st.markdown("- Covered Call")
                    st.markdown("- Cash-Secured Put")
                    st.markdown("- Collar")

                with col2:
                    st.markdown("**Directional Strategies**")
                    st.markdown("- Vertical Spreads")
                    st.markdown("- Diagonal Spreads")
                    st.markdown("- Calendar Spreads")

                with col3:
                    st.markdown("**Volatility Strategies**")
                    st.markdown("- Iron Condor")
                    st.markdown("- Straddle")
                    st.markdown("- Strangle")

    with options_tabs[1]:
        try:
            from ui.options_backtest import render_options_backtest

            render_options_backtest(st.session_state.db, st.session_state.risk_manager)
        except ImportError:
            st.info("Options backtesting will be available after module installation")

            # Show example results
            st.markdown("### 📊 Example Backtest Results")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Return", "24.5%", "↑ 12.3%")
            with col2:
                st.metric("Win Rate", "67.8%")
            with col3:
                st.metric("Avg Trade", "+$342")
            with col4:
                st.metric("Max Drawdown", "-8.2%")

    with options_tabs[2]:
        try:
            from ui.options_builder_ui import render_options_chain_viewer

            render_options_chain_viewer()
        except ImportError:
            st.info("Options chain viewer will be available after module installation")

            st.markdown("### 🔍 Options Chain Preview")
            st.markdown("View real-time options data with:")
            st.markdown("- Live bid/ask prices")
            st.markdown("- Volume and open interest")
            st.markdown("- Implied volatility")
            st.markdown("- Greeks for each strike")

    with options_tabs[3]:
        render_options_education()


def render_options_education():
    """Render options education section"""
    st.markdown("### 📚 Options Trading Guide")

    st.markdown(
        """
    Options give you the right (but not obligation) to buy or sell an asset at a specific price.
    They're powerful tools for income generation, hedging, and speculation.
    """
    )

    # Strategy cards
    st.markdown("### 🎯 Popular Strategies")

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("💰 Covered Call - Income Generation"):
            st.markdown(
                """
            **What it is:** Sell a call option against stock you own

            **When to use:**
            - You're neutral to slightly bullish
            - Want to generate income from holdings
            - OK with capping upside potential

            **Risk:** Limited (stock could go to zero)
            **Reward:** Limited (premium + stock gains to strike)

            **Example:** Own 100 shares of AAPL at $150, sell $155 call for $2
            - Max profit: $7/share ($5 stock gain + $2 premium)
            - Breakeven: $148 (stock cost - premium)
            """
            )

        with st.expander("🛡️ Protective Put - Portfolio Insurance"):
            st.markdown(
                """
            **What it is:** Buy a put option to protect stock holdings

            **When to use:**
            - You own stock but worried about downside
            - Before earnings or major events
            - Market volatility is high

            **Risk:** Limited (premium paid)
            **Reward:** Unlimited (stock gains minus premium)

            **Example:** Own 100 shares at $150, buy $145 put for $2
            - Protected below $145 (minus premium)
            - Cost: $2/share insurance premium
            """
            )

        with st.expander("📊 Iron Condor - Range Trading"):
            st.markdown(
                """
            **What it is:** Sell OTM put spread + call spread

            **When to use:**
            - Expect stock to stay in a range
            - High implied volatility
            - Want defined risk/reward

            **Risk:** Limited (width of widest spread - premium)
            **Reward:** Limited (net premium received)

            **Example:** Stock at $150
            - Sell $145/$140 put spread
            - Sell $155/$160 call spread
            - Collect $2 premium, risk $3
            """
            )

    with col2:
        with st.expander("🎯 Vertical Spread - Directional Trading"):
            st.markdown(
                """
            **What it is:** Buy and sell options at different strikes

            **When to use:**
            - Moderately bullish or bearish
            - Want defined risk
            - Lower capital requirement than stock

            **Risk:** Limited (net debit paid)
            **Reward:** Limited (spread width - debit)

            **Example:** Bullish on stock at $150
            - Buy $150 call for $5
            - Sell $155 call for $3
            - Net cost: $2, max gain: $3
            """
            )

        with st.expander("⚡ Straddle - Volatility Play"):
            st.markdown(
                """
            **What it is:** Buy call and put at same strike (ATM)

            **When to use:**
            - Big move expected (earnings, FDA approval)
            - Direction uncertain
            - Implied volatility is low

            **Risk:** Limited (total premium paid)
            **Reward:** Unlimited (in either direction)

            **Example:** Stock at $150, expecting big move
            - Buy $150 call for $5
            - Buy $150 put for $5
            - Need >$10 move to profit
            """
            )

        with st.expander("🔄 Calendar Spread - Time Decay"):
            st.markdown(
                """
            **What it is:** Sell near-term, buy longer-term option

            **When to use:**
            - Expect slow/sideways movement near-term
            - Profit from time decay difference
            - Want to own longer-term option cheaper

            **Risk:** Limited (net debit paid)
            **Reward:** Limited (difference in decay rates)

            **Example:** Stock at $150
            - Sell 30-day $150 call for $4
            - Buy 60-day $150 call for $6
            - Net cost: $2
            """
            )

    # Greeks explanation
    st.markdown("### 📐 Understanding Greeks")

    greeks_col1, greeks_col2, greeks_col3 = st.columns(3)

    with greeks_col1:
        st.markdown("**Delta (Δ)** - Price Sensitivity")
        st.markdown("- How much option price changes per $1 move in stock")
        st.markdown("- Range: 0 to 1 (calls), 0 to -1 (puts)")
        st.markdown("- Example: 0.5 delta = $0.50 move per $1 stock move")

    with greeks_col2:
        st.markdown("**Theta (Θ)** - Time Decay")
        st.markdown("- How much option loses per day")
        st.markdown("- Always negative for long options")
        st.markdown("- Example: -0.05 = loses $5/day per contract")

    with greeks_col3:
        st.markdown("**Vega (ν)** - Volatility Sensitivity")
        st.markdown("- How much option price changes per 1% IV change")
        st.markdown("- Higher for ATM options")
        st.markdown("- Example: 0.10 = $10 gain per 1% IV increase")

    # Risk warning
    st.warning(
        """
    ⚠️ **Important Risk Disclosure:**

    Options trading involves substantial risk and is not suitable for all investors.
    You can lose your entire investment. Key risks include:

    - **Time decay**: Options lose value as expiration approaches
    - **Leverage**: Small moves can result in large gains or losses
    - **Complexity**: Multi-leg strategies require active management
    - **Assignment risk**: Short options can be assigned early

    Only trade with capital you can afford to lose. Consider paper trading first.
    """
    )


def main():
    """Main application entry point"""
    configure_page()
    initialize_session_state()
    render_sidebar()

    # Header
    st.title("📈 Advanced Algorithmic Trading Platform")
    st.markdown(
        "**Full-Featured Trading System: Backtesting • Live Trading • "
        "Options • Portfolio Optimization • ML Strategies**"
    )

    # Navigation tabs (added Options tab)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
        [
            "📊 Dashboard",
            "🤖 AI Strategy Advisor",
            "🏦 AI Financial Analyst",
            "🔬 Backtest",
            "📈 Options",
            "🤖 ML Builder",
            "⚡ Live Trading",
            "📊 Portfolio Optimization",
            "⚙️ Configuration",
        ]
    )

    # Render each tab
    with tab1:
        render_dashboard(st.session_state.db)
    with tab2:
        render_strategy_advisor()
    with tab3:
        render_analyst()
    with tab4:
        render_backtest(
            st.session_state.db,
            st.session_state.risk_manager,
            st.session_state.ml_models,
            st.session_state.alert_manager,
        )

    with tab5:
        render_options_tab()

    with tab6:
        render_ml_builder(st.session_state.ml_models)

    with tab7:
        render_live_trading(
            st.session_state.db,
            st.session_state.risk_manager,
            st.session_state.ml_models,
            st.session_state.alert_manager,
        )

    with tab8:
        render_portfolio_optimization()

    with tab9:
        render_configuration(
            st.session_state.db,
            st.session_state.alert_manager,
            st.session_state.risk_manager,
        )


if __name__ == "__main__":
    main()
