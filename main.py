# # # # # """
# # # # # Advanced Algorithmic Trading Platform
# # # # # ===================================================================
# # # # # """
# # # # #
# # # # # import logging
# # # # #
# # # # # import streamlit as st
# # # # #
# # # # # from alerts.alert_manager import AlertManager
# # # # # from config import LAYOUT, PAGE_ICON, PAGE_TITLE
# # # # # from core.database import DatabaseManager
# # # # # from core.risk_manager import RiskManager
# # # # # from ui.backtest import render_backtest
# # # # # from ui.configuration import render_configuration
# # # # # from ui.dashboard import render_dashboard
# # # # # from ui.live import render_live_trading
# # # # # from ui.ml_builder import render_ml_builder
# # # # # from ui.portfolio_optimisation import render_portfolio_optimization
# # # # #
# # # # # logger = logging.getLogger(__name__)
# # # # #
# # # # #
# # # # # def initialize_session_state():
# # # # #     """Initialize Streamlit session state variables"""
# # # # #     if "db" not in st.session_state:
# # # # #         st.session_state.db = DatabaseManager()
# # # # #
# # # # #     if "alert_manager" not in st.session_state:
# # # # #         st.session_state.alert_manager = AlertManager()
# # # # #
# # # # #     if "ml_models" not in st.session_state:
# # # # #         st.session_state.ml_models = {}
# # # # #
# # # # #     if "risk_manager" not in st.session_state:
# # # # #         st.session_state.risk_manager = RiskManager()
# # # # #
# # # # #
# # # # # def configure_page():
# # # # #     """Configure Streamlit page settings"""
# # # # #     st.set_page_config(
# # # # #         page_title=PAGE_TITLE,
# # # # #         layout=LAYOUT,
# # # # #         page_icon=PAGE_ICON,
# # # # #         initial_sidebar_state="expanded",
# # # # #     )
# # # # #
# # # # #     # Custom CSS
# # # # #     st.markdown(
# # # # #         """
# # # # #         <style>
# # # # #         .main {background-color: #0e1117;}
# # # # #         .stMetric {
# # # # #             background-color: #1e2130;
# # # # #             padding: 15px;
# # # # #             border-radius: 10px;
# # # # #         }
# # # # #         .stTabs [data-baseweb="tab-list"] {
# # # # #             gap: 8px;
# # # # #         }
# # # # #         .stTabs [data-baseweb="tab"] {
# # # # #             padding: 10px 20px;
# # # # #         }
# # # # #         </style>
# # # # #     """,
# # # # #         unsafe_allow_html=True,
# # # # #     )
# # # # #
# # # # #
# # # # # def render_sidebar():
# # # # #     """Render sidebar with platform info"""
# # # # #     with st.sidebar:
# # # # #         st.title("🚀 Trading Platform")
# # # # #
# # # # #         st.markdown("---")
# # # # #
# # # # #         st.markdown("### 📊 Features")
# # # # #         st.markdown(
# # # # #             """
# # # # #         - ✅ Backtesting
# # # # #         - ✅ ML Strategies
# # # # #         - ✅ **Live Trading**
# # # # #         - ✅ **Portfolio Optimization**
# # # # #         - ✅ Risk Management
# # # # #         - ✅ Alerts
# # # # #         """
# # # # #         )
# # # # #
# # # # #         st.markdown("---")
# # # # #
# # # # #         st.markdown("### 🔗 Quick Links")
# # # # #         st.markdown("[📖 Documentation](#)")
# # # # #         st.markdown("[💬 Community](#)")
# # # # #         st.markdown("[🐛 Report Bug](#)")
# # # # #
# # # # #         st.markdown("---")
# # # # #
# # # # #         st.caption("v1.1.0 - With Live Trading & Portfolio Optimization")
# # # # #
# # # # #
# # # # # def main():
# # # # #     """Main application entry point"""
# # # # #     configure_page()
# # # # #     initialize_session_state()
# # # # #     render_sidebar()
# # # # #
# # # # #     # Header
# # # # #     st.title("📈 Advanced Algorithmic Trading Platform")
# # # # #     st.markdown(
# # # # #         "**Full-Featured Trading System: Backtesting • Live Trading • "
# # # # #         "Portfolio Optimization • ML Strategies**"
# # # # #     )
# # # # #
# # # # #     # Navigation tabs
# # # # #     tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
# # # # #         [
# # # # #             "📊 Dashboard",
# # # # #             "🔬 Backtest",
# # # # #             "🤖 ML Builder",
# # # # #             "⚡ Live Trading",
# # # # #             "📊 Portfolio Optimization",
# # # # #             "⚙️ Configuration",
# # # # #         ]
# # # # #     )
# # # # #
# # # # #     # Render each tab
# # # # #     with tab1:
# # # # #         render_dashboard(st.session_state.db)
# # # # #
# # # # #     with tab2:
# # # # #         render_backtest(
# # # # #             st.session_state.db,
# # # # #             st.session_state.risk_manager,
# # # # #             st.session_state.ml_models,
# # # # #             st.session_state.alert_manager,
# # # # #         )
# # # # #
# # # # #     with tab3:
# # # # #         render_ml_builder(st.session_state.ml_models)
# # # # #
# # # # #     with tab4:
# # # # #         render_live_trading(
# # # # #             st.session_state.db,
# # # # #             st.session_state.risk_manager,
# # # # #             st.session_state.ml_models,
# # # # #             st.session_state.alert_manager,
# # # # #         )
# # # # #
# # # # #     with tab5:
# # # # #         render_portfolio_optimization()
# # # # #
# # # # #     with tab6:
# # # # #         render_configuration(
# # # # #             st.session_state.db,
# # # # #             st.session_state.alert_manager,
# # # # #             st.session_state.risk_manager,
# # # # #         )
# # # # #
# # # # #
# # # # # if __name__ == "__main__":
# # # # #     main()
# # # #
# # # # """
# # # # Advanced Algorithmic Trading Platform
# # # # ===================================================================
# # # # """
# # # #
# # # # import logging
# # # #
# # # # import streamlit as st
# # # #
# # # # from alerts.alert_manager import AlertManager
# # # # from config import LAYOUT, PAGE_ICON, PAGE_TITLE
# # # # from core.database import DatabaseManager
# # # # from core.risk_manager import RiskManager
# # # # from ui.analyst import render_analyst
# # # # from ui.backtest import render_backtest
# # # # from ui.configuration import render_configuration
# # # # from ui.custom_strategy_builder import render_custom_strategy_builder
# # # # from ui.dashboard import render_dashboard
# # # # from ui.live import render_live_trading
# # # # from ui.ml_builder import render_ml_builder
# # # # from ui.portfolio_optimisation import render_portfolio_optimization
# # # # from ui.strategy_advisor import render_strategy_advisor
# # # #
# # # # logger = logging.getLogger(__name__)
# # # #
# # # #
# # # # def initialize_session_state():
# # # #     """Initialize Streamlit session state variables"""
# # # #     if "db" not in st.session_state:
# # # #         st.session_state.db = DatabaseManager()
# # # #
# # # #     if "alert_manager" not in st.session_state:
# # # #         st.session_state.alert_manager = AlertManager()
# # # #
# # # #     if "ml_models" not in st.session_state:
# # # #         st.session_state.ml_models = {}
# # # #
# # # #     if "risk_manager" not in st.session_state:
# # # #         st.session_state.risk_manager = RiskManager()
# # # #
# # # #
# # # # def configure_page():
# # # #     """Configure Streamlit page settings"""
# # # #     st.set_page_config(
# # # #         page_title=PAGE_TITLE,
# # # #         layout=LAYOUT,
# # # #         page_icon=PAGE_ICON,
# # # #         initial_sidebar_state="expanded",
# # # #     )
# # # #
# # # #     # Custom CSS
# # # #     st.markdown(
# # # #         """
# # # #         <style>
# # # #         .main {background-color: #0e1117;}
# # # #         .stMetric {
# # # #             background-color: #1e2130;
# # # #             padding: 15px;
# # # #             border-radius: 10px;
# # # #         }
# # # #         .stTabs [data-baseweb="tab-list"] {
# # # #             gap: 8px;
# # # #         }
# # # #         .stTabs [data-baseweb="tab"] {
# # # #             padding: 10px 20px;
# # # #         }
# # # #         /* Options-specific styling */
# # # #         .options-card {
# # # #             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
# # # #             padding: 20px;
# # # #             border-radius: 10px;
# # # #             margin: 10px 0;
# # # #         }
# # # #         .greeks-container {
# # # #             display: flex;
# # # #             gap: 10px;
# # # #             flex-wrap: wrap;
# # # #         }
# # # #         </style>
# # # #     """,
# # # #         unsafe_allow_html=True,
# # # #     )
# # # #
# # # #
# # # # def render_sidebar():
# # # #     """Render sidebar with platform info"""
# # # #     with st.sidebar:
# # # #         st.title("🚀 Trading Platform")
# # # #
# # # #         st.markdown("---")
# # # #
# # # #         st.markdown("### 📊 Features")
# # # #         st.markdown(
# # # #             """
# # # #         - ✅ **Backtesting**
# # # #         - ✅ **ML Strategies**
# # # #         - ✅ **Live Trading**
# # # #         - ✅ **Portfolio Optimization**
# # # #         - ✅ **Options Strategies** 🆕
# # # #         - ✅ **Risk Management**
# # # #         - ✅ **Alerts**
# # # #         """
# # # #         )
# # # #
# # # #         st.markdown("---")
# # # #
# # # #         # Quick stats
# # # #         if "db" in st.session_state:
# # # #             try:
# # # #                 db = st.session_state.db
# # # #                 total_trades = len(db.get_all_trades())
# # # #
# # # #                 st.markdown("### 📈 Quick Stats")
# # # #                 col1, col2 = st.columns(2)
# # # #                 with col1:
# # # #                     st.metric("Total Trades", total_trades)
# # # #                 with col2:
# # # #                     # Get options positions count if table exists
# # # #                     try:
# # # #                         options_count = db.cursor.execute(
# # # #                             "SELECT COUNT(*) FROM options_positions WHERE status='OPEN'"
# # # #                         ).fetchone()[0]
# # # #                         st.metric("Options", options_count)
# # # #                     except Exception as e:
# # # #                         logger.error(e)
# # # #                         st.metric("Options", "N/A")
# # # #             except Exception as e:
# # # #                 logger.warning(f"Could not load stats: {e}")
# # # #
# # # #         st.markdown("---")
# # # #
# # # #         st.markdown("### 🔗 Quick Links")
# # # #         st.markdown("[📖 Documentation](#)")
# # # #         st.markdown("[💬 Community](#)")
# # # #         st.markdown("[🐛 Report Bug](#)")
# # # #
# # # #         st.markdown("---")
# # # #
# # # #         st.caption("v1.2.0 - With Options Trading & Multi-Asset Support")
# # # #
# # # #
# # # # def render_options_tab():
# # # #     """Render the options strategy builder tab"""
# # # #     st.header("📈 Options Strategy Builder")
# # # #
# # # #     st.markdown(
# # # #         """
# # # #     **Professional options trading tools with:**
# # # #     - 13+ preset strategies (Covered Calls, Iron Condors, Straddles, etc.)
# # # #     - Real-time Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
# # # #     - Probability of profit analysis
# # # #     - Interactive payoff diagrams
# # # #     - Historical backtesting
# # # #     - Options chain viewer
# # # #     """
# # # #     )
# # # #
# # # #     # Create sub-tabs for options features
# # # #     options_tabs = st.tabs(
# # # #         [
# # # #             "🎯 Strategy Builder",
# # # #             "🔬 Backtest Options",
# # # #             "🔍 Options Chain",
# # # #             "📚 Learn Options",
# # # #         ]
# # # #     )
# # # #
# # # #     with options_tabs[0]:
# # # #         try:
# # # #             from ui.options_builder_ui import render_options_strategy_builder
# # # #
# # # #             render_options_strategy_builder()
# # # #         except ImportError as e:
# # # #             st.error(f"⚠️ Options module not yet installed {e}")
# # # #             st.info(
# # # #                 """
# # # #             **To enable options trading:**
# # # #             1. Install required dependencies: `pip install scipy`
# # # #             2. Create the options module files in your project
# # # #             3. Restart the application
# # # #
# # # #             See the integration guide for detailed instructions.
# # # #             """
# # # #             )
# # # #
# # # #             # Show a preview/demo
# # # #             st.markdown("### 📊 Preview: Options Strategy Example")
# # # #             st.image(
# # # #                 "https://via.placeholder.com/800x400/1e2130/00ff88?text=Options+Payoff+Diagram",
# # # #                 use_container_width=True,
# # # #             )
# # # #
# # # #             with st.expander("📋 Available Strategies (Coming Soon)"):
# # # #                 col1, col2, col3 = st.columns(3)
# # # #
# # # #                 with col1:
# # # #                     st.markdown("**Income Strategies**")
# # # #                     st.markdown("- Covered Call")
# # # #                     st.markdown("- Cash-Secured Put")
# # # #                     st.markdown("- Collar")
# # # #
# # # #                 with col2:
# # # #                     st.markdown("**Directional Strategies**")
# # # #                     st.markdown("- Vertical Spreads")
# # # #                     st.markdown("- Diagonal Spreads")
# # # #                     st.markdown("- Calendar Spreads")
# # # #
# # # #                 with col3:
# # # #                     st.markdown("**Volatility Strategies**")
# # # #                     st.markdown("- Iron Condor")
# # # #                     st.markdown("- Straddle")
# # # #                     st.markdown("- Strangle")
# # # #
# # # #     with options_tabs[1]:
# # # #         try:
# # # #             from ui.options_backtest import render_options_backtest
# # # #
# # # #             render_options_backtest(st.session_state.db, st.session_state.risk_manager)
# # # #         except ImportError:
# # # #             st.info("Options backtesting will be available after module installation")
# # # #
# # # #             # Show example results
# # # #             st.markdown("### 📊 Example Backtest Results")
# # # #             col1, col2, col3, col4 = st.columns(4)
# # # #
# # # #             with col1:
# # # #                 st.metric("Total Return", "24.5%", "↑ 12.3%")
# # # #             with col2:
# # # #                 st.metric("Win Rate", "67.8%")
# # # #             with col3:
# # # #                 st.metric("Avg Trade", "+$342")
# # # #             with col4:
# # # #                 st.metric("Max Drawdown", "-8.2%")
# # # #
# # # #     with options_tabs[2]:
# # # #         try:
# # # #             from ui.options_builder_ui import render_options_chain_viewer
# # # #
# # # #             render_options_chain_viewer()
# # # #         except ImportError:
# # # #             st.info("Options chain viewer will be available after module installation")
# # # #
# # # #             st.markdown("### 🔍 Options Chain Preview")
# # # #             st.markdown("View real-time options data with:")
# # # #             st.markdown("- Live bid/ask prices")
# # # #             st.markdown("- Volume and open interest")
# # # #             st.markdown("- Implied volatility")
# # # #             st.markdown("- Greeks for each strike")
# # # #
# # # #     with options_tabs[3]:
# # # #         render_options_education()
# # # #
# # # #
# # # # def render_options_education():
# # # #     """Render options education section"""
# # # #     st.markdown("### 📚 Options Trading Guide")
# # # #
# # # #     st.markdown(
# # # #         """
# # # #     Options give you the right (but not obligation) to buy or sell an asset at a specific price.
# # # #     They're powerful tools for income generation, hedging, and speculation.
# # # #     """
# # # #     )
# # # #
# # # #     # Strategy cards
# # # #     st.markdown("### 🎯 Popular Strategies")
# # # #
# # # #     col1, col2 = st.columns(2)
# # # #
# # # #     with col1:
# # # #         with st.expander("💰 Covered Call - Income Generation"):
# # # #             st.markdown(
# # # #                 """
# # # #             **What it is:** Sell a call option against stock you own
# # # #
# # # #             **When to use:**
# # # #             - You're neutral to slightly bullish
# # # #             - Want to generate income from holdings
# # # #             - OK with capping upside potential
# # # #
# # # #             **Risk:** Limited (stock could go to zero)
# # # #             **Reward:** Limited (premium + stock gains to strike)
# # # #
# # # #             **Example:** Own 100 shares of AAPL at $150, sell $155 call for $2
# # # #             - Max profit: $7/share ($5 stock gain + $2 premium)
# # # #             - Breakeven: $148 (stock cost - premium)
# # # #             """
# # # #             )
# # # #
# # # #         with st.expander("🛡️ Protective Put - Portfolio Insurance"):
# # # #             st.markdown(
# # # #                 """
# # # #             **What it is:** Buy a put option to protect stock holdings
# # # #
# # # #             **When to use:**
# # # #             - You own stock but worried about downside
# # # #             - Before earnings or major events
# # # #             - Market volatility is high
# # # #
# # # #             **Risk:** Limited (premium paid)
# # # #             **Reward:** Unlimited (stock gains minus premium)
# # # #
# # # #             **Example:** Own 100 shares at $150, buy $145 put for $2
# # # #             - Protected below $145 (minus premium)
# # # #             - Cost: $2/share insurance premium
# # # #             """
# # # #             )
# # # #
# # # #         with st.expander("📊 Iron Condor - Range Trading"):
# # # #             st.markdown(
# # # #                 """
# # # #             **What it is:** Sell OTM put spread + call spread
# # # #
# # # #             **When to use:**
# # # #             - Expect stock to stay in a range
# # # #             - High implied volatility
# # # #             - Want defined risk/reward
# # # #
# # # #             **Risk:** Limited (width of widest spread - premium)
# # # #             **Reward:** Limited (net premium received)
# # # #
# # # #             **Example:** Stock at $150
# # # #             - Sell $145/$140 put spread
# # # #             - Sell $155/$160 call spread
# # # #             - Collect $2 premium, risk $3
# # # #             """
# # # #             )
# # # #
# # # #     with col2:
# # # #         with st.expander("🎯 Vertical Spread - Directional Trading"):
# # # #             st.markdown(
# # # #                 """
# # # #             **What it is:** Buy and sell options at different strikes
# # # #
# # # #             **When to use:**
# # # #             - Moderately bullish or bearish
# # # #             - Want defined risk
# # # #             - Lower capital requirement than stock
# # # #
# # # #             **Risk:** Limited (net debit paid)
# # # #             **Reward:** Limited (spread width - debit)
# # # #
# # # #             **Example:** Bullish on stock at $150
# # # #             - Buy $150 call for $5
# # # #             - Sell $155 call for $3
# # # #             - Net cost: $2, max gain: $3
# # # #             """
# # # #             )
# # # #
# # # #         with st.expander("⚡ Straddle - Volatility Play"):
# # # #             st.markdown(
# # # #                 """
# # # #             **What it is:** Buy call and put at same strike (ATM)
# # # #
# # # #             **When to use:**
# # # #             - Big move expected (earnings, FDA approval)
# # # #             - Direction uncertain
# # # #             - Implied volatility is low
# # # #
# # # #             **Risk:** Limited (total premium paid)
# # # #             **Reward:** Unlimited (in either direction)
# # # #
# # # #             **Example:** Stock at $150, expecting big move
# # # #             - Buy $150 call for $5
# # # #             - Buy $150 put for $5
# # # #             - Need >$10 move to profit
# # # #             """
# # # #             )
# # # #
# # # #         with st.expander("🔄 Calendar Spread - Time Decay"):
# # # #             st.markdown(
# # # #                 """
# # # #             **What it is:** Sell near-term, buy longer-term option
# # # #
# # # #             **When to use:**
# # # #             - Expect slow/sideways movement near-term
# # # #             - Profit from time decay difference
# # # #             - Want to own longer-term option cheaper
# # # #
# # # #             **Risk:** Limited (net debit paid)
# # # #             **Reward:** Limited (difference in decay rates)
# # # #
# # # #             **Example:** Stock at $150
# # # #             - Sell 30-day $150 call for $4
# # # #             - Buy 60-day $150 call for $6
# # # #             - Net cost: $2
# # # #             """
# # # #             )
# # # #
# # # #     # Greeks explanation
# # # #     st.markdown("### 📐 Understanding Greeks")
# # # #
# # # #     greeks_col1, greeks_col2, greeks_col3 = st.columns(3)
# # # #
# # # #     with greeks_col1:
# # # #         st.markdown("**Delta (Δ)** - Price Sensitivity")
# # # #         st.markdown("- How much option price changes per $1 move in stock")
# # # #         st.markdown("- Range: 0 to 1 (calls), 0 to -1 (puts)")
# # # #         st.markdown("- Example: 0.5 delta = $0.50 move per $1 stock move")
# # # #
# # # #     with greeks_col2:
# # # #         st.markdown("**Theta (Θ)** - Time Decay")
# # # #         st.markdown("- How much option loses per day")
# # # #         st.markdown("- Always negative for long options")
# # # #         st.markdown("- Example: -0.05 = loses $5/day per contract")
# # # #
# # # #     with greeks_col3:
# # # #         st.markdown("**Vega (ν)** - Volatility Sensitivity")
# # # #         st.markdown("- How much option price changes per 1% IV change")
# # # #         st.markdown("- Higher for ATM options")
# # # #         st.markdown("- Example: 0.10 = $10 gain per 1% IV increase")
# # # #
# # # #     # Risk warning
# # # #     st.warning(
# # # #         """
# # # #     ⚠️ **Important Risk Disclosure:**
# # # #
# # # #     Options trading involves substantial risk and is not suitable for all investors.
# # # #     You can lose your entire investment. Key risks include:
# # # #
# # # #     - **Time decay**: Options lose value as expiration approaches
# # # #     - **Leverage**: Small moves can result in large gains or losses
# # # #     - **Complexity**: Multi-leg strategies require active management
# # # #     - **Assignment risk**: Short options can be assigned early
# # # #
# # # #     Only trade with capital you can afford to lose. Consider paper trading first.
# # # #     """
# # # #     )
# # # #
# # # #
# # # # def main():
# # # #     """Main application entry point"""
# # # #     configure_page()
# # # #     initialize_session_state()
# # # #     render_sidebar()
# # # #
# # # #     # Header
# # # #     st.title("📈 Advanced Algorithmic Trading Platform")
# # # #     st.markdown(
# # # #         "**Full-Featured Trading System: Backtesting • Live Trading • "
# # # #         "Options • Portfolio Optimization • ML Strategies**"
# # # #     )
# # # #
# # # #     # Navigation tabs (added Options tab)
# # # #     tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(
# # # #         [
# # # #             "📊 Dashboard",
# # # #             "🤖 AI Strategy Advisor",
# # # #             "🏦 AI Financial Analyst",
# # # #             "🔬 Backtest",
# # # #             "📈 Options",
# # # #             "🤖 ML Builder",
# # # #             "⚡ Live Trading",
# # # #             "📊 Portfolio Optimization",
# # # #             "Custom Strategy Builder",
# # # #             "⚙️ Configuration",
# # # #         ]
# # # #     )
# # # #
# # # #     # Render each tab
# # # #     with tab1:
# # # #         render_dashboard(st.session_state.db)
# # # #     with tab2:
# # # #         render_strategy_advisor()
# # # #     with tab3:
# # # #         render_analyst()
# # # #     with tab4:
# # # #         render_backtest(
# # # #             st.session_state.db,
# # # #             st.session_state.risk_manager,
# # # #             st.session_state.ml_models,
# # # #             st.session_state.alert_manager,
# # # #         )
# # # #
# # # #     with tab5:
# # # #         render_options_tab()
# # # #
# # # #     with tab6:
# # # #         render_ml_builder(st.session_state.ml_models)
# # # #
# # # #     with tab7:
# # # #         render_live_trading(
# # # #             st.session_state.db,
# # # #             st.session_state.risk_manager,
# # # #             st.session_state.ml_models,
# # # #             st.session_state.alert_manager,
# # # #         )
# # # #
# # # #     with tab8:
# # # #         render_portfolio_optimization()
# # # #
# # # #     with tab9:
# # # #         render_custom_strategy_builder()
# # # #
# # # #     with tab10:
# # # #         render_configuration(
# # # #             st.session_state.db,
# # # #             st.session_state.alert_manager,
# # # #             st.session_state.risk_manager,
# # # #         )
# # # #
# # # #
# # # # if __name__ == "__main__":
# # # #     main()
# # #
# # # # """
# # # # Advanced Algorithmic Trading Platform
# # # # =====================================
# # # # Professional Multi-Asset Trading & Research System
# # # # """
# # # #
# # # # import logging
# # # # from dataclasses import dataclass
# # # #
# # # # import streamlit as st
# # # #
# # # # from alerts.alert_manager import AlertManager
# # # # from config import LAYOUT, PAGE_ICON, PAGE_TITLE
# # # # from core.database import DatabaseManager
# # # # from core.risk_manager import RiskManager
# # # #
# # # # from ui.dashboard import render_dashboard
# # # # from ui.backtest import render_backtest
# # # # from ui.live import render_live_trading
# # # # from ui.ml_builder import render_ml_builder
# # # # from ui.portfolio_optimisation import render_portfolio_optimization
# # # # from ui.configuration import render_configuration
# # # # from ui.strategy_advisor import render_strategy_advisor
# # # # from ui.analyst import render_analyst
# # # # from ui.custom_strategy_builder import render_custom_strategy_builder
# # # #
# # # # logger = logging.getLogger(__name__)
# # # #
# # # #
# # # # # -------------------------------------------------------------------
# # # # # App Context (clean dependency passing)
# # # # # -------------------------------------------------------------------
# # # #
# # # # @dataclass
# # # # class AppContext:
# # # #     db: DatabaseManager
# # # #     risk: RiskManager
# # # #     alerts: AlertManager
# # # #     ml_models: dict
# # # #
# # # #
# # # # # -------------------------------------------------------------------
# # # # # Initialization
# # # # # -------------------------------------------------------------------
# # # #
# # # # def initialize_context() -> AppContext:
# # # #     if "context" not in st.session_state:
# # # #         st.session_state.context = AppContext(
# # # #             db=DatabaseManager(),
# # # #             risk=RiskManager(),
# # # #             alerts=AlertManager(),
# # # #             ml_models={},
# # # #         )
# # # #     return st.session_state.context
# # # #
# # # #
# # # # def configure_page():
# # # #     st.set_page_config(
# # # #         page_title=PAGE_TITLE,
# # # #         layout=LAYOUT,
# # # #         page_icon=PAGE_ICON,
# # # #         initial_sidebar_state="expanded",
# # # #     )
# # # #
# # # #     st.markdown(
# # # #         """
# # # #         <style>
# # # #         :root {
# # # #             --bg: #0E1117;
# # # #             --card: #1E2130;
# # # #             --primary: #4C78FF;
# # # #             --success: #00C853;
# # # #             --warning: #FFB300;
# # # #             --danger: #FF5252;
# # # #             --text: #E6EAF2;
# # # #             --muted: #9AA4B2;
# # # #         }
# # # #
# # # #         .main {
# # # #             background-color: var(--bg);
# # # #             color: var(--text);
# # # #             padding: 24px;
# # # #         }
# # # #
# # # #         .card {
# # # #             background-color: var(--card);
# # # #             padding: 18px;
# # # #             border-radius: 14px;
# # # #             margin-bottom: 16px;
# # # #         }
# # # #
# # # #         .card-title {
# # # #             font-size: 0.85rem;
# # # #             color: var(--muted);
# # # #             text-transform: uppercase;
# # # #             letter-spacing: 0.08em;
# # # #             margin-bottom: 6px;
# # # #         }
# # # #
# # # #         .metric-value {
# # # #             font-size: 1.6rem;
# # # #             font-weight: 600;
# # # #         }
# # # #
# # # #         .metric-positive { color: var(--success); }
# # # #         .metric-negative { color: var(--danger); }
# # # #         .metric-warning { color: var(--warning); }
# # # #         </style>
# # # #         """,
# # # #         unsafe_allow_html=True,
# # # #     )
# # # #
# # # #     # st.markdown(
# # # #     #     """
# # # #     #     <style>
# # # #     #     .main { background-color: #0e1117; }
# # # #     #     .card {
# # # #     #         background-color: #1e2130;
# # # #     #         padding: 18px;
# # # #     #         border-radius: 12px;
# # # #     #     }
# # # #     #     .status-ok { color: #00ff88; }
# # # #     #     .status-warn { color: #ffcc00; }
# # # #     #     .status-bad { color: #ff4b4b; }
# # # #     #     </style>
# # # #     #     """,
# # # #     #     unsafe_allow_html=True,
# # # #     # )
# # # #
# # # #
# # # # # -------------------------------------------------------------------
# # # # # Sidebar
# # # # # -------------------------------------------------------------------
# # # #
# # # # def render_sidebar(context: AppContext):
# # # #     with st.sidebar:
# # # #         st.title("🚀 Trading Platform")
# # # #
# # # #         st.markdown("### 🧭 Mode")
# # # #         mode = st.radio(
# # # #             "",
# # # #             ["Monitor", "Decide", "Research", "Build", "Manage"],
# # # #             label_visibility="collapsed",
# # # #         )
# # # #
# # # #         st.markdown("---")
# # # #
# # # #         st.markdown("### 🟢 System Status")
# # # #         st.success("Broker: Connected")
# # # #         st.info("Market Data: Live")
# # # #         st.warning("Risk Utilization: 72%")
# # # #
# # # #         st.markdown("---")
# # # #
# # # #         try:
# # # #             total_trades = len(context.db.get_all_trades())
# # # #             st.metric("Total Trades", total_trades)
# # # #         except Exception:
# # # #             st.metric("Total Trades", "N/A")
# # # #
# # # #         st.markdown("---")
# # # #         st.caption("v1.3.0 – Institutional Edition")
# # # #
# # # #     return mode
# # # #
# # # #
# # # # # -------------------------------------------------------------------
# # # # # Command Center Header
# # # # # -------------------------------------------------------------------
# # # #
# # # # def render_command_center():
# # # #     st.markdown("## 📊 Command Center")
# # # #
# # # #     col1, col2, col3, col4 = st.columns(4)
# # # #
# # # #     col1.metric("Equity", "$128,450", "+2.3%")
# # # #     col2.metric("Day P&L", "$1,240", "↑")
# # # #     col3.metric("Open Positions", 11)
# # # #     col4.metric("Risk Level", "Moderate")
# # # #
# # # #
# # # # # -------------------------------------------------------------------
# # # # # Main Renderer
# # # # # -------------------------------------------------------------------
# # # #
# # # # def main():
# # # #     configure_page()
# # # #     context = initialize_context()
# # # #     mode = render_sidebar(context)
# # # #
# # # #     st.title("📈 Advanced Algorithmic Trading Platform")
# # # #     st.caption(
# # # #         "Backtesting • AI-Driven Decisions • Options • Live Execution"
# # # #     )
# # # #
# # # #     render_command_center()
# # # #     st.markdown("---")
# # # #
# # # #     # ---------------------------------------------------------------
# # # #     # Mode-based navigation
# # # #     # ---------------------------------------------------------------
# # # #
# # # #     if mode == "Monitor":
# # # #         tab1, tab2 = st.tabs(["📊 Dashboard", "⚡ Execution Desk"])
# # # #         with tab1:
# # # #             render_dashboard(context.db)
# # # #         with tab2:
# # # #             render_live_trading(
# # # #                 context.db,
# # # #                 context.risk,
# # # #                 context.ml_models,
# # # #                 context.alerts,
# # # #             )
# # # #
# # # #     elif mode == "Decide":
# # # #         tab1, tab2 = st.tabs(["🤖 AI Strategy Advisor", "🏦 AI Analyst"])
# # # #         with tab1:
# # # #             render_strategy_advisor()
# # # #         with tab2:
# # # #             render_analyst()
# # # #
# # # #     elif mode == "Research":
# # # #         tab1, tab2, tab3 = st.tabs(
# # # #             ["🔬 Backtesting Lab", "📈 Options Desk", "📊 Portfolio"]
# # # #         )
# # # #         with tab1:
# # # #             render_backtest(
# # # #                 context.db,
# # # #                 context.risk,
# # # #                 context.ml_models,
# # # #                 context.alerts,
# # # #             )
# # # #         with tab2:
# # # #             from ui.options_builder_ui import render_options_strategy_builder
# # # #             render_options_strategy_builder()
# # # #         with tab3:
# # # #             render_portfolio_optimization()
# # # #
# # # #     elif mode == "Build":
# # # #         tab1, tab2 = st.tabs(
# # # #             ["🤖 ML Strategy Studio", "🧪 Strategy Lab"]
# # # #         )
# # # #         with tab1:
# # # #             render_ml_builder(context.ml_models)
# # # #         with tab2:
# # # #             render_custom_strategy_builder()
# # # #
# # # #     elif mode == "Manage":
# # # #         render_configuration(
# # # #             context.db,
# # # #             context.alerts,
# # # #             context.risk,
# # # #         )
# # # #
# # # #
# # # # # -------------------------------------------------------------------
# # # #
# # # # if __name__ == "__main__":
# # # #     main()
# # #
# # # # Advanced Algorithmic Trading Platform
# # # # ===================================================================
# # #
# # # import logging
# # #
# # # import streamlit as st
# # #
# # # from alerts.alert_manager import AlertManager
# # # from config import LAYOUT, PAGE_ICON, PAGE_TITLE
# # # from core.database import DatabaseManager
# # # from core.risk_manager import RiskManager
# # # from ui.analyst import render_analyst
# # # from ui.backtest import render_backtest
# # # from ui.configuration import render_configuration
# # # from ui.custom_strategy_builder import render_custom_strategy_builder
# # # from ui.dashboard import render_dashboard
# # # from ui.live import render_live_trading
# # # from ui.ml_builder import render_ml_builder
# # # from ui.portfolio_optimisation import render_portfolio_optimization
# # # from ui.strategy_advisor import render_strategy_advisor
# # # from ui.regime_detector import render_regime_detector
# # #
# # # logger = logging.getLogger(__name__)
# # #
# # #
# # # def initialize_session_state():
# # #     """Initialize Streamlit session state variables"""
# # #     if "db" not in st.session_state:
# # #         st.session_state.db = DatabaseManager()
# # #
# # #     if "alert_manager" not in st.session_state:
# # #         st.session_state.alert_manager = AlertManager()
# # #
# # #     if "ml_models" not in st.session_state:
# # #         st.session_state.ml_models = {}
# # #
# # #     if "risk_manager" not in st.session_state:
# # #         st.session_state.risk_manager = RiskManager()
# # #
# # #     if "regime_detector" not in st.session_state:
# # #         from analytics.market_regime_detector import MarketRegimeDetector
# # #         st.session_state.regime_detector = MarketRegimeDetector()
# # #
# # #     if "current_regime" not in st.session_state:
# # #         st.session_state.current_regime = None
# # #
# # #
# # # def configure_page():
# # #     """Configure Streamlit page settings"""
# # #     st.set_page_config(
# # #         page_title=PAGE_TITLE,
# # #         layout=LAYOUT,
# # #         page_icon=PAGE_ICON,
# # #         initial_sidebar_state="expanded",
# # #     )
# # #
# # #     # Custom CSS
# # #     st.markdown(
# # #         """
# # #         <style>
# # #         .main {background-color: #0e1117;}
# # #         .stMetric {
# # #             background-color: #1e2130;
# # #             padding: 15px;
# # #             border-radius: 10px;
# # #         }
# # #         .stTabs [data-baseweb="tab-list"] {
# # #             gap: 8px;
# # #         }
# # #         .stTabs [data-baseweb="tab"] {
# # #             padding: 10px 20px;
# # #         }
# # #         /* Regime-specific styling */
# # #         .regime-card {
# # #             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
# # #             padding: 20px;
# # #             border-radius: 10px;
# # #             margin: 10px 0;
# # #         }
# # #         .regime-bull {
# # #             border-left: 4px solid #00c853;
# # #         }
# # #         .regime-bear {
# # #             border-left: 4px solid #ff1744;
# # #         }
# # #         .regime-neutral {
# # #             border-left: 4px solid #2196f3;
# # #         }
# # #         </style>
# # #     """,
# # #         unsafe_allow_html=True,
# # #     )
# # #
# # #
# # # def render_sidebar():
# # #     """Render sidebar with platform info"""
# # #     with st.sidebar:
# # #         st.title("🚀 Trading Platform")
# # #
# # #         st.markdown("---")
# # #
# # #         st.markdown("### 📊 Features")
# # #         st.markdown(
# # #             """
# # #         - ✅ **Backtesting**
# # #         - ✅ **ML Strategies**
# # #         - ✅ **Live Trading**
# # #         - ✅ **Portfolio Optimization**
# # #         - ✅ **Options Strategies**
# # #         - ✅ **Market Regime Detection** 🆕
# # #         - ✅ **Risk Management**
# # #         - ✅ **Alerts**
# # #         """
# # #         )
# # #
# # #         st.markdown("---")
# # #
# # #         # Quick stats including regime
# # #         if "db" in st.session_state:
# # #             try:
# # #                 db = st.session_state.db
# # #                 total_trades = len(db.get_all_trades())
# # #
# # #                 st.markdown("### 📈 Quick Stats")
# # #                 col1, col2 = st.columns(2)
# # #                 with col1:
# # #                     st.metric("Total Trades", total_trades)
# # #                 with col2:
# # #                     # Get options positions count if table exists
# # #                     try:
# # #                         options_count = db.cursor.execute(
# # #                             "SELECT COUNT(*) FROM options_positions WHERE status='OPEN'"
# # #                         ).fetchone()[0]
# # #                         st.metric("Options", options_count)
# # #                     except Exception as e:
# # #                         logger.error(e)
# # #                         st.metric("Options", "N/A")
# # #
# # #                 # Display current regime
# # #                 if st.session_state.current_regime:
# # #                     regime = st.session_state.current_regime.get('regime', 'unknown')
# # #                     confidence = st.session_state.current_regime.get('confidence', 0)
# # #
# # #                     st.markdown("---")
# # #                     st.markdown("### 🎯 Market Regime")
# # #
# # #                     # Regime emoji mapping
# # #                     regime_emoji = {
# # #                         'trending_bull': '🟢',
# # #                         'trending_bear': '🔴',
# # #                         'mean_reverting': '🔵',
# # #                         'high_volatility': '🟡',
# # #                         'low_volatility': '⚪',
# # #                         'crisis': '🔴🔴',
# # #                         'recovery': '🟢🟡',
# # #                         'transition': '🟣'
# # #                     }
# # #
# # #                     emoji = regime_emoji.get(regime, '⚫')
# # #                     st.info(f"{emoji} **{regime.replace('_', ' ').title()}**\nConfidence: {confidence:.1%}")
# # #
# # #             except Exception as e:
# # #                 logger.warning(f"Could not load stats: {e}")
# # #
# # #         st.markdown("---")
# # #
# # #         st.markdown("### 🔗 Quick Links")
# # #         st.markdown("[📖 Documentation](#)")
# # #         st.markdown("[💬 Community](#)")
# # #         st.markdown("[🐛 Report Bug](#)")
# # #
# # #         st.markdown("---")
# # #
# # #         st.caption("v2.0.0 - With Market Regime Detection")
# # #
# # #
# # # def render_options_tab():
# # #     """Render the options strategy builder tab"""
# # #     st.header("📈 Options Strategy Builder")
# # #
# # #     st.markdown(
# # #         """
# # #     **Professional options trading tools with:**
# # #     - 13+ preset strategies (Covered Calls, Iron Condors, Straddles, etc.)
# # #     - Real-time Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
# # #     - Probability of profit analysis
# # #     - Interactive payoff diagrams
# # #     - Historical backtesting
# # #     - Options chain viewer
# # #     """
# # #     )
# # #
# # #     # Create sub-tabs for options features
# # #     options_tabs = st.tabs(
# # #         [
# # #             "🎯 Strategy Builder",
# # #             "🔬 Backtest Options",
# # #             "🔍 Options Chain",
# # #             "📚 Learn Options",
# # #         ]
# # #     )
# # #
# # #     with options_tabs[0]:
# # #         try:
# # #             from ui.options_builder_ui import render_options_strategy_builder
# # #
# # #             render_options_strategy_builder()
# # #         except ImportError as e:
# # #             st.error(f"⚠️ Options module not yet installed {e}")
# # #             st.info(
# # #                 """
# # #             **To enable options trading:**
# # #             1. Install required dependencies: `pip install scipy`
# # #             2. Create the options module files in your project
# # #             3. Restart the application
# # #
# # #             See the integration guide for detailed instructions.
# # #             """
# # #             )
# # #
# # #     with options_tabs[1]:
# # #         try:
# # #             from ui.options_backtest import render_options_backtest
# # #
# # #             render_options_backtest(st.session_state.db, st.session_state.risk_manager)
# # #         except ImportError:
# # #             st.info("Options backtesting will be available after module installation")
# # #
# # #     with options_tabs[2]:
# # #         try:
# # #             from ui.options_builder_ui import render_options_chain_viewer
# # #
# # #             render_options_chain_viewer()
# # #         except ImportError:
# # #             st.info("Options chain viewer will be available after module installation")
# # #
# # #     with options_tabs[3]:
# # #         render_options_education()
# # #
# # #
# # # def render_options_education():
# # #     """Render options education section"""
# # #     st.markdown("### 📚 Options Trading Guide")
# # #
# # #     st.markdown(
# # #         """
# # #     Options give you the right (but not obligation) to buy or sell an asset at a specific price.
# # #     They're powerful tools for income generation, hedging, and speculation.
# # #     """
# # #     )
# # #
# # #     st.warning(
# # #         """
# # #     ⚠️ **Important Risk Disclosure:**
# # #
# # #     Options trading involves substantial risk and is not suitable for all investors.
# # #     You can lose your entire investment. Only trade with capital you can afford to lose.
# # #     """
# # #     )
# # #
# # #
# # # def main():
# # #     """Main application entry point"""
# # #     configure_page()
# # #     initialize_session_state()
# # #     render_sidebar()
# # #
# # #     # Header
# # #     st.title("📈 Advanced Algorithmic Trading Platform")
# # #     st.markdown(
# # #         "**Full-Featured Trading System: Backtesting • Live Trading • "
# # #         "Options • Portfolio Optimization • ML Strategies • Market Regime Detection**"
# # #     )
# # #
# # #     # Navigation tabs (added Market Regime tab)
# # #     tabs = st.tabs(
# # #         [
# # #             "📊 Dashboard",
# # #             "🎯 Market Regime",  # NEW TAB
# # #             "🤖 AI Strategy Advisor",
# # #             "🏦 AI Financial Analyst",
# # #             "🔬 Backtest",
# # #             "📈 Options",
# # #             "🤖 ML Builder",
# # #             "⚡ Live Trading",
# # #             "📊 Portfolio Optimization",
# # #             "🔧 Custom Strategy",
# # #             "⚙️ Configuration",
# # #         ]
# # #     )
# # #
# # #     # Render each tab
# # #     with tabs[0]:  # Dashboard
# # #         render_dashboard(st.session_state.db)
# # #
# # #     with tabs[1]:  # Market Regime - NEW
# # #         render_regime_detector(st.session_state.db)
# # #
# # #     with tabs[2]:  # AI Strategy Advisor
# # #         render_strategy_advisor()
# # #
# # #     with tabs[3]:  # AI Financial Analyst
# # #         render_analyst()
# # #
# # #     with tabs[4]:  # Backtest
# # #         render_backtest(
# # #             st.session_state.db,
# # #             st.session_state.risk_manager,
# # #             st.session_state.ml_models,
# # #             st.session_state.alert_manager,
# # #         )
# # #
# # #     with tabs[5]:  # Options
# # #         render_options_tab()
# # #
# # #     with tabs[6]:  # ML Builder
# # #         render_ml_builder(st.session_state.ml_models)
# # #
# # #     with tabs[7]:  # Live Trading
# # #         render_live_trading(
# # #             st.session_state.db,
# # #             st.session_state.risk_manager,
# # #             st.session_state.ml_models,
# # #             st.session_state.alert_manager,
# # #         )
# # #
# # #     with tabs[8]:  # Portfolio Optimization
# # #         render_portfolio_optimization()
# # #
# # #     with tabs[9]:  # Custom Strategy Builder
# # #         render_custom_strategy_builder()
# # #
# # #     with tabs[10]:  # Configuration
# # #         render_configuration(
# # #             st.session_state.db,
# # #             st.session_state.alert_manager,
# # #             st.session_state.risk_manager,
# # #         )
# # #
# # #
# # # if __name__ == "__main__":
# # #     main()
# #
# # import logging
# # from dataclasses import dataclass
# #
# # import streamlit as st
# #
# # from alerts.alert_manager import AlertManager
# # from config import LAYOUT, PAGE_ICON, PAGE_TITLE
# # # Core System Imports
# # from core.database import DatabaseManager
# # from core.risk_manager import RiskManager
# # from ui.analyst import render_analyst
# # from ui.backtest import render_backtest
# # from ui.configuration import render_configuration
# # from ui.custom_strategy_builder import render_custom_strategy_builder
# # # UI Module Imports
# # from ui.dashboard import render_dashboard
# # from ui.live import render_live_trading
# # from ui.ml_builder import render_ml_builder
# # from ui.newsfeed import render_market_ticker
# # from ui.portfolio_optimisation import render_portfolio_optimization
# # from ui.regime_detector import render_regime_detector
# # from ui.strategy_advisor import render_strategy_advisor
# #
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)
# #
# #
# # # -------------------------------------------------------------------
# # # App Context: Centralized Dependency Injection
# # # -------------------------------------------------------------------
# # @dataclass
# # class AppContext:
# #     db: DatabaseManager
# #     risk: RiskManager
# #     alerts: AlertManager
# #     ml_models: dict
# #
# #
# # # -------------------------------------------------------------------
# # # Theme & Styling Engine
# # # -------------------------------------------------------------------
# # def apply_custom_theme(mode: str):
# #     """Injects CSS variables based on the selected theme mode"""
# #     if mode == "Dark":
# #         colors = {
# #             "bg": "#0E1117",
# #             "card": "#1E2130",
# #             "text": "#E6EAF2",
# #             "muted": "#9AA4B2",
# #             "border": "#31354A",
# #             "accent": "#4C78FF",
# #         }
# #     else:  # Light Mode
# #         colors = {
# #             "bg": "#F8F9FB",
# #             "card": "#FFFFFF",
# #             "text": "#1A1C23",
# #             "muted": "#64748B",
# #             "border": "#E2E8F0",
# #             "accent": "#2563EB",
# #         }
# #
# #     st.markdown(
# #         f"""
# #         <style>
# #         /* Base Application Styling */
# #         .stApp {{ background-color: {colors['bg']}; color: {colors['text']}; }}
# #
# #         /* Sidebar Styling */
# #         section[data-testid="stSidebar"] {{
# #             background-color: {colors['bg']};
# #             border-right: 1px solid {colors['border']};
# #         }}
# #
# #         /* Global Typography */
# #         h1, h2, h3, h4, p, span, label, .stMarkdown {{ color: {colors['text']} !important; }}
# #
# #         /* Professional Metric Cards */
# #         div[data-testid="stMetric"] {{
# #             background-color: {colors['card']};
# #             border: 1px solid {colors['border']};
# #             border-radius: 12px;
# #             padding: 20px;
# #             box-shadow: 0 2px 4px rgba(0,0,0,0.05);
# #         }}
# #
# #         /* Workspace Headers */
# #         .workspace-header {{
# #             font-size: 0.85rem;
# #             font-weight: 700;
# #             color: {colors['muted']};
# #             text-transform: uppercase;
# #             letter-spacing: 0.1em;
# #             border-bottom: 2px solid {colors['accent']};
# #             padding-bottom: 5px;
# #             margin-bottom: 25px;
# #         }}
# #
# #         /* Custom Tabs Styling */
# #         .stTabs [data-baseweb="tab"] {{
# #             color: {colors['muted']};
# #         }}
# #         .stTabs [aria-selected="true"] {{
# #             color: {colors['accent']} !important;
# #             font-weight: bold;
# #         }}
# #         </style>
# #         """,
# #         unsafe_allow_html=True,
# #     )
# #
# #
# # def initialize_system():
# #     """Configure page and initialize session-stored context"""
# #     st.set_page_config(
# #         page_title=PAGE_TITLE,
# #         layout=LAYOUT,
# #         page_icon=PAGE_ICON,
# #         initial_sidebar_state="expanded",
# #     )
# #
# #     if "context" not in st.session_state:
# #         st.session_state.context = AppContext(
# #             db=DatabaseManager(),
# #             risk=RiskManager(),
# #             alerts=AlertManager(),
# #             ml_models={},
# #         )
# #
# #     if "theme_mode" not in st.session_state:
# #         st.session_state.theme_mode = "Dark"
# #
# #
# # def render_sidebar(ctx: AppContext):
# #     """Renders the institutional-grade navigation sidebar"""
# #     with st.sidebar:
# #         st.markdown("# 🏛️ ORACULUM")
# #         st.caption("Advanced Algorithmic Suite v2.0")
# #
# #         st.markdown("### 🧭 Workspace")
# #         mode = st.radio(
# #             "Navigation",
# #             ["**Monitor**", "**Analyze**", "**Research**", "**Build**", "**Settings**"],
# #             label_visibility="collapsed",
# #         )
# #
# #         st.markdown("---")
# #         st.markdown("### 🎨 Appearance")
# #         theme_choice = st.selectbox(
# #             "Theme Mode",
# #             ["Dark", "Light"],
# #             index=0 if st.session_state.theme_mode == "Dark" else 1,
# #         )
# #
# #         # Trigger Rerun on Theme Change
# #         if theme_choice != st.session_state.theme_mode:
# #             st.session_state.theme_mode = theme_choice
# #             st.rerun()
# #
# #         st.markdown("---")
# #         st.markdown("### 🟢 System Connectivity")
# #         st.success("Exchange API: Active")
# #         st.info("DB Engine: SQLite (Local)")
# #
# #         return mode
# #
# #
# # def render_header_metrics(ctx: AppContext):
# #     # 1. Fetch live data
# #     data = ctx.db.get_header_metrics(portfolio_id=1)
# #
# #     # 2. Calculate Utilization via RiskManager
# #     risk_util = ctx.risk.get_current_risk_utilization(current_exposure=data["exposure"], portfolio_value=data["nav"])
# #
# #     # 3. Calculate NAV Change
# #     nav_change = ((data["nav"] - data["prev_nav"]) / data["prev_nav"] * 100) if data["prev_nav"] > 0 else 0.0
# #
# #     # 4. Render
# #     col1, col2, col3, col4 = st.columns(4)
# #
# #     with col1:
# #         st.metric("Net Asset Value", f"${data['nav']:,.2f}", f"{nav_change:+.2f}%")
# #
# #     with col2:
# #         exposure_pct = (data["exposure"] / data["nav"] * 100) if data["nav"] > 0 else 0
# #         st.metric("Market Exposure", f"{exposure_pct:.1f}%", f"${data['exposure']:,.0f}")
# #
# #     with col3:
# #         # Green/Red based on profit
# #         st.metric("Unrealized P&L", f"${data['unrealized_pnl']:,.2f}", "Live")
# #
# #     with col4:
# #         # Critical warning logic
# #         color = "normal" if risk_util < 70 else "inverse"
# #         st.metric(
# #             "Risk Utilization",
# #             f"{risk_util}%",
# #             "CRITICAL" if risk_util > 85 else "Nominal",
# #             delta_color=color,
# #         )
# #
# # def main():
# #     initialize_system()
# #     ctx = st.session_state.context
# #
# #     # Apply selected theme
# #     apply_custom_theme(st.session_state.theme_mode)
# #
# #     render_market_ticker(
# #         style="scrolling",  # or "grid"
# #         include_news=False,
# #         include_calendar=False
# #     )
# #
# #     st.markdown("<br>", unsafe_allow_html=True)
# #
# #     mode = render_sidebar(ctx)
# #
# #     if mode == "**Monitor**":
# #         st.markdown(
# #             '<div class="workspace-header">Monitor & Execution Desk</div>',
# #             unsafe_allow_html=True,
# #         )
# #
# #         # This fragment isolates the header for high-frequency updates
# #         @st.fragment(run_every="10s")
# #         def sync_header():
# #             render_header_metrics(ctx)
# #
# #         sync_header()
# #
# #         t1, t2, t3 = st.tabs(["📊 Portfolio Dashboard", "🎯 Market Regime", "⚡ Live Execution"])
# #         with t1:
# #             render_dashboard(ctx.db)
# #         with t2:
# #             render_regime_detector(ctx.db)
# #         with t3:
# #             render_live_trading(ctx.db, ctx.risk, ctx.ml_models, ctx.alerts)
# #
# #     elif mode == "**Analyze**":
# #         st.markdown(
# #             '<div class="workspace-header">AI Analyst & Advisory</div>',
# #             unsafe_allow_html=True,
# #         )
# #         t1, t2 = st.tabs(["🤖 Strategy Advisor", "🏦 Fundamental Analyst"])
# #         with t1:
# #             render_strategy_advisor()
# #         with t2:
# #             render_analyst()
# #
# #     elif mode == "**Research**":
# #         st.markdown(
# #             '<div class="workspace-header">Quantitative Research Lab</div>',
# #             unsafe_allow_html=True,
# #         )
# #         t1, t2, t3 = st.tabs(["🔬 Backtesting Lab", "📈 Options Desk", "📊 Portfolio Optimization"])
# #         with t1:
# #             render_backtest(ctx.db, ctx.risk, ctx.ml_models, ctx.alerts)
# #         with t2:
# #             try:
# #                 from ui.options_builder_ui import render_options_strategy_builder
# #
# #                 render_options_strategy_builder()
# #             except ImportError:
# #                 st.info("Options Module Loading...")
# #         with t3:
# #             render_portfolio_optimization()
# #
# #     elif mode == "**Build**":
# #         st.markdown(
# #             '<div class="workspace-header">Engineering Studio</div>',
# #             unsafe_allow_html=True,
# #         )
# #         t1, t2 = st.tabs(["🤖 ML Model Builder", "🧪 Logic Strategy Lab"])
# #         with t1:
# #             render_ml_builder(ctx.ml_models)
# #         with t2:
# #             render_custom_strategy_builder()
# #
# #     elif mode == "**Settings**":
# #         st.markdown(
# #             '<div class="workspace-header">System Configuration</div>',
# #             unsafe_allow_html=True,
# #         )
# #         render_configuration(ctx.db, ctx.alerts, ctx.risk)
# #
# #
# # if __name__ == "__main__":
# #     main()
#
# """
# Main entry point.
# """
#
# import streamlit as st
# from core.context import get_app_context, configure_page, AppContext
# from ui.newsfeed import render_market_ticker
#
#
# def main():
#     """Main landing page with navigation"""
#     configure_page("Home", "🚀")
#
#     # Get shared context
#     ctx = get_app_context()
#
#     # Header
#     st.markdown("""
#         <div class="page-header">
#             <h1>🏛️ ORACULUM</h1>
#             <p style="color: var(--text-secondary); margin-top: 8px;">
#                 AI-driven decisions and real-time execution
#             </p>
#         </div>
#     """, unsafe_allow_html=True)
#
#     # Market ticker
#     render_market_ticker(style="scrolling", include_news=False, include_calendar=False)
#
#     st.markdown("---")
#
#     # Navigation cards
#     st.markdown("## 🧭 Navigate to Your Workspace")
#
#     col1, col2, col3 = st.columns(3)
#
#     with col1:
#         st.markdown("""
#             <div style="background: linear-gradient(135deg, #00c85322 0%, #00c85311 100%);
#                         padding: 24px; border-radius: 12px; border-left: 4px solid #00c853;">
#                 <h3>📊 Monitor</h3>
#                 <p>Real-time Portfolio Tracking, Live Execution, and Position Management</p>
#                 <ul style="color: var(--text-secondary); font-size: 0.9rem;">
#                     <li>Portfolio Dashboard</li>
#                     <li>Live Trading Desk</li>
#                     <li>Performance Metrics</li>
#                 </ul>
#             </div>
#         """, unsafe_allow_html=True)
#
#         if st.button("Go to Monitor →", key="nav_monitor", use_container_width=True):
#             st.switch_page("pages/1_📊_Monitor.py")
#
#     with col2:
#         st.markdown("""
#             <div style="background: linear-gradient(135deg, #4c78ff22 0%, #4c78ff11 100%);
#                         padding: 24px; border-radius: 12px; border-left: 4px solid #4c78ff;">
#                 <h3>🎯 Analyze</h3>
#                 <p>AI-Powered Analysis, Market Regime Detection, and Strategy Recommendations</p>
#                 <ul style="color: var(--text-secondary); font-size: 0.9rem;">
#                     <li>AI Strategy Advisor</li>
#                     <li>Market Regime Detector</li>
#                     <li>Financial Analyst</li>
#                 </ul>
#             </div>
#         """, unsafe_allow_html=True)
#
#         if st.button("Go to Analyze →", key="nav_analyze", use_container_width=True):
#             st.switch_page("pages/2_🎯_Analyze.py")
#
#     with col3:
#         st.markdown("""
#             <div style="background: linear-gradient(135deg, #ff174422 0%, #ff174411 100%);
#                         padding: 24px; border-radius: 12px; border-left: 4px solid #ff1744;">
#                 <h3>🔬 Research</h3>
#                 <p>Backtesting, Options Strategies, and Portfolio Optimization</p>
#                 <ul style="color: var(--text-secondary); font-size: 0.9rem;">
#                     <li>Backtesting Lab</li>
#                     <li>Options Desk</li>
#                     <li>Portfolio Optimization</li>
#                 </ul>
#             </div>
#         """, unsafe_allow_html=True)
#
#         if st.button("Go to Research →", key="nav_research", use_container_width=True):
#             st.switch_page("pages/3_🔬_Research.py")
#
#     col4, col5 = st.columns(2)
#
#     with col4:
#         st.markdown("""
#             <div style="background: linear-gradient(135deg, #ffb30022 0%, #ffb30011 100%);
#                         padding: 24px; border-radius: 12px; border-left: 4px solid #ffb300;">
#                 <h3>🛠️ Build</h3>
#                 <p>Create Custom Strategies and ML models</p>
#                 <ul style="color: var(--text-secondary); font-size: 0.9rem;">
#                     <li>ML Strategy Studio</li>
#                     <li>Custom Strategy Builder</li>
#                     <li>Prompting</li>
#                 </ul>
#             </div>
#         """, unsafe_allow_html=True)
#
#         if st.button("Go to Build →", key="nav_build", use_container_width=True):
#             st.switch_page("pages/4_🛠️_Build.py")
#
#     with col5:
#         st.markdown("""
#             <div style="background: linear-gradient(135deg, #9c27b022 0%, #9c27b011 100%);
#                         padding: 24px; border-radius: 12px; border-left: 4px solid #9c27b0;">
#                 <h3>⚙️ Settings</h3>
#                 <p>Configuration, Risk Management, and Alerts</p>
#                 <ul style="color: var(--text-secondary); font-size: 0.9rem;">
#                     <li>System Configuration</li>
#                     <li>Risk Parameters</li>
#                     <li>Alert Settings</li>
#                 </ul>
#             </div>
#         """, unsafe_allow_html=True)
#
#         if st.button("Go to Settings →", key="nav_settings", use_container_width=True):
#             st.switch_page("pages/5_⚙️_Settings.py")
#
#     st.markdown("---")
#
#     # Quick stats overview
#     st.markdown("## 📈 Portfolio Overview")
#
#     def render_header_metrics(ctx: AppContext):
#         # 1. Fetch live data
#         data = ctx.db.get_header_metrics(portfolio_id=1)
#
#         # 2. Calculate Utilization via RiskManager
#         risk_util = ctx.risk_manager.get_current_risk_utilization(current_exposure=data["exposure"],
#                                                                   portfolio_value=data["nav"])
#
#         # 3. Calculate NAV Change
#         nav_change = ((data["nav"] - data["prev_nav"]) / data["prev_nav"] * 100) if data["prev_nav"] > 0 else 0.0
#
#         # 4. Render
#         col1, col2, col3, col4 = st.columns(4)
#
#         with col1:
#             st.metric("Net Asset Value", f"${data['nav']:,.2f}", f"{nav_change:+.2f}%")
#
#         with col2:
#             exposure_pct = (data["exposure"] / data["nav"] * 100) if data["nav"] > 0 else 0
#             st.metric("Market Exposure", f"{exposure_pct:.1f}%", f"${data['exposure']:,.0f}")
#
#         with col3:
#             # Green/Red based on profit
#             st.metric("Unrealized P&L", f"${data['unrealized_pnl']:,.2f}", "Live")
#
#         with col4:
#             # Critical warning logic
#             color = "normal" if risk_util < 70 else "inverse"
#             st.metric(
#                 "Risk Utilization",
#                 f"{risk_util}%",
#                 "CRITICAL" if risk_util > 85 else "Nominal",
#                 delta_color=color,
#             )
#
#     # This fragment isolates the header for high-frequency updates
#     @st.fragment(run_every="10s")
#     def sync_header():
#         render_header_metrics(ctx)
#
#     sync_header()
#
#     # Footer
#     st.markdown("---")
#     st.caption("Version 1.0.0 – Institutional Edition | © 2026")
#
#
# if __name__ == "__main__":
#     main()

"""
Main entry point with authentication.
"""

import streamlit as st

from auth.auth_manager import Permission
from auth.streamlit_auth import check_permission, init_auth_state, render_login_page, render_user_menu
from core.context import AppContext, configure_page, get_app_context
from ui.newsfeed import render_market_ticker


def main():
    """Main landing page with navigation"""

    init_auth_state()

    if not st.session_state.get("authenticated"):
        # Show login page if not authenticated
        configure_page("Login", "🔐")
        render_login_page()
        st.stop()

    # User is authenticated - proceed with normal app
    configure_page("Home", "🚀")

    # Get shared context
    ctx = get_app_context()

    # Render user menu in sidebar
    render_user_menu()

    # Header
    st.markdown(
        """
        <div class="page-header">
            <h1>🏛️ ORACULUM</h1>
            <p style="color: var(--text-secondary); margin-top: 8px;">
                AI-driven decisions and real-time execution
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Market ticker
    render_market_ticker(style="scrolling", include_news=False, include_calendar=False)

    st.markdown("---")

    # Navigation cards
    st.markdown("## 🧭 Navigate to Your Workspace")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #00c85322 0%, #00c85311 100%);
                        padding: 24px; border-radius: 12px; border-left: 4px solid #00c853;">
                <h3>📊 Monitor</h3>
                <p>Real-time Portfolio Tracking, Live Execution, and Position Management</p>
                <ul style="color: var(--text-secondary); font-size: 0.9rem;">
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
            """
            <div style="background: linear-gradient(135deg, #4c78ff22 0%, #4c78ff11 100%);
                        padding: 24px; border-radius: 12px; border-left: 4px solid #4c78ff;">
                <h3>🎯 Analyze</h3>
                <p>AI-Powered Analysis, Market Regime Detection, and Strategy Recommendations</p>
                <ul style="color: var(--text-secondary); font-size: 0.9rem;">
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
        # Check if user has ML strategies permission before showing Research
        has_ml_access = check_permission(Permission.ML_STRATEGIES)

        card_opacity = "100%" if has_ml_access else "50%"
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #ff174422 0%, #ff174411 100%);
                        padding: 24px; border-radius: 12px; border-left: 4px solid #ff1744;
                        opacity: {card_opacity};">
                <h3>🔬 Research {'🔒' if not has_ml_access else ''}</h3>
                <p>Backtesting, Options Strategies, and Portfolio Optimization</p>
                <ul style="color: var(--text-secondary); font-size: 0.9rem;">
                    <li>Backtesting Lab</li>
                    <li>Options Desk</li>
                    <li>Portfolio Optimization</li>
                </ul>
            </div>
        """,
            unsafe_allow_html=True,
        )

        if has_ml_access:
            if st.button("Go to Research →", key="nav_research", use_container_width=True):
                st.switch_page("pages/3_🔬_Research.py")
        else:
            if st.button("🔒 Upgrade to Access", key="nav_research_locked", use_container_width=True):
                st.info("Research features require BASIC tier or higher. Check Settings to upgrade.")

    col4, col5 = st.columns(2)

    with col4:
        # Check if user has ML strategies permission
        has_ml_build = check_permission(Permission.ML_STRATEGIES)

        card_opacity = "100%" if has_ml_build else "50%"
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #ffb30022 0%, #ffb30011 100%);
                        padding: 24px; border-radius: 12px; border-left: 4px solid #ffb300;
                        opacity: {card_opacity};">
                <h3>🛠️ Build {'🔒' if not has_ml_build else ''}</h3>
                <p>Create Custom Strategies and ML models</p>
                <ul style="color: var(--text-secondary); font-size: 0.9rem;">
                    <li>ML Strategy Studio</li>
                    <li>Custom Strategy Builder</li>
                    <li>Prompting</li>
                </ul>
            </div>
        """,
            unsafe_allow_html=True,
        )

        if has_ml_build:
            if st.button("Go to Build →", key="nav_build", use_container_width=True):
                st.switch_page("pages/4_🛠️_Build.py")
        else:
            if st.button("🔒 Upgrade to PRO", key="nav_build_locked", use_container_width=True):
                st.info("ML Builder requires PRO tier. Check Settings to upgrade.")

    with col5:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #9c27b022 0%, #9c27b011 100%);
                        padding: 24px; border-radius: 12px; border-left: 4px solid #9c27b0;">
                <h3>⚙️ Settings</h3>
                <p>Configuration, Risk Management, and Alerts</p>
                <ul style="color: var(--text-secondary); font-size: 0.9rem;">
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

    st.markdown("---")

    # Quick stats overview
    st.markdown("## 📈 Portfolio Overview")

    def render_header_metrics(ctx: AppContext):
        # 1. Fetch live data
        data = ctx.db.get_header_metrics(portfolio_id=1)

        # 2. Calculate Utilization via RiskManager
        risk_util = ctx.risk_manager.get_current_risk_utilization(current_exposure=data["exposure"], portfolio_value=data["nav"])

        # 3. Calculate NAV Change
        nav_change = ((data["nav"] - data["prev_nav"]) / data["prev_nav"] * 100) if data["prev_nav"] > 0 else 0.0

        # 4. Render
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Net Asset Value", f"${data['nav']:,.2f}", f"{nav_change:+.2f}%")

        with col2:
            exposure_pct = (data["exposure"] / data["nav"] * 100) if data["nav"] > 0 else 0
            st.metric("Market Exposure", f"{exposure_pct:.1f}%", f"${data['exposure']:,.0f}")

        with col3:
            st.metric("Unrealized P&L", f"${data['unrealized_pnl']:,.2f}", "Live")

        with col4:
            color = "normal" if risk_util < 70 else "inverse"
            st.metric(
                "Risk Utilization",
                f"{risk_util}%",
                "CRITICAL" if risk_util > 85 else "Nominal",
                delta_color=color,
            )

    # This fragment isolates the header for high-frequency updates
    @st.fragment(run_every="10s")
    def sync_header():
        render_header_metrics(ctx)

    sync_header()

    # Footer with user info
    st.markdown("---")
    user = st.session_state.user
    st.caption(f"Version 1.0.0 – Institutional Edition | Logged in as: {user['username']} ({user['tier'].upper()})")


if __name__ == "__main__":
    main()
