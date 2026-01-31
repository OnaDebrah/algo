"""
Options Strategy Builder UI Component
"""

from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import streamlit as st
from streamlit.core.data_fetcher import fetch_stock_data
from streamlit.strategies.options_builder import OptionsStrategyBuilder, create_preset_strategy
from streamlit.strategies.options_strategies import (
    OptionsChain,
    OptionsStrategy,
    OptionType,
    get_strategy_description,
)


def render_options_strategy_builder(ml_models: dict = None):
    """Main options strategy builder interface"""

    st.header("📈 Options Strategy Builder")

    st.markdown(
        """
    Build, analyze, and backtest complex options strategies with real-time Greeks calculation,
    payoff diagrams, and probability analysis.
    """
    )

    tab_design, tab_chain = st.tabs(["🎨 Strategy Designer", "⛓️ Options Chain"])

    with tab_design:
        st.markdown("### Strategy Blueprint")
        mode = st.radio("Design Mode", ["Preset Templates", "Custom Leg Builder"], horizontal=True, key="design_mode")
        if mode == "Preset Templates":
            render_preset_strategies(ml_models)
        else:
            render_custom_builder()

    with tab_chain:
        render_options_chain_viewer()


def render_preset_strategies(ml_models: dict = None):
    """Render preset strategy interface"""

    st.subheader("🎯 Preset Strategy Templates")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Symbol input
        symbol = st.text_input("Underlying Symbol", value="AAPL", help="Enter stock ticker").upper()

    with col2:
        # Fetch current price
        st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Fetch Price"):
            chain = OptionsChain(symbol)
            price = chain.get_current_price()
            if price > 0:
                st.session_state["current_price"] = price
                st.success(f"${price:.2f}")

    # ML Forecast Section
    if ml_models and symbol in ml_models:
        model = ml_models[symbol]
        if model.is_trained:
            st.markdown("### 🤖 ML Forecast")
            col_ml1, col_ml2 = st.columns([1, 2])
            with col_ml1:
                st.caption(f"Model: {model.name}")
                if st.button("🔮 Predict Direction"):
                    try:
                        data = fetch_stock_data(symbol, period="3mo", interval="1d")
                        signal = model.generate_signal(data)
                        if signal == 1:
                            st.success("📈 **Bullish** Prediction")
                        elif signal == -1:
                            st.error("📉 **Bearish** Prediction")
                        else:
                            st.info("➖ **Neutral** Prediction")
                    except Exception as e:
                        st.warning(f"Prediction failed: {e}")
            with col_ml2:
                st.info("Use this forecast to select a strategy below (e.g. Bullish -> Calls/Bull Spreads)")
    elif ml_models:
        st.caption(f"ℹ️ Train an ML model for {symbol} in the 'Build' tab to see forecasts here.")

    current_price = st.session_state.get("current_price", 150.0)

    st.metric("Current Price", f"${current_price:.2f}")

    # Strategy selection
    st.markdown("### Select Strategy")

    # Create strategy cards
    col1, col2, col3 = st.columns(3)

    strategies_group_1 = [
        OptionsStrategy.COVERED_CALL,
        OptionsStrategy.CASH_SECURED_PUT,
        OptionsStrategy.PROTECTIVE_PUT,
    ]

    strategies_group_2 = [
        OptionsStrategy.VERTICAL_CALL_SPREAD,
        OptionsStrategy.VERTICAL_PUT_SPREAD,
        OptionsStrategy.IRON_CONDOR,
    ]

    strategies_group_3 = [
        OptionsStrategy.STRADDLE,
        OptionsStrategy.STRANGLE,
        OptionsStrategy.BUTTERFLY_SPREAD,
    ]

    # Initialize selected_strategy to None
    selected_strategy = None

    with col1:
        for strategy in strategies_group_1:
            if st.button(strategy.value, key=f"preset_{strategy.value}", use_container_width=True):
                selected_strategy = strategy
                st.session_state["selected_strategy"] = strategy

    with col2:
        for strategy in strategies_group_2:
            if st.button(strategy.value, key=f"preset_{strategy.value}", use_container_width=True):
                selected_strategy = strategy
                st.session_state["selected_strategy"] = strategy

    with col3:
        for strategy in strategies_group_3:
            if st.button(strategy.value, key=f"preset_{strategy.value}", use_container_width=True):
                selected_strategy = strategy
                st.session_state["selected_strategy"] = strategy

    # FIXED: Check if selected_strategy exists and is not None before rendering
    if "selected_strategy" in st.session_state and st.session_state["selected_strategy"] is not None:
        strategy = st.session_state["selected_strategy"]
        _render_strategy_details(strategy, symbol, current_price)
    elif selected_strategy is not None:
        # Fallback if session state hasn't been set yet
        _render_strategy_details(selected_strategy, symbol, current_price)


def _render_strategy_details(strategy: OptionsStrategy, symbol: str, current_price: float):
    """Render details and configuration for selected strategy"""

    # FIXED: Add safety check at the beginning
    if strategy is None:
        st.warning("Please select a strategy from the buttons above.")
        return

    st.markdown("---")
    st.subheader(f"📊 {strategy.value}")

    # Show strategy description
    desc = get_strategy_description(strategy)

    if desc:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Description:** {desc.get('description', 'N/A')}")
            st.markdown(f"**Outlook:** {desc.get('outlook', 'N/A')}")
            st.markdown(f"**Best For:** {desc.get('best_for', 'N/A')}")

        with col2:
            st.markdown(f"**Max Profit:** {desc.get('max_profit', 'N/A')}")
            st.markdown(f"**Max Loss:** {desc.get('max_loss', 'N/A')}")
            st.markdown(f"**Breakeven:** {desc.get('breakeven', 'N/A')}")

    # Strategy parameters
    st.markdown("### Configure Strategy")

    col1, col2 = st.columns(2)

    with col1:
        days_to_exp = st.slider(
            "Days to Expiration",
            min_value=1,
            max_value=365,
            value=30,
            help="Number of days until options expire",
        )

    with col2:
        volatility = st.slider(
            "Implied Volatility",
            min_value=0.1,
            max_value=1.0,
            value=0.3,
            step=0.05,
            format="%.0f%%",
            help="Expected volatility (30% = 0.3)",
        )

    expiration = datetime.now() + timedelta(days=days_to_exp)

    # Strategy-specific parameters
    kwargs = _get_strategy_parameters(strategy, current_price)

    # Build strategy
    if st.button("🚀 Build & Analyze Strategy", type="primary"):
        try:
            builder = create_preset_strategy(strategy, symbol, current_price, expiration, **kwargs)
            _display_strategy_analysis(builder, volatility, current_price)
        except Exception as e:
            st.error(f"Error building strategy: {e}")


def _get_strategy_parameters(strategy: OptionsStrategy, current_price: float) -> dict:
    """Get strategy-specific parameters"""

    kwargs = {}

    if strategy == OptionsStrategy.COVERED_CALL:
        strike = st.slider(
            "Call Strike Price",
            min_value=current_price * 0.9,
            max_value=current_price * 1.2,
            value=current_price * 1.05,
            step=0.5,
        )
        kwargs["strike"] = strike

    elif strategy == OptionsStrategy.CASH_SECURED_PUT:
        strike = st.slider(
            "Put Strike Price",
            min_value=current_price * 0.8,
            max_value=current_price * 1.0,
            value=current_price * 0.95,
            step=0.5,
        )
        kwargs["strike"] = strike

    elif strategy == OptionsStrategy.VERTICAL_CALL_SPREAD:
        col1, col2 = st.columns(2)
        with col1:
            long_strike = st.number_input("Long Call Strike", value=current_price, step=0.5)
        with col2:
            short_strike = st.number_input("Short Call Strike", value=current_price * 1.05, step=0.5)
        kwargs["long_strike"] = long_strike
        kwargs["short_strike"] = short_strike

    elif strategy == OptionsStrategy.VERTICAL_PUT_SPREAD:
        col1, col2 = st.columns(2)
        with col1:
            long_strike = st.number_input("Long Put Strike", value=current_price, step=0.5)
        with col2:
            short_strike = st.number_input("Short Put Strike", value=current_price * 0.95, step=0.5)
        kwargs["long_strike"] = long_strike
        kwargs["short_strike"] = short_strike

    elif strategy == OptionsStrategy.IRON_CONDOR:
        st.markdown("**Configure Strikes:**")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            put_long = st.number_input("Put Long", value=current_price * 0.90, step=0.5)
        with col2:
            put_short = st.number_input("Put Short", value=current_price * 0.95, step=0.5)
        with col3:
            call_short = st.number_input("Call Short", value=current_price * 1.05, step=0.5)
        with col4:
            call_long = st.number_input("Call Long", value=current_price * 1.10, step=0.5)

        kwargs["put_long_strike"] = put_long
        kwargs["put_short_strike"] = put_short
        kwargs["call_short_strike"] = call_short
        kwargs["call_long_strike"] = call_long

    elif strategy == OptionsStrategy.STRADDLE:
        strike = st.slider(
            "Strike Price (ATM recommended)",
            min_value=current_price * 0.95,
            max_value=current_price * 1.05,
            value=current_price,
            step=0.5,
        )
        kwargs["strike"] = strike

    elif strategy == OptionsStrategy.STRANGLE:
        col1, col2 = st.columns(2)
        with col1:
            put_strike = st.number_input("Put Strike", value=current_price * 0.95, step=0.5)
        with col2:
            call_strike = st.number_input("Call Strike", value=current_price * 1.05, step=0.5)
        kwargs["put_strike"] = put_strike
        kwargs["call_strike"] = call_strike

    return kwargs


def render_custom_builder():
    """Render custom strategy builder"""

    st.subheader("🔧 Custom Strategy Builder")

    # Initialize session state
    if "custom_builder" not in st.session_state:
        st.session_state["custom_builder"] = None
        st.session_state["custom_legs"] = []

    # Symbol input
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        symbol = st.text_input("Underlying Symbol", value="AAPL").upper()

    with col2:
        if st.button("🔄 Initialize"):
            try:
                builder = OptionsStrategyBuilder(symbol)
                st.session_state["custom_builder"] = builder
                st.session_state["custom_legs"] = []
                st.success("Builder initialized!")
            except Exception as e:
                st.error(f"Failed to initialize: {e}")

    with col3:
        if st.button("🗑️ Clear All"):
            st.session_state["custom_legs"] = []
            if st.session_state["custom_builder"]:
                st.session_state["custom_builder"].clear_legs()
                st.success("Cleared all legs")

    if st.session_state["custom_builder"] is None:
        st.info("👆 Click 'Initialize' to start building your strategy")
        return

    builder = st.session_state["custom_builder"]

    # Current price display
    st.metric("Current Price", f"${builder.current_price:.2f}")

    # Add leg section
    st.markdown("### ➕ Add Option Leg")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        leg_type = st.selectbox("Type", [OptionType.CALL.value, OptionType.PUT.value], key="leg_type")
        option_type = OptionType.CALL if leg_type == "Call" else OptionType.PUT

    with col2:
        position = st.selectbox("Position", ["Long", "Short"], key="leg_position")
        quantity = 1 if position == "Long" else -1

    with col3:
        strike = st.number_input("Strike", value=builder.current_price, step=0.5, key="leg_strike")

    with col4:
        days = st.number_input("Days to Exp", min_value=1, max_value=365, value=30, key="leg_days")

    with col5:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ Add Leg"):
            try:
                expiry = datetime.now() + timedelta(days=days)
                builder.add_leg(option_type, strike, expiry, quantity)

                # Store leg info for display
                st.session_state["custom_legs"].append({"type": leg_type, "position": position, "strike": strike, "days": days})

                st.success(f"Added {position} {leg_type} @ ${strike}")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to add leg: {e}")

    # Display current legs
    if st.session_state["custom_legs"]:
        st.markdown("### 📋 Current Legs")

        legs_df = pd.DataFrame(st.session_state["custom_legs"])
        st.dataframe(legs_df, use_container_width=True, hide_index=True)

        # Analysis section
        st.markdown("### 📊 Strategy Analysis")

        col1, col2 = st.columns(2)

        with col1:
            volatility = st.slider(
                "Implied Volatility",
                min_value=0.1,
                max_value=1.0,
                value=0.3,
                step=0.05,
                key="custom_vol",
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🧪 Analyze Strategy", type="primary"):
                try:
                    _display_strategy_analysis(builder, volatility, builder.current_price)
                except Exception as e:
                    st.error(f"Analysis failed: {e}")


def _display_strategy_analysis(builder: OptionsStrategyBuilder, volatility: float, current_price: float):
    """Display comprehensive strategy analysis"""

    st.markdown("---")
    st.subheader("📈 Strategy Analysis Results")

    try:
        # Calculate metrics
        initial_cost = builder.get_initial_cost()
        greeks = builder.calculate_greeks(volatility)
        breakevens = builder.get_breakeven_points()
        max_profit, profit_condition = builder.get_max_profit()
        max_loss, loss_condition = builder.get_max_loss()
        pop = builder.calculate_probability_of_profit(volatility) * 100

        # Display key metrics
        st.markdown("### 💰 Key Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Initial Cost/Credit",
                f"${initial_cost:.2f}",
                delta="Credit" if initial_cost < 0 else "Debit",
            )

        with col2:
            st.metric("Max Profit", f"${max_profit:.2f}", delta=profit_condition)

        with col3:
            st.metric("Max Loss", f"${max_loss:.2f}", delta=loss_condition)

        with col4:
            st.metric("Probability of Profit", f"{pop:.1f}%")

        # Display Greeks
        st.markdown("### 📐 Greeks")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Delta", f"{greeks['delta']:.3f}")
            st.caption("Price sensitivity")

        with col2:
            st.metric("Gamma", f"{greeks['gamma']:.3f}")
            st.caption("Delta change rate")

        with col3:
            st.metric("Theta", f"{greeks['theta']:.3f}")
            st.caption("Time decay ($/day)")

        with col4:
            st.metric("Vega", f"{greeks['vega']:.3f}")
            st.caption("Volatility sensitivity")

        with col5:
            st.metric("Rho", f"{greeks['rho']:.3f}")
            st.caption("Interest rate sensitivity")

        # Breakeven points
        if breakevens:
            st.markdown("### ⌖ Breakeven Points")
            for i, be in enumerate(breakevens, 1):
                st.write(f"**Breakeven {i}:** ${be:.2f}")

        # Payoff diagram
        _plot_payoff_diagram(builder, current_price, breakevens)

        # Greeks surface (optional advanced view)
        with st.expander("Δ Advanced: Greeks Heatmap"):
            _plot_greeks_surface(builder, current_price, volatility)

    except Exception as e:
        st.error(f"Error during analysis: {e}")
        st.exception(e)


def _plot_payoff_diagram(builder: OptionsStrategyBuilder, current_price: float, breakevens: List[float]):
    """Plot payoff diagram"""

    st.markdown("### Σ Payoff Diagram")

    try:
        # Generate price range
        price_range = np.linspace(current_price * 0.7, current_price * 1.3, 200)
        payoffs = builder.calculate_payoff(price_range)

        # Create figure
        fig = go.Figure()

        # Payoff line
        fig.add_trace(
            go.Scatter(
                x=price_range,
                y=payoffs,
                mode="lines",
                name="Payoff at Expiration",
                line=dict(color="#00ff88", width=3),
            )
        )

        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)

        # Current price line
        fig.add_vline(
            x=current_price,
            line_dash="dot",
            line_color="yellow",
            annotation_text="Current Price",
        )

        # Breakeven points
        for be in breakevens:
            fig.add_vline(x=be, line_dash="dot", line_color="cyan", annotation_text=f"BE: ${be:.2f}")

        # Profit/loss zones
        fig.add_shape(
            type="rect",
            x0=price_range[0],
            x1=price_range[-1],
            y0=0,
            y1=max(payoffs),
            fillcolor="green",
            opacity=0.1,
            layer="below",
            line_width=0,
        )

        fig.add_shape(
            type="rect",
            x0=price_range[0],
            x1=price_range[-1],
            y0=min(payoffs),
            y1=0,
            fillcolor="red",
            opacity=0.1,
            layer="below",
            line_width=0,
        )

        fig.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Underlying Price at Expiration ($)",
            yaxis_title="Profit/Loss ($)",
            hovermode="x unified",
            showlegend=True,
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to plot payoff diagram: {e}")


def _plot_greeks_surface(builder: OptionsStrategyBuilder, current_price: float, volatility: float):
    """Plot Greeks sensitivity heatmap"""

    st.markdown("#### Greeks Sensitivity Analysis")

    try:
        # Generate price and volatility ranges
        price_range = np.linspace(current_price * 0.8, current_price * 1.2, 30)
        vol_range = np.linspace(0.1, 0.8, 30)

        # Calculate Greeks for each combination
        delta_surface = np.zeros((len(vol_range), len(price_range)))

        for i, vol in enumerate(vol_range):
            for j, price in enumerate(price_range):
                # Temporarily update builder price
                original_price = builder.current_price
                builder.current_price = price
                greeks = builder.calculate_greeks(vol)
                delta_surface[i, j] = greeks["delta"]
                builder.current_price = original_price

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=delta_surface,
                x=price_range,
                y=vol_range * 100,
                colorscale="RdYlGn",
                colorbar=dict(title="Delta"),
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="Underlying Price ($)",
            yaxis_title="Implied Volatility (%)",
            title="Delta Sensitivity Heatmap",
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to plot Greeks surface: {e}")


def render_options_chain_viewer():
    """Render options chain viewer"""

    st.subheader("🔍 Options Chain Viewer")

    col1, col2 = st.columns([2, 1])

    with col1:
        symbol = st.text_input("Symbol", value="AAPL", key="chain_symbol").upper()

    with col2:
        st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("📥 Load Chain"):
            try:
                chain = OptionsChain(symbol)
                expirations = chain.get_expirations()

                if expirations:
                    st.session_state["chain"] = chain
                    st.session_state["expirations"] = expirations
                    st.success(f"Found {len(expirations)} expirations")
                else:
                    st.warning("No options data available for this symbol")
            except Exception as e:
                st.error(f"Failed to load chain: {e}")

    if "chain" in st.session_state and "expirations" in st.session_state:
        chain = st.session_state["chain"]
        expirations = st.session_state["expirations"]

        # Current price
        current_price = chain.get_current_price()
        st.metric("Current Price", f"${current_price:.2f}")

        # Expiration selector
        selected_exp = st.selectbox("Expiration Date", expirations)

        if st.button("⊞ Show Chain"):
            try:
                calls, puts = chain.get_chain(selected_exp)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 📞 Calls")
                    if not calls.empty:
                        display_cols = [
                            "strike",
                            "lastPrice",
                            "bid",
                            "ask",
                            "volume",
                            "impliedVolatility",
                        ]
                        display_cols = [c for c in display_cols if c in calls.columns]
                        st.dataframe(calls[display_cols], use_container_width=True, hide_index=True)
                    else:
                        st.info("No call options available")

                with col2:
                    st.markdown("#### 📉 Puts")
                    if not puts.empty:
                        display_cols = [
                            "strike",
                            "lastPrice",
                            "bid",
                            "ask",
                            "volume",
                            "impliedVolatility",
                        ]
                        display_cols = [c for c in display_cols if c in puts.columns]
                        st.dataframe(puts[display_cols], use_container_width=True, hide_index=True)
                    else:
                        st.info("No put options available")

            except Exception as e:
                st.error(f"Failed to show chain: {e}")
