"""
Portfolio Optimization UI Component
"""

import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from optimise.optimiser import PortfolioBacktest, PortfolioOptimizer

logger = logging.getLogger(__name__)


def render_portfolio_optimization():
    """Render portfolio optimization tab"""

    st.header("📊 Portfolio Optimization")

    st.markdown(
        """
    Optimize your portfolio allocation using Modern Portfolio Theory and various strategies.
    Find the optimal balance between risk and return.
    """
    )

    # Symbol input
    st.subheader("1️⃣ Select Assets")

    symbols_input = st.text_input(
        "Enter stock symbols (comma-separated)",
        value="AAPL,MSFT,GOOGL,AMZN,TSLA",
        help="Enter 3-10 stock symbols",
    )

    symbols = [s.strip().upper() for s in symbols_input.split(",")]

    if len(symbols) < 2:
        st.error("Please enter at least 2 symbols")
        return

    if len(symbols) > 10:
        st.warning(
            "Too many symbols may slow down optimization. Consider using 3-10 symbols."
        )

    # Optimization method
    st.subheader("2️⃣ Select Optimization Method")

    method = st.selectbox(
        "Optimization Strategy",
        [
            "Maximum Sharpe Ratio",
            "Minimum Volatility",
            "Target Return",
            "Equal Weight",
            "Risk Parity",
            "Black-Litterman",
            "Efficient Frontier",
        ],
        help="Choose optimization method",
    )

    # Method-specific parameters
    params = {}

    if method == "Target Return":
        target_return = (
            st.slider(
                "Target Annual Return (%)",
                min_value=5,
                max_value=50,
                value=15,
                help="Desired annual return",
            )
            / 100
        )
        params["target_return"] = target_return

    elif method == "Black-Litterman":
        st.markdown("**Enter Your Views (Optional)**")
        st.caption(
            "Specify expected returns for symbols you have strong opinions about"
        )

        views = {}
        for symbol in symbols:
            col1, col2 = st.columns([1, 2])
            with col1:
                has_view = st.checkbox(f"{symbol}", key=f"view_{symbol}")
            with col2:
                if has_view:
                    view_return = (
                        st.slider(
                            f"Expected return for {symbol} (%)",
                            min_value=-30,
                            max_value=50,
                            value=10,
                            key=f"return_{symbol}",
                        )
                        / 100
                    )
                    views[symbol] = view_return

        if views:
            confidence = st.slider(
                "Confidence in views",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                help="How confident are you in your views? (0.1 = low, 1.0 = high)",
            )
            params["views"] = views
            params["confidence"] = confidence

    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        lookback_days = st.slider(
            "Historical Data Period (days)",
            min_value=30,
            max_value=756,
            value=252,
            help="Number of days for historical analysis (252 = 1 year)",
        )

        risk_free_rate = (
            st.slider(
                "Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.5
            )
            / 100
        )

    # Run optimization
    if st.button("🚀 Optimize Portfolio", type="primary"):
        with st.spinner("Optimizing portfolio..."):
            try:
                optimizer = PortfolioOptimizer(symbols, lookback_days)
                optimizer.fetch_data()

                # Run selected optimization
                if method == "Maximum Sharpe Ratio":
                    result = optimizer.optimize_sharpe(risk_free_rate)
                elif method == "Minimum Volatility":
                    result = optimizer.optimize_min_volatility()
                elif method == "Target Return":
                    result = optimizer.optimize_target_return(params["target_return"])
                elif method == "Equal Weight":
                    result = optimizer.equal_weight_portfolio()
                elif method == "Risk Parity":
                    result = optimizer.risk_parity_portfolio()
                elif method == "Black-Litterman":
                    if "views" in params and params["views"]:
                        result = optimizer.black_litterman(
                            params["views"], params["confidence"]
                        )
                    else:
                        st.error(
                            "Please specify at least one view for Black-Litterman optimization"
                        )
                        return
                elif method == "Efficient Frontier":
                    _render_efficient_frontier(optimizer)
                    return

                # Display results
                _display_optimization_results(result, optimizer)

                # Store in session state for backtesting
                st.session_state["optimized_weights"] = result["weights"]
                st.session_state["optimized_symbols"] = symbols

            except Exception as e:
                st.error(f"Optimization failed: {str(e)}")
                st.write("Please check:")
                st.write("- All symbols are valid")
                st.write("- Sufficient historical data available")
                st.write("- Internet connection is stable")


def _display_optimization_results(result: dict, optimizer: PortfolioOptimizer):
    """Display optimization results"""

    st.success("✅ Optimization Complete!")

    # Key metrics
    st.subheader("📊 Portfolio Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Expected Return",
            f"{result['expected_return']:.2%}",
            help="Annualized expected return",
        )

    with col2:
        st.metric(
            "Volatility (Risk)",
            f"{result['volatility']:.2%}",
            help="Annualized standard deviation",
        )

    with col3:
        st.metric(
            "Sharpe Ratio", f"{result['sharpe_ratio']:.2f}", help="Risk-adjusted return"
        )

    # Optimal weights
    st.subheader("💎 Optimal Portfolio Allocation")

    weights_df = pd.DataFrame(
        [
            {"Symbol": symbol, "Weight": weight, "Allocation %": weight * 100}
            for symbol, weight in result["weights"].items()
        ]
    ).sort_values("Weight", ascending=False)

    col1, col2 = st.columns([2, 1])

    with col1:
        # Pie chart
        fig = px.pie(
            weights_df,
            values="Weight",
            names="Symbol",
            title="Portfolio Allocation",
            hole=0.4,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Table
        display_df = weights_df[["Symbol", "Allocation %"]].copy()
        display_df["Allocation %"] = display_df["Allocation %"].apply(
            lambda x: f"{x:.2f}%"
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Dollar allocation
    st.subheader("💰 Capital Allocation")

    capital = st.number_input(
        "Investment Amount ($)",
        min_value=1000,
        max_value=10000000,
        value=100000,
        step=10000,
    )

    allocation_df = pd.DataFrame(
        [
            {
                "Symbol": symbol,
                "Weight": weight,
                "Amount": capital * weight,
                "Shares (approx)": int(
                    capital * weight / _get_current_price(symbol, optimizer)
                ),
            }
            for symbol, weight in result["weights"].items()
        ]
    ).sort_values("Amount", ascending=False)

    # Format for display
    display_alloc = allocation_df.copy()
    display_alloc["Weight"] = display_alloc["Weight"].apply(lambda x: f"{x:.2%}")
    display_alloc["Amount"] = display_alloc["Amount"].apply(lambda x: f"${x:,.2f}")

    st.dataframe(display_alloc, use_container_width=True, hide_index=True)

    # Backtest section
    st.divider()
    st.subheader("📈 Backtest Optimized Portfolio")

    col1, col2 = st.columns(2)

    with col1:
        backtest_period = st.selectbox(
            "Backtest Period", ["1mo", "3mo", "6mo", "1y", "2y"]
        )

    with col2:
        start_capital = st.number_input(
            "Starting Capital", min_value=1000, value=100000
        )

    if st.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            try:
                backtester = PortfolioBacktest(
                    list(result["weights"].keys()), result["weights"]
                )

                backtest_results = backtester.run_backtest(
                    start_capital=start_capital, period=backtest_period
                )

                _display_backtest_results(backtest_results)

            except Exception as e:
                st.error(f"Backtest failed: {str(e)}")


def _display_backtest_results(results: dict):
    """Display backtest results"""

    st.subheader("📊 Backtest Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Return", f"{results['total_return']:.2f}%")
    with col2:
        st.metric("Volatility", f"{results['volatility']:.2f}%")
    with col3:
        st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
    with col4:
        st.metric("Max Drawdown", f"{results['max_drawdown']:.2f}%")

    # Equity curve
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=results["equity_curve"].index,
            y=results["equity_curve"].values,
            mode="lines",
            name="Portfolio Value",
            line=dict(color="#00ff88", width=2),
            fill="tonexty",
        )
    )

    fig.update_layout(
        title="Portfolio Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_dark",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Returns distribution
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            results["returns"] * 100,
            nbins=50,
            title="Returns Distribution",
            labels={"value": "Daily Return (%)"},
        )
        fig.update_layout(template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Rolling volatility
        rolling_vol = results["returns"].rolling(20).std() * (252**0.5) * 100

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                mode="lines",
                name="Rolling Volatility",
                line=dict(color="orange"),
            )
        )
        fig.update_layout(
            title="20-Day Rolling Volatility",
            xaxis_title="Date",
            yaxis_title="Volatility (%)",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_efficient_frontier(optimizer: PortfolioOptimizer):
    """Render efficient frontier"""

    with st.spinner("Generating efficient frontier..."):
        try:
            # Generate frontier
            frontier_df = optimizer.efficient_frontier(num_portfolios=50)

            # Get special portfolios
            min_vol = optimizer.optimize_min_volatility()
            max_sharpe = optimizer.optimize_sharpe()

            st.success("✅ Efficient Frontier Generated!")

            # Plot
            fig = go.Figure()

            # Efficient frontier
            fig.add_trace(
                go.Scatter(
                    x=frontier_df["volatility"] * 100,
                    y=frontier_df["return"] * 100,
                    mode="lines+markers",
                    name="Efficient Frontier",
                    line=dict(color="cyan", width=2),
                    marker=dict(size=6),
                )
            )

            # Min volatility portfolio
            fig.add_trace(
                go.Scatter(
                    x=[min_vol["volatility"] * 100],
                    y=[min_vol["expected_return"] * 100],
                    mode="markers",
                    name="Min Volatility",
                    marker=dict(size=15, color="green", symbol="star"),
                )
            )

            # Max Sharpe portfolio
            fig.add_trace(
                go.Scatter(
                    x=[max_sharpe["volatility"] * 100],
                    y=[max_sharpe["expected_return"] * 100],
                    mode="markers",
                    name="Max Sharpe",
                    marker=dict(size=15, color="gold", symbol="star"),
                )
            )

            fig.update_layout(
                title="Efficient Frontier",
                xaxis_title="Volatility (Risk) %",
                yaxis_title="Expected Return %",
                template="plotly_dark",
                hovermode="closest",
                showlegend=True,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Display special portfolios
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🟢 Minimum Volatility Portfolio")
                st.write(f"**Return:** {min_vol['expected_return']:.2%}")
                st.write(f"**Risk:** {min_vol['volatility']:.2%}")
                st.write(f"**Sharpe:** {min_vol['sharpe_ratio']:.2f}")

                weights_df = pd.DataFrame(
                    [
                        {"Symbol": s, "Weight": f"{w:.2%}"}
                        for s, w in min_vol["weights"].items()
                    ]
                )
                st.dataframe(weights_df, hide_index=True)

            with col2:
                st.markdown("### 🟡 Maximum Sharpe Portfolio")
                st.write(f"**Return:** {max_sharpe['expected_return']:.2%}")
                st.write(f"**Risk:** {max_sharpe['volatility']:.2%}")
                st.write(f"**Sharpe:** {max_sharpe['sharpe_ratio']:.2f}")

                weights_df = pd.DataFrame(
                    [
                        {"Symbol": s, "Weight": f"{w:.2%}"}
                        for s, w in max_sharpe["weights"].items()
                    ]
                )
                st.dataframe(weights_df, hide_index=True)

        except Exception as e:
            st.error(f"Failed to generate efficient frontier: {str(e)}")


def _get_current_price(symbol: str, optimizer: PortfolioOptimizer) -> float:
    """Get current price for a symbol"""
    try:
        if optimizer.returns is not None and symbol in optimizer.returns.columns:
            # Use last available price from optimizer data
            return 100  # Placeholder, would need actual price data
        return 100
    except Exception as e:
        logger.exception(f"Failed to get current price: {e}")
        return 100
