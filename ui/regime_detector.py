"""
Market Regime Detector UI
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from analytics.market_regime_detector import MarketRegimeDetector
from core import fetch_stock_data


def get_regime_emoji(regime: str) -> str:
    """Get emoji for regime type"""
    emoji_map = {
        "trending_bull": "🟢",
        "trending_bear": "🔴",
        "mean_reverting": "🔵",
        "high_volatility": "🟡",
        "low_volatility": "⚪",
        "crisis": "🔴🔴",
        "recovery": "🟢🟡",
        "transition": "🟣",
        "unknown": "⚫",
    }
    return emoji_map.get(regime, "⚫")


def get_regime_color(regime: str) -> str:
    """Get color for regime type"""
    color_map = {
        "trending_bull": "#00c853",
        "trending_bear": "#ff1744",
        "mean_reverting": "#2196f3",
        "high_volatility": "#ffd600",
        "low_volatility": "#90a4ae",
        "crisis": "#d50000",
        "recovery": "#76ff03",
        "transition": "#9c27b0",
        "unknown": "#616161",
    }
    return color_map.get(regime, "#616161")


def render_regime_detector(db):
    """
    Main regime detector interface

    Args:
        db: Database manager instance
    """
    st.header("🎯 Market Regime Detector")

    st.markdown(
        """
    **Intelligent regime detection system** that identifies market conditions
    and recommends optimal strategy allocations.

    Uses statistical methods, technical indicators, and machine learning to classify
    8 distinct market regimes and provide strategy recommendations.
    """
    )

    # Initialize detector in session state if not exists
    if "regime_detector" not in st.session_state:
        st.session_state.regime_detector = MarketRegimeDetector(lookback_period=252, use_ml=True, confidence_threshold=0.7)

    detector = st.session_state.regime_detector

    # Settings in sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Regime Detector Settings")

        use_ml = st.checkbox(
            "Use Machine Learning",
            value=True,
            help="Enable ML-based ensemble detection",
        )

        period = st.selectbox(
            "Period",
            ["1mo", "3mo", "6mo", "1y", "2y"],
            help="Historical data period",
            key="regime_detector_backtest_period",
        )

        interval = st.selectbox(
            "Interval",
            ["1h", "1d", "1wk"],
            help="Data interval/timeframe",
            key="regime_detector_backtest_interval",
        )

        lookback = st.slider(
            "Lookback Period (days)",
            min_value=63,
            max_value=504,
            value=252,
            step=21,
            help="Historical period for analysis",
        )

        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.5,
            max_value=0.95,
            value=0.7,
            step=0.05,
            help="Minimum confidence for regime classification",
        )

        if st.button("Update Detector Settings", type="secondary"):
            # Reinitialize detector with new settings
            st.session_state.regime_detector = MarketRegimeDetector(
                lookback_period=lookback,
                use_ml=use_ml,
                confidence_threshold=confidence_threshold,
            )
            st.success("Settings updated!")

    st.markdown("---")

    # Data input section
    col1, col2, col3 = st.columns([3, 2, 2])

    with col1:
        symbol = st.text_input("Symbol", value="SPY", help="Ticker symbol for analysis").upper()

    with col2:
        data_source = st.selectbox(
            "Data Source",
            ["Yahoo Finance", "Database", "Generate Example"],
            help="Where to fetch price data from",
        )

    with col3:
        if st.button("🔍 Detect Regime", type="primary", use_container_width=True):
            with st.spinner(f"Analyzing {symbol}..."):
                try:
                    # Fetch data based on source
                    if data_source == "Yahoo Finance":
                        price_data = fetch_stock_data(symbol, period, interval)
                    elif data_source == "Database":
                        price_data = fetch_database_data(db, symbol, lookback)
                    else:  # Generate Example
                        price_data = generate_synthetic_data(lookback)

                    if price_data is None or len(price_data) < 63:
                        st.error("Insufficient data. Need at least 63 days of price history.")
                        return

                    # Detect regime
                    regime_info = detector.detect_current_regime(price_data)

                    # Store in session state
                    st.session_state.current_regime = regime_info
                    st.session_state.regime_symbol = symbol
                    st.session_state.regime_price_data = price_data

                    st.success(f"✅ Regime detected for {symbol}!")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error detecting regime: {str(e)}")
                    import traceback

                    st.error(traceback.format_exc())

    st.markdown("---")

    # Display results if available
    if "current_regime" in st.session_state and st.session_state.current_regime:
        render_regime_dashboard(
            st.session_state.current_regime,
            detector,
            st.session_state.get("regime_price_data"),
        )
    else:
        st.info("👆 Enter a symbol and click 'Detect Regime' to analyze market conditions")

        # Show example of what to expect
        with st.expander("ℹ️ What will this show?"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    """
                **Regime Analysis:**
                - Current market regime classification
                - Confidence and strength metrics
                - Early warning indicators
                - Probability distribution across regimes
                """
                )

            with col2:
                st.markdown(
                    """
                **Strategy Recommendations:**
                - Optimal allocation across strategy types
                - Risk-adjusted positioning
                - Regime-specific insights
                - Historical regime patterns
                """
                )


def fetch_database_data(db, symbol: str, days: int) -> pd.Series:
    """Fetch data from database"""
    try:
        # Use your existing database methods
        # This is a placeholder - adjust to your actual DB structure

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 100)

        # Example: db.get_historical_prices(symbol, start_date, end_date)
        # Adjust this to match your actual database API

        query = f"""
            SELECT date, close
            FROM price_data
            WHERE symbol = '{symbol}'
            AND date >= '{start_date.strftime('%Y-%m-%d')}'
            ORDER BY date
        """

        result = db.execute_query(query)

        if not result:
            st.warning(f"No data in database for {symbol}. Try Yahoo Finance.")
            return None

        df = pd.DataFrame(result, columns=["date", "close"])
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        return df["close"]

    except Exception as e:
        st.error(f"Database error: {str(e)}")
        st.info("Tip: Make sure you have price_data table with symbol, date, close columns")
        return None


def generate_synthetic_data(days: int) -> pd.Series:
    """Generate synthetic price data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

    # Simulate realistic market data with regime changes
    prices = [100.0]
    current_price = 100.0

    # Define regime periods
    regime_periods = [
        ("trending_bull", days // 3),
        ("high_volatility", days // 6),
        ("mean_reverting", days // 3),
        ("recovery", days - (days // 3 + days // 6 + days // 3)),
    ]

    for regime, length in regime_periods:
        for _ in range(length):
            if regime == "trending_bull":
                ret = np.random.normal(0.0008, 0.01)
            elif regime == "high_volatility":
                ret = np.random.normal(0, 0.02)
            elif regime == "mean_reverting":
                deviation = (current_price - 100) / 100
                ret = np.random.normal(-0.1 * deviation, 0.008)
            else:  # recovery
                ret = np.random.normal(0.001, 0.012)

            current_price *= 1 + ret
            prices.append(current_price)

    return pd.Series(prices[:days], index=dates)


def render_regime_dashboard(regime_info: dict, detector, price_data: pd.Series = None):
    """Render complete regime analysis dashboard"""

    regime = regime_info.get("regime", "unknown")
    confidence = regime_info.get("confidence", 0)
    strength = regime_info.get("regime_strength", 0)
    warning = regime_info.get("change_warning", {})

    # Header metrics
    st.markdown("### 🎯 Current Market Regime")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Regime", f"{get_regime_emoji(regime)} {regime.replace('_', ' ').title()}")

    with col2:
        st.metric(
            "Confidence",
            f"{confidence:.1%}",
            delta="High" if confidence > 0.7 else "Low",
            delta_color="normal" if confidence > 0.7 else "inverse",
        )

    with col3:
        st.metric(
            "Strength",
            f"{strength:.2f}σ",
            delta="Strong" if strength > 1.5 else "Weak",
            delta_color="normal" if strength > 1.5 else "inverse",
        )

    with col4:
        warning_status = "⚠️ Warning" if warning.get("warning") else "✅ Stable"
        st.metric(
            "Status",
            warning_status,
            delta="Change Detected" if warning.get("warning") else "Stable",
            delta_color="inverse" if warning.get("warning") else "normal",
        )

    # Warning panel
    if warning.get("warning"):
        st.warning(
            f"""
        ⚠️ **Regime Change Warning Detected**

        Confidence trend: {warning.get('confidence_trend', 0):.4f} (declining)

        **Recommendation:** {warning.get('recommendation', 'maintain').replace('_', ' ').title()}
        """
        )

    st.markdown("---")

    # Main content - two columns
    col_left, col_right = st.columns([1, 1])

    with col_left:
        # Regime probability scores
        st.markdown("### 📊 Regime Probabilities")
        scores = regime_info.get("scores", {})

        if scores:
            scores_df = pd.DataFrame(
                [{"Regime": k.replace("_", " ").title(), "Probability": v} for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
            )

            fig = px.bar(
                scores_df.head(5),
                x="Probability",
                y="Regime",
                orientation="h",
                color="Probability",
                color_continuous_scale="RdYlGn",
                height=300,
            )
            fig.update_layout(
                showlegend=False,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Probability",
                yaxis_title="",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Detailed analysis expander
        with st.expander("📋 Detailed Analysis", expanded=False):
            st.markdown("**Detection Methods:**")
            st.write(f"• Method: {regime_info.get('method', 'N/A').title()}")
            st.write(f"• Statistical: {regime_info.get('statistical_regime', 'N/A').replace('_', ' ').title()}")

            if regime_info.get("ml_regime"):
                st.write(f"• ML Prediction: {regime_info['ml_regime'].replace('_', ' ').title()}")

    with col_right:
        # Strategy allocation
        st.markdown("### 💼 Recommended Allocation")
        allocation = regime_info.get("strategy_allocation", {})

        if allocation:
            # Filter out very small allocations
            filtered_alloc = {k: v for k, v in allocation.items() if v > 0.01}

            alloc_df = pd.DataFrame([{"Strategy": k.replace("_", " ").title(), "Weight": v} for k, v in filtered_alloc.items()])

            fig = px.pie(
                alloc_df,
                values="Weight",
                names="Strategy",
                hole=0.4,
                height=300,
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(
                showlegend=True,
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="v", yanchor="middle", y=0.5),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Allocation table
            st.markdown("**Breakdown:**")
            for _, row in alloc_df.sort_values("Weight", ascending=False).iterrows():
                st.write(f"• {row['Strategy']}: **{row['Weight']:.1%}**")

    st.markdown("---")

    # Regime history
    if len(detector.regime_history) > 1:
        st.markdown("### 📈 Regime History")
        render_regime_history_chart(detector.regime_history, price_data)

        st.markdown("---")

        # Statistics
        render_regime_statistics(detector.regime_history)


def render_regime_history_chart(regime_history: list, price_data: pd.Series = None):
    """Render regime timeline with price overlay"""

    if len(regime_history) < 2:
        st.info("Not enough history to display timeline")
        return

    history_df = pd.DataFrame(
        [
            {
                "timestamp": entry["timestamp"],
                "regime": entry["regime"],
                "confidence": entry["confidence"],
                "strength": entry.get("strength", 0),
            }
            for entry in regime_history
        ]
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Regime Timeline", "Confidence & Strength"),
        row_heights=[0.6, 0.4],
    )

    # Add price data if available
    if price_data is not None and len(price_data) > 0:
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data.values,
                name="Price",
                line=dict(color="lightgray", width=2),
                opacity=0.6,
            ),
            row=1,
            col=1,
        )

    # Add regime color blocks
    for i in range(len(history_df) - 1):
        current = history_df.iloc[i]
        next_row = history_df.iloc[i + 1]

        fig.add_vrect(
            x0=current["timestamp"],
            x1=next_row["timestamp"],
            fillcolor=get_regime_color(current["regime"]),
            opacity=0.3,
            layer="below",
            line_width=0,
            row=1,
            col=1,
        )

    # Confidence line
    fig.add_trace(
        go.Scatter(
            x=history_df["timestamp"],
            y=history_df["confidence"],
            name="Confidence",
            line=dict(color="blue", width=2),
        ),
        row=2,
        col=1,
    )

    # Strength line
    fig.add_trace(
        go.Scatter(
            x=history_df["timestamp"],
            y=history_df["strength"],
            name="Strength",
            line=dict(color="green", width=2),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(height=500, hovermode="x unified", showlegend=True)

    fig.update_yaxes(title_text="Price / Regime", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def render_regime_statistics(regime_history: list):
    """Render regime statistics"""

    st.markdown("### 📊 Regime Statistics")

    regimes = [entry["regime"] for entry in regime_history]
    regime_counts = pd.Series(regimes).value_counts()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Regime Distribution:**")
        for regime, count in regime_counts.items():
            percentage = count / len(regimes) * 100
            emoji = get_regime_emoji(regime)
            st.write(f"{emoji} {regime.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")

    with col2:
        st.markdown("**Average Duration:**")

        # Calculate durations
        durations = {}
        current_regime = regimes[0]
        current_duration = 1

        for i in range(1, len(regimes)):
            if regimes[i] == current_regime:
                current_duration += 1
            else:
                if current_regime not in durations:
                    durations[current_regime] = []
                durations[current_regime].append(current_duration)
                current_regime = regimes[i]
                current_duration = 1

        if current_regime not in durations:
            durations[current_regime] = []
        durations[current_regime].append(current_duration)

        for regime, dur_list in durations.items():
            if dur_list:
                avg_duration = np.mean(dur_list)
                emoji = get_regime_emoji(regime)
                st.write(f"{emoji} {regime.replace('_', ' ').title()}: {avg_duration:.1f} periods")
