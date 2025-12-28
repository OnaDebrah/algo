"""
Portfolio Dashboard UI Component
"""

import pandas as pd
import streamlit as st

from core.database import DatabaseManager


def render_dashboard(db: DatabaseManager):
    """
    Render the portfolio dashboard tab

    Args:
        db: Database manager instance
    """
    st.header("Portfolio Overview")

    st.subheader("Performance Metrics")

    recent_trades = db.get_trades(limit=100)

    if recent_trades:
        trades_df = pd.DataFrame(recent_trades)
        completed_trades = trades_df[trades_df["profit"].notna()]

        if not completed_trades.empty:
            # Calculate metrics
            total_profit = completed_trades["profit"].sum()
            win_rate = (completed_trades["profit"] > 0).sum() / len(completed_trades) * 100
            avg_profit = completed_trades["profit"].mean()
            total_trades = len(completed_trades)

            # Display metrics in columns
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Total P&L",
                    f"${total_profit:.2f}",
                    delta=f"{total_profit:.2f}",
                    delta_color="normal" if total_profit >= 0 else "inverse",
                )

            with col2:
                st.metric("Win Rate", f"{win_rate:.1f}%")

            with col3:
                st.metric("Avg Profit", f"${avg_profit:.2f}")

            with col4:
                st.metric("Total Trades", total_trades)

            # Additional metrics row
            col1, col2, col3, col4 = st.columns(4)

            winning_trades = completed_trades[completed_trades["profit"] > 0]
            losing_trades = completed_trades[completed_trades["profit"] < 0]

            with col1:
                st.metric("Winning Trades", len(winning_trades))

            with col2:
                st.metric("Losing Trades", len(losing_trades))

            with col3:
                avg_win = winning_trades["profit"].mean() if len(winning_trades) > 0 else 0
                st.metric("Avg Win", f"${avg_win:.2f}")

            with col4:
                avg_loss = losing_trades["profit"].mean() if len(losing_trades) > 0 else 0
                st.metric("Avg Loss", f"${avg_loss:.2f}")

            # Recent Trades Table
            st.subheader("Recent Trades")

            display_df = (
                completed_trades[
                    [
                        "timestamp",
                        "symbol",
                        "order_type",
                        "quantity",
                        "price",
                        "profit",
                        "profit_pct",
                    ]
                ]
                .tail(20)
                .copy()
            )

            # Format columns for better display
            display_df["price"] = display_df["price"].apply(lambda x: f"${x:.2f}")
            display_df["profit"] = display_df["profit"].apply(lambda x: f"${x:.2f}")
            display_df["profit_pct"] = display_df["profit_pct"].apply(lambda x: f"{x:.2f}%")

            # Rename columns for display
            display_df.columns = [
                "Timestamp",
                "Symbol",
                "Type",
                "Quantity",
                "Price",
                "Profit",
                "Profit %",
            ]

            st.dataframe(display_df, use_container_width=True)

            # Strategy Performance
            st.subheader("Strategy Performance")

            strategy_stats = (
                completed_trades.groupby("strategy")
                .agg(
                    {
                        "profit": ["sum", "mean", "count"],
                    }
                )
                .round(2)
            )

            strategy_stats.columns = ["Total Profit", "Avg Profit", "Trade Count"]
            strategy_stats = strategy_stats.sort_values("Total Profit", ascending=False)

            st.dataframe(strategy_stats, use_container_width=True)

            # Symbol Performance
            st.subheader("Symbol Performance")

            symbol_stats = (
                completed_trades.groupby("symbol")
                .agg(
                    {
                        "profit": ["sum", "mean", "count"],
                    }
                )
                .round(2)
            )

            symbol_stats.columns = ["Total Profit", "Avg Profit", "Trade Count"]
            symbol_stats = symbol_stats.sort_values("Total Profit", ascending=False)

            st.dataframe(symbol_stats, use_container_width=True)

        else:
            st.info("📊 No completed trades yet. Run a backtest to see performance metrics!")
            st.markdown(
                """
            **Get Started:**
            1. Go to the **Backtest** tab
            2. Select a symbol and strategy
            3. Run your first backtes
            4. Come back here to see your results!
            """
            )
    else:
        st.info("📈 No trades recorded. Start your first backtest to see metrics here!")
        st.markdown(
            """
        **Welcome to your Trading Dashboard!**

        This dashboard will display:
        - Portfolio performance metrics
        - Recent trade history
        - Strategy performance breakdown
        - Symbol performance analysis

        **To get started:**
        1. Navigate to the **Backtest** tab
        2. Configure and run a backtes
        3. Return here to analyze your results
        """
        )

    # Export functionality
    st.divider()

    if recent_trades:
        col1, col2 = st.columns([1, 3])

        with col1:
            if st.button("📥 Export All Trades"):
                trades_export_df = pd.DataFrame(recent_trades)
                csv = trades_export_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download CSV",
                    data=csv,
                    file_name=f"trades_export_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
