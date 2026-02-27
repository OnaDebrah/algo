"""
Interactive visualization dashboards for crash prediction
"""

import logging
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class CrashPredictionDashboard:
    """
    Interactive dashboard for crash predictions and stress indicators
    """

    def __init__(self):
        self.colors = {
            "background": "#1f2630",
            "text": "#ffffff",
            "grid": "#2e3a47",
            "crash": "#ff4d4d",
            "bubble": "#ffa64d",
            "stress": "#4da6ff",
            "safe": "#4dff4d",
        }

    def create_dashboard(
        self, price_data: pd.DataFrame, predictions: Dict[str, pd.Series], stress_data: Optional[Dict] = None, bubble_data: Optional[Dict] = None
    ) -> go.Figure:
        """
        Create comprehensive dashboard
        """
        fig = make_subplots(
            rows=4,
            cols=2,
            subplot_titles=(
                "Price with Crash Predictions",
                "Crash Probability Timeline",
                "LPPLS Bubble Detection",
                "Market Stress Index",
                "Feature Importance",
                "Model Confidence",
                "Risk Heatmap",
                "Alert History",
            ),
            specs=[[{"colspan": 2}, None], [{}, {}], [{}, {}], [{}, {}]],
            vertical_spacing=0.08,
            horizontal_spacing=0.1,
        )

        # 1. Price with crash predictions (top row, full width)
        self._add_price_chart(fig, price_data, predictions, row=1, col=1)

        # 2. Crash probability timeline
        self._add_probability_chart(fig, predictions, row=2, col=1)

        # 3. LPPLS bubble detection
        self._add_bubble_chart(fig, bubble_data, row=2, col=2)

        # 4. Market stress index
        self._add_stress_chart(fig, stress_data, row=3, col=1)

        # 5. Feature importance
        self._add_feature_importance(fig, row=3, col=2)

        # 6. Model confidence
        self._add_confidence_chart(fig, predictions, row=4, col=1)

        # 7. Risk heatmap
        self._add_risk_heatmap(fig, price_data, predictions, row=4, col=2)

        # Update layout
        fig.update_layout(title_text="Market Crash Prediction Dashboard", template="plotly_dark", showlegend=True, height=1200, hovermode="x unified")

        return fig

    def _add_price_chart(self, fig: go.Figure, price_data: pd.DataFrame, predictions: Dict[str, pd.Series], row: int, col: int):
        """Add price chart with crash markers"""

        # Price line
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data["Close"],
                name="Price",
                line=dict(color="white", width=2),
                hovertemplate="Date: %{x}<br>Price: $%{y:.2f}<extra></extra>",
            ),
            row=row,
            col=col,
        )

        # Add moving averages
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data["Close"].rolling(50).mean(),
                name="MA(50)",
                line=dict(color="yellow", width=1, dash="dash"),
                opacity=0.7,
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data["Close"].rolling(200).mean(),
                name="MA(200)",
                line=dict(color="orange", width=1, dash="dash"),
                opacity=0.7,
            ),
            row=row,
            col=col,
        )

        # Add crash prediction markers
        if "ensemble" in predictions:
            crash_dates = predictions["ensemble"][predictions["ensemble"] >= 0.7].index
            crash_probs = predictions["ensemble"][predictions["ensemble"] >= 0.7]

            fig.add_trace(
                go.Scatter(
                    x=crash_dates,
                    y=price_data.loc[crash_dates, "Close"],
                    mode="markers",
                    name="High Crash Risk",
                    marker=dict(
                        symbol="triangle-down",
                        size=12,
                        color=crash_probs,
                        colorscale="Reds",
                        showscale=True,
                        colorbar=dict(title="Crash<br>Probability", x=1.02, y=0.9),
                    ),
                    hovertemplate="Date: %{x}<br>Price: $%{y:.2f}<br>Risk: %{marker.color:.1%}<extra></extra>",
                ),
                row=row,
                col=col,
            )

    def _add_probability_chart(self, fig: go.Figure, predictions: Dict[str, pd.Series], row: int, col: int):
        """Add crash probability timeline"""

        if "ensemble" in predictions:
            proba = predictions["ensemble"]

            # Color based on probability
            colors = ["green" if p < 0.33 else "orange" if p < 0.66 else "red" for p in proba]

            fig.add_trace(
                go.Bar(
                    x=proba.index,
                    y=proba,
                    name="Crash Probability",
                    marker_color=colors,
                    hovertemplate="Date: %{x}<br>Probability: %{y:.1%}<extra></extra>",
                ),
                row=row,
                col=col,
            )

            # Add threshold lines
            fig.add_hline(y=0.33, line_dash="dash", line_color="orange", annotation_text="Warning Threshold", row=row, col=col)

            fig.add_hline(y=0.66, line_dash="dash", line_color="red", annotation_text="Critical Threshold", row=row, col=col)

    def _add_bubble_chart(self, fig: go.Figure, bubble_data: Optional[Dict], row: int, col: int):
        """Add LPPLS bubble detection chart"""

        if bubble_data and "confidence" in bubble_data:
            # Create bubble indicator
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=bubble_data.get("confidence", 0) * 100,
                    title={"text": "Bubble Confidence"},
                    delta={"reference": 50},
                    gauge={
                        "axis": {"range": [None, 100]},
                        "bar": {"color": "darkblue"},
                        "steps": [{"range": [0, 33], "color": "green"}, {"range": [33, 66], "color": "yellow"}, {"range": [66, 100], "color": "red"}],
                        "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 80},
                    },
                ),
                row=row,
                col=col,
            )

            # Add bubble parameters
            params = bubble_data.get("parameters", {})
            fig.add_annotation(
                text=f"m: {params.get('m', 0):.3f}<br>" f"ω: {params.get('omega', 0):.1f}<br>" f"tc: {params.get('tc_days_ahead', 0):.0f} days",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.3,
                showarrow=False,
                font=dict(size=12),
                row=row,
                col=col,
            )

    def _add_stress_chart(self, fig: go.Figure, stress_data: Optional[Dict], row: int, col: int):
        """Add market stress index chart"""

        if stress_data and "history" in stress_data:
            history = stress_data["history"]

            fig.add_trace(
                go.Scatter(
                    x=history.index,
                    y=history["stress"],
                    name="Stress Index",
                    line=dict(color="red", width=2),
                    fill="tozeroy",
                    hovertemplate="Date: %{x}<br>Stress: %{y:.2%}<extra></extra>",
                ),
                row=row,
                col=col,
            )

            # Add confidence bands
            fig.add_trace(
                go.Scatter(
                    x=history.index, y=history["upper_bound"], name="Upper Bound", line=dict(color="rgba(255,0,0,0.2)", width=0), showlegend=False
                ),
                row=row,
                col=col,
            )

            fig.add_trace(
                go.Scatter(
                    x=history.index,
                    y=history["lower_bound"],
                    name="Lower Bound",
                    fill="tonexty",
                    fillcolor="rgba(255,0,0,0.1)",
                    line=dict(color="rgba(255,0,0,0.2)", width=0),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

    def _add_feature_importance(self, fig: go.Figure, row: int, col: int):
        """Add feature importance horizontal bar chart"""

        # Sample feature importance - would come from model
        features = {
            "Volatility Expansion": 0.15,
            "LPPLS Confidence": 0.12,
            "RSI": 0.11,
            "LSTM Stress": 0.10,
            "Price vs MA200": 0.09,
            "Volume Ratio": 0.08,
            "Bubble Detection": 0.07,
            "ML Consensus": 0.06,
            "Put/Call Ratio": 0.05,
            "VIX Level": 0.04,
        }

        df = pd.DataFrame(list(features.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)

        fig.add_trace(
            go.Bar(
                x=df["Importance"],
                y=df["Feature"],
                orientation="h",
                name="Feature Importance",
                marker=dict(color=df["Importance"], colorscale="Viridis", showscale=True, colorbar=dict(title="Importance", x=1.02, y=0.4)),
            ),
            row=row,
            col=col,
        )

    def _add_confidence_chart(self, fig: go.Figure, predictions: Dict[str, pd.Series], row: int, col: int):
        """Add model confidence over time"""

        if "confidence" in predictions:
            confidence = predictions["confidence"]

            fig.add_trace(
                go.Scatter(
                    x=confidence.index,
                    y=confidence,
                    name="Model Confidence",
                    line=dict(color="lightblue", width=2),
                    fill="tozeroy",
                    hovertemplate="Date: %{x}<br>Confidence: %{y:.1%}<extra></extra>",
                ),
                row=row,
                col=col,
            )

    def _add_risk_heatmap(self, fig: go.Figure, price_data: pd.DataFrame, predictions: Dict[str, pd.Series], row: int, col: int):
        """Add risk heatmap combining multiple factors"""

        # Create risk matrix
        if len(price_data) > 60:
            # Calculate various risk metrics
            returns = price_data["Close"].pct_change()

            risk_matrix = pd.DataFrame(
                {
                    "Volatility": returns.rolling(20).std(),
                    "Drawdown": (price_data["Close"] / price_data["Close"].rolling(60).max() - 1).abs(),
                    "Volume_Spike": price_data["Volume"] / price_data["Volume"].rolling(20).mean(),
                    "Crash_Prob": predictions.get("ensemble", pd.Series(0, index=price_data.index)),
                }
            ).tail(60)

            # Create heatmap
            fig.add_trace(
                go.Heatmap(
                    z=risk_matrix.T.values,
                    x=risk_matrix.index,
                    y=risk_matrix.columns,
                    colorscale="RdYlGn_r",
                    name="Risk Heatmap",
                    colorbar=dict(title="Risk Level", x=1.02, y=0.1),
                    hovertemplate="Date: %{x}<br>Metric: %{y}<br>Value: %{z:.3f}<extra></extra>",
                ),
                row=row,
                col=col,
            )

    def create_alert_dashboard(self, alerts: List[Dict]) -> go.Figure:
        """Create alerts dashboard"""

        fig = make_subplots(
            rows=2, cols=2, subplot_titles=("Alert Timeline", "Alert Severity Distribution", "Recent Alerts", "Response Success Rate")
        )

        if alerts:
            df = pd.DataFrame(alerts)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Alert timeline
            colors = {"LOW": "green", "MODERATE": "orange", "HIGH": "red", "CRITICAL": "darkred"}

            for severity in df["severity"].unique():
                mask = df["severity"] == severity
                fig.add_trace(
                    go.Scatter(
                        x=df[mask]["timestamp"],
                        y=[severity] * sum(mask),
                        mode="markers",
                        name=severity,
                        marker=dict(color=colors.get(severity, "blue"), size=10),
                        hovertemplate="Date: %{x}<br>Message: %{text}<extra></extra>",
                        text=df[mask]["message"],
                    ),
                    row=1,
                    col=1,
                )

            # Severity distribution
            severity_counts = df["severity"].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=severity_counts.index,
                    values=severity_counts.values,
                    marker=dict(colors=[colors.get(s, "blue") for s in severity_counts.index]),
                ),
                row=1,
                col=2,
            )

            # Recent alerts table
            recent = df.tail(5)
            fig.add_trace(
                go.Table(
                    header=dict(values=["Date", "Severity", "Message"]),
                    cells=dict(values=[recent["timestamp"].dt.strftime("%Y-%m-%d %H:%M"), recent["severity"], recent["message"]]),
                ),
                row=2,
                col=1,
            )

            # Response success rate
            if "success" in df.columns:
                success_rate = df["success"].mean()

                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=success_rate * 100,
                        title={"text": "Alert Success Rate"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "green"},
                            "steps": [
                                {"range": [0, 50], "color": "red"},
                                {"range": [50, 75], "color": "yellow"},
                                {"range": [75, 100], "color": "green"},
                            ],
                        },
                    ),
                    row=2,
                    col=2,
                )

        fig.update_layout(title_text="Alert System Dashboard", template="plotly_dark", height=800, showlegend=True)

        return fig
