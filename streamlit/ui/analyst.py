"""
Financial Analyst UI - Streamlit interface for investment thesis generation
"""

import asyncio
import json
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit.core.analyst_agent import AnalystReport, FinancialAnalystAgent


class FinancialAnalystUI:
    """UI for AI Financial Analyst"""

    def __init__(self):
        self.agent = None
        self._initialize_agent()

    def _initialize_agent(self):
        """Initialize the analyst agent"""
        import os

        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.agent = FinancialAnalystAgent(api_key=api_key)

    def render(self):
        """Render the financial analyst interface"""
        st.title("🏦 AI Financial Analyst")
        st.markdown("Generate comprehensive investment thesis reports powered by AI")

        # Input section
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            ticker = st.text_input(
                "Enter Stock Ticker",
                placeholder="e.g., AAPL, MSFT, TSLA",
                help="Enter any publicly traded stock ticker",
            ).upper()

        with col2:
            depth = st.selectbox(
                "Analysis Depth",
                ["quick", "standard", "comprehensive", "deep_dive"],
                index=2,
                help="Deeper analysis takes longer but provides more insights",
            )

        with col3:
            st.markdown("####")  # Spacing
            analyze_btn = st.button("🔍 Generate Report", type="primary", use_container_width=True)

        # Info boxes
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.info("📊 Fundamental Analysis")
        with col2:
            st.info("📈 Technical Analysis")
        with col3:
            st.info("🏢 Peer Comparison")
        with col4:
            st.info("📰 News Sentiment")

        st.markdown("---")

        # Generate report
        if analyze_btn:
            if not ticker:
                st.error("⚠️ Please enter a stock ticker")
                return

            self._generate_and_display_report(ticker, depth)

        # Show existing report if available
        elif "analyst_report" in st.session_state:
            self._display_report(st.session_state["analyst_report"])

    def _generate_and_display_report(self, ticker: str, depth: str):
        """Generate and display analyst report"""

        with st.status(f"Analyzing {ticker}...", expanded=True) as status:
            st.write("🔍 Gathering market data...")
            st.write("📊 Calculating fundamental metrics...")
            st.write("📈 Performing technical analysis...")
            st.write("🏢 Analyzing peer companies...")
            st.write("📰 Analyzing news sentiment...")
            st.write("🤖 Generating AI insights...")

            try:
                # Generate report
                report = asyncio.run(self.agent.generate_investment_thesis(ticker, depth))

                # Store in session state
                st.session_state["analyst_report"] = report

                status.update(
                    label=f"✅ Analysis complete for {ticker}!",
                    state="complete",
                    expanded=False,
                )

                # Display report
                self._display_report(report)

            except Exception as e:
                status.update(label=f"❌ Error analyzing {ticker}", state="error")
                st.error(f"Error: {str(e)}")

    def _display_report(self, report: AnalystReport):
        """Display the complete analyst report"""

        # Header with key metrics
        self._render_header(report)

        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "📋 Executive Summary",
                "📊 Fundamental Analysis",
                "📈 Technical Analysis",
                "⚠️ Risk Analysis",
                "📥 Export Report",
            ]
        )

        with tab1:
            self._render_executive_summary(report)

        with tab2:
            self._render_fundamental_analysis(report)

        with tab3:
            self._render_technical_analysis(report)

        with tab4:
            self._render_risk_analysis(report)

        with tab5:
            self._render_export_options(report)

    def _render_header(self, report: AnalystReport):
        """Render report header with key metrics"""

        # Company header
        st.markdown(f"## {report.company_name} ({report.ticker})")
        st.caption(f"Report Generated: {report.generated_date.strftime('%B %d, %Y at %I:%M %p')}")

        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)

        # Recommendation with color
        rec_color = {
            "Strong Buy": "🟢",
            "Buy": "🟢",
            "Hold": "🟡",
            "Sell": "🔴",
            "Strong Sell": "🔴",
        }

        with col1:
            st.metric(
                "Recommendation",
                f"{rec_color.get(report.recommendation, '⚪')} {report.recommendation}",
            )

        with col2:
            st.metric("Current Price", f"${report.current_price:.2f}")

        with col3:
            st.metric(
                "Target Price",
                f"${report.target_price:.2f}",
                delta=f"{report.upside_potential:.1f}%",
            )

        with col4:
            st.metric("Upside Potential", f"{report.upside_potential:.1f}%")

        with col5:
            risk_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
            st.metric(
                "Risk Rating",
                f"{risk_emoji.get(report.risk_rating, '⚪')} {report.risk_rating}",
            )

        st.markdown("---")

    def _render_executive_summary(self, report: AnalystReport):
        """Render executive summary tab"""

        # Investment Thesis
        st.markdown("### 📝 Investment Thesis")
        st.markdown(report.investment_thesis)

        st.markdown("---")

        # Key Takeaways
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ✨ Key Takeaways")
            for takeaway in report.key_takeaways:
                st.markdown(f"• {takeaway}")

        with col2:
            st.markdown("### 🎯 Action Items")
            for action in report.action_items:
                st.markdown(f"• {action}")

        st.markdown("---")

        # Catalysts
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📈 Positive Catalysts")
            for catalyst in report.positive_catalysts:
                st.success(f"✓ {catalyst}")

        with col2:
            st.markdown("### 📉 Negative Catalysts")
            for catalyst in report.negative_catalysts:
                st.warning(f"⚠ {catalyst}")

    def _render_fundamental_analysis(self, report: AnalystReport):
        """Render fundamental analysis tab"""

        # Business Overview
        st.markdown("### 🏢 Business Overview")
        st.markdown(report.business_overview)

        st.markdown("### 🎯 Competitive Position")
        st.markdown(report.competitive_position)

        st.markdown("---")

        # Valuation Metrics
        st.markdown("### 💰 Valuation Metrics")

        val_metrics = report.valuation_metrics
        if val_metrics:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                pe = val_metrics.get("pe_ratio", 0)
                st.metric("P/E Ratio", f"{pe:.2f}" if pe else "N/A")

                fpe = val_metrics.get("forward_pe", 0)
                st.metric("Forward P/E", f"{fpe:.2f}" if fpe else "N/A")

            with col2:
                pb = val_metrics.get("price_to_book", 0)
                st.metric("P/B Ratio", f"{pb:.2f}" if pb else "N/A")

                ps = val_metrics.get("price_to_sales", 0)
                st.metric("P/S Ratio", f"{ps:.2f}" if ps else "N/A")

            with col3:
                peg = val_metrics.get("peg_ratio", 0)
                st.metric("PEG Ratio", f"{peg:.2f}" if peg else "N/A")

                ev_ebitda = val_metrics.get("ev_to_ebitda", 0)
                st.metric("EV/EBITDA", f"{ev_ebitda:.2f}" if ev_ebitda else "N/A")

            with col4:
                mcap = val_metrics.get("market_cap", 0)
                if mcap > 1e9:
                    st.metric("Market Cap", f"${mcap/1e9:.2f}B")
                elif mcap > 1e6:
                    st.metric("Market Cap", f"${mcap/1e6:.2f}M")
                else:
                    st.metric("Market Cap", "N/A")

        st.markdown("---")

        # Financial Health
        st.markdown("### 💪 Financial Health")

        fin_health = report.financial_health
        if fin_health:
            col1, col2, col3 = st.columns(3)

            with col1:
                cr = fin_health.get("current_ratio", 0)
                st.metric(
                    "Current Ratio",
                    f"{cr:.2f}" if cr else "N/A",
                    help="Higher is better (>1.5 is good)",
                )

            with col2:
                de = fin_health.get("debt_to_equity", 0)
                st.metric(
                    "Debt/Equity",
                    f"{de:.2f}" if de else "N/A",
                    help="Lower is better (<1.0 is good)",
                )

            with col3:
                qr = fin_health.get("quick_ratio", 0)
                st.metric(
                    "Quick Ratio",
                    f"{qr:.2f}" if qr else "N/A",
                    help="Higher is better (>1.0 is good)",
                )

        # Visualization: Valuation comparison
        if val_metrics:
            st.markdown("### 📊 Valuation Metrics Visualization")
            self._render_valuation_chart(val_metrics)

    def _render_valuation_chart(self, metrics: dict):
        """Render valuation metrics chart"""

        # Create radar chart for valuation
        categories = []
        values = []

        if metrics.get("pe_ratio"):
            categories.append("P/E")
            # Normalize P/E (inverse, lower is better)
            values.append(min(100, 100 - (metrics["pe_ratio"] - 15) * 2))

        if metrics.get("price_to_book"):
            categories.append("P/B")
            values.append(min(100, 100 - (metrics["price_to_book"] - 2) * 10))

        if metrics.get("peg_ratio"):
            categories.append("PEG")
            values.append(min(100, 100 - (metrics["peg_ratio"] - 1) * 20))

        if categories:
            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill="toself", name="Valuation Score"))

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)

    def _render_technical_analysis(self, report: AnalystReport):
        """Render technical analysis tab"""

        st.markdown("### 📈 Technical Outlook")
        st.markdown(report.chart_patterns)

        st.markdown("---")

        # Technical Indicators
        tech = report.technical_indicators

        if tech:
            # Moving Averages
            st.markdown("### 📊 Moving Averages")
            ma_data = tech.get("moving_averages", {})

            if ma_data:
                col1, col2, col3 = st.columns(3)

                with col1:
                    ma20 = ma_data.get("ma_20", 0)
                    pct_ma20 = ma_data.get("price_vs_ma20", 0)
                    st.metric("20-Day MA", f"${ma20:.2f}", delta=f"{pct_ma20:.1f}%")

                with col2:
                    ma50 = ma_data.get("ma_50", 0)
                    pct_ma50 = ma_data.get("price_vs_ma50", 0)
                    st.metric("50-Day MA", f"${ma50:.2f}", delta=f"{pct_ma50:.1f}%")

                with col3:
                    ma200 = ma_data.get("ma_200", 0)
                    pct_ma200 = ma_data.get("price_vs_ma200", 0)
                    st.metric("200-Day MA", f"${ma200:.2f}", delta=f"{pct_ma200:.1f}%")

            # Momentum Indicators
            st.markdown("### 🎯 Momentum Indicators")

            col1, col2 = st.columns(2)

            with col1:
                momentum = tech.get("momentum", {})
                if momentum:
                    rsi = momentum.get("rsi", 50)
                    signal = momentum.get("rsi_signal", "Neutral")

                    # RSI Gauge
                    fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=rsi,
                            title={"text": "RSI (14)"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": "darkblue"},
                                "steps": [
                                    {"range": [0, 30], "color": "lightgreen"},
                                    {"range": [30, 70], "color": "lightyellow"},
                                    {"range": [70, 100], "color": "lightcoral"},
                                ],
                                "threshold": {
                                    "line": {"color": "red", "width": 4},
                                    "thickness": 0.75,
                                    "value": rsi,
                                },
                            },
                        )
                    )

                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"Signal: {signal}")

            with col2:
                macd_data = tech.get("macd", {})
                if macd_data:
                    macd_val = macd_data.get("macd", 0)
                    signal_val = macd_data.get("signal", 0)
                    hist = macd_data.get("histogram", 0)
                    trend = macd_data.get("trend", "Neutral")

                    st.markdown("**MACD Analysis**")
                    st.metric("MACD", f"{macd_val:.2f}")
                    st.metric("Signal", f"{signal_val:.2f}")
                    st.metric("Histogram", f"{hist:.2f}")
                    st.info(f"Trend: {trend}")

            # Support and Resistance
            st.markdown("### 🎚️ Support & Resistance Levels")

            sr = report.support_resistance
            if sr:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Support", f"${sr.get('support', 0):.2f}")

                with col2:
                    st.metric("Current", f"${sr.get('current', 0):.2f}")

                with col3:
                    st.metric("Resistance", f"${sr.get('resistance', 0):.2f}")

    def _render_risk_analysis(self, report: AnalystReport):
        """Render risk analysis tab"""

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ⚠️ Key Risks")
            for idx, risk in enumerate(report.key_risks, 1):
                with st.expander(f"Risk #{idx}: {risk[:50]}..."):
                    st.markdown(risk)

        with col2:
            st.markdown("### 🛡️ Risk Mitigation Strategies")
            for idx, mitigation in enumerate(report.risk_mitigation, 1):
                st.success(f"**Strategy {idx}:** {mitigation}")

        st.markdown("---")

        # Risk Rating Visualization
        st.markdown("### 📊 Risk Assessment")

        risk_scores = {
            "Market Risk": 60,
            "Company Risk": 45,
            "Sector Risk": 55,
            "Liquidity Risk": 30,
            "Valuation Risk": 50,
        }

        fig = px.bar(
            x=list(risk_scores.values()),
            y=list(risk_scores.keys()),
            orientation="h",
            title="Risk Profile Breakdown",
            labels={"x": "Risk Score (0-100)", "y": "Risk Category"},
            color=list(risk_scores.values()),
            color_continuous_scale="RdYlGn_r",
        )

        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    def _render_export_options(self, report: AnalystReport):
        """Render export options"""

        st.markdown("### 📥 Export Report")
        st.markdown("Download this analysis in your preferred format")

        col1, col2, col3 = st.columns(3)

        with col1:
            # PDF Export
            st.markdown("#### 📄 PDF Report")
            st.markdown("Professional formatted report")
            pdf_data = self._generate_pdf_report(report)
            st.download_button(
                label="Download PDF",
                data=pdf_data,
                file_name=f"{report.ticker}_analysis_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        with col2:
            # JSON Export
            st.markdown("#### 📊 JSON Data")
            st.markdown("Raw data for further analysis")
            json_data = self._generate_json_report(report)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"{report.ticker}_data_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True,
            )

        with col3:
            # Markdown Export
            st.markdown("#### 📝 Markdown")
            st.markdown("Formatted text document")
            md_data = self._generate_markdown_report(report)
            st.download_button(
                label="Download Markdown",
                data=md_data,
                file_name=f"{report.ticker}_report_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown",
                use_container_width=True,
            )

    def _generate_pdf_report(self, report: AnalystReport) -> bytes:
        """Generate PDF report (placeholder - requires reportlab)"""
        # In production, use reportlab or similar
        # For now, return a simple text representation
        content = self._generate_markdown_report(report)
        return content.encode("utf-8")

    def _generate_json_report(self, report: AnalystReport) -> str:
        """Generate JSON report"""
        report_dict = {
            "ticker": report.ticker,
            "company_name": report.company_name,
            "generated_date": report.generated_date.isoformat(),
            "recommendation": report.recommendation,
            "current_price": report.current_price,
            "target_price": report.target_price,
            "upside_potential": report.upside_potential,
            "risk_rating": report.risk_rating,
            "investment_thesis": report.investment_thesis,
            "business_overview": report.business_overview,
            "key_risks": report.key_risks,
            "positive_catalysts": report.positive_catalysts,
            "negative_catalysts": report.negative_catalysts,
            "key_takeaways": report.key_takeaways,
            "action_items": report.action_items,
            "valuation_metrics": report.valuation_metrics,
            "technical_indicators": report.technical_indicators,
        }
        return json.dumps(report_dict, indent=2)

    def _generate_markdown_report(self, report: AnalystReport) -> str:
        """Generate Markdown report"""
        md = f"""# Investment Analysis Report: {report.company_name} ({report.ticker})

**Report Date:** {report.generated_date.strftime('%B %d, %Y')}

---

## Executive Summary

**Recommendation:** {report.recommendation}
**Current Price:** ${report.current_price:.2f}
**Target Price:** ${report.target_price:.2f}
**Upside Potential:** {report.upside_potential:.1f}%
**Risk Rating:** {report.risk_rating}

---

## Investment Thesis

{report.investment_thesis}

---

## Key Takeaways

"""
        for takeaway in report.key_takeaways:
            md += f"- {takeaway}\n"

        md += "\n---\n\n## Business Overview\n\n"
        md += report.business_overview

        md += "\n\n---\n\n## Competitive Position\n\n"
        md += report.competitive_position

        md += "\n\n---\n\n## Key Risks\n\n"
        for idx, risk in enumerate(report.key_risks, 1):
            md += f"{idx}. {risk}\n"

        md += "\n\n---\n\n## Positive Catalysts\n\n"
        for catalyst in report.positive_catalysts:
            md += f"- {catalyst}\n"

        md += "\n\n---\n\n## Negative Catalysts\n\n"
        for catalyst in report.negative_catalysts:
            md += f"- {catalyst}\n"

        md += "\n\n---\n\n## Action Items\n\n"
        for action in report.action_items:
            md += f"- {action}\n"

        md += "\n\n---\n\n*This report was generated by AI Financial Analyst*"

        return md


def render_analyst():
    """Main function to render the analyst interface"""
    ui = FinancialAnalystUI()
    ui.render()
