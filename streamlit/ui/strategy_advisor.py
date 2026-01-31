"""
Strategy Advisor UI - Streamlit interface for AI recommendations
"""

import asyncio

import plotly.graph_objects as go

import streamlit as st
from streamlit.core.ai_advisor import AIStrategyAdvisor, StrategyRecommendation
from streamlit.core.ai_advisor_api import AIAdvisorAPI


class StrategyAdvisorUI:
    """UI for AI Strategy Recommendations"""

    def __init__(self, ai_advisor: AIAdvisorAPI):
        self.advisor = AIStrategyAdvisor(ai_advisor)

    def render(self):
        """Render the strategy advisor interface"""
        # st.title("🤖 AI Strategy Advisor")
        # st.markdown("Get personalized strategy recommendations based on your goals and preferences")

        # Check if user has completed questionnaire
        if "user_profile" not in st.session_state:
            self._render_questionnaire()
        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.success("✓ Profile Complete")
            with col2:
                if st.button("↻ Retake Quiz"):
                    del st.session_state["user_profile"]
                    st.rerun()

            self._render_recommendations()

    def _render_questionnaire(self):
        """Render the user profile questionnaire"""
        st.markdown("### 📋 Tell Us About Your Trading Goals")
        st.markdown("Answer a few questions to get personalized strategy recommendations")

        with st.form("profile_form"):
            st.markdown("#### 1. Investment Goals")
            goals = st.multiselect(
                "What are your primary goals? (Select all that apply)",
                [
                    "Steady income generation",
                    "Long-term wealth building",
                    "Short-term profits",
                    "Learning and skill development",
                    "Portfolio diversification",
                ],
                help="Your goals help us match you with appropriate strategies",
            )

            st.markdown("#### 2. Risk Tolerance")
            risk_tolerance = st.select_slider(
                "How much risk are you comfortable with?",
                options=["Very Low", "Low", "Medium", "High", "Very High"],
                value="Medium",
                help="Higher risk strategies offer higher potential returns but also larger potential losses",
            )

            st.markdown("#### 3. Time Horizon")
            time_horizon = st.radio(
                "What's your investment timeframe?",
                [
                    "Short-term (< 1 year)",
                    "Medium-term (1-3 years)",
                    "Long-term (3+ years)",
                ],
                help="Different strategies work better for different timeframes",
            )

            st.markdown("#### 4. Experience Level")
            experience = st.select_slider(
                "What's your trading experience?",
                options=["Beginner", "Intermediate", "Advanced", "Expert"],
                value="Intermediate",
            )

            st.markdown("#### 5. Time Commitment")
            time_commitment = st.radio(
                "How much time can you dedicate to trading?",
                [
                    "Minimal (set and forget)",
                    "Low (check weekly)",
                    "Medium (check daily)",
                    "High (actively manage)",
                ],
            )

            st.markdown("#### 6. Starting Capital")
            capital = st.number_input(
                "Starting capital ($)",
                min_value=1000,
                max_value=1000000,
                value=10000,
                step=1000,
                help="Your starting capital affects which strategies are most suitable",
            )

            st.markdown("#### 7. Market Preference")
            market_preference = st.selectbox(
                "Do you prefer any specific market conditions?",
                [
                    "Any (I want the AI to decide)",
                    "Trending markets",
                    "Volatile markets",
                    "Stable markets",
                ],
            )

            submitted = st.form_submit_button("🚀 Get My Recommendations", use_container_width=True)

            if submitted:
                if not goals:
                    st.error("Please select at least one investment goal")
                    return

                # Store profile
                profile = {
                    "goals": ", ".join(goals),
                    "risk_tolerance": risk_tolerance,
                    "time_horizon": time_horizon,
                    "experience": experience,
                    "time_commitment": time_commitment,
                    "capital": capital,
                    "market_preference": market_preference,
                }

                st.session_state["user_profile"] = profile
                st.success("✓ Profile saved! Generating recommendations...")
                st.rerun()

    def _render_recommendations(self):
        """Render AI-generated recommendations"""
        st.markdown("### 🎯 Your Personalized Strategy Recommendations")

        # Get recommendations
        if "recommendations" not in st.session_state:
            with st.spinner("🤖 AI is analyzing your profile and generating recommendations..."):
                profile = st.session_state["user_profile"]
                recommendations = asyncio.run(self.advisor.get_recommendations(profile))
                st.session_state["recommendations"] = recommendations

        recommendations = st.session_state["recommendations"]

        if not recommendations:
            st.error("Unable to generate recommendations. Please try again.")
            return

        # Display profile summary
        with st.expander("📊 Your Profile Summary", expanded=False):
            profile = st.session_state["user_profile"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Risk Tolerance", profile["risk_tolerance"])
                st.metric("Experience", profile["experience"])
            with col2:
                st.metric("Time Horizon", profile["time_horizon"])
                st.metric("Capital", f"${profile['capital']:,}")
            with col3:
                st.metric("Time Commitment", profile["time_commitment"])

        # Display recommendations
        st.markdown("---")

        for idx, rec in enumerate(recommendations, 1):
            self._render_recommendation_card(rec, idx)
            st.markdown("---")

        # Show comparison chart
        self._render_comparison_chart(recommendations)

        # Action buttons
        st.markdown("### 🎬 Next Steps")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📈 Backtest Top Strategy", use_container_width=True):
                st.session_state["selected_strategy"] = recommendations[0].name
                st.session_state["page"] = "backtest"
                st.rerun()

        with col2:
            if st.button("🔧 Customize Strategy", use_container_width=True):
                st.info("Navigate to Configuration to customize parameters")

        with col3:
            if st.button("📚 Learn More", use_container_width=True):
                st.session_state["show_education"] = True
                st.rerun()

    def _render_recommendation_card(self, rec: StrategyRecommendation, rank: int):
        """Render a single recommendation card"""
        # Header with rank and fit score
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### {rank}. {rec.name}")
            st.markdown(f"*{rec.description}*")
        with col2:
            # Fit score gauge
            score_color = "green" if rec.fit_score >= 80 else "orange" if rec.fit_score >= 60 else "red"
            st.markdown(
                f"<div style='text-align: center;'>"
                f"<h2 style='color: {score_color};'>{rec.fit_score}%</h2>"
                f"<p style='color: gray; font-size: 12px;'>FIT SCORE</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Risk and return info
        col1, col2, col3 = st.columns(3)
        with col1:
            risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}
            st.metric(
                "Risk Level",
                f"{risk_emoji.get(rec.risk_level, '⚪')} {rec.risk_level.title()}",
            )
        with col2:
            st.metric(
                "Expected Return",
                f"{rec.expected_return_range[0]}-{rec.expected_return_range[1]}%",
            )
        with col3:
            # Extract percentage from similar traders text if available
            st.metric("Similar Traders", "35%", help=rec.similar_traders_usage)

        # Why recommended
        st.markdown("#### 💡 Why This Strategy Fits You")
        for reason in rec.why_recommended:
            st.markdown(f"- {reason}")

        # Insight from similar traders
        st.info(f"📊 **Insight:** {rec.similar_traders_usage}")

        # Pros and Cons
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**✅ Pros:**")
            for pro in rec.pros:
                st.markdown(f"- {pro}")
        with col2:
            st.markdown("**⚠️ Cons:**")
            for con in rec.cons:
                st.markdown(f"- {con}")

        # Parameters
        with st.expander("🔧 Recommended Parameters"):
            for param, value in rec.parameters.items():
                st.code(f"{param}: {value}")

    def _render_comparison_chart(self, recommendations):
        """Render comparison chart for all recommendations"""
        st.markdown("### 📊 Strategy Comparison")

        # Create radar chart
        categories = [
            "Fit Score",
            "Avg Return",
            "Ease of Use",
            "Risk Adjusted",
            "Popularity",
        ]

        fig = go.Figure()

        for rec in recommendations:
            # Calculate metrics (normalized to 0-100)
            avg_return = sum(rec.expected_return_range) / 2
            ease_score = 90 if rec.risk_level == "low" else 70 if rec.risk_level == "medium" else 50
            risk_adj = 100 - ({"low": 20, "medium": 50, "high": 80}.get(rec.risk_level, 50))
            popularity = 65  # Could be calculated from actual usage data

            values = [
                rec.fit_score,
                min(avg_return * 5, 100),  # Scale to 100
                ease_score,
                risk_adj,
                popularity,
            ]

            fig.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],  # Close the polygon
                    theta=categories + [categories[0]],
                    name=rec.name,
                    fill="toself",
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)


def render_strategy_advisor():
    """Main function to render the strategy advisor"""
    ui = StrategyAdvisorUI(AIAdvisorAPI())
    ui.render()
