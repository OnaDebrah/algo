"""
AI Strategy Advisor - Personalized trading strategy recommendations
"""

import json
from dataclasses import dataclass
from typing import Dict, List

from core.ai_advisor_api import AIAdvisorAPI


@dataclass
class StrategyRecommendation:
    """Represents a single strategy recommendation"""

    name: str
    description: str
    risk_level: str
    expected_return_range: tuple
    fit_score: float
    why_recommended: List[str]
    similar_traders_usage: str
    parameters: Dict
    pros: List[str]
    cons: List[str]


class AIStrategyAdvisor:
    """AI-powered strategy recommendation engine"""

    def __init__(self, ai_advisor_api: AIAdvisorAPI):
        self.strategy_database = self._initialize_strategy_database()
        self._call_claude_api = ai_advisor_api

    def _initialize_strategy_database(self) -> Dict:
        """Initialize the strategy knowledge base"""
        return {
            "sma_crossover": {
                "name": "SMA Crossover",
                "description": "Uses moving average crossovers to identify trends",
                "risk_level": "low",
                "typical_return_range": (8, 15),
                "time_commitment": "low",
                "market_conditions": ["trending"],
                "parameters": {"fast_period": 20, "slow_period": 50},
                "pros": [
                    "Simple to understand",
                    "Works well in trending markets",
                    "Low false signals with proper filtering",
                ],
                "cons": [
                    "Lags in choppy markets",
                    "Can produce whipsaws",
                    "Slower entry/exit timing",
                ],
            },
            "rsi_mean_reversion": {
                "name": "RSI Mean Reversion",
                "description": "Buys oversold and sells overbought conditions",
                "risk_level": "medium",
                "typical_return_range": (12, 25),
                "time_commitment": "medium",
                "market_conditions": ["ranging", "volatile"],
                "parameters": {"period": 14, "oversold": 30, "overbought": 70},
                "pros": [
                    "Captures reversal opportunities",
                    "Works in ranging markets",
                    "Good risk/reward ratios",
                ],
                "cons": [
                    "Can get caught in strong trends",
                    "Requires quick decision making",
                    "Higher drawdowns possible",
                ],
            },
            "macd_momentum": {
                "name": "MACD Momentum",
                "description": "Follows momentum using MACD indicator signals",
                "risk_level": "medium",
                "typical_return_range": (10, 22),
                "time_commitment": "medium",
                "market_conditions": ["trending", "volatile"],
                "parameters": {"fast": 12, "slow": 26, "signal": 9},
                "pros": [
                    "Good trend confirmation",
                    "Balanced signals",
                    "Works across timeframes",
                ],
                "cons": [
                    "Can lag entries",
                    "Multiple false signals in choppy markets",
                    "Requires experience to interpret",
                ],
            },
            "ml_adaptive": {
                "name": "ML Adaptive Strategy",
                "description": "Machine learning model adapts to market conditions",
                "risk_level": "high",
                "typical_return_range": (15, 35),
                "time_commitment": "high",
                "market_conditions": ["all"],
                "parameters": {"model_type": "random_forest", "retrain_days": 30},
                "pros": [
                    "Adapts to changing markets",
                    "Highest potential returns",
                    "Data-driven decisions",
                ],
                "cons": [
                    "Requires technical knowledge",
                    "Needs regular monitoring",
                    "Higher risk and complexity",
                ],
            },
            "conservative_dividend": {
                "name": "Conservative Dividend",
                "description": "Long-term buy and hold of dividend-paying stocks",
                "risk_level": "low",
                "typical_return_range": (6, 12),
                "time_commitment": "low",
                "market_conditions": ["all"],
                "parameters": {"min_yield": 3.0, "rebalance_months": 6},
                "pros": [
                    "Very low maintenance",
                    "Steady income stream",
                    "Lower volatility",
                ],
                "cons": [
                    "Lower growth potential",
                    "Slower to react to changes",
                    "Limited excitement factor",
                ],
            },
        }

    async def get_recommendations(
        self, user_profile: Dict
    ) -> List[StrategyRecommendation]:
        """
        Generate personalized strategy recommendations using Claude API

        Args:
            user_profile: User's goals, risk tolerance, and preferences

        Returns:
            List of recommended strategies with explanations
        """
        # Build the AI prompt
        prompt = self._build_recommendation_prompt(user_profile)

        # Call Claude API
        try:
            response = await self._call_claude_api.call_claude_api(prompt)
            recommendations = self._parse_recommendations(response, user_profile)
            return recommendations
        except Exception as e:
            print(f"AI recommendation error: {e}")
            # Fallback to rule-based recommendations
            return self._fallback_recommendations(user_profile)

    def _build_recommendation_prompt(self, profile: Dict) -> str:
        """Build the prompt for Claude API"""
        strategies_info = json.dumps(self.strategy_database, indent=2)

        prompt = f"""You are an expert trading strategy advisor. Based on the user's profile, recommend 3-5 optimal trading strategies.

USER PROFILE:
- Investment Goals: {profile.get('goals', 'Not specified')}
- Risk Tolerance: {profile.get('risk_tolerance', 'medium')}
- Time Horizon: {profile.get('time_horizon', 'medium')}
- Experience Level: {profile.get('experience', 'intermediate')}
- Time Commitment: {profile.get('time_commitment', 'medium')}
- Capital: ${profile.get('capital', 10000):,}
- Preferred Market Conditions: {profile.get('market_preference', 'any')}

AVAILABLE STRATEGIES:
{strategies_info}

Please respond ONLY with a JSON object in this exact format (no preamble, no markdown):
{{
  "recommendations": [
    {{
      "strategy_key": "sma_crossover",
      "fit_score": 85,
      "why_recommended": [
        "Matches your low risk tolerance perfectly",
        "Suitable for your available time commitment",
        "Good for beginners and intermediates"
      ],
      "personalized_insight": "Based on traders with similar profiles, this strategy averages 12% annual returns with 8% drawdowns",
      "risk_adjustment": "Consider starting with 50% position sizes until comfortable"
    }}
  ]
}}

Rank by fit_score (0-100). Consider risk tolerance, time commitment, experience, and goals. Be specific and actionable."""

        return prompt

    def _parse_recommendations(
        self, ai_response: str, profile: Dict
    ) -> List[StrategyRecommendation]:
        """Parse AI response into recommendation objects"""
        try:
            data = json.loads(ai_response)
            recommendations = []

            for rec in data.get("recommendations", []):
                strategy_key = rec["strategy_key"]
                strategy_data = self.strategy_database.get(strategy_key)

                if not strategy_data:
                    continue

                recommendation = StrategyRecommendation(
                    name=strategy_data["name"],
                    description=strategy_data["description"],
                    risk_level=strategy_data["risk_level"],
                    expected_return_range=strategy_data["typical_return_range"],
                    fit_score=rec["fit_score"],
                    why_recommended=rec["why_recommended"],
                    similar_traders_usage=rec["personalized_insight"],
                    parameters=strategy_data["parameters"],
                    pros=strategy_data["pros"],
                    cons=strategy_data["cons"],
                )
                recommendations.append(recommendation)

            return recommendations

        except json.JSONDecodeError:
            # Fallback if parsing fails
            return self._fallback_recommendations(profile)

    def _fallback_recommendations(self, profile: Dict) -> List[StrategyRecommendation]:
        """Rule-based fallback recommendations"""
        risk = profile.get("risk_tolerance", "medium").lower()
        experience = profile.get("experience", "intermediate").lower()
        time_commit = profile.get("time_commitment", "medium").lower()

        recommendations = []

        # Rule-based matching
        if risk == "low":
            strategies = ["sma_crossover", "conservative_dividend"]
        elif risk == "high":
            strategies = ["ml_adaptive", "macd_momentum", "rsi_mean_reversion"]
        else:
            strategies = ["sma_crossover", "macd_momentum", "rsi_mean_reversion"]

        for idx, strategy_key in enumerate(strategies[:3]):
            strategy_data = self.strategy_database[strategy_key]

            fit_score = 75 - (idx * 10)  # Decreasing scores

            recommendation = StrategyRecommendation(
                name=strategy_data["name"],
                description=strategy_data["description"],
                risk_level=strategy_data["risk_level"],
                expected_return_range=strategy_data["typical_return_range"],
                fit_score=fit_score,
                why_recommended=[
                    f"Matches your {risk} risk tolerance",
                    f"Suitable for {experience} traders",
                    f"Fits your {time_commit} time commitment",
                ],
                similar_traders_usage="Used by 35% of traders with similar profiles",
                parameters=strategy_data["parameters"],
                pros=strategy_data["pros"],
                cons=strategy_data["cons"],
            )
            recommendations.append(recommendation)

        return recommendations
