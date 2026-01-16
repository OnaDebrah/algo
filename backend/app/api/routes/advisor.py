from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.ai_advisor import AIStrategyAdvisor, StrategyRecommendation
from backend.app.core.ai_advisor_api import AIAdvisorAPI
from backend.app.api.deps import get_current_active_user, get_db
from backend.app.models import User
from backend.app.services.auth_service import AuthService

router = APIRouter(prefix="/advisor", tags=["Advisor"])


# Initialize AI advisor (singleton pattern)
ai_advisor_api = AIAdvisorAPI()
ai_advisor = AIStrategyAdvisor(ai_advisor_api)

class GuideRequest(BaseModel):
    goal: str
    risk: str
    experience: str
    capital: float
    timeHorizon: str
    markets: List[str]

class Theme(BaseModel):
    primary: str
    secondary: str
    bg: str
    border: str
    text: str
    icon: str  # String identifier for the icon

class Recommendation(BaseModel):
    id: int
    name: str
    tagline: str
    description: str
    fit_score: int
    risk_level: str
    theme: Theme
    expected_return: str
    similar_traders: str
    time_commitment: str
    success_rate: str
    min_capital: float
    why: List[str]
    pros: List[str]
    cons: List[str]
    best_for: List[str]
    performance_data: List[Dict[str, Any]]
    allocation_data: List[Dict[str, Any]]
    tags: List[str]
    icon: str # String identifier

class GuideResponse(BaseModel):
    recommendations: List[Recommendation]
    radar_data: List[Dict[str, Any]]


# Theme mapping for different risk levels
THEME_MAP = {
    "low": {
        "primary": '#4ECDC4',
        "secondary": '#44A08D',
        "bg": 'bg-gradient-to-br from-teal-500/10 to-cyan-500/10',
        "border": 'border-teal-500/30',
        "text": 'text-teal-400',
        "icon": "Shield"
    },
    "medium": {
        "primary": '#FFD166',
        "secondary": '#FFB347',
        "bg": 'bg-gradient-to-br from-amber-500/10 to-yellow-500/10',
        "border": 'border-amber-500/30',
        "text": 'text-amber-400',
        "icon": "CircuitBoard"
    },
    "high": {
        "primary": '#FF6B6B',
        "secondary": '#FF8E53',
        "bg": 'bg-gradient-to-br from-red-500/10 to-orange-500/10',
        "border": 'border-red-500/30',
        "text": 'text-red-400',
        "icon": "Flame"
    }
}

# Mock performance and allocation data generators
def generate_performance_data(risk_level: str) -> List[Dict[str, Any]]:
    """Generate realistic performance data based on risk level"""
    if risk_level == "high":
        return [
            {"month": 'Jan', "return": 15},
            {"month": 'Feb', "return": 22},
            {"month": 'Mar', "return": -5},
            {"month": 'Apr', "return": 28},
            {"month": 'May', "return": 18},
            {"month": 'Jun', "return": 31},
        ]
    elif risk_level == "medium":
        return [
            {"month": 'Jan', "return": 8},
            {"month": 'Feb', "return": 11},
            {"month": 'Mar', "return": 9},
            {"month": 'Apr', "return": 13},
            {"month": 'May', "return": 7},
            {"month": 'Jun', "return": 12},
        ]
    else:  # low
        return [
            {"month": 'Jan', "return": 5},
            {"month": 'Feb', "return": 6},
            {"month": 'Mar', "return": 4},
            {"month": 'Apr', "return": 7},
            {"month": 'May', "return": 5},
            {"month": 'Jun', "return": 6},
        ]

def generate_allocation_data(risk_level: str) -> List[Dict[str, Any]]:
    """Generate allocation data based on risk level"""
    if risk_level == "high":
        return [
            {"name": 'Tech Stocks', "value": 40, "color": '#FF6B6B'},
            {"name": 'Crypto', "value": 35, "color": '#FF8E53'},
            {"name": 'Options', "value": 15, "color": '#FFA726'},
            {"name": 'Cash Reserve', "value": 10, "color": '#FFB74D'},
        ]
    elif risk_level == "medium":
        return [
            {"name": 'Blue Chips', "value": 50, "color": '#FFD166'},
            {"name": 'Growth Stocks', "value": 30, "color": '#FFB347'},
            {"name": 'Bonds', "value": 15, "color": '#FFA726'},
            {"name": 'Cash', "value": 5, "color": '#FF9800'},
        ]
    else:  # low
        return [
            {"name": 'Blue Chips', "value": 50, "color": '#4ECDC4'},
            {"name": 'Dividend Stocks', "value": 30, "color": '#44A08D'},
            {"name": 'Bonds', "value": 15, "color": '#26A69A'},
            {"name": 'Cash', "value": 5, "color": '#00897B'},
        ]

def map_strategy_to_frontend(
    rec: StrategyRecommendation, 
    idx: int,
    request: GuideRequest
) -> Recommendation:
    """Map core StrategyRecommendation to frontend Recommendation format"""
    
    theme = THEME_MAP.get(rec.risk_level, THEME_MAP["medium"])
    
    # Generate tagline based on strategy characteristics
    taglines = {
        "SMA Crossover": "Trend-following simplicity",
        "RSI Mean Reversion": "Statistical AI for steady returns",
        "MACD Momentum": "Momentum-based precision",
        "ML Adaptive Strategy": "Hyper-growth ML-powered strategy",
        "Conservative Dividend": "Steady income generator"
    }
    
    # Best for mapping
    best_for_map = {
        "low": ["Conservative investors", "Long-term holders", "Risk-averse traders"],
        "medium": ["Balanced traders", "Growth seekers", "Active investors"],
        "high": ["Tech-savvy traders", "Growth-focused investors", "Risk-tolerant operators"]
    }
    
    # Time commitment mapping
    time_map = {
        "low": "5-15 min/day",
        "medium": "15-30 min/day",
        "high": "30-60 min/day"
    }
    
    # Success rate estimation (inverse to risk)
    success_rates = {"low": "85%", "medium": "78%", "high": "72%"}
    
    # Min capital based on risk
    min_capitals = {"low": 1000, "medium": 2500, "high": 5000}
    
    # Tags based on characteristics
    tags = []
    if "ML" in rec.name or "Adaptive" in rec.name:
        tags.extend(["AI-Powered", "Advanced"])
    if rec.risk_level == "low":
        tags.extend(["Conservative", "Steady"])
    elif rec.risk_level == "high":
        tags.extend(["High Growth", "Volatility"])
    else:
        tags.extend(["Balanced", "Moderate"])
    
    return Recommendation(
        id=idx + 1,
        name=rec.name,
        tagline=taglines.get(rec.name, "Algorithmic trading strategy"),
        description=rec.description,
        fit_score=int(rec.fit_score),
        risk_level=rec.risk_level,
        theme=Theme(**theme),
        expected_return=f"{rec.expected_return_range[0]}-{rec.expected_return_range[1]}%",
        similar_traders=rec.similar_traders_usage,
        time_commitment=time_map.get(rec.risk_level, "15-30 min/day"),
        success_rate=success_rates.get(rec.risk_level, "75%"),
        min_capital=min_capitals.get(rec.risk_level, 2000),
        why=rec.why_recommended,
        pros=rec.pros,
        cons=rec.cons,
        best_for=best_for_map.get(rec.risk_level, ["General traders"]),
        performance_data=generate_performance_data(rec.risk_level),
        allocation_data=generate_allocation_data(rec.risk_level),
        tags=tags,
        icon=theme["icon"]
    )

@router.post("/guide", response_model=GuideResponse)
async def generate_guide(
    request: GuideRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate personalized strategy recommendations using AI advisor
    """
    # Track usage
    await AuthService.track_usage(db, current_user.id, "ai_advisor_guide", {
        "goal": request.goal,
        "risk": request.risk
    })
    # Build user profile from request
    user_profile = {
        "goals": request.goal,
        "risk_tolerance": request.risk.lower(),
        "time_horizon": request.timeHorizon,
        "experience": request.experience.lower(),
        "capital": request.capital,
        "market_preference": ", ".join(request.markets) if request.markets else "any",
        "time_commitment": "low" if request.timeHorizon == "Short-term" else "medium"
    }
    
    # Get AI-powered recommendations
    core_recommendations = await ai_advisor.get_recommendations(user_profile)
    
    # Map to frontend format
    recommendations = [
        map_strategy_to_frontend(rec, idx, request)
        for idx, rec in enumerate(core_recommendations)
    ]
    
    # Generate radar data based on top 3 recommendations
    radar_data = []
    if len(recommendations) >= 3:
        radar_data = [
            {
                "subject": 'AI Fit',
                recommendations[0].name.split()[0]: recommendations[0].fit_score,
                recommendations[1].name.split()[0]: recommendations[1].fit_score,
                recommendations[2].name.split()[0]: recommendations[2].fit_score
            },
            {
                "subject": 'Returns',
                recommendations[0].name.split()[0]: 85 if recommendations[0].risk_level == "high" else 70,
                recommendations[1].name.split()[0]: 75 if recommendations[1].risk_level == "high" else 65,
                recommendations[2].name.split()[0]: 65 if recommendations[2].risk_level == "high" else 60
            },
            {
                "subject": 'Risk Adj',
                recommendations[0].name.split()[0]: 90 if recommendations[0].risk_level == "low" else 65,
                recommendations[1].name.split()[0]: 85 if recommendations[1].risk_level == "low" else 70,
                recommendations[2].name.split()[0]: 80 if recommendations[2].risk_level == "low" else 75
            },
            {
                "subject": 'Ease',
                recommendations[0].name.split()[0]: 85 if recommendations[0].risk_level == "low" else 50,
                recommendations[1].name.split()[0]: 80 if recommendations[1].risk_level == "low" else 60,
                recommendations[2].name.split()[0]: 75 if recommendations[2].risk_level == "low" else 65
            },
            {
                "subject": 'Growth',
                recommendations[0].name.split()[0]: 90 if recommendations[0].risk_level == "high" else 55,
                recommendations[1].name.split()[0]: 85 if recommendations[1].risk_level == "high" else 60,
                recommendations[2].name.split()[0]: 75 if recommendations[2].risk_level == "high" else 65
            },
            {
                "subject": 'Stability',
                recommendations[0].name.split()[0]: 85 if recommendations[0].risk_level == "low" else 45,
                recommendations[1].name.split()[0]: 80 if recommendations[1].risk_level == "low" else 55,
                recommendations[2].name.split()[0]: 75 if recommendations[2].risk_level == "low" else 60
            },
        ]

    return GuideResponse(
        recommendations=recommendations,
        radar_data=radar_data
    )
