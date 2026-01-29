"""
Testing script for AI Strategy Advisor
Run this to test the integration
"""

import asyncio

from config import ANTHROPIC_API_KEY
from streamlit.core.ai_advisor_api import AIAdvisorAPI


def test_advisor_without_api():
    """Test advisor with fallback (no API key needed)"""
    print("🧪 Testing AI Advisor (Fallback Mode)")
    print("=" * 60)

    from streamlit.core.ai_advisor import AIStrategyAdvisor

    advisor = AIStrategyAdvisor(AIAdvisorAPI())

    # Test profile 1: Conservative investor
    test_profile_1 = {
        "goals": "Long-term wealth building, Steady income generation",
        "risk_tolerance": "Low",
        "time_horizon": "Long-term (3+ years)",
        "experience": "Beginner",
        "time_commitment": "Minimal (set and forget)",
        "capital": 10000,
        "market_preference": "Stable markets",
    }

    print("\n📋 Test Profile 1: Conservative Beginner")
    print(f"   Risk: {test_profile_1['risk_tolerance']}")
    print(f"   Experience: {test_profile_1['experience']}")
    print(f"   Capital: ${test_profile_1['capital']:,}")

    try:
        recommendations = asyncio.run(advisor.get_recommendations(test_profile_1))
        print(f"\n✓ Received {len(recommendations)} recommendations")

        for idx, rec in enumerate(recommendations, 1):
            print(f"\n   {idx}. {rec.name}")
            print(f"      Fit Score: {rec.fit_score}%")
            print(f"      Risk Level: {rec.risk_level}")
            print(f"      Expected Return: {rec.expected_return_range[0]}-{rec.expected_return_range[1]}%")
            print(f"      Why: {rec.why_recommended[0]}")

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    # Test profile 2: Aggressive investor
    test_profile_2 = {
        "goals": "Short-term profits, Learning and skill development",
        "risk_tolerance": "High",
        "time_horizon": "Short-term (< 1 year)",
        "experience": "Advanced",
        "time_commitment": "High (actively manage)",
        "capital": 50000,
        "market_preference": "Volatile markets",
    }

    print("\n" + "=" * 60)
    print("\n📋 Test Profile 2: Aggressive Advanced Trader")
    print(f"   Risk: {test_profile_2['risk_tolerance']}")
    print(f"   Experience: {test_profile_2['experience']}")
    print(f"   Capital: ${test_profile_2['capital']:,}")

    try:
        recommendations = asyncio.run(advisor.get_recommendations(test_profile_2))
        print(f"\n✓ Received {len(recommendations)} recommendations")

        for idx, rec in enumerate(recommendations, 1):
            print(f"\n   {idx}. {rec.name}")
            print(f"      Fit Score: {rec.fit_score}%")
            print(f"      Risk Level: {rec.risk_level}")
            print(f"      Expected Return: {rec.expected_return_range[0]}-{rec.expected_return_range[1]}%")

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    print("\n✅ All tests passed!")
    return True


def test_advisor_with_api():
    """Test advisor with real Claude API"""
    print("\n🧪 Testing AI Advisor (Real API Mode)")
    print("=" * 60)

    # Check for API key
    api_key = ANTHROPIC_API_KEY
    if not api_key:
        print("⚠️ ANTHROPIC_API_KEY not found in environment")
        print("   Set it with: export ANTHROPIC_API_KEY='your-key'")
        print("   Or add it to .env file")
        return False

    print(f"✓ API Key found: {api_key[:8]}...{api_key[-4:]}")

    from streamlit.core.ai_advisor import AIStrategyAdvisor

    advisor = AIStrategyAdvisor(AIAdvisorAPI())

    test_profile = {
        "goals": "Portfolio diversification, Long-term wealth building",
        "risk_tolerance": "Medium",
        "time_horizon": "Medium-term (1-3 years)",
        "experience": "Intermediate",
        "time_commitment": "Medium (check daily)",
        "capital": 25000,
        "market_preference": "Any (I want the AI to decide)",
    }

    print("\n📋 Test Profile: Balanced Intermediate Trader")

    try:
        print("\n⏳ Calling Claude API...")
        recommendations = asyncio.run(advisor.get_recommendations(test_profile))

        print(f"✓ Received {len(recommendations)} AI-generated recommendations\n")

        for idx, rec in enumerate(recommendations, 1):
            print(f"{idx}. {rec.name} (Fit Score: {rec.fit_score}%)")
            print(f"   Risk: {rec.risk_level.title()}")
            print(f"   Returns: {rec.expected_return_range[0]}-{rec.expected_return_range[1]}%")
            print("   Why recommended:")
            for reason in rec.why_recommended[:2]:
                print(f"      • {reason}")
            print(f"   Insight: {rec.similar_traders_usage}")
            print()

        print("✅ API test passed!")
        return True

    except Exception as e:
        print(f"❌ API Error: {e}")
        return False


def test_ui_components():
    """Test UI rendering (requires Streamlit)"""
    print("\n🧪 Testing UI Components")
    print("=" * 60)

    try:
        from streamlit.ui import StrategyAdvisorUI

        print("✓ UI module imported successfully")

        # Create UI instance
        _ = StrategyAdvisorUI(AIAdvisorAPI())
        print("✓ UI instance created")

        print("\n⚠️  To test the full UI, run:")
        print("   streamlit run main.py")

        return True

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Make sure all dependencies are installed")
        return False


def run_all_tests():
    """Run complete test suite"""
    print("\n" + "=" * 60)
    print("AI STRATEGY ADVISOR - COMPLETE TEST SUITE")
    print("=" * 60)

    results = {
        "Fallback Mode": test_advisor_without_api(),
        "Real API Mode": test_advisor_with_api(),
        "UI Components": test_ui_components(),
    }

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<40} {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 All tests passed! Your AI Strategy Advisor is ready to use.")
        print("\nNext steps:")
        print("1. Run: streamlit run main.py")
        print("2. Navigate to 'AI Strategy Advisor'")
        print("3. Fill out the questionnaire")
        print("4. Get personalized recommendations!")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")

    return all_passed


def demo_mode():
    """Interactive demo of the advisor"""
    print("\n" + "=" * 60)
    print("AI STRATEGY ADVISOR - INTERACTIVE DEMO")
    print("=" * 60)

    print("\nLet's create a sample user profile...")

    profiles = {
        "1": {
            "name": "Conservative Beginner",
            "profile": {
                "goals": "Steady income, Long-term wealth",
                "risk_tolerance": "Low",
                "time_horizon": "Long-term (3+ years)",
                "experience": "Beginner",
                "time_commitment": "Minimal",
                "capital": 10000,
                "market_preference": "Stable markets",
            },
        },
        "2": {
            "name": "Balanced Intermediate",
            "profile": {
                "goals": "Portfolio diversification, Wealth building",
                "risk_tolerance": "Medium",
                "time_horizon": "Medium-term (1-3 years)",
                "experience": "Intermediate",
                "time_commitment": "Medium",
                "capital": 25000,
                "market_preference": "Any",
            },
        },
        "3": {
            "name": "Aggressive Advanced",
            "profile": {
                "goals": "Short-term profits, Skill development",
                "risk_tolerance": "High",
                "time_horizon": "Short-term (< 1 year)",
                "experience": "Advanced",
                "time_commitment": "High",
                "capital": 50000,
                "market_preference": "Volatile markets",
            },
        },
    }

    print("\nChoose a profile:")
    for key, data in profiles.items():
        print(f"{key}. {data['name']}")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice not in profiles:
        print("Invalid choice. Using profile 2.")
        choice = "2"

    selected = profiles[choice]
    print(f"\n✓ Selected: {selected['name']}")

    from streamlit.core.ai_advisor import AIStrategyAdvisor

    advisor = AIStrategyAdvisor(AIAdvisorAPI())

    print("\n⏳ Generating personalized recommendations...")
    recommendations = asyncio.run(advisor.get_recommendations(selected["profile"]))

    print(f"\n{'=' * 60}")
    print(f"RECOMMENDATIONS FOR: {selected['name']}")
    print(f"{'=' * 60}\n")

    for idx, rec in enumerate(recommendations, 1):
        print(f"\n{idx}. {rec.name}")
        print(f"   {'─' * 55}")
        print(f"   Fit Score: {rec.fit_score}% | Risk: {rec.risk_level.title()}")
        print(f"   Expected Returns: {rec.expected_return_range[0]}-{rec.expected_return_range[1]}%")
        print("\n   Why This Strategy Fits You:")
        for reason in rec.why_recommended:
            print(f"      • {reason}")
        print(f"\n   💡 {rec.similar_traders_usage}")
        print(f"\n   Pros: {', '.join(rec.pros[:2])}")
        print(f"   Cons: {', '.join(rec.cons[:2])}")

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_mode()
    else:
        run_all_tests()
