from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ...config import settings
from ...core.analyst_agent import FinancialAnalystAgent
from ...core.data.providers.providers import ProviderFactory
from ...core.permissions import Permission
from ...models import User
from ...schemas.analyst import AnalystReport, FundamentalData, MACDData, RisksData, SentimentData, TechnicalData, ValuationMetric
from ...services.auth_service import AuthService
from ...utils.errors import safe_detail
from ..deps import check_permission, get_current_active_user, get_db

router = APIRouter(prefix="/analyst", tags=["Analyst"])

analyst_agent = FinancialAnalystAgent(api_key=settings.ANTHROPIC_API_KEY)


def convert_core_report_to_api(core_report, ticker_info: Dict) -> AnalystReport:
    """Convert core AnalystReport to API schema"""

    tech = core_report.technical_indicators
    ma_data = tech.get("moving_averages", {})
    momentum = tech.get("momentum", {})
    macd_data = tech.get("macd", {})
    volume = tech.get("volume", {})
    support_res = tech.get("support_resistance", {})

    fund = core_report.valuation_metrics
    health = core_report.financial_health
    profitability = core_report.profitability
    growth = core_report.growth_metrics
    dividends = core_report.dividends

    valuation_metrics = [
        ValuationMetric(
            subject="P/E Ratio",
            score=min(100, max(0, 100 - abs(fund.get("pe_ratio", 20) - 20) * 2)),
            benchmark=70,
            description="Compared to sector average",
        ),
        ValuationMetric(
            subject="P/B Ratio",
            score=min(100, max(0, 100 - abs(fund.get("price_to_book", 3) - 3) * 10)),
            benchmark=60,
            description="Asset valuation multiple",
        ),
        ValuationMetric(
            subject="PEG Ratio",
            score=min(100, max(0, 100 - abs(fund.get("peg_ratio", 1.5) - 1) * 30)),
            benchmark=75,
            description="Growth-adjusted P/E",
        ),
        ValuationMetric(subject="ROE", score=min(100, max(0, int(profitability.get("roe", 15)))), benchmark=65, description="Return on equity"),
    ]

    # Categorize risks
    all_risks = core_report.key_risks
    risks_data = RisksData(
        regulatory=all_risks[:2] if len(all_risks) > 1 else ["Regulatory oversight"],
        competitive=all_risks[2:4] if len(all_risks) > 3 else ["Market competition"],
        market=all_risks[4:6] if len(all_risks) > 5 else ["Market volatility"],
        financial=["Valuation risk", "Liquidity risk"],
        operational=["Execution risk"],
    )

    # Generate sentiment scores (normalized to 0-100 for the UI)
    sentiment_score = core_report.sentiment_score  # Already in AnalystReport dataclass
    # If the analyst agent added the score to the news dict, we should use it

    # We use the raw score (-1 to 1) and map it to 0-100 for the radar chart
    # -1 -> 0, 0 -> 50, 1 -> 100
    display_score = int((sentiment_score + 1) * 50)

    # Analyst consensus score based on recommendation
    analyst_consensus_map = {
        "Strong Buy": 90, "Buy": 75, "Hold": 50, "Sell": 25, "Strong Sell": 10
    }
    analyst_score = analyst_consensus_map.get(core_report.recommendation, 50)

    # Options sentiment: derive from put/call implied vol skew if available
    options_score = max(0, min(100, int(50 + (sentiment_score * 30))))

    sentiment_data = SentimentData(
        institutional=display_score,
        retail=display_score - 5 if display_score > 5 else 0,
        analyst=analyst_score,
        news=display_score,
        social=display_score + 5 if display_score < 95 else 100,
        options=options_score,
    )

    # Calculate confidence from multiple factors
    factor_count = 0
    confidence_sum = 0.0
    # Technical alignment (RSI not extreme + trend direction matches recommendation)
    rsi_val = momentum.get("rsi", 50)
    if 30 < rsi_val < 70:
        confidence_sum += 75
    else:
        confidence_sum += 55
    factor_count += 1
    # Fundamental backing
    pe = fund.get("pe_ratio", 0)
    if 0 < pe < 40:
        confidence_sum += 80
    else:
        confidence_sum += 50
    factor_count += 1
    # Sentiment alignment
    confidence_sum += display_score
    factor_count += 1
    # Recommendation strength
    if core_report.recommendation in ["Strong Buy", "Strong Sell"]:
        confidence_sum += 90
    elif core_report.recommendation in ["Buy", "Sell"]:
        confidence_sum += 70
    else:
        confidence_sum += 50
    factor_count += 1
    confidence = int(confidence_sum / factor_count) if factor_count > 0 else 65

    # Format market cap
    market_cap_val = fund.get("market_cap", 0)
    if market_cap_val > 1e12:
        market_cap = f"{market_cap_val / 1e12:.1f}T"
    elif market_cap_val > 1e9:
        market_cap = f"{market_cap_val / 1e9:.1f}B"
    elif market_cap_val > 1e6:
        market_cap = f"{market_cap_val / 1e6:.1f}M"
    else:
        market_cap = "N/A"

    return AnalystReport(
        company_name=core_report.company_name,
        ticker=core_report.ticker,
        recommendation=core_report.recommendation,
        recommendation_confidence=confidence,
        current_price=core_report.current_price,
        target_price=core_report.target_price,
        upside=core_report.upside_potential,
        risk_rating=core_report.risk_rating,
        investment_thesis=core_report.investment_thesis,
        sector=ticker_info.get("sector", "N/A"),
        industry=ticker_info.get("industry", "N/A"),
        market_cap=market_cap,
        last_updated=core_report.generated_date.isoformat(),
        valuation=valuation_metrics,
        technical=TechnicalData(
            rsi=momentum.get("rsi", 50),
            rsi_signal=momentum.get("rsi_signal", "Neutral"),
            ma_20=ma_data.get("ma_20", core_report.current_price),
            ma_50=ma_data.get("ma_50", core_report.current_price),
            ma_200=ma_data.get("ma_200", core_report.current_price),
            support_levels=[support_res.get("support", core_report.current_price * 0.95)],
            resistance_levels=[support_res.get("resistance", core_report.current_price * 1.05)],
            trend_strength=min(100, max(0, int(abs(momentum.get("rsi", 50) - 50) * 2 + abs(macd_data.get("histogram", 0)) * 10))),
            macd=MACDData(value=macd_data.get("macd", 0), signal=macd_data.get("signal", 0), histogram=macd_data.get("histogram", 0)),
            volume_trend=volume.get("volume_trend", "Average"),
        ),
        fundamental=FundamentalData(
            pe_ratio=fund.get("pe_ratio", 0),
            pb_ratio=fund.get("price_to_book", 0),
            peg_ratio=fund.get("peg_ratio", 0),
            debt_to_equity=health.get("debt_to_equity", 0),
            roe=profitability.get("roe", 0),
            revenue_growth=growth.get("revenue_growth", 0),
            eps_growth=growth.get("earnings_growth", 0),
            profit_margin=profitability.get("profit_margin", 0),
            dividend_yield=dividends.get("dividend_yield", 0),
        ),
        sentiment=sentiment_data,
        risks=risks_data,
    )


@router.get("/report/{ticker}", response_model=AnalystReport)
async def get_analyst_report(
    ticker: str,
    depth: str = Query("standard", description="Analysis depth: quick, standard, comprehensive, deep_dive"),
    current_user: User = Depends(check_permission(Permission.UNLIMITED_BACKTESTS)),
    db: AsyncSession = Depends(get_db),
):
    """
    Generate comprehensive analyst report for a ticker using AI-powered analysis
    """

    await AuthService.track_usage(db, current_user.id, "ai_analyst_report", {"ticker": ticker, "depth": depth})
    ticker = ticker.upper()

    try:
        core_report = await analyst_agent.generate_investment_thesis(ticker, depth=depth)

        ticker_info = await ProviderFactory().get_ticker_info(ticker)

        # Convert to API schema
        api_report = convert_core_report_to_api(core_report, ticker_info)

        return api_report

    except Exception as e:
        # If analysis fails, return a basic error report
        raise HTTPException(status_code=500, detail=safe_detail(f"Failed to generate analyst report for {ticker}", e))


@router.get("/sentiment/{ticker}")
async def get_sentiment_analysis(
    ticker: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get real-time sentiment analysis for a ticker
    """
    await AuthService.track_usage(db, current_user.id, "get_sentiment", {"ticker": ticker})
    ticker = ticker.upper()

    try:
        sentiment = await analyst_agent.sentiment_service.get_sentiment(ticker)
        return sentiment
    except Exception as e:
        raise HTTPException(status_code=500, detail=safe_detail(f"Sentiment analysis failed for {ticker}", e))
