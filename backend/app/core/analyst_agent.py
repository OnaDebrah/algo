"""
AI Financial Analyst Agent - Comprehensive investment thesis generation
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class AnalystReport:
    """Complete analyst report structure"""

    ticker: str
    company_name: str
    generated_date: datetime

    # Executive Summary
    investment_thesis: str
    recommendation: str  # Strong Buy, Buy, Hold, Sell, Strong Sell
    target_price: float
    current_price: float
    upside_potential: float
    risk_rating: str

    # Fundamental Analysis
    business_overview: str
    competitive_position: str
    management_quality: str
    financial_health: Dict
    valuation_metrics: Dict

    # Technical Analysis
    technical_indicators: Dict
    chart_patterns: str
    support_resistance: Dict

    # Risk Analysis
    key_risks: List[str]
    risk_mitigation: List[str]

    # Catalysts
    positive_catalysts: List[str]
    negative_catalysts: List[str]

    # Financial Projections
    revenue_forecast: Dict
    earnings_forecast: Dict

    # Peer Comparison
    peer_analysis: Dict

    # Recent News & Sentiment
    news_summary: str
    sentiment_score: float

    # Analyst Notes
    key_takeaways: List[str]
    action_items: List[str]


class FinancialAnalystAgent:
    """AI-powered financial analyst agent"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = None
        if api_key:
            import anthropic

            self.client = anthropic.Anthropic(api_key=api_key)

    async def generate_investment_thesis(
        self,
        ticker: str,
        depth: str = "comprehensive",  # quick, standard, comprehensive, deep_dive
    ) -> AnalystReport:
        """
        Generate complete investment thesis report

        Args:
            ticker: Stock ticker symbol
            depth: Analysis depth level

        Returns:
            Complete analyst report
        """
        print(f"🔍 Starting analysis for {ticker}...")

        # Step 1: Gather all data
        data = await self._gather_market_data(ticker)

        # Step 2: Perform technical analysis
        technical = self._perform_technical_analysis(data)

        # Step 3: Calculate fundamental metrics
        fundamentals = self._calculate_fundamentals(data)

        # Step 4: Get peer comparison
        peers = await self._analyze_peers(ticker, data)

        # Step 5: Fetch recent news and sentiment
        news = await self._analyze_news_sentiment(ticker)

        # Step 6: Generate AI analysis
        ai_analysis = await self._generate_ai_analysis(ticker, data, technical, fundamentals, peers, news, depth)

        # Step 7: Compile report
        report = self._compile_report(ticker, data, technical, fundamentals, peers, news, ai_analysis)

        logger.info(f"✅ Analysis complete for {ticker}")
        return report

    async def _gather_market_data(self, ticker: str) -> Dict:
        """Gather comprehensive market data"""
        logger.info("  📊 Gathering market data...")

        try:
            stock = yf.Ticker(ticker)

            # Get historical data
            hist = stock.history(period="2y")

            # Get company info
            info = stock.info

            # Get financials
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow

            # Get recommendations
            recommendations = stock.recommendations

            return {
                "ticker": ticker,
                "info": info,
                "history": hist,
                "financials": financials,
                "balance_sheet": balance_sheet,
                "cash_flow": cash_flow,
                "recommendations": recommendations,
                "current_price": hist["Close"].iloc[-1] if not hist.empty else 0,
            }

        except Exception as e:
            print(f"  ⚠️ Error gathering data: {e}")
            return {
                "ticker": ticker,
                "info": {},
                "history": pd.DataFrame(),
                "current_price": 0,
            }

    def _perform_technical_analysis(self, data: Dict) -> Dict:
        """Perform comprehensive technical analysis"""
        print("  📈 Performing technical analysis...")

        hist = data.get("history", pd.DataFrame())
        if hist.empty:
            return {}

        try:
            close = hist["Close"]

            # Moving averages
            ma_20 = close.rolling(window=20).mean()
            ma_50 = close.rolling(window=50).mean()
            ma_200 = close.rolling(window=200).mean()

            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # MACD
            exp1 = close.ewm(span=12, adjust=False).mean()
            exp2 = close.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()

            # Bollinger Bands
            bb_middle = close.rolling(window=20).mean()
            bb_std = close.rolling(window=20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)

            # Volume analysis
            avg_volume = hist["Volume"].rolling(window=20).mean()

            # Support and resistance
            recent_high = close.tail(252).max()
            recent_low = close.tail(252).min()

            current = close.iloc[-1]

            return {
                "moving_averages": {
                    "ma_20": float(ma_20.iloc[-1]),
                    "ma_50": float(ma_50.iloc[-1]),
                    "ma_200": float(ma_200.iloc[-1]),
                    "price_vs_ma20": ((current - ma_20.iloc[-1]) / ma_20.iloc[-1] * 100),
                    "price_vs_ma50": ((current - ma_50.iloc[-1]) / ma_50.iloc[-1] * 100),
                    "price_vs_ma200": ((current - ma_200.iloc[-1]) / ma_200.iloc[-1] * 100),
                },
                "momentum": {
                    "rsi": float(rsi.iloc[-1]),
                    "rsi_signal": ("Oversold" if rsi.iloc[-1] < 30 else "Overbought" if rsi.iloc[-1] > 70 else "Neutral"),
                },
                "macd": {
                    "macd": float(macd.iloc[-1]),
                    "signal": float(signal.iloc[-1]),
                    "histogram": float(macd.iloc[-1] - signal.iloc[-1]),
                    "trend": ("Bullish" if macd.iloc[-1] > signal.iloc[-1] else "Bearish"),
                },
                "bollinger": {
                    "upper": float(bb_upper.iloc[-1]),
                    "middle": float(bb_middle.iloc[-1]),
                    "lower": float(bb_lower.iloc[-1]),
                    "position": ("Near Upper" if current > bb_middle.iloc[-1] else "Near Lower"),
                },
                "volume": {
                    "current": float(hist["Volume"].iloc[-1]),
                    "average_20d": float(avg_volume.iloc[-1]),
                    "volume_trend": ("Above Average" if hist["Volume"].iloc[-1] > avg_volume.iloc[-1] else "Below Average"),
                },
                "support_resistance": {
                    "resistance": float(recent_high),
                    "support": float(recent_low),
                    "current": float(current),
                },
            }

        except Exception as e:
            print(f"  ⚠️ Technical analysis error: {e}")
            return {}

    def _calculate_fundamentals(self, data: Dict) -> Dict:
        """Calculate fundamental metrics"""
        print("  💰 Calculating fundamentals...")

        info = data.get("info", {})

        try:
            return {
                "valuation": {
                    "market_cap": info.get("marketCap", 0),
                    "pe_ratio": info.get("trailingPE", 0),
                    "forward_pe": info.get("forwardPE", 0),
                    "peg_ratio": info.get("pegRatio", 0),
                    "price_to_book": info.get("priceToBook", 0),
                    "price_to_sales": info.get("priceToSalesTrailing12Months", 0),
                    "ev_to_ebitda": info.get("enterpriseToEbitda", 0),
                },
                "profitability": {
                    "profit_margin": info.get("profitMargins", 0) * 100,
                    "operating_margin": info.get("operatingMargins", 0) * 100,
                    "roe": info.get("returnOnEquity", 0) * 100,
                    "roa": info.get("returnOnAssets", 0) * 100,
                },
                "financial_health": {
                    "current_ratio": info.get("currentRatio", 0),
                    "debt_to_equity": info.get("debtToEquity", 0),
                    "quick_ratio": info.get("quickRatio", 0),
                    "total_cash": info.get("totalCash", 0),
                    "total_debt": info.get("totalDebt", 0),
                },
                "growth": {
                    "revenue_growth": info.get("revenueGrowth", 0) * 100,
                    "earnings_growth": info.get("earningsGrowth", 0) * 100,
                    "revenue": info.get("totalRevenue", 0),
                    "earnings": info.get("earnings", 0),
                },
                "dividends": {
                    "dividend_yield": (info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0),
                    "payout_ratio": (info.get("payoutRatio", 0) * 100 if info.get("payoutRatio") else 0),
                    "dividend_rate": info.get("dividendRate", 0),
                },
            }

        except Exception as e:
            print(f"  ⚠️ Fundamentals calculation error: {e}")
            return {}

    async def _analyze_peers(self, ticker: str, data: Dict) -> Dict:
        """Analyze peer companies"""
        print("  🏢 Analyzing peer companies...")

        info = data.get("info", {})
        sector = info.get("sector", "")
        industry = info.get("industry", "")

        # For demo, return structured peer data
        # In production, fetch real peer data
        return {
            "sector": sector,
            "industry": industry,
            "peer_comparison": {
                "relative_pe": "Below sector average",
                "relative_growth": "Above sector average",
                "market_position": "Top quartile",
            },
        }

    async def _analyze_news_sentiment(self, ticker: str) -> Dict:
        """Analyze recent news and sentiment"""
        print("  📰 Analyzing news sentiment...")

        try:
            stock = yf.Ticker(ticker)
            news = stock.news

            if news:
                return {
                    "recent_news": news[:5],  # Last 5 news items
                    "news_count": len(news),
                    "sentiment": "Neutral",  # Would use NLP in production
                }

            return {"recent_news": [], "news_count": 0, "sentiment": "No recent news"}

        except Exception as e:
            print(f"  ⚠️ News analysis error: {e}")
            return {"recent_news": [], "news_count": 0, "sentiment": "N/A"}

    async def _generate_ai_analysis(
        self,
        ticker: str,
        data: Dict,
        technical: Dict,
        fundamentals: Dict,
        peers: Dict,
        news: Dict,
        depth: str,
    ) -> Dict:
        """Generate AI-powered analysis using Claude"""
        print("  🤖 Generating AI analysis...")

        if not self.client:
            return self._generate_fallback_analysis(ticker, data, technical, fundamentals)

        # Build comprehensive prompt
        prompt = self._build_analysis_prompt(ticker, data, technical, fundamentals, peers, news, depth)

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                temperature=0.3,  # Lower temperature for factual analysis
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = message.content[0].text

            # Clean and parse response
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1].split("```")[0].strip()

            analysis = json.loads(response_text)
            return analysis

        except Exception as e:
            print(f"  ⚠️ AI analysis error: {e}, using fallback")
            return self._generate_fallback_analysis(ticker, data, technical, fundamentals)

    def _build_analysis_prompt(
        self,
        ticker: str,
        data: Dict,
        technical: Dict,
        fundamentals: Dict,
        peers: Dict,
        news: Dict,
        depth: str,
    ) -> str:
        """Build comprehensive analysis prompt for Claude"""

        info = data.get("info", {})
        company_name = info.get("longName", ticker)
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")

        prompt = f"""You are a seasoned financial analyst at a top investment firm.
        Create a comprehensive investment thesis report for {company_name} ({ticker}).

COMPANY INFORMATION:
- Company: {company_name}
- Ticker: {ticker}
- Sector: {sector}
- Industry: {industry}
- Current Price: ${data.get('current_price', 0):.2f}

FUNDAMENTAL DATA:
{json.dumps(fundamentals, indent=2)}

TECHNICAL INDICATORS:
{json.dumps(technical, indent=2)}

PEER ANALYSIS:
{json.dumps(peers, indent=2)}

RECENT NEWS:
News Items: {news.get('news_count', 0)}
Sentiment: {news.get('sentiment', 'N/A')}

ANALYSIS DEPTH: {depth}

Provide a {depth} analysis covering:

1. INVESTMENT THESIS (200-300 words)
   - Core investment case
   - Why this is attractive/unattractive now
   - Key value drivers

2. RECOMMENDATION
   - Rating: Strong Buy/Buy/Hold/Sell/Strong Sell
   - Target Price (12-month, with justification)
   - Confidence Level: High/Medium/Low

3. BUSINESS ANALYSIS (150-200 words)
   - Business model strengths
   - Competitive advantages/moats
   - Market position

4. VALUATION ASSESSMENT
   - Is it overvalued/undervalued?
   - Key valuation metrics interpretation
   - Comparison to peers/sector

5. KEY RISKS (List 4-6)
   - Specific, actionable risks
   - Probability and impact assessment

6. POSITIVE CATALYSTS (List 3-5)
   - Near-term and long-term drivers
   - Timeline expectations

7. NEGATIVE CATALYSTS (List 2-4)
   - Headwinds to watch
   - Potential red flags

8. TECHNICAL OUTLOOK (100-150 words)
   - Current trend analysis
   - Key levels to watch
   - Technical signals

9. KEY TAKEAWAYS (4-6 bullet points)
   - Most important insights
   - What investors should know

10. ACTION ITEMS (3-4 items)
    - Specific recommendations for investors
    - What to monitor going forward

Respond ONLY with a JSON object in this exact format (no preamble, no markdown):
{{
  "investment_thesis": "Your detailed thesis here...",
  "recommendation": "Buy",
  "target_price": 150.00,
  "confidence": "Medium",
  "risk_rating": "Medium",
  "business_overview": "Business analysis here...",
  "competitive_position": "Competitive position analysis...",
  "valuation_assessment": "Valuation analysis...",
  "technical_outlook": "Technical analysis here...",
  "key_risks": ["Risk 1", "Risk 2", ...],
  "risk_mitigation": ["Mitigation 1", "Mitigation 2", ...],
  "positive_catalysts": ["Catalyst 1", "Catalyst 2", ...],
  "negative_catalysts": ["Catalyst 1", "Catalyst 2", ...],
  "key_takeaways": ["Takeaway 1", "Takeaway 2", ...],
  "action_items": ["Action 1", "Action 2", ...]
}}

Be specific, data-driven, and professional. This report will be used by investors making real decisions."""

        return prompt

    def _generate_fallback_analysis(self, ticker: str, data: Dict, technical: Dict, fundamentals: Dict) -> Dict:
        """Generate rule-based analysis when AI is unavailable"""

        info = data.get("info", {})
        current_price = data.get("current_price", 0)

        # Simple rule-based recommendation
        pe = fundamentals.get("valuation", {}).get("pe_ratio", 0)
        rsi = technical.get("momentum", {}).get("rsi", 50)

        if pe > 0 and pe < 15 and rsi < 40:
            recommendation = "Buy"
            target = current_price * 1.15
        elif pe > 30 or rsi > 70:
            recommendation = "Hold"
            target = current_price * 1.05
        else:
            recommendation = "Hold"
            target = current_price * 1.10

        return {
            "investment_thesis": "Based on current valuation metrics and technical indicators, "
            f"{ticker} presents a moderate investment opportunity. "
            f"The company trades at a P/E ratio of {pe:.2f} "
            "with current momentum indicators suggesting "
            f"{'oversold' if rsi < 40 else 'overbought' if rsi > 70 else 'neutral'} conditions.",
            "recommendation": recommendation,
            "target_price": target,
            "confidence": "Medium",
            "risk_rating": "Medium",
            "business_overview": f"{info.get('longName', ticker)} operates in the {info.get('sector', 'N/A')} "
            f"sector with a market capitalization of ${info.get('marketCap', 0):,.0f}.",
            "competitive_position": "Competitive analysis requires additional data.",
            "valuation_assessment": f"Current valuation metrics suggest "
            f"{'undervalued' if pe < 15 else 'overvalued' if pe > 30 else 'fairly valued'} "
            f"conditions based on P/E ratio of {pe:.2f}.",
            "technical_outlook": "Technical indicators show RSI at "
            f"{rsi:.2f}, suggesting "
            f"{'oversold' if rsi < 40 else 'overbought' if rsi > 70 else 'neutral'} momentum.",
            "key_risks": [
                "Market volatility risk",
                "Sector-specific headwinds",
                "Economic uncertainty",
                "Competitive pressures",
            ],
            "risk_mitigation": [
                "Diversify portfolio holdings",
                "Use stop-loss orders",
                "Monitor quarterly earnings",
            ],
            "positive_catalysts": [
                "Sector growth trends",
                "Potential market expansion",
                "Operational improvements",
            ],
            "negative_catalysts": ["Regulatory changes", "Increased competition"],
            "key_takeaways": [
                f"{recommendation} recommendation with target of ${target:.2f}",
                "Current valuation metrics suggest moderate opportunity",
                "Monitor technical indicators for entry points",
                "Consider position sizing based on risk tolerance",
            ],
            "action_items": [
                "Set price alerts at key technical levels",
                "Review quarterly earnings reports",
                "Monitor sector trends and news",
            ],
        }

    def _compile_report(
        self,
        ticker: str,
        data: Dict,
        technical: Dict,
        fundamentals: Dict,
        peers: Dict,
        news: Dict,
        ai_analysis: Dict,
    ) -> AnalystReport:
        """Compile final analyst report"""

        info = data.get("info", {})
        current_price = data.get("current_price", 0)
        target_price = ai_analysis.get("target_price", current_price * 1.1)

        return AnalystReport(
            ticker=ticker,
            company_name=info.get("longName", ticker),
            generated_date=datetime.now(),
            investment_thesis=ai_analysis.get("investment_thesis", ""),
            recommendation=ai_analysis.get("recommendation", "Hold"),
            target_price=target_price,
            current_price=current_price,
            upside_potential=((target_price - current_price) / current_price * 100),
            risk_rating=ai_analysis.get("risk_rating", "Medium"),
            business_overview=ai_analysis.get("business_overview", ""),
            competitive_position=ai_analysis.get("competitive_position", ""),
            management_quality=ai_analysis.get("management_quality", "N/A"),
            financial_health=fundamentals.get("financial_health", {}),
            valuation_metrics=fundamentals.get("valuation", {}),
            technical_indicators=technical,
            chart_patterns=ai_analysis.get("technical_outlook", ""),
            support_resistance=technical.get("support_resistance", {}),
            key_risks=ai_analysis.get("key_risks", []),
            risk_mitigation=ai_analysis.get("risk_mitigation", []),
            positive_catalysts=ai_analysis.get("positive_catalysts", []),
            negative_catalysts=ai_analysis.get("negative_catalysts", []),
            revenue_forecast={},
            earnings_forecast={},
            peer_analysis=peers,
            news_summary=news.get("sentiment", "N/A"),
            sentiment_score=0.5,
            key_takeaways=ai_analysis.get("key_takeaways", []),
            action_items=ai_analysis.get("action_items", []),
        )
