from typing import List

from pydantic import BaseModel


class ValuationMetric(BaseModel):
    subject: str
    score: float
    benchmark: float
    description: str


class MACDData(BaseModel):
    value: float
    signal: float
    histogram: float


class TechnicalData(BaseModel):
    rsi: float
    rsi_signal: str  # 'Oversold' | 'Neutral' | ...
    ma_20: float
    ma_50: float
    ma_200: float
    support_levels: List[float]
    resistance_levels: List[float]
    trend_strength: float
    macd: MACDData
    volume_trend: str


class FundamentalData(BaseModel):
    pe_ratio: float
    pb_ratio: float
    peg_ratio: float
    debt_to_equity: float
    roe: float
    revenue_growth: float
    eps_growth: float
    profit_margin: float
    dividend_yield: float


class SentimentData(BaseModel):
    institutional: float
    retail: float
    analyst: float
    news: float
    social: float
    options: float


class RisksData(BaseModel):
    regulatory: List[str]
    competitive: List[str]
    market: List[str]
    financial: List[str]
    operational: List[str]


class AnalystReport(BaseModel):
    company_name: str
    ticker: str
    recommendation: str  # 'Strong Buy' | 'Buy' | ...
    recommendation_confidence: float
    current_price: float
    target_price: float
    upside: float
    risk_rating: str  # 'Low' | 'Medium' ...
    investment_thesis: str
    sector: str
    industry: str
    market_cap: str
    last_updated: str
    valuation: List[ValuationMetric]
    technical: TechnicalData
    fundamental: FundamentalData
    sentiment: SentimentData
    risks: RisksData
