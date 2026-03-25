"""
Crypto market data routes.

Provides cryptocurrency market data via CoinGecko:
  - Top coins by market cap with sparklines
  - Trending coins
  - Global crypto market overview
  - Individual coin quotes with extended crypto-specific data
  - OHLCV historical data for backtesting
  - Advanced analytics: correlation, momentum, sectors, volatility,
    predictions, fear/greed, technical signals, BTC dominance, portfolio optimization
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Body, Depends, Query
from pydantic import BaseModel

from ...api.deps import get_current_active_user
from ...core.data.providers.coingecko_provider import (
    CoinGeckoProvider,
    get_crypto_market_overview,
    get_top_coins,
    get_trending_coins,
)
from ...models.user import User
from ...services.crypto_analytics import (
    compute_btc_dominance_analysis,
    compute_correlation_matrix,
    compute_fear_greed_index,
    compute_momentum_signals,
    compute_portfolio_optimization,
    compute_predictions,
    compute_sector_performance,
    compute_technical_signals,
    compute_volatility_analysis,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/crypto", tags=["Crypto"])

_provider = CoinGeckoProvider()


@router.get("/market-overview")
async def market_overview(
    current_user: User = Depends(get_current_active_user),
):
    """
    Get global crypto market statistics.

    Returns total market cap, BTC/ETH dominance, 24h volume, and active cryptocurrencies.
    """
    return await get_crypto_market_overview()


@router.get("/top")
async def top_coins(
    limit: int = Query(50, ge=1, le=250),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get top cryptocurrencies ranked by market cap.

    Includes 7-day sparkline, price changes (1h, 24h, 7d, 30d), supply data, and ATH info.
    """
    coins = await get_top_coins(limit=limit)
    return {"count": len(coins), "coins": coins}


@router.get("/trending")
async def trending(
    current_user: User = Depends(get_current_active_user),
):
    """Get currently trending coins on CoinGecko."""
    coins = await get_trending_coins()
    return {"count": len(coins), "coins": coins}


@router.get("/quote/{symbol}")
async def crypto_quote(
    symbol: str,
    current_user: User = Depends(get_current_active_user),
):
    """
    Get detailed quote for a cryptocurrency.

    Includes: price, 24h change, volume, market cap, supply data, ATH, and rank.
    Supports both formats: 'BTC' or 'BTC-USD'.
    """
    import asyncio

    quote = await asyncio.to_thread(_provider.get_quote, symbol)
    return quote


@router.get("/coin/{coin_id}")
async def coin_detail(
    coin_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """
    Get detailed information about a single cryptocurrency.

    Returns description, links, genesis date, categories, market data,
    developer stats, and community stats from CoinGecko.
    """
    import httpx

    async with httpx.AsyncClient(timeout=20) as client:
        try:
            resp = await client.get(
                f"https://api.coingecko.com/api/v3/coins/{coin_id}",
                params={
                    "localization": "false",
                    "tickers": "false",
                    "market_data": "true",
                    "community_data": "true",
                    "developer_data": "true",
                    "sparkline": "true",
                },
                headers={"Accept": "application/json"},
            )
            if resp.status_code != 200:
                return {"error": f"CoinGecko returned {resp.status_code}"}
            data = resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch coin detail for {coin_id}: {e}")
            return {"error": str(e)}

    # Extract key fields
    market = data.get("market_data") or {}
    desc_raw = (data.get("description") or {}).get("en", "")
    # Strip HTML tags for clean text
    import re

    description = re.sub(r"<[^>]+>", "", desc_raw).strip()

    return {
        "id": data.get("id"),
        "symbol": (data.get("symbol") or "").upper(),
        "name": data.get("name"),
        "image": (data.get("image") or {}).get("large"),
        "description": description[:3000],  # cap at 3000 chars
        "categories": data.get("categories") or [],
        "genesis_date": data.get("genesis_date"),
        "hashing_algorithm": data.get("hashing_algorithm"),
        "links": {
            "homepage": ((data.get("links") or {}).get("homepage") or [""])[0],
            "blockchain_site": [u for u in ((data.get("links") or {}).get("blockchain_site") or []) if u][:3],
            "subreddit": (data.get("links") or {}).get("subreddit_url"),
            "twitter": f"https://twitter.com/{(data.get('links') or {}).get('twitter_screen_name', '')}",
            "github": ((data.get("links") or {}).get("repos_url") or {}).get("github", [])[:2],
        },
        "market_data": {
            "current_price": (market.get("current_price") or {}).get("usd", 0),
            "market_cap": (market.get("market_cap") or {}).get("usd", 0),
            "market_cap_rank": market.get("market_cap_rank"),
            "total_volume": (market.get("total_volume") or {}).get("usd", 0),
            "high_24h": (market.get("high_24h") or {}).get("usd", 0),
            "low_24h": (market.get("low_24h") or {}).get("usd", 0),
            "price_change_24h": market.get("price_change_24h", 0),
            "price_change_pct_24h": market.get("price_change_percentage_24h", 0),
            "price_change_pct_7d": market.get("price_change_percentage_7d", 0),
            "price_change_pct_30d": market.get("price_change_percentage_30d", 0),
            "price_change_pct_1y": market.get("price_change_percentage_1y", 0),
            "ath": (market.get("ath") or {}).get("usd", 0),
            "ath_change_pct": (market.get("ath_change_percentage") or {}).get("usd", 0),
            "ath_date": (market.get("ath_date") or {}).get("usd"),
            "atl": (market.get("atl") or {}).get("usd", 0),
            "atl_date": (market.get("atl_date") or {}).get("usd"),
            "circulating_supply": market.get("circulating_supply", 0),
            "total_supply": market.get("total_supply"),
            "max_supply": market.get("max_supply"),
            "fully_diluted_valuation": (market.get("fully_diluted_valuation") or {}).get("usd"),
            "sparkline_7d": (market.get("sparkline_7d") or {}).get("price", []),
        },
        "community": {
            "twitter_followers": (data.get("community_data") or {}).get("twitter_followers"),
            "reddit_subscribers": (data.get("community_data") or {}).get("reddit_subscribers"),
            "reddit_active_accounts": (data.get("community_data") or {}).get("reddit_accounts_active_48h"),
        },
        "developer": {
            "github_forks": (data.get("developer_data") or {}).get("forks"),
            "github_stars": (data.get("developer_data") or {}).get("stars"),
            "github_subscribers": (data.get("developer_data") or {}).get("subscribers"),
            "github_total_issues": (data.get("developer_data") or {}).get("total_issues"),
            "github_closed_issues": (data.get("developer_data") or {}).get("closed_issues"),
            "commit_count_4_weeks": (data.get("developer_data") or {}).get("commit_count_4_weeks"),
        },
        "sentiment_votes_up_pct": data.get("sentiment_votes_up_percentage"),
        "sentiment_votes_down_pct": data.get("sentiment_votes_down_percentage"),
        "coingecko_rank": data.get("coingecko_rank"),
        "coingecko_score": data.get("coingecko_score"),
        "liquidity_score": data.get("liquidity_score"),
    }


@router.get("/historical/{symbol}")
async def crypto_historical(
    symbol: str,
    period: str = Query("1y", description="1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max"),
    interval: str = Query("1d", description="Data interval (auto-selected by CoinGecko based on period)"),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get OHLCV historical data for a cryptocurrency.

    Data sourced from CoinGecko. Interval is auto-determined:
    - 1-2 days: 30-minute candles
    - 3-30 days: 4-hour candles
    - 31+ days: daily candles
    """
    import asyncio

    df = await asyncio.to_thread(_provider.fetch_data, symbol, period, interval)

    if df.empty:
        return {"symbol": symbol, "period": period, "data": []}

    # Convert to list of dicts
    df_reset = df.reset_index()
    data = []
    for _, row in df_reset.iterrows():
        data.append(
            {
                "date": row.get("Date", row.name).isoformat() if hasattr(row.get("Date", row.name), "isoformat") else str(row.get("Date", row.name)),
                "open": float(row.get("Open", 0)),
                "high": float(row.get("High", 0)),
                "low": float(row.get("Low", 0)),
                "close": float(row.get("Close", 0)),
                "volume": float(row.get("Volume", 0)),
            }
        )

    return {
        "symbol": symbol,
        "period": period,
        "interval": interval,
        "data": data,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Analytics endpoints
# ═══════════════════════════════════════════════════════════════════════════


class PortfolioHolding(BaseModel):
    coin_id: str
    amount_usd: float


class PortfolioRequest(BaseModel):
    holdings: List[PortfolioHolding]


@router.get("/analytics/correlation")
async def correlation_matrix(
    coins: str = Query(
        "bitcoin,ethereum,solana,cardano,avalanche-2,chainlink,dogecoin,polkadot",
        description="Comma-separated CoinGecko IDs",
    ),
    period: str = Query("30d", description="7d, 30d, 90d"),
    current_user: User = Depends(get_current_active_user),
):
    """
    Compute a Pearson correlation matrix of daily returns across selected coins.

    Useful for diversification analysis — low or negative correlations
    indicate assets that don't move in lockstep.
    """
    coin_list = [c.strip() for c in coins.split(",") if c.strip()]
    return await compute_correlation_matrix(coin_list, period)


@router.get("/analytics/momentum")
async def momentum_scanner(
    current_user: User = Depends(get_current_active_user),
):
    """
    Scan top 50 coins for momentum signals (RSI, MACD, short-term momentum).

    Returns each coin with a signal (STRONG_BUY → STRONG_SELL), RSI value,
    and composite score from -100 to 100.
    """
    coins = await get_top_coins(limit=50)
    return await compute_momentum_signals(coins)


@router.get("/analytics/sectors")
async def sector_performance(
    current_user: User = Depends(get_current_active_user),
):
    """
    Get average 24h and 7d performance for predefined crypto sectors
    (Layer 1, Layer 2, DeFi, Meme, Store of Value, AI & Data).
    """
    return await compute_sector_performance()


@router.get("/analytics/volatility")
async def volatility_analysis(
    coins: str = Query(
        "bitcoin,ethereum,solana,cardano,avalanche-2,chainlink,dogecoin,polkadot,arbitrum,optimism",
        description="Comma-separated CoinGecko IDs",
    ),
    period: str = Query("30d", description="7d, 30d, 90d"),
    current_user: User = Depends(get_current_active_user),
):
    """
    Compute realized volatility, Bollinger width, and risk classification
    for each coin. Sorted by volatility descending.
    """
    coin_list = [c.strip() for c in coins.split(",") if c.strip()]
    return await compute_volatility_analysis(coin_list, period)


@router.get("/analytics/predictions")
async def price_predictions(
    current_user: User = Depends(get_current_active_user),
):
    """
    Multi-factor scoring model for short-term price direction.

    Combines momentum, mean reversion (RSI), volume, and trend signals
    into a composite score with BULLISH / NEUTRAL / BEARISH prediction.
    """
    coins = await get_top_coins(limit=30)
    coin_ids = [c.get("id") for c in coins if c.get("id")]
    return await compute_predictions(coin_ids[:20])


@router.get("/analytics/fear-greed")
async def fear_greed_index(
    current_user: User = Depends(get_current_active_user),
):
    """
    Custom crypto Fear & Greed index (0-100).

    Components: BTC volatility (25%), market volume (25%),
    BTC dominance (25%), and momentum (25% — pct of top 20 coins green).
    """
    overview = await get_crypto_market_overview()
    coins = await get_top_coins(limit=50)
    return await compute_fear_greed_index(overview, coins)


@router.get("/analytics/technical-signals")
async def technical_signals(
    coins: str = Query(
        "bitcoin,ethereum,solana,cardano,avalanche-2,chainlink,dogecoin,polkadot",
        description="Comma-separated CoinGecko IDs",
    ),
    current_user: User = Depends(get_current_active_user),
):
    """
    Full technical analysis per coin: RSI(14), SMA(20/50) cross,
    MACD(12,26,9), and Bollinger Bands(20,2).

    Each indicator returns a BUY/SELL/NEUTRAL signal. Overall signal
    is determined by majority vote.
    """
    coin_list = [c.strip() for c in coins.split(",") if c.strip()]
    return await compute_technical_signals(coin_list)


@router.get("/analytics/btc-dominance")
async def btc_dominance(
    current_user: User = Depends(get_current_active_user),
):
    """
    Analyze BTC dominance trend and detect altcoin season.

    Returns dominance value, trend (rising/falling/stable),
    altcoin season index (0-100), and analysis text.
    """
    overview = await get_crypto_market_overview()
    coins = await get_top_coins(limit=50)
    return await compute_btc_dominance_analysis(overview, coins)


@router.post("/analytics/portfolio-optimize")
async def portfolio_optimize(
    request: PortfolioRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    """
    Mean-variance portfolio optimization.

    Accepts a list of holdings (coin_id + amount_usd) and returns
    current weights, minimum-variance optimized weights, and
    equal-weight comparison with expected return, volatility, and Sharpe ratio.
    """
    holdings = [{"coin_id": h.coin_id, "amount_usd": h.amount_usd} for h in request.holdings]
    return await compute_portfolio_optimization(holdings)
