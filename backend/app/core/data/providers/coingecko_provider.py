"""
CoinGecko Data Provider — Free crypto market data.

Provides:
  - Real-time prices and 24h stats for any cryptocurrency
  - OHLCV historical data
  - Market cap rankings and trending coins
  - Global crypto market overview (dominance, total market cap, DeFi TVL)

Rate limit: 10-30 req/min on free tier (no API key required).
Docs: https://docs.coingecko.com/reference/introduction
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd

from ....config import COINGECKO_BASE_URL
from .base_provider import DataProvider, QuoteProvider

logger = logging.getLogger(__name__)

# Map common ticker symbols to CoinGecko IDs
TICKER_TO_ID = {
    "BTC": "bitcoin",
    "BTC-USD": "bitcoin",
    "ETH": "ethereum",
    "ETH-USD": "ethereum",
    "BNB": "binancecoin",
    "BNB-USD": "binancecoin",
    "SOL": "solana",
    "SOL-USD": "solana",
    "XRP": "ripple",
    "XRP-USD": "ripple",
    "ADA": "cardano",
    "ADA-USD": "cardano",
    "DOGE": "dogecoin",
    "DOGE-USD": "dogecoin",
    "DOT": "polkadot",
    "DOT-USD": "polkadot",
    "MATIC": "matic-network",
    "MATIC-USD": "matic-network",
    "AVAX": "avalanche-2",
    "AVAX-USD": "avalanche-2",
    "LINK": "chainlink",
    "LINK-USD": "chainlink",
    "UNI": "uniswap",
    "UNI-USD": "uniswap",
    "ATOM": "cosmos",
    "ATOM-USD": "cosmos",
    "LTC": "litecoin",
    "LTC-USD": "litecoin",
    "NEAR": "near",
    "NEAR-USD": "near",
    "ARB": "arbitrum",
    "ARB-USD": "arbitrum",
    "OP": "optimism",
    "OP-USD": "optimism",
    "APT": "aptos",
    "APT-USD": "aptos",
    "SUI": "sui",
    "SUI-USD": "sui",
    "PEPE": "pepe",
    "PEPE-USD": "pepe",
    "SHIB": "shiba-inu",
    "SHIB-USD": "shiba-inu",
    "FIL": "filecoin",
    "FIL-USD": "filecoin",
    "AAVE": "aave",
    "AAVE-USD": "aave",
    "MKR": "maker",
    "MKR-USD": "maker",
}

# Reverse mapping for display
ID_TO_TICKER = {v: k.replace("-USD", "") for k, v in TICKER_TO_ID.items() if k.endswith("-USD")}


def _resolve_id(symbol: str) -> str:
    """Resolve a ticker symbol to CoinGecko coin ID."""
    sym = symbol.upper().strip()
    if sym in TICKER_TO_ID:
        return TICKER_TO_ID[sym]
    # Try without -USD suffix
    base = sym.replace("-USD", "").replace("-USDT", "")
    if base in TICKER_TO_ID:
        return TICKER_TO_ID[base]
    # Last resort: lowercase symbol as ID
    return base.lower()


def _period_to_days(period: str) -> int:
    """Convert period string to days for CoinGecko API."""
    mapping = {
        "1d": 1,
        "5d": 5,
        "1mo": 30,
        "3mo": 90,
        "6mo": 180,
        "1y": 365,
        "2y": 730,
        "5y": 1825,
        "max": 3650,
        "ytd": 90,  # Approximate YTD
    }
    return mapping.get(period, 365)


class CoinGeckoProvider(DataProvider, QuoteProvider):
    """CoinGecko data provider for cryptocurrency market data."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key  # Optional Pro API key
        self._last_request = 0.0
        self._min_interval = 2.0  # Min seconds between requests (rate limit safety)

    def _rate_limit(self):
        """Simple rate limiter."""
        elapsed = time.monotonic() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.monotonic()

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["x-cg-demo-api-key"] = self._api_key
        return headers

    def fetch_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        start: Optional[Any] = None,
        end: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from CoinGecko."""
        coin_id = _resolve_id(symbol)
        days = _period_to_days(period)

        self._rate_limit()

        try:
            with httpx.Client(timeout=20) as client:
                # CoinGecko OHLC endpoint
                resp = client.get(
                    f"{COINGECKO_BASE_URL}/coins/{coin_id}/ohlc",
                    params={"vs_currency": "usd", "days": str(days)},
                    headers=self._get_headers(),
                )

                if resp.status_code == 429:
                    logger.warning("CoinGecko rate limited, retrying after 60s...")
                    time.sleep(60)
                    resp = client.get(
                        f"{COINGECKO_BASE_URL}/coins/{coin_id}/ohlc",
                        params={"vs_currency": "usd", "days": str(days)},
                        headers=self._get_headers(),
                    )

                if resp.status_code != 200:
                    logger.error(f"CoinGecko OHLC failed: HTTP {resp.status_code}")
                    return pd.DataFrame()

                data = resp.json()
                if not data:
                    return pd.DataFrame()

                # CoinGecko returns: [[timestamp_ms, open, high, low, close], ...]
                df = pd.DataFrame(data, columns=["timestamp", "Open", "High", "Low", "Close"])
                df["Date"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("Date", inplace=True)
                df.drop("timestamp", axis=1, inplace=True)

                # CoinGecko OHLC doesn't include volume — fetch from market_chart
                try:
                    vol_resp = client.get(
                        f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart",
                        params={"vs_currency": "usd", "days": str(days)},
                        headers=self._get_headers(),
                    )
                    if vol_resp.status_code == 200:
                        vol_data = vol_resp.json()
                        volumes = vol_data.get("total_volumes", [])
                        if volumes:
                            vol_df = pd.DataFrame(volumes, columns=["timestamp", "Volume"])
                            vol_df["Date"] = pd.to_datetime(vol_df["timestamp"], unit="ms")
                            vol_df.set_index("Date", inplace=True)
                            vol_df.drop("timestamp", axis=1, inplace=True)
                            # Merge on nearest timestamp
                            df = df.join(vol_df, how="left")
                            df["Volume"] = df["Volume"].fillna(0)
                except Exception:
                    df["Volume"] = 0

                if "Volume" not in df.columns:
                    df["Volume"] = 0

                return df

        except Exception as e:
            logger.error(f"CoinGecko fetch_data failed for {symbol}: {e}")
            return pd.DataFrame()

    def get_quote(self, symbol: str) -> Dict:
        """Get real-time crypto quote from CoinGecko."""
        coin_id = _resolve_id(symbol)

        self._rate_limit()

        try:
            with httpx.Client(timeout=15) as client:
                resp = client.get(
                    f"{COINGECKO_BASE_URL}/coins/{coin_id}",
                    params={
                        "localization": "false",
                        "tickers": "false",
                        "community_data": "false",
                        "developer_data": "false",
                    },
                    headers=self._get_headers(),
                )

                if resp.status_code != 200:
                    return {"symbol": symbol, "price": 0, "error": f"HTTP {resp.status_code}"}

                data = resp.json()
                market = data.get("market_data", {})

                price = market.get("current_price", {}).get("usd", 0)
                change_24h = market.get("price_change_24h", 0) or 0
                change_pct = market.get("price_change_percentage_24h", 0) or 0

                return {
                    "symbol": symbol.upper(),
                    "name": data.get("name", symbol),
                    "price": price,
                    "change": change_24h,
                    "changePercent": change_pct,
                    "volume": market.get("total_volume", {}).get("usd", 0),
                    "marketCap": market.get("market_cap", {}).get("usd", 0),
                    "high": market.get("high_24h", {}).get("usd", 0),
                    "low": market.get("low_24h", {}).get("usd", 0),
                    "open": price - change_24h,
                    "previousClose": price - change_24h,
                    "circulatingSupply": market.get("circulating_supply", 0),
                    "totalSupply": market.get("total_supply", 0),
                    "maxSupply": market.get("max_supply"),
                    "ath": market.get("ath", {}).get("usd", 0),
                    "athDate": market.get("ath_date", {}).get("usd", ""),
                    "athChangePercent": market.get("ath_change_percentage", {}).get("usd", 0),
                    "marketCapRank": data.get("market_cap_rank", 0),
                    "priceChange7d": market.get("price_change_percentage_7d", 0),
                    "priceChange30d": market.get("price_change_percentage_30d", 0),
                    "timestamp": datetime.now().isoformat(),
                    "asset_class": "crypto",
                }

        except Exception as e:
            logger.error(f"CoinGecko quote failed for {symbol}: {e}")
            return {"symbol": symbol, "price": 0, "error": str(e)}


# ── Standalone utility functions (not part of provider interface) ────────


async def get_crypto_market_overview() -> Dict:
    """Get global crypto market stats: total market cap, BTC dominance, etc."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{COINGECKO_BASE_URL}/global",
                headers={"Accept": "application/json"},
            )
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}"}

            data = resp.json().get("data", {})
            return {
                "total_market_cap_usd": data.get("total_market_cap", {}).get("usd", 0),
                "total_volume_24h_usd": data.get("total_volume", {}).get("usd", 0),
                "bitcoin_dominance": data.get("market_cap_percentage", {}).get("btc", 0),
                "ethereum_dominance": data.get("market_cap_percentage", {}).get("eth", 0),
                "active_cryptocurrencies": data.get("active_cryptocurrencies", 0),
                "markets": data.get("markets", 0),
                "market_cap_change_24h_pct": data.get("market_cap_change_percentage_24h_usd", 0),
                "updated_at": data.get("updated_at", 0),
            }
    except Exception as e:
        logger.error(f"CoinGecko global market overview failed: {e}")
        return {"error": str(e)}


async def get_trending_coins() -> List[Dict]:
    """Get currently trending coins on CoinGecko."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{COINGECKO_BASE_URL}/search/trending",
                headers={"Accept": "application/json"},
            )
            if resp.status_code != 200:
                return []

            data = resp.json()
            coins = data.get("coins", [])
            return [
                {
                    "id": c["item"]["id"],
                    "name": c["item"]["name"],
                    "symbol": c["item"]["symbol"],
                    "market_cap_rank": c["item"].get("market_cap_rank", 0),
                    "thumb": c["item"].get("thumb", ""),
                    "score": c["item"].get("score", 0),
                }
                for c in coins[:15]
            ]
    except Exception as e:
        logger.error(f"CoinGecko trending failed: {e}")
        return []


async def get_top_coins(limit: int = 50) -> List[Dict]:
    """Get top coins by market cap."""
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                f"{COINGECKO_BASE_URL}/coins/markets",
                params={
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": str(limit),
                    "page": "1",
                    "sparkline": "true",
                    "price_change_percentage": "1h,24h,7d,30d",
                },
                headers={"Accept": "application/json"},
            )
            if resp.status_code != 200:
                return []

            data = resp.json()
            return [
                {
                    "id": c["id"],
                    "symbol": c["symbol"].upper(),
                    "name": c["name"],
                    "image": c.get("image", ""),
                    "price": c.get("current_price", 0),
                    "market_cap": c.get("market_cap", 0),
                    "market_cap_rank": c.get("market_cap_rank", 0),
                    "volume_24h": c.get("total_volume", 0),
                    "change_1h": c.get("price_change_percentage_1h_in_currency", 0),
                    "change_24h": c.get("price_change_percentage_24h", 0),
                    "change_7d": c.get("price_change_percentage_7d_in_currency", 0),
                    "change_30d": c.get("price_change_percentage_30d_in_currency", 0),
                    "circulating_supply": c.get("circulating_supply", 0),
                    "total_supply": c.get("total_supply", 0),
                    "ath": c.get("ath", 0),
                    "ath_change_pct": c.get("ath_change_percentage", 0),
                    "sparkline": c.get("sparkline_in_7d", {}).get("price", []),
                }
                for c in data
            ]
    except Exception as e:
        logger.error(f"CoinGecko top coins failed: {e}")
        return []
