"""
Crypto Analytics Service — advanced analytics computations for cryptocurrency markets.

Provides:
  - Correlation matrix between coins
  - Momentum signal scanning
  - Sector performance breakdown
  - Volatility analysis
  - Price prediction scoring
  - Fear & Greed index
  - Technical signal analysis
  - BTC dominance / altcoin season detection
  - Portfolio optimization (mean-variance)

All functions are async and use httpx for CoinGecko API calls.
CPU-bound numpy/pandas work is offloaded via asyncio.to_thread.
"""

import asyncio
import logging
import time as _time
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_URL = "https://api.coingecko.com/api/v3"
_HEADERS = {"Accept": "application/json"}

# ── In-memory TTL cache for CoinGecko responses ─────────────────────
_cache: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 300  # 5 minutes


def _cache_get(key: str) -> Any | None:
    """Return cached value if still valid, else None."""
    entry = _cache.get(key)
    if entry and (_time.monotonic() - entry[0]) < _CACHE_TTL:
        return entry[1]
    return None


def _cache_set(key: str, value: Any) -> None:
    """Store value in cache with current timestamp."""
    _cache[key] = (_time.monotonic(), value)
    # Prune stale entries periodically (keep cache bounded)
    if len(_cache) > 200:
        now = _time.monotonic()
        stale = [k for k, (t, _) in _cache.items() if now - t > _CACHE_TTL]
        for k in stale:
            del _cache[k]


# ── Sector definitions ───────────────────────────────────────────────────

CRYPTO_SECTORS = {
    "Layer 1": ["bitcoin", "ethereum", "solana", "cardano", "avalanche-2", "near", "aptos", "sui"],
    "Layer 2": ["matic-network", "arbitrum", "optimism"],
    "DeFi": ["uniswap", "aave", "maker", "chainlink"],
    "Meme": ["dogecoin", "shiba-inu", "pepe"],
    "Store of Value": ["bitcoin", "litecoin"],
    "AI & Data": ["chainlink", "filecoin", "near"],
}

# Collect all unique coin IDs used across sectors
_ALL_SECTOR_COINS: set[str] = set()
for _coins in CRYPTO_SECTORS.values():
    _ALL_SECTOR_COINS.update(_coins)


# ── Helper: rate-limited CoinGecko fetcher ───────────────────────────────


async def _cg_get(client: httpx.AsyncClient, path: str, params: dict | None = None) -> Any:
    """Make a GET request to CoinGecko with caching and rate limiting."""
    cache_key = f"{path}:{sorted((params or {}).items())}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        resp = await client.get(f"{BASE_URL}{path}", params=params or {}, headers=_HEADERS)
        if resp.status_code == 429:
            logger.warning("CoinGecko rate limited — waiting 60s")
            await asyncio.sleep(60)
            resp = await client.get(f"{BASE_URL}{path}", params=params or {}, headers=_HEADERS)
        if resp.status_code != 200:
            logger.error(f"CoinGecko {path} returned HTTP {resp.status_code}")
            return None
        data = resp.json()
        _cache_set(cache_key, data)
        return data
    except Exception as e:
        logger.error(f"CoinGecko request failed ({path}): {e}")
        return None


def _period_to_days(period: str) -> int:
    mapping = {"7d": 7, "14d": 14, "30d": 30, "60d": 60, "90d": 90, "180d": 180, "365d": 365}
    return mapping.get(period, 30)


async def _fetch_market_data_batch(coin_ids: list[str], sparkline: bool = True) -> list[dict]:
    """Fetch market data for multiple coins in a single API call (up to 250)."""
    cache_key = f"markets_batch:{','.join(sorted(coin_ids))}:{sparkline}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    ids_str = ",".join(coin_ids)
    async with httpx.AsyncClient(timeout=30) as client:
        data = await _cg_get(
            client,
            "/coins/markets",
            {
                "vs_currency": "usd",
                "ids": ids_str,
                "order": "market_cap_desc",
                "per_page": "250",
                "page": "1",
                "sparkline": str(sparkline).lower(),
                "price_change_percentage": "1h,24h,7d,14d,30d",
            },
        )
    if data:
        _cache_set(cache_key, data)
    return data or []


async def _fetch_price_histories(coin_ids: list[str], days: int) -> dict[str, list[float]]:
    """Fetch daily price histories for multiple coins, using cache to avoid redundant calls."""
    price_series: dict[str, list[float]] = {}
    uncached = []

    for coin_id in coin_ids:
        cache_key = f"prices:{coin_id}:{days}"
        cached = _cache_get(cache_key)
        if cached is not None:
            price_series[coin_id] = cached
        else:
            uncached.append(coin_id)

    if uncached:
        async with httpx.AsyncClient(timeout=30) as client:
            for coin_id in uncached:
                data = await _cg_get(
                    client,
                    f"/coins/{coin_id}/market_chart",
                    {
                        "vs_currency": "usd",
                        "days": str(days),
                    },
                )
                if data and "prices" in data and len(data["prices"]) > 1:
                    prices = [p[1] for p in data["prices"]]
                    _cache_set(f"prices:{coin_id}:{days}", prices)
                    price_series[coin_id] = prices
                await asyncio.sleep(2.5)

    return price_series


# ── Technical indicator helpers (pure numpy, no external TA library) ─────


def _compute_rsi(prices: np.ndarray, period: int = 14) -> float:
    """Compute RSI from a price array. Returns value 0-100."""
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _compute_sma(prices: np.ndarray, period: int) -> float:
    """Simple moving average of the last *period* values."""
    if len(prices) < period:
        return float(np.mean(prices)) if len(prices) > 0 else 0.0
    return float(np.mean(prices[-period:]))


def _compute_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average (full series)."""
    if len(prices) == 0:
        return np.array([])
    alpha = 2.0 / (period + 1)
    ema = np.empty_like(prices, dtype=float)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema


def _compute_macd(prices: np.ndarray) -> tuple[float, float, float]:
    """MACD(12,26,9) — returns (macd_line, signal_line, histogram)."""
    if len(prices) < 26:
        return 0.0, 0.0, 0.0
    ema12 = _compute_ema(prices, 12)
    ema26 = _compute_ema(prices, 26)
    macd_line = ema12 - ema26
    signal = _compute_ema(macd_line, 9)
    hist = macd_line - signal
    return float(macd_line[-1]), float(signal[-1]), float(hist[-1])


def _compute_bollinger(prices: np.ndarray, period: int = 20, num_std: float = 2.0) -> dict:
    """Bollinger Bands — returns dict with upper, middle, lower, width."""
    if len(prices) < period:
        mid = float(np.mean(prices)) if len(prices) > 0 else 0.0
        return {"upper": mid, "middle": mid, "lower": mid, "width": 0.0}
    window = prices[-period:]
    mid = float(np.mean(window))
    std = float(np.std(window, ddof=1)) if len(window) > 1 else 0.0
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid if mid != 0 else 0.0
    return {"upper": upper, "middle": mid, "lower": lower, "width": width}


# ═════════════════════════════════════════════════════════════════════════
# Public analytics functions
# ═════════════════════════════════════════════════════════════════════════


async def compute_correlation_matrix(coin_ids: list[str], period: str = "30d") -> dict:
    """
    Fetch daily price histories and compute a Pearson correlation matrix
    of daily returns across coins.
    """
    days = _period_to_days(period)
    price_series = await _fetch_price_histories(coin_ids, days)

    if len(price_series) < 2:
        return {"coins": list(price_series.keys()), "matrix": [], "period": period, "error": "Not enough data to compute correlations"}

    def _compute():
        df = pd.DataFrame(price_series)
        # Forward-fill then drop remaining NaNs
        df = df.ffill().bfill().dropna(axis=1)
        returns = df.pct_change().dropna()
        if returns.empty or returns.shape[1] < 2:
            return list(df.columns), []
        corr = returns.corr()
        return list(corr.columns), corr.values.tolist()

    coins_out, matrix = await asyncio.to_thread(_compute)
    # Round matrix values
    matrix_rounded = [[round(v, 4) for v in row] for row in matrix]
    return {"coins": coins_out, "matrix": matrix_rounded, "period": period}


async def compute_momentum_signals(coins_data: list[dict]) -> list[dict]:
    """
    Compute momentum signals for a list of coins (from get_top_coins with sparkline).
    Uses RSI, pseudo-MACD, and volume momentum.
    """

    def _analyse(coins: list[dict]) -> list[dict]:
        results = []
        for coin in coins:
            sparkline = coin.get("sparkline", [])
            if not sparkline or len(sparkline) < 20:
                results.append(
                    {
                        "coin": coin.get("id", ""),
                        "symbol": coin.get("symbol", ""),
                        "name": coin.get("name", ""),
                        "price": coin.get("price", 0),
                        "signal": "NEUTRAL",
                        "rsi": 50.0,
                        "score": 0,
                        "factors": {"rsi": "neutral", "macd": "neutral", "momentum": "neutral"},
                    }
                )
                continue

            prices = np.array(sparkline, dtype=float)

            # RSI
            rsi = _compute_rsi(prices, 14)

            # MACD-like from sparkline
            macd_line, signal_line, hist = _compute_macd(prices)

            # Short-term momentum: last 24h vs last 7d avg
            recent = prices[-24:] if len(prices) >= 24 else prices
            older = prices[:-24] if len(prices) > 24 else prices
            recent_avg = float(np.mean(recent))
            older_avg = float(np.mean(older)) if len(older) > 0 else recent_avg
            momentum_pct = ((recent_avg - older_avg) / older_avg * 100) if older_avg != 0 else 0.0

            # Composite score (-100 to 100)
            score = 0.0
            # RSI contribution
            if rsi < 30:
                score += 30
            elif rsi < 40:
                score += 15
            elif rsi > 70:
                score -= 30
            elif rsi > 60:
                score -= 15

            # MACD contribution
            if hist > 0:
                score += min(25, abs(hist) / (abs(macd_line) + 1e-9) * 25)
            else:
                score -= min(25, abs(hist) / (abs(macd_line) + 1e-9) * 25)

            # Momentum contribution
            score += max(-25, min(25, momentum_pct * 2.5))

            # Volume contribution (use 24h change as proxy)
            change_24h = coin.get("change_24h") or 0
            score += max(-20, min(20, change_24h * 2))

            score = max(-100, min(100, score))

            # Signal label
            if score >= 50:
                signal = "STRONG_BUY"
            elif score >= 20:
                signal = "BUY"
            elif score <= -50:
                signal = "STRONG_SELL"
            elif score <= -20:
                signal = "SELL"
            else:
                signal = "NEUTRAL"

            factors = {
                "rsi": "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral",
                "macd": "bullish" if hist > 0 else "bearish",
                "momentum": "positive" if momentum_pct > 0 else "negative",
                "change_24h": round(change_24h, 2),
            }

            results.append(
                {
                    "coin": coin.get("id", ""),
                    "symbol": coin.get("symbol", ""),
                    "name": coin.get("name", ""),
                    "price": coin.get("price", 0),
                    "market_cap": coin.get("market_cap", 0),
                    "rsi": round(rsi, 2),
                    "score": round(score, 1),
                    "signal": signal,
                    "factors": factors,
                }
            )

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    return await asyncio.to_thread(_analyse, coins_data)


async def compute_sector_performance() -> list[dict]:
    """
    Calculate average performance per crypto sector using CoinGecko bulk market data.
    """
    all_ids = list(_ALL_SECTOR_COINS)
    ids_str = ",".join(all_ids)

    async with httpx.AsyncClient(timeout=30) as client:
        data = await _cg_get(
            client,
            "/coins/markets",
            {
                "vs_currency": "usd",
                "ids": ids_str,
                "order": "market_cap_desc",
                "per_page": "250",
                "page": "1",
                "sparkline": "false",
                "price_change_percentage": "24h,7d",
            },
        )

    if not data:
        return []

    # Index by id
    coin_map: dict[str, dict] = {c["id"]: c for c in data}

    def _build_sectors():
        results = []
        for sector_name, coin_ids in CRYPTO_SECTORS.items():
            coins_info = []
            changes_24h = []
            changes_7d = []
            total_mcap = 0.0

            for cid in coin_ids:
                c = coin_map.get(cid)
                if not c:
                    continue
                ch24 = c.get("price_change_percentage_24h_in_currency") or c.get("price_change_percentage_24h") or 0
                ch7d = c.get("price_change_percentage_7d_in_currency") or 0
                mcap = c.get("market_cap") or 0
                coins_info.append(
                    {
                        "id": cid,
                        "symbol": (c.get("symbol") or "").upper(),
                        "name": c.get("name", ""),
                        "price": c.get("current_price", 0),
                        "change_24h": round(ch24, 2),
                        "change_7d": round(ch7d, 2),
                        "market_cap": mcap,
                    }
                )
                changes_24h.append(ch24)
                changes_7d.append(ch7d)
                total_mcap += mcap

            if not coins_info:
                continue

            avg_24h = float(np.mean(changes_24h)) if changes_24h else 0.0
            avg_7d = float(np.mean(changes_7d)) if changes_7d else 0.0
            top = max(coins_info, key=lambda x: x["change_24h"])

            results.append(
                {
                    "sector": sector_name,
                    "coins": coins_info,
                    "avg_change_24h": round(avg_24h, 2),
                    "avg_change_7d": round(avg_7d, 2),
                    "total_market_cap": total_mcap,
                    "top_performer": top,
                }
            )

        results.sort(key=lambda x: x["avg_change_24h"], reverse=True)
        return results

    return await asyncio.to_thread(_build_sectors)


async def compute_volatility_analysis(coin_ids: list[str], period: str = "30d") -> list[dict]:
    """
    Compute realized volatility, Bollinger width, and risk classification for each coin.
    """
    days = _period_to_days(period)
    raw_prices = await _fetch_price_histories(coin_ids, days)
    coin_data = {k: {"prices": np.array(v, dtype=float)} for k, v in raw_prices.items() if len(v) > 2}

    if not coin_data:
        return []

    def _compute():
        results = []
        for coin_id, info in coin_data.items():
            prices = info["prices"]
            returns = np.diff(np.log(prices))
            returns = returns[np.isfinite(returns)]

            if len(returns) < 2:
                continue

            daily_vol = float(np.std(returns, ddof=1))
            annualized_vol = daily_vol * np.sqrt(365) * 100  # percentage

            # Bollinger width
            bb = _compute_bollinger(prices, 20, 2.0)
            bollinger_width = round(bb["width"] * 100, 2)  # percentage

            # Average true range proxy (mean of abs daily returns)
            atr_proxy = float(np.mean(np.abs(returns))) * 100

            # Risk classification
            if annualized_vol < 40:
                risk = "LOW"
            elif annualized_vol < 80:
                risk = "MEDIUM"
            elif annualized_vol < 120:
                risk = "HIGH"
            else:
                risk = "EXTREME"

            results.append(
                {
                    "coin": coin_id,
                    "volatility_annual": round(annualized_vol, 2),
                    "volatility_daily": round(daily_vol * 100, 4),
                    "bollinger_width": bollinger_width,
                    "atr_proxy": round(atr_proxy, 4),
                    "risk_level": risk,
                    "current_price": float(prices[-1]),
                    "period": period,
                }
            )

        # Sort by volatility descending and assign rank
        results.sort(key=lambda x: x["volatility_annual"], reverse=True)
        for i, r in enumerate(results, 1):
            r["volatility_rank"] = i

        return results

    return await asyncio.to_thread(_compute)


async def compute_predictions(coin_ids: list[str]) -> list[dict]:
    """
    Simple multi-factor scoring model for short-term price direction.
    Uses momentum, mean-reversion, volume, and trend signals.
    """
    # Fetch bulk market data with sparkline and extended change percentages
    ids_str = ",".join(coin_ids[:20])

    async with httpx.AsyncClient(timeout=30) as client:
        market_data = await _cg_get(
            client,
            "/coins/markets",
            {
                "vs_currency": "usd",
                "ids": ids_str,
                "order": "market_cap_desc",
                "per_page": "250",
                "page": "1",
                "sparkline": "true",
                "price_change_percentage": "1h,24h,7d,14d,30d",
            },
        )

    if not market_data:
        return []

    def _score(coins: list[dict]) -> list[dict]:
        results = []
        for c in coins:
            sparkline = (c.get("sparkline_in_7d") or {}).get("price", [])
            prices = np.array(sparkline, dtype=float) if sparkline else np.array([])

            ch_1d = c.get("price_change_percentage_24h_in_currency") or c.get("price_change_percentage_24h") or 0
            ch_7d = c.get("price_change_percentage_7d_in_currency") or 0
            ch_14d = c.get("price_change_percentage_14d_in_currency") or 0
            ch_30d = c.get("price_change_percentage_30d_in_currency") or 0

            # 1. Momentum score: weighted recent returns
            # Approximate 3d from 1d and 7d
            ch_3d = (ch_1d * 0.4 + ch_7d * 0.6) * 0.43  # rough approximation
            momentum = ch_1d * 0.30 + ch_3d * 0.25 + ch_7d * 0.25 + ch_14d * 0.20
            momentum_score = max(-30, min(30, momentum))

            # 2. Mean reversion (RSI-based)
            rsi = _compute_rsi(prices, 14) if len(prices) > 15 else 50.0
            if rsi < 25:
                reversion_score = 25.0
            elif rsi < 35:
                reversion_score = 15.0
            elif rsi > 75:
                reversion_score = -25.0
            elif rsi > 65:
                reversion_score = -15.0
            else:
                reversion_score = 0.0

            # 3. Volume score (use 24h volume vs market cap as proxy)
            vol = c.get("total_volume") or 0
            mcap = c.get("market_cap") or 1
            vol_ratio = vol / mcap if mcap > 0 else 0
            # Higher vol/mcap ratio can indicate interest
            volume_score = max(-15, min(15, (vol_ratio - 0.05) * 150))

            # 4. Trend score: price vs SMA approximation
            if len(prices) >= 20:
                sma20 = _compute_sma(prices, 20)
                current = float(prices[-1])
                pct_above_sma = ((current - sma20) / sma20 * 100) if sma20 != 0 else 0
                trend_score = max(-20, min(20, pct_above_sma * 2))
            else:
                trend_score = 0.0

            # Composite
            total_score = momentum_score + reversion_score + volume_score + trend_score
            total_score = max(-100, min(100, total_score))

            # Confidence: higher when signals agree
            signals_positive = sum(1 for s in [momentum_score, reversion_score, volume_score, trend_score] if s > 0)
            signals_negative = sum(1 for s in [momentum_score, reversion_score, volume_score, trend_score] if s < 0)
            agreement = max(signals_positive, signals_negative)
            confidence = min(95, int(abs(total_score) * 0.6 + agreement * 10))

            if total_score >= 20:
                prediction = "BULLISH"
            elif total_score <= -20:
                prediction = "BEARISH"
            else:
                prediction = "NEUTRAL"

            # Projected 7d change — very rough heuristic
            projected = total_score * 0.15  # +-15% max

            # Build key_factors: top 3 contributing factors sorted by absolute score
            factor_scores = {
                "momentum": momentum_score,
                "mean_reversion": reversion_score,
                "volume": volume_score,
                "trend": trend_score,
            }
            sorted_factors = sorted(factor_scores.items(), key=lambda x: abs(x[1]), reverse=True)
            key_factors = [f"{name} {val:+.1f}" for name, val in sorted_factors[:3]]
            if rsi < 30:
                key_factors.append(f"RSI oversold ({rsi:.0f})")
            elif rsi > 70:
                key_factors.append(f"RSI overbought ({rsi:.0f})")

            results.append(
                {
                    "coin_id": c.get("id", ""),
                    "symbol": (c.get("symbol") or "").upper(),
                    "name": c.get("name", ""),
                    "price": c.get("current_price", 0),
                    "prediction": prediction,
                    "confidence": confidence,
                    "score": round(total_score, 1),
                    "factors": {
                        "momentum": round(momentum_score, 2),
                        "mean_reversion": round(reversion_score, 2),
                        "volume": round(volume_score, 2),
                        "trend": round(trend_score, 2),
                        "rsi": round(rsi, 2),
                    },
                    "key_factors": key_factors[:3],
                    "projected_7d": round(projected, 2),
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    return await asyncio.to_thread(_score, market_data)


async def compute_fear_greed_index(overview_data: dict, top_coins: list[dict]) -> dict:
    """
    Custom crypto Fear & Greed index calculated from volatility, volume,
    BTC dominance, and market momentum.
    """
    btc_prices_raw = await _fetch_price_histories(["bitcoin"], 30)
    btc_prices = btc_prices_raw.get("bitcoin", [])

    def _compute():
        # 1. Volatility component (25%) — lower vol = more greed
        vol_score = 50.0
        if btc_prices and len(btc_prices) > 2:
            btc_arr = np.array(btc_prices, dtype=float)
            returns = np.diff(np.log(btc_arr))
            returns = returns[np.isfinite(returns)]
            if len(returns) > 1:
                daily_vol = float(np.std(returns, ddof=1))
                annual_vol = daily_vol * np.sqrt(365) * 100
                vol_score = max(0, min(100, 100 - (annual_vol - 30) * 1.2))

        # 2. Volume component (25%) — high volume = more greed
        total_vol_24h = overview_data.get("total_volume_24h_usd", 0)
        total_mcap = overview_data.get("total_market_cap_usd", 1)
        vol_mcap_ratio = total_vol_24h / total_mcap if total_mcap > 0 else 0
        # Typical range: 0.03-0.10
        volume_score = max(0, min(100, (vol_mcap_ratio - 0.02) * 1000))

        # 3. BTC Dominance component (25%)
        btc_dom = overview_data.get("bitcoin_dominance", 50)
        # Higher dominance = more fear (flight to safety), lower = greed (risk-on)
        dom_score = max(0, min(100, (70 - btc_dom) * 2.5))

        # 4. Momentum component (25%) — % of top coins in green
        green_count = sum(1 for c in top_coins[:20] if (c.get("change_24h") or 0) > 0)
        total_count = min(20, len(top_coins))
        green_pct = (green_count / total_count * 100) if total_count > 0 else 50
        momentum_score = green_pct

        # Weighted composite
        composite = vol_score * 0.25 + volume_score * 0.25 + dom_score * 0.25 + momentum_score * 0.25
        composite = max(0, min(100, composite))

        # Label
        if composite <= 20:
            label = "Extreme Fear"
            desc = "Market is in extreme fear. Historically this can be a buying opportunity, but caution is warranted."
        elif composite <= 40:
            label = "Fear"
            desc = "Investors are fearful. Sentiment is negative and risk appetite is low."
        elif composite <= 60:
            label = "Neutral"
            desc = "Market sentiment is balanced. Neither fear nor greed dominates."
        elif composite <= 80:
            label = "Greed"
            desc = "Investors are greedy. Prices may be overextended — consider taking profits."
        else:
            label = "Extreme Greed"
            desc = "Market euphoria. Extreme caution advised — corrections often follow extreme greed."

        return {
            "value": round(composite, 1),
            "label": label,
            "description": desc,
            "components": {
                "volatility": {"score": round(vol_score, 1), "weight": 0.25},
                "volume": {"score": round(volume_score, 1), "weight": 0.25},
                "dominance": {"score": round(dom_score, 1), "weight": 0.25},
                "momentum": {
                    "score": round(momentum_score, 1),
                    "weight": 0.25,
                    "green_coins": green_count,
                    "total_coins": total_count,
                },
            },
        }

    return await asyncio.to_thread(_compute)


async def compute_technical_signals(coin_ids: list[str]) -> list[dict]:
    """
    Full technical analysis for each coin: RSI, SMA cross, MACD, Bollinger.
    Uses 90 days of daily data.
    """
    raw_prices = await _fetch_price_histories(coin_ids, 90)
    coin_prices = {k: np.array(v, dtype=float) for k, v in raw_prices.items() if len(v) > 10}

    if not coin_prices:
        return []

    def _analyse():
        results = []
        for coin_id, prices in coin_prices.items():
            signals: dict[str, dict] = {}
            buy_signals = 0
            sell_signals = 0

            # RSI(14)
            rsi = _compute_rsi(prices, 14)
            if rsi < 30:
                rsi_signal = "BUY"
                buy_signals += 1
            elif rsi > 70:
                rsi_signal = "SELL"
                sell_signals += 1
            else:
                rsi_signal = "NEUTRAL"
            signals["rsi"] = {"value": round(rsi, 2), "signal": rsi_signal}

            # SMA Cross (20 vs 50)
            sma20 = _compute_sma(prices, 20)
            sma50 = _compute_sma(prices, 50)
            current_price = float(prices[-1])
            if sma20 > sma50 and current_price > sma20:
                sma_signal = "BUY"
                buy_signals += 1
            elif sma20 < sma50 and current_price < sma20:
                sma_signal = "SELL"
                sell_signals += 1
            else:
                sma_signal = "NEUTRAL"
            signals["sma_cross"] = {
                "sma20": round(sma20, 2),
                "sma50": round(sma50, 2),
                "price": round(current_price, 2),
                "signal": sma_signal,
            }

            # MACD(12,26,9)
            macd_line, signal_line, histogram = _compute_macd(prices)
            if histogram > 0 and macd_line > signal_line:
                macd_signal = "BUY"
                buy_signals += 1
            elif histogram < 0 and macd_line < signal_line:
                macd_signal = "SELL"
                sell_signals += 1
            else:
                macd_signal = "NEUTRAL"
            signals["macd"] = {
                "macd_line": round(macd_line, 4),
                "signal_line": round(signal_line, 4),
                "histogram": round(histogram, 4),
                "signal": macd_signal,
            }

            # Bollinger Bands(20,2)
            bb = _compute_bollinger(prices, 20, 2.0)
            if current_price < bb["lower"]:
                bb_signal = "BUY"
                buy_signals += 1
            elif current_price > bb["upper"]:
                bb_signal = "SELL"
                sell_signals += 1
            else:
                bb_signal = "NEUTRAL"
            signals["bollinger"] = {
                "upper": round(bb["upper"], 2),
                "middle": round(bb["middle"], 2),
                "lower": round(bb["lower"], 2),
                "width": round(bb["width"] * 100, 2),
                "signal": bb_signal,
            }

            # Overall — majority vote
            if buy_signals > sell_signals:
                overall = "BUY"
            elif sell_signals > buy_signals:
                overall = "SELL"
            else:
                overall = "NEUTRAL"

            results.append(
                {
                    "coin": coin_id,
                    "current_price": round(current_price, 2),
                    "signals": signals,
                    "overall": overall,
                    "buy_signals": buy_signals,
                    "sell_signals": sell_signals,
                    "neutral_signals": 4 - buy_signals - sell_signals,
                }
            )

        return results

    return await asyncio.to_thread(_analyse)


async def compute_btc_dominance_analysis(overview_data: dict, top_coins: list[dict]) -> dict:
    """
    Analyze BTC dominance and detect altcoin season.
    """
    btc_prices_raw = await _fetch_price_histories(["bitcoin"], 90)
    btc_chart_prices = btc_prices_raw.get("bitcoin", [])

    def _analyse():
        btc_dominance = overview_data.get("bitcoin_dominance", 0)

        # Dominance trend — approximate from price trend relative to total market
        trend = "stable"
        if btc_chart_prices and len(btc_chart_prices) > 30:
            recent_avg = np.mean(btc_chart_prices[-7:])
            older_avg = np.mean(btc_chart_prices[-30:-7])
            pct_change = ((recent_avg - older_avg) / older_avg * 100) if older_avg else 0
            if pct_change > 3:
                trend = "rising"
            elif pct_change < -3:
                trend = "falling"

        # Altcoin season index: % of top 50 alts outperforming BTC in 90d
        btc_change_90d = 0
        btc_coin = None
        for c in top_coins:
            if c.get("id") == "bitcoin":
                btc_coin = c
                btc_change_90d = c.get("change_30d", 0) or 0  # Use 30d as proxy for available data
                break

        alts = [c for c in top_coins if c.get("id") != "bitcoin"][:49]
        outperform_count = 0
        for alt in alts:
            alt_change = alt.get("change_30d", 0) or 0
            if alt_change > btc_change_90d:
                outperform_count += 1

        total_alts = len(alts) if alts else 1
        altcoin_season_index = round(outperform_count / total_alts * 100, 1)
        is_altcoin_season = altcoin_season_index >= 75

        # Analysis text
        if is_altcoin_season:
            analysis = (
                f"Altcoin Season detected. {outperform_count} of {total_alts} altcoins are "
                f"outperforming Bitcoin. BTC dominance at {btc_dominance:.1f}% is {trend}. "
                "Risk-on environment favors altcoin allocation."
            )
        elif altcoin_season_index <= 25:
            analysis = (
                f"Bitcoin Season. Only {outperform_count} of {total_alts} altcoins outperform BTC. "
                f"BTC dominance at {btc_dominance:.1f}% is {trend}. "
                "Capital is flowing to Bitcoin — consider BTC-heavy positioning."
            )
        else:
            analysis = (
                f"Mixed market. {outperform_count} of {total_alts} altcoins outperform BTC. "
                f"BTC dominance at {btc_dominance:.1f}% is {trend}. "
                "Selective altcoin exposure is appropriate."
            )

        return {
            "btc_dominance": round(btc_dominance, 2),
            "trend": trend,
            "altcoin_season_index": altcoin_season_index,
            "is_altcoin_season": is_altcoin_season,
            "alts_outperforming_btc": outperform_count,
            "total_alts_tracked": total_alts,
            "analysis": analysis,
        }

    return await asyncio.to_thread(_analyse)


async def compute_portfolio_optimization(holdings: list[dict]) -> dict:
    """
    Mean-variance portfolio optimization.
    Input: list of {"coin_id": str, "amount_usd": float}
    """
    if not holdings or len(holdings) < 2:
        return {"error": "At least 2 holdings are required for portfolio optimization"}

    coin_ids = [h["coin_id"] for h in holdings]
    amounts = {h["coin_id"]: h["amount_usd"] for h in holdings}
    total_value = sum(amounts.values())
    if total_value <= 0:
        return {"error": "Total portfolio value must be positive"}

    # Fetch 90d price data for each coin
    price_series = await _fetch_price_histories(coin_ids, 90)

    valid_coins = [c for c in coin_ids if c in price_series]
    if len(valid_coins) < 2:
        return {"error": "Not enough price data for optimization"}

    def _optimize():
        # Build returns DataFrame
        df = pd.DataFrame({c: price_series[c] for c in valid_coins})
        df = df.ffill().bfill().dropna(axis=1)

        remaining_coins = list(df.columns)
        if len(remaining_coins) < 2:
            return {"error": "Not enough aligned data for optimization"}

        returns = df.pct_change().dropna()
        if returns.empty:
            return {"error": "Not enough return data"}

        mean_returns = returns.mean().values  # daily
        cov_matrix = returns.cov().values  # daily

        n = len(remaining_coins)

        # Current weights
        current_total = sum(amounts.get(c, 0) for c in remaining_coins)
        current_weights = np.array([amounts.get(c, 0) / current_total for c in remaining_coins])

        # Equal weight
        equal_weights = np.ones(n) / n

        # Minimum variance portfolio (analytical solution for long-only constraint approx)
        try:
            inv_cov = np.linalg.inv(cov_matrix)
            ones = np.ones(n)
            min_var_weights = inv_cov @ ones / (ones @ inv_cov @ ones)
            # Clip negative weights (long-only) and renormalize
            min_var_weights = np.maximum(min_var_weights, 0)
            w_sum = min_var_weights.sum()
            if w_sum > 0:
                min_var_weights = min_var_weights / w_sum
            else:
                min_var_weights = equal_weights.copy()
        except np.linalg.LinAlgError:
            min_var_weights = equal_weights.copy()

        def _portfolio_stats(weights):
            port_return = float(np.sum(mean_returns * weights) * 365)  # annualized
            port_vol = float(np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(365)) * 100
            sharpe = port_return / (port_vol / 100) if port_vol > 0 else 0
            return port_return * 100, port_vol, sharpe  # return as percentage

        cur_ret, cur_vol, cur_sharpe = _portfolio_stats(current_weights)
        opt_ret, opt_vol, opt_sharpe = _portfolio_stats(min_var_weights)
        eq_ret, eq_vol, eq_sharpe = _portfolio_stats(equal_weights)

        def _build_allocations(weights):
            return [
                {"coin": remaining_coins[i], "weight": round(float(weights[i]), 4), "amount_usd": round(float(weights[i] * total_value), 2)}
                for i in range(n)
            ]

        return {
            "coins": remaining_coins,
            "total_value_usd": round(total_value, 2),
            "current": {
                "allocations": _build_allocations(current_weights),
                "expected_return_annual": round(cur_ret, 2),
                "expected_volatility": round(cur_vol, 2),
                "sharpe_ratio": round(cur_sharpe, 3),
            },
            "optimized": {
                "allocations": _build_allocations(min_var_weights),
                "expected_return_annual": round(opt_ret, 2),
                "expected_volatility": round(opt_vol, 2),
                "sharpe_ratio": round(opt_sharpe, 3),
                "method": "minimum_variance",
            },
            "equal_weight": {
                "allocations": _build_allocations(equal_weights),
                "expected_return_annual": round(eq_ret, 2),
                "expected_volatility": round(eq_vol, 2),
                "sharpe_ratio": round(eq_sharpe, 3),
            },
        }

    return await asyncio.to_thread(_optimize)
