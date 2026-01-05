"""
Market data service
"""

import asyncio
from datetime import datetime
from typing import Dict, List

import yfinance as yf


class MarketService:
    """Service for fetching market data"""

    @staticmethod
    async def get_quote(symbol: str) -> Dict:
        """Get real-time quote for a symbol"""

        def _fetch():
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                "symbol": symbol,
                "price": info.get("currentPrice", 0),
                "change": info.get("regularMarketChange", 0),
                "changePct": info.get("regularMarketChangePercent", 0),
                "volume": info.get("volume", 0),
                "marketCap": info.get("marketCap", 0),
                "high": info.get("dayHigh", 0),
                "low": info.get("dayLow", 0),
                "open": info.get("open", 0),
                "previousClose": info.get("previousClose", 0),
                "timestamp": datetime.now().isoformat(),
            }

        return await asyncio.to_thread(_fetch)

    @staticmethod
    async def get_quotes(symbols: List[str]) -> List[Dict]:
        """Get quotes for multiple symbols"""
        tasks = [MarketService.get_quote(symbol) for symbol in symbols]
        return await asyncio.gather(*tasks)

    @staticmethod
    async def get_historical_data(symbol: str, period: str = "1mo", interval: str = "1d") -> Dict:
        """Get historical data"""

        def _fetch():
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)

            return {"symbol": symbol, "data": hist.reset_index().to_dict(orient="records")}

        return await asyncio.to_thread(_fetch)

    @staticmethod
    async def search_symbols(query: str) -> List[Dict]:
        """Search for symbols"""

        def _search():
            # This is a simple implementation
            # In production, use a proper symbol search API
            try:
                ticker = yf.Ticker(query.upper())
                info = ticker.info
                return [
                    {
                        "symbol": query.upper(),
                        "name": info.get("longName", query),
                        "type": info.get("quoteType", "EQUITY"),
                        "exchange": info.get("exchange", ""),
                    }
                ]
            except Exception:
                return []

        return await asyncio.to_thread(_search)
