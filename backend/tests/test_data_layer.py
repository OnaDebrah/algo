import asyncio
import logging

from backend.app.config import settings
from backend.app.core.data_fetcher import fetch_stock_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_providers():
    symbol = "AAPL"
    
    # 1. Test Yahoo (Default)
    logger.info("--- Testing Yahoo Provider (Default) ---")
    settings.DATA_PROVIDER = "yahoo"
    df_yahoo = fetch_stock_data(symbol, "1mo", "1d")
    if not df_yahoo.empty:
        logger.info(f"Yahoo success: {len(df_yahoo)} rows fetched.")
        print(df_yahoo.head())
    else:
        logger.error("Yahoo failed.")

    # 2. Test Polygon (Requires API Key)
    logger.info("\n--- Testing Polygon Provider ---")
    if settings.POLYGON_API_KEY:
        settings.DATA_PROVIDER = "polygon"
        df_poly = fetch_stock_data(symbol, "1mo", "1d")
        if not df_poly.empty:
            logger.info(f"Polygon success: {len(df_poly)} rows fetched.")
            print(df_poly.head())
        else:
            logger.error("Polygon failed.")
    else:
        logger.info("Skipping Polygon test (no API Key set).")

if __name__ == "__main__":
    asyncio.run(test_providers())
