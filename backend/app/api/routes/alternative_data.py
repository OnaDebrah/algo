"""
Alternative data routes — SEC EDGAR filings, insider transactions, and company facts.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query

from ...api.deps import get_current_active_user
from ...models.user import User
from ...services.edgar_service import get_company_facts, get_company_filings, get_insider_transactions

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/alternative-data", tags=["Alternative Data"])


@router.get("/filings/{ticker}")
async def get_filings(
    ticker: str,
    form_type: Optional[str] = Query(None, description="Filter by form type: 10-K, 10-Q, 8-K, 4, etc."),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get recent SEC filings for a company.

    Common form types:
    - **10-K**: Annual report
    - **10-Q**: Quarterly report
    - **8-K**: Current report (material events)
    - **4**: Insider transaction report
    - **S-1**: IPO registration
    - **DEF 14A**: Proxy statement
    """
    filings = await get_company_filings(ticker, filing_type=form_type, limit=limit)
    return {
        "ticker": ticker,
        "form_type": form_type,
        "count": len(filings),
        "filings": filings,
    }


@router.get("/insider-transactions/{ticker}")
async def get_insider_trades(
    ticker: str,
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
):
    """Get recent insider transaction filings (Form 4) for a company."""
    transactions = await get_insider_transactions(ticker, limit=limit)
    return {
        "ticker": ticker,
        "count": len(transactions),
        "transactions": transactions,
    }


@router.get("/company-facts/{ticker}")
async def get_facts(
    ticker: str,
    current_user: User = Depends(get_current_active_user),
):
    """
    Get XBRL financial facts from SEC EDGAR.

    Returns key metrics extracted from the company's most recent filings:
    revenue, net income, total assets, liabilities, EPS, etc.
    """
    facts = await get_company_facts(ticker)
    return facts
