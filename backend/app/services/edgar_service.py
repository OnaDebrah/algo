"""
SEC EDGAR Service — Fetch company filings and insider trading data.

Uses the free SEC EDGAR API (no key required, just a User-Agent header).
Endpoints:
  - Company filings (10-K, 10-Q, 8-K, etc.)
  - Insider transactions (Form 4)
  - Company facts (XBRL financial data)

Rate limit: 10 requests/second per SEC policy.
Docs: https://www.sec.gov/search#/dateRange=custom&forms=4
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

SEC_BASE_URL = "https://efts.sec.gov/LATEST"
SEC_DATA_URL = "https://data.sec.gov"
SEC_EDGAR_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
USER_AGENT = "Oraculum Trading Platform support@oraculum.io"

# Map common ticker symbols to CIK numbers (SEC uses CIK, not tickers)
# We'll look these up dynamically via the SEC company search API.

_cik_cache: Dict[str, str] = {}


async def _get_cik(ticker: str) -> Optional[str]:
    """Look up SEC CIK number for a ticker symbol."""
    ticker_upper = ticker.upper()
    if ticker_upper in _cik_cache:
        return _cik_cache[ticker_upper]

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            # SEC provides a JSON mapping of tickers to CIKs
            resp = await client.get(
                f"{SEC_DATA_URL}/submissions/CIK{ticker_upper}.json",
                headers={"User-Agent": USER_AGENT},
            )
            if resp.status_code == 200:
                data = resp.json()
                cik = str(data.get("cik", "")).zfill(10)
                _cik_cache[ticker_upper] = cik
                return cik

            # Fallback: search by company ticker
            resp = await client.get(
                "https://efts.sec.gov/LATEST/search-index?q=%22" + ticker_upper + "%22&dateRange=custom&startdt=2020-01-01&forms=10-K",
                headers={"User-Agent": USER_AGENT},
            )
    except Exception as e:
        logger.warning(f"EDGAR CIK lookup failed for {ticker}: {e}")

    return None


async def get_company_filings(
    ticker: str,
    filing_type: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Get recent SEC filings for a company.

    Args:
        ticker: Stock ticker symbol
        filing_type: Filter by form type (e.g., '10-K', '10-Q', '8-K', '4')
        limit: Max number of filings to return

    Returns:
        List of filing dicts with: form, date, description, url, accession_number
    """
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            # Use the full-text search API (more reliable than CIK lookup)
            params = {
                "q": f'"{ticker}"',
                "dateRange": "custom",
                "startdt": "2020-01-01",
                "enddt": datetime.now().strftime("%Y-%m-%d"),
            }
            if filing_type:
                params["forms"] = filing_type

            resp = await client.get(
                f"{SEC_BASE_URL}/search-index",
                params=params,
                headers={"User-Agent": USER_AGENT},
            )

            if resp.status_code != 200:
                # Fallback: use submissions endpoint
                return await _get_filings_via_submissions(ticker, filing_type, limit)

            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])

            filings = []
            for hit in hits[:limit]:
                source = hit.get("_source", {})
                form = source.get("forms", source.get("form_type", ""))
                if filing_type and form != filing_type:
                    continue

                accession = source.get("file_num", source.get("accession_no", ""))
                filings.append(
                    {
                        "form": form,
                        "date": source.get("file_date", source.get("period_of_report", "")),
                        "description": source.get("display_names", [source.get("entity_name", "")])[0]
                        if source.get("display_names")
                        else source.get("entity_name", ""),
                        "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type={form}&dateb=&owner=include&count=10",
                        "accession_number": accession,
                    }
                )

            return filings

    except Exception as e:
        logger.error(f"EDGAR filing fetch failed for {ticker}: {e}")
        return await _get_filings_via_submissions(ticker, filing_type, limit)


async def _get_filings_via_submissions(
    ticker: str,
    filing_type: Optional[str],
    limit: int,
) -> List[Dict[str, Any]]:
    """Fallback: fetch filings via the SEC submissions API."""
    try:
        # Use the ticker-based submissions endpoint
        async with httpx.AsyncClient(timeout=20) as client:
            # Try direct CIK lookup
            tickers_resp = await client.get(
                f"{SEC_DATA_URL}/submissions/CIK{ticker.upper()}.json",
                headers={"User-Agent": USER_AGENT},
            )

            if tickers_resp.status_code != 200:
                # Try company tickers JSON for mapping
                mapping_resp = await client.get(
                    "https://www.sec.gov/files/company_tickers.json",
                    headers={"User-Agent": USER_AGENT},
                )
                if mapping_resp.status_code == 200:
                    mapping = mapping_resp.json()
                    cik = None
                    for entry in mapping.values():
                        if entry.get("ticker", "").upper() == ticker.upper():
                            cik = str(entry["cik_str"]).zfill(10)
                            break

                    if not cik:
                        return []

                    tickers_resp = await client.get(
                        f"{SEC_DATA_URL}/submissions/CIK{cik}.json",
                        headers={"User-Agent": USER_AGENT},
                    )
                    if tickers_resp.status_code != 200:
                        return []

            data = tickers_resp.json()
            recent = data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            descriptions = recent.get("primaryDocument", [])
            accessions = recent.get("accessionNumber", [])

            filings = []
            company_name = data.get("name", ticker)

            for i in range(min(len(forms), limit * 3)):
                form = forms[i] if i < len(forms) else ""
                if filing_type and form != filing_type:
                    continue

                accession = accessions[i].replace("-", "") if i < len(accessions) else ""
                doc = descriptions[i] if i < len(descriptions) else ""

                filings.append(
                    {
                        "form": form,
                        "date": dates[i] if i < len(dates) else "",
                        "description": company_name,
                        "url": f"{SEC_DATA_URL}/Archives/edgar/data/{data.get('cik', '')}/{accession}/{doc}",
                        "accession_number": accessions[i] if i < len(accessions) else "",
                    }
                )

                if len(filings) >= limit:
                    break

            return filings

    except Exception as e:
        logger.error(f"EDGAR submissions fallback failed for {ticker}: {e}")
        return []


async def get_insider_transactions(ticker: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Get recent insider transactions (Form 4 filings) for a company.

    Returns list of: insider_name, title, transaction_type, shares, price, date, value
    """
    # Form 4 filings are insider transaction reports
    filings = await get_company_filings(ticker, filing_type="4", limit=limit)

    # The filings list from the search doesn't contain transaction details,
    # so we'll return the filing metadata. For full transaction parsing,
    # we'd need to fetch and parse each Form 4 XML — which is a deeper integration.
    transactions = []
    for filing in filings:
        transactions.append(
            {
                "type": "insider_filing",
                "form": filing["form"],
                "date": filing["date"],
                "description": filing["description"],
                "url": filing["url"],
                "accession_number": filing.get("accession_number", ""),
            }
        )

    return transactions


async def get_company_facts(ticker: str) -> Dict[str, Any]:
    """
    Get XBRL financial facts from SEC EDGAR for a company.

    Returns key financial metrics extracted from SEC filings:
    revenue, net_income, total_assets, total_liabilities, etc.
    """
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            # First get the CIK
            mapping_resp = await client.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers={"User-Agent": USER_AGENT},
            )

            cik = None
            if mapping_resp.status_code == 200:
                mapping = mapping_resp.json()
                for entry in mapping.values():
                    if entry.get("ticker", "").upper() == ticker.upper():
                        cik = str(entry["cik_str"]).zfill(10)
                        break

            if not cik:
                return {"error": f"CIK not found for {ticker}"}

            # Fetch company facts
            facts_resp = await client.get(
                f"{SEC_DATA_URL}/api/xbrl/companyfacts/CIK{cik}.json",
                headers={"User-Agent": USER_AGENT},
            )

            if facts_resp.status_code != 200:
                return {"error": f"Failed to fetch facts: HTTP {facts_resp.status_code}"}

            data = facts_resp.json()
            us_gaap = data.get("facts", {}).get("us-gaap", {})

            # Extract key financial metrics
            def _latest_value(concept: str) -> Optional[float]:
                """Get the most recent value for an XBRL concept."""
                concept_data = us_gaap.get(concept, {})
                units = concept_data.get("units", {})
                usd_values = units.get("USD", [])
                if not usd_values:
                    return None
                # Sort by end date, get most recent
                sorted_vals = sorted(usd_values, key=lambda x: x.get("end", ""), reverse=True)
                return sorted_vals[0].get("val") if sorted_vals else None

            return {
                "ticker": ticker,
                "cik": cik,
                "company_name": data.get("entityName", ""),
                "revenue": _latest_value("Revenues") or _latest_value("RevenueFromContractWithCustomerExcludingAssessedTax"),
                "net_income": _latest_value("NetIncomeLoss"),
                "total_assets": _latest_value("Assets"),
                "total_liabilities": _latest_value("Liabilities"),
                "stockholders_equity": _latest_value("StockholdersEquity"),
                "operating_income": _latest_value("OperatingIncomeLoss"),
                "earnings_per_share": _latest_value("EarningsPerShareBasic"),
                "cash_and_equivalents": _latest_value("CashAndCashEquivalentsAtCarryingValue"),
                "long_term_debt": _latest_value("LongTermDebt") or _latest_value("LongTermDebtNoncurrent"),
                "shares_outstanding": _latest_value("CommonStockSharesOutstanding"),
            }

    except Exception as e:
        logger.error(f"EDGAR company facts failed for {ticker}: {e}")
        return {"error": str(e)}
