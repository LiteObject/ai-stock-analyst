"""SEC EDGAR tools for fetching company filings.

This module provides functions for fetching SEC filings (10-K, 10-Q, 8-K, etc.)
from the SEC EDGAR database.
"""

import logging
from typing import Any, Dict, List

import requests

from tools.cache import cached

# Configure module logger
logger = logging.getLogger(__name__)

# SEC EDGAR base URL
SEC_EDGAR_BASE = "https://data.sec.gov"


@cached(ttl_hours=24)
def get_sec_filings(
    ticker: str, filing_type: str = "10-K", limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Fetch recent SEC filings for a company.

    Args:
        ticker: Stock ticker symbol
        filing_type: Type of filing (10-K, 10-Q, 8-K, etc.)
        limit: Maximum filings to return

    Returns:
        List of filings with date, type, and URL
    """
    try:
        headers = {
            "User-Agent": "AI-Stock-Analyst/1.0 (Educational Project)",
            "Accept": "application/json",
        }

        # Use SEC's company tickers JSON file
        tickers_url = "https://www.sec.gov/files/company_tickers.json"
        response = requests.get(tickers_url, headers=headers, timeout=30)
        response.raise_for_status()

        tickers_data = response.json()

        # Find CIK for ticker
        cik = None
        for entry in tickers_data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                cik = str(entry.get("cik_str", "")).zfill(10)
                break

        if not cik:
            logger.warning(f"Could not find CIK for ticker {ticker}")
            return []

        # Get filings for the CIK
        submissions_url = f"{SEC_EDGAR_BASE}/submissions/CIK{cik}.json"
        response = requests.get(submissions_url, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()
        filings = data.get("filings", {}).get("recent", {})

        if not filings:
            return []

        forms = filings.get("form", [])
        dates = filings.get("filingDate", [])
        accession_numbers = filings.get("accessionNumber", [])
        primary_docs = filings.get("primaryDocument", [])

        result = []
        for i, form in enumerate(forms):
            if filing_type.upper() in form.upper():
                accession = accession_numbers[i].replace("-", "")
                doc_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{cik.lstrip('0')}/{accession}/{primary_docs[i]}"
                )

                result.append(
                    {
                        "form": form,
                        "filing_date": dates[i],
                        "accession_number": accession_numbers[i],
                        "document_url": doc_url,
                    }
                )

                if len(result) >= limit:
                    break

        logger.info(f"Retrieved {len(result)} {filing_type} filings for {ticker}")
        return result

    except Exception as e:
        logger.error(f"Failed to fetch SEC filings for {ticker}: {e}")
        return []
