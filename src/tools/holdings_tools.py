"""Holdings data tools for institutional and major holder information.

This module provides functions for fetching institutional holders,
major holders breakdown, and ownership information from Yahoo Finance.
"""

import logging
from typing import Any, Dict, List

from tools.cache import cached
from tools.data_sources import DataSourceRegistry, DataSourceType

# Configure module logger
logger = logging.getLogger(__name__)


@cached(ttl_hours=24)
def get_institutional_holders(ticker: str) -> List[Dict[str, Any]]:
    """
    Fetch institutional holders from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol

    Returns:
        List of top institutional holders with shares and percentage
    """
    if not DataSourceRegistry.is_available(DataSourceType.YAHOO_FINANCE):
        logger.warning("Yahoo Finance not available")
        return []

    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        holders = stock.institutional_holders

        if holders is None or (hasattr(holders, "empty") and holders.empty):
            logger.warning(f"No institutional holders found for {ticker}")
            return []

        result = []
        for _, row in holders.iterrows():
            result.append(
                {
                    "holder": row.get("Holder", "Unknown"),
                    "shares": int(row.get("Shares", 0)),
                    "date_reported": str(row.get("Date Reported", "")),
                    "percent_out": (
                        float(row.get("% Out", 0)) if row.get("% Out") else None
                    ),
                    "value": int(row.get("Value", 0)) if row.get("Value") else None,
                }
            )

        logger.info(f"Retrieved {len(result)} institutional holders for {ticker}")
        return result

    except Exception as e:
        logger.error(f"Failed to fetch institutional holders for {ticker}: {e}")
        return []


@cached(ttl_hours=24)
def get_major_holders_breakdown(ticker: str) -> Dict[str, Any]:
    """
    Fetch major holders breakdown (insider vs institutional).

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dict with insider and institutional ownership percentages
    """
    if not DataSourceRegistry.is_available(DataSourceType.YAHOO_FINANCE):
        return {}

    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        major_holders = stock.major_holders

        if major_holders is None or (
            hasattr(major_holders, "empty") and major_holders.empty
        ):
            return {}

        # Parse the major holders DataFrame
        result: Dict[str, Any] = {}
        for _, row in major_holders.iterrows():
            key = str(row.iloc[1]).lower().replace(" ", "_").replace("%", "pct")
            try:
                value = float(str(row.iloc[0]).replace("%", ""))
                result[key] = value
            except (ValueError, TypeError):
                result[key] = str(row.iloc[0])

        logger.info(f"Retrieved major holders breakdown for {ticker}")
        return result

    except Exception as e:
        logger.error(f"Failed to fetch major holders for {ticker}: {e}")
        return {}
