"""Economic data tools for macro indicators and FRED data.

This module provides functions for fetching economic indicators from FRED
(Federal Reserve Economic Data), including interest rates, inflation,
and yield curve data.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import requests

from tools.cache import cached

# Configure module logger
logger = logging.getLogger(__name__)

# FRED API configuration
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Common FRED series IDs for financial analysis
FRED_SERIES = {
    "fed_funds_rate": "DFF",  # Effective Federal Funds Rate
    "treasury_10y": "DGS10",  # 10-Year Treasury Rate
    "treasury_2y": "DGS2",  # 2-Year Treasury Rate
    "treasury_3m": "DTB3",  # 3-Month Treasury Bill
    "inflation_cpi": "CPIAUCSL",  # Consumer Price Index
    "unemployment": "UNRATE",  # Unemployment Rate
    "gdp_growth": "A191RL1Q225SBEA",  # Real GDP Growth Rate
    "vix": "VIXCLS",  # VIX Volatility Index
    "sp500": "SP500",  # S&P 500 Index
}


@cached(ttl_hours=24)
def get_fred_series(
    series_id: str,
    observation_start: Optional[str] = None,
    observation_end: Optional[str] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Fetch economic data from FRED.

    Note: FRED requires a free API key from
    https://fred.stlouisfed.org/docs/api/api_key.html

    Args:
        series_id: FRED series ID (e.g., "DFF" for Fed Funds Rate)
        observation_start: Start date (YYYY-MM-DD)
        observation_end: End date (YYYY-MM-DD)
        limit: Maximum observations to return

    Returns:
        List of observations with date and value
    """
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        logger.warning(
            "FRED_API_KEY not set. Get free key at: "
            "https://fred.stlouisfed.org/docs/api/api_key.html"
        )
        return []

    try:
        params: Dict[str, Any] = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        }

        if observation_start:
            params["observation_start"] = observation_start
        if observation_end:
            params["observation_end"] = observation_end

        response = requests.get(FRED_BASE_URL, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        observations = data.get("observations", [])

        result = []
        for obs in observations:
            try:
                value = float(obs["value"]) if obs["value"] != "." else None
                result.append({"date": obs["date"], "value": value})
            except (ValueError, KeyError):
                continue

        logger.info(f"Retrieved {len(result)} observations for FRED series {series_id}")
        return result

    except Exception as e:
        logger.error(f"Failed to fetch FRED series {series_id}: {e}")
        return []


@cached(ttl_hours=24)
def get_economic_indicators() -> Dict[str, Any]:
    """
    Fetch common economic indicators from FRED.

    Returns:
        Dict with latest values for key economic indicators
    """
    if not os.environ.get("FRED_API_KEY"):
        return {}

    indicators: Dict[str, Any] = {}

    for name, series_id in FRED_SERIES.items():
        try:
            data = get_fred_series(series_id, limit=1)
            if data:
                indicators[name] = {
                    "value": data[0]["value"],
                    "date": data[0]["date"],
                    "series_id": series_id,
                }
        except Exception as e:
            logger.warning(f"Failed to fetch {name}: {e}")

    return indicators


@cached(ttl_hours=24)
def get_yield_curve() -> Dict[str, Any]:
    """
    Calculate yield curve data (spread between long and short rates).

    Returns:
        Dict with yield curve indicators
    """
    if not os.environ.get("FRED_API_KEY"):
        return {}

    try:
        t10y = get_fred_series("DGS10", limit=1)
        t2y = get_fred_series("DGS2", limit=1)
        t3m = get_fred_series("DTB3", limit=1)

        result: Dict[str, Any] = {}

        if t10y and t2y and t10y[0]["value"] and t2y[0]["value"]:
            spread_10y_2y = t10y[0]["value"] - t2y[0]["value"]
            result["spread_10y_2y"] = round(spread_10y_2y, 3)
            result["is_inverted"] = spread_10y_2y < 0

        if t10y and t3m and t10y[0]["value"] and t3m[0]["value"]:
            spread_10y_3m = t10y[0]["value"] - t3m[0]["value"]
            result["spread_10y_3m"] = round(spread_10y_3m, 3)

        if t10y:
            result["treasury_10y"] = t10y[0]["value"]
        if t2y:
            result["treasury_2y"] = t2y[0]["value"]
        if t3m:
            result["treasury_3m"] = t3m[0]["value"]

        logger.info("Retrieved yield curve data")
        return result

    except Exception as e:
        logger.error(f"Failed to calculate yield curve: {e}")
        return {}
