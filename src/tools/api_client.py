"""Core API client utilities for financial data fetching.

This module provides shared HTTP session management, authentication headers,
and base URL configuration used by all tool modules.
"""

import logging
import os
from typing import Dict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure module logger
logger = logging.getLogger(__name__)

# Constants
BASE_URL = "https://api.financialdatasets.ai"
DEFAULT_TIMEOUT = 30  # seconds
MAX_RETRIES = 3

# Check if yfinance is available
try:
    import yfinance  # noqa: F401

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not installed. Yahoo Finance fallback unavailable.")


def get_session() -> requests.Session:
    """Create a requests session with retry logic.

    Returns:
        requests.Session: Configured session with retry strategy
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def get_headers() -> Dict[str, str]:
    """Get API headers with authentication.

    Returns:
        Dict[str, str]: Headers dictionary with API key if available
    """
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key
    return headers


def get_yfinance_ticker(ticker: str):
    """Get a yfinance Ticker object if yfinance is available.

    Args:
        ticker: Stock ticker symbol

    Returns:
        yfinance.Ticker object

    Raises:
        ImportError: If yfinance is not installed
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError(
            "yfinance is not installed. Install with: pip install yfinance"
        )
    import yfinance as yf

    return yf.Ticker(ticker)
