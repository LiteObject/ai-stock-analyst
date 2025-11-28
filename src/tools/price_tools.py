"""Price data tools for fetching OHLCV stock data.

This module provides functions for fetching price data from Financial Datasets API
with Yahoo Finance fallback.
"""

import logging
from typing import Any, Dict, List, cast

import pandas as pd
import requests

from tools.api_client import (BASE_URL, DEFAULT_TIMEOUT, YFINANCE_AVAILABLE,
                              get_headers, get_session, get_yfinance_ticker)
from tools.cache import cached

# Configure module logger
logger = logging.getLogger(__name__)


def _get_prices_from_yfinance(
    ticker: str, start_date: str, end_date: str
) -> List[Dict[str, Any]]:
    """
    Fetch prices from Yahoo Finance as a fallback.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        List of price records with OHLCV data
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError(
            "yfinance is not installed. Install with: pip install yfinance"
        )

    logger.info(f"Fetching prices from Yahoo Finance for {ticker}")

    try:
        stock = get_yfinance_ticker(ticker)
        df = stock.history(start=start_date, end=end_date)

        if df.empty:
            raise ValueError(f"No data returned from Yahoo Finance for {ticker}")

        # Handle both single-level and MultiIndex columns (yfinance behavior varies)
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten MultiIndex by taking first level
            df.columns = df.columns.get_level_values(0)

        prices = []
        for idx, row in df.iterrows():
            # idx is a pandas Timestamp when iterating over DatetimeIndex
            date_str = cast(pd.Timestamp, idx).strftime("%Y-%m-%d")
            prices.append(
                {
                    "time": date_str,
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]),
                }
            )

        logger.info(
            f"Retrieved {len(prices)} price records from Yahoo Finance for {ticker}"
        )
        return prices

    except Exception as e:
        logger.error(f"Failed to fetch prices from Yahoo Finance for {ticker}: {e}")
        raise


@cached
def get_prices(ticker: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """
    Fetch daily price data for a ticker.
    Falls back to Yahoo Finance if Financial Datasets API fails.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        List of price records with OHLCV data

    Raises:
        requests.RequestException: If both APIs fail
        ValueError: If no price data is returned
    """
    logger.info(f"Fetching prices for {ticker} from {start_date} to {end_date}")

    url = (
        f"{BASE_URL}/prices/"
        f"?ticker={ticker}"
        f"&interval=day"
        f"&interval_multiplier=1"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
    )

    try:
        session = get_session()
        response = session.get(url, headers=get_headers(), timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()

        data = response.json()
        prices = data.get("prices")

        if not prices:
            logger.warning(f"No price data returned for {ticker}")
            raise ValueError("No price data returned")

        logger.debug(f"Retrieved {len(prices)} price records for {ticker}")
        return prices

    except (requests.RequestException, ValueError) as e:
        logger.warning(f"Financial Datasets API failed for {ticker}: {e}")

        # Try Yahoo Finance fallback
        if YFINANCE_AVAILABLE:
            logger.info(f"Attempting Yahoo Finance fallback for {ticker}")
            try:
                return _get_prices_from_yfinance(ticker, start_date, end_date)
            except Exception as yf_error:
                logger.error(f"Yahoo Finance fallback also failed: {yf_error}")
                raise
        else:
            logger.error(
                "Yahoo Finance fallback not available (yfinance not installed)"
            )
            raise


def prices_to_df(prices: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert prices list to a pandas DataFrame.

    Args:
        prices: List of price records with OHLCV data

    Returns:
        DataFrame with Date index and OHLCV columns
    """
    df = pd.DataFrame(prices)
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Get price data as a DataFrame.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with Date index and OHLCV columns
    """
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)
