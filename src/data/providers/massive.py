"""
Massive/Polygon.io Data Provider.

This module implements the DataProvider interface using Polygon.io's
Massive API for professional-grade market data.

API Documentation: https://massive.com/docs/rest/quickstart

Features:
- Historical OHLCV data
- Real-time quotes (with appropriate subscription)
- Company details and financials
- Rate limiting and retry logic
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from core.exceptions import (
    APIConnectionError,
    DataProviderError,
    InsufficientDataError,
    RateLimitError,
)
from core.interfaces import DataProvider

logger = logging.getLogger(__name__)


class MassiveDataProvider(DataProvider):
    """
    Data provider implementation using Polygon.io/Massive API.

    Environment Variables:
        MASSIVE_API_KEY or POLYGON_API_KEY: API key for authentication
        MASSIVE_BASE_URL: Base URL for API (default: https://api.polygon.io)

    Example:
        provider = MassiveDataProvider(api_key="your_api_key")
        data = await provider.get_historical_data("AAPL", start, end)
    """

    DEFAULT_BASE_URL = "https://api.polygon.io"
    DEFAULT_TIMEOUT = 30
    MAX_RETRIES = 3

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        """
        Initialize the Massive/Polygon data provider.

        Args:
            api_key: API key (uses MASSIVE_API_KEY or POLYGON_API_KEY env var if not provided)
            base_url: Base URL for API
            timeout: Request timeout in seconds
        """
        self._api_key = api_key or os.environ.get("MASSIVE_API_KEY") or os.environ.get("POLYGON_API_KEY")

        if not self._api_key:
            logger.warning("No Massive/Polygon API key provided. Some features may not work.")

        self._base_url = base_url or os.environ.get("MASSIVE_BASE_URL", self.DEFAULT_BASE_URL)
        self._timeout = timeout
        self._session = self._create_session()

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "Massive/Polygon.io"

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic."""
        session = requests.Session()
        retry_strategy = Retry(
            total=self.MAX_RETRIES,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        method: str = "GET",
    ) -> Dict[str, Any]:
        """
        Make an API request.

        Args:
            endpoint: API endpoint (relative to base URL)
            params: Query parameters
            method: HTTP method

        Returns:
            Response JSON

        Raises:
            RateLimitError: If rate limited
            APIConnectionError: If connection fails
            DataProviderError: For other API errors
        """
        url = f"{self._base_url}{endpoint}"

        # Add API key to params for Polygon's URL-based auth
        if params is None:
            params = {}
        if self._api_key:
            params["apiKey"] = self._api_key

        try:
            response = self._session.request(
                method=method,
                url=url,
                params=params,
                headers=self._get_headers(),
                timeout=self._timeout,
            )

            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                raise RateLimitError(
                    f"Rate limited by {self.name}",
                    retry_after=retry_after,
                    provider=self.name,
                )

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            raise APIConnectionError(
                f"Request to {self.name} timed out",
                url=url,
                provider=self.name,
            )
        except requests.exceptions.ConnectionError as e:
            raise APIConnectionError(
                f"Failed to connect to {self.name}: {e}",
                url=url,
                provider=self.name,
            )
        except requests.exceptions.HTTPError as e:
            raise DataProviderError(
                f"HTTP error from {self.name}: {e}",
                provider=self.name,
            )

    async def get_historical_data(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Polygon.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data
            timeframe: Data timeframe (1min, 5min, 1h, 1d, etc.)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, vwap
        """
        ticker = ticker.upper()

        # Map timeframe to Polygon format
        timeframe_map = {
            "1min": ("minute", 1),
            "5min": ("minute", 5),
            "15min": ("minute", 15),
            "30min": ("minute", 30),
            "1h": ("hour", 1),
            "4h": ("hour", 4),
            "1d": ("day", 1),
            "1w": ("week", 1),
            "1mo": ("month", 1),
        }

        if timeframe not in timeframe_map:
            raise DataProviderError(
                f"Unsupported timeframe: {timeframe}",
                provider=self.name,
                ticker=ticker,
            )

        timespan, multiplier = timeframe_map[timeframe]

        # Format dates
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        # Make request
        endpoint = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_str}/{end_str}"

        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
        }

        try:
            data = self._make_request(endpoint, params)
        except DataProviderError as e:
            logger.warning(f"Polygon API failed for {ticker}: {e}")
            raise

        # Check for results
        results = data.get("results", [])
        if not results:
            raise InsufficientDataError(
                f"No data returned for {ticker} from {start_str} to {end_str}",
                provider=self.name,
                ticker=ticker,
                available_rows=0,
            )

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Rename columns to standard format
        column_map = {
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vw": "vwap",
            "n": "trades",
        }
        df = df.rename(columns=column_map)

        # Convert timestamp from milliseconds
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Select and order columns
        columns = ["timestamp", "open", "high", "low", "close", "volume"]
        if "vwap" in df.columns:
            columns.append("vwap")

        df = df[columns]
        df = df.set_index("timestamp")

        logger.info(f"Retrieved {len(df)} bars for {ticker} from {self.name}")
        return df

    async def get_current_price(self, ticker: str) -> float:
        """
        Get current price for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Current price
        """
        ticker = ticker.upper()

        # Use previous close endpoint for delayed data
        # For real-time, use snapshot endpoint (requires higher tier)
        endpoint = f"/v2/aggs/ticker/{ticker}/prev"

        try:
            data = self._make_request(endpoint)
            results = data.get("results", [])

            if results:
                return float(results[0].get("c", 0))

            raise DataProviderError(
                f"No price data for {ticker}",
                provider=self.name,
                ticker=ticker,
            )

        except DataProviderError:
            # Try snapshot endpoint as fallback
            try:
                endpoint = f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
                data = self._make_request(endpoint)

                ticker_data = data.get("ticker", {})
                if "day" in ticker_data:
                    return float(ticker_data["day"].get("c", 0))
                elif "prevDay" in ticker_data:
                    return float(ticker_data["prevDay"].get("c", 0))

            except Exception:
                pass

            raise

    async def get_quote(self, ticker: str) -> Dict[str, Any]:
        """
        Get current quote with bid/ask/last.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Quote data
        """
        ticker = ticker.upper()

        try:
            # Get snapshot for comprehensive quote data
            endpoint = f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
            data = self._make_request(endpoint)

            ticker_data = data.get("ticker", {})

            quote = {
                "ticker": ticker,
                "last": None,
                "bid": None,
                "ask": None,
                "volume": None,
                "open": None,
                "high": None,
                "low": None,
                "prev_close": None,
                "timestamp": datetime.now().isoformat(),
            }

            # Extract last trade
            if "lastTrade" in ticker_data:
                quote["last"] = ticker_data["lastTrade"].get("p")

            # Extract quote (bid/ask)
            if "lastQuote" in ticker_data:
                quote["bid"] = ticker_data["lastQuote"].get("P")  # Bid price
                quote["ask"] = ticker_data["lastQuote"].get("p")  # Ask price

            # Extract day data
            if "day" in ticker_data:
                day = ticker_data["day"]
                quote["open"] = day.get("o")
                quote["high"] = day.get("h")
                quote["low"] = day.get("l")
                quote["volume"] = day.get("v")

            # Extract previous day data
            if "prevDay" in ticker_data:
                quote["prev_close"] = ticker_data["prevDay"].get("c")

            return quote

        except Exception as e:
            logger.warning(f"Failed to get quote for {ticker}: {e}")
            # Return minimal quote with just the price
            price = await self.get_current_price(ticker)
            return {
                "ticker": ticker,
                "last": price,
                "timestamp": datetime.now().isoformat(),
            }

    async def get_company_details(self, ticker: str) -> Dict[str, Any]:
        """
        Get company details and metadata.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Company information
        """
        ticker = ticker.upper()
        endpoint = f"/v3/reference/tickers/{ticker}"

        data = self._make_request(endpoint)
        results = data.get("results", {})

        return {
            "ticker": ticker,
            "name": results.get("name"),
            "description": results.get("description"),
            "sector": results.get("sic_description"),
            "industry": results.get("sic_description"),
            "market_cap": results.get("market_cap"),
            "employees": results.get("total_employees"),
            "homepage": results.get("homepage_url"),
            "list_date": results.get("list_date"),
            "exchange": results.get("primary_exchange"),
        }

    async def get_financials(
        self,
        ticker: str,
        limit: int = 4,
        timeframe: str = "quarterly",
    ) -> List[Dict[str, Any]]:
        """
        Get financial statements.

        Args:
            ticker: Stock ticker symbol
            limit: Number of periods to fetch
            timeframe: 'quarterly' or 'annual'

        Returns:
            List of financial statement data
        """
        ticker = ticker.upper()
        endpoint = "/vX/reference/financials"

        params = {
            "ticker": ticker,
            "limit": limit,
            "timeframe": timeframe,
            "order": "desc",
            "sort": "period_of_report_date",
        }

        data = self._make_request(endpoint, params)
        return data.get("results", [])

    async def health_check(self) -> bool:
        """Check if the API is accessible."""
        try:
            # Use a lightweight endpoint
            endpoint = "/v1/marketstatus/now"
            self._make_request(endpoint)
            return True
        except Exception as e:
            logger.warning(f"Health check failed for {self.name}: {e}")
            return False
