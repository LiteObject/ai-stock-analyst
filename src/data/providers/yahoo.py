"""
Yahoo Finance Data Provider.

This module implements the DataProvider interface using Yahoo Finance
via the yfinance library. This serves as a free fallback when no
API key is configured.

Note: Yahoo Finance data may have limitations and delays.
For production trading, use a professional data provider like Polygon.io.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict

import pandas as pd

from core.exceptions import DataProviderError, InsufficientDataError
from core.interfaces import DataProvider

logger = logging.getLogger(__name__)

# Check if yfinance is available
try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not installed. Yahoo Finance provider unavailable.")


class YahooDataProvider(DataProvider):
    """
    Data provider implementation using Yahoo Finance (yfinance).

    This is a free fallback provider for development and testing.
    For production use, consider using Polygon.io or another
    professional data provider.

    Limitations:
    - Data may be delayed
    - Rate limiting is not documented
    - Historical data availability varies
    - No real-time streaming

    Example:
        provider = YahooDataProvider()
        data = await provider.get_historical_data("AAPL", start, end)
    """

    def __init__(self, **kwargs):
        """
        Initialize the Yahoo Finance data provider.

        Args:
            **kwargs: Ignored (for interface compatibility)
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance is required for YahooDataProvider. " "Install it with: pip install yfinance")

        self._cache: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "Yahoo Finance"

    def _get_ticker(self, symbol: str) -> "yf.Ticker":
        """
        Get a yfinance Ticker object with caching.

        Args:
            symbol: Stock ticker symbol

        Returns:
            yfinance Ticker object
        """
        symbol = symbol.upper()
        if symbol not in self._cache:
            self._cache[symbol] = yf.Ticker(symbol)
        return self._cache[symbol]

    async def get_historical_data(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data
            timeframe: Data timeframe (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)

        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        ticker = ticker.upper()

        # Map timeframe to yfinance interval
        interval_map = {
            "1min": "1m",
            "1m": "1m",
            "5min": "5m",
            "5m": "5m",
            "15min": "15m",
            "15m": "15m",
            "30min": "30m",
            "30m": "30m",
            "1h": "1h",
            "4h": "1h",  # yfinance doesn't support 4h, use 1h
            "1d": "1d",
            "1w": "1wk",
            "1wk": "1wk",
            "1mo": "1mo",
        }

        interval = interval_map.get(timeframe, "1d")

        # For intraday data, yfinance has limitations
        if interval in ["1m", "5m", "15m", "30m", "1h"]:
            # Intraday data limited to last 7-60 days depending on interval
            max_days = {"1m": 7, "5m": 60, "15m": 60, "30m": 60, "1h": 730}
            days_limit = max_days.get(interval, 60)

            if (end_date - start_date).days > days_limit:
                logger.warning(f"Yahoo Finance limits {interval} data to {days_limit} days. " f"Adjusting start date.")
                start_date = end_date - timedelta(days=days_limit)

        try:
            # Run yfinance download in a thread pool to not block async
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                lambda: yf.download(
                    ticker,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                ),
            )

            if df.empty:
                raise InsufficientDataError(
                    f"No data returned for {ticker}",
                    provider=self.name,
                    ticker=ticker,
                    available_rows=0,
                )

            # Handle multi-level columns from yfinance first
            if isinstance(df.columns, pd.MultiIndex):
                # yfinance often returns (Price, Ticker) MultiIndex
                # We want just the Price level (Open, High, Low, Close, Volume)
                # Usually level 0 is the price type
                df.columns = df.columns.get_level_values(0)

            # Standardize column names
            df.columns = df.columns.str.lower()

            # Ensure we have required columns
            required = ["open", "high", "low", "close", "volume"]
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise DataProviderError(
                    f"Missing required columns: {missing}",
                    provider=self.name,
                    ticker=ticker,
                )

            # Select and reorder columns
            df = df[required]

            # Ensure index is named 'timestamp' for consistency
            df.index.name = "timestamp"

            logger.info(f"Retrieved {len(df)} bars for {ticker} from {self.name}")
            return df

        except Exception as e:
            if isinstance(e, (InsufficientDataError, DataProviderError)):
                raise
            raise DataProviderError(
                f"Failed to fetch data for {ticker}: {e}",
                provider=self.name,
                ticker=ticker,
            )

    async def get_current_price(self, ticker: str) -> float:
        """
        Get current price for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Current/last price
        """
        ticker = ticker.upper()

        try:
            loop = asyncio.get_event_loop()
            yf_ticker = self._get_ticker(ticker)

            # Get fast info (cached)
            info = await loop.run_in_executor(
                None,
                lambda: yf_ticker.fast_info,
            )

            # Try different price fields
            price = info.get("lastPrice") or info.get("regularMarketPrice")

            if price is None:
                # Fall back to historical data
                df = await self.get_historical_data(
                    ticker,
                    datetime.now() - timedelta(days=5),
                    datetime.now(),
                    "1d",
                )
                if not df.empty:
                    price = float(df["close"].iloc[-1])

            if price is None:
                raise DataProviderError(
                    f"Could not get price for {ticker}",
                    provider=self.name,
                    ticker=ticker,
                )

            return float(price)

        except Exception as e:
            if isinstance(e, DataProviderError):
                raise
            raise DataProviderError(
                f"Failed to get price for {ticker}: {e}",
                provider=self.name,
                ticker=ticker,
            )

    async def get_quote(self, ticker: str) -> Dict[str, Any]:
        """
        Get current quote data.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Quote data including bid, ask, last, volume, etc.
        """
        ticker = ticker.upper()

        try:
            loop = asyncio.get_event_loop()
            yf_ticker = self._get_ticker(ticker)

            info = await loop.run_in_executor(
                None,
                lambda: yf_ticker.info,
            )

            return {
                "ticker": ticker,
                "last": info.get("regularMarketPrice") or info.get("currentPrice"),
                "bid": info.get("bid"),
                "ask": info.get("ask"),
                "bid_size": info.get("bidSize"),
                "ask_size": info.get("askSize"),
                "volume": info.get("volume") or info.get("regularMarketVolume"),
                "open": info.get("open") or info.get("regularMarketOpen"),
                "high": info.get("dayHigh") or info.get("regularMarketDayHigh"),
                "low": info.get("dayLow") or info.get("regularMarketDayLow"),
                "prev_close": info.get("previousClose") or info.get("regularMarketPreviousClose"),
                "market_cap": info.get("marketCap"),
                "fifty_day_avg": info.get("fiftyDayAverage"),
                "two_hundred_day_avg": info.get("twoHundredDayAverage"),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.warning(f"Failed to get full quote for {ticker}: {e}")
            # Return minimal quote
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

        try:
            loop = asyncio.get_event_loop()
            yf_ticker = self._get_ticker(ticker)

            info = await loop.run_in_executor(
                None,
                lambda: yf_ticker.info,
            )

            return {
                "ticker": ticker,
                "name": info.get("longName") or info.get("shortName"),
                "description": info.get("longBusinessSummary"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": info.get("marketCap"),
                "employees": info.get("fullTimeEmployees"),
                "homepage": info.get("website"),
                "exchange": info.get("exchange"),
                "currency": info.get("currency"),
                "country": info.get("country"),
            }

        except Exception as e:
            raise DataProviderError(
                f"Failed to get company details for {ticker}: {e}",
                provider=self.name,
                ticker=ticker,
            )

    async def get_financials(
        self,
        ticker: str,
        statement_type: str = "income",
        quarterly: bool = False,
    ) -> pd.DataFrame:
        """
        Get financial statements.

        Args:
            ticker: Stock ticker symbol
            statement_type: 'income', 'balance', or 'cash'
            quarterly: If True, get quarterly data; else annual

        Returns:
            DataFrame with financial data
        """
        ticker = ticker.upper()

        try:
            loop = asyncio.get_event_loop()
            yf_ticker = self._get_ticker(ticker)

            statement_map = {
                "income": (yf_ticker.quarterly_income_stmt if quarterly else yf_ticker.income_stmt),
                "balance": (yf_ticker.quarterly_balance_sheet if quarterly else yf_ticker.balance_sheet),
                "cash": (yf_ticker.quarterly_cashflow if quarterly else yf_ticker.cashflow),
            }

            if statement_type not in statement_map:
                raise ValueError(f"Unknown statement type: {statement_type}")

            df = await loop.run_in_executor(
                None,
                lambda: statement_map[statement_type],
            )

            return df

        except Exception as e:
            raise DataProviderError(
                f"Failed to get financials for {ticker}: {e}",
                provider=self.name,
                ticker=ticker,
            )

    async def health_check(self) -> bool:
        """Check if Yahoo Finance is accessible."""
        try:
            await self.get_current_price("AAPL")
            return True
        except Exception:
            return False
