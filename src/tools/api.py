"""API tools for fetching financial data from external sources."""

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from tools.cache import cached

# Configure module logger
logger = logging.getLogger(__name__)

# Constants
BASE_URL = "https://api.financialdatasets.ai"
DEFAULT_TIMEOUT = 30  # seconds
MAX_RETRIES = 3

# Check if yfinance is available
try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not installed. Yahoo Finance fallback unavailable.")


def _get_session() -> requests.Session:
    """Create a requests session with retry logic."""
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


def _get_headers() -> Dict[str, str]:
    """Get API headers with authentication."""
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key
    return headers


@cached
def get_financial_metrics(ticker: str, report_period: str, period: str = "ttm", limit: int = 1) -> List[Dict[str, Any]]:
    """
    Fetch financial metrics from the API.
    Falls back to Yahoo Finance if Financial Datasets API fails.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        report_period: Report period date (YYYY-MM-DD)
        period: Period type ('ttm', 'quarterly', 'annual')
        limit: Maximum number of results to return

    Returns:
        List of financial metrics dictionaries

    Raises:
        requests.RequestException: If both APIs fail
        ValueError: If no data is returned
    """
    logger.info(f"Fetching financial metrics for {ticker}")

    url = (
        f"{BASE_URL}/financial-metrics/"
        f"?ticker={ticker}"
        f"&report_period_lte={report_period}"
        f"&limit={limit}"
        f"&period={period}"
    )

    try:
        session = _get_session()
        response = session.get(url, headers=_get_headers(), timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()

        data = response.json()
        financial_metrics = data.get("financial_metrics")

        if not financial_metrics:
            logger.warning(f"No financial metrics returned for {ticker}")
            raise ValueError("No financial metrics returned")

        logger.debug(f"Retrieved {len(financial_metrics)} financial metrics for {ticker}")
        return financial_metrics

    except (requests.RequestException, ValueError) as e:
        logger.warning(f"Financial Datasets API failed for {ticker}: {e}")

        # Try Yahoo Finance fallback
        if YFINANCE_AVAILABLE:
            logger.info("Attempting Yahoo Finance fallback for financial metrics")
            try:
                return _get_financial_metrics_from_yfinance(ticker)
            except Exception as yf_error:
                logger.error(f"Yahoo Finance fallback also failed: {yf_error}")
                raise
        else:
            logger.error("Yahoo Finance fallback not available (yfinance not installed)")
            raise


@cached
def search_line_items(ticker: str, line_items: List[str], period: str = "ttm", limit: int = 1) -> List[Dict[str, Any]]:
    """
    Search for specific financial line items.
    Falls back to Yahoo Finance if Financial Datasets API fails.

    Args:
        ticker: Stock ticker symbol
        line_items: List of line items to search for
        period: Period type ('ttm', 'quarterly', 'annual')
        limit: Maximum number of results per line item

    Returns:
        List of search results with line item data

    Raises:
        requests.RequestException: If both APIs fail
        ValueError: If no results are returned
    """
    logger.info(f"Searching line items for {ticker}: {line_items}")

    url = f"{BASE_URL}/financials/search/line-items"
    body = {
        "tickers": [ticker],
        "line_items": line_items,
        "period": period,
        "limit": limit,
    }

    try:
        session = _get_session()
        response = session.post(url, headers=_get_headers(), json=body, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()

        data = response.json()
        search_results = data.get("search_results")

        if not search_results:
            logger.warning(f"No search results returned for {ticker}")
            raise ValueError("No search results returned")

        return search_results

    except requests.RequestException as e:
        # Check for payment/auth errors and try Yahoo Finance fallback
        is_payment_error = hasattr(e, "response") and e.response is not None and e.response.status_code in (401, 402)

        if is_payment_error:
            logger.warning(f"Financial Datasets API failed for {ticker}: {e}")
            if YFINANCE_AVAILABLE:
                logger.info("Attempting Yahoo Finance fallback for line items")
                try:
                    return _search_line_items_from_yfinance(ticker, line_items, limit)
                except Exception as yf_error:
                    logger.error(f"Yahoo Finance fallback also failed: {yf_error}")
                    raise
            else:
                logger.error("Yahoo Finance fallback not available (yfinance not installed)")

        logger.error(f"Failed to search line items for {ticker}: {e}")
        raise


@cached
def get_insider_trades(
    ticker: str,
    end_date: str,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    Fetch insider trades for a given ticker and date range.

    Args:
        ticker: Stock ticker symbol
        end_date: End date for filtering trades (YYYY-MM-DD)
        limit: Maximum number of trades to return

    Returns:
        List of insider trade records

    Raises:
        requests.RequestException: If API request fails
        ValueError: If no trades are returned
    """
    logger.info(f"Fetching insider trades for {ticker} up to {end_date}")

    url = f"{BASE_URL}/insider-trades/" f"?ticker={ticker}" f"&filing_date_lte={end_date}" f"&limit={limit}"

    try:
        session = _get_session()
        response = session.get(url, headers=_get_headers(), timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch insider trades for {ticker}: {e}")
        raise

    data = response.json()
    insider_trades = data.get("insider_trades")

    if not insider_trades:
        logger.warning(f"No insider trades returned for {ticker}")
        raise ValueError("No insider trades returned")

    logger.debug(f"Retrieved {len(insider_trades)} insider trades for {ticker}")
    return insider_trades


@cached
def get_market_cap(ticker: str) -> Optional[float]:
    """
    Fetch market capitalization for a ticker.
    Falls back to Yahoo Finance if Financial Datasets API fails.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Market capitalization value in USD, or None if not available

    Raises:
        requests.RequestException: If both APIs fail
        ValueError: If no company facts are returned
    """
    logger.info(f"Fetching market cap for {ticker}")

    url = f"{BASE_URL}/company/facts?ticker={ticker}"

    try:
        session = _get_session()
        response = session.get(url, headers=_get_headers(), timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()

        data = response.json()
        company_facts = data.get("company_facts")

        if not company_facts:
            logger.warning(f"No company facts returned for {ticker}")
            raise ValueError("No company facts returned")

        market_cap = company_facts.get("market_cap")
        logger.debug(f"Market cap for {ticker}: ${market_cap:,.2f}" if market_cap else f"No market cap for {ticker}")
        return market_cap

    except (requests.RequestException, ValueError) as e:
        logger.warning(f"Financial Datasets API failed for market cap {ticker}: {e}")

        # Try Yahoo Finance fallback
        if YFINANCE_AVAILABLE:
            logger.info("Attempting Yahoo Finance fallback for market cap")
            try:
                return _get_market_cap_from_yfinance(ticker)
            except Exception as yf_error:
                logger.error(f"Yahoo Finance fallback also failed: {yf_error}")
                raise
        else:
            logger.error("Yahoo Finance fallback not available (yfinance not installed)")
            raise


def _get_prices_from_yfinance(ticker: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """
    Fetch price data from Yahoo Finance as a fallback.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        List of price records with OHLCV data
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance is not installed. Install with: pip install yfinance")

    logger.info(f"Fetching prices from Yahoo Finance for {ticker}")

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval="1d")

        if df.empty:
            raise ValueError(f"No price data available from Yahoo Finance for {ticker}")

        # Convert to the same format as Financial Datasets API
        prices = []
        for date, row in df.iterrows():
            prices.append(
                {
                    "time": date.strftime("%Y-%m-%d"),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]),
                }
            )

        logger.info(f"Retrieved {len(prices)} price records from Yahoo Finance for {ticker}")
        return prices

    except Exception as e:
        logger.error(f"Failed to fetch prices from Yahoo Finance for {ticker}: {e}")
        raise


def _get_financial_metrics_from_yfinance(ticker: str) -> List[Dict[str, Any]]:
    """
    Fetch financial metrics from Yahoo Finance as a fallback.

    Args:
        ticker: Stock ticker symbol

    Returns:
        List with a single financial metrics dictionary
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance is not installed. Install with: pip install yfinance")

    logger.info(f"Fetching financial metrics from Yahoo Finance for {ticker}")

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Map Yahoo Finance fields to Financial Datasets API format
        # Field names must match what agents expect
        metrics = {
            "ticker": ticker,
            # Valuation metrics
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "pb_ratio": info.get("priceToBook"),
            "ps_ratio": info.get("priceToSalesTrailing12Months"),
            "price_to_book": info.get("priceToBook"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "price_to_earnings": info.get("trailingPE"),
            # Profitability metrics (names matching fundamentals agent)
            "return_on_equity": info.get("returnOnEquity"),
            "return_on_assets": info.get("returnOnAssets"),
            "net_margin": info.get("profitMargins"),  # Maps to net_margin
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "gross_margin": info.get("grossMargins"),
            # Growth metrics (may not be available from yfinance)
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "book_value_growth": None,  # Not available in yfinance
            # Financial health
            "current_ratio": info.get("currentRatio"),
            "quick_ratio": info.get("quickRatio"),
            "debt_to_equity": info.get("debtToEquity"),
            # Income statement
            "revenue": info.get("totalRevenue"),
            "net_income": info.get("netIncomeToCommon"),
            "earnings_per_share": info.get("trailingEps"),
            "free_cash_flow": info.get("freeCashflow"),
            # Dividend
            "dividend_yield": info.get("dividendYield"),
            # Risk
            "beta": info.get("beta"),
            # Price levels
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            # Volume
            "average_volume": info.get("averageVolume"),
            # Source indicator
            "source": "yahoo_finance",
        }

        logger.info(f"Retrieved financial metrics from Yahoo Finance for {ticker}")
        return [metrics]

    except Exception as e:
        logger.error(f"Failed to fetch financial metrics from Yahoo Finance for {ticker}: {e}")
        raise


def _get_market_cap_from_yfinance(ticker: str) -> Optional[float]:
    """
    Fetch market cap from Yahoo Finance as a fallback.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Market capitalization value in USD
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance is not installed. Install with: pip install yfinance")

    logger.info(f"Fetching market cap from Yahoo Finance for {ticker}")

    try:
        stock = yf.Ticker(ticker)
        market_cap = stock.info.get("marketCap")

        if market_cap:
            logger.info(f"Market cap from Yahoo Finance for {ticker}: ${market_cap:,.2f}")
        return market_cap

    except Exception as e:
        logger.error(f"Failed to fetch market cap from Yahoo Finance for {ticker}: {e}")
        raise


def _search_line_items_from_yfinance(ticker: str, line_items: List[str], limit: int = 1) -> List[Dict[str, Any]]:
    """
    Search for financial line items from Yahoo Finance as a fallback.

    Args:
        ticker: Stock ticker symbol
        line_items: List of line items to search for
        limit: Number of periods to return

    Returns:
        List of dictionaries with line item data
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance is not installed. Install with: pip install yfinance")

    logger.info(f"Fetching line items from Yahoo Finance for {ticker}: {line_items}")

    try:
        stock = yf.Ticker(ticker)

        # Get financial statements
        cashflow = stock.cashflow
        income = stock.income_stmt
        balance = stock.balance_sheet
        info = stock.info

        results = []

        # Get up to 'limit' periods (columns are periods in yfinance)
        num_periods = min(limit, len(cashflow.columns) if not cashflow.empty else 0)
        if num_periods == 0:
            num_periods = 1  # At least return one record with info data

        for i in range(num_periods):
            record = {"ticker": ticker}

            for item in line_items:
                value = None

                # Map line items to yfinance data
                if item == "free_cash_flow":
                    if not cashflow.empty and i < len(cashflow.columns):
                        value = _safe_get_value(cashflow, "Free Cash Flow", i)
                        if value is None:
                            # Calculate from operating cash flow - capex
                            op_cf = _safe_get_value(cashflow, "Operating Cash Flow", i)
                            capex = _safe_get_value(cashflow, "Capital Expenditure", i)
                            if op_cf is not None and capex is not None:
                                value = op_cf + capex  # capex is usually negative
                    if value is None:
                        value = info.get("freeCashflow")

                elif item == "net_income":
                    if not income.empty and i < len(income.columns):
                        value = _safe_get_value(income, "Net Income", i)
                    if value is None:
                        value = info.get("netIncomeToCommon")

                elif item == "depreciation_and_amortization":
                    if not cashflow.empty and i < len(cashflow.columns):
                        value = _safe_get_value(cashflow, "Depreciation And Amortization", i)

                elif item == "capital_expenditure":
                    if not cashflow.empty and i < len(cashflow.columns):
                        value = _safe_get_value(cashflow, "Capital Expenditure", i)

                elif item == "working_capital":
                    if not balance.empty and i < len(balance.columns):
                        current_assets = _safe_get_value(balance, "Current Assets", i)
                        current_liabilities = _safe_get_value(balance, "Current Liabilities", i)
                        if current_assets is not None and current_liabilities is not None:
                            value = current_assets - current_liabilities

                elif item == "revenue" or item == "total_revenue":
                    if not income.empty and i < len(income.columns):
                        value = _safe_get_value(income, "Total Revenue", i)
                    if value is None:
                        value = info.get("totalRevenue")

                elif item == "operating_income":
                    if not income.empty and i < len(income.columns):
                        value = _safe_get_value(income, "Operating Income", i)
                    if value is None:
                        value = info.get("operatingIncome")

                elif item == "gross_profit":
                    if not income.empty and i < len(income.columns):
                        value = _safe_get_value(income, "Gross Profit", i)
                    if value is None:
                        value = info.get("grossProfit")

                elif item == "total_assets":
                    if not balance.empty and i < len(balance.columns):
                        value = _safe_get_value(balance, "Total Assets", i)

                elif item == "total_liabilities":
                    if not balance.empty and i < len(balance.columns):
                        value = _safe_get_value(balance, "Total Liabilities Net Minority Interest", i)

                elif item == "total_equity" or item == "shareholders_equity":
                    if not balance.empty and i < len(balance.columns):
                        value = _safe_get_value(balance, "Stockholders Equity", i)

                elif item == "total_debt":
                    if not balance.empty and i < len(balance.columns):
                        value = _safe_get_value(balance, "Total Debt", i)

                elif item == "cash_and_equivalents":
                    if not balance.empty and i < len(balance.columns):
                        value = _safe_get_value(balance, "Cash And Cash Equivalents", i)

                record[item] = value

            record["source"] = "yahoo_finance"
            results.append(record)

        logger.info(f"Retrieved {len(results)} line item records from Yahoo Finance for {ticker}")
        return results

    except Exception as e:
        logger.error(f"Failed to fetch line items from Yahoo Finance for {ticker}: {e}")
        raise


def _safe_get_value(df: pd.DataFrame, row_name: str, col_idx: int) -> Optional[float]:
    """Safely extract a value from a DataFrame."""
    try:
        if row_name in df.index and col_idx < len(df.columns):
            value = df.loc[row_name].iloc[col_idx]
            if pd.notna(value):
                return float(value)
    except (KeyError, IndexError):
        pass
    return None


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
        session = _get_session()
        response = session.get(url, headers=_get_headers(), timeout=DEFAULT_TIMEOUT)
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
            logger.error("Yahoo Finance fallback not available (yfinance not installed)")
            raise


def prices_to_df(prices: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame(prices)
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


# Update the get_price_data function to use the new functions
def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)
