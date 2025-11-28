"""Fundamental analysis tools for financial metrics and line items.

This module provides functions for fetching financial metrics, line items,
insider trades, and market cap data from Financial Datasets API with
Yahoo Finance fallback.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from tools.api_client import (
    BASE_URL,
    DEFAULT_TIMEOUT,
    YFINANCE_AVAILABLE,
    get_headers,
    get_session,
    get_yfinance_ticker,
)
from tools.cache import cached

# Configure module logger
logger = logging.getLogger(__name__)


def _safe_get_value(df: pd.DataFrame, row_name: str, col_idx: int) -> Optional[float]:
    """Safely extract a value from a DataFrame.

    Args:
        df: DataFrame to extract from
        row_name: Row index name
        col_idx: Column index position

    Returns:
        Float value or None if not found
    """
    try:
        if row_name in df.index and col_idx < len(df.columns):
            value = df.loc[row_name].iloc[col_idx]
            # Check if it's a scalar value and not NaN
            if not isinstance(value, pd.Series) and pd.notna(value):
                return float(value)
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    return None


def _get_financial_metrics_from_yfinance(ticker: str) -> List[Dict[str, Any]]:
    """
    Fetch financial metrics from Yahoo Finance as a fallback.

    Args:
        ticker: Stock ticker symbol

    Returns:
        List with a single financial metrics dictionary
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError(
            "yfinance is not installed. Install with: pip install yfinance"
        )

    logger.info(f"Fetching financial metrics from Yahoo Finance for {ticker}")

    try:
        stock = get_yfinance_ticker(ticker)
        info = stock.info

        # Map Yahoo Finance fields to Financial Datasets API format
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
            # Profitability metrics
            "return_on_equity": info.get("returnOnEquity"),
            "return_on_assets": info.get("returnOnAssets"),
            "net_margin": info.get("profitMargins"),
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "gross_margin": info.get("grossMargins"),
            # Growth metrics
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "book_value_growth": None,
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
        logger.error(
            f"Failed to fetch financial metrics from Yahoo Finance for {ticker}: {e}"
        )
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
        raise ImportError(
            "yfinance is not installed. Install with: pip install yfinance"
        )

    logger.info(f"Fetching market cap from Yahoo Finance for {ticker}")

    try:
        stock = get_yfinance_ticker(ticker)
        market_cap = stock.info.get("marketCap")

        if market_cap:
            logger.info(
                f"Market cap from Yahoo Finance for {ticker}: ${market_cap:,.2f}"
            )
        return market_cap

    except Exception as e:
        logger.error(f"Failed to fetch market cap from Yahoo Finance for {ticker}: {e}")
        raise


def _search_line_items_from_yfinance(
    ticker: str, line_items: List[str], limit: int = 1
) -> List[Dict[str, Any]]:
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
        raise ImportError(
            "yfinance is not installed. Install with: pip install yfinance"
        )

    logger.info(f"Fetching line items from Yahoo Finance for {ticker}: {line_items}")

    try:
        stock = get_yfinance_ticker(ticker)

        # Get financial statements
        cashflow = stock.cashflow
        income = stock.income_stmt
        balance = stock.balance_sheet
        info = stock.info

        results = []

        # Get up to 'limit' periods (columns are periods in yfinance)
        num_periods = min(limit, len(cashflow.columns) if not cashflow.empty else 0)
        if num_periods == 0:
            num_periods = 1

        for i in range(num_periods):
            record: Dict[str, Any] = {"ticker": ticker}

            for item in line_items:
                value = None

                if item == "free_cash_flow":
                    if not cashflow.empty and i < len(cashflow.columns):
                        value = _safe_get_value(cashflow, "Free Cash Flow", i)
                        if value is None:
                            op_cf = _safe_get_value(cashflow, "Operating Cash Flow", i)
                            capex = _safe_get_value(cashflow, "Capital Expenditure", i)
                            if op_cf is not None and capex is not None:
                                value = op_cf + capex
                    if value is None:
                        value = info.get("freeCashflow")

                elif item == "net_income":
                    if not income.empty and i < len(income.columns):
                        value = _safe_get_value(income, "Net Income", i)
                    if value is None:
                        value = info.get("netIncomeToCommon")

                elif item == "depreciation_and_amortization":
                    if not cashflow.empty and i < len(cashflow.columns):
                        value = _safe_get_value(
                            cashflow, "Depreciation And Amortization", i
                        )

                elif item == "capital_expenditure":
                    if not cashflow.empty and i < len(cashflow.columns):
                        value = _safe_get_value(cashflow, "Capital Expenditure", i)

                elif item == "working_capital":
                    if not balance.empty and i < len(balance.columns):
                        current_assets = _safe_get_value(balance, "Current Assets", i)
                        current_liabilities = _safe_get_value(
                            balance, "Current Liabilities", i
                        )
                        if (
                            current_assets is not None
                            and current_liabilities is not None
                        ):
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
                        value = _safe_get_value(
                            balance, "Total Liabilities Net Minority Interest", i
                        )

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

        logger.info(
            f"Retrieved {len(results)} line item records from Yahoo Finance for {ticker}"
        )
        return results

    except Exception as e:
        logger.error(f"Failed to fetch line items from Yahoo Finance for {ticker}: {e}")
        raise


@cached
def get_financial_metrics(
    ticker: str, report_period: str, period: str = "ttm", limit: int = 1
) -> List[Dict[str, Any]]:
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
        session = get_session()
        response = session.get(url, headers=get_headers(), timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()

        data = response.json()
        financial_metrics = data.get("financial_metrics")

        if not financial_metrics:
            logger.warning(f"No financial metrics returned for {ticker}")
            raise ValueError("No financial metrics returned")

        logger.debug(
            f"Retrieved {len(financial_metrics)} financial metrics for {ticker}"
        )
        return financial_metrics

    except (requests.RequestException, ValueError) as e:
        logger.warning(f"Financial Datasets API failed for {ticker}: {e}")

        if YFINANCE_AVAILABLE:
            logger.info(f"Attempting Yahoo Finance fallback for {ticker}")
            try:
                return _get_financial_metrics_from_yfinance(ticker)
            except Exception as yf_error:
                logger.error(f"Yahoo Finance fallback also failed: {yf_error}")
                raise
        else:
            logger.error(
                "Yahoo Finance fallback not available (yfinance not installed)"
            )
            raise


@cached
def search_line_items(
    ticker: str,
    line_items: List[str],
    report_period: str,
    period: str = "ttm",
    limit: int = 1,
) -> List[Dict[str, Any]]:
    """
    Search for specific line items in financial statements.
    Falls back to Yahoo Finance if Financial Datasets API fails.

    Args:
        ticker: Stock ticker symbol
        line_items: List of line item names to search for
        report_period: Report period date (YYYY-MM-DD)
        period: Period type ('ttm', 'quarterly', 'annual')
        limit: Maximum number of results to return

    Returns:
        List of line item results

    Raises:
        requests.RequestException: If both APIs fail
        ValueError: If no data is returned
    """
    logger.info(f"Searching line items for {ticker}: {line_items}")

    url = f"{BASE_URL}/financials/search/line-items"
    payload = {
        "tickers": [ticker],
        "line_items": line_items,
        "end_date": report_period,
        "period": period,
        "limit": limit,
    }

    try:
        session = get_session()
        response = session.post(
            url, headers=get_headers(), json=payload, timeout=DEFAULT_TIMEOUT
        )
        response.raise_for_status()

        data = response.json()
        results = data.get("search_results")

        if not results:
            logger.warning(f"No line items returned for {ticker}")
            raise ValueError("No line items returned")

        logger.debug(f"Retrieved {len(results)} line items for {ticker}")
        return results

    except (requests.RequestException, ValueError) as e:
        logger.warning(f"Financial Datasets API failed for {ticker}: {e}")

        if YFINANCE_AVAILABLE:
            logger.info(f"Attempting Yahoo Finance fallback for {ticker}")
            try:
                return _search_line_items_from_yfinance(ticker, line_items, limit)
            except Exception as yf_error:
                logger.error(f"Yahoo Finance fallback also failed: {yf_error}")
                raise
        else:
            logger.error(
                "Yahoo Finance fallback not available (yfinance not installed)"
            )
            raise


@cached
def get_insider_trades(
    ticker: str, start_date: str, end_date: str, limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Fetch insider trading data for a company.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        limit: Maximum number of results to return

    Returns:
        List of insider trade records

    Raises:
        requests.RequestException: If the API call fails
        ValueError: If no data is returned
    """
    logger.info(f"Fetching insider trades for {ticker}")

    url = (
        f"{BASE_URL}/insider-trades/"
        f"?ticker={ticker}"
        f"&filing_date_gte={start_date}"
        f"&filing_date_lte={end_date}"
        f"&limit={limit}"
    )

    try:
        session = get_session()
        response = session.get(url, headers=get_headers(), timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()

        data = response.json()
        trades = data.get("insider_trades")

        if not trades:
            logger.warning(f"No insider trades returned for {ticker}")
            raise ValueError("No insider trades returned")

        logger.debug(f"Retrieved {len(trades)} insider trades for {ticker}")
        return trades

    except requests.RequestException as e:
        logger.error(f"Failed to fetch insider trades for {ticker}: {e}")
        raise


@cached
def get_market_cap(ticker: str, report_period: str) -> Optional[float]:
    """
    Get market capitalization for a company.
    Falls back to Yahoo Finance if Financial Datasets API fails.

    Args:
        ticker: Stock ticker symbol
        report_period: Report period date (YYYY-MM-DD)

    Returns:
        Market capitalization value in USD
    """
    logger.info(f"Fetching market cap for {ticker}")

    try:
        metrics = get_financial_metrics(ticker, report_period)
        if metrics and isinstance(metrics, list) and len(metrics) > 0:
            market_cap = metrics[0].get("market_cap")
            if market_cap:
                logger.info(f"Market cap for {ticker}: ${market_cap:,.2f}")
                return market_cap

        logger.warning(f"No market cap in financial metrics for {ticker}")
        raise ValueError("No market cap in financial metrics")

    except Exception as e:
        logger.warning(f"Financial metrics failed for market cap: {e}")

        if YFINANCE_AVAILABLE:
            logger.info("Attempting Yahoo Finance fallback for market cap")
            try:
                return _get_market_cap_from_yfinance(ticker)
            except Exception as yf_error:
                logger.error(f"Yahoo Finance fallback also failed: {yf_error}")
                return None
        else:
            logger.error("Yahoo Finance fallback not available")
            return None
