"""
Free Data Sources for AI Stock Analyst.

This module provides access to various free financial data sources
to supplement the primary data from Financial Datasets API.

Supported Data Sources:
- Yahoo Finance (yfinance): Extended market data, analyst info, options
- FRED (Federal Reserve): Economic indicators
- SEC EDGAR: Company filings (10-K, 10-Q, 8-K)
- Fear & Greed Index: Market sentiment

All functions are cached to reduce API calls and improve performance.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import requests

from tools.cache import cached

logger = logging.getLogger(__name__)


# =============================================================================
# Data Source Registry - Modular Architecture
# =============================================================================


class DataSourceType(Enum):
    """Types of data sources available."""

    YAHOO_FINANCE = "yahoo_finance"
    FRED = "fred"
    SEC_EDGAR = "sec_edgar"
    FEAR_GREED = "fear_greed"


@dataclass
class DataSourceStatus:
    """Status of a data source."""

    source: DataSourceType
    available: bool
    reason: str = ""


class DataSourceRegistry:
    """Registry for managing available data sources."""

    _sources: dict[DataSourceType, bool] = {}
    _initialized: bool = False

    @classmethod
    def initialize(cls) -> None:
        """Initialize and check availability of all data sources."""
        if cls._initialized:
            return

        # Check Yahoo Finance
        try:
            import yfinance  # noqa: F401

            cls._sources[DataSourceType.YAHOO_FINANCE] = True
        except ImportError:
            cls._sources[DataSourceType.YAHOO_FINANCE] = False
            logger.warning("yfinance not installed. Install with: pip install yfinance")

        # FRED requires API key but is always "available" as a source
        cls._sources[DataSourceType.FRED] = bool(os.environ.get("FRED_API_KEY"))
        if not cls._sources[DataSourceType.FRED]:
            logger.info("FRED API key not set. Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html")

        # SEC EDGAR is always available (free, no API key)
        cls._sources[DataSourceType.SEC_EDGAR] = True

        # Fear & Greed is always available (scraping)
        cls._sources[DataSourceType.FEAR_GREED] = True

        cls._initialized = True

    @classmethod
    def is_available(cls, source: DataSourceType) -> bool:
        """Check if a data source is available."""
        if not cls._initialized:
            cls.initialize()
        return cls._sources.get(source, False)

    @classmethod
    def get_status(cls) -> list[DataSourceStatus]:
        """Get status of all data sources."""
        if not cls._initialized:
            cls.initialize()

        statuses = []
        for source, available in cls._sources.items():
            reason = "" if available else "Not configured or missing dependency"
            statuses.append(DataSourceStatus(source, available, reason))
        return statuses


# Initialize on module load
DataSourceRegistry.initialize()


# =============================================================================
# Yahoo Finance Extended Data
# =============================================================================


@cached(ttl_hours=24)
def get_analyst_recommendations(ticker: str) -> list[dict]:
    """
    Fetch analyst recommendations from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol

    Returns:
        List of analyst recommendations with firm, rating, and date
    """
    if not DataSourceRegistry.is_available(DataSourceType.YAHOO_FINANCE):
        logger.warning("Yahoo Finance not available")
        return []

    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        recommendations = stock.recommendations

        if recommendations is None or recommendations.empty:
            logger.warning(f"No analyst recommendations found for {ticker}")
            return []

        # Get recent recommendations (last 20)
        recent = recommendations.tail(20)
        result = []

        for idx, row in recent.iterrows():
            result.append(
                {
                    "date": (idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)),
                    "firm": row.get("Firm", "Unknown"),
                    "to_grade": row.get("To Grade", row.get("toGrade", "Unknown")),
                    "from_grade": row.get("From Grade", row.get("fromGrade", "")),
                    "action": row.get("Action", row.get("action", "")),
                }
            )

        logger.info(f"Retrieved {len(result)} analyst recommendations for {ticker}")
        return result

    except Exception as e:
        logger.error(f"Failed to fetch analyst recommendations for {ticker}: {e}")
        return []


@cached(ttl_hours=24)
def get_analyst_price_targets(ticker: str) -> dict:
    """
    Fetch analyst price targets from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dict with low, high, mean, median, and current price targets
    """
    if not DataSourceRegistry.is_available(DataSourceType.YAHOO_FINANCE):
        logger.warning("Yahoo Finance not available")
        return {}

    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        info = stock.info

        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        target_mean = info.get("targetMeanPrice")

        targets = {
            "current_price": current_price,
            "target_low": info.get("targetLowPrice"),
            "target_high": info.get("targetHighPrice"),
            "target_mean": target_mean,
            "target_median": info.get("targetMedianPrice"),
            "number_of_analysts": info.get("numberOfAnalystOpinions"),
            "recommendation": info.get("recommendationKey"),
            "recommendation_mean": info.get("recommendationMean"),  # 1=Strong Buy, 5=Sell
            "upside_potential": None,
        }

        # Calculate upside/downside potential
        if current_price and target_mean:
            targets["upside_potential"] = round((target_mean - current_price) / current_price * 100, 2)

        logger.info(f"Retrieved price targets for {ticker}")
        return targets

    except Exception as e:
        logger.error(f"Failed to fetch price targets for {ticker}: {e}")
        return {}


@cached(ttl_hours=24)
def get_institutional_holders(ticker: str) -> list[dict]:
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

        if holders is None or holders.empty:
            logger.warning(f"No institutional holders found for {ticker}")
            return []

        result = []
        for _, row in holders.iterrows():
            result.append(
                {
                    "holder": row.get("Holder", "Unknown"),
                    "shares": int(row.get("Shares", 0)),
                    "date_reported": str(row.get("Date Reported", "")),
                    "percent_out": (float(row.get("% Out", 0)) if row.get("% Out") else None),
                    "value": int(row.get("Value", 0)) if row.get("Value") else None,
                }
            )

        logger.info(f"Retrieved {len(result)} institutional holders for {ticker}")
        return result

    except Exception as e:
        logger.error(f"Failed to fetch institutional holders for {ticker}: {e}")
        return []


@cached(ttl_hours=24)
def get_major_holders_breakdown(ticker: str) -> dict:
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

        if major_holders is None or major_holders.empty:
            return {}

        # Parse the major holders DataFrame
        result = {}
        for idx, row in major_holders.iterrows():
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


@cached(ttl_hours=6)  # Options data changes more frequently
def get_options_data(ticker: str) -> dict:
    """
    Fetch options data including implied volatility from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dict with options statistics and implied volatility
    """
    if not DataSourceRegistry.is_available(DataSourceType.YAHOO_FINANCE):
        return {}

    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)

        # Get available expiration dates
        expirations = stock.options
        if not expirations:
            logger.warning(f"No options data available for {ticker}")
            return {}

        # Get the nearest expiration
        nearest_expiry = expirations[0]
        options = stock.option_chain(nearest_expiry)

        calls = options.calls
        puts = options.puts

        # Calculate put/call ratio and average IV
        total_call_volume = calls["volume"].sum() if "volume" in calls.columns else 0
        total_put_volume = puts["volume"].sum() if "volume" in puts.columns else 0
        total_call_oi = calls["openInterest"].sum() if "openInterest" in calls.columns else 0
        total_put_oi = puts["openInterest"].sum() if "openInterest" in puts.columns else 0

        put_call_volume_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else None
        put_call_oi_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else None

        # Average implied volatility (weighted by open interest)
        avg_call_iv = None
        avg_put_iv = None

        if "impliedVolatility" in calls.columns and "openInterest" in calls.columns:
            call_weights = calls["openInterest"].fillna(0)
            if call_weights.sum() > 0:
                avg_call_iv = (calls["impliedVolatility"] * call_weights).sum() / call_weights.sum()

        if "impliedVolatility" in puts.columns and "openInterest" in puts.columns:
            put_weights = puts["openInterest"].fillna(0)
            if put_weights.sum() > 0:
                avg_put_iv = (puts["impliedVolatility"] * put_weights).sum() / put_weights.sum()

        result = {
            "nearest_expiry": nearest_expiry,
            "days_to_expiry": (datetime.strptime(nearest_expiry, "%Y-%m-%d") - datetime.now()).days,
            "total_call_volume": int(total_call_volume) if total_call_volume else 0,
            "total_put_volume": int(total_put_volume) if total_put_volume else 0,
            "total_call_oi": int(total_call_oi) if total_call_oi else 0,
            "total_put_oi": int(total_put_oi) if total_put_oi else 0,
            "put_call_volume_ratio": (round(put_call_volume_ratio, 3) if put_call_volume_ratio else None),
            "put_call_oi_ratio": (round(put_call_oi_ratio, 3) if put_call_oi_ratio else None),
            "avg_call_iv": round(avg_call_iv, 4) if avg_call_iv else None,
            "avg_put_iv": round(avg_put_iv, 4) if avg_put_iv else None,
            "num_call_contracts": len(calls),
            "num_put_contracts": len(puts),
        }

        logger.info(f"Retrieved options data for {ticker} (expiry: {nearest_expiry})")
        return result

    except Exception as e:
        logger.error(f"Failed to fetch options data for {ticker}: {e}")
        return {}


@cached(ttl_hours=24)
def get_earnings_history(ticker: str) -> list[dict]:
    """
    Fetch earnings history and surprises from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol

    Returns:
        List of earnings with actual, estimate, and surprise
    """
    if not DataSourceRegistry.is_available(DataSourceType.YAHOO_FINANCE):
        return []

    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        earnings = stock.earnings_history

        if earnings is None or earnings.empty:
            return []

        result = []
        for idx, row in earnings.iterrows():
            surprise_pct = None
            if row.get("epsActual") and row.get("epsEstimate"):
                surprise_pct = ((row["epsActual"] - row["epsEstimate"]) / abs(row["epsEstimate"])) * 100

            result.append(
                {
                    "date": (idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)),
                    "eps_actual": row.get("epsActual"),
                    "eps_estimate": row.get("epsEstimate"),
                    "surprise_pct": round(surprise_pct, 2) if surprise_pct else None,
                }
            )

        logger.info(f"Retrieved {len(result)} earnings records for {ticker}")
        return result

    except Exception as e:
        logger.error(f"Failed to fetch earnings history for {ticker}: {e}")
        return []


@cached(ttl_hours=168)  # ESG data doesn't change often (1 week cache)
def get_esg_scores(ticker: str) -> dict:
    """
    Fetch ESG (Environmental, Social, Governance) scores from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dict with ESG scores and ratings
    """
    if not DataSourceRegistry.is_available(DataSourceType.YAHOO_FINANCE):
        return {}

    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        esg = stock.sustainability

        if esg is None or esg.empty:
            logger.warning(f"No ESG data available for {ticker}")
            return {}

        # Extract ESG scores
        result = {}
        score_keys = [
            "totalEsg",
            "environmentScore",
            "socialScore",
            "governanceScore",
            "percentile",
            "esgPerformance",
            "peerGroup",
        ]

        for key in score_keys:
            if key in esg.index:
                value = esg.loc[key].values[0] if hasattr(esg.loc[key], "values") else esg.loc[key]
                result[key] = float(value) if isinstance(value, (int, float)) else str(value)

        logger.info(f"Retrieved ESG scores for {ticker}")
        return result

    except Exception as e:
        logger.error(f"Failed to fetch ESG scores for {ticker}: {e}")
        return {}


# =============================================================================
# FRED (Federal Reserve Economic Data)
# =============================================================================

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
) -> list[dict]:
    """
    Fetch economic data from FRED.

    Note: FRED requires a free API key from https://fred.stlouisfed.org/docs/api/api_key.html

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
        logger.warning("FRED_API_KEY not set. Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return []

    try:
        params = {
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
def get_economic_indicators() -> dict:
    """
    Fetch common economic indicators from FRED.

    Returns:
        Dict with latest values for key economic indicators
    """
    if not os.environ.get("FRED_API_KEY"):
        return {}

    indicators = {}

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
def get_yield_curve() -> dict:
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

        result = {}

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


# =============================================================================
# Fear & Greed Index (CNN Money)
# =============================================================================


@cached(ttl_hours=6)  # Updates frequently
def get_fear_greed_index() -> dict:
    """
    Fetch the CNN Fear & Greed Index.

    This is a composite index of 7 indicators:
    - Stock Price Momentum (S&P 500 vs 125-day MA)
    - Stock Price Strength (52-week highs vs lows)
    - Stock Price Breadth (McClellan Volume Summation Index)
    - Put/Call Ratio
    - Market Volatility (VIX)
    - Safe Haven Demand (stocks vs bonds)
    - Junk Bond Demand (yield spread)

    Returns:
        Dict with fear/greed score and classification
    """
    try:
        # CNN Fear & Greed API endpoint
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Extract the current score
        fear_greed_data = data.get("fear_and_greed", {})
        score = fear_greed_data.get("score")
        rating = fear_greed_data.get("rating")
        previous_close = fear_greed_data.get("previous_close")
        previous_1_week = fear_greed_data.get("previous_1_week")
        previous_1_month = fear_greed_data.get("previous_1_month")
        previous_1_year = fear_greed_data.get("previous_1_year")

        # Classify the score
        def classify_score(s):
            if s is None:
                return "unknown"
            if s <= 25:
                return "extreme_fear"
            elif s <= 45:
                return "fear"
            elif s <= 55:
                return "neutral"
            elif s <= 75:
                return "greed"
            else:
                return "extreme_greed"

        result = {
            "score": score,
            "rating": rating or classify_score(score),
            "previous_close": previous_close,
            "previous_1_week": previous_1_week,
            "previous_1_month": previous_1_month,
            "previous_1_year": previous_1_year,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Fear & Greed Index: {score} ({result['rating']})")
        return result

    except Exception as e:
        logger.error(f"Failed to fetch Fear & Greed Index: {e}")
        return {}


# =============================================================================
# SEC EDGAR Data
# =============================================================================

SEC_EDGAR_BASE = "https://data.sec.gov"


@cached(ttl_hours=24)
def get_sec_filings(ticker: str, filing_type: str = "10-K", limit: int = 5) -> list[dict]:
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
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}/{primary_docs[i]}"

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


# =============================================================================
# Aggregate Data Functions
# =============================================================================


@cached(ttl_hours=12)
def get_extended_stock_data(ticker: str) -> dict:
    """
    Fetch all available extended data for a stock from free sources.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dict with all available extended data
    """
    logger.info(f"Fetching extended stock data for {ticker}")

    data = {
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "sources": [],
    }

    # Yahoo Finance data
    if DataSourceRegistry.is_available(DataSourceType.YAHOO_FINANCE):
        data["sources"].append("yahoo_finance")
        data["analyst_recommendations"] = get_analyst_recommendations(ticker)
        data["price_targets"] = get_analyst_price_targets(ticker)
        data["institutional_holders"] = get_institutional_holders(ticker)
        data["major_holders"] = get_major_holders_breakdown(ticker)
        data["options"] = get_options_data(ticker)
        data["earnings_history"] = get_earnings_history(ticker)
        data["esg_scores"] = get_esg_scores(ticker)

    # SEC EDGAR
    if DataSourceRegistry.is_available(DataSourceType.SEC_EDGAR):
        data["sources"].append("sec_edgar")
        data["sec_filings"] = get_sec_filings(ticker)

    return data


@cached(ttl_hours=6)
def get_market_sentiment() -> dict:
    """
    Fetch aggregate market sentiment indicators.

    Returns:
        Dict with fear/greed index and economic indicators
    """
    logger.info("Fetching market sentiment data")

    sentiment = {
        "timestamp": datetime.now().isoformat(),
        "sources": [],
    }

    # Fear & Greed Index
    fear_greed = get_fear_greed_index()
    if fear_greed:
        sentiment["sources"].append("cnn_fear_greed")
        sentiment["fear_greed"] = fear_greed

    # FRED economic indicators
    if DataSourceRegistry.is_available(DataSourceType.FRED):
        sentiment["sources"].append("fred")
        sentiment["economic_indicators"] = get_economic_indicators()
        sentiment["yield_curve"] = get_yield_curve()

    return sentiment


# =============================================================================
# Utility Functions
# =============================================================================


def list_available_sources() -> list[str]:
    """List all available data sources."""
    return [status.source.value for status in DataSourceRegistry.get_status() if status.available]


def get_source_status() -> dict:
    """Get detailed status of all data sources."""
    statuses = DataSourceRegistry.get_status()
    return {
        status.source.value: {
            "available": status.available,
            "reason": status.reason,
        }
        for status in statuses
    }
