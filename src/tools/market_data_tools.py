"""Market data tools for options, earnings, and ESG data.

This module provides functions for fetching options data, earnings history,
and ESG (Environmental, Social, Governance) scores from Yahoo Finance.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, cast

import pandas as pd

from tools.cache import cached
from tools.data_sources import DataSourceRegistry, DataSourceType

# Configure module logger
logger = logging.getLogger(__name__)


@cached(ttl_hours=6)  # Options data changes more frequently
def get_options_data(ticker: str) -> Dict[str, Any]:
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
        total_call_oi = (
            calls["openInterest"].sum() if "openInterest" in calls.columns else 0
        )
        total_put_oi = (
            puts["openInterest"].sum() if "openInterest" in puts.columns else 0
        )

        put_call_volume_ratio = (
            total_put_volume / total_call_volume if total_call_volume > 0 else None
        )
        put_call_oi_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else None

        # Average implied volatility (weighted by open interest)
        avg_call_iv = None
        avg_put_iv = None

        if "impliedVolatility" in calls.columns and "openInterest" in calls.columns:
            call_weights = calls["openInterest"].fillna(0)
            if call_weights.sum() > 0:
                avg_call_iv = (
                    calls["impliedVolatility"] * call_weights
                ).sum() / call_weights.sum()

        if "impliedVolatility" in puts.columns and "openInterest" in puts.columns:
            put_weights = puts["openInterest"].fillna(0)
            if put_weights.sum() > 0:
                avg_put_iv = (
                    puts["impliedVolatility"] * put_weights
                ).sum() / put_weights.sum()

        result = {
            "nearest_expiry": nearest_expiry,
            "days_to_expiry": (
                datetime.strptime(nearest_expiry, "%Y-%m-%d") - datetime.now()
            ).days,
            "total_call_volume": int(total_call_volume) if total_call_volume else 0,
            "total_put_volume": int(total_put_volume) if total_put_volume else 0,
            "total_call_oi": int(total_call_oi) if total_call_oi else 0,
            "total_put_oi": int(total_put_oi) if total_put_oi else 0,
            "put_call_volume_ratio": (
                round(put_call_volume_ratio, 3) if put_call_volume_ratio else None
            ),
            "put_call_oi_ratio": (
                round(put_call_oi_ratio, 3) if put_call_oi_ratio else None
            ),
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
def get_earnings_history(ticker: str) -> List[Dict[str, Any]]:
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

        if earnings is None or (hasattr(earnings, "empty") and earnings.empty):
            return []

        result = []
        for idx, row in earnings.iterrows():
            surprise_pct = None
            if row.get("epsActual") and row.get("epsEstimate"):
                surprise_pct = (
                    (row["epsActual"] - row["epsEstimate"]) / abs(row["epsEstimate"])
                ) * 100

            # idx is typically a Timestamp
            date_str = (
                cast(pd.Timestamp, idx).strftime("%Y-%m-%d")
                if hasattr(idx, "strftime")
                else str(idx)
            )
            result.append(
                {
                    "date": date_str,
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
def get_esg_scores(ticker: str) -> Dict[str, Any]:
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

        if esg is None or (hasattr(esg, "empty") and esg.empty):
            logger.warning(f"No ESG data available for {ticker}")
            return {}

        # Extract ESG scores
        result: Dict[str, Any] = {}
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
                value = (
                    esg.loc[key].values[0]
                    if hasattr(esg.loc[key], "values")
                    else esg.loc[key]
                )
                result[key] = (
                    float(value) if isinstance(value, (int, float)) else str(value)
                )

        logger.info(f"Retrieved ESG scores for {ticker}")
        return result

    except Exception as e:
        logger.error(f"Failed to fetch ESG scores for {ticker}: {e}")
        return {}
