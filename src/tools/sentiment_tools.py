"""Sentiment analysis tools for market and stock sentiment data.

This module provides functions for fetching analyst recommendations,
price targets, and market sentiment indicators like Fear & Greed Index.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, cast

import pandas as pd
import requests

from tools.cache import cached
from tools.data_sources import DataSourceRegistry, DataSourceType

# Configure module logger
logger = logging.getLogger(__name__)


@cached(ttl_hours=24)
def get_analyst_recommendations(ticker: str) -> List[Dict[str, Any]]:
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

        if recommendations is None:
            logger.warning(f"No analyst recommendations found for {ticker}")
            return []

        # Handle both DataFrame and dict returns from yfinance
        if isinstance(recommendations, dict):
            logger.warning(f"No analyst recommendations found for {ticker}")
            return []

        if recommendations.empty:
            logger.warning(f"No analyst recommendations found for {ticker}")
            return []

        # Get recent recommendations (last 20)
        recent = recommendations.tail(20)
        result = []

        for idx, row in recent.iterrows():
            # idx is typically a Timestamp
            date_str = (
                cast(pd.Timestamp, idx).strftime("%Y-%m-%d")
                if hasattr(idx, "strftime")
                else str(idx)
            )
            result.append(
                {
                    "date": date_str,
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
def get_analyst_price_targets(ticker: str) -> Dict[str, Any]:
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

        targets: Dict[str, Any] = {
            "current_price": current_price,
            "target_low": info.get("targetLowPrice"),
            "target_high": info.get("targetHighPrice"),
            "target_mean": target_mean,
            "target_median": info.get("targetMedianPrice"),
            "number_of_analysts": info.get("numberOfAnalystOpinions"),
            "recommendation": info.get("recommendationKey"),
            "recommendation_mean": info.get(
                "recommendationMean"
            ),  # 1=Strong Buy, 5=Sell
            "upside_potential": None,
        }

        # Calculate upside/downside potential
        if current_price and target_mean:
            targets["upside_potential"] = round(
                (target_mean - current_price) / current_price * 100, 2
            )

        logger.info(f"Retrieved price targets for {ticker}")
        return targets

    except Exception as e:
        logger.error(f"Failed to fetch price targets for {ticker}: {e}")
        return {}


@cached(ttl_hours=6)  # Updates frequently
def get_fear_greed_index() -> Dict[str, Any]:
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

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

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
        def classify_score(s: Any) -> str:
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
