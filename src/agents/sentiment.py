"""Sentiment analysis agent using multiple free data sources.

Uses:
- Insider trading activity
- Analyst recommendations and price targets
- Options market sentiment (put/call ratio)
- Fear & Greed Index (market-wide sentiment)
- Institutional holder changes
"""

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage

from graph.state import AgentState, show_agent_reasoning
from tools.api import get_insider_trades

# Configure logger
logger = logging.getLogger(__name__)

# Try to import free data sources
try:
    from tools.free_data_sources import (
        get_analyst_price_targets,
        get_analyst_recommendations,
        get_fear_greed_index,
        get_options_data,
    )

    FREE_DATA_AVAILABLE = True
except ImportError:
    FREE_DATA_AVAILABLE = False
    logger.warning("Free data sources module not available")


def _analyze_insider_trades(ticker: str, end_date: str) -> dict[str, Any]:
    """Analyze insider trading activity."""
    try:
        insider_trades = get_insider_trades(ticker=ticker, end_date=end_date, limit=5)

        if not insider_trades:
            return {"signal": "neutral", "confidence": 0, "details": "No data"}

        transaction_shares = pd.Series([t.get("transaction_shares") for t in insider_trades]).dropna()

        if transaction_shares.empty:
            return {"signal": "neutral", "confidence": 0, "details": "No data"}

        bearish_condition = transaction_shares < 0
        signals = np.where(bearish_condition, "bearish", "bullish").tolist()

        bullish = signals.count("bullish")
        bearish = signals.count("bearish")
        total = len(signals)

        if bullish > bearish:
            signal = "bullish"
        elif bearish > bullish:
            signal = "bearish"
        else:
            signal = "neutral"

        confidence = round(max(bullish, bearish) / total * 100, 1) if total > 0 else 0

        return {
            "signal": signal,
            "confidence": confidence,
            "details": f"Buys: {bullish}, Sells: {bearish}",
        }
    except Exception as e:
        logger.warning(f"Failed to analyze insider trades: {e}")
        return {"signal": "neutral", "confidence": 0, "details": str(e)}


def _analyze_analyst_sentiment(ticker: str) -> dict[str, Any]:
    """Analyze analyst recommendations and price targets."""
    if not FREE_DATA_AVAILABLE:
        return {"signal": "neutral", "confidence": 0, "details": "Not available"}

    try:
        # Get price targets
        targets = get_analyst_price_targets(ticker)
        recommendations = get_analyst_recommendations(ticker)

        signals = []
        details = []

        # Analyze price target upside
        if targets.get("upside_potential") is not None:
            upside = targets["upside_potential"]
            if upside > 15:
                signals.append("bullish")
                details.append(f"Upside: {upside:.1f}%")
            elif upside < -15:
                signals.append("bearish")
                details.append(f"Downside: {upside:.1f}%")
            else:
                signals.append("neutral")
                details.append(f"Target: {upside:+.1f}%")

        # Analyze recommendation mean (1=Strong Buy, 5=Sell)
        rec_mean = targets.get("recommendation_mean")
        if rec_mean:
            if rec_mean <= 2.0:
                signals.append("bullish")
                details.append(f"Rec: {rec_mean:.1f} (Buy)")
            elif rec_mean >= 4.0:
                signals.append("bearish")
                details.append(f"Rec: {rec_mean:.1f} (Sell)")
            else:
                signals.append("neutral")
                details.append(f"Rec: {rec_mean:.1f} (Hold)")

        # Analyze recent recommendation changes
        if recommendations:
            recent = recommendations[:5]
            upgrades = sum(1 for r in recent if "upgrade" in r.get("action", "").lower())
            downgrades = sum(1 for r in recent if "downgrade" in r.get("action", "").lower())

            if upgrades > downgrades:
                signals.append("bullish")
                details.append(f"Recent: {upgrades} upgrades")
            elif downgrades > upgrades:
                signals.append("bearish")
                details.append(f"Recent: {downgrades} downgrades")

        if not signals:
            return {"signal": "neutral", "confidence": 0, "details": "No data"}

        bullish = signals.count("bullish")
        bearish = signals.count("bearish")

        if bullish > bearish:
            signal = "bullish"
        elif bearish > bullish:
            signal = "bearish"
        else:
            signal = "neutral"

        confidence = round(max(bullish, bearish) / len(signals) * 100, 1)

        return {
            "signal": signal,
            "confidence": confidence,
            "details": ", ".join(details),
        }
    except Exception as e:
        logger.warning(f"Failed to analyze analyst sentiment: {e}")
        return {"signal": "neutral", "confidence": 0, "details": str(e)}


def _analyze_options_sentiment(ticker: str) -> dict[str, Any]:
    """Analyze options market sentiment via put/call ratio."""
    if not FREE_DATA_AVAILABLE:
        return {"signal": "neutral", "confidence": 0, "details": "Not available"}

    try:
        options = get_options_data(ticker)

        if not options:
            return {"signal": "neutral", "confidence": 0, "details": "No data"}

        # Put/Call ratio analysis
        # < 0.7 = bullish (more calls), > 1.0 = bearish (more puts)
        pc_ratio = options.get("put_call_volume_ratio") or options.get("put_call_oi_ratio")

        if pc_ratio is None:
            return {"signal": "neutral", "confidence": 0, "details": "No P/C ratio"}

        if pc_ratio < 0.7:
            signal = "bullish"
            details = f"P/C: {pc_ratio:.2f} (bullish)"
            confidence = min((0.7 - pc_ratio) / 0.3 * 100, 100)
        elif pc_ratio > 1.0:
            signal = "bearish"
            details = f"P/C: {pc_ratio:.2f} (bearish)"
            confidence = min((pc_ratio - 1.0) / 0.5 * 100, 100)
        else:
            signal = "neutral"
            details = f"P/C: {pc_ratio:.2f} (neutral)"
            confidence = 50

        # Add implied volatility context
        avg_iv = options.get("avg_call_iv") or options.get("avg_put_iv")
        if avg_iv:
            details += f", IV: {avg_iv*100:.1f}%"

        return {
            "signal": signal,
            "confidence": round(confidence, 1),
            "details": details,
        }
    except Exception as e:
        logger.warning(f"Failed to analyze options sentiment: {e}")
        return {"signal": "neutral", "confidence": 0, "details": str(e)}


def _analyze_market_fear_greed() -> dict[str, Any]:
    """Analyze market-wide Fear & Greed Index."""
    if not FREE_DATA_AVAILABLE:
        return {"signal": "neutral", "confidence": 0, "details": "Not available"}

    try:
        fg = get_fear_greed_index()

        if not fg or fg.get("score") is None:
            return {"signal": "neutral", "confidence": 0, "details": "No data"}

        score = fg["score"]

        # Contrarian approach: extreme fear = buying opportunity, extreme greed = caution
        if score <= 25:
            signal = "bullish"  # Contrarian: extreme fear = buy
            details = f"F&G: {score} (Extreme Fear - contrarian bullish)"
        elif score <= 45:
            signal = "neutral"
            details = f"F&G: {score} (Fear)"
        elif score <= 55:
            signal = "neutral"
            details = f"F&G: {score} (Neutral)"
        elif score <= 75:
            signal = "neutral"
            details = f"F&G: {score} (Greed)"
        else:
            signal = "bearish"  # Contrarian: extreme greed = sell
            details = f"F&G: {score} (Extreme Greed - contrarian bearish)"

        # Confidence based on extremity
        confidence = abs(score - 50) * 2

        return {
            "signal": signal,
            "confidence": round(confidence, 1),
            "details": details,
        }
    except Exception as e:
        logger.warning(f"Failed to analyze Fear & Greed: {e}")
        return {"signal": "neutral", "confidence": 0, "details": str(e)}


def sentiment_agent(state: AgentState):
    """Analyzes market sentiment using multiple data sources.

    Uses:
    - Insider trading activity
    - Analyst recommendations and price targets
    - Options market sentiment (put/call ratio)
    - Fear & Greed Index (market-wide sentiment)

    Args:
        state: Current agent state with ticker and date info

    Returns:
        Updated state with sentiment analysis signals
    """
    data = state.get("data", {})
    end_date = data.get("end_date")
    ticker = data.get("ticker")

    logger.info(f"Running sentiment analysis for {ticker}")

    # Collect signals from all sources
    sentiment_signals = {}

    # 1. Insider trading (weight: 30%)
    sentiment_signals["insider_trading"] = _analyze_insider_trades(ticker, end_date)

    # 2. Analyst recommendations (weight: 30%)
    sentiment_signals["analyst_sentiment"] = _analyze_analyst_sentiment(ticker)

    # 3. Options market (weight: 25%)
    sentiment_signals["options_sentiment"] = _analyze_options_sentiment(ticker)

    # 4. Fear & Greed Index (weight: 15% - market-wide, less stock-specific)
    sentiment_signals["fear_greed"] = _analyze_market_fear_greed()

    # Weight the signals
    weights = {
        "insider_trading": 0.30,
        "analyst_sentiment": 0.30,
        "options_sentiment": 0.25,
        "fear_greed": 0.15,
    }

    # Calculate weighted score
    # bullish = +1, neutral = 0, bearish = -1
    signal_values = {"bullish": 1, "neutral": 0, "bearish": -1}

    weighted_score = 0
    total_weight = 0
    weighted_confidence = 0

    for source, result in sentiment_signals.items():
        signal = result.get("signal", "neutral")
        confidence = result.get("confidence", 0)
        weight = weights.get(source, 0)

        # Only count signals with some confidence
        if confidence > 0:
            weighted_score += signal_values.get(signal, 0) * weight * (confidence / 100)
            weighted_confidence += confidence * weight
            total_weight += weight

    # Normalize
    if total_weight > 0:
        weighted_score /= total_weight
        weighted_confidence /= total_weight

    # Determine overall signal
    if weighted_score > 0.2:
        overall_signal = "bullish"
    elif weighted_score < -0.2:
        overall_signal = "bearish"
    else:
        overall_signal = "neutral"

    # Build reasoning
    reasoning = {
        source: {
            "signal": result.get("signal"),
            "confidence": result.get("confidence"),
            "details": result.get("details"),
        }
        for source, result in sentiment_signals.items()
    }

    message_content = {
        "signal": overall_signal,
        "confidence": round(weighted_confidence, 1),
        "reasoning": reasoning,
    }

    # Print the reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(message_content, "Sentiment Analysis Agent")

    # Create the sentiment message
    message = HumanMessage(
        content=json.dumps(message_content),
        name="sentiment_agent",
    )

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["sentiment_agent"] = {
        "signal": overall_signal,
        "confidence": round(weighted_confidence, 1),
        "reasoning": reasoning,
    }

    return {
        "messages": [message],
        "data": data,
    }
