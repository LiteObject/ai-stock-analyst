"""
Hedge Fund Manager Agent.

This agent acts as the chief investment officer, synthesizing signals from all
analysts and making sophisticated trading decisions with detailed reasoning.
It considers market context, news, and applies hedge fund-style risk management.
"""

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from graph.state import AgentState, show_agent_reasoning
from llm import get_llm

# Configure logger
logger = logging.getLogger(__name__)

# Web search functions (initialized to None, set if available)
_search_stock_news = None
_get_search_summary = None
_is_web_search_available = None
WEB_SEARCH_AVAILABLE = False

# Try to import web search for news context
try:
    from tools.web_search import get_search_summary as _gss
    from tools.web_search import is_web_search_available as _iswa
    from tools.web_search import search_stock_news as _ssn

    _search_stock_news = _ssn
    _get_search_summary = _gss
    _is_web_search_available = _iswa
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    logger.info("Web search module not available")


def _get_news_context(ticker: str) -> str:
    """Fetch recent news context for the ticker."""
    if not WEB_SEARCH_AVAILABLE:
        return "No recent news available (web search not configured)."

    if _is_web_search_available is None or not _is_web_search_available():
        return "No recent news available (web search not configured)."

    try:
        if _search_stock_news is None or _get_search_summary is None:
            return "No recent news available (web search not configured)."
        news = _search_stock_news(ticker, max_results=3)
        if news:
            return _get_search_summary(news)
        return "No recent news found for this ticker."
    except Exception as e:
        logger.warning(f"Failed to fetch news for {ticker}: {e}")
        return "Unable to fetch recent news."


def _calculate_signal_agreement(analyst_signals: dict[str, Any]) -> dict[str, Any]:
    """Calculate the level of agreement between analyst signals."""
    signals = []
    confidences = []

    signal_mapping = {"bullish": 1, "neutral": 0, "bearish": -1}

    for agent_name, signal_data in analyst_signals.items():
        if agent_name == "risk_management_agent":
            continue  # Skip risk management

        if isinstance(signal_data, dict):
            signal = signal_data.get("signal", "neutral")
            confidence = signal_data.get("confidence", 50)

            if signal in signal_mapping:
                signals.append(signal_mapping[signal])
                confidences.append(confidence)

    if not signals:
        return {
            "consensus": "neutral",
            "agreement_score": 0,
            "avg_confidence": 0,
            "bull_count": 0,
            "bear_count": 0,
            "neutral_count": 0,
        }

    avg_signal = sum(signals) / len(signals)
    avg_confidence = sum(confidences) / len(confidences)

    bull_count = sum(1 for s in signals if s > 0)
    bear_count = sum(1 for s in signals if s < 0)
    neutral_count = sum(1 for s in signals if s == 0)

    # Determine consensus
    if avg_signal > 0.3:
        consensus = "bullish"
    elif avg_signal < -0.3:
        consensus = "bearish"
    else:
        consensus = "neutral"

    # Calculate agreement score (how much analysts agree)
    if len(signals) > 1:
        variance = sum((s - avg_signal) ** 2 for s in signals) / len(signals)
        agreement_score = max(0, 100 - variance * 100)
    else:
        agreement_score = 100

    return {
        "consensus": consensus,
        "agreement_score": round(agreement_score, 1),
        "avg_confidence": round(avg_confidence, 1),
        "avg_signal_strength": round(avg_signal, 2),
        "bull_count": bull_count,
        "bear_count": bear_count,
        "neutral_count": neutral_count,
    }


# === Hedge Fund Manager Agent ===
def hedge_fund_manager_agent(state: AgentState) -> dict[str, Any]:
    """
    Chief Investment Officer that synthesizes all analyst signals.

    This agent acts as a sophisticated hedge fund manager, considering:
    - All analyst signals and their confidence levels
    - Market context and recent news
    - Risk management constraints
    - Portfolio composition and available capital
    - Signal agreement/disagreement between analysts

    It applies hedge fund-style thinking:
    - Position sizing based on conviction
    - Risk-adjusted returns consideration
    - Contrarian thinking when appropriate
    - Clear reasoning for the decision

    Args:
        state: Current agent state with analyst signals and portfolio info

    Returns:
        Updated state with final trading decision
    """
    logger.info("Running hedge fund manager agent")

    data = state["data"]
    ticker = data.get("ticker", "UNKNOWN")
    portfolio = data["portfolio"]
    analyst_signals = data.get("analyst_signals", {})

    # Calculate signal agreement
    signal_analysis = _calculate_signal_agreement(analyst_signals)

    # Get news context if available
    news_context = _get_news_context(ticker)

    # Create the sophisticated prompt
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a world-class hedge fund manager with decades of experience.
Your job is to synthesize all analyst inputs and make a final trading decision.

You think like a professional investor:
1. You weigh each analyst's signal by their confidence and track record
2. You consider whether analysts agree or disagree (disagreement = uncertainty)
3. You factor in risk management constraints
4. You consider position sizing based on conviction
5. You look for asymmetric risk/reward opportunities
6. You're willing to go against consensus if you have strong conviction

Key principles:
- High conviction + high agreement = larger position
- Low conviction or disagreement = smaller position or pass
- Never risk more than risk management allows
- Consider news and market context
- Protect capital first, grow it second

Output your decision as JSON with these fields:
- "action": "buy" | "sell" | "hold"
- "quantity": <positive integer or 0>
- "confidence": <float between 0 and 1>
- "reasoning": <detailed explanation of your decision process>
- "risk_reward_assessment": <brief assessment of the risk/reward>
- "key_factors": [<list of the top 3 factors that influenced your decision>]""",
            ),
            (
                "human",
                """Make your trading decision for {ticker}.

=== ANALYST SIGNALS ===

Technical Analysis:
{technical_signal}

Fundamental Analysis:
{fundamentals_signal}

Sentiment Analysis:
{sentiment_signal}

Valuation Analysis:
{valuation_signal}

=== SIGNAL AGREEMENT ANALYSIS ===
Consensus: {consensus}
Agreement Score: {agreement_score}% (higher = more agreement between analysts)
Average Confidence: {avg_confidence}%
Signal Breakdown: {bull_count} bullish, {bear_count} bearish, {neutral_count} neutral

=== RISK MANAGEMENT ===
Maximum Position Size: {max_position_size} shares
Risk Assessment: {risk_reasoning}

=== RECENT NEWS & CONTEXT ===
{news_context}

=== PORTFOLIO STATUS ===
Available Cash: ${portfolio_cash}
Current Position: {portfolio_stock} shares of {ticker}

=== DECISION RULES ===
- You can only BUY if you have available cash
- You can only SELL if you have shares to sell
- Quantity must not exceed max_position_size
- If holding, quantity should be 0

Provide your decision as JSON (no markdown formatting):""",
            ),
        ]
    )

    # Get risk management info
    risk_info = analyst_signals.get("risk_management_agent", {})

    # Generate the prompt
    prompt = template.invoke(
        {
            "ticker": ticker,
            "technical_signal": json.dumps(analyst_signals.get("technical_analyst_agent", {}), indent=2),
            "fundamentals_signal": json.dumps(analyst_signals.get("fundamentals_agent", {}), indent=2),
            "sentiment_signal": json.dumps(analyst_signals.get("sentiment_agent", {}), indent=2),
            "valuation_signal": json.dumps(analyst_signals.get("valuation_agent", {}), indent=2),
            "consensus": signal_analysis["consensus"],
            "agreement_score": signal_analysis["agreement_score"],
            "avg_confidence": signal_analysis["avg_confidence"],
            "bull_count": signal_analysis["bull_count"],
            "bear_count": signal_analysis["bear_count"],
            "neutral_count": signal_analysis["neutral_count"],
            "max_position_size": risk_info.get("max_position_size", 0),
            "risk_reasoning": risk_info.get("reasoning", "No risk assessment available"),
            "news_context": news_context,
            "portfolio_cash": f"{portfolio['cash']:.2f}",
            "portfolio_stock": portfolio["stock"],
        }
    )

    # Invoke the LLM
    llm = get_llm()
    result = llm.invoke(prompt)

    # Create the message
    message = HumanMessage(
        content=result.content,
        name="hedge_fund_manager",
    )

    # Check if web search was used
    news_used = False
    if WEB_SEARCH_AVAILABLE and _is_web_search_available is not None:
        news_used = _is_web_search_available()

    # Store additional analysis in the state
    updated_data = {
        **state["data"],
        "signal_analysis": signal_analysis,
        "news_context_used": news_used,
    }

    # Show reasoning if requested
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(message.content, "Hedge Fund Manager")

        # Also show signal analysis
        print("\n" + "=" * 48)
        print("Signal Analysis Summary".center(48))
        print("=" * 48)
        print(f"Consensus: {signal_analysis['consensus'].upper()}")
        print(f"Agreement Score: {signal_analysis['agreement_score']}%")
        print(f"Average Confidence: {signal_analysis['avg_confidence']}%")
        print(
            f"Breakdown: {signal_analysis['bull_count']}🟢 "
            f"{signal_analysis['bear_count']}🔴 "
            f"{signal_analysis['neutral_count']}⚪"
        )
        print("=" * 48)

    return {
        "messages": list(state["messages"]) + [message],
        "data": updated_data,
    }
