"""
Web Search Tool for AI Stock Analyst.

This module provides web search capabilities using Tavily API
to fetch real-time news and information about stocks and markets.
"""

import logging
import os
from datetime import datetime
from typing import Any, Optional

import requests

from tools.cache import cached

logger = logging.getLogger(__name__)


# Check for Tavily availability
TAVILY_AVAILABLE = bool(os.environ.get("TAVILY_API_KEY"))


def is_web_search_available() -> bool:
    """Check if web search is available."""
    return TAVILY_AVAILABLE


@cached(ttl_hours=1)  # Short cache for news freshness
def search_web(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    include_domains: Optional[list[str]] = None,
    exclude_domains: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """
    Search the web for relevant information using Tavily API.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)
        search_depth: "basic" or "advanced" (advanced is slower but more thorough)
        include_domains: List of domains to include in search
        exclude_domains: List of domains to exclude from search

    Returns:
        List of search results with title, url, content, and score
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        logger.warning("TAVILY_API_KEY not set. Get your API key at: https://tavily.com/")
        return []

    try:
        url = "https://api.tavily.com/search"

        payload: dict[str, Any] = {
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": True,
            "include_raw_content": False,
        }

        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains

        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()

        data = response.json()

        results: list[dict[str, Any]] = []
        for result in data.get("results", []):
            results.append(
                {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0),
                    "published_date": result.get("published_date"),
                }
            )

        # Include the AI-generated answer if available
        answer = data.get("answer")
        if answer:
            results.insert(
                0,
                {
                    "title": "AI Summary",
                    "url": "",
                    "content": answer,
                    "score": 1.0,
                    "is_summary": True,
                },
            )

        logger.info(f"Web search for '{query}' returned {len(results)} results")
        return results

    except requests.exceptions.RequestException as e:
        logger.error(f"Web search failed: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in web search: {e}")
        return []


@cached(ttl_hours=1)
def search_stock_news(
    ticker: str,
    company_name: Optional[str] = None,
    max_results: int = 5,
) -> list[dict[str, Any]]:
    """
    Search for recent news about a specific stock.

    Args:
        ticker: Stock ticker symbol
        company_name: Optional company name for better search results
        max_results: Maximum number of results to return

    Returns:
        List of news articles with title, url, content, and sentiment hints
    """
    # Build search query
    if company_name:
        query = f"{company_name} ({ticker}) stock news latest"
    else:
        query = f"{ticker} stock news latest market analysis"

    # Prefer financial news sources
    include_domains = [
        "reuters.com",
        "bloomberg.com",
        "cnbc.com",
        "wsj.com",
        "ft.com",
        "marketwatch.com",
        "seekingalpha.com",
        "fool.com",
        "yahoo.com",
        "investing.com",
        "barrons.com",
    ]

    results = search_web(
        query=query,
        max_results=max_results,
        search_depth="basic",
        include_domains=include_domains,
    )

    # Add ticker context to results
    for result in results:
        result["ticker"] = ticker
        result["search_time"] = datetime.now().isoformat()

    return results


@cached(ttl_hours=1)
def search_market_news(
    topic: str = "stock market",
    max_results: int = 5,
) -> list[dict[str, Any]]:
    """
    Search for general market news and analysis.

    Args:
        topic: Market topic to search for (e.g., "stock market", "fed interest rates")
        max_results: Maximum number of results to return

    Returns:
        List of market news articles
    """
    query = f"{topic} latest news today"

    # Financial news sources
    include_domains = [
        "reuters.com",
        "bloomberg.com",
        "cnbc.com",
        "wsj.com",
        "ft.com",
        "marketwatch.com",
    ]

    return search_web(
        query=query,
        max_results=max_results,
        search_depth="basic",
        include_domains=include_domains,
    )


@cached(ttl_hours=2)
def search_company_analysis(
    ticker: str,
    company_name: Optional[str] = None,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """
    Search for analyst reports and company analysis.

    Args:
        ticker: Stock ticker symbol
        company_name: Optional company name
        max_results: Maximum number of results

    Returns:
        List of analyst reports and analysis
    """
    if company_name:
        query = f"{company_name} {ticker} analyst report price target rating"
    else:
        query = f"{ticker} stock analyst report price target rating 2024"

    return search_web(
        query=query,
        max_results=max_results,
        search_depth="advanced",  # Use advanced for more thorough analysis
    )


def get_search_summary(results: list[dict[str, Any]]) -> str:
    """
    Generate a text summary from search results for LLM consumption.

    Args:
        results: List of search results

    Returns:
        Formatted string summary of search results
    """
    if not results:
        return "No web search results available."

    summary_parts = []
    for i, result in enumerate(results, 1):
        if result.get("is_summary"):
            summary_parts.append(f"**AI Summary**: {result['content']}\n")
        else:
            title = result.get("title", "Untitled")
            content = result.get("content", "")[:500]  # Truncate long content
            url = result.get("url", "")
            summary_parts.append(f"{i}. **{title}**\n   {content}\n   Source: {url}\n")

    return "\n".join(summary_parts)


# Alternative: DuckDuckGo search (free, no API key required)
@cached(ttl_hours=1)
def search_duckduckgo(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """
    Search using DuckDuckGo (free, no API key required).

    This is a fallback when Tavily API is not available.

    Args:
        query: Search query
        max_results: Maximum results to return

    Returns:
        List of search results
    """
    try:
        # DuckDuckGo Instant Answer API
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        results: list[dict[str, Any]] = []

        # Abstract (main result)
        if data.get("Abstract"):
            results.append(
                {
                    "title": data.get("Heading", "Summary"),
                    "url": data.get("AbstractURL", ""),
                    "content": data.get("Abstract", ""),
                    "source": data.get("AbstractSource", ""),
                }
            )

        # Related topics
        for topic in data.get("RelatedTopics", [])[: max_results - 1]:
            if isinstance(topic, dict) and "Text" in topic:
                results.append(
                    {
                        "title": topic.get("Text", "")[:100],
                        "url": topic.get("FirstURL", ""),
                        "content": topic.get("Text", ""),
                    }
                )

        logger.info(f"DuckDuckGo search for '{query}' returned {len(results)} results")
        return results

    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {e}")
        return []


def smart_search(
    query: str,
    max_results: int = 5,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """
    Smart search that uses Tavily if available, falls back to DuckDuckGo.

    Args:
        query: Search query
        max_results: Maximum results
        **kwargs: Additional arguments for Tavily

    Returns:
        Search results from best available source
    """
    if TAVILY_AVAILABLE:
        results = search_web(query, max_results, **kwargs)
        if results:
            return results

    # Fallback to DuckDuckGo
    logger.info("Using DuckDuckGo fallback for web search")
    return search_duckduckgo(query, max_results)
