"""
Pytest configuration and shared fixtures.
"""

import os
import sys

import pytest

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(autouse=True)
def disable_cache():
    """Disable caching for all tests."""
    os.environ["CACHE_ENABLED"] = "false"
    yield
    # Reset after test
    if "CACHE_ENABLED" in os.environ:
        del os.environ["CACHE_ENABLED"]


@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio for testing."""
    return {
        "cash": 100000.0,
        "stock": 0,
    }


@pytest.fixture
def sample_analyst_signals():
    """Create sample analyst signals for testing."""
    return {
        "technical_analyst_agent": {
            "signal": "bullish",
            "confidence": 75,
            "reasoning": {"trend": "upward"},
        },
        "fundamentals_agent": {
            "signal": "bullish",
            "confidence": 80,
            "reasoning": {"profitability": "strong"},
        },
        "sentiment_agent": {
            "signal": "neutral",
            "confidence": 50,
            "reasoning": "Mixed signals from insider trades",
        },
        "valuation_agent": {
            "signal": "bearish",
            "confidence": 60,
            "reasoning": {"dcf_analysis": "overvalued"},
        },
        "risk_management_agent": {
            "max_position_size": 20000,
            "reasoning": "Based on portfolio size and liquidity",
        },
    }


@pytest.fixture
def mock_price_data():
    """Create mock price data for testing."""
    return [
        {
            "time": "2024-01-01",
            "open": 100.0,
            "close": 105.0,
            "high": 106.0,
            "low": 99.0,
            "volume": 1000000,
        },
        {
            "time": "2024-01-02",
            "open": 105.0,
            "close": 108.0,
            "high": 109.0,
            "low": 104.0,
            "volume": 1100000,
        },
        {
            "time": "2024-01-03",
            "open": 108.0,
            "close": 110.0,
            "high": 112.0,
            "low": 107.0,
            "volume": 1200000,
        },
        {
            "time": "2024-01-04",
            "open": 110.0,
            "close": 107.0,
            "high": 111.0,
            "low": 106.0,
            "volume": 900000,
        },
        {
            "time": "2024-01-05",
            "open": 107.0,
            "close": 109.0,
            "high": 110.0,
            "low": 106.0,
            "volume": 950000,
        },
    ]
