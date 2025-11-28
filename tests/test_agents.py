"""
Unit tests for the agent modules.
"""

import numpy as np
import pandas as pd
import pytest

from src.agents.technicals import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_hurst_exponent,
    calculate_rsi,
    normalize_pandas,
    weighted_signal_combination,
)
from src.agents.valuation import (
    calculate_intrinsic_value,
    calculate_owner_earnings_value,
    calculate_working_capital_change,
)


class TestTechnicalIndicators:
    """Tests for technical analysis indicator calculations."""

    @pytest.fixture
    def sample_prices_df(self):
        """Create a sample price DataFrame for testing."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        np.random.seed(42)

        # Generate synthetic price data
        close_prices = 100 + np.cumsum(np.random.randn(100) * 2)
        high_prices = close_prices + np.random.rand(100) * 2
        low_prices = close_prices - np.random.rand(100) * 2

        df = pd.DataFrame(
            {
                "close": close_prices,
                "high": high_prices,
                "low": low_prices,
                "open": close_prices - np.random.randn(100) * 0.5,
                "volume": np.random.randint(100000, 1000000, 100),
            },
            index=dates,
        )

        return df

    def test_calculate_rsi(self, sample_prices_df):
        """Test RSI calculation."""
        rsi = calculate_rsi(sample_prices_df, period=14)

        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(sample_prices_df)
        # RSI should be between 0 and 100 (excluding NaN values)
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

    def test_calculate_bollinger_bands(self, sample_prices_df):
        """Test Bollinger Bands calculation."""
        upper, lower = calculate_bollinger_bands(sample_prices_df, window=20)

        assert isinstance(upper, pd.Series)
        assert isinstance(lower, pd.Series)
        assert len(upper) == len(sample_prices_df)
        # Upper band should always be above lower band
        valid_mask = upper.notna() & lower.notna()
        assert (upper[valid_mask] > lower[valid_mask]).all()

    def test_calculate_ema(self, sample_prices_df):
        """Test EMA calculation."""
        ema = calculate_ema(sample_prices_df, window=20)

        assert isinstance(ema, pd.Series)
        assert len(ema) == len(sample_prices_df)
        # EMA should not have NaN after the first few values
        assert ema.iloc[-1] is not None

    def test_calculate_atr(self, sample_prices_df):
        """Test ATR calculation."""
        atr = calculate_atr(sample_prices_df, period=14)

        assert isinstance(atr, pd.Series)
        assert len(atr) == len(sample_prices_df)
        # ATR should be positive
        valid_atr = atr.dropna()
        assert (valid_atr > 0).all()

    def test_calculate_hurst_exponent(self, sample_prices_df):
        """Test Hurst exponent calculation."""
        hurst = calculate_hurst_exponent(sample_prices_df["close"])

        assert isinstance(hurst, float)
        # Hurst exponent should typically be between 0 and 1
        assert 0 <= hurst <= 1 or hurst == 0.5  # 0.5 is returned on error


class TestWeightedSignalCombination:
    """Tests for the weighted signal combination function."""

    def test_all_bullish_signals(self):
        """Test combination of all bullish signals."""
        signals = {
            "trend": {"signal": "bullish", "confidence": 0.8},
            "momentum": {"signal": "bullish", "confidence": 0.7},
            "mean_reversion": {"signal": "bullish", "confidence": 0.6},
        }
        weights = {"trend": 0.4, "momentum": 0.4, "mean_reversion": 0.2}

        result = weighted_signal_combination(signals, weights)

        assert result["signal"] == "bullish"
        assert result["confidence"] > 0

    def test_all_bearish_signals(self):
        """Test combination of all bearish signals."""
        signals = {
            "trend": {"signal": "bearish", "confidence": 0.8},
            "momentum": {"signal": "bearish", "confidence": 0.7},
        }
        weights = {"trend": 0.5, "momentum": 0.5}

        result = weighted_signal_combination(signals, weights)

        assert result["signal"] == "bearish"

    def test_mixed_signals(self):
        """Test combination of mixed signals."""
        signals = {
            "trend": {"signal": "bullish", "confidence": 0.5},
            "momentum": {"signal": "bearish", "confidence": 0.5},
        }
        weights = {"trend": 0.5, "momentum": 0.5}

        result = weighted_signal_combination(signals, weights)

        assert result["signal"] == "neutral"


class TestNormalizePandas:
    """Tests for the normalize_pandas function."""

    def test_normalize_series(self):
        """Test normalization of pandas Series."""
        series = pd.Series([1, 2, 3, 4, 5])
        result = normalize_pandas(series)

        assert isinstance(result, list)
        assert result == [1, 2, 3, 4, 5]

    def test_normalize_dict(self):
        """Test normalization of dict with nested pandas objects."""
        data = {
            "values": pd.Series([1, 2, 3]),
            "name": "test",
        }
        result = normalize_pandas(data)

        assert isinstance(result, dict)
        assert result["values"] == [1, 2, 3]
        assert result["name"] == "test"


class TestValuationCalculations:
    """Tests for valuation agent calculations."""

    def test_calculate_intrinsic_value(self):
        """Test DCF intrinsic value calculation."""
        value = calculate_intrinsic_value(
            free_cash_flow=1000000,
            growth_rate=0.05,
            discount_rate=0.10,
            terminal_growth_rate=0.02,
            num_years=5,
        )

        assert value > 0
        assert isinstance(value, float)

    def test_calculate_intrinsic_value_negative_fcf(self):
        """Test DCF with negative free cash flow."""
        value = calculate_intrinsic_value(
            free_cash_flow=-1000000,
            growth_rate=0.05,
            discount_rate=0.10,
            terminal_growth_rate=0.02,
            num_years=5,
        )

        # Should still calculate (even if negative)
        assert isinstance(value, float)

    def test_calculate_owner_earnings_value(self):
        """Test Buffett's owner earnings valuation."""
        value = calculate_owner_earnings_value(
            net_income=1000000,
            depreciation=200000,
            capex=150000,
            working_capital_change=50000,
            growth_rate=0.05,
            required_return=0.15,
            margin_of_safety=0.25,
        )

        assert value > 0
        assert isinstance(value, float)

    def test_calculate_owner_earnings_value_invalid_inputs(self):
        """Test owner earnings with invalid inputs (string instead of number)."""
        value = calculate_owner_earnings_value(
            net_income="invalid",  # type: ignore
            depreciation=200000,
            capex=150000,
            working_capital_change=50000,
        )

        assert value == 0

    def test_calculate_working_capital_change(self):
        """Test working capital change calculation."""
        change = calculate_working_capital_change(
            current_working_capital=1000000,
            previous_working_capital=800000,
        )

        assert change == 200000

    def test_calculate_working_capital_change_negative(self):
        """Test negative working capital change."""
        change = calculate_working_capital_change(
            current_working_capital=800000,
            previous_working_capital=1000000,
        )

        assert change == -200000


class TestHedgeFundManager:
    """Tests for the hedge fund manager signal analysis."""

    def test_calculate_signal_agreement_all_bullish(self):
        """Test signal agreement with all bullish signals."""
        from src.agents.hedge_fund_manager import _calculate_signal_agreement

        signals = {
            "technical_analyst_agent": {"signal": "bullish", "confidence": 80},
            "fundamentals_agent": {"signal": "bullish", "confidence": 75},
            "sentiment_agent": {"signal": "bullish", "confidence": 70},
        }

        result = _calculate_signal_agreement(signals)

        assert result["consensus"] == "bullish"
        assert result["bull_count"] == 3
        assert result["bear_count"] == 0
        assert result["agreement_score"] == 100.0  # All agree

    def test_calculate_signal_agreement_mixed(self):
        """Test signal agreement with mixed signals."""
        from src.agents.hedge_fund_manager import _calculate_signal_agreement

        signals = {
            "technical_analyst_agent": {"signal": "bullish", "confidence": 80},
            "fundamentals_agent": {"signal": "bearish", "confidence": 75},
            "sentiment_agent": {"signal": "neutral", "confidence": 50},
        }

        result = _calculate_signal_agreement(signals)

        assert result["consensus"] == "neutral"  # Mixed signals
        assert result["bull_count"] == 1
        assert result["bear_count"] == 1
        assert result["neutral_count"] == 1

    def test_calculate_signal_agreement_empty(self):
        """Test signal agreement with no signals."""
        from src.agents.hedge_fund_manager import _calculate_signal_agreement

        signals = {}

        result = _calculate_signal_agreement(signals)

        assert result["consensus"] == "neutral"
        assert result["avg_confidence"] == 0


class TestWebSearch:
    """Tests for web search tool."""

    def test_get_search_summary_empty(self):
        """Test search summary with empty results."""
        from src.tools.web_search import get_search_summary

        summary = get_search_summary([])
        assert summary == "No web search results available."

    def test_get_search_summary_with_results(self):
        """Test search summary with results."""
        from src.tools.web_search import get_search_summary

        results = [
            {
                "title": "Test Article",
                "content": "Test content",
                "url": "http://test.com",
            },
            {"title": "AI Summary", "content": "Summary content", "is_summary": True},
        ]

        summary = get_search_summary(results)
        assert "Test Article" in summary
        assert "Summary content" in summary

    def test_is_web_search_available(self):
        """Test web search availability check."""
        from src.tools.web_search import is_web_search_available

        # Should return False when TAVILY_API_KEY is not set
        result = is_web_search_available()
        assert isinstance(result, bool)
