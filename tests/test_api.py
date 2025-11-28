"""
Unit tests for the API tools module.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tools.api import (  # noqa: E402
    get_financial_metrics,
    get_insider_trades,
    get_market_cap,
    get_prices,
    prices_to_df,
)


class TestGetFinancialMetrics:
    """Tests for the get_financial_metrics function."""

    @patch("tools.api._get_session")
    def test_get_financial_metrics_success(self, mock_get_session):
        """Test successful retrieval of financial metrics."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "financial_metrics": [
                {
                    "return_on_equity": 0.25,
                    "net_margin": 0.15,
                    "operating_margin": 0.20,
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        mock_get_session.return_value = mock_session

        result = get_financial_metrics("AAPL", "2024-01-01")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["return_on_equity"] == 0.25

    @patch("tools.api._get_financial_metrics_from_yfinance")
    @patch("tools.api._get_session")
    def test_get_financial_metrics_api_error(self, mock_get_session, mock_yfinance):
        """Test handling of API errors when both primary and fallback fail."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = requests.RequestException("Unauthorized")

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        mock_get_session.return_value = mock_session

        # Mock fallback to also fail
        mock_yfinance.side_effect = requests.RequestException("Fallback failed")

        with pytest.raises(requests.RequestException):
            get_financial_metrics("AAPL", "2024-01-01")

    @patch("tools.api._get_financial_metrics_from_yfinance")
    @patch("tools.api._get_session")
    def test_get_financial_metrics_no_data(self, mock_get_session, mock_yfinance):
        """Test handling when no data is returned and fallback also fails."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"financial_metrics": None}
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        mock_get_session.return_value = mock_session

        # Mock fallback to fail with ValueError
        mock_yfinance.side_effect = ValueError("No data from Yahoo Finance")

        with pytest.raises(ValueError) as exc_info:
            get_financial_metrics("INVALID", "2024-01-01")

        assert "No data from Yahoo Finance" in str(exc_info.value)

    @patch("tools.api._get_financial_metrics_from_yfinance")
    @patch("tools.api._get_session")
    def test_get_financial_metrics_with_fallback(self, mock_get_session, mock_yfinance):
        """Test Yahoo Finance fallback when primary API fails."""
        # Primary API fails
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = requests.RequestException("Unauthorized")

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        mock_get_session.return_value = mock_session

        # Yahoo Finance fallback succeeds
        mock_yfinance.return_value = [{"return_on_equity": 0.20, "source": "yahoo_finance"}]

        result = get_financial_metrics("AAPL", "2024-01-01")

        assert isinstance(result, list)
        assert result[0]["source"] == "yahoo_finance"
        mock_yfinance.assert_called_once()


class TestGetPrices:
    """Tests for the get_prices function."""

    @patch("tools.api._get_session")
    def test_get_prices_success(self, mock_get_session):
        """Test successful retrieval of price data."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "prices": [
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
                    "close": 110.0,
                    "high": 111.0,
                    "low": 104.0,
                    "volume": 1200000,
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        mock_get_session.return_value = mock_session

        result = get_prices("AAPL", "2024-01-01", "2024-01-31")

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["close"] == 105.0

    @patch("tools.api.YFINANCE_AVAILABLE", False)
    @patch("tools.api._get_session")
    def test_get_prices_no_data(self, mock_get_session):
        """Test handling when no price data is returned and no fallback available."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"prices": None}
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        mock_get_session.return_value = mock_session

        with pytest.raises(ValueError) as exc_info:
            get_prices("INVALID", "2024-01-01", "2024-01-31")

        assert "No price data returned" in str(exc_info.value)


class TestGetInsiderTrades:
    """Tests for the get_insider_trades function."""

    @patch("tools.api._get_session")
    def test_get_insider_trades_success(self, mock_get_session):
        """Test successful retrieval of insider trades."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "insider_trades": [
                {"transaction_shares": 1000, "transaction_type": "P"},
                {"transaction_shares": -500, "transaction_type": "S"},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        mock_get_session.return_value = mock_session

        result = get_insider_trades("AAPL", "2024-01-31")

        assert isinstance(result, list)
        assert len(result) == 2


class TestPricesToDf:
    """Tests for the prices_to_df function."""

    def test_prices_to_df_conversion(self):
        """Test conversion of price data to DataFrame."""
        prices = [
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
                "close": 110.0,
                "high": 111.0,
                "low": 104.0,
                "volume": 1200000,
            },
        ]

        df = prices_to_df(prices)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "close" in df.columns
        assert "volume" in df.columns
        assert df.index.name == "Date"

    def test_prices_to_df_sorting(self):
        """Test that prices are sorted by date."""
        prices = [
            {
                "time": "2024-01-02",
                "open": 105.0,
                "close": 110.0,
                "high": 111.0,
                "low": 104.0,
                "volume": 1200000,
            },
            {
                "time": "2024-01-01",
                "open": 100.0,
                "close": 105.0,
                "high": 106.0,
                "low": 99.0,
                "volume": 1000000,
            },
        ]

        df = prices_to_df(prices)

        # First row should be earlier date
        assert df.iloc[0]["close"] == 105.0


class TestGetMarketCap:
    """Tests for the get_market_cap function."""

    @patch("tools.api._get_session")
    def test_get_market_cap_success(self, mock_get_session):
        """Test successful retrieval of market cap."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"company_facts": {"market_cap": 3000000000000}}
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        mock_get_session.return_value = mock_session

        result = get_market_cap("AAPL")

        assert result == 3000000000000
