"""
Data providers module for the AI Stock Analyst.

This module provides data provider implementations for fetching market data.
The architecture is extensible, allowing easy addition of new data sources.

Currently supported:
- MassiveDataProvider (Polygon.io via Massive)
- YahooDataProvider (free fallback)

Future providers (easy to add):
- AlphaVantageDataProvider
- TwelveDataProvider
- FinancialModelingPrepProvider
"""

from data.providers.base import DataProviderFactory, get_data_provider
from data.providers.massive import MassiveDataProvider
from data.providers.yahoo import YahooDataProvider

__all__ = [
    "DataProviderFactory",
    "get_data_provider",
    "MassiveDataProvider",
    "YahooDataProvider",
]
