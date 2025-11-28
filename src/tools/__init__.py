"""Tools package for AI Stock Analyst.

This package provides modular tools for fetching financial data from various sources.

Module Structure:
- api_client: Core HTTP client utilities (session, headers, base URL)
- price_tools: Stock price data (OHLCV)
- fundamental_tools: Financial metrics, line items, insider trades, market cap
- sentiment_tools: Analyst recommendations, Fear & Greed Index
- economic_tools: FRED data, yield curve, economic indicators
- holdings_tools: Institutional holders, major holders breakdown
- market_data_tools: Options data, earnings history, ESG scores
- sec_tools: SEC EDGAR filings
- data_sources: Data source registry and availability tracking
- cache: Caching utilities
- web_search: Web search functionality

For backward compatibility, the original api.py and free_data_sources.py are maintained.
"""

# Re-export from modular tools for convenience
from tools.api_client import (BASE_URL, DEFAULT_TIMEOUT, MAX_RETRIES,
                              YFINANCE_AVAILABLE, get_headers, get_session,
                              get_yfinance_ticker)
from tools.data_sources import (DataSourceRegistry, DataSourceStatus,
                                DataSourceType, get_source_status,
                                list_available_sources)
from tools.economic_tools import (FRED_SERIES, get_economic_indicators,
                                  get_fred_series, get_yield_curve)
from tools.fundamental_tools import (get_financial_metrics, get_insider_trades,
                                     get_market_cap, search_line_items)
from tools.holdings_tools import (get_institutional_holders,
                                  get_major_holders_breakdown)
from tools.market_data_tools import (get_earnings_history, get_esg_scores,
                                     get_options_data)
from tools.price_tools import get_price_data, get_prices, prices_to_df
from tools.sec_tools import get_sec_filings
from tools.sentiment_tools import (get_analyst_price_targets,
                                   get_analyst_recommendations,
                                   get_fear_greed_index)

__all__ = [
    # API Client
    "BASE_URL",
    "DEFAULT_TIMEOUT",
    "MAX_RETRIES",
    "YFINANCE_AVAILABLE",
    "get_headers",
    "get_session",
    "get_yfinance_ticker",
    # Data Sources
    "DataSourceRegistry",
    "DataSourceStatus",
    "DataSourceType",
    "get_source_status",
    "list_available_sources",
    # Price Tools
    "get_prices",
    "get_price_data",
    "prices_to_df",
    # Fundamental Tools
    "get_financial_metrics",
    "search_line_items",
    "get_insider_trades",
    "get_market_cap",
    # Sentiment Tools
    "get_analyst_recommendations",
    "get_analyst_price_targets",
    "get_fear_greed_index",
    # Economic Tools
    "FRED_SERIES",
    "get_fred_series",
    "get_economic_indicators",
    "get_yield_curve",
    # Holdings Tools
    "get_institutional_holders",
    "get_major_holders_breakdown",
    # Market Data Tools
    "get_options_data",
    "get_earnings_history",
    "get_esg_scores",
    # SEC Tools
    "get_sec_filings",
]
