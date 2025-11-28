"""Data Source Registry for managing available data sources.

This module provides a registry for tracking and managing different
financial data sources (Yahoo Finance, FRED, SEC EDGAR, etc.).
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

# Configure module logger
logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Types of data sources available."""

    YAHOO_FINANCE = "yahoo_finance"
    FRED = "fred"
    SEC_EDGAR = "sec_edgar"
    FEAR_GREED = "fear_greed"


@dataclass
class DataSourceStatus:
    """Status of a data source."""

    source: DataSourceType
    available: bool
    reason: str = ""


class DataSourceRegistry:
    """Registry for managing available data sources."""

    _sources: Dict[DataSourceType, bool] = {}
    _initialized: bool = False

    @classmethod
    def initialize(cls) -> None:
        """Initialize and check availability of all data sources."""
        if cls._initialized:
            return

        # Check Yahoo Finance
        try:
            import yfinance  # noqa: F401

            cls._sources[DataSourceType.YAHOO_FINANCE] = True
        except ImportError:
            cls._sources[DataSourceType.YAHOO_FINANCE] = False
            logger.warning("yfinance not installed. Install with: pip install yfinance")

        # FRED requires API key but is always "available" as a source
        cls._sources[DataSourceType.FRED] = bool(os.environ.get("FRED_API_KEY"))
        if not cls._sources[DataSourceType.FRED]:
            logger.info(
                "FRED API key not set. Get free key at: "
                "https://fred.stlouisfed.org/docs/api/api_key.html"
            )

        # SEC EDGAR is always available (free, no API key)
        cls._sources[DataSourceType.SEC_EDGAR] = True

        # Fear & Greed is always available (scraping)
        cls._sources[DataSourceType.FEAR_GREED] = True

        cls._initialized = True

    @classmethod
    def is_available(cls, source: DataSourceType) -> bool:
        """Check if a data source is available."""
        if not cls._initialized:
            cls.initialize()
        return cls._sources.get(source, False)

    @classmethod
    def get_status(cls) -> List[DataSourceStatus]:
        """Get status of all data sources."""
        if not cls._initialized:
            cls.initialize()

        statuses = []
        for source, available in cls._sources.items():
            reason = "" if available else "Not configured or missing dependency"
            statuses.append(DataSourceStatus(source, available, reason))
        return statuses


def list_available_sources() -> List[str]:
    """List all available data sources."""
    return [
        status.source.value
        for status in DataSourceRegistry.get_status()
        if status.available
    ]


def get_source_status() -> Dict[str, Dict[str, object]]:
    """Get detailed status of all data sources."""
    statuses = DataSourceRegistry.get_status()
    return {
        status.source.value: {
            "available": status.available,
            "reason": status.reason,
        }
        for status in statuses
    }


# Initialize on module load
DataSourceRegistry.initialize()
