"""
Base data provider and factory.

This module provides the factory pattern for creating data providers,
enabling easy swapping and configuration of different data sources.
"""

import logging
import os
from typing import Dict, Optional, Type

from core.interfaces import DataProvider

logger = logging.getLogger(__name__)


class DataProviderFactory:
    """
    Factory for creating data provider instances.

    Supports registration of custom providers and automatic selection
    based on configuration.

    Example:
        # Register a custom provider
        DataProviderFactory.register("custom", CustomDataProvider)

        # Get a provider instance
        provider = DataProviderFactory.create("massive", api_key="...")

        # Or use the default based on config
        provider = DataProviderFactory.get_default()
    """

    _providers: Dict[str, Type[DataProvider]] = {}
    _default_provider: Optional[str] = None

    @classmethod
    def register(cls, name: str, provider_class: Type[DataProvider]) -> None:
        """
        Register a data provider class.

        Args:
            name: Unique name for the provider
            provider_class: DataProvider implementation class
        """
        cls._providers[name.lower()] = provider_class
        logger.debug(f"Registered data provider: {name}")

    @classmethod
    def create(cls, name: str, **kwargs) -> DataProvider:
        """
        Create a data provider instance.

        Args:
            name: Name of the registered provider
            **kwargs: Provider-specific configuration

        Returns:
            DataProvider instance

        Raises:
            ValueError: If provider name is not registered
        """
        name_lower = name.lower()

        if name_lower not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(f"Unknown data provider: {name}. Available: {available}")

        provider_class = cls._providers[name_lower]
        return provider_class(**kwargs)

    @classmethod
    def get_default(cls, **kwargs) -> DataProvider:
        """
        Get the default data provider based on configuration.

        Checks environment variable DATA_PROVIDER or uses the registered default.
        Falls back to Yahoo Finance if no API key is configured.

        Args:
            **kwargs: Provider-specific configuration

        Returns:
            DataProvider instance
        """
        # Check for explicit provider setting
        provider_name = os.environ.get("DATA_PROVIDER", "").lower()

        if not provider_name:
            # Auto-detect based on available API keys
            if os.environ.get("MASSIVE_API_KEY") or os.environ.get("POLYGON_API_KEY"):
                provider_name = "massive"
            else:
                provider_name = "yahoo"
                logger.info("No API key found, using Yahoo Finance as fallback")

        return cls.create(provider_name, **kwargs)

    @classmethod
    def set_default(cls, name: str) -> None:
        """Set the default provider name."""
        if name.lower() not in cls._providers:
            raise ValueError(f"Unknown data provider: {name}")
        cls._default_provider = name.lower()

    @classmethod
    def list_providers(cls) -> list:
        """List all registered provider names."""
        return list(cls._providers.keys())


def get_data_provider(**kwargs) -> DataProvider:
    """
    Convenience function to get the default data provider.

    Args:
        **kwargs: Provider-specific configuration

    Returns:
        DataProvider instance
    """
    return DataProviderFactory.get_default(**kwargs)


# =============================================================================
# Register providers (imports done here to avoid circular imports)
# =============================================================================


def _register_providers():
    """Register all available data providers."""
    try:
        from data.providers.massive import MassiveDataProvider

        DataProviderFactory.register("massive", MassiveDataProvider)
        DataProviderFactory.register("polygon", MassiveDataProvider)  # Alias
    except ImportError as e:
        logger.warning(f"Could not register Massive provider: {e}")

    try:
        from data.providers.yahoo import YahooDataProvider

        DataProviderFactory.register("yahoo", YahooDataProvider)
        DataProviderFactory.register("yfinance", YahooDataProvider)  # Alias
    except ImportError as e:
        logger.warning(f"Could not register Yahoo provider: {e}")


# Register on module load
_register_providers()
