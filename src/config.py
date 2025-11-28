"""
Centralized configuration management for the AI Stock Analyst.

This module provides a single source of truth for all configuration settings,
using environment variables with sensible defaults.
"""

import logging
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    financial_datasets_api_key: Optional[str] = Field(
        default=None,
        alias="FINANCIAL_DATASETS_API_KEY",
        description="API key for Financial Datasets API",
    )
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY", description="API key for OpenAI")
    anthropic_api_key: Optional[str] = Field(
        default=None, alias="ANTHROPIC_API_KEY", description="API key for Anthropic"
    )
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY", description="API key for Google AI")

    # LLM Configuration
    llm_provider: str = Field(
        default="openai",
        alias="LLM_PROVIDER",
        description="LLM provider to use (openai, anthropic, ollama, google)",
    )
    default_model: str = Field(default="gpt-4o", description="Default LLM model to use")
    model_temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Temperature for LLM responses")
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        alias="OLLAMA_BASE_URL",
        description="Base URL for Ollama server",
    )

    # Caching Configuration
    cache_enabled: bool = Field(
        default=True,
        alias="CACHE_ENABLED",
        description="Enable API response caching",
    )
    cache_ttl_hours: int = Field(
        default=24,
        alias="CACHE_TTL_HOURS",
        gt=0,
        description="Cache time-to-live in hours",
    )

    # Trading Configuration
    default_initial_capital: float = Field(default=100000.0, gt=0, description="Default initial capital for trading")
    max_position_percentage: float = Field(
        default=0.20,
        gt=0,
        le=1.0,
        description="Maximum position size as percentage of portfolio",
    )
    liquidity_limit_percentage: float = Field(
        default=0.10,
        gt=0,
        le=1.0,
        description="Maximum percentage of daily volume to trade",
    )

    # API Configuration
    api_timeout_seconds: int = Field(default=30, gt=0, description="Timeout for API requests in seconds")
    max_retries: int = Field(default=3, ge=0, description="Maximum number of retries for API requests")

    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate that log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v_upper

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def setup_logging(level: Optional[str] = None) -> logging.Logger:
    """
    Configure application logging.

    Args:
        level: Optional override for log level

    Returns:
        Configured root logger
    """
    settings = get_settings()
    log_level = level or settings.log_level

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    return logging.getLogger(__name__)


# Create a global settings instance
settings = get_settings()
