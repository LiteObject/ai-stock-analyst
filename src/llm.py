"""
LLM Provider Abstraction Layer.

This module provides a unified interface for different LLM providers,
allowing easy switching between OpenAI, Anthropic, Ollama, and others.
"""

import logging
from enum import Enum
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from config import get_settings

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    GOOGLE = "google"


# Default models for each provider
DEFAULT_MODELS = {
    LLMProvider.OPENAI: "gpt-4o",
    LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    LLMProvider.OLLAMA: "llama3.2",
    LLMProvider.GOOGLE: "gemini-1.5-pro",
}


def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    **kwargs,
) -> BaseChatModel:
    """
    Get an LLM instance based on the specified provider.

    Args:
        provider: LLM provider name ('openai', 'anthropic', 'ollama', 'google')
                  Defaults to LLM_PROVIDER config setting or 'openai'
        model: Model name. Defaults to provider's default model
        temperature: Temperature for generation (0.0 - 2.0). Defaults to config setting
        **kwargs: Additional arguments passed to the LLM constructor

    Returns:
        Configured LLM instance

    Raises:
        ValueError: If provider is not supported
        ImportError: If required provider package is not installed
    """
    settings = get_settings()

    # Get provider from config if not specified
    provider_str = provider or settings.llm_provider

    try:
        llm_provider = LLMProvider(provider_str.lower())
    except ValueError:
        valid_providers = [p.value for p in LLMProvider]
        raise ValueError(f"Unsupported LLM provider: {provider_str}. " f"Valid providers: {valid_providers}")

    # Get model from config if not specified
    model_name = model or (
        settings.default_model
        if settings.default_model != "gpt-4o" or llm_provider == LLMProvider.OPENAI
        else DEFAULT_MODELS[llm_provider]
    )

    # Get temperature from config if not specified
    temp = temperature if temperature is not None else settings.model_temperature

    logger.info(f"Initializing LLM: provider={llm_provider.value}, model={model_name}")

    if llm_provider == LLMProvider.OPENAI:
        return _get_openai_llm(model_name, temp, settings, **kwargs)
    elif llm_provider == LLMProvider.ANTHROPIC:
        return _get_anthropic_llm(model_name, temp, settings, **kwargs)
    elif llm_provider == LLMProvider.OLLAMA:
        return _get_ollama_llm(model_name, temp, settings, **kwargs)
    elif llm_provider == LLMProvider.GOOGLE:
        return _get_google_llm(model_name, temp, settings, **kwargs)
    else:
        raise ValueError(f"Unsupported provider: {llm_provider}")


def _get_openai_llm(
    model: str,
    temperature: float,
    settings,
    **kwargs,
) -> BaseChatModel:
    """Create an OpenAI LLM instance."""
    try:
        from langchain_openai import ChatOpenAI  # type: ignore[import-not-found]
    except ImportError:
        raise ImportError(
            "langchain-openai package is required for OpenAI provider. " "Install it with: pip install langchain-openai"
        )

    api_key = settings.openai_api_key
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        **kwargs,
    )


def _get_anthropic_llm(
    model: str,
    temperature: float,
    settings,
    **kwargs,
) -> BaseChatModel:
    """Create an Anthropic LLM instance."""
    try:
        from langchain_anthropic import ChatAnthropic  # type: ignore[import-not-found]
    except ImportError:
        raise ImportError(
            "langchain-anthropic package is required for Anthropic provider. "
            "Install it with: pip install langchain-anthropic"
        )

    api_key = settings.anthropic_api_key
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    return ChatAnthropic(
        model=model,
        temperature=temperature,
        api_key=api_key,
        **kwargs,
    )


def _get_ollama_llm(
    model: str,
    temperature: float,
    settings,
    **kwargs,
) -> BaseChatModel:
    """Create an Ollama LLM instance."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise ImportError(
            "langchain-ollama package is required for Ollama provider. " "Install it with: pip install langchain-ollama"
        )

    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url=settings.ollama_base_url,
        **kwargs,
    )


def _get_google_llm(
    model: str,
    temperature: float,
    settings,
    **kwargs,
) -> BaseChatModel:
    """Create a Google Generative AI LLM instance."""
    try:
        from langchain_google_genai import (
            ChatGoogleGenerativeAI,  # type: ignore[import-not-found]
        )
    except ImportError:
        raise ImportError(
            "langchain-google-genai package is required for Google provider. "
            "Install it with: pip install langchain-google-genai"
        )

    api_key = settings.google_api_key
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is required")

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
        **kwargs,
    )


def list_available_providers() -> list[str]:
    """
    List available LLM providers based on installed packages and API keys.

    Returns:
        List of available provider names
    """
    settings = get_settings()
    available = []

    # Check OpenAI
    try:
        from langchain_openai import (  # type: ignore[import-not-found]  # noqa: F401
            ChatOpenAI,
        )

        if settings.openai_api_key:
            available.append("openai")
    except ImportError:
        pass

    # Check Anthropic
    try:
        from langchain_anthropic import (  # type: ignore[import-not-found]  # noqa: F401
            ChatAnthropic,
        )

        if settings.anthropic_api_key:
            available.append("anthropic")
    except ImportError:
        pass

    # Check Ollama (no API key required)
    try:
        from langchain_ollama import (  # type: ignore[import-not-found]  # noqa: F401
            ChatOllama,
        )

        available.append("ollama")
    except ImportError:
        pass

    # Check Google
    try:
        from langchain_google_genai import (  # type: ignore[import-not-found]  # noqa: F401
            ChatGoogleGenerativeAI,
        )

        if settings.google_api_key:
            available.append("google")
    except ImportError:
        pass

    return available
