"""Caching utilities for API responses.

This module provides a disk-based caching layer to reduce API calls,
improve performance during development, and reduce costs.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Default cache directory
CACHE_DIR = Path(__file__).parent.parent.parent / ".cache"
DEFAULT_TTL_HOURS = 24  # Cache TTL in hours


class DiskCache:
    """Simple disk-based cache for API responses."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        ttl_hours: int = DEFAULT_TTL_HOURS,
        enabled: bool = True,
    ):
        """
        Initialize the disk cache.

        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time-to-live for cache entries in hours
            enabled: Whether caching is enabled
        """
        self.cache_dir = cache_dir or CACHE_DIR
        self.ttl_hours = ttl_hours
        self.enabled = enabled

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cache initialized at {self.cache_dir}")

    def _get_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate a unique cache key based on function name and arguments."""
        # Create a deterministic string representation
        key_data = {
            "func": func_name,
            "args": args,
            "kwargs": {k: v for k, v in sorted(kwargs.items())},
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{cache_key}.json"

    def _is_expired(self, cache_path: Path, ttl_hours: int = None) -> bool:
        """Check if a cache entry has expired.

        Args:
            cache_path: Path to the cache file
            ttl_hours: Custom TTL in hours (uses default if None)
        """
        if not cache_path.exists():
            return True

        ttl = ttl_hours if ttl_hours is not None else self.ttl_hours

        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)
                cached_time = datetime.fromisoformat(cached.get("timestamp", ""))
                expiry_time = cached_time + timedelta(hours=ttl)
                return datetime.now() > expiry_time
        except (json.JSONDecodeError, ValueError, KeyError):
            return True

    def get(self, cache_key: str, ttl_hours: int = None) -> Optional[Any]:
        """
        Retrieve a value from the cache.

        Args:
            cache_key: The cache key
            ttl_hours: Custom TTL in hours (uses default if None)

        Returns:
            The cached value, or None if not found or expired
        """
        if not self.enabled:
            return None

        cache_path = self._get_cache_path(cache_key)

        if self._is_expired(cache_path, ttl_hours):
            return None

        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)
                logger.debug(f"Cache hit for key {cache_key[:16]}...")
                return cached.get("data")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read cache: {e}")
            return None

    def set(self, cache_key: str, value: Any) -> None:
        """
        Store a value in the cache.

        Args:
            cache_key: The cache key
            value: The value to cache
        """
        if not self.enabled:
            return

        cache_path = self._get_cache_path(cache_key)

        try:
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "data": value,
            }
            with open(cache_path, "w") as f:
                json.dump(cache_data, f)
            logger.debug(f"Cached data for key {cache_key[:16]}...")
        except (IOError, TypeError) as e:
            logger.warning(f"Failed to write cache: {e}")

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        if not self.cache_dir.exists():
            return 0

        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                count += 1
            except IOError:
                pass

        logger.info(f"Cleared {count} cache entries")
        return count

    def clear_expired(self) -> int:
        """
        Clear only expired cache entries.

        Returns:
            Number of entries cleared
        """
        if not self.cache_dir.exists():
            return 0

        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            if self._is_expired(cache_file):
                try:
                    cache_file.unlink()
                    count += 1
                except IOError:
                    pass

        logger.info(f"Cleared {count} expired cache entries")
        return count

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        if not self.cache_dir.exists():
            return {"total_entries": 0, "total_size_mb": 0, "expired": 0}

        total_entries = 0
        total_size = 0
        expired = 0

        for cache_file in self.cache_dir.glob("*.json"):
            total_entries += 1
            total_size += cache_file.stat().st_size
            if self._is_expired(cache_file):
                expired += 1

        return {
            "total_entries": total_entries,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "expired": expired,
            "valid": total_entries - expired,
        }


# Global cache instance
_cache: Optional[DiskCache] = None


def get_cache() -> DiskCache:
    """Get or create the global cache instance."""
    global _cache
    if _cache is None:
        # Import settings here to avoid circular imports
        try:
            from config import get_settings

            settings = get_settings()
            enabled = settings.cache_enabled
            ttl_hours = settings.cache_ttl_hours
        except ImportError:
            # Fallback to environment variables if config not available
            enabled = os.environ.get("CACHE_ENABLED", "true").lower() == "true"
            ttl_hours = int(os.environ.get("CACHE_TTL_HOURS", DEFAULT_TTL_HOURS))
        _cache = DiskCache(enabled=enabled, ttl_hours=ttl_hours)
    return _cache


def cached(func: Callable = None, *, ttl_hours: int = None) -> Callable:
    """
    Decorator to cache function results.

    Usage:
        @cached
        def get_prices(ticker: str, start_date: str, end_date: str):
            ...

        @cached(ttl_hours=6)
        def get_options_data(ticker: str):
            ...
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            cache_key = cache._get_cache_key(fn.__name__, args, kwargs)

            # Try to get from cache (with custom TTL if provided)
            cached_result = cache.get(cache_key, ttl_hours=ttl_hours)
            if cached_result is not None:
                return cached_result

            # Call the function and cache the result
            result = fn(*args, **kwargs)
            cache.set(cache_key, result)
            return result

        return wrapper

    # Handle both @cached and @cached(ttl_hours=X) syntax
    if func is not None:
        return decorator(func)
    return decorator
