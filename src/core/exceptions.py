"""
Custom exceptions for the AI Stock Analyst.

This module defines a hierarchy of exceptions used throughout the application
to provide clear and actionable error information.
"""

from typing import Any, Dict, Optional


class StockAnalystError(Exception):
    """Base exception for all AI Stock Analyst errors."""

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code or "UNKNOWN_ERROR"
        self.details = details or {}

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


# =============================================================================
# Data Provider Exceptions
# =============================================================================


class DataProviderError(StockAnalystError):
    """Base exception for data provider errors."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        ticker: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, code="DATA_PROVIDER_ERROR", **kwargs)
        self.provider = provider
        self.ticker = ticker
        self.details.update({"provider": provider, "ticker": ticker})


class InsufficientDataError(DataProviderError):
    """Raised when there is not enough data for analysis."""

    def __init__(
        self,
        message: str,
        required_rows: Optional[int] = None,
        available_rows: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, code="INSUFFICIENT_DATA", **kwargs)
        self.required_rows = required_rows
        self.available_rows = available_rows
        self.details.update(
            {
                "required_rows": required_rows,
                "available_rows": available_rows,
            }
        )


class RateLimitError(DataProviderError):
    """Raised when API rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, code="RATE_LIMIT_EXCEEDED", **kwargs)
        self.retry_after = retry_after
        self.details.update({"retry_after_seconds": retry_after})


class APIConnectionError(DataProviderError):
    """Raised when unable to connect to data provider API."""

    def __init__(self, message: str, url: Optional[str] = None, **kwargs):
        super().__init__(message, code="API_CONNECTION_ERROR", **kwargs)
        self.url = url
        self.details.update({"url": url})


# =============================================================================
# Validation Exceptions
# =============================================================================


class ValidationError(StockAnalystError):
    """Raised when data validation fails."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(message, code="VALIDATION_ERROR", **kwargs)
        self.field = field
        self.value = value
        self.details.update({"field": field, "value": str(value)})


class ConfigurationError(ValidationError):
    """Raised when configuration is invalid."""

    def __init__(self, message: str, setting: Optional[str] = None, **kwargs):
        super().__init__(message, code="CONFIGURATION_ERROR", **kwargs)
        self.setting = setting
        self.details.update({"setting": setting})


# =============================================================================
# Risk Management Exceptions
# =============================================================================


class RiskLimitExceededError(StockAnalystError):
    """Raised when a risk limit is exceeded."""

    def __init__(
        self,
        message: str,
        limit_type: Optional[str] = None,
        limit_value: Optional[float] = None,
        actual_value: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(message, code="RISK_LIMIT_EXCEEDED", **kwargs)
        self.limit_type = limit_type
        self.limit_value = limit_value
        self.actual_value = actual_value
        self.details.update(
            {
                "limit_type": limit_type,
                "limit_value": limit_value,
                "actual_value": actual_value,
            }
        )


class PositionSizeError(RiskLimitExceededError):
    """Raised when position size exceeds limits."""

    def __init__(self, message: str, ticker: Optional[str] = None, **kwargs):
        super().__init__(message, code="POSITION_SIZE_ERROR", **kwargs)
        self.ticker = ticker
        self.details.update({"ticker": ticker})


class DrawdownLimitError(RiskLimitExceededError):
    """Raised when drawdown limit is exceeded."""

    def __init__(
        self,
        message: str,
        max_drawdown: Optional[float] = None,
        current_drawdown: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(message, code="DRAWDOWN_LIMIT_ERROR", **kwargs)
        self.max_drawdown = max_drawdown
        self.current_drawdown = current_drawdown
        self.details.update(
            {
                "max_drawdown": max_drawdown,
                "current_drawdown": current_drawdown,
            }
        )


# =============================================================================
# Order Execution Exceptions
# =============================================================================


class OrderExecutionError(StockAnalystError):
    """Base exception for order execution errors."""

    def __init__(
        self,
        message: str,
        order_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, code="ORDER_EXECUTION_ERROR", **kwargs)
        self.order_id = order_id
        self.details.update({"order_id": order_id})


class InsufficientFundsError(OrderExecutionError):
    """Raised when there are insufficient funds for an order."""

    def __init__(
        self,
        message: str,
        required: Optional[float] = None,
        available: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(message, code="INSUFFICIENT_FUNDS", **kwargs)
        self.required = required
        self.available = available
        self.details.update({"required": required, "available": available})


class InsufficientSharesError(OrderExecutionError):
    """Raised when there are insufficient shares to sell."""

    def __init__(
        self,
        message: str,
        requested: Optional[int] = None,
        available: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, code="INSUFFICIENT_SHARES", **kwargs)
        self.requested = requested
        self.available = available
        self.details.update({"requested": requested, "available": available})


class OrderRejectedError(OrderExecutionError):
    """Raised when an order is rejected."""

    def __init__(self, message: str, reason: Optional[str] = None, **kwargs):
        super().__init__(message, code="ORDER_REJECTED", **kwargs)
        self.reason = reason
        self.details.update({"reason": reason})


# =============================================================================
# Database Exceptions
# =============================================================================


class DatabaseError(StockAnalystError):
    """Base exception for database errors."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, code="DATABASE_ERROR", **kwargs)
        self.operation = operation
        self.details.update({"operation": operation})


class RecordNotFoundError(DatabaseError):
    """Raised when a requested record is not found."""

    def __init__(
        self,
        message: str,
        entity: Optional[str] = None,
        entity_id: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(message, code="RECORD_NOT_FOUND", **kwargs)
        self.entity = entity
        self.entity_id = entity_id
        self.details.update({"entity": entity, "entity_id": str(entity_id)})


class DuplicateRecordError(DatabaseError):
    """Raised when attempting to create a duplicate record."""

    def __init__(
        self,
        message: str,
        entity: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, code="DUPLICATE_RECORD", **kwargs)
        self.entity = entity
        self.details.update({"entity": entity})


# =============================================================================
# ML Model Exceptions
# =============================================================================


class MLModelError(StockAnalystError):
    """Base exception for ML model errors."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, code="ML_MODEL_ERROR", **kwargs)
        self.model_name = model_name
        self.details.update({"model_name": model_name})


class ModelNotTrainedError(MLModelError):
    """Raised when trying to predict with an untrained model."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, code="MODEL_NOT_TRAINED", **kwargs)


class FeatureEngineeringError(MLModelError):
    """Raised when feature engineering fails."""

    def __init__(
        self,
        message: str,
        missing_features: Optional[list] = None,
        **kwargs,
    ):
        super().__init__(message, code="FEATURE_ENGINEERING_ERROR", **kwargs)
        self.missing_features = missing_features
        self.details.update({"missing_features": missing_features})
