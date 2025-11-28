"""
Core module for the AI Stock Analyst.

This module provides core abstractions, interfaces, and base classes
used throughout the application.
"""

from core.events import Event, EventBus, EventType
from core.exceptions import (
    DatabaseError,
    DataProviderError,
    InsufficientDataError,
    OrderExecutionError,
    RiskLimitExceededError,
    StockAnalystError,
    ValidationError,
)
from core.interfaces import (
    BacktestEngine,
    DataProvider,
    MLPredictor,
    OrderExecutor,
    PositionSizer,
    Repository,
    RiskCalculator,
)
from core.models import (
    OHLCV,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    PerformanceMetrics,
    Portfolio,
    Position,
    Signal,
    SignalType,
    Ticker,
    Trade,
)

__all__ = [
    # Interfaces
    "DataProvider",
    "RiskCalculator",
    "PositionSizer",
    "BacktestEngine",
    "OrderExecutor",
    "Repository",
    "MLPredictor",
    # Exceptions
    "StockAnalystError",
    "DataProviderError",
    "InsufficientDataError",
    "ValidationError",
    "RiskLimitExceededError",
    "OrderExecutionError",
    "DatabaseError",
    # Events
    "Event",
    "EventBus",
    "EventType",
    # Models
    "Ticker",
    "OHLCV",
    "Trade",
    "Position",
    "Portfolio",
    "Signal",
    "SignalType",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "Order",
    "PerformanceMetrics",
]
