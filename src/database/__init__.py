"""
Database module for the AI Stock Analyst.

This module provides database models, repositories, and utilities
using SQLAlchemy with a repository pattern for easy database swapping.
"""

from database.connection import (
    DatabaseSession,
    get_database_url,
    get_engine,
    get_session_factory,
    init_database,
)
from database.models import (
    BacktestResultModel,
    Base,
    DailyPerformanceModel,
    OrderModel,
    PortfolioModel,
    PositionModel,
    SignalModel,
    TradeModel,
)
from database.repositories import (
    SQLiteBacktestRepository,
    SQLiteOrderRepository,
    SQLitePerformanceRepository,
    SQLiteTradeRepository,
)

__all__ = [
    # Connection
    "get_database_url",
    "get_engine",
    "get_session_factory",
    "init_database",
    "DatabaseSession",
    # Models
    "Base",
    "TradeModel",
    "OrderModel",
    "PositionModel",
    "PortfolioModel",
    "DailyPerformanceModel",
    "BacktestResultModel",
    "SignalModel",
    # Repositories
    "SQLiteTradeRepository",
    "SQLiteOrderRepository",
    "SQLitePerformanceRepository",
    "SQLiteBacktestRepository",
]
