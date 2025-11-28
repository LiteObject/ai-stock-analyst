"""
SQLAlchemy database models.

These models define the database schema for persisting trades, orders,
portfolio state, and performance metrics.

The models are designed to work with both SQLite and PostgreSQL.
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


# =============================================================================
# Helper Functions
# =============================================================================


def generate_uuid() -> str:
    """Generate a UUID string for primary keys."""
    return str(uuid4())


# =============================================================================
# Trade & Order Models
# =============================================================================


class TradeModel(Base):
    """Database model for executed trades."""

    __tablename__ = "trades"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    order_id = Column(String(36), ForeignKey("orders.id"), nullable=True)
    ticker = Column(String(10), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # 'buy' or 'sell'
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    commission = Column(Float, default=0.0)
    slippage = Column(Float, default=0.0)
    timestamp = Column(DateTime, nullable=False, default=datetime.now, index=True)
    strategy_name = Column(String(100), nullable=True, index=True)
    signals = Column(JSON, default=dict)
    metadata_ = Column("metadata", JSON, default=dict)

    # Relationships
    order = relationship("OrderModel", back_populates="trades")

    # Indexes for common queries
    __table_args__ = (
        Index("ix_trades_ticker_timestamp", "ticker", "timestamp"),
        Index("ix_trades_strategy_timestamp", "strategy_name", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<Trade {self.id[:8]}: {self.side} {self.quantity} {self.ticker} @ {self.price}>"


class OrderModel(Base):
    """Database model for orders."""

    __tablename__ = "orders"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    ticker = Column(String(10), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # 'buy' or 'sell'
    order_type = Column(String(20), nullable=False, default="market")
    quantity = Column(Integer, nullable=False)
    limit_price = Column(Float, nullable=True)
    stop_price = Column(Float, nullable=True)
    status = Column(String(20), nullable=False, default="pending", index=True)
    filled_quantity = Column(Integer, default=0)
    average_fill_price = Column(Float, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.now, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)
    filled_at = Column(DateTime, nullable=True)
    metadata_ = Column("metadata", JSON, default=dict)

    # Relationships
    trades = relationship("TradeModel", back_populates="order")

    # Indexes
    __table_args__ = (
        Index("ix_orders_ticker_status", "ticker", "status"),
        Index("ix_orders_created_at_status", "created_at", "status"),
    )

    def __repr__(self) -> str:
        return f"<Order {self.id[:8]}: {self.side} {self.quantity} {self.ticker} ({self.status})>"


# =============================================================================
# Portfolio Models
# =============================================================================


class PositionModel(Base):
    """Database model for positions."""

    __tablename__ = "positions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    portfolio_id = Column(String(36), ForeignKey("portfolios.id"), nullable=False)
    ticker = Column(String(10), nullable=False, index=True)
    quantity = Column(Integer, nullable=False)
    average_cost = Column(Float, nullable=False)
    current_price = Column(Float, nullable=True)
    opened_at = Column(DateTime, nullable=False, default=datetime.now)
    last_updated = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)

    # Relationships
    portfolio = relationship("PortfolioModel", back_populates="positions")

    # Indexes
    __table_args__ = (Index("ix_positions_portfolio_ticker", "portfolio_id", "ticker", unique=True),)

    def __repr__(self) -> str:
        return f"<Position {self.ticker}: {self.quantity} shares @ {self.average_cost}>"


class PortfolioModel(Base):
    """Database model for portfolios."""

    __tablename__ = "portfolios"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(100), nullable=False, default="Default")
    cash = Column(Float, nullable=False, default=0.0)
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)

    # Relationships
    positions = relationship("PositionModel", back_populates="portfolio", cascade="all, delete-orphan")
    daily_performance = relationship(
        "DailyPerformanceModel",
        back_populates="portfolio",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Portfolio {self.name}: ${self.cash:.2f} cash>"


# =============================================================================
# Performance Models
# =============================================================================


class DailyPerformanceModel(Base):
    """Database model for daily performance snapshots."""

    __tablename__ = "daily_performance"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    portfolio_id = Column(String(36), ForeignKey("portfolios.id"), nullable=False)
    date = Column(DateTime, nullable=False, index=True)
    portfolio_value = Column(Float, nullable=False)
    daily_return = Column(Float, nullable=False)
    daily_return_pct = Column(Float, nullable=False)
    cumulative_return = Column(Float, nullable=False)
    drawdown = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    positions_value = Column(Float, nullable=False)

    # Additional metrics
    sharpe_ratio = Column(Float, nullable=True)
    sortino_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)

    # Relationships
    portfolio = relationship("PortfolioModel", back_populates="daily_performance")

    # Indexes
    __table_args__ = (Index("ix_daily_perf_portfolio_date", "portfolio_id", "date", unique=True),)

    def __repr__(self) -> str:
        return f"<DailyPerformance {self.date.date()}: ${self.portfolio_value:.2f}>"


# =============================================================================
# Backtest Models
# =============================================================================


class BacktestResultModel(Base):
    """Database model for backtest results."""

    __tablename__ = "backtest_results"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(200), nullable=True)

    # Configuration
    tickers = Column(JSON, nullable=False)  # List of tickers
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float, nullable=False)
    config = Column(JSON, nullable=False)  # Full config as JSON

    # Results
    total_return = Column(Float, nullable=False)
    total_return_pct = Column(Float, nullable=False)
    cagr = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    sortino_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=False)
    max_drawdown_pct = Column(Float, nullable=False)
    calmar_ratio = Column(Float, nullable=True)

    # Trade statistics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)

    # Detailed results (stored as JSON for flexibility)
    trades_json = Column(JSON, default=list)
    daily_performance_json = Column(JSON, default=list)
    final_portfolio_json = Column(JSON, default=dict)

    # Metadata
    execution_time_seconds = Column(Float, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.now, index=True)
    notes = Column(Text, nullable=True)

    # Indexes
    __table_args__ = (Index("ix_backtest_dates", "start_date", "end_date"),)

    def __repr__(self) -> str:
        return f"<BacktestResult {self.id[:8]}: {self.total_return_pct:.2f}% return>"


# =============================================================================
# Signal Models
# =============================================================================


class SignalModel(Base):
    """Database model for trading signals."""

    __tablename__ = "signals"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    ticker = Column(String(10), nullable=False, index=True)
    signal_type = Column(String(20), nullable=False)  # strong_buy, buy, hold, sell, strong_sell
    confidence = Column(Float, nullable=False)
    source = Column(String(100), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.now, index=True)
    reasoning = Column(Text, nullable=True)
    metadata_ = Column("metadata", JSON, default=dict)

    # Whether the signal was acted upon
    was_executed = Column(Boolean, default=False)
    execution_order_id = Column(String(36), nullable=True)

    # Indexes
    __table_args__ = (
        Index("ix_signals_ticker_timestamp", "ticker", "timestamp"),
        Index("ix_signals_source_timestamp", "source", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<Signal {self.ticker}: {self.signal_type} ({self.confidence:.2f}) from {self.source}>"


# =============================================================================
# Market Data Cache (Optional)
# =============================================================================


class MarketDataCacheModel(Base):
    """
    Database model for caching market data.

    This is optional and can be used to reduce API calls.
    For production, consider using Redis instead.
    """

    __tablename__ = "market_data_cache"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    ticker = Column(String(10), nullable=False)
    timeframe = Column(String(10), nullable=False)  # 1d, 1h, etc.
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    vwap = Column(Float, nullable=True)
    cached_at = Column(DateTime, nullable=False, default=datetime.now)

    # Unique constraint on ticker + timeframe + timestamp
    __table_args__ = (
        Index("ix_market_data_unique", "ticker", "timeframe", "timestamp", unique=True),
        Index("ix_market_data_ticker_timeframe", "ticker", "timeframe"),
    )

    def __repr__(self) -> str:
        return f"<MarketData {self.ticker} {self.timestamp}: O={self.open} C={self.close}>"
