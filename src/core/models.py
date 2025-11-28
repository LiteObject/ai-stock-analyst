"""
Core data models for the AI Stock Analyst.

This module defines Pydantic models used throughout the application
for data validation and serialization.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

# =============================================================================
# Enumerations
# =============================================================================


class SignalType(str, Enum):
    """Types of trading signals."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class OrderSide(str, Enum):
    """Order side (buy or sell)."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Types of orders."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(str, Enum):
    """Order status."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeFrame(str, Enum):
    """Data time frames."""

    MINUTE_1 = "1min"
    MINUTE_5 = "5min"
    MINUTE_15 = "15min"
    MINUTE_30 = "30min"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1mo"


# =============================================================================
# Base Models
# =============================================================================


class Ticker(BaseModel):
    """Stock ticker information."""

    symbol: str = Field(..., min_length=1, max_length=10, description="Ticker symbol")
    name: Optional[str] = Field(None, description="Company name")
    exchange: Optional[str] = Field(None, description="Exchange name")
    sector: Optional[str] = Field(None, description="Sector")
    industry: Optional[str] = Field(None, description="Industry")

    @field_validator("symbol")
    @classmethod
    def uppercase_symbol(cls, v: str) -> str:
        """Ensure ticker symbol is uppercase."""
        return v.upper().strip()


class OHLCV(BaseModel):
    """OHLCV (Open, High, Low, Close, Volume) price bar."""

    timestamp: datetime = Field(..., description="Bar timestamp")
    open: float = Field(..., gt=0, description="Open price")
    high: float = Field(..., gt=0, description="High price")
    low: float = Field(..., gt=0, description="Low price")
    close: float = Field(..., gt=0, description="Close price")
    volume: int = Field(..., ge=0, description="Trading volume")
    vwap: Optional[float] = Field(None, gt=0, description="Volume-weighted average price")

    @model_validator(mode="after")
    def validate_prices(self) -> "OHLCV":
        """Validate OHLCV price relationships."""
        if self.high < self.low:
            raise ValueError("High price must be >= low price")
        if self.open > self.high or self.open < self.low:
            raise ValueError("Open price must be between low and high")
        if self.close > self.high or self.close < self.low:
            raise ValueError("Close price must be between low and high")
        return self

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# =============================================================================
# Trading Models
# =============================================================================


class Signal(BaseModel):
    """Trading signal from an analysis agent."""

    ticker: str = Field(..., description="Ticker symbol")
    signal_type: SignalType = Field(..., description="Type of signal")
    confidence: float = Field(..., ge=0, le=1, description="Confidence level (0-1)")
    source: str = Field(..., description="Signal source (agent name)")
    timestamp: datetime = Field(default_factory=datetime.now, description="Signal timestamp")
    reasoning: Optional[str] = Field(None, description="Reasoning for the signal")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, v: str) -> str:
        return v.upper().strip()


class Position(BaseModel):
    """A position in a security."""

    ticker: str = Field(..., description="Ticker symbol")
    quantity: int = Field(..., description="Number of shares (negative for short)")
    average_cost: float = Field(..., gt=0, description="Average cost per share")
    current_price: Optional[float] = Field(None, gt=0, description="Current market price")
    opened_at: datetime = Field(default_factory=datetime.now, description="Position open timestamp")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")

    @property
    def market_value(self) -> float:
        """Calculate current market value."""
        if self.current_price is None:
            return self.quantity * self.average_cost
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        """Calculate total cost basis."""
        return abs(self.quantity) * self.average_cost

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        if self.current_price is None:
            return 0.0
        return self.market_value - (self.quantity * self.average_cost)

    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L percentage."""
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / self.cost_basis


class Portfolio(BaseModel):
    """Portfolio state."""

    id: UUID = Field(default_factory=uuid4, description="Portfolio ID")
    name: str = Field(default="Default", description="Portfolio name")
    cash: float = Field(..., ge=0, description="Cash balance")
    positions: Dict[str, Position] = Field(default_factory=dict, description="Open positions")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")

    @property
    def total_value(self) -> float:
        """Calculate total portfolio value."""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value

    @property
    def total_cost_basis(self) -> float:
        """Calculate total cost basis of all positions."""
        return sum(pos.cost_basis for pos in self.positions.values())

    def get_position(self, ticker: str) -> Optional[Position]:
        """Get position for a ticker."""
        return self.positions.get(ticker.upper())

    def position_weight(self, ticker: str) -> float:
        """Get position weight as fraction of total portfolio."""
        position = self.get_position(ticker)
        if position is None or self.total_value == 0:
            return 0.0
        return position.market_value / self.total_value


class Order(BaseModel):
    """Trading order."""

    id: UUID = Field(default_factory=uuid4, description="Order ID")
    ticker: str = Field(..., description="Ticker symbol")
    side: OrderSide = Field(..., description="Buy or sell")
    order_type: OrderType = Field(default=OrderType.MARKET, description="Order type")
    quantity: int = Field(..., gt=0, description="Order quantity")
    limit_price: Optional[float] = Field(None, gt=0, description="Limit price")
    stop_price: Optional[float] = Field(None, gt=0, description="Stop price")
    status: OrderStatus = Field(default=OrderStatus.PENDING, description="Order status")
    filled_quantity: int = Field(default=0, ge=0, description="Filled quantity")
    average_fill_price: Optional[float] = Field(None, description="Average fill price")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    filled_at: Optional[datetime] = Field(None, description="Fill timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, v: str) -> str:
        return v.upper().strip()

    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.status == OrderStatus.FILLED

    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in (
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED,
        )

    @property
    def fill_percentage(self) -> float:
        """Calculate fill percentage."""
        if self.quantity == 0:
            return 0.0
        return self.filled_quantity / self.quantity


class Trade(BaseModel):
    """Executed trade record."""

    id: UUID = Field(default_factory=uuid4, description="Trade ID")
    order_id: Optional[UUID] = Field(None, description="Associated order ID")
    ticker: str = Field(..., description="Ticker symbol")
    side: OrderSide = Field(..., description="Buy or sell")
    quantity: int = Field(..., gt=0, description="Trade quantity")
    price: float = Field(..., gt=0, description="Execution price")
    commission: float = Field(default=0.0, ge=0, description="Commission paid")
    slippage: float = Field(default=0.0, description="Slippage amount")
    timestamp: datetime = Field(default_factory=datetime.now, description="Execution timestamp")
    strategy_name: Optional[str] = Field(None, description="Strategy that generated the trade")
    signals: Dict[str, Any] = Field(default_factory=dict, description="Signals that led to trade")

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, v: str) -> str:
        return v.upper().strip()

    @property
    def total_value(self) -> float:
        """Calculate total trade value including commission."""
        return (self.quantity * self.price) + self.commission


# =============================================================================
# Performance Models
# =============================================================================


class PerformanceMetrics(BaseModel):
    """Performance metrics for backtesting and live trading."""

    # Basic metrics
    total_return: float = Field(..., description="Total return (decimal)")
    total_return_pct: float = Field(..., description="Total return percentage")
    cagr: Optional[float] = Field(None, description="Compound annual growth rate")

    # Risk metrics
    sharpe_ratio: Optional[float] = Field(None, description="Sharpe ratio (annualized)")
    sortino_ratio: Optional[float] = Field(None, description="Sortino ratio (annualized)")
    calmar_ratio: Optional[float] = Field(None, description="Calmar ratio")

    # Drawdown metrics
    max_drawdown: float = Field(..., description="Maximum drawdown (decimal)")
    max_drawdown_pct: float = Field(..., description="Maximum drawdown percentage")
    avg_drawdown: Optional[float] = Field(None, description="Average drawdown")
    max_drawdown_duration_days: Optional[int] = Field(None, description="Max drawdown duration in days")

    # Trade metrics
    total_trades: int = Field(default=0, ge=0, description="Total number of trades")
    winning_trades: int = Field(default=0, ge=0, description="Number of winning trades")
    losing_trades: int = Field(default=0, ge=0, description="Number of losing trades")
    win_rate: Optional[float] = Field(None, ge=0, le=1, description="Win rate (0-1)")

    # P&L metrics
    avg_win: Optional[float] = Field(None, description="Average winning trade")
    avg_loss: Optional[float] = Field(None, description="Average losing trade")
    profit_factor: Optional[float] = Field(None, ge=0, description="Profit factor")
    expectancy: Optional[float] = Field(None, description="Trade expectancy")

    # Risk metrics
    volatility: Optional[float] = Field(None, ge=0, description="Annualized volatility")
    var_95: Optional[float] = Field(None, description="Value at Risk (95%)")
    cvar_95: Optional[float] = Field(None, description="Conditional VaR (95%)")

    # Period info
    start_date: Optional[datetime] = Field(None, description="Analysis start date")
    end_date: Optional[datetime] = Field(None, description="Analysis end date")
    trading_days: Optional[int] = Field(None, ge=0, description="Number of trading days")

    # Benchmark comparison
    benchmark_return: Optional[float] = Field(None, description="Benchmark return")
    alpha: Optional[float] = Field(None, description="Alpha vs benchmark")
    beta: Optional[float] = Field(None, description="Beta vs benchmark")

    @property
    def risk_adjusted_return(self) -> Optional[float]:
        """Calculate risk-adjusted return."""
        if self.volatility is None or self.volatility == 0:
            return None
        return self.total_return / self.volatility


class DailyPerformance(BaseModel):
    """Daily performance snapshot."""

    date: datetime = Field(..., description="Date")
    portfolio_value: float = Field(..., gt=0, description="Portfolio value")
    daily_return: float = Field(..., description="Daily return (decimal)")
    daily_return_pct: float = Field(..., description="Daily return percentage")
    cumulative_return: float = Field(..., description="Cumulative return (decimal)")
    drawdown: float = Field(..., le=0, description="Current drawdown (negative)")
    cash: float = Field(..., ge=0, description="Cash balance")
    positions_value: float = Field(..., ge=0, description="Positions market value")


# =============================================================================
# Backtesting Models
# =============================================================================


class BacktestConfig(BaseModel):
    """Configuration for backtesting."""

    tickers: List[str] = Field(..., min_length=1, description="Tickers to backtest")
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    initial_capital: float = Field(default=100000.0, gt=0, description="Initial capital")
    commission_rate: float = Field(default=0.001, ge=0, description="Commission rate per trade")
    slippage_rate: float = Field(default=0.0005, ge=0, description="Slippage rate per trade")

    # Walk-forward settings
    use_walk_forward: bool = Field(default=False, description="Use walk-forward analysis")
    training_window_days: int = Field(default=252, gt=0, description="Training window in days")
    test_window_days: int = Field(default=21, gt=0, description="Test window in days")

    # Risk settings
    max_position_size_pct: float = Field(default=0.20, gt=0, le=1, description="Max position size %")
    stop_loss_pct: Optional[float] = Field(None, gt=0, le=1, description="Stop loss percentage")
    take_profit_pct: Optional[float] = Field(None, gt=0, description="Take profit percentage")

    @field_validator("tickers")
    @classmethod
    def uppercase_tickers(cls, v: List[str]) -> List[str]:
        return [t.upper().strip() for t in v]

    @model_validator(mode="after")
    def validate_dates(self) -> "BacktestConfig":
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        return self


class BacktestResult(BaseModel):
    """Results from a backtest run."""

    config: BacktestConfig = Field(..., description="Backtest configuration")
    metrics: PerformanceMetrics = Field(..., description="Performance metrics")
    trades: List[Trade] = Field(default_factory=list, description="All trades")
    daily_performance: List[DailyPerformance] = Field(default_factory=list, description="Daily snapshots")
    final_portfolio: Portfolio = Field(..., description="Final portfolio state")
    execution_time_seconds: float = Field(..., ge=0, description="Execution time")
    created_at: datetime = Field(default_factory=datetime.now, description="Result creation time")
