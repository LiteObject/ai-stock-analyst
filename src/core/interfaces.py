"""
Abstract interfaces for the AI Stock Analyst.

This module defines abstract base classes that establish contracts
for different components of the system. This enables:
- Dependency injection
- Easy testing with mocks
- Swappable implementations (e.g., different data providers, databases)
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar
from uuid import UUID

import pandas as pd

from core.models import (
    BacktestConfig,
    BacktestResult,
    Order,
    OrderSide,
    OrderType,
    PerformanceMetrics,
    Portfolio,
    Signal,
    Trade,
)

# =============================================================================
# Generic Type Variables
# =============================================================================

T = TypeVar("T")  # Generic type for repository entities


# =============================================================================
# Data Provider Interface
# =============================================================================


class DataProvider(ABC):
    """
    Abstract base class for market data providers.

    Implementations:
    - MassiveDataProvider (Polygon.io/Massive)
    - YahooDataProvider (Yahoo Finance fallback)
    - AlphaVantageDataProvider (future)

    Example:
        provider = MassiveDataProvider(api_key="...")
        data = await provider.get_historical_data("AAPL", start, end)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""
        pass

    @abstractmethod
    async def get_historical_data(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data
            timeframe: Data timeframe (1min, 5min, 1h, 1d, etc.)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume

        Raises:
            DataProviderError: If data fetch fails
            InsufficientDataError: If not enough data available
        """
        pass

    @abstractmethod
    async def get_current_price(self, ticker: str) -> float:
        """
        Get current price for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Current price

        Raises:
            DataProviderError: If price fetch fails
        """
        pass

    @abstractmethod
    async def get_quote(self, ticker: str) -> Dict[str, Any]:
        """
        Get current quote with bid/ask/last.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Quote data with bid, ask, last, volume, etc.
        """
        pass

    async def get_multiple_tickers(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple tickers.

        Default implementation calls get_historical_data for each ticker.
        Override for batch API support.
        """
        result = {}
        for ticker in tickers:
            result[ticker] = await self.get_historical_data(ticker, start_date, end_date, timeframe)
        return result

    async def health_check(self) -> bool:
        """Check if the data provider is accessible."""
        try:
            await self.get_current_price("AAPL")
            return True
        except Exception:
            return False


# =============================================================================
# Risk Calculator Interface
# =============================================================================


class RiskCalculator(ABC):
    """
    Abstract base class for risk calculations.

    Implementations:
    - VaRCalculator
    - PortfolioRiskCalculator
    """

    @abstractmethod
    def calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        horizon_days: int = 1,
    ) -> float:
        """
        Calculate Value at Risk.

        Args:
            returns: Historical returns series
            confidence: Confidence level (e.g., 0.95 for 95%)
            horizon_days: Time horizon in days

        Returns:
            VaR value (as positive number representing potential loss)
        """
        pass

    @abstractmethod
    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).

        Args:
            returns: Historical returns series
            confidence: Confidence level

        Returns:
            CVaR value
        """
        pass

    @abstractmethod
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown from equity curve.

        Args:
            equity_curve: Series of portfolio values over time

        Returns:
            Maximum drawdown as decimal (e.g., -0.25 for 25% drawdown)
        """
        pass


# =============================================================================
# Position Sizer Interface
# =============================================================================


class PositionSizer(ABC):
    """
    Abstract base class for position sizing algorithms.

    Implementations:
    - FixedFractionalSizer
    - KellyCriterionSizer
    - VolatilityTargetSizer
    """

    @abstractmethod
    def calculate_position_size(
        self,
        portfolio: Portfolio,
        ticker: str,
        current_price: float,
        signal: Optional[Signal] = None,
        **kwargs,
    ) -> int:
        """
        Calculate recommended position size.

        Args:
            portfolio: Current portfolio state
            ticker: Ticker to size position for
            current_price: Current price of the ticker
            signal: Optional trading signal for confidence-based sizing
            **kwargs: Additional parameters for specific algorithms

        Returns:
            Recommended number of shares to trade
        """
        pass

    @abstractmethod
    def calculate_risk_per_trade(
        self,
        portfolio: Portfolio,
        stop_loss_pct: float,
    ) -> float:
        """
        Calculate dollar risk per trade.

        Args:
            portfolio: Current portfolio state
            stop_loss_pct: Stop loss percentage

        Returns:
            Dollar amount to risk per trade
        """
        pass


# =============================================================================
# Backtest Engine Interface
# =============================================================================


class BacktestEngine(ABC):
    """
    Abstract base class for backtesting engines.

    Implementations:
    - SimpleBacktestEngine
    - WalkForwardBacktestEngine
    """

    @abstractmethod
    async def run(
        self,
        config: BacktestConfig,
        strategy: Any,  # Strategy callable or object
    ) -> BacktestResult:
        """
        Run a backtest.

        Args:
            config: Backtest configuration
            strategy: Trading strategy to test

        Returns:
            Backtest results with metrics and trades
        """
        pass

    @abstractmethod
    def calculate_metrics(
        self,
        trades: List[Trade],
        equity_curve: pd.Series,
        config: BacktestConfig,
    ) -> PerformanceMetrics:
        """
        Calculate performance metrics from trades and equity curve.

        Args:
            trades: List of executed trades
            equity_curve: Portfolio value over time
            config: Backtest configuration

        Returns:
            Calculated performance metrics
        """
        pass


# =============================================================================
# Order Executor Interface
# =============================================================================


class OrderExecutor(ABC):
    """
    Abstract base class for order execution.

    Implementations:
    - PaperTradingExecutor
    - AlpacaExecutor (future)
    - InteractiveBrokersExecutor (future)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the executor name."""
        pass

    @property
    @abstractmethod
    def is_paper_trading(self) -> bool:
        """Return True if this is paper trading."""
        pass

    @abstractmethod
    async def submit_order(self, order: Order) -> Order:
        """
        Submit an order for execution.

        Args:
            order: Order to submit

        Returns:
            Updated order with status

        Raises:
            OrderExecutionError: If order submission fails
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: UUID) -> Order:
        """
        Cancel an existing order.

        Args:
            order_id: ID of order to cancel

        Returns:
            Updated order with cancelled status
        """
        pass

    @abstractmethod
    async def get_order(self, order_id: UUID) -> Order:
        """
        Get current order status.

        Args:
            order_id: Order ID

        Returns:
            Order with current status
        """
        pass

    @abstractmethod
    async def get_portfolio(self) -> Portfolio:
        """
        Get current portfolio state.

        Returns:
            Current portfolio with positions and cash
        """
        pass

    async def create_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Order:
        """
        Create and submit a new order.

        Convenience method that creates and submits an order.
        """
        order = Order(
            ticker=ticker,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
        )
        return await self.submit_order(order)


# =============================================================================
# Repository Interface (Database Access)
# =============================================================================


class Repository(ABC, Generic[T]):
    """
    Abstract base class for data access (Repository pattern).

    This abstraction allows easy swapping of database implementations:
    - SQLiteRepository
    - PostgreSQLRepository
    - InMemoryRepository (for testing)

    Type parameter T represents the entity type (Trade, Order, etc.)
    """

    @abstractmethod
    async def get(self, id: UUID) -> Optional[T]:
        """
        Get entity by ID.

        Args:
            id: Entity ID

        Returns:
            Entity if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_all(self, skip: int = 0, limit: int = 100) -> List[T]:
        """
        Get all entities with pagination.

        Args:
            skip: Number of records to skip
            limit: Maximum records to return

        Returns:
            List of entities
        """
        pass

    @abstractmethod
    async def create(self, entity: T) -> T:
        """
        Create a new entity.

        Args:
            entity: Entity to create

        Returns:
            Created entity (may include generated fields)
        """
        pass

    @abstractmethod
    async def update(self, entity: T) -> T:
        """
        Update an existing entity.

        Args:
            entity: Entity with updated values

        Returns:
            Updated entity
        """
        pass

    @abstractmethod
    async def delete(self, id: UUID) -> bool:
        """
        Delete an entity by ID.

        Args:
            id: Entity ID

        Returns:
            True if deleted, False if not found
        """
        pass


class TradeRepository(Repository[Trade]):
    """Repository interface specifically for trades."""

    @abstractmethod
    async def get_by_ticker(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Trade]:
        """Get trades for a specific ticker within date range."""
        pass

    @abstractmethod
    async def get_by_strategy(self, strategy_name: str) -> List[Trade]:
        """Get trades for a specific strategy."""
        pass


class PerformanceRepository(ABC):
    """Repository interface for performance metrics."""

    @abstractmethod
    async def save_daily_performance(
        self,
        date: datetime,
        portfolio_value: float,
        daily_return: float,
        **metrics,
    ) -> None:
        """Save daily performance snapshot."""
        pass

    @abstractmethod
    async def get_performance_history(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Get performance history as DataFrame."""
        pass


# =============================================================================
# ML Model Interface
# =============================================================================


class MLPredictor(ABC):
    """
    Abstract base class for ML prediction models.

    Implementations (Basic):
    - RandomForestPredictor
    - XGBoostPredictor

    Future Implementations (Intermediate):
    - LSTMPredictor
    - GRUPredictor

    Future Implementations (Advanced):
    - TransformerPredictor
    - ReinforcementLearningAgent

    TODO: Implement intermediate and advanced models for better predictions.
    Consider using:
    - LSTM/GRU for sequential pattern learning
    - Attention mechanisms for long-range dependencies
    - Reinforcement learning for adaptive strategy optimization
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return model name."""
        pass

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """Check if model is trained."""
        pass

    @abstractmethod
    def train(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        validation_split: float = 0.2,
    ) -> Dict[str, float]:
        """
        Train the model.

        Args:
            features: Feature DataFrame
            target: Target Series
            validation_split: Fraction for validation

        Returns:
            Training metrics (accuracy, loss, etc.)
        """
        pass

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Make predictions.

        Args:
            features: Feature DataFrame

        Returns:
            Predictions as Series
        """
        pass

    @abstractmethod
    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Predict class probabilities.

        Args:
            features: Feature DataFrame

        Returns:
            DataFrame with probability for each class
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
        pass

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores,
            or None if model doesn't support feature importance.
        """
        return None
