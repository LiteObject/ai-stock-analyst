"""
Repository implementations for database access.

This module implements the Repository pattern for data access,
allowing easy swapping of database implementations.

Current implementation: SQLite via SQLAlchemy
Future: PostgreSQL, async SQLAlchemy, etc.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import pandas as pd
from sqlalchemy import and_, desc

from core.interfaces import PerformanceRepository, Repository, TradeRepository
from core.models import (
    BacktestResult,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Signal,
    SignalType,
    Trade,
)
from database.connection import DatabaseSession, get_db_session
from database.models import (
    BacktestResultModel,
    DailyPerformanceModel,
    OrderModel,
    PortfolioModel,
    PositionModel,
    SignalModel,
    TradeModel,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def trade_model_to_entity(model: TradeModel) -> Trade:
    """Convert TradeModel to Trade entity."""
    return Trade(
        id=UUID(model.id),
        order_id=UUID(model.order_id) if model.order_id else None,
        ticker=model.ticker,
        side=OrderSide(model.side),
        quantity=model.quantity,
        price=model.price,
        commission=model.commission,
        slippage=model.slippage,
        timestamp=model.timestamp,
        strategy_name=model.strategy_name,
        signals=model.signals or {},
    )


def trade_entity_to_model(entity: Trade) -> TradeModel:
    """Convert Trade entity to TradeModel."""
    return TradeModel(
        id=str(entity.id),
        order_id=str(entity.order_id) if entity.order_id else None,
        ticker=entity.ticker,
        side=entity.side.value,
        quantity=entity.quantity,
        price=entity.price,
        commission=entity.commission,
        slippage=entity.slippage,
        timestamp=entity.timestamp,
        strategy_name=entity.strategy_name,
        signals=entity.signals,
    )


def order_model_to_entity(model: OrderModel) -> Order:
    """Convert OrderModel to Order entity."""
    return Order(
        id=UUID(model.id),
        ticker=model.ticker,
        side=OrderSide(model.side),
        order_type=OrderType(model.order_type),
        quantity=model.quantity,
        limit_price=model.limit_price,
        stop_price=model.stop_price,
        status=OrderStatus(model.status),
        filled_quantity=model.filled_quantity,
        average_fill_price=model.average_fill_price,
        created_at=model.created_at,
        updated_at=model.updated_at,
        filled_at=model.filled_at,
        metadata=model.metadata_ or {},
    )


def order_entity_to_model(entity: Order) -> OrderModel:
    """Convert Order entity to OrderModel."""
    return OrderModel(
        id=str(entity.id),
        ticker=entity.ticker,
        side=entity.side.value,
        order_type=entity.order_type.value,
        quantity=entity.quantity,
        limit_price=entity.limit_price,
        stop_price=entity.stop_price,
        status=entity.status.value,
        filled_quantity=entity.filled_quantity,
        average_fill_price=entity.average_fill_price,
        created_at=entity.created_at,
        updated_at=entity.updated_at,
        filled_at=entity.filled_at,
        metadata_=entity.metadata,
    )


def position_model_to_entity(model: PositionModel) -> Position:
    """Convert PositionModel to Position entity."""
    return Position(
        ticker=model.ticker,
        quantity=model.quantity,
        average_cost=model.average_cost,
        current_price=model.current_price,
        opened_at=model.opened_at,
        last_updated=model.last_updated,
    )


def signal_model_to_entity(model: SignalModel) -> Signal:
    """Convert SignalModel to Signal entity."""
    return Signal(
        ticker=model.ticker,
        signal_type=SignalType(model.signal_type),
        confidence=model.confidence,
        source=model.source,
        timestamp=model.timestamp,
        reasoning=model.reasoning,
        metadata=model.metadata_ or {},
    )


# =============================================================================
# Trade Repository
# =============================================================================


class SQLiteTradeRepository(TradeRepository):
    """
    SQLite implementation of TradeRepository.

    This can be easily swapped with a PostgreSQL implementation
    by creating a new class that implements the same interface.
    """

    def __init__(self, session_factory=None):
        """
        Initialize repository.

        Args:
            session_factory: SQLAlchemy session factory (uses default if None)
        """
        self._session_factory = session_factory

    def _get_session(self):
        """Get a database session."""
        if self._session_factory:
            return self._session_factory()
        return DatabaseSession()

    async def get(self, id: UUID) -> Optional[Trade]:
        """Get trade by ID."""
        with get_db_session() as session:
            model = session.query(TradeModel).filter(TradeModel.id == str(id)).first()
            return trade_model_to_entity(model) if model else None

    async def get_all(self, skip: int = 0, limit: int = 100) -> List[Trade]:
        """Get all trades with pagination."""
        with get_db_session() as session:
            models = session.query(TradeModel).order_by(desc(TradeModel.timestamp)).offset(skip).limit(limit).all()
            return [trade_model_to_entity(m) for m in models]

    async def create(self, entity: Trade) -> Trade:
        """Create a new trade."""
        with get_db_session() as session:
            model = trade_entity_to_model(entity)
            session.add(model)
            session.flush()
            logger.debug(f"Created trade: {entity.id}")
            return entity

    async def update(self, entity: Trade) -> Trade:
        """Update an existing trade."""
        with get_db_session() as session:
            model = session.query(TradeModel).filter(TradeModel.id == str(entity.id)).first()
            if model:
                model.ticker = entity.ticker
                model.side = entity.side.value
                model.quantity = entity.quantity
                model.price = entity.price
                model.commission = entity.commission
                model.slippage = entity.slippage
                model.strategy_name = entity.strategy_name
                model.signals = entity.signals
                session.flush()
                logger.debug(f"Updated trade: {entity.id}")
            return entity

    async def delete(self, id: UUID) -> bool:
        """Delete a trade by ID."""
        with get_db_session() as session:
            model = session.query(TradeModel).filter(TradeModel.id == str(id)).first()
            if model:
                session.delete(model)
                logger.debug(f"Deleted trade: {id}")
                return True
            return False

    async def get_by_ticker(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Trade]:
        """Get trades for a specific ticker within date range."""
        with get_db_session() as session:
            query = session.query(TradeModel).filter(TradeModel.ticker == ticker.upper())

            if start_date:
                query = query.filter(TradeModel.timestamp >= start_date)
            if end_date:
                query = query.filter(TradeModel.timestamp <= end_date)

            models = query.order_by(desc(TradeModel.timestamp)).all()
            return [trade_model_to_entity(m) for m in models]

    async def get_by_strategy(self, strategy_name: str) -> List[Trade]:
        """Get trades for a specific strategy."""
        with get_db_session() as session:
            models = (
                session.query(TradeModel)
                .filter(TradeModel.strategy_name == strategy_name)
                .order_by(desc(TradeModel.timestamp))
                .all()
            )
            return [trade_model_to_entity(m) for m in models]

    async def get_trades_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get summary statistics for trades."""
        with get_db_session() as session:
            query = session.query(TradeModel)

            if start_date:
                query = query.filter(TradeModel.timestamp >= start_date)
            if end_date:
                query = query.filter(TradeModel.timestamp <= end_date)

            trades = query.all()

            if not trades:
                return {
                    "total_trades": 0,
                    "total_volume": 0,
                    "total_commission": 0,
                    "unique_tickers": 0,
                }

            return {
                "total_trades": len(trades),
                "total_volume": sum(t.quantity * t.price for t in trades),
                "total_commission": sum(t.commission for t in trades),
                "unique_tickers": len(set(t.ticker for t in trades)),
                "buy_trades": sum(1 for t in trades if t.side == "buy"),
                "sell_trades": sum(1 for t in trades if t.side == "sell"),
            }


# =============================================================================
# Order Repository
# =============================================================================


class SQLiteOrderRepository(Repository[Order]):
    """SQLite implementation of Order repository."""

    async def get(self, id: UUID) -> Optional[Order]:
        """Get order by ID."""
        with get_db_session() as session:
            model = session.query(OrderModel).filter(OrderModel.id == str(id)).first()
            return order_model_to_entity(model) if model else None

    async def get_all(self, skip: int = 0, limit: int = 100) -> List[Order]:
        """Get all orders with pagination."""
        with get_db_session() as session:
            models = session.query(OrderModel).order_by(desc(OrderModel.created_at)).offset(skip).limit(limit).all()
            return [order_model_to_entity(m) for m in models]

    async def create(self, entity: Order) -> Order:
        """Create a new order."""
        with get_db_session() as session:
            model = order_entity_to_model(entity)
            session.add(model)
            session.flush()
            logger.debug(f"Created order: {entity.id}")
            return entity

    async def update(self, entity: Order) -> Order:
        """Update an existing order."""
        with get_db_session() as session:
            model = session.query(OrderModel).filter(OrderModel.id == str(entity.id)).first()
            if model:
                model.status = entity.status.value
                model.filled_quantity = entity.filled_quantity
                model.average_fill_price = entity.average_fill_price
                model.updated_at = datetime.now()
                model.filled_at = entity.filled_at
                model.metadata_ = entity.metadata
                session.flush()
                logger.debug(f"Updated order: {entity.id}")
            return entity

    async def delete(self, id: UUID) -> bool:
        """Delete an order by ID."""
        with get_db_session() as session:
            model = session.query(OrderModel).filter(OrderModel.id == str(id)).first()
            if model:
                session.delete(model)
                logger.debug(f"Deleted order: {id}")
                return True
            return False

    async def get_active_orders(self, ticker: Optional[str] = None) -> List[Order]:
        """Get all active orders."""
        with get_db_session() as session:
            query = session.query(OrderModel).filter(
                OrderModel.status.in_(["pending", "submitted", "partially_filled"])
            )

            if ticker:
                query = query.filter(OrderModel.ticker == ticker.upper())

            models = query.order_by(OrderModel.created_at).all()
            return [order_model_to_entity(m) for m in models]


# =============================================================================
# Performance Repository
# =============================================================================


class SQLitePerformanceRepository(PerformanceRepository):
    """SQLite implementation of PerformanceRepository."""

    async def save_daily_performance(
        self,
        date: datetime,
        portfolio_value: float,
        daily_return: float,
        portfolio_id: str = None,
        **metrics,
    ) -> None:
        """Save daily performance snapshot."""
        with get_db_session() as session:
            # Get or create default portfolio
            if portfolio_id is None:
                portfolio = session.query(PortfolioModel).filter(PortfolioModel.is_active.is_(True)).first()
                if portfolio is None:
                    portfolio = PortfolioModel(name="Default", cash=0, is_active=True)
                    session.add(portfolio)
                    session.flush()
                portfolio_id = portfolio.id

            # Check if record exists for this date
            existing = (
                session.query(DailyPerformanceModel)
                .filter(
                    and_(
                        DailyPerformanceModel.portfolio_id == portfolio_id,
                        DailyPerformanceModel.date == date,
                    )
                )
                .first()
            )

            if existing:
                # Update existing
                existing.portfolio_value = portfolio_value
                existing.daily_return = daily_return
                existing.daily_return_pct = daily_return * 100
                for key, value in metrics.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
            else:
                # Create new
                model = DailyPerformanceModel(
                    portfolio_id=portfolio_id,
                    date=date,
                    portfolio_value=portfolio_value,
                    daily_return=daily_return,
                    daily_return_pct=daily_return * 100,
                    cumulative_return=metrics.get("cumulative_return", 0),
                    drawdown=metrics.get("drawdown", 0),
                    cash=metrics.get("cash", 0),
                    positions_value=metrics.get("positions_value", 0),
                    sharpe_ratio=metrics.get("sharpe_ratio"),
                    sortino_ratio=metrics.get("sortino_ratio"),
                    max_drawdown=metrics.get("max_drawdown"),
                )
                session.add(model)

            session.flush()
            logger.debug(f"Saved daily performance for {date}")

    async def get_performance_history(
        self,
        start_date: datetime,
        end_date: datetime,
        portfolio_id: str = None,
    ) -> pd.DataFrame:
        """Get performance history as DataFrame."""
        with get_db_session() as session:
            query = session.query(DailyPerformanceModel).filter(
                and_(
                    DailyPerformanceModel.date >= start_date,
                    DailyPerformanceModel.date <= end_date,
                )
            )

            if portfolio_id:
                query = query.filter(DailyPerformanceModel.portfolio_id == portfolio_id)

            models = query.order_by(DailyPerformanceModel.date).all()

            if not models:
                return pd.DataFrame()

            data = []
            for m in models:
                data.append(
                    {
                        "date": m.date,
                        "portfolio_value": m.portfolio_value,
                        "daily_return": m.daily_return,
                        "daily_return_pct": m.daily_return_pct,
                        "cumulative_return": m.cumulative_return,
                        "drawdown": m.drawdown,
                        "cash": m.cash,
                        "positions_value": m.positions_value,
                        "sharpe_ratio": m.sharpe_ratio,
                        "sortino_ratio": m.sortino_ratio,
                        "max_drawdown": m.max_drawdown,
                    }
                )

            return pd.DataFrame(data)


# =============================================================================
# Backtest Repository
# =============================================================================


class SQLiteBacktestRepository:
    """Repository for storing and retrieving backtest results."""

    async def save_result(self, result: BacktestResult) -> str:
        """Save a backtest result."""
        with get_db_session() as session:
            model = BacktestResultModel(
                tickers=result.config.tickers,
                start_date=result.config.start_date,
                end_date=result.config.end_date,
                initial_capital=result.config.initial_capital,
                config=result.config.model_dump(mode="json"),
                total_return=result.metrics.total_return,
                total_return_pct=result.metrics.total_return_pct,
                cagr=result.metrics.cagr,
                sharpe_ratio=result.metrics.sharpe_ratio,
                sortino_ratio=result.metrics.sortino_ratio,
                max_drawdown=result.metrics.max_drawdown,
                max_drawdown_pct=result.metrics.max_drawdown_pct,
                calmar_ratio=result.metrics.calmar_ratio,
                total_trades=result.metrics.total_trades,
                winning_trades=result.metrics.winning_trades,
                losing_trades=result.metrics.losing_trades,
                win_rate=result.metrics.win_rate,
                profit_factor=result.metrics.profit_factor,
                trades_json=[t.model_dump(mode="json") for t in result.trades],
                daily_performance_json=[d.model_dump(mode="json") for d in result.daily_performance],
                final_portfolio_json=result.final_portfolio.model_dump(mode="json"),
                execution_time_seconds=result.execution_time_seconds,
            )
            session.add(model)
            session.flush()
            logger.info(f"Saved backtest result: {model.id}")
            return model.id

    async def get_result(self, id: str) -> Optional[BacktestResultModel]:
        """Get a backtest result by ID."""
        with get_db_session() as session:
            return session.query(BacktestResultModel).filter(BacktestResultModel.id == id).first()

    async def get_all_results(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[BacktestResultModel]:
        """Get all backtest results with pagination."""
        with get_db_session() as session:
            return (
                session.query(BacktestResultModel)
                .order_by(desc(BacktestResultModel.created_at))
                .offset(skip)
                .limit(limit)
                .all()
            )

    async def get_best_results(
        self,
        metric: str = "sharpe_ratio",
        limit: int = 10,
    ) -> List[BacktestResultModel]:
        """Get best backtest results by a specific metric."""
        with get_db_session() as session:
            order_col = getattr(BacktestResultModel, metric, None)
            if order_col is None:
                order_col = BacktestResultModel.sharpe_ratio

            return (
                session.query(BacktestResultModel)
                .filter(order_col.isnot(None))
                .order_by(desc(order_col))
                .limit(limit)
                .all()
            )

    async def delete_result(self, id: str) -> bool:
        """Delete a backtest result."""
        with get_db_session() as session:
            model = session.query(BacktestResultModel).filter(BacktestResultModel.id == id).first()
            if model:
                session.delete(model)
                logger.debug(f"Deleted backtest result: {id}")
                return True
            return False
