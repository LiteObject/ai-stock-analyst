"""
Event system for the AI Stock Analyst.

This module implements a simple event bus for decoupled communication
between components. Useful for:
- Logging trade executions
- Triggering alerts (future)
- Updating UI components
- Audit trail
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events in the system."""

    # Data events
    PRICE_UPDATE = auto()
    DATA_RECEIVED = auto()
    DATA_ERROR = auto()

    # Trading events
    SIGNAL_GENERATED = auto()
    ORDER_CREATED = auto()
    ORDER_SUBMITTED = auto()
    ORDER_FILLED = auto()
    ORDER_CANCELLED = auto()
    ORDER_REJECTED = auto()
    TRADE_EXECUTED = auto()

    # Portfolio events
    POSITION_OPENED = auto()
    POSITION_CLOSED = auto()
    POSITION_UPDATED = auto()
    PORTFOLIO_UPDATED = auto()

    # Risk events
    RISK_LIMIT_WARNING = auto()
    RISK_LIMIT_EXCEEDED = auto()
    DRAWDOWN_WARNING = auto()
    STOP_LOSS_TRIGGERED = auto()
    TAKE_PROFIT_TRIGGERED = auto()

    # System events
    SYSTEM_STARTED = auto()
    SYSTEM_STOPPED = auto()
    ERROR_OCCURRED = auto()

    # Backtest events
    BACKTEST_STARTED = auto()
    BACKTEST_COMPLETED = auto()
    BACKTEST_DAY_COMPLETED = auto()

    # ML events
    MODEL_TRAINED = auto()
    MODEL_PREDICTION = auto()
    MODEL_ERROR = auto()


@dataclass
class Event:
    """
    Base event class.

    All events in the system are instances of this class with
    different event_type values.
    """

    event_type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: UUID = field(default_factory=uuid4)
    source: Optional[str] = None

    def __str__(self) -> str:
        return f"Event({self.event_type.name}, source={self.source}, id={self.event_id})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_id": str(self.event_id),
            "event_type": self.event_type.name,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "data": self.data,
        }


# Type alias for event handlers
EventHandler = Callable[[Event], None]
AsyncEventHandler = Callable[[Event], Any]  # Coroutine


class EventBus:
    """
    Simple event bus for pub/sub messaging.

    Example:
        bus = EventBus()

        def on_trade(event: Event):
            print(f"Trade executed: {event.data}")

        bus.subscribe(EventType.TRADE_EXECUTED, on_trade)

        bus.publish(Event(
            event_type=EventType.TRADE_EXECUTED,
            data={"ticker": "AAPL", "quantity": 100}
        ))
    """

    def __init__(self):
        self._handlers: Dict[EventType, List[EventHandler]] = {}
        self._async_handlers: Dict[EventType, List[AsyncEventHandler]] = {}
        self._all_event_handlers: List[EventHandler] = []
        self._history: List[Event] = []
        self._max_history: int = 1000
        self._enabled: bool = True

    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
    ) -> None:
        """
        Subscribe to events of a specific type.

        Args:
            event_type: Type of event to subscribe to
            handler: Callback function to handle the event
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug(f"Handler subscribed to {event_type.name}")

    def subscribe_async(
        self,
        event_type: EventType,
        handler: AsyncEventHandler,
    ) -> None:
        """
        Subscribe an async handler to events of a specific type.

        Args:
            event_type: Type of event to subscribe to
            handler: Async callback function to handle the event
        """
        if event_type not in self._async_handlers:
            self._async_handlers[event_type] = []
        self._async_handlers[event_type].append(handler)
        logger.debug(f"Async handler subscribed to {event_type.name}")

    def subscribe_all(self, handler: EventHandler) -> None:
        """
        Subscribe to all events.

        Args:
            handler: Callback function to handle all events
        """
        self._all_event_handlers.append(handler)
        logger.debug("Handler subscribed to all events")

    def unsubscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
    ) -> None:
        """
        Unsubscribe from events of a specific type.

        Args:
            event_type: Type of event to unsubscribe from
            handler: Handler to remove
        """
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
                logger.debug(f"Handler unsubscribed from {event_type.name}")
            except ValueError:
                pass

    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event: Event to publish
        """
        if not self._enabled:
            return

        # Add to history
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        # Notify type-specific handlers
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler for {event.event_type}: {e}")

        # Notify all-event handlers
        for handler in self._all_event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in all-event handler: {e}")

    async def publish_async(self, event: Event) -> None:
        """
        Publish an event and await async handlers.

        Args:
            event: Event to publish
        """
        if not self._enabled:
            return

        # First publish to sync handlers
        self.publish(event)

        # Then await async handlers
        handlers = self._async_handlers.get(event.event_type, [])
        if handlers:
            await asyncio.gather(
                *[handler(event) for handler in handlers],
                return_exceptions=True,
            )

    def get_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100,
    ) -> List[Event]:
        """
        Get event history.

        Args:
            event_type: Filter by event type (None for all)
            limit: Maximum number of events to return

        Returns:
            List of events (most recent first)
        """
        events = self._history[::-1]  # Reverse for most recent first
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[:limit]

    def clear_history(self) -> None:
        """Clear event history."""
        self._history = []

    def enable(self) -> None:
        """Enable event publishing."""
        self._enabled = True

    def disable(self) -> None:
        """Disable event publishing."""
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        """Check if event bus is enabled."""
        return self._enabled


# Global event bus instance
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def reset_event_bus() -> None:
    """Reset the global event bus (useful for testing)."""
    global _event_bus
    _event_bus = None


# =============================================================================
# Event Factory Functions
# =============================================================================


def create_trade_event(
    ticker: str,
    side: str,
    quantity: int,
    price: float,
    **kwargs,
) -> Event:
    """Create a trade executed event."""
    return Event(
        event_type=EventType.TRADE_EXECUTED,
        data={
            "ticker": ticker,
            "side": side,
            "quantity": quantity,
            "price": price,
            **kwargs,
        },
        source="trading_engine",
    )


def create_signal_event(
    ticker: str,
    signal_type: str,
    confidence: float,
    source: str,
    **kwargs,
) -> Event:
    """Create a signal generated event."""
    return Event(
        event_type=EventType.SIGNAL_GENERATED,
        data={
            "ticker": ticker,
            "signal_type": signal_type,
            "confidence": confidence,
            **kwargs,
        },
        source=source,
    )


def create_risk_warning_event(
    warning_type: str,
    current_value: float,
    limit_value: float,
    **kwargs,
) -> Event:
    """Create a risk warning event."""
    return Event(
        event_type=EventType.RISK_LIMIT_WARNING,
        data={
            "warning_type": warning_type,
            "current_value": current_value,
            "limit_value": limit_value,
            **kwargs,
        },
        source="risk_manager",
    )


def create_error_event(
    error_type: str,
    message: str,
    **kwargs,
) -> Event:
    """Create an error event."""
    return Event(
        event_type=EventType.ERROR_OCCURRED,
        data={
            "error_type": error_type,
            "message": message,
            **kwargs,
        },
        source="system",
    )
