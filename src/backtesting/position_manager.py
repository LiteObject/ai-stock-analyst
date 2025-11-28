"""
Position Manager for Realistic Trade Execution.

This module provides:
- Slippage modeling (fixed, percentage, volume-based)
- Transaction cost calculation
- Position sizing with risk constraints
- Trade execution simulation
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SlippageModel(Enum):
    """Slippage model types."""

    FIXED = "fixed"
    PERCENTAGE = "percentage"
    VOLUME_IMPACT = "volume_impact"
    ADAPTIVE = "adaptive"


class ExecutionModel(Enum):
    """Order execution model."""

    MARKET = "market"
    LIMIT = "limit"
    VWAP = "vwap"
    TWAP = "twap"


@dataclass
class TransactionCosts:
    """Transaction cost configuration."""

    commission_rate: float = 0.001  # 0.1% per trade
    min_commission: float = 1.0  # Minimum commission
    sec_fee_rate: float = 0.0000278  # SEC fee (sells only)
    taf_fee_rate: float = 0.000166  # TAF fee
    spread_cost: float = 0.0005  # Half spread cost

    def calculate(
        self,
        quantity: int,
        price: float,
        is_sell: bool = False,
    ) -> Dict[str, float]:
        """
        Calculate all transaction costs.

        Args:
            quantity: Number of shares
            price: Execution price
            is_sell: Whether this is a sell order

        Returns:
            Dictionary with cost breakdown
        """
        notional = quantity * price

        # Commission
        commission = max(notional * self.commission_rate, self.min_commission)

        # SEC fee (sells only)
        sec_fee = notional * self.sec_fee_rate if is_sell else 0.0

        # TAF fee
        taf_fee = quantity * self.taf_fee_rate

        # Spread cost (implicit)
        spread = notional * self.spread_cost

        total = commission + sec_fee + taf_fee + spread

        return {
            "commission": commission,
            "sec_fee": sec_fee,
            "taf_fee": taf_fee,
            "spread_cost": spread,
            "total": total,
            "cost_bps": (total / notional) * 10000 if notional > 0 else 0,
        }


@dataclass
class SlippageConfig:
    """Slippage configuration."""

    model: SlippageModel = SlippageModel.PERCENTAGE
    fixed_slippage: float = 0.01  # Fixed $ amount
    pct_slippage: float = 0.001  # 0.1% of price
    volume_impact_coef: float = 0.1  # Volume impact coefficient
    max_slippage_pct: float = 0.02  # Maximum slippage cap


@dataclass
class ExecutionResult:
    """Result of trade execution."""

    ticker: str
    side: str  # "buy" or "sell"
    requested_quantity: int
    executed_quantity: int
    requested_price: float
    executed_price: float
    slippage: float
    slippage_pct: float
    costs: Dict[str, float]
    total_cost: float
    net_proceeds: float  # For sells
    timestamp: datetime
    fill_rate: float = 1.0
    execution_model: ExecutionModel = ExecutionModel.MARKET


class PositionManager:
    """
    Manages position sizing, execution, and transaction costs.

    Features:
    - Realistic slippage modeling
    - Transaction cost calculation
    - Volume-based position sizing
    - Risk-adjusted position limits
    """

    def __init__(
        self,
        transaction_costs: Optional[TransactionCosts] = None,
        slippage_config: Optional[SlippageConfig] = None,
        max_position_pct: float = 0.20,
        max_daily_volume_pct: float = 0.10,
        min_position_size: int = 1,
    ):
        """
        Initialize the position manager.

        Args:
            transaction_costs: Transaction cost configuration
            slippage_config: Slippage configuration
            max_position_pct: Maximum position as % of portfolio
            max_daily_volume_pct: Maximum % of daily volume to trade
            min_position_size: Minimum shares to trade
        """
        self.costs = transaction_costs or TransactionCosts()
        self.slippage = slippage_config or SlippageConfig()
        self.max_position_pct = max_position_pct
        self.max_daily_volume_pct = max_daily_volume_pct
        self.min_position_size = min_position_size

    def calculate_slippage(
        self,
        price: float,
        quantity: int,
        is_buy: bool,
        daily_volume: Optional[int] = None,
        volatility: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Calculate slippage for a trade.

        Args:
            price: Current market price
            quantity: Number of shares
            is_buy: Whether this is a buy order
            daily_volume: Average daily volume (for volume impact)
            volatility: Price volatility (for adaptive slippage)

        Returns:
            Tuple of (slippage amount, slippage percentage)
        """
        if self.slippage.model == SlippageModel.FIXED:
            slippage = self.slippage.fixed_slippage

        elif self.slippage.model == SlippageModel.PERCENTAGE:
            slippage = price * self.slippage.pct_slippage

        elif self.slippage.model == SlippageModel.VOLUME_IMPACT:
            if daily_volume and daily_volume > 0:
                # Market impact model: slippage increases with order size relative to volume
                participation_rate = quantity / daily_volume
                # Square-root market impact model
                impact = self.slippage.volume_impact_coef * np.sqrt(participation_rate)
                slippage = price * impact
            else:
                slippage = price * self.slippage.pct_slippage

        elif self.slippage.model == SlippageModel.ADAPTIVE:
            # Combines percentage and volatility-based slippage
            base_slippage = price * self.slippage.pct_slippage

            if volatility and volatility > 0:
                # Higher volatility = higher slippage
                vol_multiplier = 1 + (volatility / 0.20)  # Normalize to ~20% vol
                slippage = base_slippage * vol_multiplier
            else:
                slippage = base_slippage
        else:
            slippage = price * self.slippage.pct_slippage

        # Cap slippage
        max_slippage = price * self.slippage.max_slippage_pct
        slippage = min(slippage, max_slippage)

        slippage_pct = slippage / price if price > 0 else 0

        return slippage, slippage_pct

    def calculate_execution_price(
        self,
        price: float,
        quantity: int,
        is_buy: bool,
        daily_volume: Optional[int] = None,
        volatility: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """
        Calculate the expected execution price including slippage.

        Args:
            price: Current market price
            quantity: Number of shares
            is_buy: Whether this is a buy order
            daily_volume: Average daily volume
            volatility: Price volatility

        Returns:
            Tuple of (execution_price, slippage, slippage_pct)
        """
        slippage, slippage_pct = self.calculate_slippage(
            price, quantity, is_buy, daily_volume, volatility
        )

        if is_buy:
            # Buying: price goes up (worse for buyer)
            execution_price = price + slippage
        else:
            # Selling: price goes down (worse for seller)
            execution_price = price - slippage

        return execution_price, slippage, slippage_pct

    def calculate_position_size(
        self,
        portfolio_value: float,
        price: float,
        signal_strength: float = 1.0,
        volatility: Optional[float] = None,
        daily_volume: Optional[int] = None,
        risk_per_trade: float = 0.02,
        stop_loss_pct: Optional[float] = None,
    ) -> int:
        """
        Calculate optimal position size.

        Uses multiple constraints:
        1. Maximum portfolio percentage
        2. Volatility-adjusted sizing
        3. Volume-based limits
        4. Risk-per-trade limits (with stop-loss)

        Args:
            portfolio_value: Total portfolio value
            price: Current price
            signal_strength: Prediction confidence (0-1)
            volatility: Asset volatility
            daily_volume: Average daily volume
            risk_per_trade: Maximum risk per trade as % of portfolio
            stop_loss_pct: Stop loss percentage (for risk-based sizing)

        Returns:
            Optimal number of shares to trade
        """
        if price <= 0:
            return 0

        # 1. Maximum position by portfolio percentage
        max_by_pct = int(
            (portfolio_value * self.max_position_pct * signal_strength) / price
        )

        # 2. Volatility-adjusted sizing (Kelly-inspired)
        if volatility and volatility > 0:
            # Lower position for higher volatility
            vol_adjustment = min(0.20 / volatility, 2.0)  # Cap at 2x
            max_by_vol = int(max_by_pct * vol_adjustment)
        else:
            max_by_vol = max_by_pct

        # 3. Volume-based limit
        if daily_volume and daily_volume > 0:
            max_by_volume = int(daily_volume * self.max_daily_volume_pct)
        else:
            max_by_volume = max_by_pct

        # 4. Risk-per-trade limit (if stop-loss provided)
        if stop_loss_pct and stop_loss_pct > 0:
            # Maximum loss = portfolio_value * risk_per_trade
            max_loss = portfolio_value * risk_per_trade
            # Risk per share = price * stop_loss_pct
            risk_per_share = price * stop_loss_pct
            max_by_risk = int(max_loss / risk_per_share)
        else:
            max_by_risk = max_by_pct

        # Take the minimum of all constraints
        position_size = min(max_by_pct, max_by_vol, max_by_volume, max_by_risk)

        # Ensure minimum size
        if position_size < self.min_position_size:
            position_size = 0  # Don't trade if below minimum

        return position_size

    def execute_trade(
        self,
        ticker: str,
        quantity: int,
        price: float,
        is_buy: bool,
        timestamp: datetime,
        daily_volume: Optional[int] = None,
        volatility: Optional[float] = None,
        execution_model: ExecutionModel = ExecutionModel.MARKET,
    ) -> ExecutionResult:
        """
        Simulate trade execution with realistic costs and slippage.

        Args:
            ticker: Stock ticker
            quantity: Number of shares
            price: Market price
            is_buy: Whether this is a buy order
            timestamp: Execution timestamp
            daily_volume: Average daily volume
            volatility: Price volatility
            execution_model: Type of execution

        Returns:
            ExecutionResult with all execution details
        """
        # Calculate execution price with slippage
        exec_price, slippage, slippage_pct = self.calculate_execution_price(
            price, quantity, is_buy, daily_volume, volatility
        )

        # Calculate fill rate based on volume
        if daily_volume and daily_volume > 0:
            participation = quantity / daily_volume
            if participation > self.max_daily_volume_pct:
                # Partial fill
                fill_rate = self.max_daily_volume_pct / participation
                executed_qty = int(quantity * fill_rate)
            else:
                fill_rate = 1.0
                executed_qty = quantity
        else:
            fill_rate = 1.0
            executed_qty = quantity

        # Calculate transaction costs
        costs = self.costs.calculate(executed_qty, exec_price, is_sell=not is_buy)

        # Calculate total cost or net proceeds
        notional = executed_qty * exec_price

        if is_buy:
            total_cost = notional + costs["total"]
            net_proceeds = 0.0
        else:
            total_cost = costs["total"]
            net_proceeds = notional - costs["total"]

        return ExecutionResult(
            ticker=ticker,
            side="buy" if is_buy else "sell",
            requested_quantity=quantity,
            executed_quantity=executed_qty,
            requested_price=price,
            executed_price=exec_price,
            slippage=slippage * executed_qty,
            slippage_pct=slippage_pct,
            costs=costs,
            total_cost=total_cost,
            net_proceeds=net_proceeds,
            timestamp=timestamp,
            fill_rate=fill_rate,
            execution_model=execution_model,
        )


@dataclass
class PositionState:
    """Current state of a position."""

    ticker: str
    quantity: int
    average_cost: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    opened_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        """Total cost basis of position."""
        return self.quantity * self.average_cost

    @property
    def return_pct(self) -> float:
        """Return percentage on position."""
        if self.cost_basis > 0:
            return (self.market_value - self.cost_basis) / self.cost_basis
        return 0.0


class PortfolioTracker:
    """
    Tracks portfolio positions and calculates P&L.

    Features:
    - Position tracking with average cost
    - Realized and unrealized P&L
    - Stop-loss and take-profit management
    - Trade history
    """

    def __init__(self, initial_cash: float = 100000.0):
        """
        Initialize the portfolio tracker.

        Args:
            initial_cash: Starting cash balance
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, PositionState] = {}
        self.trade_history: List[ExecutionResult] = []
        self.realized_pnl = 0.0

    @property
    def total_value(self) -> float:
        """Total portfolio value."""
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash + positions_value

    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized P&L."""
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def return_pct(self) -> float:
        """Total return percentage."""
        return (self.total_value - self.initial_cash) / self.initial_cash

    def update_prices(self, prices: Dict[str, float], timestamp: datetime) -> None:
        """
        Update current prices for all positions.

        Args:
            prices: Dictionary of ticker -> current price
            timestamp: Current timestamp
        """
        for ticker, position in self.positions.items():
            if ticker in prices:
                position.current_price = prices[ticker]
                position.unrealized_pnl = (
                    position.current_price - position.average_cost
                ) * position.quantity
                position.last_updated = timestamp

    def open_position(
        self,
        execution: ExecutionResult,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
    ) -> bool:
        """
        Open or add to a position.

        Args:
            execution: Execution result from trade
            stop_loss_pct: Optional stop-loss percentage
            take_profit_pct: Optional take-profit percentage

        Returns:
            True if position was opened/added successfully
        """
        if execution.side != "buy":
            return False

        if execution.executed_quantity <= 0:
            return False

        ticker = execution.ticker

        # Deduct cash
        if self.cash < execution.total_cost:
            logger.warning(f"Insufficient cash for {ticker}")
            return False

        self.cash -= execution.total_cost

        # Update or create position
        if ticker in self.positions:
            # Add to existing position (update average cost)
            pos = self.positions[ticker]
            total_qty = pos.quantity + execution.executed_quantity
            total_cost = (pos.average_cost * pos.quantity) + (
                execution.executed_price * execution.executed_quantity
            )
            pos.average_cost = total_cost / total_qty
            pos.quantity = total_qty
            pos.current_price = execution.executed_price
            pos.last_updated = execution.timestamp
        else:
            # Create new position
            stop_loss = None
            take_profit = None

            if stop_loss_pct:
                stop_loss = execution.executed_price * (1 - stop_loss_pct)
            if take_profit_pct:
                take_profit = execution.executed_price * (1 + take_profit_pct)

            self.positions[ticker] = PositionState(
                ticker=ticker,
                quantity=execution.executed_quantity,
                average_cost=execution.executed_price,
                current_price=execution.executed_price,
                unrealized_pnl=0.0,
                opened_at=execution.timestamp,
                last_updated=execution.timestamp,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

        self.trade_history.append(execution)
        return True

    def close_position(
        self,
        execution: ExecutionResult,
    ) -> Tuple[bool, float]:
        """
        Close or reduce a position.

        Args:
            execution: Execution result from trade

        Returns:
            Tuple of (success, realized_pnl)
        """
        if execution.side != "sell":
            return False, 0.0

        ticker = execution.ticker

        if ticker not in self.positions:
            logger.warning(f"No position to close for {ticker}")
            return False, 0.0

        pos = self.positions[ticker]
        qty_to_close = min(execution.executed_quantity, pos.quantity)

        # Calculate realized P&L
        cost_basis = pos.average_cost * qty_to_close
        proceeds = execution.executed_price * qty_to_close
        realized = proceeds - cost_basis - execution.costs["total"]

        # Update cash
        self.cash += execution.net_proceeds

        # Update position
        pos.quantity -= qty_to_close
        pos.realized_pnl += realized
        self.realized_pnl += realized

        if pos.quantity <= 0:
            del self.positions[ticker]

        self.trade_history.append(execution)
        return True, realized

    def check_stops(
        self,
        prices: Dict[str, float],
    ) -> List[str]:
        """
        Check for stop-loss or take-profit triggers.

        Args:
            prices: Current prices

        Returns:
            List of tickers that triggered stops
        """
        triggered = []

        for ticker, pos in self.positions.items():
            if ticker not in prices:
                continue

            current_price = prices[ticker]

            # Check stop-loss
            if pos.stop_loss and current_price <= pos.stop_loss:
                logger.info(f"Stop-loss triggered for {ticker} at {current_price}")
                triggered.append(ticker)
                continue

            # Check take-profit
            if pos.take_profit and current_price >= pos.take_profit:
                logger.info(f"Take-profit triggered for {ticker} at {current_price}")
                triggered.append(ticker)

        return triggered

    def get_summary(self) -> Dict[str, Any]:
        """Get portfolio summary."""
        return {
            "total_value": self.total_value,
            "cash": self.cash,
            "positions_value": sum(p.market_value for p in self.positions.values()),
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "total_pnl": self.total_pnl,
            "return_pct": self.return_pct,
            "n_positions": len(self.positions),
            "n_trades": len(self.trade_history),
        }
