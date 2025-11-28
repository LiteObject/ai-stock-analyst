"""
Risk limits and monitoring.

This module implements checks for various risk limits to ensure
the portfolio stays within defined safety boundaries.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from core.models import Order, OrderSide, Portfolio

logger = logging.getLogger(__name__)


@dataclass
class RiskCheckResult:
    """Result of a risk check."""

    passed: bool
    message: Optional[str] = None
    limit_type: Optional[str] = None
    current_value: Optional[float] = None
    limit_value: Optional[float] = None


class RiskLimitManager:
    """
    Manages and enforces risk limits.
    """

    def __init__(
        self,
        max_position_size_pct: float = 0.20,
        max_portfolio_leverage: float = 1.0,
        max_drawdown_pct: float = 0.25,
        max_concentration_sector: float = 0.40,
        restricted_tickers: Optional[List[str]] = None,
    ):
        """
        Initialize risk limits.

        Args:
            max_position_size_pct: Max size of single position as % of portfolio
            max_portfolio_leverage: Max leverage (1.0 = no leverage)
            max_drawdown_pct: Max allowed drawdown before halting
            max_concentration_sector: Max exposure to a single sector
            restricted_tickers: List of tickers that cannot be traded
        """
        self.max_position_size_pct = max_position_size_pct
        self.max_portfolio_leverage = max_portfolio_leverage
        self.max_drawdown_pct = max_drawdown_pct
        self.max_concentration_sector = max_concentration_sector
        self.restricted_tickers = set(restricted_tickers or [])

    def check_order(self, portfolio: Portfolio, order: Order) -> RiskCheckResult:
        """
        Check if an order violates any risk limits.

        Args:
            portfolio: Current portfolio state
            order: Proposed order

        Returns:
            RiskCheckResult indicating pass/fail
        """
        # 1. Check restricted list
        if order.ticker in self.restricted_tickers:
            return RiskCheckResult(
                False,
                f"Ticker {order.ticker} is on restricted list",
                "RESTRICTED_TICKER",
            )

        # 2. Check position size limit (for buy orders)
        if order.side == OrderSide.BUY:
            # Estimate post-trade position value
            current_pos = portfolio.get_position(order.ticker)
            current_val = current_pos.market_value if current_pos else 0.0

            # Assuming order price is limit price or current market price (need to pass price)
            # For now, using a rough estimate if price not in order
            price = order.limit_price or 100.0  # Fallback, should be improved

            trade_val = order.quantity * price
            new_val = current_val + trade_val

            # Calculate projected portfolio value
            proj_portfolio_val = portfolio.total_value  # Simplified

            if proj_portfolio_val > 0:
                new_weight = new_val / proj_portfolio_val
                if new_weight > self.max_position_size_pct:
                    return RiskCheckResult(
                        False,
                        f"Position size {new_weight:.1%} exceeds limit {self.max_position_size_pct:.1%}",
                        "MAX_POSITION_SIZE",
                        new_weight,
                        self.max_position_size_pct,
                    )

        return RiskCheckResult(True)

    def check_portfolio_health(self, portfolio: Portfolio, drawdown: float) -> RiskCheckResult:
        """
        Check overall portfolio health metrics.

        Args:
            portfolio: Current portfolio
            drawdown: Current drawdown (positive number, e.g. 0.10 for 10% down)

        Returns:
            RiskCheckResult
        """
        # 1. Check Max Drawdown
        if drawdown > self.max_drawdown_pct:
            return RiskCheckResult(
                False,
                f"Drawdown {drawdown:.1%} exceeds limit {self.max_drawdown_pct:.1%}",
                "MAX_DRAWDOWN",
                drawdown,
                self.max_drawdown_pct,
            )

        # 2. Check Leverage
        total_exposure = sum(pos.market_value for pos in portfolio.positions.values())
        equity = portfolio.total_value

        if equity > 0:
            leverage = total_exposure / equity
            if leverage > self.max_portfolio_leverage:
                return RiskCheckResult(
                    False,
                    f"Leverage {leverage:.2f}x exceeds limit {self.max_portfolio_leverage:.2f}x",
                    "MAX_LEVERAGE",
                    leverage,
                    self.max_portfolio_leverage,
                )

        return RiskCheckResult(True)
