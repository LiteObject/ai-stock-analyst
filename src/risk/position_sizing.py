"""
Position sizing algorithms.

This module implements various position sizing strategies to manage risk
and optimize capital allocation.
"""

import logging
from typing import Optional

from core.exceptions import ConfigurationError
from core.interfaces import PositionSizer
from core.models import Portfolio, Signal

logger = logging.getLogger(__name__)


class FixedFractionalSizer(PositionSizer):
    """
    Fixed Fractional Position Sizing.

    Risks a fixed percentage of the portfolio equity on each trade.
    This is a conservative and robust method.
    """

    def __init__(self, risk_per_trade_pct: float = 0.02):
        """
        Initialize.

        Args:
            risk_per_trade_pct: Percentage of equity to risk per trade (default 2%)
        """
        if not 0 < risk_per_trade_pct <= 1.0:
            raise ConfigurationError("Risk percentage must be between 0 and 1")
        self.risk_per_trade_pct = risk_per_trade_pct

    def calculate_risk_per_trade(
        self,
        portfolio: Portfolio,
        stop_loss_pct: float,
    ) -> float:
        """Calculate dollar risk amount."""
        return portfolio.total_value * self.risk_per_trade_pct

    def calculate_position_size(
        self,
        portfolio: Portfolio,
        ticker: str,
        current_price: float,
        signal: Optional[Signal] = None,
        stop_loss_pct: float = 0.05,
        **kwargs,
    ) -> int:
        """
        Calculate position size based on fixed risk.

        Formula: Position Size = (Account Value * Risk %) / (Stop Loss Distance)
        """
        if stop_loss_pct <= 0:
            raise ValueError("Stop loss percentage must be positive")

        # Calculate dollar amount to risk
        risk_amount = self.calculate_risk_per_trade(portfolio, stop_loss_pct)

        # Calculate stop loss distance in dollars per share
        stop_loss_dist = current_price * stop_loss_pct

        # Calculate number of shares
        shares = int(risk_amount / stop_loss_dist)

        # Ensure we don't exceed available cash
        max_shares_cash = int(portfolio.cash / current_price)
        shares = min(shares, max_shares_cash)

        logger.debug(f"Fixed Fractional Sizing for {ticker}: " f"Risk=${risk_amount:.2f}, Shares={shares}")
        return shares


class KellyCriterionSizer(PositionSizer):
    """
    Kelly Criterion Position Sizing.

    Calculates optimal position size based on win rate and win/loss ratio.
    Often used with "Half Kelly" to reduce volatility.
    """

    def __init__(
        self,
        kelly_fraction: float = 0.5,
        max_position_size_pct: float = 0.25,
    ):
        """
        Initialize.

        Args:
            kelly_fraction: Fraction of Kelly to use (e.g., 0.5 for Half Kelly)
            max_position_size_pct: Maximum position size cap
        """
        self.kelly_fraction = kelly_fraction
        self.max_position_size_pct = max_position_size_pct

    def calculate_risk_per_trade(
        self,
        portfolio: Portfolio,
        stop_loss_pct: float,
    ) -> float:
        """
        Not directly applicable for Kelly, but returns max risk based on cap.
        """
        return portfolio.total_value * self.max_position_size_pct * stop_loss_pct

    def calculate_position_size(
        self,
        portfolio: Portfolio,
        ticker: str,
        current_price: float,
        signal: Optional[Signal] = None,
        win_rate: float = 0.5,
        win_loss_ratio: float = 1.5,
        **kwargs,
    ) -> int:
        """
        Calculate position size using Kelly Criterion.

        Formula: f* = (bp - q) / b
        where:
            f* = fraction of bankroll to wager
            b = odds received (win/loss ratio)
            p = probability of winning
            q = probability of losing (1-p)
        """
        if win_loss_ratio <= 0:
            return 0

        p = win_rate
        q = 1 - p
        b = win_loss_ratio

        # Calculate Kelly fraction
        kelly_f = (b * p - q) / b

        # Apply fractional Kelly (e.g., Half Kelly)
        adjusted_f = kelly_f * self.kelly_fraction

        # Cap at max position size and ensure non-negative
        target_pct = max(0.0, min(adjusted_f, self.max_position_size_pct))

        # Calculate shares
        target_value = portfolio.total_value * target_pct
        shares = int(target_value / current_price)

        # Ensure we don't exceed available cash
        max_shares_cash = int(portfolio.cash / current_price)
        shares = min(shares, max_shares_cash)

        logger.debug(f"Kelly Sizing for {ticker}: " f"Kelly={kelly_f:.2f}, Target%={target_pct:.2%}, Shares={shares}")
        return shares


class VolatilityTargetSizer(PositionSizer):
    """
    Volatility Targeting Position Sizing.

    Sizes positions inversely proportional to their volatility to maintain
    a constant portfolio risk target.
    """

    def __init__(self, target_volatility_annual: float = 0.15):
        """
        Initialize.

        Args:
            target_volatility_annual: Target annualized volatility (e.g., 0.15 for 15%)
        """
        self.target_vol = target_volatility_annual

    def calculate_risk_per_trade(self, portfolio: Portfolio, stop_loss_pct: float) -> float:
        return 0.0  # Not used directly

    def calculate_position_size(
        self,
        portfolio: Portfolio,
        ticker: str,
        current_price: float,
        signal: Optional[Signal] = None,
        volatility_annual: Optional[float] = None,
        **kwargs,
    ) -> int:
        """
        Calculate position size based on volatility target.

        Formula: Weight = (Target Volatility) / (Asset Volatility)
        """
        if volatility_annual is None or volatility_annual <= 0:
            # Fallback if volatility not provided
            logger.warning(f"No volatility provided for {ticker}, using default sizing")
            return int((portfolio.total_value * 0.05) / current_price)

        # Calculate target weight
        # Note: This assumes single asset contribution. For portfolio context,
        # correlations should be considered (handled in PortfolioOptimizer).
        target_weight = self.target_vol / volatility_annual

        # Cap leverage at 1.0 (no borrowing)
        target_weight = min(target_weight, 1.0)

        target_value = portfolio.total_value * target_weight
        shares = int(target_value / current_price)

        # Ensure we don't exceed available cash
        max_shares_cash = int(portfolio.cash / current_price)
        shares = min(shares, max_shares_cash)

        logger.debug(
            f"Vol Target Sizing for {ticker}: "
            f"AssetVol={volatility_annual:.2%}, Target%={target_weight:.2%}, Shares={shares}"
        )
        return shares
