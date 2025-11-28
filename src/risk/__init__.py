"""
Risk management module for the AI Stock Analyst.

This module provides tools for:
- Position sizing (Kelly Criterion, Fixed Fractional)
- Value at Risk (VaR) calculation
- Portfolio optimization
- Risk limits and checks
"""

from risk.limits import RiskCheckResult, RiskLimitManager
from risk.portfolio import MeanVarianceOptimizer, PortfolioOptimizer
from risk.position_sizing import (
    FixedFractionalSizer,
    KellyCriterionSizer,
    PositionSizer,
    VolatilityTargetSizer,
)
from risk.var import HistoricalVaR, ParametricVaR, RiskCalculator

__all__ = [
    # Position Sizing
    "PositionSizer",
    "FixedFractionalSizer",
    "KellyCriterionSizer",
    "VolatilityTargetSizer",
    # VaR
    "RiskCalculator",
    "HistoricalVaR",
    "ParametricVaR",
    # Portfolio
    "PortfolioOptimizer",
    "MeanVarianceOptimizer",
    # Limits
    "RiskLimitManager",
    "RiskCheckResult",
]
