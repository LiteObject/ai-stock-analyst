"""
Value at Risk (VaR) calculations.

This module implements methods to calculate Value at Risk (VaR) and
Conditional Value at Risk (CVaR/Expected Shortfall).
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

from core.interfaces import RiskCalculator

logger = logging.getLogger(__name__)


class HistoricalVaR(RiskCalculator):
    """
    Historical Simulation VaR.

    Calculates VaR based on actual historical returns distribution.
    Does not assume a normal distribution.
    """

    def calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        horizon_days: int = 1,
    ) -> float:
        """
        Calculate Historical VaR.

        Args:
            returns: Historical returns series
            confidence: Confidence level (e.g., 0.95)
            horizon_days: Time horizon in days

        Returns:
            VaR value (positive number representing loss %)
        """
        if returns.empty:
            return 0.0

        # Calculate percentile
        percentile = (1 - confidence) * 100
        var_daily = -np.percentile(returns, percentile)

        # Scale to horizon
        return var_daily * np.sqrt(horizon_days)

    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """
        Calculate Historical CVaR (Expected Shortfall).

        Average of losses exceeding VaR.
        """
        if returns.empty:
            return 0.0

        cutoff = np.percentile(returns, (1 - confidence) * 100)
        tail_losses = returns[returns <= cutoff]

        if tail_losses.empty:
            return 0.0

        return -tail_losses.mean()

    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate Maximum Drawdown."""
        if equity_curve.empty:
            return 0.0

        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown.min()


class ParametricVaR(RiskCalculator):
    """
    Parametric (Variance-Covariance) VaR.

    Assumes returns follow a normal distribution.
    Computationally faster but less accurate for non-normal distributions.
    """

    def calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        horizon_days: int = 1,
    ) -> float:
        """
        Calculate Parametric VaR.

        VaR = mu - z * sigma
        """
        if returns.empty:
            return 0.0

        mu = returns.mean()
        sigma = returns.std()

        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence)

        var_daily = -(mu + z_score * sigma)

        return var_daily * np.sqrt(horizon_days)

    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """
        Calculate Parametric CVaR.

        CVaR = mu - (sigma / (1-confidence)) * pdf(z_score)
        """
        if returns.empty:
            return 0.0

        mu = returns.mean()
        sigma = returns.std()
        alpha = 1 - confidence
        z_score = stats.norm.ppf(alpha)

        cvar = -(mu - (sigma / alpha) * stats.norm.pdf(z_score))
        return cvar

    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate Maximum Drawdown (same as Historical)."""
        if equity_curve.empty:
            return 0.0

        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown.min()
