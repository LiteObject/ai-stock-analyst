"""
Performance metrics calculation for backtesting.
"""

from typing import Dict

import numpy as np
import pandas as pd


def calculate_metrics(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> Dict[str, float]:
    """
    Calculate performance metrics from a series of returns.

    Args:
        returns: Series of percentage returns (e.g., 0.01 for 1%)
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of trading periods in a year (252 for daily)

    Returns:
        Dictionary of metrics
    """
    if returns.empty:
        return {}

    # Cumulative Return
    cumulative_prod = (1 + returns).prod()
    total_return: float = float(cumulative_prod) - 1  # type: ignore[arg-type]

    # CAGR
    n_years = len(returns) / periods_per_year
    if n_years > 0:
        cagr: float = (1 + total_return) ** (1 / n_years) - 1
    else:
        cagr = 0.0

    # Volatility (Annualized)
    volatility = returns.std() * np.sqrt(periods_per_year)

    # Sharpe Ratio
    excess_returns = returns - (risk_free_rate / periods_per_year)
    if returns.std() > 0:
        sharpe_ratio = (excess_returns.mean() / returns.std()) * np.sqrt(periods_per_year)
    else:
        sharpe_ratio = 0.0

    # Sortino Ratio
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0 and downside_returns.std() > 0:
        sortino_ratio = (excess_returns.mean() / downside_returns.std()) * np.sqrt(periods_per_year)
    else:
        sortino_ratio = 0.0

    # Max Drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    # Win Rate
    win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0.0

    # Profit Factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
    }
